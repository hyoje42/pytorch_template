from typing import List, Optional
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import BasicBlock, Bottleneck

from .layers import SNConv2d

class Colorizer(nn.Module):
    def __init__(self, in_channels: int = 16, norm: bool = False):
        super().__init__()
        self.encoder = Encoder(in_channel=5, channels=in_channels, depth=5, blockwise_depth=2)
        self.decoder_draft = Decoder(channels=in_channels, return_tanh=norm)
        self.decoder_refine = Decoder(channels=in_channels, return_tanh=norm)
        self.resnet_backbone = ResNetBackbone(34)

    def encode_embed(self, x: Tensor) -> List[Tensor]:
        """
        Args:
            x: (B, 5, H, W)
        """
        return self.encoder(x)

    def decode_draft(self, embeds: List[Tensor]) -> Tensor:
        return self.decoder_draft(embeds) 

    def decode_refine(self, embeds: List[Tensor]) -> Tensor:
        return self.decoder_refine(embeds)

    def encode_resblock(self, x: Tensor) -> Tensor:
        return self.resnet_backbone(x)

    def forward(self, x, x_for_refine: Optional[Tensor]=None):
        embeds = self.encoder(x)
        out_draft = self.decoder_draft(embeds)
        if x_for_refine is None:
            embeds[-1] += self.resnet_backbone(out_draft)
        else:
            embeds[-1] += self.resnet_backbone(x_for_refine)
        out_refine = self.decoder_refine(embeds)

        return out_draft, out_refine

class Encoder(nn.Module):
    def __init__(self, in_channel, channels, depth=5, blockwise_depth=2):
        super(Encoder, self).__init__()
        cnum = channels
        self.layers = nn.ModuleList(
            self.build_layers(in_channel, cnum, depth, blockwise_depth),)
        self.depth = depth

    def build_layers(self, in_channel, cnum, depth, blockwise_depth):
        layers = []
        for d in range(depth):
            if d > 0:
                layers.append(
                    encoder_block(cnum, 2*cnum, blockwise_depth)
                )
                cnum *= 2
            else:
                layers.append(
                    encoder_block(in_channel, cnum, blockwise_depth)
                )
        return layers

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs

class Decoder(nn.Module):
    def __init__(self, channels, depth=5, blockwise_depth=2, return_tanh=False):
        super(Decoder, self).__init__()
        cnum = channels
        self.layers = nn.ModuleList(
            self.build_layers(cnum, depth, blockwise_depth),)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(cnum, cnum, 3, stride=2, padding=1, output_padding=1),   # x2
            nn.LeakyReLU(0.2),
            nn.Conv2d(cnum, 3, 3, stride=1, padding=1),
        )

        self.return_tanh = return_tanh

    def build_layers(self, cnum, depth, blockwise_depth):
        dim = cnum*2**(depth-1)
        layers = []
        for _ in range(depth - 1):
            layers.append(
                decoder_block(dim, dim//2, blockwise_depth)
            )
            dim = dim // 2
        return layers

    def forward(self, s):
        x = s[-1]
        
        s_idx = -2
        for layer in self.layers:
            x = layer(x)
            x = F.interpolate(x, scale_factor=2)
            x = x + s[s_idx]
            s_idx += -1
        out = self.final(x)
        return torch.tanh(out) if self.return_tanh else torch.clamp(out, 0, 1)

class ResNetBackbone(nn.Module):

    def __init__(self, resnet_type):
	
        resnet_spec = {18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'),
		       34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'),
		       50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
		       101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'),
		       152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')}
        block, layers, channels, name = resnet_spec[resnet_type]
        
        self.name = name
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

class Discriminator(nn.Module):
    """Defines a Spectral Normalized PatchGAN discriminator"""

    def __init__(self, input_nc=4, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        use_bias = False
        kw = 4
        padw = 1
        sequence = [
            SNConv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                SNConv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                # norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            SNConv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            # norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2),
        ]

        sequence += [
            SNConv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        """Standard forward."""
        return self.model(x)

def encoder_block(in_channel, out_channel, blockwise_depth=2):
    layers = []
    stride = 2-(in_channel == 3 or in_channel == 1)
    layers.extend([
        nn.Conv2d(in_channel, out_channel, 3,
                  stride, 1, ),  # remove padding
        nn.LeakyReLU(0.2),
    ])
    for i in range(blockwise_depth-1):
        layers.extend([
            nn.Conv2d(out_channel, out_channel, 3,
                      1, 1, ),  # remove padding
            nn.LeakyReLU(0.2),
        ])
    return nn.Sequential(*layers)


def decoder_block(in_channel, out_channel, blockwise_depth=2):
    layers = []
    layers.extend([
        nn.Conv2d(in_channel, 2*out_channel, 3, 1, 1, ),  # remove padding
        nn.LeakyReLU(0.2),
    ])
    if blockwise_depth > 2:
        for i in range(blockwise_depth-2):
            layers.extend([
                nn.Conv2d(2*out_channel, 2*out_channel, 3,
                          1, 1, ),  # remove padding
                nn.LeakyReLU(0.2),
            ])
    layers.extend([
        nn.Conv2d(2*out_channel, out_channel, 3, 1, 1, ),  # remove padding
        nn.LeakyReLU(0.2),
    ])
    return nn.Sequential(*layers)

def compute_dim_ratio(depth):
    cnum = 2
    dim = 0
    for d in range(depth - 1):
        dim += cnum
        cnum *= 2
    return dim