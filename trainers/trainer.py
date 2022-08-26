from typing import Dict, List, Optional

import torch
import torchvision
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from utils.util import AverageMeter, ProgressMeter
from utils.config import Config

class Trainer_Base(object):
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.epoch = 0
        self.step = 0
        self.tb_writer = SummaryWriter(log_dir=cfg.tb_dir)
        self.log = cfg.logger

        self._build_model()
        self._set_meters()
        self._set_loss_optimizer()
    
    def _build_model(self):
        pass
    
    def _set_loss_optimizer(self):
        pass
    
    def set_loader(self, train_loader, valid_loader):
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def _set_meters(self):
        self.meters: Dict[str, AverageMeter] = {}
        self.meters['batch_time'] = AverageMeter('Time', ':.1f')
        self.meters['data_time'] = AverageMeter('Data', ':.3f')
        self.meters['loss'] = AverageMeter('loss', ':.4f')

        self.progress = ProgressMeter(self.meters)

    def update_log(self, losses: Dict[str, Tensor], progress: ProgressMeter, size: int):
        for key in losses.keys():
            if key not in progress.keys():
                progress.add(key, fmt=':.4f', mode='avg_only')
            progress.update(key, val=losses[key].item(), n=size)
        
    def state_dict(self):
        state_dict = {
            'step': self.step,
            'epoch': self.epoch,
            'config': self.cfg,
            'optim_G': self.optim_G.state_dict(),
            'optim_D': self.optim_D.state_dict(),
        }
        state_dict.update({
            'netG': self.netG.state_dict(),
            'netD': self.netD.state_dict(),
        })
            
        return state_dict

    def save(self, mode='step'):
        """
        Args:
            mode: 'step', or 'best'
        """
        if mode == 'step':
            path = f"{self.cfg.save_dir}/model_step_{self.step}"
        else:
            path = f"{self.cfg.save_dir}/model"

        path += ".pth"
        torch.save(self.state_dict, path)

    def write_tb(self, meters: Dict[str, AverageMeter], keys: List[str], name: str = 'train'):
        """
        Args:
            meters: meters to write on tensorboard
            keys: string keys to write. (loss, score, ...)
        """
        for c_key in meters.keys():
            if any([key in c_key for key in keys]):
                self.tb_writer.add_scalar(f"{name}/{c_key}", meters[c_key].avg, self.step)
        self.tb_writer.flush()
    
    def visualize_tb(self, images: List[Tensor], name: str = 'train/output', num: Optional[int] = None) -> None:
        """
        Args:
            images: a list of images for vis. [(B, 3, H, W), (B, 3, H, W), ...]
            num: the number of visualized images.
        """
        h, w = images[0].shape[-2:]
        concat = torch.cat(images, dim=0).reshape(len(images), -1, 3, h, w)
        if num is not None:
            concat = concat[:, :num, :, :, :]

        grid = torchvision.utils.make_grid(
            concat.transpose(0, 1).reshape(-1, 3, h, w), nrow=len(images)
            ).cpu().numpy().transpose(1, 2, 0)

        self.tb_writer.add_image(name, grid[..., ::-1], self.step, dataformats='HWC')

    def train(self):
        pass
    
    def eval(self):
        pass

    def cuda(self):
        pass

