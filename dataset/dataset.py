"""

"""
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils.config import Config

class Dataset_custom(Dataset):
    """

    """
    def __init__(self, cfg: Config, mode: str = 'train'):
        """
        mode: 'train', 'valid'
        """
        self.cfg = cfg
        self.logger = cfg.logger
        self.mode = mode
        self.shape = cfg.input_shape    # (h, w)

        # transforms
        self.get_trasforms()

    def get_trasforms(self):
        transform_list = [transforms.ToTensor()]
        if True:
            self.transform = transforms.Compose(transform_list + 
                [transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
            self.transform_1d = transforms.Compose(transform_list + 
                [transforms.Normalize(mean=[0.5], std=[0.5])])
        else:
            self.transform = transforms.Compose(transform_list)
            self.transform_1d = transforms.Compose(transform_list)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        pass
