from typing import Dict, List, Optional
import os
import random
import numpy as np
from datetime import datetime

import torch
from torch import nn

def cal_num_parameters(*models: nn.Module):
    """
    Args:
        (nn.Module)
    Return:
        (str) "The number of parameters : "
    """
    num_params = 0
    for model in models:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        num_params += sum([np.prod(p.size()) for p in model_parameters])
    print_str = f"The number of parameters : {num_params/1000000:.2f}M"
    return print_str

def current_time(readable=False):
    """
    return : 
        if readable==False, '20190212_070531'
        if readable==True, '2019-02-12 07:05:31'
    """
    now = datetime.now()
    if not readable:
        current_time = '{0.year:04}{0.month:02}{0.day:02}_{0.hour:02}{0.minute:02}{0.second:02}'.format(now)
    else:
        current_time = '{0.year:04}-{0.month:02}-{0.day:02} {0.hour:02}:{0.minute:02}:{0.second:02}'.format(now)

    return current_time

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name='', fmt=':f', mode='avg'):
        """
        Args:
            fmt: print format. .4f
            mode: 'avg', 'sum', 'val'
        """
        self.name = name
        self.fmt = fmt
        self.mode = mode
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        if self.mode == 'avg':
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        elif self.mode == 'sum':
            fmtstr = '{name} {val' + self.fmt + '} ({sum' + self.fmt + '})'
        elif self.mode == 'val':
            fmtstr = '{name} {val' + self.fmt + '}'
        elif self.mode == 'avg_only':
            fmtstr = '{name} {avg' + self.fmt + '}'
        else:
            raise NotImplemented(f"{self.mode} Mode not implemented")
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, meters: Optional[Dict[str, AverageMeter]] = None):
        self.meters: Dict[str, AverageMeter] = {} if meters is None else meters

    def display(self, batch):
        entries = [f"Step [{batch}]"]
        entries += [str(meter) for meter in self.meters.values()]
        print('\t'.join(entries))
        return '\t'.join(entries)

    def add(self, name='', fmt=':f', mode='avg'):
        self.meters.update({name: AverageMeter(name, fmt, mode)})

    def update(self, name: str, val: int, n: int = 1):
        self.meters[name].update(val, n)

    def reset(self):
        for key in self.meters.keys():
            self.meters[key].reset()

    def keys(self):
        return self.meters.keys()

def do_seed(seed_num, cudnn_ok=True):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num) # if use multi-GPU
    # It could be slow
    if cudnn_ok:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def set_backend(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    do_seed(cfg.seed)

def merge_meters(list_meters:List[Dict[str, AverageMeter]], 
                 meters: Dict[str, AverageMeter]) -> Dict[str, AverageMeter]:
    for key in meters.keys():
        meters[key].sum = np.sum([m[key].sum for m in list_meters])
        meters[key].count = np.sum([m[key].count for m in list_meters])

    return meters
