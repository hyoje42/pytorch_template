from typing import Tuple
import os
import yaml
import shutil
import argparse

from utils.util import current_time

class Logger():
    def __init__(self, logfile, overlap=False) -> None:
        self.logfile = logfile
        if not overlap:
            f = open(self.logfile, 'w')
            f.close()

    def print_args(self, args):
        self.log(f'Strat time : {current_time(easy=True)}')
        for key in args.__dict__.keys():
            self.log(f'{key} : {args.__dict__[key]}')
    
    def log(self, text: str, consol: bool = True) -> None:
        with open(self.logfile, 'a') as f:
            print(text, file=f)
        if consol:
            print(text)
    
    def __call__(self, text: str, consol: bool = True):
        self.log(text, consol)

class Config(object):
    """ default param setting
    """

    def __init__(self, args) -> None:
        # training settings
        self.exp_name: str = args.exp_name
        self.gpu: str = args.gpu
        self.seed: int = args.seed
        self.num_workers: int = args.num_workers
        self.freq_log: int = args.freq_log
        self.freq_save: int = args.freq_save
        self.freq_vis: int = args.freq_vis
        self.freq_valid: int = args.freq_valid

        # hyper-paramters
        self.method: str = args.method
        self.epochs: int = args.epochs
        self.batch_size: int = args.batch_size
        self.lr: float = args.lr
        self.weight_decay: float =  args.weight_decay
        self.input_shape: Tuple[int] = tuple(args.input_shape)  # (h, w)

        assert len(self.__dict__) == len(args.__dict__), "Check argparse"

        # params of expname
        self.expname_params = {

        }

        self._build()

    def __str__(self) -> str:
        _str = "==== params setting ====\n"
        for k, v in self.__dict__.items():
            if not type(v) in [int, float, str, bool, list, dict, tuple]:
                continue
            _str += f"{k} : {v}\n"
        return _str

    def _build(self):
        """ 
        Define expname. Create exp dir.
        """
        for k, v in self.expname_params.items():
            self.exp_name += f"_{v}{self.__getattribute__(k)}"

        if self.exp_name[0] == '_': self.exp_name = self.exp_name[1:]

        self.exp_dir = os.path.join("./exps", self.exp_name)
        self.save_dir = os.path.join(self.exp_dir, "saved_models")
        self.tb_dir = os.path.join(self.exp_dir, "tb")
        self.log_path = os.path.join(self.exp_dir, "log.txt")
        
        self._save()
        self.logger = Logger(self.log_path)

    def _save(self) -> None:
        if os.path.exists(self.exp_dir):
            print(self.exp_dir)
            isdelete = input("delete exist exp dir (y/n): ") if not 'debug' in self.exp_dir else "y"
            if isdelete == "y":
                shutil.rmtree(self.exp_dir)
            elif isdelete == "n":
                raise FileExistsError
            else:
                raise FileExistsError

        os.makedirs(self.exp_dir)
        os.makedirs(self.save_dir)
        os.makedirs(self.tb_dir)

        yaml_path = os.path.join(self.exp_dir, "params.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(self.__dict__, f)

def parse_args() -> Tuple[Config, Logger]:
    parser = argparse.ArgumentParser()
    ## 
    parser.add_argument('--nm', '--exp_name', type=str, dest='exp_name', default='exp', help="the name of experiment")
    parser.add_argument('-g', '--gpu', type=str, default='0', dest='gpu', help="gpu number (default: 0)")
    parser.add_argument('--seed', type=int, default=2021, metavar='N', help='seed number. if 0, do not fix seed (default: 2021)')
    parser.add_argument('--num_workers', type=int, default=4, metavar='N', help="the number of workers (default: 4)")
    parser.add_argument('--freq_log', type=int, default=10, metavar='N', help='log step (default: 10)')
    parser.add_argument('--freq_save', type=int, default=2000, metavar='N', help='save step (default: 2000)')
    parser.add_argument('--freq_vis', type=int, default=2000, metavar='N', help='vis step (default: 2000)')
    parser.add_argument('--freq_valid', type=int, default=2000, metavar='N', help='valid step (default: 2000)')

    ## hyper-parameters
    parser.add_argument('--method', type=str, default='', help="")
    parser.add_argument('--epochs', type=int, default=1000, metavar='N', help="epochs (default: 1000)")
    parser.add_argument('-b', '--batch_size', type=int, dest='batch_size', default=2, metavar='N', help="batch size (default: 84)")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--wd', '--weight_decay', type=float, dest='weight_decay', default=0.0, help="weight decay (default: 0.0)")
    parser.add_argument('--is', '--input_shape', nargs='+', type=int, default=[512, 512], 
                        metavar='(N, N)', dest='input_shape', help='input shape. (h, w) (default: (512, 512))')

    args, unknown_args = parser.parse_known_args()
    # check for jupyter
    if len(unknown_args) and unknown_args[0] == '-f' and 'jupyter' in unknown_args[1]:
        unknown_args = unknown_args[2:]
    # check for invalid args
    assert len(unknown_args) == 0, f"Invalid Arguments: {str(unknown_args)}"

    config = Config(args)
    
    return config, config.logger



