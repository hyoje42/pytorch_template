"""
"""
import sys
from torch.utils.data import DataLoader

# custom
from dataset.dataset import Dataset_hint
from trainers.trainer import Trainer_Base
from utils.config import parse_args
from utils.util import current_time, set_backend

def main():
    cfg, log = parse_args()
    log(' '.join(sys.argv))
    log(str(cfg))
    set_backend(cfg)

    ######## Dataset ########
    train_dataset = Dataset_hint(cfg, mode='train')
    valid_dataset = Dataset_hint(cfg, mode='valid')

    train_loader = DataLoader(
        train_dataset, shuffle=True, 
        batch_size=cfg.batch_size, 
        pin_memory=True,
        num_workers=cfg.num_workers,
    )
    
    valid_loader = DataLoader(
        valid_dataset, shuffle=False,
        batch_size=cfg.batch_size,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )

    # ######## Model ########
    trainer = Trainer_Base(cfg)
    trainer.set_loader(train_loader, valid_loader)
    
    for epoch in range(1, cfg.epochs+1):
        # trainer.train_loop(epoch)
        pass
    # end time
    log(current_time(readable=True))

if __name__ == "__main__":
    ## debug
    # torch.autograd.set_detect_anomaly(True)
    main()