import glob
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from omegaconf import OmegaConf
import sys

sys.path.append(os.getcwd())

def search_and_load(dir):
    
    pass

@hydra.main(config_path="../confs", config_name="MultiTriplaneSDF")
def main(opt):
    pl.seed_everything(opt.seed)
    torch.set_printoptions(precision=6)
    print(f"Switch to {os.getcwd()}")

    # load ckpts
    datamodule = hydra.utils.instantiate(opt.dataset, _recursive_=False)

    # ckpts_dir = search_and_load('./checkpoints')
    ckpts_dir = './checkpoints/model_epoch_1500_loss_0.0090.pt'

    model = hydra.utils.instantiate(opt.model, datamodule=datamodule, _recursive_=False)

    model.load_ckpts_own(ckpts_dir)
    obj_idx=torch.Tensor([7]).cuda().to(torch.int)

    # inference.
    model.test(obj_idx)

if __name__ == "__main__":
    main()
    

