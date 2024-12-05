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

    ckpts = sorted(glob.glob(f"{dir}/*.pt"))
    print(ckpts)
    
    return ckpts[0]

@hydra.main(config_path="../confs", config_name="MultiTriplaneSDF")
def main(opt):
    pl.seed_everything(opt.seed)
    torch.set_printoptions(precision=6)
    print(f"Switch to {os.getcwd()}")

    # load ckpts
    datamodule = hydra.utils.instantiate(opt.dataset, _recursive_=False)
    ckpts_dir = search_and_load('./checkpoints')
    # ckpts_dir = './checkpoints/model_epoch_1000_loss_0.7850.pt'
    # import ipdb; ipdb.set_trace()

    model = hydra.utils.instantiate(opt.model, datamodule=datamodule,is_train=False, _recursive_=False)

    model.load_ckpts_own(ckpts_dir)
    # model.load_decoder()

    val_obj=[0,1,2,3,4,5,6,7,8,9,20,30,40,50,60]
    for obj in val_obj:
        # obj_idx=torch.Tensor([obj]).cuda().to(torch.int)

        # inference.
        model.test(obj)

if __name__ == "__main__":
    main()
    

