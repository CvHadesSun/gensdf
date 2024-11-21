import glob
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from omegaconf import OmegaConf
import sys
import argparse
import concurrent.futures
from tqdm import tqdm

sys.path.append(os.getcwd())

# @hydra.main(config_path="../confs", config_name="MultiTriplaneSDF")
def main(opt):
    pl.seed_everything(opt.seed)
    torch.set_printoptions(precision=6)
    # print(f"Switch to {os.getcwd()}")

    wk_dir = os.getcwd()
    run_dir = f"{wk_dir}/{opt.hydra.run.dir}"
    os.makedirs(run_dir,exist_ok=True)
    opt.model.opt.work_dir = run_dir
    # os.chdir(run_dir)


    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     dirpath=f"checkpoints/",
    #     filename="epoch={epoch:04d}",
    #     auto_insert_metric_name=False,
    #     save_last=True,
    #     **opt.checkpoint,
    # )

    if os.path.exists(f"{run_dir}/checkpoints"):
    
        ckpts = len(os.listdir(f"{run_dir}/checkpoints"))
        # print('-------',run_dir)

        ckpts_f = []
        
        for item in ckpts:
            if '.pt' in item:
                ckpts_f.append(item)

        num_ckpts = len(ckpts_f)
    else:
        num_ckpts = 0


    if num_ckpts >=2:
        return
    
    # lr_monitor = pl.callbacks.LearningRateMonitor()

    pl_logger = TensorBoardLogger("tensorboard", name="default", version=0)

    datamodule = hydra.utils.instantiate(opt.dataset, _recursive_=False)

    model = hydra.utils.instantiate(opt.model, datamodule=datamodule, _recursive_=False)

    trainer = pl.Trainer(devices=1,
                         accelerator="gpu",
                         callbacks=[],
                         num_sanity_val_steps=0,  # disable sanity check
                         logger=pl_logger,
                        #  gradient_clip_val=0.1,
                        #  profiler=pl_profiler,
                         **opt.train)
    

    # restore_ckpt_dir = opt.restore_ckpt
    # checkpoints = sorted(glob.glob(f"{restore_ckpt_dir}/checkpoints/*.pt"))
    # print("Saving configs.")
    # OmegaConf.save(opt, "config.yaml")




    
    # if len(checkpoints) > 0 and opt.resume:
    #     print("Resume from", checkpoints[-1])
    #     model.load_ckpts_own(checkpoints[-1])
    #     trainer.fit(model)
    # else:

    # try:
    if opt.load_decoder:
        model.load_decoder()
    trainer.fit(model)
    # except:
    #     pass


    # while 1:
    #     pass

def argp():

    parser = argparse.ArgumentParser(description="multi train")

    parser.add_argument('-b', '--begin_id', type=int, help="the begin obj id")
    parser.add_argument('-n', '--num_objs', type=int, help="the number of this gpu training objs")
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    cfg_dir = f"confs/MultiTriplaneSDF.yaml"

    cfg = OmegaConf.load(cfg_dir)

    args = argp()

    all_cfgs = []

    for i in range(args.num_objs):
        obj_id = i + args.begin_id
        cfg = OmegaConf.load(cfg_dir)
        cfg.dataset.opt.obj_id = obj_id

        cfg.subject = f"{obj_id:05d}"

        all_cfgs.append(cfg)


    with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
        # Submit tasks to the executor
        futures = [executor.submit(main, item) for item in all_cfgs]
        # Wait for all the futures to complete and get the results
        for future in concurrent.futures.as_completed(futures):
            # print(future.result())
            pass

    # for item in all_cfgs:
    #     main(item)

    #     break