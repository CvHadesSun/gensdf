from pathlib import Path
import os
import numpy as np
import hydra
import torch
import cv2
import glob
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import random
from scripts.vis_gt import vis_sdf


class CustomDataset(torch.utils.data.Dataset):

    """
    000
    |
    -mesh.txt
    -colors
    -masks
    -depths
    -masks
    -camears.npz
    -samples.npz

    """
    def __init__(self, root, subject, opt):
        
        # random.seed(1991)
        # np.random.seed(1991)
        # torch.manual_seed(1991)

        self.opt = opt
        self.root = f"{root}/{subject}"
        self.num_sample = opt.num_sample

        aabb = np.load(f"{self.root}/aabb.npy")

        a= torch.from_numpy(aabb[0]).cuda()
        b= torch.from_numpy(aabb[1]).cuda()

        self.aabb = a,b

        # todo load cameras, color, depth, normal if trian with those gt

        # load sdf samples

        self.samples = np.load(f"{self.root}/samples.npy") # 





    def __len__(self):
        # return len(self.img_lists)
        return 1
    
    # def get_sample_sdf(self,data_dir):
    #     sample_dict = dict(
    #         np.load(f"{data_dir}/samples.npz",allow_pickle=True))
        
    #     in_samples = torch.from_numpy(sample_dict["sample_points_in"])
    #     out_samles = torch.from_numpy(sample_dict["sample_points_out"])
        
    #     random_idx1 = (torch.rand(self.num_sample//2) *
    #                   in_samples.shape[0]).long()
        
    #     random_idx2 = (torch.rand(self.num_sample//2) *
    #                   out_samles.shape[0]).long()

    #     _in_sample = in_samples[random_idx1,:]
    #     _out_sample = out_samles[random_idx2,:]

    #     sample_points = torch.cat([_in_sample[:,:3],_out_sample[:,:3]],dim=0)
    #     labels_sdf = torch.cat([_in_sample[:,3],_out_sample[:,3]],dim=0)

    #     labels_01 =torch.cat([torch.ones(self.num_sample//2),torch.zeros(self.num_sample//2)],dim=0)

    def get_sample_sdf(self,data_dir,idx):

        indices = np.random.randint(low=0, high=self.samples.shape[0], size=self.num_sample)
        sampled_data = self.samples[indices] # self.points_batch_size, 4

        xyz = sampled_data[:,:3]
        occ = sampled_data[:,3:4]
        sdf = sampled_data[:,4:5]

        # vis_sdf(xyz,np.abs(sdf),f"./gt_{idx}.ply")

        return {
            'xyz': torch.from_numpy(xyz).float(),
            'labels_sdf': torch.from_numpy(sdf).float(),
            "labels_01": torch.from_numpy(occ).float()
        }

    def __getitem__(self, idx):

        datum = self.get_sample_sdf(self.root,idx)
        return datum


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, opt, **kwargs):
        super().__init__()

        data_dir = Path(hydra.utils.to_absolute_path(opt.dataroot))
        for split in ("train", "val", "test"):
            dataset = CustomDataset(data_dir, opt.subject, opt)
            setattr(self, f"{split}set", dataset)
        self.opt = opt

    def train_dataloader(self):
        if hasattr(self, "trainset"):
            return DataLoader(self.trainset,
                              shuffle=True,
                              num_workers=self.opt.train_num_workers,
                              persistent_workers=True and self.opt.train_num_workers> 0,
                              pin_memory=True,
                              batch_size=1)
        else:
            return super().train_dataloader()

    def val_dataloader(self):
        if hasattr(self, "valset"):
            return DataLoader(self.valset,
                              shuffle=False,
                              num_workers=self.opt.val_num_workers,
                              persistent_workers=True and self.opt.val_num_workers > 0,
                              pin_memory=True,
                              batch_size=1)
        else:
            return super().test_dataloader()

    def test_dataloader(self):
        if hasattr(self, "testset"):
            return DataLoader(self.testset,
                              shuffle=False,
                              num_workers=self.opt.test_num_workers,
                              persistent_workers=True and self.opt.test_num_workers > 0,
                              pin_memory=True,
                              batch_size=1)
        else:
            return super().test_dataloader()