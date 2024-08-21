import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np
import math

import os 
from pathlib import Path
import time 
from skimage import measure
import mcubes


from model.triplane_net.networks import TriplaneFeatureNet,SurfaceClassifier,init_weights,CartesianPlaneNonSirenEmbeddingNetwork
import pytorch_lightning as pl
from utils.sdf import create_grid_torch,create_cube
from scripts.vis_gt import vis_sdf
import trimesh

import logging
logger = logging.getLogger("TriplaneSDF")
logger.addHandler(logging.FileHandler("tri-sdf.log"))


class TriplaneSDF(pl.LightningModule):
    def __init__(self, opt, datamodule):
        super().__init__()
        self.automatic_optimization = False

        self.cfg = opt
        # self.aabb = opt.AABB
        self.datamodule = datamodule
        # self.aabb = torch.from_numpy(np.array([[-1,-1,-1],[1,1,1]])).cuda()
        self.aabb = datamodule.trainset.aabb
        if self.cfg.last_op == "relu":
            self.last_op = nn.ReLU()
        elif self.cfg.last_op == "sigmoid":
            self.last_op = nn.Sigmoid()
        elif self.cfg.last_op == "tanh":
            self.last_op = nn.Tanh()
        else:
            raise NotImplementedError()

        self.build_model()

        if opt.loss_fn == "l1":
            self.loss_fn = nn.L1Loss(reduction='none')
        elif opt.loss_fn == "mse":
            self.loss_fn = nn.MSELoss()
        elif opt.loss_fn == "bce":
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError()

    def build_model(self):
        # self.triplane_feat = TriplaneFeatureNet(**self.cfg.triplane,aabb=self.aabb)
        # self.mlp_decoder = SurfaceClassifier(self.cfg.mlp_filters,last_op=self.last_op)

        # init_weights(self.triplane_feat)
        # init_weights(self.mlp_decoder)

        self.auto_decoder = CartesianPlaneNonSirenEmbeddingNetwork()

    def configure_optimizers(self):
    
        # optimizer_feature = torch.optim.Adam(self.triplane_feat.parameters(), self.cfg.lr_feature_init)
        # # optimizer_decoder = torch.optim.Adam(self.mlp_decoder.parameters(), self.cfg.lr_decoder_init)
        # optimizer_decoder = torch.optim.RMSprop(self.mlp_decoder.parameters(), lr=self.cfg.lr_decoder_init,momentum=0, weight_decay=0)


        # feature_scheduler = torch.optim.lr_scheduler.StepLR(
        #                 optimizer_feature, self.cfg.feature_lr_step, self.cfg.feature_lr_gamma)
        # decoder_scheduler = torch.optim.lr_scheduler.StepLR(
        #                 optimizer_decoder, self.cfg.decoder_lr_step, self.cfg.decoder_lr_gamma)
        

        # self.optimizer_list = [optimizer_feature,optimizer_decoder]
        # self.scheduler_list = [feature_scheduler,decoder_scheduler]

        optimizer = torch.optim.Adam(params=self.auto_decoder.parameters(), lr=1e-3, 
            betas=(0.9, 0.999))
        
        scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer, 10_000, 0.1)
        
        self.optimizer_list=[optimizer]
        self.scheduler_list=[scheduler]

        return self.optimizer_list, self.scheduler_list
    
    def forward(self,xyz):
        # query feature from triplane.
        # feature_tirplane = self.triplane_feat(xyz)

        # feature_tirplane = feature_tirplane.permute(1,0)
        # # decoder
        # pred = self.mlp_decoder(feature_tirplane)
        pred_occupancies = self.auto_decoder(xyz)
        return pred_occupancies

    def training_step(self, batch, batch_idx):

        xyz = batch['xyz'] # [B,N,3]

        if self.cfg.train_sdf:
            label= batch['labels_sdf']
        else:
            label = batch["labels_01"]
        # forward
        pred_occupancies = self(xyz)

        if self.cfg.train_sdf:
            pred_occupancies = torch.clamp(pred_occupancies, -0.1, 0.1)
            label = torch.clamp(label, -0.1, 0.1)
        # sdf loss
        # loss = self.loss_fn(pred_occupancies, label.reshape((label.shape[0], label.shape[1], -1)))
        # error = self.mse_loss(pred,label)
        loss = F.l1_loss(pred_occupancies, label.reshape((label.shape[0], label.shape[1], -1)))
        # loss = self.loss_fn(pred_occupancies,label.reshape((label.shape[0], label.shape[1], -1)))
        # loss = loss_.mean()
        self.log(f"train_loss", loss.item(),prog_bar=True)
        # 
        for opt in self.optimizer_list:
            opt.zero_grad()
        loss.backward()
        for opt in self.optimizer_list:
            opt.step()

        for scd in self.scheduler_list:
            # print(scd.get_lr())
            scd.step()

        # todo save ckpts into file.

        os.makedirs("checkpoints",exist_ok=True)

        if self.current_epoch in self.cfg.save_epochs:
                # print(f'Saving checkpoint at step {step}')
                torch.save({
                    'epoch': self.current_epoch,
                    'model_state_dict': self.auto_decoder.state_dict(),
                    'optimizer_state_dict': self.optimizer_list[0].state_dict(),
                    'loss': loss.item(),
                }, f'checkpoints/model_epoch_{self.current_epoch}_loss_{loss.item():.4f}.pt')


    @torch.no_grad()
    def validation_step(self, batch, batch_idx, *args, **kwargs):
        # todo: test error to log, or save some middle results or resconstruction ?
        
        grids = create_cube(self.cfg.res)

        grids = grids.cuda()

        num_pts = grids.shape[0]

        sdf = torch.zeros(num_pts)

        num_samples = self.cfg.num_samples

        num_batch = num_pts // num_samples

        for i in range(num_batch):
            batch_pts = grids[i*num_samples:i*num_samples+num_samples]
            pred_sdf = self(batch_pts[None,...])
            sdf[i*num_samples:i*num_samples+num_samples] = pred_sdf[0,:,0]

        if num_pts % num_samples:
            sdf[num_batch*num_samples:] = self(grids[num_batch*num_samples:][None,...])[0,:,0]

        sdf = sdf.reshape(self.cfg.res,self.cfg.res,self.cfg.res)

        if 0:
            out_dir = "./test"
            os.makedirs(out_dir,exist_ok=True)
            # xyz = batch['xyz'] # [B,N,3]
            # label= batch['labels_sdf']
            vis_sdf(grids.cpu().numpy(),sdf.cpu().numpy(),f"{out_dir}/{self.current_epoch:04d}.ply")

        try:
            out_dir = "./val_mesh"
            os.makedirs(out_dir,exist_ok=True)
            # print(sdf.min(),sdf.max())
            sdf_numpy = sdf.cpu().numpy()
            # verts, faces, normals, values = measure.marching_cubes(sdf_numpy, 0.0) # todo:use diffcu.
            vertices, triangles = mcubes.marching_cubes(sdf_numpy, 0.0)

            vertices /=self.cfg.res
            new_vertices = (vertices*2-1)*0.9
            mcubes.export_obj(new_vertices,triangles,f"{out_dir}/{self.current_epoch:04d}.obj")

        except:
            print("error cannot marching cubes")
            return -1

    ######################
    # DATA RELATED HOOKS #
    ######################
    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()

# if train_idx % opt.freq_save == 0 and train_idx != 0:
#     torch.save(netG.state_dict(), 'experiments/%s/%s/netG_latest' % (opt.checkpoints_path, opt.name))
#     torch.save(netG.state_dict(), 'experiments/%s/%s/netG_epoch_%d' % (opt.checkpoints_path, opt.name, epoch))








