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
import diso

from model.triplane_net.networks import TriplaneFeatureNet,SurfaceClassifier,init_weights
import pytorch_lightning as pl
from utils.sdf import create_grid_torch,create_cube
from scripts.vis_gt import vis_sdf
import trimesh
from model.triplane_net.multi_network import MLPDecoder,MultiTriplane
import mcubes

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
        if self.cfg.last_op == "relu":
            self.last_op = nn.ReLU()
        elif self.cfg.last_op == "sigmoid":
            self.last_op = nn.Sigmoid()
        elif self.cfg.last_op == "tanh":
            self.last_op = nn.Tanh()
        else:
            self.last_op = None

        self.num_objs = len(datamodule.trainset) # for trainset objs.

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
        self.triplane_feat = MultiTriplane(num_objs=self.num_objs,**self.cfg.triplane)
        in_channls = self.triplane_feat.feature_dim * 3 + self.triplane_feat.embedding_channels
        self.mlp_decoder = MLPDecoder(in_channls,latend_dim=self.cfg.deocder.latend_dim,last_op=self.last_op)



    def configure_optimizers(self):
    
        optimizer_feature = torch.optim.Adam(self.triplane_feat.parameters(), self.cfg.lr_feature_init)
        optimizer_decoder = torch.optim.Adam(self.mlp_decoder.parameters(), self.cfg.lr_decoder_init)
        # optimizer_decoder = torch.optim.RMSprop(self.mlp_decoder.parameters(), lr=self.cfg.lr_decoder_init,momentum=0, weight_decay=0)


        feature_scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer_feature, self.cfg.feature_lr_step, self.cfg.feature_lr_gamma)
        decoder_scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer_decoder, self.cfg.decoder_lr_step, self.cfg.decoder_lr_gamma)
        
        self.optimizer_list = [optimizer_feature,optimizer_decoder]
        self.scheduler_list = [feature_scheduler,decoder_scheduler]

        return self.optimizer_list, self.scheduler_list


    
    def forward(self,obj_idx,xyz):
        # query feature from triplane.
        feature_tirplane = self.triplane_feat(obj_idx,xyz)
        # decoder
        pred = self.mlp_decoder(feature_tirplane)

        return pred

    def training_step(self, batch, batch_idx):
        if self.current_epoch > self.cfg.fix_mlp_util:
            self.mlp_decoder.eval() # fix the decoder weights.

        obj_idx = batch['obj_idx']
        xyz = batch['xyz'] # [B,N,3]

        if self.cfg.train_sdf:
            label= batch['labels_sdf']
        else:
            label = batch["labels_01"]
        # forward
        pred = self(obj_idx,xyz)

        # sdf loss
        error = self.loss_fn(pred,label)
        # error = self.mse_loss(pred,label)
        loss = error.mean()

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

        os.makedirs("checkpoints",exist_ok=True)

        if batch_idx == 0:
            if self.current_epoch in self.cfg.save_epochs:
                    # print(f'Saving checkpoint at step {step}')
                    torch.save({
                        'epoch': self.current_epoch,
                        'triplane_state_dict': self.triplane_feat.state_dict(),
                        'decoder_state_dict': self.mlp_decoder.state_dict(),
                        'optimizer_state_dict': self.optimizer_list[0].state_dict(),
                        'loss': loss.item(),
                    }, f'checkpoints/model_epoch_{self.current_epoch}_loss_{loss.item():.4f}.pt')

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, *args, **kwargs):
        # todo: test error to log, or save some middle results or resconstruction ?
        # if batch_idx == 0:
        if 1:
            grids = create_cube(self.cfg.res)

            grids = grids.cuda()

            num_pts = grids.shape[0]

            sdf = torch.zeros(num_pts)

            num_samples = self.cfg.num_samples

            num_batch = num_pts // num_samples
            obj_idx = batch['obj_idx']

            for i in range(num_batch):
                batch_pts = grids[i*num_samples:i*num_samples+num_samples]
                pred_sdf = self(obj_idx,batch_pts[None,...])
                sdf[i*num_samples:i*num_samples+num_samples] = pred_sdf[0,:,0]

            if num_pts % num_samples:
                sdf[num_batch*num_samples:] = self(obj_idx,grids[num_batch*num_samples:][None,...])[0,:,0]

            sdf = sdf.reshape(self.cfg.res,self.cfg.res,self.cfg.res) *-1
            if 0:
                out_dir = "./test"
                os.makedirs(out_dir,exist_ok=True)
                vis_sdf(grids.cpu().numpy(),sdf.cpu().numpy(),f"{out_dir}/{self.current_epoch:04d}.ply")
            # try:
            if 1:
                vertices, triangles, normals, values = measure.marching_cubes(sdf.cpu().numpy(), 0.0) # todo:use diffcu.
                # vertices, triangles = diso.DiffMC().cuda().forward(sdf.cuda())
                out_dir = "./val_mesh"
                os.makedirs(out_dir,exist_ok=True)
                vertices /=self.cfg.res
                new_vertices = (vertices*2-1)*0.9
                mcubes.export_obj(new_vertices,triangles,f"{out_dir}/{self.current_epoch:04d}.obj")
                # mcubes.export_obj(new_vertices.cpu().numpy(),triangles.cpu().numpy(),f"{out_dir}/{self.current_epoch:04d}.obj")
            # except:
            #     print("error cannot marching cubes")
            #     return -1

    @torch.no_grad()
    def test(self, obj_idx):
        # todo: test error to log, or save some middle results or resconstruction ?
        # if batch_idx == 0:
        if 1:
            grids = create_cube(self.cfg.res)

            grids = grids.cuda()

            num_pts = grids.shape[0]

            sdf = torch.zeros(num_pts)

            num_samples = self.cfg.num_samples

            num_batch = num_pts // num_samples
            # obj_idx = batch['obj_idx']

            for i in range(num_batch):
                batch_pts = grids[i*num_samples:i*num_samples+num_samples]
                pred_sdf = self(obj_idx,batch_pts[None,...])
                sdf[i*num_samples:i*num_samples+num_samples] = pred_sdf[0,:,0]

            if num_pts % num_samples:
                sdf[num_batch*num_samples:] = self(obj_idx,grids[num_batch*num_samples:][None,...])[0,:,0]

            sdf = sdf.reshape(self.cfg.res,self.cfg.res,self.cfg.res)*-1
            if 0:
                out_dir = "./test"
                os.makedirs(out_dir,exist_ok=True)
                vis_sdf(grids.cpu().numpy(),sdf.cpu().numpy(),f"{out_dir}/{self.current_epoch:04d}.ply")
            # try:
            if 1:
                # verts, faces, normals, values = measure.marching_cubes(sdf_numpy, 0.0) # todo:use diffcu.
                vertices, triangles = diso.DiffMC().cuda().forward(sdf.cuda())
                out_dir = "./test_mesh"
                os.makedirs(out_dir,exist_ok=True)
                vertices /=self.cfg.res
                new_vertices = (vertices*2-1)*0.9
                mcubes.export_obj(new_vertices.cpu().numpy(),triangles.cpu().numpy(),f"{out_dir}/{self.current_epoch:04d}.obj")
            # except:
            #     print("error cannot marching cubes")
            #     return -1

    ######################
    # DATA RELATED HOOKS #
    ######################
    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader() #  only verify in trainset.

    def test_dataloader(self):
        return self.datamodule.test_dataloader()



    def load_ckpts_own(self,dir):
        print(f'load ckpt from {dir}')
        ckpts = torch.load(dir)
        self.triplane_feat.load_state_dict(ckpts['triplane_state_dict'])
        self.mlp_decoder.load_state_dict(ckpts['decoder_state_dict'])

        self.triplane_feat.cuda()
        self.mlp_decoder.cuda()









