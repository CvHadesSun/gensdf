import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np
import math

import os 
from pathlib import Path
import time 


from model.triplane_net.networks import TriplaneFeatureNet,SurfaceClassifier
import pytorch_lightning as pl

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
        else:
            raise NotImplementedError()

        self.build_model()

        if opt.loss_fn == "l1":
            self.err_fn = nn.L1Loss(reduction='none')
        elif opt.loss_fn == "mse":
            self.err_fn = nn.MSELoss()
        else:
            raise NotImplementedError()
        



    def build_model(self):
        self.triplane_feat = TriplaneFeatureNet(**self.cfg.triplane,aabb=self.aabb)
        self.mlp_decoder = SurfaceClassifier(self.cfg.mlp_filters,last_op=self.last_op)

    def configure_optimizers(self):
    
        optimizer_feature = torch.optim.Adam(self.triplane_feat.parameters(), self.cfg.lr_feature_init)
        # optimizer_decoder = torch.optim.Adam(self.mlp_decoder.parameters(), self.cfg.lr_decoder_init)
        optimizer_decoder = torch.optim.RMSprop(self.mlp_decoder.parameters(), lr=self.cfg.lr_decoder_init,momentum=0, weight_decay=0)


        feature_scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer_feature, self.cfg.feature_lr_step, self.cfg.feature_lr_gamma)
        decoder_scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer_decoder, self.cfg.decoder_lr_step, self.cfg.decoder_lr_gamma)
        

        self.optimizer_list = [optimizer_feature,optimizer_decoder]
        self.scheduler_list = [feature_scheduler,decoder_scheduler]

        return self.optimizer_list, self.scheduler_list
    
    def forward(self,xyz):
        # query feature from triplane.
        feature_tirplane = self.triplane_feat(xyz)

        feature_tirplane = feature_tirplane.transpose(1,0)
        # decoder
        pred = self.mlp_decoder(feature_tirplane)

        return pred

    def training_step(self, batch, batch_idx):

        
        xyz = batch['xyz'] # [B,N,3]
        label= batch['labels_sdf']

        # forward
        pred = self(xyz)

        # sdf loss
        error = self.err_fn(pred,label)
        # error = self.mse_loss(pred,label)

        print("sdf loss",error.min().item())

        self.log(f"train_loss", error.mean().item())
        # 
        for opt in self.optimizer_list:
            opt.zero_grad()

        error.mean().backward()

        for opt in self.optimizer_list:
            opt.step()

        for scd in self.scheduler_list:
            scd.step()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, *args, **kwargs):
        # todo: test error to log, or save some middle results or resconstruction ?
        pass

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








