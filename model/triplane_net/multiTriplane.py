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
from model.triplane_net.kplanes import Triplane,Network
import mcubes

import logging

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)



logger = logging.getLogger("TriplaneSDF")
file_handler = logging.FileHandler("tri-sdf.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger.setLevel(logging.DEBUG) 
logger.addHandler(file_handler)

class TriplaneSDF(pl.LightningModule):
    def __init__(self, opt, datamodule,is_train=True):
        super().__init__()
        self.automatic_optimization = False
        self.is_train = is_train

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
            self.loss_fn = nn.L1Loss()
        elif opt.loss_fn == "mse":
            self.loss_fn = nn.MSELoss()
        elif opt.loss_fn == "bce":
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError()

        if opt.train_triplane:
            self.triplane_feat.train()
        else:
            self.triplane_feat.eval()

        if opt.train_decoder:
            self.mlp_decoder.train()
        else:
            self.mlp_decoder.eval()

        # if self.cfg.train_sdf:
        logger.info(f"training the process with sdf {self.cfg.train_sdf}")

    def build_model_v1(self):
        self.triplane_feat = MultiTriplane(num_objs=self.num_objs,**self.cfg.triplane)
        in_channls = self.triplane_feat.feature_dim*3 + self.triplane_feat.embedding_channels
        self.mlp_decoder = MLPDecoder(in_channls,latend_dim=self.cfg.deocder.latend_dim,last_op=self.last_op)

    def build_model_v2(self):
        if self.is_train:
            self.triplane_feat = Triplane(n=self.num_objs,reso=self.cfg.triplane.resolution // (2 ** len(self.cfg.c2f_scale)))
        else:
            self.triplane_feat = Triplane(n=self.num_objs,reso=self.cfg.triplane.resolution)
        # in_channls = self.triplane_feat.feature_dim*3 + self.triplane_feat.embedding_channels
        self.mlp_decoder = Network()

    def build_model(self):
        # self.build_model_v1()
        self.build_model_v2()

    def configure_optimizers_v1(self):
        self.optimizer_list=[]
        self.scheduler_list=[]
        if self.cfg.train_triplane:
            optimizer_feature = torch.optim.Adam(self.triplane_feat.parameters(), self.cfg.lr_feature_init)
            feature_scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer_feature, self.cfg.feature_lr_step, self.cfg.feature_lr_gamma)
            self.optimizer_list.append(optimizer_feature)
            self.scheduler_list.append(feature_scheduler)

        if self.cfg.train_decoder:
            optimizer_decoder = torch.optim.Adam(self.mlp_decoder.parameters(), self.cfg.lr_decoder_init)
            decoder_scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer_decoder, self.cfg.decoder_lr_step, self.cfg.decoder_lr_gamma)
        
            self.optimizer_list.append(optimizer_decoder)
            self.scheduler_list.append(decoder_scheduler)

        return self.optimizer_list, self.scheduler_list
    
    def configure_optimizers_v2(self,net=False,triplane=True):   
        params_to_train = []
        if net:
            params_to_train += [{'name':'net', 'params':self.mlp_decoder.parameters(), 'lr':1e-3}]
        if triplane:
            params_to_train += [{'name':'tri', 'params':self.triplane_feat.parameters(), 'lr':5e-2}]

        self.optimizer_list = [torch.optim.Adam(params_to_train)]
        return self.optimizer_list, []
    
    def configure_optimizers(self):
        return self.configure_optimizers_v2(net=self.cfg.train_decoder,triplane=self.cfg.train_triplane)


    def forward(self,obj_idx,xyz):
        # query feature from triplane.
        feature_tirplane = self.triplane_feat(obj_idx,xyz)
        # decoder
        pred = self.mlp_decoder(feature_tirplane)

        return pred
    
    def training_step(self,batch,batch_idx):
        # if self.cfg.version == 0:
        #     self.training_step_v0(batch,batch_idx)
        # elif self.cfg.version == 1:
        #     self.training_step_v1(batch,batch_idx)
        # else:
        #     raise NotImplementedError()

        self.training_step_v2(batch,batch_idx)


    def training_step_v0(self, batch, batch_idx):
        # if self.current_epoch > self.cfg.fix_mlp_util:
        #     self.mlp_decoder.eval() # fix the decoder weights.

        obj_idx = batch['obj_idx']
        xyz = batch['xyz'] # [B,N,3]

        if self.cfg.train_sdf:
            label= batch['labels_sdf']
        else:
            label = batch["labels_01"]
        # forward
        pred = self(obj_idx,xyz)
        if self.cfg.train_sdf:
            clamp_value = self.cfg.clamped
            pred = torch.clamp(pred, -1*clamp_value, clamp_value)
            label = torch.clamp(label, -1*clamp_value, clamp_value)
        # sdf loss
        # loss = self.loss_fn(pred,label)
        loss = (pred - label).pow(2).mean()

        self.log(f"train_loss", loss.item(),prog_bar=True)
        if batch_idx ==0 :
            logger.info(f"Epoch {self.current_epoch}, Loss: {loss.item()}'")
        # 
        for opt in self.optimizer_list:
            opt.zero_grad()

        loss.backward()

        for opt in self.optimizer_list:
            opt.step()

        for scd in self.scheduler_list:
            # print(scd.get_lr())
            scd.step()

        if batch_idx == 0:
            if self.current_epoch in self.cfg.save_epochs:
                    # print(f'Saving checkpoint at step {step}')
                    os.makedirs(f"checkpoints",exist_ok=True)
                    torch.save({
                        'epoch': self.current_epoch,
                        'triplane_state_dict': self.triplane_feat.state_dict(),
                        'decoder_state_dict': self.mlp_decoder.state_dict(),
                        # 'optimizer_state_dict': self.optimizer_list[0].state_dict(),
                        'loss': loss.item(),
                    }, f'checkpoints/model_epoch_{self.current_epoch}_loss_{loss.item():.4f}.pt')

                    logger.info(f"Epoch {self.current_epoch}: save the ckpt into:checkpoints/model_epoch_{self.current_epoch}_loss_{loss.item():.4f}.pt")

    def training_step_v1(self, batch, batch_idx):
        # if self.current_epoch > self.cfg.fix_mlp_util:
        #     self.mlp_decoder.eval() # fix the decoder weights.

        obj_idx = batch['obj_idx']
        xyz = batch['xyz'] # [B,N,3]

        if self.cfg.train_sdf:
            label= batch['labels_sdf']
        else:
            label = batch["labels_01"]
        # forward
        pred = self(obj_idx,xyz)
        if self.cfg.train_sdf:
            clamp_value = self.cfg.clamped
            pred = torch.clamp(pred, -1*clamp_value, clamp_value)
            label = torch.clamp(label, -1*clamp_value, clamp_value)

        # sdf loss
        loss = self.loss_fn(pred, label.reshape((label.shape[0], label.shape[1], -1)))

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

        if batch_idx == 0:
            if self.current_epoch in self.cfg.save_epochs:
                    # print(f'Saving checkpoint at step {step}')
                    os.makedirs("checkpoints",exist_ok=True)
                    torch.save({
                        'epoch': self.current_epoch,
                        'triplane_state_dict': self.triplane_feat.state_dict(),
                        'decoder_state_dict': self.mlp_decoder.state_dict(),
                        'optimizer_state_dict': self.optimizer_list[0].state_dict(),
                        'loss': loss.item(),
                    }, f'checkpoints/model_epoch_{self.current_epoch}_loss_{loss.item():.4f}.pt')

    def training_step_v2(self, batch, batch_idx):

        # "obj_idx": idx,
        # 'surf_xyz': torch.from_numpy(surf_xyz).float(),
        # 'surf_normal': torch.from_numpy(surf_normal).float(),
        # 'vol_xyz': torch.from_numpy(vol_xyz).float(),
        # 'vol_sdf': torch.from_numpy(vol_sdf).float()

        # random_pts = torch.uniform(-1,1,(10_0000,3)).cuda().unsqueeze(0)
        c2f_scale = self.cfg.c2f_scale
        if self.current_epoch in c2f_scale:
            new_reso = int(self.cfg.triplane.resolution  / (2 ** (len(c2f_scale) - c2f_scale.index(self.current_epoch) - 1)))
            self.triplane_feat.update_resolution(new_reso)
            self.configure_optimizers_v2(net=self.cfg.train_decoder,triplane=self.cfg.train_triplane)
            update_lr(self.optimizer_list[0], self.current_epoch - 1,max_epoch=self.cfg.max_epoch)
            torch.cuda.empty_cache()

        theta=1e-6
        beta = 5.0


        obj_idx = batch['obj_idx']
        surf_xyz = batch['surf_xyz']
        surf_normal = batch['surf_normal']
        vol_xyz = batch['vol_xyz']
        vol_sdf = batch['vol_sdf']

        lw1 = 100.0
        lw2=50.0
        lw3=5.0
        lw4=2.0

        loss = 0

        # loss += self.triplane_feat.tvreg() * 1e-2
        # loss += self.triplane_feat.l2reg() * 1e-3
        # loss 1
        pred_surface_sdf = self(obj_idx,surf_xyz)
        loss1 = pred_surface_sdf.abs().mean() 
        # loss1 = (0.5*torch.exp(beta*pred_surface_sdf.abs())-0.5).mean() * lw1
        # loss1=0

        # loss 2
        pred_vol_sdf = self(obj_idx,vol_xyz)
        # pred_vol_sdf = torch.clamp(pred_vol_sdf,-0.1,0.1)
        # vol_sdf = torch.clamp(vol_sdf,-0.1,0.1)
        loss2 = (pred_vol_sdf - vol_sdf).abs().mean() 
        # loss2 = (0.5*torch.exp((pred_vol_sdf - vol_sdf).abs())-0.5).mean() * lw2
        # loss2 = (pred_vol_sdf - vol_sdf).pow(2).mean() * lw2
        # import ipdb;ipdb.set_trace()

        # loss 3 
        grad_xyz= self.construct_grad_v3(surf_xyz,theta,obj_idx)
        # normed_grad = grad_xyz  / (grad_xyz.norm(dim=-1,keepdim=True) + 1e-6)
        
        # import ipdb;ipdb.set_trace()
        loss3 = (grad_xyz - surf_normal.squeeze(0)).norm(2,dim=1).mean()
        # todo test normal loss.
        # import ipdb;ipdb.set_trace()

        # loss 4
        # grad_xyz_rdn = self.construct_grad(vol_xyz,theta,obj_idx)
        loss4 = ((grad_xyz.norm(2,dim=-1) - 1)**2).mean()
        # import ipdb;ipdb.set_trace()

        loss = lw1* loss1 + lw2* loss2  + lw3 * loss3 + loss4 *lw4
        # self.log(f"train_loss", loss.item(),prog_bar=True)

        # if batch_idx ==0 :
        #     logger.info(f"Epoch {self.current_epoch}, Loss: {loss.item()}'")


        self.log(f"train_loss", loss.item(),prog_bar=True)
        if batch_idx ==0 :
            logger.info(f"Epoch {self.current_epoch}, Loss: {loss.item()}'")
        # 
        for opt in self.optimizer_list:
            opt.zero_grad()

        loss.backward()

        for opt in self.optimizer_list:
            opt.step()

        try:
            for scd in self.scheduler_list:
                # print(scd.get_lr())
                scd.step()
        except:
            pass

        update_lr(self.optimizer_list[0],self.current_epoch,max_epoch=self.cfg.max_epoch)

        self.loss_it = loss.item()


        # if self.current_epoch ==600:
        #     vis_model(self.mlp_decoder, self.triplane_feat, 1, '.')
            # save_model(net, triplane, '.')

        # # if batch_idx == 0:
        # if self.trainer.is_global_zero:
        #     if self.current_epoch in self.cfg.save_epochs:
        #             # print(f'Saving checkpoint at step {step}')
        #             os.makedirs(f"checkpoints",exist_ok=True)
        #             torch.save({
        #                 'epoch': self.current_epoch,
        #                 'triplane_state_dict': self.triplane_feat.state_dict(),
        #                 'decoder_state_dict': self.mlp_decoder.state_dict(),
        #                 # 'optimizer_state_dict': self.optimizer_list[0].state_dict(),
        #                 'loss': loss.item(),
        #             }, f'checkpoints/model_epoch_{self.current_epoch}_loss_{loss.item():.4f}.pt')

        #             logger.info(f"Epoch {self.current_epoch}: save the ckpt into:checkpoints/model_epoch_{self.current_epoch}_loss_{loss.item():.4f}.pt")
    
    def on_train_epoch_end(self):
        """
        在每个 epoch 结束时保存最后一个 step 的 checkpoint
        """
        if self.trainer.is_global_zero and self.current_epoch in self.cfg.save_epochs:
            save_path = f'checkpoints/model_epoch_{self.current_epoch}_{self.loss_it:0.4f}.pt'
            os.makedirs("checkpoints", exist_ok=True)
            
            torch.save({
                'epoch': self.current_epoch,
                'triplane_state_dict': self.triplane_feat.state_dict(),
                'decoder_state_dict': self.mlp_decoder.state_dict(),
            }, save_path)

            logger.info(f"Checkpoint saved at {save_path}")
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx,*args, **kwargs):
        # todo: test error to log, or save some middle results or resconstruction ?
        if not self.cfg.val_mesh:
            return
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

            sdf = sdf.reshape(self.cfg.res,self.cfg.res,self.cfg.res)# *-1
            if 0:
                out_dir = f"./test"
                os.makedirs(out_dir,exist_ok=True)
                vis_sdf(grids.cpu().numpy(),sdf.cpu().numpy(),f"{out_dir}/{self.current_epoch:04d}.ply")
            try:
            # if 1:
                # vertices, triangles, normals, values = measure.marching_cubes(sdf.cpu().numpy(), 0.0) # todo:use diffcu.
                # vertices, triangles = diso.DiffMC().cuda().forward(sdf.cuda()-2/self.cfg.res)
                vertices, triangles = diso.DiffMC().cuda().forward(sdf.cuda())
                out_dir = f"./val_mesh"
                os.makedirs(out_dir,exist_ok=True)
                vertices /=self.cfg.res
                new_vertices = (vertices*2-1)*0.9
                mcubes.export_obj(new_vertices,triangles,f"{out_dir}/{self.current_epoch:04d}.obj")
                # mcubes.export_obj(new_vertices.cpu().numpy(),triangles.cpu().numpy(),f"{out_dir}/{self.current_epoch:04d}.obj")
                logger.info(f"Epoch {self.current_epoch} output val mesh idx: {obj_idx}")
            except:
                # print("error cannot marching cubes")
                logger.error(f"Epoch {self.current_epoch} error cannot marching cubes，val mesh idx: {obj_idx}")
                return -1

    @torch.no_grad()
    def test(self, obj_idx):
        # todo: test error to log, or save some middle results or resconstruction ?
        # if batch_idx == 0:
        if 1:
            grids = create_cube(self.cfg.res)

            grids = grids.cuda()

            num_pts = grids.shape[0]

            # grids_np = np.load('/home/wanhu/workspace/gensdf/data/vehicle/dataset/000/samples.npy')[:,:3]

            # grids = torch.from_numpy(grids_np).cuda().float()

            # num_pts=  grids.shape[0]

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

            # out_data = np.concatenate([grids.cpu().numpy(),sdf.cpu().numpy().reshape(-1,1)],axis=1)
            # np.save('/home/wanhu/workspace/gensdf/data/vehicle/dataset/000/predict.npy',out_data)
            # exit()

            sdf = sdf.reshape(self.cfg.res,self.cfg.res,self.cfg.res)#*-1
            if 0:
                out_dir = "./test"
                os.makedirs(out_dir,exist_ok=True)
                vis_sdf(grids.cpu().numpy(),sdf.cpu().numpy(),f"{out_dir}/{self.current_epoch:04d}.ply")
            # try:
            if 0:
                # sdf = torch.clamp(sdf,-0.05,0.05)
                sdf_numpy = sdf.cpu().numpy()
                print(sdf_numpy.min(),sdf_numpy.max())
                verts, faces, normals, values = measure.marching_cubes(sdf_numpy, 0.0) # todo:use diffcu.
                # vertices, triangles = diso.DiffMC().cuda().forward(sdf.cuda()-2/self.cfg.res)
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                if 0:
                    components = mesh.split(only_watertight=False)
                    bbox = []
                    for c in components:
                        bbmin = c.vertices.min(0)
                        bbmax = c.vertices.max(0)
                        bbox.append((bbmax - bbmin).max())
                    max_component = np.argmax(bbox)
                    mesh = components[max_component]
                # mesh.export(filename)
                out_dir = "./test_mesh"
                os.makedirs(out_dir,exist_ok=True)
                verts = mesh.vertices
                verts /=self.cfg.res
                new_vertices = (verts*2-1)*0.9
                # mcubes.export_obj(new_vertices.cpu().numpy(),triangles.cpu().numpy(),f"{out_dir}/{self.current_epoch:04d}_{obj_idx[0]}.obj")
                # mcubes.export_obj(new_vertices,faces,f"{out_dir}/{self.current_epoch:04d}_{obj_idx[0]}.obj")
                mesh.vertices = new_vertices
                mesh.export(f"{out_dir}/{self.current_epoch:04d}_{obj_idx[0]}.obj")
            # except:
            #     print("error cannot marching cubes")
            #     return -1

        if 1:
            vis_model(self.mlp_decoder, self.triplane_feat, 1, './tes_recon',oid=obj_idx)

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
        ckpts = torch.load(dir,map_location='cpu')

        self.triplane_feat.load_state_dict(ckpts['triplane_state_dict'],)
        # self.mlp_decoder.load_state_dict(ckpts['decoder_state_dict'])

        self.triplane_feat.cuda()
        self.mlp_decoder.cuda()


    def load_decoder(self):

        # dir = f"/home/wanhu/workspace/gensdf/outputs/decoder_1.0.pt"
        dir = f"/DATA/local4T_0/wanhu/gensdf/decoder/decoder_100_obj.pt"

        print(f"load decoder weights from {dir}")
        ckpts = torch.load(dir,map_location='cpu')

        self.mlp_decoder.load_state_dict(ckpts)
        # self.triplane_feat.load_state_dict(ckpts['triplane_state_dict'],)
        self.triplane_feat.cuda()
        self.mlp_decoder.cuda()
    
    def construct_grad(self,surf_xyz,theta,obj_idx):
        num_pts = surf_xyz.shape[1]
        axis_x = surf_xyz[0]
        axis_x_p = axis_x + torch.tensor([theta,0,0]).cuda()
        axis_x_n = axis_x - torch.tensor([theta,0,0]).cuda()

        axis_y = surf_xyz[0]
        axis_y_p = axis_y + torch.tensor([0,theta,0]).cuda()
        axis_y_n = axis_y - torch.tensor([0,theta,0]).cuda()

        axis_z = surf_xyz[0]
        axis_z_p = axis_z + torch.tensor([0,0,theta]).cuda()
        axis_z_n = axis_z - torch.tensor([0,0,theta]).cuda()
        
        # import ipdb;ipdb.set_trace()

        axis_points = torch.cat([axis_x_p,axis_x_n,axis_y_p,axis_y_n,axis_z_p,axis_z_n],dim=0)[None,...]

        pred_axis_sdf = self(obj_idx,axis_points)

        # split the axis points

        pred_axis_x_p_sdf = pred_axis_sdf[0,:num_pts]
        pred_axis_x_n_sdf = pred_axis_sdf[0,num_pts:num_pts*2]

        pred_axis_y_p_sdf = pred_axis_sdf[0,num_pts*2:num_pts*3]
        pred_axis_y_n_sdf = pred_axis_sdf[0,num_pts*3:num_pts*4]

        pred_axis_z_p_sdf = pred_axis_sdf[0,num_pts*4:num_pts*5]
        pred_axis_z_n_sdf = pred_axis_sdf[0,num_pts*5:num_pts*6]

        grad_x = (pred_axis_x_p_sdf - pred_axis_x_n_sdf) / (2*theta)
        grad_y = (pred_axis_y_p_sdf - pred_axis_y_n_sdf) / (2*theta)
        grad_z = (pred_axis_z_p_sdf - pred_axis_z_n_sdf) / (2*theta)

        grad_xyz = torch.cat([grad_x,grad_y,grad_z],dim=1)[None,...] # [n,3]

        return grad_xyz
    
    def construct_grad_v2(self,surf_xyz,theta,obj_idx):
        axis_xyz = surf_xyz[0]

        axis_xyz_p = axis_xyz + torch.tensor([theta,theta,theta]).cuda()
        axis_xyz_n = axis_xyz - torch.tensor([theta,theta,theta]).cuda()  

        p_sdf = self(obj_idx,axis_xyz_p[None,...])
        n_sdf = self(obj_idx,axis_xyz_n[None,...])

        grad_xyz = (p_sdf - n_sdf) / (2*theta)

        return grad_xyz
    
    def construct_grad_v3(self,mnfd_points,eps_s,obj_idx):
        mnfd_points = mnfd_points.squeeze(0)

        points_all = torch.cat([
            mnfd_points + torch.as_tensor([[eps_s, 0.0, 0.0]]).to(mnfd_points),
            mnfd_points + torch.as_tensor([[-eps_s, 0.0, 0.0]]).to(mnfd_points),
            mnfd_points + torch.as_tensor([[0.0, eps_s, 0.0]]).to(mnfd_points),
            mnfd_points + torch.as_tensor([[0.0, -eps_s, 0.0]]).to(mnfd_points),
            mnfd_points + torch.as_tensor([[0.0, 0.0, eps_s]]).to(mnfd_points),
            mnfd_points + torch.as_tensor([[0.0, 0.0, -eps_s]]).to(mnfd_points),
            # rndm_points + torch.as_tensor([[eps_v, 0.0, 0.0]]).to(points),
            # rndm_points + torch.as_tensor([[-eps_v, 0.0, 0.0]]).to(points),
            # rndm_points + torch.as_tensor([[0.0, eps_v, 0.0]]).to(points),
            # rndm_points + torch.as_tensor([[0.0, -eps_v, 0.0]]).to(points),
            # rndm_points + torch.as_tensor([[0.0, 0.0, eps_v]]).to(points),
            # rndm_points + torch.as_tensor([[0.0, 0.0, -eps_v]]).to(points),
        ], dim=0).unsqueeze(0)  # [b,N,3]


        sdfs_all = self(obj_idx,points_all).squeeze(0) # [b,N,3]
        len_d = mnfd_points.shape[0]

        
        mnfd_grad = torch.cat([
            0.5 * (sdfs_all[0:len_d] - sdfs_all[len_d:2*len_d]) / eps_s,
            0.5 * (sdfs_all[2*len_d:3*len_d] - sdfs_all[3*len_d:4*len_d]) / eps_s,
            0.5 * (sdfs_all[4*len_d:5*len_d] - sdfs_all[5*len_d:]) / eps_s,
        ], dim=-1) # [N,3]

        # import ipdb;ipdb.set_trace()

        return mnfd_grad
    
def update_lr(optimizer, epoch,max_epoch=600):
    learning_factor = (np.cos(np.pi * epoch / max_epoch) + 1.0) * 0.5 * (1 - 0.001) + 0.001
    for param_group in optimizer.param_groups:
        if "net" in param_group['name']:
            param_group['lr'] = 1e-3 * learning_factor
        if "tri" in param_group['name']:
            param_group['lr'] = 5e-2 * learning_factor



def vis_model(net, triplane, n_labels, savedir, oid=0, rank=0):
    os.makedirs(savedir, exist_ok=True)
    for pid in range(n_labels):
        plot_shape(net, triplane, triplane.R * 4, n_labels, 0.0, os.path.join(savedir, f"triplane_{oid}.ply"),pid, oid)
def plot_shape(net, triplane, resolution, channel, threshold, savedir, pid, oid):
    u = extract_fields(
        bound_min=[-1.0, -1.0, -1.0],
        bound_max=[ 1.0,  1.0,  1.0],
        resolution=resolution,
        query_func=lambda xyz: -net(triplane( oid,xyz.unsqueeze(0))),
        channel=channel,
    )
    if pid<0:
        u = np.max(u, -1)  # sdf of scene
    else:
        u = u[..., pid]  # sdf of part
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    vertices = vertices / (resolution - 1.0) * 2 - 1
    mesh = trimesh.Trimesh(vertices, triangles)
    mesh.export(savedir)

def extract_fields(bound_min, bound_max, resolution, query_func, channel):
    N = 128 # 64. Change it when memory is insufficient!
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution, channel], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).cuda()
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs), channel).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u