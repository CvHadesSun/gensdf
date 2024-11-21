import torch
import numpy as np
from torch import nn
import os
import torch.nn.functional as F
from torch.nn import init
from .networks import get_embedder,frequency_init,first_layer_sine_init


class MultiTriplane(nn.Module):
    def __init__(self, num_objs, resolution = 256,feature_dim=32,embed_dim=4, noise_val = None, device = 'cuda'):
        super(MultiTriplane,self).__init__()
        self.device = device
        self.num_objs = num_objs
        self.feature_dim_=feature_dim
        self.resolution_ = resolution
        self.triplane_cpu_intermediate = False
        self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(1, feature_dim, resolution, resolution)*0.1) for _ in range(3*num_objs)])
        # self.embeddings = [torch.nn.Embedding(1, 3 * self.feature_dim_ * self.resolution * self.resolution) for i in range(num_objs)]
        # self.embeddings = [torch.nn.Embedding(1, 3 * feature_dim * resolution * resolution) for i in range(num_objs)]
        # self.embeddings = [nn.Parameter(torch.randn(1, feature_dim, resolution, resolution)*0.1) for _ in range(3*num_objs)]
        self.noise_val = noise_val
        self.xyz_embeddings,self.num_channel = get_embedder(embed_dim,0)

        if self.triplane_cpu_intermediate: # todo add into config.
            # We need to store the currently used triplanes on GPU memory, but don't want to load them each time we make a forward pass.
            self.current_embeddings = None  # Embedding object within list of embeddings. Need this intermediate step for gradient to pass through
            self.current_triplanes = None
            self.current_obj_idx = None

        self.use_tanh = False # todo add into config.

        if not self.triplane_cpu_intermediate:
            for embedding in self.embeddings:
                    embedding = embedding.to(self.device)

    def tvreg(self):
        l = 0
        for embed in self.embeddings:
            l += ((embed[:, :, 1:] - embed[:, :, :-1])**2).sum()**0.5
            l += ((embed[:, :, :, 1:] - embed[:, :, :, :-1])**2).sum()**0.5
        return l/self.num_objs
    
    def l2reg(self):
        l = 0
        for embed in self.embeddings:
            l += (embed**2).sum()**0.5
        return l/self.num_objs

    def sample_plane(self, coords2d, plane):

        # assert len(coords2d.shape) == 3, coords2d.shape
        if len(coords2d.shape) == 2:
            coords2d = coords2d[None,...]
        sampled_features = torch.nn.functional.grid_sample(plane,
                                                           coords2d.reshape(coords2d.shape[0], 1, -1, coords2d.shape[-1]),
                                                           mode='bilinear', padding_mode='zeros', align_corners=True)
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.reshape(N, C, H*W).permute(0, 2, 1)

        # assert len(coords2d.shape) == 3, coords2d.shape
        # sampled_features = torch.nn.functional.grid_sample(plane,
        #                                                    coords2d.reshape(coords2d.shape[0], 1, -1, coords2d.shape[-1]),
        #                                                    mode='bilinear', padding_mode='zeros', align_corners=True)
        # N, C, H, W = sampled_features.shape
        # sampled_features = sampled_features.reshape(N, C, H*W).permute(0, 2, 1)

        return sampled_features


    @property
    def embedding_channels(self):
        return self.num_channel
        # return 0

    @property
    def resolution(self):
        return self.resolution_
    
    @property
    def feature_dim(self):
        return self.feature_dim_
    

    def forward(self, obj_idx, coordinates, debug=False):

        batch_tris=[]
        for i in range(coordinates.shape[0]):
            _idx = obj_idx[i]
            xy_embed = self.sample_plane(coordinates[i,:, 0:2], self.embeddings[3*_idx+0])
            yz_embed = self.sample_plane(coordinates[i,:, 1:3], self.embeddings[3*_idx+1])
            xz_embed = self.sample_plane(coordinates[i,:, :3:2], self.embeddings[3*_idx+2])
            tri_features_ = torch.cat([xy_embed, yz_embed, xz_embed],dim=-1)
            batch_tris.append(tri_features_)

        tri_features = torch.cat(batch_tris,dim=0)
        # tri_features = torch.sum(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0) 
        # tri_features = xy_embed + yz_embed + xz_embed
        # tri_features = xy_embed * yz_embed * xz_embed

        embedding_features = self.xyz_embeddings(coordinates)

        all_features = torch.cat([tri_features,embedding_features],dim=-1)

        return all_features
        # import ipdb; ipdb.set_trace()
        # return tri_features
    
    def forward_new(self, obj_idx, coordinates, ray_d=None, debug=False):

        if len(coordinates.shape) == 3:
            batch_size, n_coords, n_dims = coordinates.shape
        elif len(coordinates.shape) == 4:
            batch_size, ray_batch_size, n_coords, n_dims = coordinates.shape
        assert batch_size == obj_idx.shape[0]
        if self.triplane_cpu_intermediate:
            # Move triplane from CPU to GPU. Only happens once per scene.
            if obj_idx != self.current_obj_idx:
                print(f'Moving triplane at obj_idx {obj_idx} from CPU to GPU...')
                self.current_obj_idx = obj_idx
                self.current_embeddings = self.embeddings[obj_idx.to('cpu')].to(self.device)
            
            self.current_triplanes = self.current_embeddings(torch.tensor(0, dtype=torch.int64).to(self.device)).view(batch_size, 3, self.feature_dim_, self.resolution, self.resolution)
            triplanes = self.current_triplanes
        else:
            triplanes = self.embeddings[obj_idx[0]].view(batch_size, 3, self.feature_dim_, self.resolution, self.resolution)
        
        # Use tanh to clamp triplanes
        if self.use_tanh:
            triplanes = torch.tanh(triplanes)

        xy_embed = self.sample_plane(coordinates[..., 0:2], triplanes[:, 0])  # ex: [batch_size, 20000, 64]
        yz_embed = self.sample_plane(coordinates[..., 1:3], triplanes[:, 1])
        xz_embed = self.sample_plane(coordinates[..., :3:2], triplanes[:, 2])

        embedding_features = self.xyz_embeddings(coordinates)
        all_features = torch.cat([xy_embed,yz_embed,xz_embed,embedding_features],dim=-1)
        return all_features

    def tvreg(self):
        l = 0
        for embed in self.embeddings:
            l += ((embed[:, :, 1:] - embed[:, :, :-1])**2).sum()**0.5
            l += ((embed[:, :, :, 1:] - embed[:, :, :, :-1])**2).sum()**0.5
        return l/self.num_objs
    
    def l2reg(self):
        l = 0
        for embed in self.embeddings:
            l += (embed**2).sum()**0.5
        return l/self.num_objs


class MLPDecoder(nn.Module):
    def __init__(self,in_channel,latend_dim=128,last_op=None,device='cuda'):
        super().__init__()

        self.device = device
        self.net = nn.Sequential(
            nn.Linear(in_channel, latend_dim),
            nn.ReLU(),
            nn.Linear(latend_dim, latend_dim),
            nn.ReLU(),
            nn.Linear(latend_dim, latend_dim),
            nn.ReLU(),
            nn.Linear(latend_dim, latend_dim),
            nn.ReLU(),
            nn.Linear(latend_dim, 1),
        ).to(self.device)

        # init mlp
        self.net.apply(frequency_init(30))
        self.net[0].apply(first_layer_sine_init)
        self.last_op = last_op

    def forward(self,feature):
        # 
        x = self.net(feature)

        if self.last_op is not None:
            x = self.last_op(x)
        return x