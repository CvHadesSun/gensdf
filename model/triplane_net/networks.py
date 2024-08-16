import torch
import numpy as np
from torch import nn
import os
import torch.nn.functional as F


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
    
def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


class TriplaneFeatureNet(nn.Module):

    def __init__(self,resolution,feature_dim,aabb)->None:
        super(TriplaneFeatureNet, self).__init__()
        # todo: add into cfg file.
        self.AABB= aabb
        self.plane_size = resolution
        self.feature_dim = feature_dim

        self.triplane = nn.Parameter(torch.zeros(3, self.feature_dim, self.plane_size, self.plane_size).float().cuda().requires_grad_(True))
        self.init_parameters()
        self.position_encoding, self.num_channel = get_embedder(6, 0)



    def init_parameters(self) -> None:
        # Important for performance
        nn.init.uniform_(self.triplane, -1e-2, 1e-2)


    def forward(self, p):
        _texc = (p.view(-1, 3) - self.AABB[0][None, ...]) / (self.AABB[1][None, ...] - self.AABB[0][None, ...])
        xyz =  2. * torch.clamp(_texc, min=0, max=1)-1
        xyz_contraction = xyz[None,None,:,:]
        # xyz_contraction = 2. * xyz_contraction - 1.
        xy_plane_feature = F.grid_sample(self.triplane[0:1], xyz_contraction[:,:,:,:2], padding_mode='border', align_corners=True, mode='bilinear').squeeze(0).view(self.triplane.shape[1], -1).transpose(1,0)
        yz_plane_feature = F.grid_sample(self.triplane[1:2], xyz_contraction[:,:,:,1:], padding_mode='border', align_corners=True, mode='bilinear').squeeze(0).view(self.triplane.shape[1], -1).transpose(1,0)
        xz_plane_feature = F.grid_sample(self.triplane[2:3], torch.cat((xyz_contraction[:,:,:,0:1],xyz_contraction[:,:,:,2:3]), dim=-1), padding_mode='border', align_corners=True, mode='bilinear').squeeze(0).view(self.triplane.shape[1], -1).transpose(1,0)
        triplane_features = torch.cat((xy_plane_feature, yz_plane_feature, xz_plane_feature, self.position_encoding(xyz_contraction[0,0])), dim=-1)
        return triplane_features
    
class SurfaceClassifier(nn.Module):
    def __init__(self, filter_channels, no_residual=True, last_op=None):
        super(SurfaceClassifier, self).__init__()
        self.filters = []
        self.num_views = 1
        self.no_residual = no_residual
        filter_channels = filter_channels

        self.last_op = last_op


        if self.no_residual:
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1))
                self.add_module("conv%d" % l, self.filters[l])
        else:
            for l in range(0, len(filter_channels) - 1):
                if 0 != l:
                    self.filters.append(
                        nn.Conv1d(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1],
                            1))
                else:
                    self.filters.append(nn.Conv1d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        1))

                self.add_module("conv%d" % l, self.filters[l])

    def forward(self, feature):
        '''
        :param feature: list of [BxC_inxHxW] tensors of  features
        :param xy: [Bx3xN] tensor of (x,y) coodinates in the  plane
        :return: [BxC_outxN] tensor of features extracted at the coordinates
        '''

        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters):
            if self.no_residual:
                y = self._modules['conv' + str(i)](y)
            else:
                y = self._modules['conv' + str(i)](
                    y if i == 0
                    else torch.cat([y, tmpy], 1)
                )
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)

            if self.num_views > 1 and i == len(self.filters) // 2:
                y = y.view(
                    -1, self.num_views, y.shape[1], y.shape[2]
                ).mean(dim=1)
                tmpy = feature.view(
                    -1, self.num_views, feature.shape[1], feature.shape[2]
                ).mean(dim=1)

        if self.last_op:
            y = self.last_op(y)

        return y
    


