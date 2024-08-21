import torch
import numpy as np
from torch import nn
import os
import torch.nn.functional as F
from torch.nn import init


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
        # self.init_parameters()
        self.position_encoding, self.num_channel = get_embedder(4, 0)



    def init_parameters(self) -> None:
        # Important for performance
        # nn.init.uniform_(self.triplane, -1e-2, 1e-2)
        nn.init.normal_(self.triplane, mean=0.0, std=0.1)


    def forward(self, p):
        # print("the triplane feature,",self.triplane.min(),self.triplane.max())
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

        # self.init_parameters()

    
    def init_parameters(self) -> None:
        # Important for performance
        # nn.init.uniform_(self.triplane, -1e-2, 1e-2)

        for layer in self.filters:
            nn.init.normal_(layer.weight, mean=0.0, std=0.1)

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
    


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


class MultiTriplane(nn.Module):
    def __init__(self, resolution,feature_dim,num_objs)->None:
        super().__init__()

        self.num_objs = num_objs
        self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(1, feature_dim, resolution, resolution)*0.001) for _ in range(3*num_objs)])

    def sample_plane(self, coords2d, plane):
        assert len(coords2d.shape) == 3, coords2d.shape
        sampled_features = torch.nn.functional.grid_sample(plane,
                                                           coords2d.reshape(coords2d.shape[0], 1, -1, coords2d.shape[-1]),
                                                           mode='bilinear', padding_mode='zeros', align_corners=True)
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.reshape(N, C, H*W).permute(0, 2, 1)
        return sampled_features
    
    def forward(self, obj_idx, coordinates, debug=False):

        xy_embed = self.sample_plane(coordinates[..., 0:2], self.embeddings[3*obj_idx+0])
        yz_embed = self.sample_plane(coordinates[..., 1:3], self.embeddings[3*obj_idx+1])
        xz_embed = self.sample_plane(coordinates[..., :3:2], self.embeddings[3*obj_idx+2])
        
        #if self.noise_val != None:
        #    xy_embed = xy_embed + self.noise_val*torch.empty(xy_embed.shape).normal_(mean = 0, std = 0.5).to(self.device)
        #    yz_embed = yz_embed + self.noise_val*torch.empty(yz_embed.shape).normal_(mean = 0, std = 0.5).to(self.device)
        #    xz_embed = xz_embed + self.noise_val*torch.empty(xz_embed.shape).normal_(mean = 0, std = 0.5).to(self.device)

                
        features = torch.sum(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0) 
        # if self.noise_val != None and self.training:
        #     features = features + self.noise_val*torch.empty(features.shape).normal_(mean = 0, std = 0.5).to(self.device)
        return features
    
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



class FourierFeatureTransform(nn.Module):
    def __init__(self, num_input_channels, mapping_size, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size)) * scale, requires_grad=False)

    def forward(self, x):
        B, N, C = x.shape
        x = (x.reshape(B*N, C) @ self._B).reshape(B, N, -1)
        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
    

class MLPDecoder:
    def __init__(self,):
        super().__init__()
        # self.net = nn.Sequential(
        #     FourierFeatureTransform(32, 64, scale=1),
        #     nn.Linear(128, 128),
        #     nn.ReLU(inplace=True),
            
        #     nn.Linear(128, 128),
        #     nn.ReLU(inplace=True),

        #     nn.Linear(128, 1),
        # )

        self.net = nn.Sequential(
            # https://arxiv.org/abs/2006.10739 - Fourier FN

            nn.Linear(self.channels + self.view_embed_dim, 128),
            nn.ReLU(),
            
            nn.Linear(128, 128),
            nn.ReLU(),
            
            nn.Linear(128, 1),
        ).to(self.device)

    def forward(self,feature):
        # 
        return self.net(feature)
def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            # if hasattr(m, 'weight'):
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
    return init

def first_layer_sine_init(m):
    with torch.no_grad():
        # if hasattr(m, 'weight'):
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)

# For single-scene fitting
class CartesianPlaneNonSirenEmbeddingNetwork(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, aggregate_fn='cat'):  # vs sum
        super().__init__()

        self.aggregate_fn = aggregate_fn
        print(f'Using aggregate_fn {aggregate_fn}')
        self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(1, 32, 256, 256)*0.1) for _ in range(3)])
        self.position_encoding, self.num_channel = get_embedder(4, 0)
        self.net = nn.Sequential(
            # https://arxiv.org/abs/2006.10739

            nn.Linear(32*3+self.num_channel, 128),
            nn.ReLU(),
            
            nn.Linear(128, 128),
            nn.ReLU(),
            
            nn.Linear(128, output_dim),
        )
        self.net.apply(frequency_init(30))
        self.net[0].apply(first_layer_sine_init)


    def sample_plane(self, coords2d, plane):
        assert len(coords2d.shape) == 3, coords2d.shape
        sampled_features = torch.nn.functional.grid_sample(plane,
                                                           coords2d.reshape(coords2d.shape[0], 1, -1, coords2d.shape[-1]),
                                                           mode='bilinear', padding_mode='zeros', align_corners=True)
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.reshape(N, C, H*W).permute(0, 2, 1)
        return sampled_features

    def forward(self, coordinates, debug=False):
        # batch_size, n_coords, n_dims = coordinates.shape
        
        xy_embed = self.sample_plane(coordinates[..., 0:2], self.embeddings[0])  # ex: [1, 20000, 64]
        yz_embed = self.sample_plane(coordinates[..., 1:3], self.embeddings[1])
        xz_embed = self.sample_plane(coordinates[..., :3:2], self.embeddings[2])
        # (M, C)
        
        # aggregate - product or sum?
        if self.aggregate_fn == 'sum':
            features = torch.sum(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0)
        elif self.aggregate_fn == 'prob':
            features = torch.prod(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0)
        elif self.aggregate_fn == 'cat':
            features = torch.cat([xy_embed, yz_embed, xz_embed],dim=-1)
        else:
            raise NotImplementedError()

        pts_embedding = self.position_encoding(coordinates)

        all_feature = torch.cat([features,pts_embedding],dim=-1)

        # todo concate triplane feature.
        # (M, C)
        # decoder

        x = self.net(all_feature)
        # y = torch.tanh(x)
        # x = torch.sigmoid(x)
        # x = torch.nn.Sigmoid()(x)
        # x = self.net(features)
        return x