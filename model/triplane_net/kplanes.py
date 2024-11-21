import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Triplane(nn.Module):
    def __init__(self,
                 n=1,
                 reso=256,
                 channel=32,
                 init_type="geo_init",
                 objname=None,
                 ):
        super().__init__()
        self.n = n
        self.objname = objname
        # assert len(self.objname) == n
        if init_type == "geo_init":
            sdf_proxy = nn.Sequential(
                nn.Linear(3, channel), nn.Softplus(beta=100),
                nn.Linear(channel, channel),
            )
            torch.nn.init.constant_(sdf_proxy[0].bias, 0.0)
            torch.nn.init.normal_(sdf_proxy[0].weight, 0.0, np.sqrt(2) / np.sqrt(channel))
            torch.nn.init.constant_(sdf_proxy[2].bias, 0.0)
            torch.nn.init.normal_(sdf_proxy[2].weight, 0.0, np.sqrt(2) / np.sqrt(channel))

            ini_sdf = torch.zeros([3, channel, reso, reso])
            X = torch.linspace(-1.0, 1.0, reso)
            (U, V) = torch.meshgrid(X, X, indexing="ij")
            Z = torch.zeros(reso, reso)
            inputx = torch.stack([Z, U, V], -1).reshape(-1, 3)
            inputy = torch.stack([U, Z, V], -1).reshape(-1, 3)
            inputz = torch.stack([U, V, Z], -1).reshape(-1, 3)
            ini_sdf[0] = sdf_proxy(inputx).permute(1, 0).reshape(channel, reso, reso)
            ini_sdf[1] = sdf_proxy(inputy).permute(1, 0).reshape(channel, reso, reso)
            ini_sdf[2] = sdf_proxy(inputz).permute(1, 0).reshape(channel, reso, reso)

            self.triplane = torch.nn.Parameter(ini_sdf.unsqueeze(0).repeat(self.n, 1, 1, 1, 1) / 3, requires_grad=True) #
        elif init_type == "zero_init":
            self.triplane = torch.nn.Parameter(torch.zeros([self.n, 3, channel, reso, reso]), requires_grad=True)

        self.R = reso
        self.C = channel
        self.register_buffer("plane_axes", torch.tensor(
            [[[0, 1, 0],
              [1, 0, 0],
              [0, 0, 1]],
             [[0, 0, 1],
              [1, 0, 0],
              [0, 1, 0]],
             [[0, 1, 0],
              [0, 0, 1],
              [1, 0, 0]]], dtype=torch.float32)
                             )

        # xy xz yz

    def project_onto_planes(self, xyz):
        M, _ = xyz.shape
        xyz = xyz.unsqueeze(0).expand(3, -1, -1).reshape(3, M, 3)
        inv_planes = torch.linalg.inv(self.plane_axes).reshape(3, 3, 3)
        projections = torch.bmm(xyz, inv_planes)
        return projections[..., :2]  # [3, M, 2]

    def forward(self, oid, xyz):
        # pts: [M,3]
        xyz = xyz.squeeze(0)  # [1,M,3]
        M, _ = xyz.shape
        plane_features = self.triplane[oid:oid + 1].view(3, self.C, self.R, self.R)
        projected_coordinates = self.project_onto_planes(xyz).unsqueeze(1)
        feats = F.grid_sample(
            plane_features,  # [3,C,R,R]
            projected_coordinates.float(),  # [3,1,M,2]
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        )  # [3,C,1,M]
        feats = feats.permute(0, 3, 2, 1).reshape(3, M, self.C).sum(0)
        return feats.unsqueeze(0)  # [M,C]

    def update_resolution(self, new_reso):
        old_tri = self.triplane.data.view(self.n * 3, self.C, self.R, self.R)
        new_tri = F.interpolate(old_tri, size=(new_reso, new_reso), mode='bilinear', align_corners=True)
        self.R = new_reso
        self.triplane = torch.nn.Parameter(new_tri.view(self.n, 3, self.C, self.R, self.R), requires_grad=True)

class Network(nn.Module):
    def __init__(self,
                 d_in=32,
                 d_hid=128,
                 n_layers=3,
                 d_out=1,
                 init_type="geo_init",
                 weight_norm=True,
                 bias=0.5,
                 inside_outside=False
                 ):
        super().__init__()
        dims = [d_in] + [d_hid for _ in range(n_layers)] + [d_out]
        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            in_dim = dims[l]
            out_dim = dims[l + 1]
            lin = nn.Linear(in_dim, out_dim)

            if init_type == "geo_init":
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, feats):
        x = feats
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.activation(x)
        return x
