import numpy as np
import trimesh
import random
import torch

def vis_sdf(pts,sdf,out_dir):

    color = np.array([0,255,0]).reshape(1,3)

    sdf = sdf.reshape(-1,1)

    pts = pts.reshape(-1,3)

    colors = sdf @ color

    cloud = trimesh.points.PointCloud(pts, colors=colors)

    cloud.export(out_dir)


data_dir = "/home/wanhu/workspace/gensdf/data/rings_test/000/samples.npy"
data = np.load(data_dir)

pts = data[:,:3]

occ = data[:,3]

mask = np.where(occ<0.5)

pts_in = pts[mask]
occ_in = occ[mask]
# sout = torch.from_numpy(data["sample_points_out"])
# sin = torch.from_numpy(data["sample_points_in"])

# index_out = (torch.rand(250_000//2)*sout.shape[0]).long()
# index_in = (torch.rand(250_000//2)*sin.shape[0]).long()

# out_sample = sout[index_out]
# in_sample = sin[index_in]

# # print(out_sample.min(),in_sample.min())
# # print(out_sample.max(),in_sample.max())


# pts = torch.cat([out_sample[:,:3],in_sample[:,:3]],dim=0)

# sdf = torch.cat([-1*out_sample[:,3],in_sample[:,3]],dim=0)


# print(pts.min(),pts.max())
# print(sdf.min(),sdf.max())


vis_sdf(pts_in,occ_in,'out_occ.ply')



