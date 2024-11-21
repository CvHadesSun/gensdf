import torch
import os
import sys
import glob
from tqdm import tqdm
import numpy as np
sys.path.append(os.getcwd())
import random
import matplotlib.pyplot as plt
from utils.sdf import create_cube
import diso
import mcubes


from model.triplane_net.multi_network import MultiTriplane,MLPDecoder


def forward(tri_model,mlp_model,batch_pts,obj_idx,):
    pred_feature = tri_model(obj_idx,batch_pts[None,...])
    pred_sdf = mlp_model(pred_feature)

    return pred_sdf


def main(dir,out_dir):

    os.makedirs(out_dir,exist_ok=True)

    triplane_feat = MultiTriplane(num_objs=1,feature_dim=32,resolution=256)

    in_channls =  triplane_feat.feature_dim * 3 + triplane_feat.embedding_channels

    mlp_decoder = MLPDecoder(in_channls,latend_dim=128,last_op=None)

    sub_objects = os.listdir(dir)

    decoder_ckpts = torch.load('/home/wanhu/workspace/gensdf/outputs/decoder_1.0.pt')
    ori_weights = decoder_ckpts['decoder_state_dict']
    mlp_decoder.load_state_dict(decoder_ckpts['decoder_state_dict'])
    mlp_decoder = mlp_decoder.cuda()
    triplane_feat = triplane_feat.cuda()

    res=128
    obj_idx = torch.tensor([0]).cuda().reshape(1,-1)
    grids = create_cube(res)
    grids = grids.cuda()
    num_pts = grids.shape[0]
    sdf = torch.zeros(num_pts)
    num_samples = 100_000

    num_batch = num_pts // num_samples

    # selected_subjects  = random.sample(sub_objects,1)
    selected_subjects = sub_objects
    triplane_feat.eval()
    mlp_decoder.eval()

    for item in tqdm(selected_subjects):
        # if int(item) !=661:continue
        if os.path.exists(f"{dir}/{item}"):
            checkpoints = np.load(f"{dir}/{item}")
            # checkpoints = sorted(glob.glob(f"{dir}/{item}/checkpoints/*.pt"))
            # print(checkpoints[-1])
            checkpoints = checkpoints.reshape(3,32,256,256)
            for i in range(checkpoints.shape[0]):

                tri_data = torch.from_numpy(checkpoints[i]).cuda().view(1,32,256,256)
                triplane_feat.embeddings[i].data = tri_data

            # ckpts = torch.load(checkpoints[-1])
            # triplane_feat.load_state_dict(ckpts['triplane_state_dict'])
            # mlp_decoder.load_state_dict(ckpts['decoder_state_dict'])

            # triplane_feat.cuda()
            for i in range(num_batch):
                batch_pts = grids[i*num_samples:i*num_samples+num_samples]
                pred_feature = triplane_feat(obj_idx,batch_pts[None,...])
                pred_sdf = mlp_decoder(pred_feature)

                sdf[i*num_samples:i*num_samples+num_samples] = pred_sdf[0,:,0].detach()

            if num_pts % num_samples:
                sdf[num_batch*num_samples:] = (forward(triplane_feat,mlp_decoder,grids[num_batch*num_samples:],obj_idx)[0,:,0]).detach()

            sdf = sdf.reshape(res,res,res)*-1
            # vertices, triangles = diso.DiffMC().cuda().forward(sdf.cuda())
            vertices, triangles = mcubes.marching_cubes(sdf.cpu().detach().numpy(), 0)

            vertices /=res
            new_vertices = (vertices*2-1)*0.9
            # mcubes.export_obj(new_vertices.cpu().detach().numpy(),triangles.cpu().numpy(),f"{out_dir}/{item}.obj")
            mcubes.export_obj(new_vertices,triangles,f"{out_dir}/{item}.obj")
            sdf = sdf.reshape(-1)
            # del vertices
            # del triangles
            # break
            # triplanes=[]
            # for tri in triplane_feat.embeddings:
            #     triplanes.append(tri.cpu().detach())

            # triplane = torch.cat(triplanes,dim=0)
            # np.save(f"{out_dir}/{item}_triplane_feat.npy",triplane.numpy())

main('/home/wanhu/workspace/gensdf/outputs/samples_3/triplanes',"/home/wanhu/workspace/gensdf/outputs/samples_3/objects")

