import torch
import os
import sys
import glob
from tqdm import tqdm
import numpy as np
sys.path.append(os.getcwd())

from model.triplane_net.multi_network import MultiTriplane,MLPDecoder


def main(dir,out_dir):

    os.makedirs(out_dir,exist_ok=True)

    triplane_feat = MultiTriplane(num_objs=1,feature_dim=32,resolution=256)

    in_channls =  triplane_feat.feature_dim * 3 + triplane_feat.embedding_channels

    mlp_decoder = MLPDecoder(in_channls,latend_dim=128,last_op=None)

    sub_objects = os.listdir(dir)

    for item in tqdm(sub_objects):
        if os.path.exists(f"{dir}/{item}/checkpoints"):
            checkpoints = sorted(glob.glob(f"{dir}/{item}/checkpoints/*.pt"))

            ckpts = torch.load(checkpoints[-1])
            triplane_feat.load_state_dict(ckpts['triplane_state_dict'])

            triplanes=[]
            for tri in triplane_feat.embeddings:
                triplanes.append(tri.cpu().detach())

            triplane = torch.cat(triplanes,dim=0)
            np.save(f"{out_dir}/{item}_triplane_feat.npy",triplane.numpy())





main('/home/wanhu/workspace/gensdf/outputs/triplane_sdf_1000_new',"outputs/triplane_features_1000")

