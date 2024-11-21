import os
import sys
import trimesh
sys.path.append(os.getcwd())
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

ckpt_dir = f"/home/wanhu/workspace/gensdf/outputs/triplane_sdf_500_debug/checkpoints/model_epoch_1500_loss_0.0011.pt"

ckpts_state = torch.load(ckpt_dir)

decoder_state = ckpts_state["decoder_state_dict"]

torch.save({
    "decoder_state_dict": decoder_state
},f"/home/wanhu/workspace/gensdf/outputs/decoder_1.0.pt")



