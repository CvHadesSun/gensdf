# get_watertight_mesh

import os
import sys
import trimesh
from tqdm import tqdm
sys.path.append(os.getcwd())
from utils.mesh import get_watertight_mesh,get_watertight_mesh_cpu


data_dir = "/home/wanhu/workspace/gensdf/data/rings_norm"
out_dir = f"{data_dir}/wmesh"
os.makedirs(out_dir,exist_ok=True)

objs = os.listdir(f"{data_dir}/mesh")

res=256

for obj in tqdm(objs):
    if '.obj' not in obj:
        continue
    obj_dir = f"{data_dir}/mesh/{obj}"
    out_path = f"{out_dir}/{obj}"
    get_watertight_mesh(obj_dir,res,out_path)

    # break

