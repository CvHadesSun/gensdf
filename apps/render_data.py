import os
import sys
import trimesh
sys.path.append(os.getcwd())
import numpy as np
from scripts.render import PyRender
from utils.sample_utils import generate_volume_dataset



rdr = PyRender(256,5)

root_dir = "/home/wanhu/workspace/gensdf/data/rings_norm/wmesh"
mesh_dirs = os.listdir(root_dir)
out_dir = "/home/wanhu/workspace/gensdf/data/rings_test"

for i,obj in enumerate(mesh_dirs[:]):
    if not 'obj' in obj: continue
    obj_dir = f"{root_dir}/{obj}"
    out_path=f"{out_dir}/{i:03d}"
    os.makedirs(out_path,exist_ok=True)
    # 
    with open(f"{out_path}/mesh.txt",'w') as fp:
        fp.write(obj_dir)
        fp.close()
    # 
    # rdr.render_mesh(obj_dir,out_path)
    #
    generate_volume_dataset(obj_dir,f"{out_path}/samples.npy",2_000_000,0.01)

    # break 