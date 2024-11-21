import os
import sys
import trimesh
sys.path.append(os.getcwd())
import numpy as np
from scripts.render import PyRender
from utils.sample_utils import generate_volume_dataset,sample_from_sdf_field,voxel_sample_from_sdf_field
from utils.io import write_list2file
from tqdm import tqdm



# rdr = PyRender(256,5)

# root_dir = "/home/wanhu/workspace/gensdf/data/rings_norm/wmesh"
# root_dir = "/home/shiyao/datasets/rings1000"
root_dir = '/home/wanhu/workspace/gensdf/data/test_tmp'
mesh_dirs = os.listdir(root_dir)
out_dir = "/home/wanhu/workspace/gensdf/data/test_tmp/dataset"

failed_objs=[]
os.makedirs(out_dir,exist_ok=True)


for i,obj in tqdm(enumerate(mesh_dirs[:])):
    if not 'npy' in obj: continue

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
    # try:
    if 1:
        # generate_volume_dataset(obj_dir,f"{out_path}/samples.npy",20_000_000,0.001,normalized_mesh=True)
        sample_from_sdf_field(obj_dir,'/home/wanhu/workspace/gensdf/data/vehicle/dataset/000/samples.npy',f"{out_path}")
        voxel_sample_from_sdf_field(obj_dir,f"{out_path}")
    # except:
    #     failed_objs.append(obj_dir)
    # break 

# write the failed obj into txt.
# write_list2file(f"{out_dir}/precess_failed.txt",failed_objs)

