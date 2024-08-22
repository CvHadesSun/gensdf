import os
import sys
import trimesh
sys.path.append(os.getcwd())

from utils.mesh import get_watertight_mesh,post_processing



mesh_dir = "/home/wanhu/workspace/gensdf/outputs/triplane_occ_new_sample_512_0/rings_103/000/val_mesh/9999.obj"

out_dir = "/home/wanhu/workspace/gensdf/outputs/triplane_occ_new_sample_512_0/rings_103/000/val_mesh/9999_p.obj"

# post_processing(mesh_dir,256,out_dir)
get_watertight_mesh(out_dir,384,out_dir)