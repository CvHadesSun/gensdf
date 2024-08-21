import os
import sys
import trimesh
sys.path.append(os.getcwd())
from utils.sample_utils import normalize_mesh
from tqdm import tqdm


data_dir = "data/rings"
objs = os.listdir(data_dir)
out_dir = "data/rings_norm/mesh"
os.makedirs(out_dir,exist_ok=True)

for obj in tqdm(objs):
    if not '.obj' in obj:continue
    obj_dir = f"{data_dir}/{obj}"
    # print(obj)

    mesh = trimesh.load(obj_dir, process=False, force='mesh', skip_materials=True)
    normalize_mesh(mesh)
    mesh.export(f"{out_dir}/{obj}")
