import trimesh
import numpy as np

from typing import Optional,List
import os
from tqdm import tqdm



# TODO: normalize mesh into[-1,1]
def normalize_mesh(mesh_dir: str,out_dir: str):
    # Calculate the bounding box extents
    os.makedirs(out_dir,exist_ok=True)
    mesh = trimesh.load(mesh_dir, force='mesh')
    extents = mesh.bounds
    min_extent = extents[0]
    max_extent = extents[1]

    # Find the center and scale of the mesh
    center = (min_extent + max_extent) / 2.0
    scale = (max_extent - min_extent).max() / 2.0

    # Normalize the vertices to [-1, 1]
    normalized_vertices = (mesh.vertices - center) / scale

    # Move the center of the mesh to [0, 0, 0]
    # translated_vertices = normalized_vertices - np.mean(normalized_vertices, axis=0)

    # Update the mesh with normalized and translated vertices
    mesh.vertices = normalized_vertices

    mesh_name = mesh_dir.split('/')[-1]

    # Save or use the transformed mesh
    mesh.export(f'{out_dir}/{mesh_name}')

    # Optional: View the transformed mesh

def normalize_mesh_v2(mesh_dir: str,out_dir: str):
    os.makedirs(out_dir,exist_ok=True)
    mesh = trimesh.load(mesh_dir, force='mesh')
    mesh_scale = 1.0
    vertices = mesh.vertices
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale
    mesh.vertices = mesh.vertices / scale + center
    mesh_name = mesh_dir.split('/')[-1]
    mesh.export(f'{out_dir}/{mesh_name}')

# TODO: sample mesh and points sdf
def sample_mesh(mesh,scale):
    pass



# data_dir = "data/rings"

# objs = os.listdir(data_dir)

# for obj in tqdm(objs):
#     if not '.obj' in obj:continue
#     obj_dir = f"{data_dir}/{obj}"
#     # print(obj)
#     normalize_mesh(obj_dir,"data/rings_norm/mesh")


def export_pcd(points: np.array,output_dir: str,colors: np.array=None)->None:
    pcd = trimesh.points.PointCloud(points)
    pcd.export(f"{output_dir}")