import trimesh
import numpy as np

from typing import Optional,List
import os
from tqdm import tqdm
from sklearn.neighbors import KDTree


# TODO: normalize mesh into[-1,1]
def normalize_mesh(mesh_dir: str,out_dir: str):
    # Calculate the bounding box extents
    os.makedirs(out_dir,exist_ok=True)
    mesh = trimesh.load(mesh_dir, force='mesh')
    extents = mesh.bounds
    min_extent = extents[0]
    max_extent = extents[1]

    # Find the center and scale of the mesh
    center = (min_extent + max_extent) / 2.1
    scale = (max_extent - min_extent).max() / 2.1

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


def split(mesh, points, num_sample = 250_000):
    inside = mesh.contains(points)
    inside_points = points[inside]
    outside_points = points[np.logical_not(inside)]
    
    nin = inside_points.shape[0]
    inside_points = inside_points[:num_sample // 2] if nin > num_sample// 2 else inside_points
    outside_points = outside_points[:num_sample // 2] if nin > num_sample // 2 else outside_points[:(num_sample - nin)]
    return inside_points, outside_points

# TODO: sample mesh and points sdf
def sample_mesh(mesh_dir,num_sample=250_000):
    mesh = trimesh.load(mesh_dir, force='mesh')
    box = mesh.bounds
    # B_MIN = box[0]
    # B_MAX = box[1]
    B_MIN = np.array([-1,-1,-1])
    B_MAX = np.array([1,1,1])
    length = B_MAX - B_MIN
    sigma = length.max()*0.05
    surface_points, _ = trimesh.sample.sample_surface(mesh, num_sample * 2)
    surface_points = surface_points + np.random.normal(scale=sigma, size=surface_points.shape)


    random_points = np.random.rand(num_sample, 3) * length + B_MIN 
    sample_points = np.concatenate([surface_points,random_points],0)

    surface_inside, surface_outside = split(mesh, sample_points, num_sample * 2)
    # surface_inside, surface_outside = split(mesh, random_points, num_sample * 2)

    sample_point_count = 100_000

    surface_point_cloud = mesh.sample(sample_point_count, return_index=False)
    kd_tree = KDTree(surface_point_cloud)
    distances_in, ind_in = kd_tree.query(surface_inside)
    distances_out, ind_out = kd_tree.query(surface_outside)

    din = distances_in.astype(np.float32).reshape(-1,1)
    dout = -1*distances_out.astype(np.float32).reshape(-1,1)

    

    dist = abs(length)
    din = din/dist.max()
    dout = dout/dist.max()
    # print(dist.max(),length)
    # print(din.min(),din.max())
    # print(dout.min(),dout.max())

    in_sample = np.concatenate([surface_inside,din],1)
    out_sample = np.concatenate([surface_outside,dout],1)

    return {
        "sample_points_out": out_sample,
        "sample_points_in": in_sample
    }










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