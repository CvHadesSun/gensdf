"""borrow from nfd src."""

import argparse
import trimesh
import numpy as np
from .inside_mesh import inside_mesh
from pykdtree.kdtree import KDTree
from tqdm import tqdm
import torch
import mcubes
import mesh2sdf


def normalize_mesh(mesh):
    # print("Scaling Parameters: ", mesh.bounding_box.extents)
    mesh.vertices -= mesh.bounding_box.centroid
    mesh.vertices /= np.max(mesh.bounding_box.extents / 2)


def compute_volume_points(intersector, count, max_batch_size = 1000000):
    coordinates = np.random.rand(count, 3) * 2 - 1

    occupancies = np.zeros((count, 1), dtype=int)
    head = 0
    with tqdm(total = coordinates.shape[0]) as pbar:
        while head < coordinates.shape[0]:
            occupancies[head:head+max_batch_size] = intersector.query(coordinates[head:head+max_batch_size]).astype(int).reshape(-1, 1)
            head += max_batch_size
            pbar.update(min(max_batch_size, coordinates.shape[0] - head))
        return np.concatenate([coordinates, occupancies], -1)


def compute_near_surface_points(mesh, intersector, count, epsilon, max_batch_size = 1000000):
    coordinates = trimesh.sample.sample_surface(mesh, count)[0] + np.random.randn(*(count, 3)) * epsilon

    occupancies = np.zeros((count, 1), dtype=int)
    head = 0
    with tqdm(total = coordinates.shape[0]) as pbar:
        while head < coordinates.shape[0]:
            occupancies[head:head+max_batch_size] = intersector.query(coordinates[head:head+max_batch_size]).astype(int).reshape(-1, 1)
            head += max_batch_size
            pbar.update(min(max_batch_size, coordinates.shape[0] - head))
    return np.concatenate([coordinates, occupancies], -1)

def compute_obj(mesh, intersector, max_batch_size = 1000000, res = 256):
    xx = torch.linspace(-1, 1, res)
    yy = torch.linspace(-1, 1, res)
    zz = torch.linspace(-1, 1, res)

    (x_coords, y_coords, z_coords) = torch.meshgrid([xx, yy, zz])
    coords = torch.cat([x_coords.unsqueeze(-1), y_coords.unsqueeze(-1), z_coords.unsqueeze(-1)], -1)

    coordinates = coords.reshape(res*res*res, 3).numpy()

    occupancies = np.zeros((res*res*res, 1), dtype=int)
    head = 0
    with tqdm(total = coordinates.shape[0]) as pbar:
        while head < coordinates.shape[0]:
            occupancies[head:head+max_batch_size] = intersector.query(coordinates[head:head+max_batch_size]).astype(int).reshape(-1, 1)
            head += max_batch_size
            pbar.update(min(max_batch_size, coordinates.shape[0] - head))
    
    print(occupancies.min(),occupancies.max())
    occupancies = occupancies.reshape(res, res, res)
    vertices, triangles = mcubes.marching_cubes(occupancies, 0)
    mcubes.export_obj(vertices, triangles, "car_gt_"+str(res)+".obj")

def generate_gt_obj(filepath):
    print("Loading mesh...")
    mesh = trimesh.load(filepath, process=False, force='mesh', skip_materials=True)
    normalize_mesh(mesh)
    intersector = inside_mesh.MeshIntersector(mesh, 512)

    compute_obj(mesh, intersector, res = 256)


def generate_volume_dataset(filepath, output_filepath, num_surface, epsilon,normalized_mesh=False):
    # print("Loading mesh...")  
    mesh = trimesh.load(filepath, process=False, force='mesh', skip_materials=True)
    if normalized_mesh:
        normalize_mesh(mesh)

    mesh_surface_points, _ = trimesh.sample.sample_surface(mesh, 10_000_000)
    kd_tree = KDTree(mesh_surface_points)

    intersector = inside_mesh.MeshIntersector(mesh, 512)

    # print("Computing near surface points...")
    surface_points_with_occ= compute_near_surface_points(mesh, intersector, num_surface, epsilon)
    surface_points = surface_points_with_occ[:,:3]
    dist_near, _ = kd_tree.query(surface_points, k=1)

    mask = surface_points_with_occ[:,3]

    outside_mask = np.where(mask<0.5)

    dist_near[outside_mask] *=-1

    near_samples = np.concatenate([surface_points_with_occ, dist_near.reshape(-1,1)], -1)


    # print("Computing volume points...")
    volume_points_with_occ = compute_volume_points(intersector, num_surface)
    volume_points = volume_points_with_occ[:,:3]
    dist_vol, _ = kd_tree.query(volume_points, k=1)

    mask_vol = volume_points_with_occ[:,3]
    outside_mask_vol = np.where(mask_vol<0.5)

    dist_vol[outside_mask_vol] *=-1
    vol_samples = np.concatenate([volume_points_with_occ, dist_vol.reshape(-1,1)], -1)

    all_points = np.concatenate([near_samples, vol_samples], 0)
    np.save(output_filepath, all_points)

def generate_border_occupancy_dataset(filepath, output_filepath, count, wall_thickness = 0.0025):
    mesh = trimesh.load(filepath, process=False, force='mesh', skip_materials=True)
    normalize_mesh(mesh)
    
    # intersector = inside_mesh.MeshIntersector(mesh, 512)

    surface_points, _ = trimesh.sample.sample_surface(mesh, 10_000_000)
    kd_tree = KDTree(surface_points)
    
    volume_points = np.random.rand(count, 3) * 2 - 1
    dist, _ = kd_tree.query(volume_points, k=1)

    volume_occupancy = np.where(dist < wall_thickness, np.ones_like(dist), np.zeros_like(dist))
    
    near_surface_points = trimesh.sample.sample_surface(mesh, count)[0] + np.random.randn(count, 3) * EPSILON
    dist, _ = kd_tree.query(near_surface_points, k=1)

    
    near_surface_occupancy = np.where(dist < wall_thickness, np.ones_like(dist), np.zeros_like(dist))
    
    points = np.concatenate([volume_points, near_surface_points], 0)
    occ = np.concatenate([volume_occupancy, near_surface_occupancy], 0).reshape(-1, 1)
    
    dataset = np.concatenate([points, occ], -1)
    # np.save(output_filepath, dataset)

# EPSILON=0.01


# mesh_dir = "/home/wanhu/workspace/gensdf/data/rings/5.obj"

# generate_border_occupancy_dataset(mesh_dir,'./',5000)#20_000_000
# generate_gt_obj(mesh_dir)
# generate_volume_dataset(mesh_dir,'./',5000,EPSILON)

# indices = np.random.randint(low=0, high=obj_data.shape[0], size=self.points_batch_size)
# sampled_data = obj_data[indices] # self.points_batch_size, 4