#!/usr/bin/env python3

import logging
import math
import numpy as np
import plyfile
import skimage.measure
import time
import torch
import torchcumesh2sdf
import trimesh
import mcubes
import diso
import mesh2sdf
import skimage.measure

# N: resolution of grid; 256 is typically sufficient 
# max batch: as large as GPU memory will allow
# shape_feature is either point cloud, mesh_idx (neuralpull), or generated latent code (deepsdf)
def create_mesh(
    model, shape_feature, filename, N=256, max_batch=1000000, level_set=0.0, occupancy=False
):
    
    start_time = time.time()
    ply_filename = filename

    model.eval()

    # the voxel_origin is the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)
    cube = create_cube(N).cuda()
    cube_points = cube.shape[0]

    head = 0
    while head < cube_points:
        
        query = cube[head : min(head + max_batch, cube_points), 0:3].unsqueeze(0)
        
        # inference defined in forward function per pytorch lightning convention
        pred_sdf = model(shape_feature.cuda(), query)

        cube[head : min(head + max_batch, cube_points), 3] = pred_sdf
            
        head += max_batch
    
    # for occupancy instead of SDF, subtract 0.5 so the surface boundary becomes 0
    sdf_values = cube[:, 3] - 0.5 if occupancy else cube[:, 3] 
    sdf_values = sdf_values.reshape(N, N, N).detach().cpu()

    #print("inference time: {}".format(time.time() - start_time))

    convert_sdf_samples_to_ply(
        sdf_values.data,
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        level_set
    )


# create cube from (-1,-1,-1) to (1,1,1) and uniformly sample points for marching cube
def create_cube(N):

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # the voxel_origin is the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)
    
    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long().float() / N) % N
    samples[:, 0] = ((overall_index.long().float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    samples.requires_grad = False

    return samples



def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    level_set=0.0
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    # use marching_cubes_lewiner or marching_cubes depending on pytorch version 
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, level=level_set, spacing=[voxel_size] * 3
        )
    except Exception as e:
        print("skipping {}; error: {}".format(ply_filename_out, e))
        return

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)




def load_and_precess(mesh_dir,band):
    mesh = trimesh.load(mesh_dir, process=False, force='mesh', skip_materials=True)

    tris = np.array(mesh.triangles, dtype=np.float32, subok=False)
    # tris[..., [1, 2]] = tris[..., [2, 1]]
    tris = tris - tris.min(0).min(0)
    tris = (tris / tris.max() + band) / (1 + band * 2)
    return torch.tensor(tris, dtype=torch.float32, device='cuda:0')


def get_watertight_mesh(mesh_dir,res,out_dir,batch_size=10_000):
    band = 8/res
    tris = load_and_precess(mesh_dir,band)
    sdf = torchcumesh2sdf.get_sdf(tris, res, band, batch_size)-2/res
    v, f = diso.DiffMC().cuda().forward(sdf) # todo: how to smooth?
    # v, f, _, _ = skimage.measure.marching_cubes(sdf.cpu().numpy(), 2/res)
    # v,f = mcubes.marching_cubes(sdf.cpu().numpy(), 2/res)
    # to [0,1]
    v_01 = v/res
    # to (-1,1)
    new_v = (v_01 *2 - 1.0)*0.9
    mcubes.export_obj(new_v.cpu().numpy(), f.cpu().numpy(), out_dir)


def get_watertight_mesh_cpu(mesh_dir,res,out_dir,mesh_scale=0.8):

    mesh = trimesh.load(mesh_dir, process=False, force='mesh', skip_materials=True)
    vertices = mesh.vertices
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale
    sdf, mesh = mesh2sdf.compute(
        vertices, mesh.faces, res, fix=True, level=2/res, return_mesh=True)
    
    mesh.vertices = mesh.vertices / scale + center
    # mesh.export(out_dir)


def post_processing(mesh_dir,size,out_dir):

    mesh = mesh = trimesh.load(mesh_dir, process=False, force='mesh', skip_materials=True)

    # keep the max component of the extracted mesh
    components = mesh.split(only_watertight=False)
    bbox = []
    for c in components:
        bbmin = c.vertices.min(0)
        bbmax = c.vertices.max(0)
        bbox.append((bbmax - bbmin).max())
    max_component = np.argmax(bbox)
    mesh = components[max_component]
    mesh.vertices = mesh.vertices * (2.0 / size) - 1.0  # normalize it to [-1, 1]
    mesh.export(out_dir)

    # return mesh

    # re-compute sdf
    # sdf = mesh2sdf.core.compute(mesh.vertices, mesh.faces, size)
    # return (sdf, mesh) if return_mesh else sdf