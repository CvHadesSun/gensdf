import cv2
import torch
import torch.nn.functional as tfunc
from typing import Union,List
from .tools import export_pcd
import glm
import numpy as np

import open3d as o3d
import copy
import os

from pytorch3d.renderer import look_at_view_transform
import random

def calc_face_normals(
        vertices:torch.Tensor, #V,3 first vertex may be unreferenced
        faces:torch.Tensor, #F,3 long, first face may be all zero
        normalize:bool=False,
        )->torch.Tensor: #F,3
    """
         n
         |
         c0     corners ordered counterclockwise when
        / \     looking onto surface (in neg normal direction)
      c1---c2
    """
    full_vertices = vertices[faces] #F,C=3,3
    v0,v1,v2 = full_vertices.unbind(dim=1) #F,3
    face_normals = torch.cross(v1-v0,v2-v0, dim=1) #F,3
    if normalize:
        face_normals = tfunc.normalize(face_normals, eps=1e-6, dim=1) 
    return face_normals #F,3

def calc_vertex_normals(
        vertices:torch.Tensor, #V,3 first vertex may be unreferenced
        faces:torch.Tensor, #F,3 long, first face may be all zero
        face_normals:torch.Tensor=None, #F,3, not normalized
        )->torch.Tensor: #F,3

    F = faces.shape[0]

    if face_normals is None:
        face_normals = calc_face_normals(vertices,faces)
    
    vertex_normals = torch.zeros((vertices.shape[0],3,3),dtype=vertices.dtype,device=vertices.device) #V,C=3,3
    vertex_normals.scatter_add_(dim=0,index=faces[:,:,None].expand(F,3,3),src=face_normals[:,None,:].expand(F,3,3))
    vertex_normals = vertex_normals.sum(dim=1) #V,3
    return tfunc.normalize(vertex_normals, eps=1e-6, dim=1)


def sample_viewpoints(radius: float ,num_viewpoints: int)->Union[torch.Tensor,torch.Tensor]:

    # todo: sampling some viewpoints in a sphere with radius, origin is [0,0,0]

    phi = torch.acos(1 - 2 * torch.linspace(0, 1, num_viewpoints))
    theta = torch.linspace(0, 2 * torch.pi, num_viewpoints)

    # Create a camera at each position
    R, T = look_at_view_transform(dist=1.5, elev=phi * 180 / torch.pi, azim=theta * 180 / torch.pi)


    # tts = []

    # for i in range(phi.shape[0]):

    #     rot = R[i].numpy()
    #     t = T[i].numpy()
    #     Tr = np.eye(4)
    #     Tr[:3,:3] = rot
    #     Tr[:3,3] = t
    #     inv_T = np.linalg.inv(Tr)
    #     tts.append(inv_T)

    # vis_cameras(tts,(0,0,0),"./cam.ply")
    return R,T


    

def vis_cameras_from_file(cam_file, origin, out_pth=None):
    # cam file is npy file
    # origin: the origin location
    cameras_data = np.load(cam_file, allow_pickle=True)
    camera_previews = []
    for index, camera in enumerate(cameras_data):
        preview = o3d.geometry.TriangleMesh.create_cone(radius=2, height=4)
        preview = copy.deepcopy(preview).transform(camera)
        camera_previews.append(preview)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
    axis.scale(10.0, center=origin)

    if out_pth is not None:
        whole_geo = camera_previews[0]
        for i in range(1, len(camera_previews)):
            whole_geo += camera_previews[i]
        o3d.io.write_triangle_mesh(out_pth, whole_geo)
    else:
        o3d.visualization.draw_geometries([axis] + camera_previews)


def vis_cameras(cameras_data, origin, out_pth=None):
    # cam file is npy file
    # origin: the origin location
    camera_previews = []
    for index, camera in enumerate(cameras_data):
        # preview = o3d.geometry.TriangleMesh.create_cone(radius=0.1, height=0.3)
        preview = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        preview = copy.deepcopy(preview).transform(camera)
        camera_previews.append(preview)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
    axis.scale(1.0, center=origin)

    if out_pth is not None:
        whole_geo = camera_previews[0]
        for i in range(1, len(camera_previews)):
            whole_geo += camera_previews[i]
        o3d.io.write_triangle_mesh(out_pth, whole_geo)
    else:
        o3d.visualization.draw_geometries([axis] + camera_previews)

sample_viewpoints(1.5,10)



