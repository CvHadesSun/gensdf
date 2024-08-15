# TODO: use pytorch3d to render mesh into color, normal, depth, mask maps

import torch
import trimesh
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    TexturesVertex,
    BlendParams,
    SoftSilhouetteShader
)
from pytorch3d.renderer.mesh import Textures
from pytorch3d.renderer.lighting import DirectionalLights
from pytorch3d.io import load_objs_as_meshes
import matplotlib.pyplot as plt
import cv2
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer.blending import softmax_rgb_blend


# Load and normalize the mesh using trimesh
mesh = trimesh.load('data/rings_norm/mesh/5.obj',device="cuda")
vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
faces = torch.tensor(mesh.faces, dtype=torch.int64)

# Create a PyTorch3D Meshes object
textures = TexturesVertex(verts_features=torch.ones_like(vertices)[None])  # White color for all vertices
mesh = Meshes(verts=[vertices], faces=[faces], textures=textures).cuda()

# Define the camera and lighting
R, T = look_at_view_transform(1.5, 0, 0)  # Distance, elevation, azimuth
cameras = OpenGLPerspectiveCameras(device="cuda", R=R, T=T)
lights = DirectionalLights(device="cuda", direction=[[0, 0, -1]])

# Define the rasterization and shading settings
raster_settings = RasterizationSettings(
    image_size=256,
    blur_radius=0.0,
    faces_per_pixel=1,
)

# Create a phong renderer with lighting
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=HardPhongShader(
        device="cuda",
        cameras=cameras,
        lights=lights
    )
)


# Render the images
color_image = renderer(mesh)
plt.figure(figsize=(10, 10))
plt.imshow(color_image[0, ..., :3].cpu().numpy())
# plt.show()
plt.imsave('./outputs/color.png',color_image[0, ..., :3].cpu().numpy())


rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
fragments = rasterizer(mesh)
depth_image = fragments.zbuf
plt.figure(figsize=(10, 10))
# plt.imshow(depth_image[0, ..., 0].cpu().numpy(), cmap="gray")
# plt.title('Depth Image')
# plt.show()

depth = depth_image[0, ..., 0].cpu().numpy() * 1000
depth_img = depth.astype(np.uint16)
# plt.imsave('./outputs/depth.png',depth_img)
cv2.imwrite('./outputs/depth.png',depth_img)


# normal 
faces = mesh.faces_packed()  # (F, 3)
vertex_normals = mesh.verts_normals_packed()  # (V, 3)
faces_normals = vertex_normals[faces]
ones = torch.ones_like(fragments.bary_coords)
pixel_normals = interpolate_face_attributes(
            fragments.pix_to_face, ones, faces_normals
        )
bkgd=(0.,0.,0.)
blend_params = BlendParams(1e-4, 1e-4, bkgd)
normal_map = softmax_rgb_blend(
            pixel_normals, fragments, blend_params
)
print(normal_map.min(),normal_map.max())
# normals = mesh.verts_normals_packed()
# faces_normals = torch.index_select(normals, 0, mesh.faces_packed().reshape(-1)).reshape_as(mesh.faces_packed())
# faces_normals = faces_normals.mean(dim=1)  # Get the average normal per face
# normal_image = torch.sum(fragments.pix_to_face >= 0, dim=-1, dtype=torch.float32)[:, :, None] * faces_normals[fragments.pix_to_face]
# plt.figure(figsize=(10, 10))
# plt.imshow(normal_image[0, ..., :3].cpu().numpy())
# # plt.title('Normal Image')
# # plt.show()
# plt.imsave('./outputs/normal.png',normal_image[0, ..., :3].cpu().numpy())

silhouette_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-4, gamma=1e-4))
)

mask_image = silhouette_renderer(mesh)
plt.figure(figsize=(10, 10))
plt.imshow(mask_image[0, ..., 3].cpu().numpy(), cmap="gray")
cv2.imwrite('./outputs/mask.png',mask_image[0, ..., 3].cpu().numpy())
# plt.title('Mask Image')
# plt.show()


# depth_image = depth_renderer(mesh)
# plt.figure(figsize=(10, 10))
# plt.imshow(depth_image[0, ..., 0].cpu().numpy(), cmap="gray")
# # plt.show()
# plt.imsave('./outputs/color.png',depth_image[0, ..., 0].cpu().numpy())


# Normal Image
# normal_renderer = MeshRenderer(
#     rasterizer=MeshRasterizer(
#         cameras=cameras,
#         raster_settings=raster_settings
#     ),
#     shader=NormalShader(device="cuda")
# )

# normal_image = normal_renderer(mesh)
# plt.figure(figsize=(10, 10))
# plt.imshow(normal_image[0, ..., :3].cpu().numpy())
# # plt.show()
# plt.imsave('./outputs/color.png',normal_image[0, ..., :3].cpu().numpy())

# # Silhouette/Mask Image
# silhouette_renderer = MeshRenderer(
#     rasterizer=MeshRasterizer(
#         cameras=cameras,
#         raster_settings=raster_settings
#     ),
#     shader=SoftSilhouetteShader(device="cuda", blend_params=BlendParams(sigma=1e-4, gamma=1e-4))
# )

# mask_image = silhouette_renderer(mesh)
# plt.figure(figsize=(10, 10))
# plt.imshow(mask_image[0, ..., 3].cpu().numpy(), cmap="gray")
# plt.show()

# Random viewpoints sampling on a sphere
# import random
# elevations = [random.uniform(-90, 90) for _ in range(num_views)]
# azimuths = [random.uniform(-180, 180) for _ in range(num_views)]
# Rs, Ts = [], []
# for elev, azim in zip(elevations, azimuths):
#     R, T = look_at_view_transform(2.7, elev, azim)
#     Rs.append(R)
#     Ts.append(T)

# for i, (R, T) in enumerate(zip(Rs, Ts)):
#     cameras = OpenGLPerspectiveCameras(device="cuda", R=R, T=T)
#     color_image = renderer(mesh)
#     plt.figure(figsize=(10, 10))
#     plt.imshow(color_image[0, ..., :3].cpu().numpy())
#     plt.show()


