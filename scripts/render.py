from general_utils import calc_vertex_normals

import torch
import torch.nn.functional as tfunc
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.shader import ShaderBase
from pytorch3d.renderer import (
    RasterizationSettings,
    PerspectiveCameras,
    MeshRendererWithFragments,
    TexturesVertex,
    MeshRasterizer,
    BlendParams,
    FoVOrthographicCameras,
    look_at_view_transform,
    hard_rgb_blend,
    OpenGLPerspectiveCameras,
    MeshRenderer,
    HardPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    
)
from pytorch3d.renderer.mesh.shader import SoftDepthShader,HardDepthShader

from pytorch3d.renderer.lighting import DirectionalLights,PointLights
import trimesh
import cv2
from general_utils import sample_viewpoints
import numpy as np
import os
from PIL import Image
from tools import sample_mesh

class VertexColorShader(ShaderBase):
    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        return hard_rgb_blend(texels, fragments, blend_params)

def render_mesh_vertex_color(mesh, cameras, H, W, blur_radius=0.0, faces_per_pixel=1, bkgd=(0., 0., 0.), dtype=torch.float32, device="cuda"):
    if len(mesh) != len(cameras):
        if len(cameras) % len(mesh) == 0:
            mesh = mesh.extend(len(cameras))
        else:
            raise NotImplementedError()
    
    # render requires everything in float16 or float32
    input_dtype = dtype
    blend_params = BlendParams(1e-4, 1e-4, bkgd)

    # Define the settings for rasterization and shading
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=blur_radius,
        faces_per_pixel=faces_per_pixel,
        clip_barycentric_coords=True,
        bin_size=None,
        max_faces_per_bin=None,
    )

    # Create a renderer by composing a rasterizer and a shader
    # We simply render vertex colors through the custom VertexColorShader (no lighting, materials are used)
    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=VertexColorShader(
            device=device,
            cameras=cameras,
            blend_params=blend_params
        )
    )

    # render RGB and depth, get mask
    with torch.autocast(dtype=input_dtype, device_type=torch.device(device).type):
        images, _ = renderer(mesh)
    return images   # BHW4
class Pytorch3DNormalsRenderer: # 100 times slower!!!
    def __init__(self, cameras, image_size, device):
        self.cameras = cameras.to(device)
        self._image_size = image_size
        self.device = device
    
    def render(self,
            vertices: torch.Tensor, #V,3 float
            normals: torch.Tensor, #V,3 float   in [-1, 1]
            faces: torch.Tensor, #F,3 long
            ) ->torch.Tensor: #C,H,W,4
        mesh = Meshes(verts=[vertices], faces=[faces], textures=TexturesVertex(verts_features=[(normals + 1) / 2])).to(self.device)
        return render_mesh_vertex_color(mesh, self.cameras, self._image_size[0], self._image_size[1], device=self.device)
    
class PyRender:
    # TODO: render mesh into color, depth, mask, normal
    def __init__(self,res: int,num_view: int,radius: float=2.0)->None:
        
        # viewpoints
        self.res = res
        self.R,self.T = sample_viewpoints(radius,num_viewpoints=num_view)
        self.device = "cuda:0"
        self.num_views=num_view
        self.set_light()

        self.fx = 150.0
        self.fy = 150.0

        self.cx = self.res/2
        self.cy = self.res/2

        self.cameras = PerspectiveCameras(
                image_size=[[self.res, self.res]],
                R=self.R,
                T=self.T,
                focal_length=torch.tensor([[self.fx, self.fy]], dtype=torch.float32),
                principal_point=torch.tensor([[self.cx, self.cy]], dtype=torch.float32),
                in_ndc=False,
            ).cuda()


    
    def color_render(self,mesh, out_dir: str):

        color_dir = f"{out_dir}/colors"
        depth_dir = f"{out_dir}/depths"
        mask_dir = f"{out_dir}/masks"

        os.makedirs(color_dir,exist_ok=True)
        os.makedirs(depth_dir,exist_ok=True)
        os.makedirs(mask_dir,exist_ok=True)

        raster_settings = RasterizationSettings(
            image_size=self.res,
            blur_radius=0.0,
            faces_per_pixel=1,)

        cameras = self.cameras # OpenGLPerspectiveCameras(device=self.device,R=self.R.cuda(), T=self.T.cuda())

        if len(mesh) != len(self.cameras):
            if len(cameras) % len(mesh) == 0:
                mesh = mesh.extend(len(cameras))
            else:
                raise NotImplementedError()
            
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras ,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=self.cameras ,
                lights=self.lights
            )
        )
        color_image = renderer(mesh)
        # 
        # raster_settings.perspective_correct = True
        rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings)
        fragments = rasterizer(mesh)
        depth_image = fragments.zbuf
        mask_image= fragments.pix_to_face > 0
        depth_image[~mask_image]=0

        for i in range(self.num_views):
            color_img = color_image.cpu().numpy()*255
            color_img = Image.fromarray((color_img[i,:,:,:3]).astype(np.uint8))
            color_img.save(f"{color_dir}/view_{i:03d}.png")

            depth_img = (depth_image[i,:,:,0].cpu().numpy()*1000).astype(np.uint16)
            depth_img = Image.fromarray(depth_img)
            depth_img.save(f"{depth_dir}/view_{i:03d}.png")

            mask_img = mask_image[i,:,:,0].cpu().numpy()*255
            mask_img = Image.fromarray(mask_img.astype(np.uint8))
            mask_img.save(f"{mask_dir}/view_{i:03d}.png")


    def normal_render(self,vertices,faces,out_dir):

        normal_dir = f"{out_dir}/normals"
        os.makedirs(normal_dir,exist_ok=True)
        normal_render = Pytorch3DNormalsRenderer(self.cameras,[self.res,self.res],self.device)
        normals = calc_vertex_normals(vertices,faces)
        normal_image = normal_render.render(vertices,normals,faces)

        for i in range(self.num_views):
            normal_img = normal_image[i,:,:,:3].cpu().numpy() * 255
            image_pil = Image.fromarray(normal_img.astype(np.uint8))
            image_pil.save(f"{normal_dir}/view_{i:03d}.png") 


    def set_light(self)->None:
        light_locs = [[0.0, 0.0, -3.0] for _ in range(self.num_views)]
        self.lights = PointLights(device=self.device, location=light_locs).cuda()


    def getAABB(self,verts):

        min_ = torch.min(verts, dim=0).values
        max_ = torch.max(verts, dim=0).values

        aabb = torch.cat([min_,max_],dim=0).reshape(-1,3)

        return aabb.cpu().numpy()

    def render_mesh(self,mesh_dir: str,out_dir: str)->None:
        mesh = trimesh.load(mesh_dir,device="cuda")
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
        faces = torch.tensor(mesh.faces, dtype=torch.int64)

        textures = TexturesVertex(verts_features=torch.ones_like(vertices)[None])  # White color for all vertices
        mesh = Meshes(verts=[vertices], faces=[faces], textures=textures).cuda()

        aabb = self.getAABB(vertices)
        self.color_render(mesh,out_dir)
        self.normal_render(vertices,faces,out_dir)
        self.save_cameras(out_dir)

        np.save(f'{out_dir}/aabb.npy',aabb)


    def save_cameras(self,out_dir: str):
        k = np.eye(3)
        k[0,0] = self.fx
        k[1,1] = self.fy
        k[0,2] = self.cx
        k[1,2] = self.cy

        Rs = self.R.cpu().numpy()
        Ts = self.T.cpu().numpy()

        data ={
            'K':k,
            'width': self.res,
            'height': self.res,
            'Rs': Rs,
            'Ts': Ts
        }
        np.savez(f"{out_dir}/cameras.npz",**data)

rdr = PyRender(256,5)

root_dir = "/home/wanhu/workspace/gensdf/data/rings_norm/mesh"
mesh_dirs = os.listdir(root_dir)
out_dir = "/home/wanhu/workspace/gensdf/data/rings_test"

for i,obj in enumerate(mesh_dirs[:5]):
    if not 'obj' in obj: continue
    obj_dir = f"{root_dir}/{obj}"
    out_path=f"{out_dir}/{i:03d}"
    os.makedirs(out_path,exist_ok=True)
    # 
    with open(f"{out_path}/mesh.txt",'w') as fp:
        fp.write(obj_dir)
        fp.close()
    # 
    rdr.render_mesh(obj_dir,out_path)
    #
    result = sample_mesh(obj_dir)
    np.savez(f'{out_path}/samples.npz',**result)

    # break