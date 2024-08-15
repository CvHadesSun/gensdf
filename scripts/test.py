
import torch
import torch.nn.functional as tfunc
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.shader import ShaderBase
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRendererWithFragments,
    TexturesVertex,
    MeshRasterizer,
    BlendParams,
    FoVOrthographicCameras,
    look_at_view_transform,
    hard_rgb_blend,
    OpenGLPerspectiveCameras,
)
from pytorch3d.renderer.lighting import DirectionalLights
import trimesh
import cv2


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


class VertexColorShader(ShaderBase):
    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        return hard_rgb_blend(texels, fragments, blend_params)
    
def render_mesh_vertex_color(mesh, cameras, H, W, blur_radius=0.0, faces_per_pixel=1, bkgd=(0., 0., 0.), dtype=torch.float32, device="cuda"):
    # if len(mesh) != len(cameras):
    #     if len(cameras) % len(mesh) == 0:
    #         mesh = mesh.extend(len(cameras))
    #     else:
    #         raise NotImplementedError()
    
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
    
mesh = trimesh.load('data/rings_norm/mesh/5.obj',device="cuda")
vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
faces = torch.tensor(mesh.faces, dtype=torch.int64)

# Create a PyTorch3D Meshes object
# textures = TexturesVertex(verts_features=torch.ones_like(vertices)[None])  # White color for all vertices
# mesh = Meshes(verts=[vertices], faces=[faces], textures=textures).cuda()

# Define the camera and lighting
R, T = look_at_view_transform(1.5, 0, 0)  # Distance, elevation, azimuth
cameras = OpenGLPerspectiveCameras(device="cuda", R=R, T=T)
lights = DirectionalLights(device="cuda", direction=[[0, 0, -1]])


normal_render = Pytorch3DNormalsRenderer(cameras,[256,256],"cuda")

normals = calc_vertex_normals(vertices,faces)


normal_img = normal_render.render(vertices,normals,faces)

# print(normal_img.min(),normal_img.max())

cv2.imwrite("outputs/normal.png",normal_img[0,:,:,:3].cpu().numpy()*255)