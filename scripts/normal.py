import torch
from pytorch3d.renderer import (
    BlendParams,
    SoftPhongShader,
    TexturesUV,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PerspectiveCameras,
    PointLights,
    look_at_view_transform,
    softmax_rgb_blend,
    TexturesVertex,
    OpenGLPerspectiveCameras,
    hard_rgb_blend,
)
from pytorch3d.structures import Meshes
import trimesh
from pytorch3d.ops import interpolate_face_attributes
import torchvision
import cv2
from pytorch3d.renderer.lighting import DirectionalLights

def phong_normal_shading(meshes, fragments):
    # Get the faces and vertex normals from the mesh
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    
    # Gather the normals for the faces
    faces_normals = vertex_normals[faces]
    
    # Create an array of ones with the same shape as the barycentric coordinates
    ones = torch.ones_like(fragments.bary_coords)
    
    
    # Interpolate the face normals to get the normals at each pixel
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, ones, faces_normals
    )
    # pixel_normals = pixel_normals / torch.norm(pixel_normals, dim=-1, keepdim=True)
    # print(pixel_normals.max())
    return pixel_normals


mesh = trimesh.load('data/rings_norm/mesh/5.obj',device="cuda")
vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
faces = torch.tensor(mesh.faces, dtype=torch.int64)

# Create a PyTorch3D Meshes object
textures = TexturesVertex(verts_features=torch.ones_like(vertices)[None])  # White color for all vertices
mesh = Meshes(verts=[vertices], faces=[faces], textures=textures).cuda()
# Setup basic parameters for rendering
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Define the camera and light settings
R, T = look_at_view_transform(1.5, 0, 0)  # Distance, elevation, azimuth
cameras = OpenGLPerspectiveCameras(device="cuda", R=R, T=T)
lights = DirectionalLights(device="cuda", direction=[[0, 0, -1]])

# Define the rasterization settings
raster_settings = RasterizationSettings(
    image_size=512, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)

# Create a renderer (can be used for other shading types as well)
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=cameras,
        lights=lights
    )
)

# Assume `meshes` is already loaded or created with vertices and faces
# fragments: output of MeshRasterizer which contains pixel-to-face mapping
fragments = renderer.rasterizer(mesh)

# Get the pixel normals from phong_normal_shading
pixel_normals = phong_normal_shading(mesh, fragments)



# Define blending parameters
blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

# Render the normal map using softmax_rgb_blend
normal_map = softmax_rgb_blend(
    pixel_normals, fragments, blend_params
)
print(normal_map.min(),normal_map.max(),normal_map.shape)
# The resulting `normal_map` can now be used for further processing or visualization
# torchvision.utils.save_image(normal_map[0],"outputs/normal.png")

cv2.imwrite("outputs/normal.png",normal_map[0,:,:,:3].cpu().numpy()*255)