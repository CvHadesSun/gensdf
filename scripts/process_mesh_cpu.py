import trimesh
import mesh2sdf
import numpy as np


mesh_dir = '/home/wanhu/workspace/gensdf/data/test_data/de20071db0e8412cacc941ec5bf071ba.glb'
mesh = trimesh.load(mesh_dir, force='mesh')
mesh_scale = 0.95
size = 384

# rescale mesh to [-1, 1] for mesh2sdf, note the factor **mesh_scale**
vertices = mesh.vertices
bbmin, bbmax = vertices.min(0), vertices.max(0)
center = (bbmin + bbmax) * 0.5
scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
vertices = (vertices - center) * scale

# run mesh2sdf
sdf,mesh_new= mesh2sdf.compute(vertices, mesh.faces, size, fix=True,level=2/384, return_mesh=True)
mesh_new.vertices = mesh_new.vertices #* shape_scale
mesh_new.export('/home/wanhu/workspace/gensdf/data/test_data/de20071db0e8412cacc941ec5bf071ba_sdf.obj')

# np.save(filename_npy, sdf)