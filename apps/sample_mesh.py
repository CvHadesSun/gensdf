import trimesh
import os


obj_dir = "/home/shiyao/datasets/rings1000/R66370G01H.obj"
mesh_0 = trimesh.load(obj_dir, force='scene')
mesh = mesh_0.dump(concatenate=True)
near_surface_points = trimesh.sample.sample_surface(mesh, 10_000)[0]

cloud = trimesh.points.PointCloud(near_surface_points)

cloud.export("./2.ply")
