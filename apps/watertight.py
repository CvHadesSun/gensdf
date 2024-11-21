import time
import diso
import numpy
import torch
import mcubes
import trimesh
import torchcumesh2sdf
import matplotlib.pyplot as plotlib
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# R = 512
# band = 2 / R
def load_and_preprocess(p,band):
    # mesh = trimesh.load(p)
    mesh_0 = trimesh.load(p, force='scene')
    mesh = mesh_0.dump(concatenate=True)
    tris = numpy.array(mesh.triangles, dtype=numpy.float32, subok=False)
    # tris[..., [1, 2]] = tris[..., [2, 1]]
    tris = tris - tris.min(0).min(0)
    tris = (tris / tris.max() + band) / (1 + band * 2)
    return torch.tensor(tris, dtype=torch.float32, device='cuda:0')


def get_watertight_mesh(mesh_dir,res,out_dir,batch_size=10_000):
    band = 8/res
    tris = load_and_preprocess(mesh_dir,band)
    sdf = torchcumesh2sdf.get_sdf(tris, res, band, batch_size)-2/res
    v, f = diso.DiffMC().cuda().forward(sdf) # todo: how to smooth?
    # v, f, _, _ = skimage.measure.marching_cubes(sdf.cpu().numpy(), 2/res)
    # v,f = mcubes.marching_cubes(sdf.cpu().numpy(), 2/res)
    # to [0,1]
    v_01 = v/res
    # to (-1,1)
    new_v = (v_01 *2 - 1.0)*0.9
    mcubes.export_obj(new_v.cpu().numpy(), f.cpu().numpy(), out_dir)

if __name__ == "__main__":

    root_dir = "/home/wanhu/workspace/gensdf/data/vehicle"
    mesh_dirs = os.listdir(root_dir)

    out_dir = "/home/wanhu/workspace/gensdf/data/vehicle"
    os.makedirs(out_dir,exist_ok=True)

    all_jobs=[]

    for i,item in tqdm(enumerate(mesh_dirs)):

        if not 'off' in item: continue

        obj_dir = f"{root_dir}/{item}"
        out_path = f"{out_dir}/{i:04d}.obj"

        if os.path.exists(out_path):continue



        all_jobs.append([obj_dir,out_path])

        # get_watertight_mesh(obj_dir,512,out_path)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(get_watertight_mesh, item[0],512,item[1]) for item in all_jobs]
        
        # 使用tqdm显示多线程的进度条
        for future in tqdm(as_completed(futures),total=len(all_jobs)):
            # task_id = 0#futures[future]
            pass
            # try:
            #     item, success = future.result()
            #     if success:
            #         true_list.append(item)
            #     else:
            #         false_list.append(item)
            # except Exception as exc:
            #     print(f"Task {task_id} generated an exception: {exc}")
            #     with open(f'shape_selected_{sub_name}_{int(time.time())}.txt', 'w') as out:
            #         for p in true_list:
            # 


