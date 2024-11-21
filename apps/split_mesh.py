import trimesh
import os

# 读取网格
mesh = trimesh.load('/home/shiyao/datasets/rings1000/R72907R01MH.obj')

out_dir = "/home/wanhu/workspace/gensdf/outputs/skel_data/geo"
os.makedirs(out_dir,exist_ok=True)

# 如果加载的对象是一个 Scene，我们需要提取网格
if isinstance(mesh, trimesh.Scene):
    # 将 Scene 中的所有几何体提取出来
    geometries = mesh.dump()

    # 处理每个单独的网格
    for i, geom in enumerate(geometries):
        if isinstance(geom, trimesh.Trimesh):
            # 分割连通域
            components = geom.split(only_watertight=False)
            for j, component in enumerate(components):
                print(f"Component {i}-{j} has {len(component.faces)} faces")
                # 将每个连通域保存为单独的文件
                if len(component.faces)<100:continue
                component.export(f'{out_dir}/component_{i}_{j}.obj')
else:
    # 如果是单个网格对象，直接进行连通域分割
    components = mesh.split(only_watertight=False)
    for i, component in enumerate(components):
        print(f"Component {i} has {len(component.faces)} faces")
        if len(component.faces)<100:continue
        component.export(f'{out_dir}/component_{i}.obj')