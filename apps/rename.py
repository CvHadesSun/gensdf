import os

root_dir = "/home/wanhu/workspace/gensdf/data/rings_1000"
# folders = sorted(,key=lambda s:int(s))
folders = os.listdir(root_dir)
folders.remove('precess_failed.txt')

st_fs = sorted(folders,key= lambda s: int(s))


for i,item in enumerate(st_fs):
    new_name = f"{i:05d}"

    cmd_str = f"mv {root_dir}/{item} {root_dir}/{new_name}"

    os.system(cmd_str)


