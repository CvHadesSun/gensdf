import os
import shutil

root_dir = "/DATA/local4T_1/wanhu_data/vae_dataset/vehicle"
out_dir = "/DATA/local4T_1/wanhu_data/vae_dataset/vehicle_sub"

all_files = os.listdir(root_dir)

num=1000

for i in range(3):
    tmp_files = all_files[i*num:(i+1)*num]

    for file in tmp_files:
        save_path = f"{out_dir}/{i}"
        os.makedirs(save_path,exist_ok=True)
        shutil.move(f"{root_dir}/{file}",f"{save_path}/{file}")


