import os
import shutil

# root_dir = "/DATA/local4T_1/wanhu_data/vae_dataset/vehicle"
# out_dir = "/DATA/local4T_1/wanhu_data/vae_dataset/vehicle_part0"
root_dir = "/DATA/local4T_2/wanhu_data/vae_dataset/vehicle"
out_dir = "/DATA/local4T_2/wanhu_data/vae_dataset/vehicle_part0"

all_files = os.listdir(root_dir)

l = len(all_files)

num=1000

num_f = l//num

for i in range(num_f):
    tmp_files = all_files[i*num:(i+1)*num]

    for file in tmp_files:
        save_path = f"{out_dir}/{i}"
        os.makedirs(save_path,exist_ok=True)
        shutil.move(f"{root_dir}/{file}",f"{save_path}/{file}")


j = i+1
if l >num_f*num:
    last_files = all_files[num_f*num:]

    for file in last_files:
        save_path = f"{out_dir}/{j}"
        os.makedirs(save_path,exist_ok=True)
        shutil.move(f"{root_dir}/{file}",f"{save_path}/{file}")
