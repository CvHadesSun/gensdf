import os


root_dir = '/DATA/local4T_2/wanhu_data/vae_dataset/vehicle'

all_files = os.listdir(root_dir)

for file in all_files:

    ff = f'{root_dir}/{file}'

    sub_fs = os.listdir(ff)

    if len(sub_fs) == 0:
        print(ff)
        os.rmdir(ff)
        continue