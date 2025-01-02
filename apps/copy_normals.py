import os

uids = [line.strip() for line in open("/DATA/local4T_0/wanhu/TriplaneDiffusion/scripts/uids_random.txt")]

img_path = "/DATA/local4T_1/wanhu_data/vae_dataset/renderings"

out_dir = "output_normals"
os.makedirs(out_dir,exist_ok=True)

for i, uid in enumerate(uids):
    src_pth = f"{img_path}/{uid}/{uid}/normals/0.png"
    os.system(f"cp {src_pth} {out_dir}/{i}.png")
    # print(uid)

