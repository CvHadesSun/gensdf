import os
import sys
import trimesh
sys.path.append(os.getcwd())
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random


# net = CartesianPlaneNonSirenEmbeddingNetwork()

# ckpts = torch.load("/home/wanhu/workspace/gensdf/outputs/triplane_sdf_tri_cat_xyz/rings_103/000/checkpoints/model_epoch_9999_loss_0.0003.pt")
# net.load_state_dict(ckpts['model_state_dict'])



# fig = plt.figure(figsize=(256, 256))
# for idx, triplane in enumerate(net.embeddings):

#     triplane = triplane[0].detach().permute(1,2,0).cpu().numpy()
#     norm_tri = (triplane+1)/2
#     norm_img = (norm_tri.mean(-1)*255).astype(np.uint8)

#     # cv2.imwrite(f'./outputs/{idx}.png',norm_img)

#     a = fig.add_subplot(1, 3, idx+1)
#     mgplot = plt.imshow(norm_img)
#     a.axis("off")


out_dir = f"outputs/triplane_featres_results"
os.makedirs(out_dir,exist_ok=True)

dir = "outputs/triplane_features_1000"

tris = os.listdir(dir)

selected_tris = random.sample(tris,10)
fig = plt.figure(figsize=(256, 256))


for item in selected_tris:
    tri = np.load(f"{dir}/{item}")
    for idx in range(tri.shape[0]):
        single_tri = tri[0].transpose(1,2,0)
        norm_tri = (single_tri+1)/2
        norm_img = (norm_tri.mean(-1)*255).astype(np.uint8)
    #     print(norm_img.shape)
        a = fig.add_subplot(1, 3, idx+1)
        mgplot = plt.imshow(norm_img)
        a.axis("off")

    plt.savefig(f'{out_dir}/{item}.jpg',bbox_inches='tight')

    # break





# plt.savefig(str('./outputs/feature_maps_cat_all.jpg'), bbox_inches='tight')
    


# position_encoding, num_channel = get_embedder(4, 0)
# print(num_channel)

# triplane.detach().numpy().reshape(3,256,256)
# print(f'Saving to {args.outdir}/{str(idx + args.subset_start_idx).zfill(4)}.npy...')
# np.save(f'{args.outdir}/{str(idx + args.subset_start_idx).zfill(4)}', triplane.detach().numpy().reshape(3, args.channels, args.resolution, args.resolution))
