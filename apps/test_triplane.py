import numpy as np 
import torch
import os
import sys
sys.path.append(os.getcwd())

from model.triplane_net.kplanes import Triplane,Network
from model.triplane_net.multiTriplane import vis_model
def build_net(num_obj,reso=32):
    triplane = Triplane(n = num_obj,reso=reso,init_type='zero_init')
    net = Network()

    return triplane,net

decoder_ckpt= "/DATA/local4T_0/wanhu/gensdf/decoder/decoder_100_obj.pt"
decoder_state = torch.load(decoder_ckpt)

trip_net,net = build_net(100)

net.load_state_dict(decoder_state)

# trip_net.eval()
# net.eval()

# print(net)

# exit()
if 0:
    trip_dir = "/DATA/local4T_1/wanhu_data/vae_dataset/6k_triplanes/700"

    # trip_files = os.listdir(trip_dir)[:10]
    # with open("/DATA/local4T_0/wanhu/TriplaneDiffusion/scripts/uids_random.txt")
    uids = [line.strip() for line in open("/DATA/local4T_0/wanhu/TriplaneDiffusion/scripts/uids_random.txt")]
    with torch.no_grad():
        for i,trip_file in enumerate(uids):
            # trip_path = os.path.join(trip_dir,trip_file)
            trip_path = f"{trip_dir}/{trip_file}.npy"
            trip = np.load(trip_path)
            trip = torch.from_numpy(trip)
            trip_net.triplane[i] = trip

            vis_model(net.cuda(),trip_net.cuda(),1,'./output',oid=i)

            # break
            # print(i)

if 1:
    out_dir ="/DATA/local4T_0/wanhu/TriplaneDiffusion/outputs/geo"
    src_dir = "/DATA/local4T_0/wanhu/TriplaneDiffusion/outputs/latent"
    os.makedirs(out_dir,exist_ok=True)

    l = len(os.listdir(src_dir))
    with torch.no_grad():
        for i in range(l):
            trip_dir = os.path.join(src_dir,f"sample_{i}.npy")
            #
            # trip_dir = "/home/wanhu/workspace/TriplaneDiffusion/outputs/latent/sample_0.npy"
            # trip_dir = "/home/wanhu/dataset/triplanes_test/700/yellow-dune-buggy-244547eca9654b28a3ceb35393e4faf9.npy"
            trip = np.load(trip_dir)
            trip = torch.from_numpy(trip)
            print(trip.min(),trip.max())
            print(trip.shape)
            trip_net.triplane[i] = trip
            vis_model(net.cuda(),trip_net.cuda(),1,out_dir,oid=i)




