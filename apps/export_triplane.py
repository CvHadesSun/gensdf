import torch
import argparse



def export_triplane(pth_dir, output_path):
    whole_net = torch.load(pth_dir)
    triplane_net =whole_net['triplane_state_dict']
    tris = triplane_net['triplane']
    torch.save(tris, output_path)


def export_decoder(pth_dir,output_path):
    whole_net = torch.load(pth_dir)
    decoder_net = whole_net['decoder_state_dict']
    torch.save(decoder_net, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export triplane from whole net')
    parser.add_argument('--pth_dir', type=str, help='The path of the whole net')
    parser.add_argument('--output_path', type=str, help='The path of the output triplane')
    parser.add_argument('--decoder', action='store_true', help='Export decoder instead of triplane')
    args = parser.parse_args()
    
    if args.decoder:
        export_decoder(args.pth_dir, args.output_path)
    else:
        export_triplane(args.pth_dir, args.output_path)