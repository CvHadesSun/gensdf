
export CUDA_VISIBLE_DEVICES=7
python -m torch.distributed.launch --nproc_per_node 1 --master_port 65534  apps/gen_samples.py --ddpm_ckpt /home/wanhu/workspace/gensdf/outputs/diffusion_out_0.0/ema_0.9999_340000.pt \
    --decoder_ckpt models/cars/car_decoder.pt --stats_dir "" \
    --save_dir outputs/samples_3 --num_samples 8 --num_steps 250 --shape_resolution 256 --resolution 256