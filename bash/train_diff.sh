
export CUDA_VISIBLE_DEVICES=5,6,7

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
TRAIN_FLAGS="--lr 1e-4 --batch_size 12 "

python -m torch.distributed.launch --nproc_per_node 3 model/diffusion/network.py \
--data_dir /home/wanhu/workspace/gensdf/outputs/triplane_features_1000 --in_out_channels 96 --save_interval 10000 $MODEL_FLAGS $TRAIN_FLAGS --explicit_normalization False

# python model/diffusion/inference.py --save_dir /home/wanhu/workspace/gensdf/outputs/tri_diffusion_eval \
# --model_path /home/wanhu/workspace/gensdf/outputs/diffusion_out_0.0/ema_0.9999_073000.pt \
# --num_samples 8 \
# --image_size 256 $MODEL_FLAGS  --explicit_normalization False
# --num_steps 250

