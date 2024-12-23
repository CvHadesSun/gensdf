
export CUDA_VISIBLE_DEVICES=0 #,1,2,3,4,5,6,7

root_dir="/DATA/local4T_1/wanhu_data/vae_dataset/vehicle_part0"
output_dir="/DATA/local4T_1/wanhu_data/vae_dataset/triplanes/triplanes_part0"
mkdir -p $output_dir

CONDA_ENV="nfd"
export NCCL_NVLS_ENABLE=0
source /DATA/disk2T/wanhu/miniconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV

for i in {0..9} ; do
    # python apps/train.py
    input_dir="${root_dir}/${i}"
    subdir_count=$(find "$input_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
    # torchrun --nproc_per_node=8 --master_port=29501    apps/train.py \
    python apps/val.py \
        experiment="triplane_sdf_${i}" \
        dataroot=${input_dir} \
        load_decoder=True \
        dataset.opt.num_objs=${subdir_count} \
        model.opt.train_triplane=True \
        model.opt.train_sdf=False \
        hydra.run.dir="${output_dir}/triplane_sdf_1000_${i}" 
        # break
    # echo ${input_dir}
done



# dir 0: /DATA/local4T_2/wanhu_data/vae_dataset/vehicle