
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

root_dir="/DATA/local4T_1/wanhu_data/vae_dataset/vehicle_sub"
output_dir="/DATA/local4T_1/wanhu_data/vae_dataset/vehicle_sub/triplanes"
mkdir -p $output_dir

CONDA_ENV="nfd"

source /DATA/disk2T/wanhu/miniconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV

for i in {0..2} ; do
    # python apps/train.py
    input_dir="${root_dir}/${i}"
    torchrun --nproc_per_node=8 --master_port=29501    apps/train.py \
        experiment="triplane_sdf_1000_${i}" \
        dataroot=${input_dir} \
        load_decoder=True \
        dataset.opt.num_objs=1001 \
        model.opt.train_triplane=True \
        model.opt.train_sdf=False \
        hydra.run.dir="${output_dir}/triplane_sdf_1000_${i}" \

        # break

        
    # echo ${input_dir}
done