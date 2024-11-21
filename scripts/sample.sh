#!/bin/bash

# 设置源文件夹路径和Python脚本路径
source_folder="/home/wanhu/dataset/subfolder_8"

 # 替换为您的 Conda 环境名称


# 设置线程数
num_threads=4


# # 导出文件列表
file_list=($(find "$source_folder" -type f))

# 定义一个函数，激活环境并处理文件
process_file() {
    file=$1
    conda_env="sdf" 
    output_folder="/home/wanhu/dataset/subfolder_8_sample"

    # 提取文件名去掉后缀部分
    filename=$(basename "$file")
    file_basename="${filename%.*}"
    
    # 激活 Conda 环境
    source "$(conda info --base)/etc/profile.d/conda.sh"  # 初始化 Conda
    conda activate "$conda_env"
    # echo "Using Python from: $(which python)"
    
    # 运行 Python 脚本，传递文件路径和去掉后缀的文件名作为参数
    python_script="/home/wanhu/workspace/point2sdf/apps/vol_surf.py"
    # echo "$python_script"
    python "$python_script" --input "$file" --output "$output_folder/$file_basename" --cuda
    
    # 退出 Conda 环境
    conda deactivate
}

# 导出函数，以便并行执行时能够调用
export -f process_file

# 使用xargs并行处理文件
echo "${file_list[@]}" | tr ' ' '\n' | xargs -n 1 -P "$num_threads" -I {} bash -c 'process_file "$@"' _ {}

echo "所有文件已处理完成"