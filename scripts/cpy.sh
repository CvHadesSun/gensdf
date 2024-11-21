#!/bin/bash

# 源文件夹和目标文件夹路径
source_folder="/home/wanhu/dataset/subfolder_8_500_sample"
destination_folder="/home/wanhu/dataset/subfolder_8_10_debug"

# 确保目标文件夹存在
mkdir -p "$destination_folder"

# 获取源文件夹中的所有文件并随机抽取100个
# selected_files=$(find "$source_folder" -type f | shuf -n 10)
selected_files=$(find "$source_folder" -type d | shuf -n 10)

# 复制文件到目标文件夹
for file in $selected_files; do
    cp -r "$file" "$destination_folder"
done

echo "已成功随机复制100个文件到目标文件夹"