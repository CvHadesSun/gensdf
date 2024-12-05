import os
import shutil

def split_objs_into_subfolders(src_folder, num_subfolders=3):
    # Get all .obj files in the source folder
    # obj_files = [f for f in os.listdir(src_folder) if f.endswith('.obj')]
    files = os.listdir(src_folder)
    
    # Create subfolders if they don't exist
    for i in range(1, num_subfolders + 1):
        subfolder_path = os.path.join(src_folder, f"subfolder_{i}")
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
    
    # Distribute the files into subfolders
    for idx, obj_file in enumerate(files):
        subfolder_index = (idx % num_subfolders) + 1
        subfolder_path = os.path.join(src_folder, f"subfolder_{subfolder_index}")
        
        # Move the .obj file to the selected subfolder
        src_file_path = os.path.join(src_folder, obj_file)
        dst_file_path = os.path.join(subfolder_path, obj_file)
        shutil.move(src_file_path, dst_file_path)

    print(f"Successfully split {len(files)} .obj files into {num_subfolders} subfolders.")

# Example usage
src_folder = '/home/wanhu/dataset/subfolder_8_sample'  # Replace with your folder path
split_objs_into_subfolders(src_folder)