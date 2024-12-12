import os

def replace_starting_2_with_1(file_path):
    # 读取文件内容并替换每行开头的2为1
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines = [line.replace('2', '1', 1) for line in lines]

    # 将替换后的内容写回文件
    with open(file_path, 'w') as file:
        file.writelines(lines)

def replace_starting_2_with_1_in_folder(folder_path):
    # 遍历文件夹内所有txt文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            replace_starting_2_with_1(file_path)

# 指定要处理的文件夹路径
folder_path = 'E:/ML_Prj/CS2/CS2_Autoaim/2yolo/yolov5-7.0-full4train/data/yolo_dataset/labels/val'

# 调用函数处理文件夹内所有txt文件
replace_starting_2_with_1_in_folder(folder_path)
