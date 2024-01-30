import yaml

def modify_classes(yolov7_config_path, new_nc):
    with open(yolov7_config_path, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if line.strip().startswith("nc:"):
            lines[i] = f"nc: {new_nc}\n"
            break

    with open(yolov7_config_path, 'w') as file:
        file.writelines(lines)

yolov7_config_path = 'cfg/training/yolov7.yaml'
new_nc = 2
modify_classes(yolov7_config_path, new_nc)
