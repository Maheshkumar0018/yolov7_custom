import cv2
import torch
from pathlib import Path
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, xyxy2xywh,  increment_path
from utils.torch_utils import select_device, time_synchronized
import time
from numpy import random

from utils.plots import plot_one_box



def detect_objects(image_path, weights='yolov7.pt', img_size=640, conf_thres=0.25, iou_thres=0.45, device='', 
                   view_img=False, save_txt=False, save_conf=False, classes=None, agnostic_nms=False, augment=False, 
                   project='runs/detect', name='sample_testing', exist_ok=False, no_trace=False, save_img=False):
    
    save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir


    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    t0 = time.time()  # Start time of detection

    # Load model
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())  # model stride

    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Load image using OpenCV
    img0 = cv2.imread(image_path)  # BGR

    # Resize input image to the appropriate size
    img = cv2.resize(img0, (img_size, img_size))

    # Convert BGR to RGB
    img = img[..., ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = img.copy()  # Make a copy to ensure positive strides
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    with torch.no_grad():
        pred = model(img, augment=augment)[0]
    t2 = time_synchronized()

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    t3 = time_synchronized()

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        p, s, im0, frame = image_path, '', img0, 0

        save_path = str(Path(project) / name)  # img.jpg
        txt_path = str(Path(project) / 'labels' / Path(image_path).stem)  # img.txt
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if save_img or view_img:  # Add bbox to image
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

        # Print time (inference + NMS)
        print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
    
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    if save_img:
        save_path = str(save_dir / f"{name}.jpg")  # Ensure the path has the correct file extension
        # Write the image using OpenCV
        cv2.imwrite(save_path, im0)
        print(f"The image with the result is saved in: {save_path}")

    print(f'Done. ({time.time() - t0:.3f}s)')


# Example usage
image_path = "./inference/images/bus.jpg"
detect_objects(image_path, save_img=True)




---------------------------------------------------------------------------------------------------------------------
# Assuming box contains [x_min, y_min, x_max, y_max] coordinates
x_min, y_min, x_max, y_max = box

x_center = (x_min + x_max) / 2
y_center = (y_min + y_max) / 2
width = x_max - x_min
height = y_max - y_min

# Now the bounding box is represented as [x_center, y_center, width, height]
bounding_box = [x_center, y_center, width, height]



-------------------------
import os

def check_path_type(path):
    if os.path.isdir(path):
        print("The entered path is a folder.")
        # Add your code here for handling folder input
    elif os.path.isfile(path):
        print("The entered path is an image file.")
        # Add your code here for handling image file input
    else:
        print("The entered path is invalid or does not exist.")

# Example usage:
path = input("Enter a folder path or an image file path: ")
check_path_type(path)

