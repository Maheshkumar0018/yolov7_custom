import cv2
import torch
from pathlib import Path
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, increment_path
from utils.torch_utils import select_device, time_synchronized
import time

def detect_objects(image_path, weights='yolov7.pt', img_size=640, conf_thres=0.25, iou_thres=0.45, device='', 
                   project='./sample', name='sample_testing', exist_ok=False, save_img=False, save_txt=False, show_img=True):

    # Extract filename from the image path
    filename = Path(image_path).stem

    save_dir = Path(project) 
    #save_dir.mkdir(parents=True, exist_ok=exist_ok)  # make dir if it doesn't exist

    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    t0 = time.time()  # Start time of detection

    # Load model
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())  # model stride

    if half:
        model.half()  # to FP16

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
        pred = model(img)[0]
    t2 = time_synchronized()

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    t3 = time_synchronized()

    # Lists to store bounding box coordinates, classes, and scores
    bounding_boxes = []
    classes = []
    scores = []

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Append bounding box coordinates, classes, and scores
            for *xyxy, conf, cls in det:
                bounding_boxes.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                classes.append(int(cls))
                scores.append(float(conf))

    if save_txt:
        with open(save_dir / f"{filename}.txt", 'w') as f:
            for box, cls, score in zip(bounding_boxes, classes, scores):
                f.write(f"{box[0]} {box[1]} {box[2]} {box[3]} {cls} {score}\n")

    # Draw bounding boxes with class names and scores on the image
    if save_img:
        for (x_min, y_min, x_max, y_max), cls, conf in zip(bounding_boxes, classes, scores):
            color = (0, 255, 0)  # Green color for bounding boxes
            img0 = cv2.rectangle(img0, (x_min, y_min), (x_max, y_max), color, 2)
            label = f'{model.names[int(cls)]} {conf:.2f}'
            cv2.putText(img0, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imwrite(str(save_dir / f"{filename}_detected.jpg"), img0)

    # Display the image
    if show_img:
        cv2.imshow('Detected Image', img0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f'Done. ({(time.time() - t0):.3f}s)')

    return bounding_boxes, classes, scores

# Example usage:
image_path = "./inference/images/bus.jpg"
bounding_boxes, classes, scores = detect_objects(image_path, save_img=True, save_txt=True)

print("Bounding boxes:", bounding_boxes)
print("Classes:", classes)
print("Scores:", scores)
