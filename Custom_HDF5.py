import h5py
import cv2
import torch
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import non_max_suppression, scale_coords
from yolov7.utils.torch_utils import select_device

# Load YOLOv7 model
device = select_device('')
model = attempt_load(weights='yolov7.pt', map_location=device)

# Read HDF5 file
hdf5_file = 'your_file.h5'
with h5py.File(hdf5_file, 'r') as f:
    images = f['images'][:]  # Assuming 'images' is the dataset containing your images

# Resize images to 1280x1280
resized_images = [cv2.resize(img, (1280, 1280)) for img in images]

# Convert images to torch tensor
img_tensor = torch.from_numpy(resized_images).float().div(255.0).unsqueeze(0).to(device)

# Inference
pred = model(img_tensor)[0]

# Post-process predictions
pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)[0]

# If detections are found
if pred is not None:
    # Rescale coordinates to original image size
    pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], images.shape[1:])

    # Print results
    for det in pred:
        print(det)  # Print detection coordinates and class probabilities
