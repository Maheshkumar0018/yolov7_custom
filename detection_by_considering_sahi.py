import cv2
import torch
import numpy as np
import os
import threading
from queue import Queue
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, TracedModel
from collections import defaultdict


def split_image_into_tiles(image, num_rows=3, num_cols=4, overlap_ratio=0.2):
    tiles = []
    positions = []

    h, w = image.shape[:2]

    # Compute tile size with desired overlap
    tile_h = int(h / (num_rows - (num_rows - 1) * overlap_ratio))
    tile_w = int(w / (num_cols - (num_cols - 1) * overlap_ratio))

    stride_y = int(tile_h * (1 - overlap_ratio))
    stride_x = int(tile_w * (1 - overlap_ratio))

    for row in range(num_rows):
        for col in range(num_cols):
            y = row * stride_y
            x = col * stride_x

            if y + tile_h > h:
                y = h - tile_h
            if x + tile_w > w:
                x = w - tile_w

            tile = image[y:y + tile_h, x:x + tile_w]
            if tile.shape[0] == tile_h and tile.shape[1] == tile_w:
                tiles.append(tile)
                positions.append((x, y))

    return tiles, positions



def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True,
              scaleFill=False, scaleup=True, stride=32):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def detect_tile(model, tile, tile_pos, device, img_size, stride, half, conf_thres, iou_thres, results_queue):
    im0 = tile.copy()
    img, ratio, (dw, dh) = letterbox(im0, new_shape=img_size, stride=stride)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres)

    tile_coords = []
    class_ids = []
    scores = []

    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls_id in det:
                x0, y0, x1, y1 = map(int, xyxy)
                x0 += tile_pos[0]
                x1 += tile_pos[0]
                y0 += tile_pos[1]
                y1 += tile_pos[1]
                tile_coords.append([x0, y0, x1, y1])
                class_ids.append(int(cls_id))
                scores.append(float(conf))

    results_queue.put((tile_coords, class_ids, scores))


def merge_boxes_sahi_style(boxes, class_ids, scores, iou_threshold=0.5):
    grouped = defaultdict(list)
    for box, cls_id, score in zip(boxes, class_ids, scores):
        grouped[cls_id].append((box, score))

    final_boxes = []
    final_classes = []

    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    for cls_id, items in grouped.items():
        items.sort(key=lambda x: x[1], reverse=True)
        used = [False] * len(items)
        for i, (box, score) in enumerate(items):
            if used[i]:
                continue
            cluster = [(box, score)]
            used[i] = True
            for j in range(i + 1, len(items)):
                if not used[j] and iou(box, items[j][0]) > iou_threshold:
                    cluster.append(items[j])
                    used[j] = True
            x0 = sum(b[0][0] * b[1] for b in cluster) / sum(b[1] for b in cluster)
            y0 = sum(b[0][1] * b[1] for b in cluster) / sum(b[1] for b in cluster)
            x1 = sum(b[0][2] * b[1] for b in cluster) / sum(b[1] for b in cluster)
            y1 = sum(b[0][3] * b[1] for b in cluster) / sum(b[1] for b in cluster)
            final_boxes.append([int(x0), int(y0), int(x1), int(y1)])
            final_classes.append(cls_id)

    return final_boxes, final_classes


def detect(source, weights, img_size=640, conf_thres=0.25, iou_thres=0.45, device='', trace=False):
    device = select_device(device)
    half = device.type != 'cpu'

    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    img_size = check_img_size(img_size, s=stride)
    if trace:
        model = TracedModel(model, device, img_size)
    if half:
        model.half()

    source_img = source.copy()
    tiles, positions = split_image_into_tiles(source_img)
    print("Total tiles: ", len(tiles))

    threads = []
    results_queue = Queue()

    for i, tile in enumerate(tiles):
        thread = threading.Thread(
            target=detect_tile,
            args=(model, tile, positions[i], device, img_size, stride, half, conf_thres, iou_thres, results_queue)
        )
        thread.start()
        threads.append(thread)

    for t in threads:
        t.join()

    all_coords, all_classes, all_scores = [], [], []
    while not results_queue.empty():
        coords, classes, scores = results_queue.get()
        all_coords.extend(coords)
        all_classes.extend(classes)
        all_scores.extend(scores)

    merged_coords, merged_classes = merge_boxes_sahi_style(all_coords, all_classes, all_scores)
    return source_img, merged_coords, merged_classes, all_coords, all_classes


if __name__ == "__main__":
    path = './inference/images/horses.jpg'
    weights = './yolov7.pt'

    source = cv2.imread(path)
    source = cv2.resize(source, (4000, 1000), interpolation=cv2.INTER_LINEAR)

    original_img, tile_coordinates, class_ids, all_coords, all_classes = detect(
        source, weights, img_size=640, conf_thres=0.25, iou_thres=0.45, device=''
    )

    raw_img = original_img.copy()
    merged_img = original_img.copy()

    for box, cls_id in zip(all_coords, all_classes):
        label = f"raw-{cls_id}"
        plot_one_box(box, raw_img, label=label, color=[255, 0, 0], line_thickness=1)

    for box, cls_id in zip(tile_coordinates, class_ids):
        label = f"merged-{cls_id}"
        plot_one_box(box, merged_img, label=label, color=[0, 255, 0], line_thickness=2)

    os.makedirs("output", exist_ok=True)
    raw_output_path = "output/raw_detections.jpg"
    merged_output_path = "output/merged_detections.jpg"

    cv2.imwrite(raw_output_path, raw_img)
    cv2.imwrite(merged_output_path, merged_img)

    print(f"Saved raw detections to: {raw_output_path}")
    print(f"Saved merged detections to: {merged_output_path}")
