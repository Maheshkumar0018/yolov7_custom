# main_script.py

from custom_detect import detect

if __name__ == '__main__':
    source = './inference/output_video.mp4'
    weights = './yolov7.pt'  # or the path to your trained weights
    detect(source=source, weights=weights, view_img=True, save_txt=True, save_conf=True, target_cls=17)
    #detect(source=source, weights=weights, target_class=18, save_img=False, view_img=True)

