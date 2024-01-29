# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""
import math
import threading
import time
import argparse
import csv
import os
import platform
import sys
from pathlib import Path
import numpy as np
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox

from ScreenShot import screenshot
from SendInput import *

import pynput.mouse
from pynput.mouse import Listener

import win32gui
import win32con

IsX2Pressed = False

def mouse_click(x,y,button,pressed):
    global IsX2Pressed
    print(x,y,button,pressed)
    if pressed and button == pynput.mouse.Button.x2:
        IsX2Pressed=True
        print(IsX2Pressed)
    else:
        IsX2Pressed=False

def mouse_listener():
    with Listener(on_click=mouse_click) as listener:
        listener.join()
@smart_inference_mode()
def run():
    # Load model
    device = torch.device("cuda:0")
    model = DetectMultiBackend(weights="./weight/yolov5n.pt", device=device, dnn=False, data=False, fp16=True)
    global IsX2Pressed
    while True:
        #read images
        im = screenshot()
        im0=im
        #process images
        im = letterbox(im,(640,640),stride=32,auto=True)[0]#paddle resize
        im = im.transpose((2,0,1))[::-1]#HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im) #contiguous

        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        #推理
        start = time.time()
        pred = model(im, augment=False, visualize=False)#debug

        pred = non_max_suppression(pred, conf_thres=0.6, iou_thres=0.45, classes=0, max_det=1000)#pred 0:person,(0,2)
        end = time.time()
        #print(f'推理所需时间{end-start}s')

        # Process predictions
        for i, det in enumerate(pred):  # per image
            #gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0,line_width=1)
            if len(det):
                distance_list=[]
                target_list=[]
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):#target info process

                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
                    #line = cls, *xywh, conf  # label format
                    #print(xywh)
                    X=xywh[0]-320
                    Y=xywh[1]-320
                    distance = math.sqrt(X**2+Y**2)
                    xywh.append(distance)
                    annotator.box_label(xyxy, label=f'[{int(cls)}Distance:{round(distance,2)}]',
                                        color=(34, 139, 34),
                                        txt_color=(0, 191, 255))

                    distance_list.append(distance)
                    target_list.append(xywh)

                target_info= target_list[distance_list.index(min(distance_list))]
                if IsX2Pressed:
                    print('kaile')
                    mouse_xy(int(target_info[0]-320),int(target_info[1]-320))
                    time.sleep(0.002)
            im0 = annotator.result()
            # cv2.imshow('window', im0)
            # hwnd = win32gui.FindWindow(None, 'window')
            # win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
            #                       win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
            # cv2.waitKey(1)









if __name__ == "__main__":
    print("start")
    threading.Thread(target=mouse_listener).start()
    run()
