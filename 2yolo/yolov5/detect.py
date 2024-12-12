
import math
import threading
import time
import os
import sys
from pathlib import Path
import numpy as np
import torch

import mediapipe as mp

from gesture_judgment import detect_all_finger_state, detect_hand_state

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
from pynput import keyboard
from pynput.mouse import Button, Controller

import win32gui
import win32con

from simple_pid import PID
# from ctrl import PID
# PID 控制器参数，暂时PD控制，因为识别时间不确定且加了前馈
pid_x = PID(0.75, 0, 0.008, setpoint=0,sample_time=None ) #可以略微超调
pid_y = PID(0.75, 0, 0.008, setpoint=0,sample_time=None) # 压枪是个问题，建议不要超调

IsX2Pressed = False

# 创建一个事件，用于通知线程停止
stop_event = threading.Event()
should_run = True
toggle_target = 0
toggle_start = 0

global image_size
image_size = 1280
ScrSht_size_h = 520

save_count = 0

last_target = [0,0]

fire_flag = False




class HandDetectionThread(threading.Thread):
    def __init__(self, result_dict):
        super().__init__()
        self.result_dict = result_dict  # 共享字典
        self.running = True

    def run(self):
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils

        recent_states = [''] * 5  # 存储最近 5 次的手势判断结果
        cap = cv2.VideoCapture(0)

        prev_time = 0
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Horizontal mirror flipping
            h, w = frame.shape[:2]
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            keypoints = hands.process(image)

            if keypoints.multi_hand_landmarks:
                lm = keypoints.multi_hand_landmarks[0]
                lmHand = mp_hands.HandLandmark

                landmark_list = [[] for _ in range(6)]

                for index, landmark in enumerate(lm.landmark):
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    if index == lmHand.WRIST:
                        landmark_list[0].append((x, y))
                    elif 1 <= index <= 4:
                        landmark_list[1].append((x, y))
                    elif 5 <= index <= 8:
                        landmark_list[2].append((x, y))
                    elif 9 <= index <= 12:
                        landmark_list[3].append((x, y))
                    elif 13 <= index <= 16:
                        landmark_list[4].append((x, y))
                    elif 17 <= index <= 20:
                        landmark_list[5].append((x, y))

                all_points = {
                    'point0': landmark_list[0][0],
                    'point1': landmark_list[1][0], 'point2': landmark_list[1][1], 'point3': landmark_list[1][2], 'point4': landmark_list[1][3],
                    'point5': landmark_list[2][0], 'point6': landmark_list[2][1], 'point7': landmark_list[2][2], 'point8': landmark_list[2][3],
                    'point9': landmark_list[3][0], 'point10': landmark_list[3][1], 'point11': landmark_list[3][2], 'point12': landmark_list[3][3],
                    'point13': landmark_list[4][0], 'point14': landmark_list[4][1], 'point15': landmark_list[4][2], 'point16': landmark_list[4][3],
                    'point17': landmark_list[5][0], 'point18': landmark_list[5][1], 'point19': landmark_list[5][2], 'point20': landmark_list[5][3]
                }

                bend_states, straighten_states = detect_all_finger_state(all_points)
                current_state = detect_hand_state(all_points, bend_states, straighten_states)

                recent_states.pop(0)
                recent_states.append(current_state)

                if len(set(recent_states)) == 1:
                    current_hand_state = recent_states[0]
                else:
                    current_hand_state = 'None'  # 设定一个默认状态

                self.result_dict['hand_state'] = current_hand_state  # 更新共享字典

                cv2.putText(frame, current_hand_state, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                for hand_landmarks in keypoints.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                               mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                                               mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))

            else:
                self.result_dict['hand_state'] = 'None'  # 没有检测到手势，更新为默认状态

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Hand Detection", frame)
            if cv2.waitKey(1) == ord("l"):  # Press 'l' to stop
                break

        cap.release()
        cv2.destroyAllWindows()


    def stop(self):
        self.running = False



result_dict = {'hand_state': None}  # 创建共享字典

def on_press(key):
    global toggle_start
    try:
        # 检查按键是否是"f"键（包括大写和小写）
        if key.char.lower() == 'f' or key.char.upper() == 'F':
            toggle_start = 1
            print('The "f" key was pressed')
    except AttributeError:
        # 如果按键不是字符类型，则可能是一个特殊按键
        pass

def on_release(key):
    global toggle_target, toggle_start
    try:
        if toggle_start:
            toggle_target = 1 if toggle_target == 0 else 0
            print("target changed to ", toggle_target)
            toggle_start = 0
    except AttributeError:
        # 如果按键不是字符类型，则可能是一个特殊按键
        pass

def shift_target():
    global toggle_start, toggle_target

# 设置监听器
# def keyboard_listener():
#     global should_run
#     with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
#         while True:
#             if should_run:
#                 listener.join()
#             else:
#                 listener.stop()
#                 break
def keyboard_listener():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        while not stop_event.is_set():
            listener.join()
            # time.sleep(0.001)
            pass
        listener.stop()  # 退出监听
        print("Keyboard listener stopped")

def mouse_click(x,y,button,pressed):
    global IsX2Pressed
    print(x,y,button,pressed)
    if (pressed and button == pynput.mouse.Button.x2):
        IsX2Pressed = True
    else:
        IsX2Pressed = False

# def mouse_listener():
#     global should_run
#     with Listener(on_click=mouse_click) as listener:
#         while True:
#             if should_run:
#                 listener.join()
#             else:
#                 listener.stop()
#                 break
def mouse_listener():
    with Listener(on_click=mouse_click) as listener:
        while not stop_event.is_set():
            listener.join()
            # time.sleep(0.001)
            pass
        listener.stop()  # 退出监听

mouse = Controller()

def gesture_state():
    global IsX2Pressed , fire_flag
    if (result_dict['hand_state']=='Left') or (result_dict['hand_state']=='Return'):
        IsX2Pressed = True
        # print('kaile')
        if(result_dict['hand_state']=='Left'):
            fire_flag = True
            mouse.press(Button.left)
            # print('fire')
    else:
        IsX2Pressed = False
        fire_flag = False
        mouse.release(Button.left)
        # print('guanle')


def gesture_listener():
    while True:
        gesture_state()
        time.sleep(0.0001)

@smart_inference_mode()


# def get_highest_confidence_target(pred):
#     """
#     从预测结果中获取置信度最高的目标信息。

#     参数：
#         pred (list): 预测结果，包含检测到的所有目标信息。

#     返回：
#         list: 置信度最高的目标信息，格式为 [x_min, y_min, x_max, y_max, confidence, class, distance]。
#               如果没有检测到目标，返回 None。
#     """
#     max_confidence = 0
#     best_detection = None

#     for det in pred:
#         for *xyxy, conf, cls in det:
#             if conf > max_confidence:
#                 max_confidence = conf
#                 best_detection = [*xyxy, conf, cls]

#     if best_detection is not None:
#         xywh = (xyxy2xywh(torch.tensor(best_detection[:4]).view(1, 4))).view(-1).tolist()
#         distance = math.sqrt((xywh[0]-320)**2 + (xywh[1]-320)**2)
#         xywh.append(distance)
#         return xywh
#     else:
#         return None


def run():
    # Load model
    device = torch.device("cuda:0")
    model = DetectMultiBackend(weights="./weight/best_CS2n6_2.pt", device=device, dnn=False, data=False, fp16=True)
    # model1 = DetectMultiBackend(weights="./weight/best_CS2s.pt", device=device, dnn=False, data=False, fp16=True)
    global IsX2Pressed, toggle_target
    while True:
        if should_run !=True:
            break
        #read images
        im = screenshot()
        im0=im
        im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        #process images
        im = letterbox(im,(image_size,image_size),stride=32,auto=True)[0]#paddle resize
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
        # pred1 = model1(im, augment=False, visualize=False)#debug

        pred = non_max_suppression(pred, conf_thres=0.45, iou_thres=0.45, classes=toggle_target, max_det=5)
        # pred1 = non_max_suppression(pred1, conf_thres=0.6, iou_thres=0.45, classes=toggle_target, max_det=5)
        end = time.time()


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
                    # line = cls, *xywh, conf  # label format
                    #print(xywh)
                    X=xywh[0]-ScrSht_size_h
                    Y=xywh[1]-ScrSht_size_h

                    distance = math.sqrt(X**2+Y**2)
                    xywh.append(distance)
                    annotator.box_label(xyxy, label=f'[{int(cls)}Distance:{round(distance,2)}]',
                                        color=(34, 139, 34),
                                        txt_color=(0, 191, 255))

                    distance_list.append(distance)
                    target_list.append(xywh)

                target_info= target_list[distance_list.index(min(distance_list))]
                # target_info = get_highest_confidence_target([det])

                global last_target
                # 计算需要移动的距离
                target_x = target_info[0] - ScrSht_size_h
                target_y = target_info[1] - target_info[3]/4 - ScrSht_size_h # 大约是胸颈部

                if IsX2Pressed:
                    print('kaile')
                    print(f'推理所需时间{end-start}s')
                    # 计算PID输出
                    move_x = pid_x(-target_x)
                    move_y = pid_y(-target_y)

                    # mouse_xy(int(move_x+0.46*last_target[0]),int(move_y+0.3*last_target[1]))
                    mouse_xy(int(move_x),int(move_y))
                    # mouse_xy(int(0.46*target_x+move_x),int(0.46*target_y+move_y))
                last_target = [target_x,target_y]

            else:
                # if IsX2Pressed:
                #     global save_count
                #     save_dir = './nd'
                #     if not os.path.exists(save_dir):
                #         os.makedirs(save_dir)
                #     save_path = f'{save_dir}/no_detections_{save_count}.png'
                #     cv2.imwrite(save_path,im0)
                #     save_count += 1
                last_target = [0,0]
            # im0 = annotator.result()
            # cv2.imshow('window', im0)
            # hwnd = win32gui.FindWindow(None, 'window')
            # win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
            #                       win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
            # cv2.waitKey(1)




# if __name__ == "__main__":
#     print("start")
#     try:
#         threading.Thread(target=mouse_listener).start()
#         threading.Thread(target=keyboard_listener).start()
#         run()
#     except KeyboardInterrupt:
#         should_run = False

#         stop_event.set()  # 发送停止信号
#         print("Received KeyboardInterrupt, stopping threads...")
#         threading.Thread(target=mouse_listener).join()
#         threading.Thread(target=keyboard_listener).join()
#         print("end")
if __name__ == "__main__":
    print("start")
    try:
        # 启动鼠标和键盘监听线程
        mouse_thread = threading.Thread(target=mouse_listener)
        keyboard_thread = threading.Thread(target=keyboard_listener)
        gesture_listener_thread = threading.Thread(target=gesture_listener)

        mouse_thread.start()
        keyboard_thread.start()
        gesture_listener_thread.daemon = True  # 使线程为守护线程，程序退出时自动关闭
        gesture_listener_thread.start()



        hand_detection_thread = HandDetectionThread(result_dict)
        hand_detection_thread.start()

        run()  # 运行主程序逻辑

    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, stopping threads...")
        stop_event.set()  # 发送停止信号
    finally:
        # 等待鼠标和键盘监听线程结束
        mouse_thread.stop()
        mouse_thread.join()

        keyboard_thread.stop()
        keyboard_thread.join()

        hand_detection_thread.stop()
        hand_detection_thread.join()
        print("end")


