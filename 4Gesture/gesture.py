import mediapipe as mp
import cv2
import time  # 导入 time 模块

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode
# 回调函数，显示手势识别的结果
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global img  # 需要全局变量 img
    if result.gestures:  # 如果识别到了手势
        # 获取第一个手势的序号（假设每次识别的手势列表只有一个手势）
        gesture_id = result.gestures[0][0].category_index  # 手势模型的序号
        gesture_name = result.gestures[0][0].category_name  # 手势模型的名称
        
        # 在图像上显示识别到的手势序号和名称
        cv2.putText(img, f'Gesture: {gesture_name} ({gesture_id})', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 绘制手部关键点
    if result.hand_landmarks:  # 检查是否有检测到手部关键点
        for hand_landmarks in result.hand_landmarks:
            for landmark in hand_landmarks:
                # 获取关键点的 x 和 y 坐标，并转换为图像坐标
                x = int(landmark.x * img.shape[1])
                y = int(landmark.y * img.shape[0])
                # 在图像上绘制关键点
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='F:\!繁务区\School_work\2024-2025秋季学期\IPCV\syeda\gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM)

print("000")
cap = cv2.VideoCapture(0)  # 打开摄像头
print("111")

with GestureRecognizer.create_from_options(options) as recognizer:
    print("222")
    while True:  # 无限循环
        ret, img = cap.read()  # 读取摄像头的图像帧
        img = cv2.flip(img, 1)  # 对 img 图像进行水平翻转

        if ret:
            print("999")
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)

            # 获取当前时间戳（毫秒）
            frame_timestamp_ms = int(time.time() * 1000)

            # 进行手势识别
            recognizer.recognize_async(mp_image, frame_timestamp_ms)

            # 显示图像并展示手势识别结果
            cv2.imshow("Hand Gesture Recognition", img)
        else:
            print("cannot open camera")
        

cap.release()
cv2.destroyAllWindows()
