import threading
import time

# 创建一个事件，用于通知线程停止
stop_event = threading.Event()

def mouse_listener():
    while not stop_event.is_set():
        print("Listening for mouse events...")
        time.sleep(1)  # 模拟鼠标事件监听

def keyboard_listener():
    while not stop_event.is_set():
        print("Listening for keyboard events...")
        time.sleep(1)  # 模拟键盘事件监听

def run():
    # 这里是你的主程序逻辑
    while not stop_event.is_set():
        print("Running main program...")
        time.sleep(1)  # 模拟主程序的工作

if __name__ == "__main__":
    print("start")
    try:
        # 启动线程
        mouse_thread = threading.Thread(target=mouse_listener)
        keyboard_thread = threading.Thread(target=keyboard_listener)

        mouse_thread.start()
        keyboard_thread.start()

        run()  # 运行主程序

    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, stopping threads...")
        stop_event.set()  # 发送停止信号
    finally:
        # 等待线程结束
        mouse_thread.join()
        keyboard_thread.join()
        print("end")
