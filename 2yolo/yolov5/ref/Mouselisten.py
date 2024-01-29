import pynput.mouse
from pynput.mouse import Listener

def mouse_click(x,y,button,pressed):
    print(x,y,button,pressed)
    if pressed and button == pynput.mouse.Button.x2:
        print('pressed')

with Listener(on_click=mouse_click) as listener:
    listener.join()