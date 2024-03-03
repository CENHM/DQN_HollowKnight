import ctypes
import random
import time
import mss
import mss.tools
import cv2
import numpy as np
import pyautogui as pyautogui
import win32api
import win32gui

from keys import *
from utils import init_window

jSW_MAXIMIZE = 3


user = ctypes.WinDLL('user32', use_last_error=True)
hwnd = win32gui.FindWindow(None, 'Hollow Knight')

init_window(hwnd)

action = ActionThread(1, 'thread-action')
direction = DirectionThread(2, 'thread-direction')
action.start()
direction.start()
# action.action.append(4)


capture_width, capture_height = 1280, 720

# screenshot1 = pyautogui.screenshot()

with mss.mss() as sct:
    # 获取屏幕尺寸
    monitor = sct.monitors[1]

    start_time = time.time()

    screenshot = sct.grab(monitor)

        # 处理截图，例如保存或进行 YOLO 推理
        # 例如，保存为文件（注释掉下一行代码以避免保存大量文件）
        # mss.tools.to_png(screenshot.rgb, screenshot.size, output=f'screenshot-{start_time}.png')

a = 0

# 将截图转换为OpenCV图像
frame = np.array(screenshot)[:, :, :3]

# 调整捕获窗口的大小
frame = cv2.resize(frame, (capture_width, capture_height))

frame1 = frame[145:606, 174:1105, :]

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

frame1 = cv2.resize(frame1, (224, 224))

# hp = get_agent_hp(frame)kwa

# cv2.imshow('Screen Capture', frame)
#
cv2.imwrite(f'./t/{a}.jpg', frame1)


# action.action.append(random.randint(0, 4))
# action.action.append(random.randint(0, 3))


