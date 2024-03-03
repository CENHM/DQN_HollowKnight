import ctypes

import win32api
import win32gui
import win32process
import time
from keys import press_esc, ActionThread, DirectionThread, press_slight_jump, press_up, press_enter


# Kernel32 = ctypes.WinDLL('kernel32.dll')


def get_window():
    hwnd = win32gui.FindWindow(None, 'Hollow Knight')
    init_window(hwnd)
    # a_thread = ActionThread(1, 'thread-action')
    # d_thread = DirectionThread(2, 'thread-direction')
    return hwnd


def init_window(hwnd):
    SW_MAXIMIZE = 3
    ctypes.windll.user32.ShowWindow(hwnd, SW_MAXIMIZE)
    press_esc()
    time.sleep(2)


def get_start():
    press_slight_jump()
    time.sleep(2)
    press_up()
    time.sleep(1)
    press_enter()





