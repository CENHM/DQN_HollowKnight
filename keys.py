import ctypes
import random
import threading
import time

import win32api
import win32con

from WindowsAPI import INPUT, KEYBDINPUT

INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2

KEY_EVENT_KEYUP = 0x0002

ESC = 27
ENTER = 13

LEFT = 65
UP = 87
RIGHT = 68
DOWN = 83

JUMP = 38
ATTACK = 74
DASH = 75
HEAL = 81

def ReleaseKey(user, hexKeyCode):
    x = INPUT(type=INPUT_KEYBOARD,
              ki=KEYBDINPUT(wVk=hexKeyCode,
                            dwFlags=KEY_EVENT_KEYUP))
    user.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))


def PressKey(user, hexKeyCode):
    x = INPUT(type=INPUT_KEYBOARD,
              ki=KEYBDINPUT(wVk=hexKeyCode))
    user.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))


def press_esc():
    win32api.keybd_event(ESC, 0, 0, 0)
    win32api.keybd_event(ESC, 0, win32con.KEYEVENTF_KEYUP, 0)


def press_nothing():
    time.sleep(0.01)


def press_left():
    win32api.keybd_event(LEFT, 0, 0, 0)
    time.sleep(0.1)
    win32api.keybd_event(LEFT, 0, win32con.KEYEVENTF_KEYUP, 0)


def press_up():
    win32api.keybd_event(UP, 0, 0, 0)
    time.sleep(0.1)
    win32api.keybd_event(UP, 0, win32con.KEYEVENTF_KEYUP, 0)


def press_right():
    win32api.keybd_event(RIGHT, 0, 0, 0)
    time.sleep(0.1)
    win32api.keybd_event(RIGHT, 0, win32con.KEYEVENTF_KEYUP, 0)


def press_down():
    win32api.keybd_event(DOWN, 0, 0, 0)
    time.sleep(0.1)
    win32api.keybd_event(DOWN, 0, win32con.KEYEVENTF_KEYUP, 0)


def press_jump():
    win32api.keybd_event(JUMP, 0, 0, 0)
    time.sleep(0.3)
    win32api.keybd_event(JUMP, 0, win32con.KEYEVENTF_KEYUP, 0)


def press_slight_jump():
    win32api.keybd_event(JUMP, 0, 0, 0)
    time.sleep(0.05)
    win32api.keybd_event(JUMP, 0, win32con.KEYEVENTF_KEYUP, 0)


def press_enter():
    win32api.keybd_event(ENTER, 0, 0, 0)
    time.sleep(0.05)
    win32api.keybd_event(ENTER, 0, win32con.KEYEVENTF_KEYUP, 0)


def press_attack():
    win32api.keybd_event(ATTACK, 0, 0, 0)
    time.sleep(0.01)
    win32api.keybd_event(ATTACK, 0, win32con.KEYEVENTF_KEYUP, 0)


def press_dash():
    win32api.keybd_event(DASH, 0, 0, 0)
    time.sleep(0.01)
    win32api.keybd_event(DASH, 0, win32con.KEYEVENTF_KEYUP, 0)


def press_heal():
    win32api.keybd_event(HEAL, 0, 0, 0)
    time.sleep(3)
    win32api.keybd_event(HEAL, 0, win32con.KEYEVENTF_KEYUP, 0)


actions = [press_jump, press_attack, press_dash, press_heal]
directions = [press_right, press_left, press_right, press_left]


# Run the action
def take_action(action):
    actions[action]()


def take_direction(direction):
    directions[direction]()


class ActionThread(threading.Thread):
    def __init__(self, thread_id, name):
        threading.Thread.__init__(self)
        self.threadID = thread_id
        self.name = name
        self.action = None

    def run(self):
        take_action(self.action)

    def append_action(self, action):
        self.action = action


class DirectionThread(threading.Thread):
    def __init__(self, thread_id, name):
        threading.Thread.__init__(self)
        self.threadID = thread_id
        self.name = name
        self.action = None

    def run(self):
        take_direction(self.action)

    def append_action(self, action):
        self.action = action

