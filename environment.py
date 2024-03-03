import cv2
import numpy as np

from utils import get_window
import mss.tools
import time


class Environment(object):
    def __init__(self):
        self.window = get_window()
        self.frame_hp = None
        self.frame_state = None
        self.hornet_hurt = False

        self.prev_knight_hp = 100.0
        self.prev_hornet_hp = 100.0

    def reset(self):
        self.frame_hp = None
        self.frame_state = None
        self.hornet_hurt = False

        self.prev_knight_hp = 100.0
        self.prev_hornet_hp = 100.0

    def get_frame(self):
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            screenshot = sct.grab(monitor)
        frame = np.array(screenshot)[:, :, :3]
        capture_width, capture_height = 1280, 720
        frame_hp = cv2.resize(frame, (capture_width, capture_height))
        self.frame_hp = cv2.cvtColor(frame_hp, cv2.COLOR_BGR2GRAY)
        self.frame_state = cv2.resize(frame_hp[145:606, 174:1105, :], (224, 224))

    def get_hornet_hp(self):
        if not self.hornet_hurt \
                and self.frame_hp[657, 159] in range(48, 53) \
                and self.frame_hp[657, 1119] in range(48, 53):
            self.hornet_hurt = True

        if self.frame_hp[657, 162] < 20 and self.hornet_hurt:
            return 0.0
        elif self.frame_hp[657, 162] > 130:
            for i in range(1, 960):
                if self.frame_hp[657, i + 159] < 130:
                    return i / 960 * 100
        else:
            return 100.

    def get_knight_hp(self):
        mask = np.zeros(9)
        mask[0] = 1 if self.frame_hp[80, 225] > 150 else 0
        mask[1] = 1 if self.frame_hp[80, 261] > 150 else 0
        mask[2] = 1 if self.frame_hp[80, 297] > 150 else 0
        mask[3] = 1 if self.frame_hp[80, 332] > 150 else 0
        mask[4] = 1 if self.frame_hp[80, 368] > 150 else 0
        mask[5] = 1 if self.frame_hp[80, 403] > 150 else 0
        mask[6] = 1 if self.frame_hp[80, 439] > 150 else 0
        mask[7] = 1 if self.frame_hp[80, 475] > 150 else 0
        mask[8] = 1 if self.frame_hp[80, 510] > 150 else 0
        return float(np.sum(mask)) / 9 * 100

    def get_reward(self, end):
        knight_hp = self.get_knight_hp()
        hornet_hp = self.get_hornet_hp()
        r_a, r_d = 0., 0.
        if knight_hp < self.prev_knight_hp:
            r_a += -10
            r_d += -10
        if hornet_hp < self.prev_hornet_hp:
            r_a += 20
            r_d += 20
        if self.get_hornet_hp() == 0.0 and not end:
            print("win")
            end = True
            r_a += 120
            r_d += 120
        elif self.get_knight_hp() == 0.0 and not end:
            print("lose")
            end = True
            r_a -= 120
            r_d -= 120
        self.prev_knight_hp = knight_hp
        self.prev_hornet_hp = hornet_hp
        return r_a, r_d, end


