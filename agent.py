import numpy as np
from keys import ActionThread, DirectionThread
from network import DQN


class Agent(object):
    def __init__(self):
        self.a_thread = ActionThread(1, 'thread-action')
        self.d_thread = DirectionThread(2, 'thread-direction')
        self.a_brain = DQN()
        self.d_brain = DQN()

    def perform_action(self):
        self.a_thread.run()
        self.d_thread.run()



