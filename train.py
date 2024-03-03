import time

from environment import Environment
from agent import Agent
from utils import get_start

EPISODE = 10


if __name__ == "__main__":

    environment = Environment()
    agent = Agent()

    for i in range(EPISODE):
        print(f"round {i}")
        environment.reset()
        episode_r_a, episode_r_d = 0, 0
        end = False

        EPSILON = 0.1
        counter = 0

        get_start()

        time.sleep(5)

        environment.get_frame()
        while not end:
            # counter += 1
            s_c = environment.frame_state
            a_a = agent.a_brain.choose_action(s_c, EPSILON)
            a_d = agent.d_brain.choose_action(s_c, EPSILON)

            # EPSILON /= counter

            agent.a_thread.append_action(a_a)
            agent.d_thread.append_action(a_d)

            # print(f"{a_a}, {a_d}")
            agent.perform_action()

            environment.get_frame()
            s_n = environment.frame_state
            r_a, r_d, end = environment.get_reward(end)

            agent.a_brain.store_transition(s_c, a_a, r_a, s_n)
            agent.d_brain.store_transition(s_c, a_d, r_d, s_n)

            episode_r_a += r_a  # 逐步加上一个episode内每个step的reward
            episode_r_a += r_d

            agent.a_brain.check_learn()
            agent.d_brain.check_learn()
        print(f"episode_r_a: {episode_r_a}, episode_r_d: {episode_r_a}")
        time.sleep(10)

