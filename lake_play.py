import time

import gymnasium as gym
import numpy as np

if __name__ == "__main__":
    environment = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
    environment.reset()
    environment.render()

    Q = np.load("agent.npy")
    episodes = 1000  # Total number of episodes
    alpha = 0.5  # Learning rate
    gamma = 0.9  # Discount factor

    action_set = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}

    for _ in range(episodes):
        s, _ = environment.reset()
        terminated = False
        while not terminated:
            a = np.argmax(Q[s])  # Pick the best action that you know about
            print(action_set[a])
            next_s, r, terminated, truncated, info = environment.step(a)
            s = next_s
            time.sleep(1)
