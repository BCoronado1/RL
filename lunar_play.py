import pickle

import gymnasium as gym
import numpy as np


def obs2hash(obs: np.ndarray) -> int:
    return hash(tuple([round(v, 1) for v in obs.tolist()]))


if __name__ == "__main__":
    environment = gym.make("LunarLander-v2", render_mode="human")

    Q = np.load("lunar_agent.npy")
    with open("lunar_agent_state_mapping.pickle", "rb") as f:
        state_mapping = pickle.load(f)
    episodes = 1000  # Total number of episodes
    alpha = 0.5  # Learning rate
    gamma = 0.9  # Discount factor
    action_set = {0: "x", 1: "<", 2: "^", 3: ">"}

    for _ in range(episodes):
        raw_s, _ = environment.reset()
        raw_s_hash = obs2hash(raw_s)
        if raw_s_hash in state_mapping:
            s = state_mapping[raw_s_hash]
        else:
            s = 0
        terminated = False
        while not terminated:
            a = np.argmax(Q[s])  # Pick the best action that you know about
            # a = environment.action_space.sample()
            print(action_set[a])
            raw_next_s, r, terminated, truncated, info = environment.step(a)
            next_s_hash = obs2hash(raw_next_s)
            if next_s_hash in state_mapping:
                next_s = state_mapping[next_s_hash]
            else:
                next_s = 0
            s = next_s
