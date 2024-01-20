import pickle
import random
import time

import gymnasium as gym
import numpy as np


def obs2hash(obs: np.ndarray) -> int:
    return hash(tuple([round(v, 1) for v in obs.tolist()]))


if __name__ == "__main__":
    environment = gym.make("LunarLander-v2")
    Q = None
    state_mapping = dict()
    episodes = 20000  # Total number of episodes
    alpha = 0.9  # Learning rate
    gamma = 0.9  # Discount factor
    epsilon = 0.25  # Exploration vs Exploitation
    for ep_num in range(episodes):
        if ep_num % 1000 == 0:
            print(f"Episode {ep_num}")
        raw_s, _ = environment.reset()
        s_hash = obs2hash(raw_s)
        if Q is None:  # Fix this bad smell
            Q = np.random.random((1, environment.action_space.n))
            state_mapping[s_hash] = 0
            s = 0
        else:
            if s_hash not in state_mapping:
                Q = np.vstack([Q, np.random.random(environment.action_space.n)])
                state_mapping[s_hash] = len(state_mapping)
            s = state_mapping[s_hash]

        terminated = False
        while not terminated:
            a = environment.action_space.sample() if random.uniform(0, 1) <= epsilon else np.argmax(Q[s])
            raw_next_s, r, terminated, truncated, info = environment.step(a)
            next_s_hash = obs2hash(raw_next_s)
            if next_s_hash not in state_mapping:
                next_state_idx = len(state_mapping)
                state_mapping[next_s_hash] = next_state_idx
                Q = np.vstack([Q, np.random.random(environment.action_space.n)])
            next_s = state_mapping[next_s_hash]
            Q[s, a] = (1 - alpha) * Q[s, a] + alpha * (r + gamma * np.max(Q[next_s]))
            s = next_s
    np.save("lunar_agent", Q)
    with open("lunar_agent_state_mapping.pickle", 'wb') as f:
        pickle.dump(state_mapping, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote to file! Q size {len(Q)} State mapping size: {len(state_mapping)}")
    time.sleep(1)
