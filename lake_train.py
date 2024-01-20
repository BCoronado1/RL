import random

import gymnasium as gym
import numpy as np

if __name__ == "__main__":
    environment = gym.make("FrozenLake-v1", is_slippery=False)
    Q = np.random.random((environment.observation_space.n, environment.action_space.n))
    episodes = 100000  # Total number of episodes
    alpha = 0.9  # Learning rate
    gamma = 0.9  # Discount factor
    epsilon = 0.25  # Exploration vs Exploitation
    for ep_num in range(episodes):
        if ep_num % 1000 == 0:
            print(f"Episode {ep_num}")
        s, _ = environment.reset()
        terminated = False
        while not terminated:
            a = environment.action_space.sample() if random.uniform(0, 1) <= epsilon else np.argmax(Q[s])
            next_s, r, terminated, truncated, info = environment.step(a)
            Q[s, a] = (1 - alpha) * Q[s, a] + alpha * (r + gamma * np.max(Q[next_s]))
            s = next_s
    np.save("lake_agent", Q)
