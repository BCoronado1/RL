import gymnasium as gym
from stable_baselines3 import PPO

if __name__ == "__main__":
    episodes = 10
    action_set = {0: "x", 1: "<", 2: "^", 3: ">"}
    environment = gym.make("LunarLander-v2", render_mode="human")
    model = PPO.load("lunar_ppo_model.zip")
    for _ in range(episodes):
        obs, _ = environment.reset()
        terminated = False
        while not terminated:
            a, _states = model.predict(obs, deterministic=False)
            print(action_set[a.tolist()])
            next_obs, r, terminated, truncated, info = environment.step(a)
            obs = next_obs
    environment.close()
