import gymnasium as gym
from stable_baselines3 import PPO

if __name__ == "__main__":
    time_steps = int(1e6)
    environment = gym.make("LunarLander-v2")
    model = PPO("MlpPolicy", environment, verbose=1)
    model.learn(total_timesteps=time_steps)
    environment.close()
    model.save("lunar_ppo_model.zip")
