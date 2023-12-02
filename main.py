import humanoid_envs
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from callbacks import TensorboardCallback
import os
import torch
import time


def start_cuda():
    device = 0
    torch.cuda.set_device(device)
    while not torch.cuda.is_available():
        device += 1
        print(f"Device num: {torch.cuda.current_device()}")
        torch.cuda.set_device(device)

    torch.cuda.empty_cache()


def log_name():
    i = 1
    while i < 1000:
        path_name = f"logs/exp{i}"
        if not os.path.exists(path_name):
            print(f"---------------Logging to {path_name}")
            return path_name
        i += 1
    print("TOO MANY LOGS!")


start_cuda()
training = True

if training:
    env = gym.make("nao", gui=False, action_space_size=2, obs_type="angles", reward_type="delta_inv_timestep")
    modelSAC = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        buffer_size=int(1e6),
        learning_rate=1e-4,
        gamma=0.999,
        tau=0.005,
        policy_kwargs=dict(net_arch=dict(pi=[512, 512], qf=[512, 512])),
        tensorboard_log="./tensorboard/"
    )

    modelSAC.set_env(env)
    checkpoint = CheckpointCallback(5000, log_name())
    modelSAC.learn(total_timesteps=5000000, log_interval=16, callback=[checkpoint, TensorboardCallback()])
else:
    env = gym.make("nao", gui=False, action_space_size=2, obs_type="xyz", reward_type="delta")
    model = SAC.load("logs/exp84/rl_model_105000_steps.zip")
    model.set_env(env)
    vec_env = model.get_env()
    obs = vec_env.reset()
    total_reward = 0
    t0 = time.time()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        total_reward += reward
        if done:
            obs = env.reset()
            break
    t1 = time.time()
    print(f"Total reward: {total_reward}. Time: {round(t1 - t0,3)}")

env.close()
