from gymnasium.envs.registration import register

register(
     id="nao",
     entry_point="humanoid_envs.envs:NaoEnv",
     #max_episode_steps=300,
)