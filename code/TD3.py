import gymnasium as gym
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("highway-fast-v0", render_mode="rgb_array")
env.configure({
    "action": {
        "type": "ContinuousAction"#将其变成连续空间
    }
})
env.reset()
print("Action space shape:", env.action_space.shape)
# The noise objects for TD3
n_actions = env.action_space.shape[-1]#从动作空间的形状中获取了连续动作空间的维度，即向量的长度
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3("MlpPolicy", env,
            action_noise=action_noise,
            verbose=1,
            tensorboard_log="highway_TD3/")
model.learn(total_timesteps=10000, log_interval=10)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
#使用evaluate_policy方法来评估模型，n_eval_episodes代表评估的次数，该值建议为10-30，越高评估结果越可靠。evaluate_policy返回这若干次测试后模型的得分的均值和方差，
env.close()
print(mean_reward)
print(std_reward)
model.save("highway_TD3/td3_pendulum")