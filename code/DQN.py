import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("highway-fast-v0",render_mode='rgb_array')
model = DQN('MlpPolicy',#"MlpPolicy"定义了DQN的策略网络是一个MLP网络
              env,
              policy_kwargs=dict(net_arch=[256, 256]),
            #在stable_baselines3中，DQN的MLP的激活函数默认是Tanh，隐层为两层，每层节点数量为64。通过policy_kwargs参数我们可以自定义DQN的MLP策略网络的结构。"net_arch":[256,256]代表隐层为两层，节点数量为256和256。
              learning_rate=5e-4,
              buffer_size=15000,
              learning_starts=200,
              batch_size=32,
              gamma=0.8,
              train_freq=1,
              gradient_steps=1,
              target_update_interval=50,
              verbose=1,#2打印调试信息
              tensorboard_log="CubicSpline/highway_dqn/")
model.learn(int(1e4))#模拟能够得到的state,action,reward,next state的采样数量，而不是模型的轮数
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
#使用evaluate_policy方法来评估模型，n_eval_episodes代表评估的次数，该值建议为10-30，越高评估结果越可靠。evaluate_policy返回这若干次测试后模型的得分的均值和方差，
env.close()
print(mean_reward)
print(std_reward)
model.save("CubicSpline/highway_dqn/model8_22_2")
# def callback(*params):#回调函数
#     info_dict = params[0]
#     episode_rewards = info_dict['episode_rewards']
#     print(f"episode total reward: {sum(episode_rewards)}")
#
# env = CartPoleBulletEnv(renders=False, discrete_actions=True)
#
# model = DQN(policy="MlpPolicy", env=env)
#
# print("开始训练，稍等片刻")
# model.learn(total_timesteps=100000, callback=callback)
