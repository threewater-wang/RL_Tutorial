import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Load and test saved model
env = gym.make("highway-v0",render_mode='rgb_array')
model = DQN.load("CubicSpline/highway_dqn/model8_22_2")
done = truncated = False
env.config["lanes_count"] = 2
obs, info = env.reset()# gym风格的env开头都需要reset一下以获取起点的状态
while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action) # 将动作扔进环境中，从而实现和模拟器的交互
    env.render()#将当前的状态化成一个frame，再将该frame渲染到小窗口上
env.close()