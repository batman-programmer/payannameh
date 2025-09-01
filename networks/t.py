
import wntr
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

# --- تنظیم لاگ ---
tmp_path = "./logs/"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

# --- بارگذاری شبکه ---
inp_file = "network.inp"
wn = wntr.network.WaterNetworkModel(inp_file)

# --- تعریف محیط RL ---
class WaterNetworkEnv(gym.Env):
    def __init__(self, wn):
        super().__init__()
        self.wn = wn
        self.prvs = [link for link in wn.links() if wn.get_link(link).link_type == 'PRV']
        self.action_space = gym.spaces.Box(low=10.0, high=50.0, shape=(len(self.prvs),), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(len(wn.junction_name_list),), dtype=np.float32)
        self.state = None

    def reset(self):
        # تنظیم اولیه PRV ها
        for i, prv_name in enumerate(self.prvs):
            self.wn.get_link(prv_name).setting = 30.0
        # محاسبه فشار اولیه
        sim = wntr.sim.EpanetSimulator(self.wn)
        results = sim.run_sim()
        self.state = results.node['pressure'].iloc[0].values
        return self.state

    def step(self, action):
        # اعمال اقدامات PRV
        for i, prv_name in enumerate(self.prvs):
            self.wn.get_link(prv_name).setting = float(action[i])
        # شبیه سازی شبکه
        sim = wntr.sim.EpanetSimulator(self.wn)
        results = sim.run_sim()
        pressures = results.node['pressure'].iloc[0].values
        self.state = pressures
        # محاسبه پاداش
        reward = -np.sum(np.abs(pressures - 35))  # فشار مطلوب بین 25 تا 40
        done = False
        info = {"mean_pressure": np.mean(pressures), "PRV_change": action}
        return self.state, reward, done, info

# --- ایجاد محیط ---
env = WaterNetworkEnv(wn)

# --- نمونه گیری اولیه ---
obs = env.reset()
print("مشاهده اولیه شبکه:", obs)

# --- تعریف مدل RL ---
model = PPO("MlpPolicy", env, verbose=1)
model.set_logger(new_logger)

# --- آموزش مدل ---
model.learn(total_timesteps=5000)

# --- نمودار زنده ---
plt.ion()
for step in range(10):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    plt.clf()
    plt.plot(obs, label="Pressure at junctions")
    plt.title(f"Step {step} | Reward={reward:.2f}")
    plt.xlabel("Junction index")
    plt.ylabel("Pressure (m)")
    plt.legend()
    plt.pause(0.5)
plt.ioff()
plt.show()
