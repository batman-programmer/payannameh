import os
from dotenv import load_dotenv
import wntr
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# =========================
# بارگذاری مسیر فایل شبکه
# =========================
load_dotenv()
inp_file = os.getenv("INP_FILE")
if not inp_file:
    raise ValueError("EPANET_FILE not set in .env")

wn = wntr.network.WaterNetworkModel(inp_file)

# =========================
# پارامترهای DQN
# =========================
NUM_EPISODES = 500
HORIZON = 24  # ساعت
GAMMA = 0.99
LR = 0.001
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
BATCH_SIZE = 32
MEMORY_SIZE = 10000
ALPHA_PER = 0.6  # اولویت تجربه
BETA_PER = 0.4

# =========================
# محیط ساده شبیه‌سازی
# =========================
prv_list = [v for v in wn.valves() if v[1].valve_type == "PRV"]

def reset_env():
    for valve in prv_list:
        valve[1].initial_setting = 50  # مقدار اولیه
    return get_state()

def step_env(action):
    # action: تغییر فشار هدف هر شیر ±5 متر
    for i, valve in enumerate(prv_list):
        valve[1].initial_setting = max(20, min(100, valve[1].initial_setting + action[i]))
    
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    pressure = results.node['pressure'].loc[0]  # فشار ساعت اول
    
    # Reward: مجموع خطای فاصله از فشار هدف 50 متر
    reward = -np.sum(np.abs(pressure.values - 50))
    
    next_state = get_state()
    done = True  # هر step یک ساعت فرض شده
    return next_state, reward, done, pressure

def get_state():
    pressures = [v[1].initial_setting for v in prv_list]
    return np.array(pressures, dtype=np.float32)

# =========================
# شبکه عصبی DQN
# =========================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# =========================
# حافظه تجربه با PER ساده
# =========================
class PrioritizedReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.priorities = []
    
    def push(self, transition, error):
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
            self.priorities.append((abs(error)+1e-5)**ALPHA_PER)
        else:
            idx = np.argmin(self.priorities)
            self.memory[idx] = transition
            self.priorities[idx] = (abs(error)+1e-5)**ALPHA_PER
    
    def sample(self, batch_size):
        probs = np.array(self.priorities)/sum(self.priorities)
        idx = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[i] for i in idx]
        weights = (len(self.memory)*probs[idx])**(-BETA_PER)
        weights /= weights.max()
        return samples, weights, idx
    
    def update_priorities(self, idx, errors):
        for i, e in zip(idx, errors):
            self.priorities[i] = (abs(e)+1e-5)**ALPHA_PER

# =========================
# راه‌اندازی DQN و حافظه
# =========================
n_actions = len(prv_list)*3  # کاهش/افزایش/بدون تغییر
state_dim = len(prv_list)

policy_net = DQN(state_dim, n_actions)
target_net = DQN(state_dim, n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = PrioritizedReplayMemory(MEMORY_SIZE)
eps = EPS_START

# =========================
# حلقه اصلی شبیه‌سازی
# =========================
print("شروع شبیه‌سازی DRL با DQN + PER...")
for ep in range(NUM_EPISODES):
    state = reset_env()
    for t in range(HORIZON):
        # epsilon-greedy
        if random.random() < eps:
            action = np.random.randint(-5,6,size=len(prv_list))
        else:
            with torch.no_grad():
                q_values = policy_net(torch.FloatTensor(state))
                action = np.array([np.argmax(q_values[i*3:(i+1)*3].numpy())-1 for i in range(len(prv_list))])
        
        next_state, reward, done, pressure = step_env(action)
        error = reward  # ساده برای اولویت
        memory.push((state, action, reward, next_state, done), error)
        
        state = next_state
    
    eps = max(EPS_END, eps*EPS_DECAY)

# =========================
# شبیه‌سازی 24 ساعته با سیاست یادگرفته شده
# =========================
print("\nنتایج شبیه‌سازی DRL (سیاست نهایی):")
state = reset_env()
for t in range(HORIZON):
    with torch.no_grad():
        q_values = policy_net(torch.FloatTensor(state))
        action = np.array([np.argmax(q_values[i*3:(i+1)*3].numpy())-1 for i in range(len(prv_list))])
    next_state, reward, done, pressure = step_env(action)
    
    print(f"ساعت {t+1}:")
    for i, node in enumerate(pressure.index):
        print(f"  نود {node}: {pressure[node]:.2f} متر")
    print("-----------------------------")
    
    state = next_state
