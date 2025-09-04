import numpy as np
import wntr
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# SumTree for Prioritized Experience Replay
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_pointer = 0
        self.n_entries = 0
    
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        return self.tree[0]
    
    def add(self, priority, data):
        idx = self.write_pointer + self.capacity - 1
        self.data[self.write_pointer] = data
        self.update(idx, priority)
        self.write_pointer = (self.write_pointer + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

# Prioritized Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = 0.01
        self.capacity = capacity
    
    def add(self, error, sample):
        priority = (abs(error) + self.epsilon) ** self.alpha
        self.tree.add(priority, sample)
    
    def sample(self, n, beta=0.4):
        samples = []
        indices = []
        priorities = []
        segment = self.tree.total() / n
        for i in range(n):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            samples.append(data)
            indices.append(idx)
            priorities.append(priority)
        
        priorities = np.array(priorities)
        probs = priorities / self.tree.total()
        weights = (self.tree.n_entries * probs) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights
    
    def update(self, indices, errors):
        for idx, error in zip(indices, errors):
            priority = (abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)

# Reward Function
def reward_function(node_pressures, min_pressure=30.0, max_pressure=100.0, target_pressure=50.0):
    pressures = np.array(node_pressures)
    reward = 0.0
    
    below_min = pressures < min_pressure
    if np.any(below_min):
        reward -= np.sum(min_pressure - pressures[below_min]) * 10.0
    
    above_max = pressures > max_pressure
    if np.any(above_max):
        reward -= np.sum(pressures[above_max] - max_pressure) * 5.0
    
    avg_pressure = np.mean(pressures)
    reward += -np.abs(avg_pressure - target_pressure) * 2.0
    
    if min_pressure <= avg_pressure <= max_pressure:
        reward += (max_pressure - avg_pressure) * 1.0
    
    return reward, avg_pressure  # Return average pressure for plotting

# DQN Neural Network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# EPANET Environment
class WaterNetworkEnv:
    def __init__(self, inp_file):
        self.wn = wntr.network.WaterNetworkModel(inp_file)
        self.sim = wntr.sim.EpanetSimulator(self.wn)
        self.node_list = self.wn.junction_name_list
        self.prv_list = [valve_name for valve_name, valve in self.wn.valves() if valve.valve_type == 'PRV']
        self.action_space = np.arange(30, 101, 10)
        self.state_dim = len(self.node_list)
        self.action_dim = len(self.action_space) * len(self.prv_list)
    
    def reset(self):
        for valve_name in self.prv_list:
            valve = self.wn.get_link(valve_name)
            valve.initial_setting = 50.0
        results = self.sim.run_sim()
        pressures = results.node['pressure'].loc[:, self.node_list].iloc[0].values
        return pressures
    
    def step(self, action):
        for i, valve_name in enumerate(self.prv_list):
            valve = self.wn.get_link(valve_name)
            valve.initial_setting = self.action_space[action // len(self.action_space)]
        results = self.sim.run_sim()
        pressures = results.node['pressure'].loc[:, self.node_list].iloc[0].values
        reward, avg_pressure = reward_function(pressures)
        done = False
        return pressures, reward, done, {'avg_pressure': avg_pressure}

# DQN Agent with PER
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.001
        self.memory = PrioritizedReplayBuffer(capacity=2000, alpha=self.alpha)
        self.batch_size = 32
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.update_target_model()
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        q_value = self.model(state_tensor)[0, action]
        next_q_value = self.target_model(next_state_tensor).max(1)[0]
        target = reward + (1 - done) * self.gamma * next_q_value
        td_error = abs(q_value - target).item()
        self.memory.add(td_error, (state, action, reward, next_state, done))
    
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()
    
    def replay(self):
        if self.memory.tree.n_entries < self.batch_size:
            return
        samples, indices, weights = self.memory.sample(self.batch_size, self.beta)
        states, actions, rewards, next_states, dones = zip(*samples)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        weights = torch.FloatTensor(weights)
        
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        targets = rewards + (1 - dones) * self.gamma * next_q_values
        
        td_errors = q_values - targets.detach()
        loss = (weights * td_errors ** 2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.memory.update(indices, td_errors.detach().numpy())
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Training Loop with Matplotlib Plotting for Rewards and Average Pressure
def train_dqn(inp_file, episodes=1000):
    env = WaterNetworkEnv(inp_file)
    agent = DQNAgent(env.state_dim, env.action_dim)
    rewards = []
    avg_pressures = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        for step in range(100):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward
            if done:
                break
        agent.update_target_model()
        rewards.append(total_reward)
        avg_pressures.append(info['avg_pressure'])
        print(f"اپیزود {episode + 1}/{episodes}، پاداش کل: {total_reward:.2f}، اپسیلون: {agent.epsilon:.3f}، بتا: {agent.beta:.3f}")
    
    # Plot rewards
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, episodes + 1), rewards, label='پاداش کل', color='blue')
    plt.xlabel('اپیزود')
    plt.ylabel('پاداش کل')
    plt.title('روند پاداش‌های آموزش')
    plt.legend()
    plt.grid(True)
    
    # Plot average pressures
    plt.subplot(1, 2, 2)
    plt.plot(range(1, episodes + 1), avg_pressures, label='میانگین فشار', color='green')
    plt.xlabel('اپیزود')
    plt.ylabel('میانگین فشار (متر)')
    plt.title('روند میانگین فشار گره‌ها')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    inp_file = r"networks\wolf3.inp"
    train_dqn(inp_file)