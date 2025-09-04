# PPO for EPANET (continuous actions for PRVs)
# Dependencies: wntr, torch, numpy, matplotlib
# Usage: set inp_file path and run. Uses GPU if available.

import numpy as np
import wntr
import torch
import torch.nn as nn
import torch.optim as optim
import random
import itertools
import matplotlib.pyplot as plt
from collections import deque, namedtuple

# ------------------------------
# Environment wrapper for EPANET
# ------------------------------
class WaterNetworkEnv:
    def __init__(self, inp_file, action_low=30.0, action_high=100.0):
        self.wn = wntr.network.WaterNetworkModel(inp_file)
        self.sim = wntr.sim.EpanetSimulator(self.wn)
        self.node_list = self.wn.junction_name_list
        # list of PRVs (may be empty)
        self.prv_list = [valve_name for valve_name, valve in self.wn.valves() if valve.valve_type == 'PRV']
        self.n_prv = len(self.prv_list)
        self.state_dim = len(self.node_list)
        # continuous action per PRV in [action_low, action_high]
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        if self.n_prv == 0:
            # still allow a dummy action (no-op) to keep interfaces consistent
            self.action_dim = 0
        else:
            self.action_dim = self.n_prv

    def reset(self):
        # set a neutral starting setpoint, run sim, return pressures vector
        for valve_name in self.prv_list:
            valve = self.wn.get_link(valve_name)
            valve.initial_setting = float((self.action_low + self.action_high) / 2.0)
        results = self.sim.run_sim()
        pressures = results.node['pressure'].loc[:, self.node_list].iloc[0].values.astype(np.float32)
        return pressures

    def step(self, action):
        """
        action: numpy array shape (n_prv,) with continuous values in [action_low, action_high]
        if n_prv == 0, action can be None or empty and env is simulated with defaults.
        """
        if self.n_prv > 0:
            # clip to valid range and set PRVs
            action = np.clip(action, self.action_low, self.action_high)
            for valve_name, setting in zip(self.prv_list, action):
                valve = self.wn.get_link(valve_name)
                valve.initial_setting = float(setting)
        # run simulation (single timestep / hydraulic steady-state)
        results = self.sim.run_sim()
        pressures = results.node['pressure'].loc[:, self.node_list].iloc[0].values.astype(np.float32)
        reward, avg_pressure = self._reward_function(pressures)
        done = False  # you may customize terminal conditions if desired
        info = {'avg_pressure': float(avg_pressure)}
        return pressures, float(reward), done, info

    def _reward_function(self, node_pressures, target_pressure=50.0, clip_min=-100.0, clip_max=100.0):
        pressures = np.array(node_pressures, dtype=np.float32)
        # mean absolute deviation from target
        mad = float(np.mean(np.abs(pressures - target_pressure)))
        # base reward: higher if closer to target
        reward = 100.0 - mad
        # additional small penalty for large variance to prefer uniform pressures
        reward -= 0.1 * float(np.std(pressures))
        reward = float(np.clip(reward, clip_min, clip_max))
        avg_pressure = float(np.mean(pressures))
        return reward, avg_pressure

# ------------------------------
# PPO Actor-Critic networks
# ------------------------------
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0.0)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(128, 64)):
        super().__init__()
        if action_dim == 0:
            # dummy actor (no PRVs)
            self.net = nn.Sequential(nn.Linear(state_dim, 1))  # unused
            self.log_std = nn.Parameter(torch.tensor([]))
            return
        layers = []
        dims = [state_dim] + list(hidden_sizes)
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], action_dim))  # outputs mean for each action dim
        self.net = nn.Sequential(*layers)
        # learnable log_std (one per action dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)  # initial stdev ~ exp(-0.5)=0.61
        self.apply(init_weights)

    def forward(self, x):
        if self.log_std.numel() == 0:
            # no actions
            mean = self.net(x) * 0.0
            std = torch.zeros_like(mean)
            return mean, std
        mean = self.net(x)
        std = torch.exp(self.log_std)
        # expand std to batch dimension
        std = std.unsqueeze(0).expand_as(mean)
        return mean, std

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_sizes=(128, 64)):
        super().__init__()
        layers = []
        dims = [state_dim] + list(hidden_sizes)
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ------------------------------
# PPO buffer (on-policy)
# ------------------------------
Transition = namedtuple('Transition', ['state', 'action', 'logp', 'reward', 'done', 'value'])

class RolloutBuffer:
    def __init__(self):
        self.storage = []

    def add(self, *args):
        self.storage.append(Transition(*args))

    def clear(self):
        self.storage = []

    def get_batches(self, advantages, batch_size):
        # returns indices shuffled for minibatch updates
        n = len(self.storage)
        idxs = np.arange(n)
        np.random.shuffle(idxs)
        for start in range(0, n, batch_size):
            batch_idx = idxs[start:start + batch_size]
            yield batch_idx

# ------------------------------
# PPO Agent
# ------------------------------
class PPOAgent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_low=30.0,
                 action_high=100.0,
                 device=None,
                 gamma=0.99,
                 lam=0.95,
                 clip_eps=0.2,
                 lr=3e-4,
                 epochs=10,
                 minibatch_size=64,
                 value_coef=0.5,
                 entropy_coef=0.01):
        self.device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        print(f"[PPO] using device: {self.device}")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = float(action_low)
        self.action_high = float(action_high)

        # actor-critic
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)

        # optimizers
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

        # hyperparams
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # buffer
        self.buffer = RolloutBuffer()

    def select_action(self, state_np):
        # state_np: numpy array (state_dim,)
        state = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mean, std = self.actor(state)
            if self.action_dim == 0:
                # no actions -> return empty
                return np.array([]), 0.0, 0.0
            dist = torch.distributions.Normal(mean, std)
            raw_action = dist.sample()
            logp = dist.log_prob(raw_action).sum(dim=-1)
            value = self.critic(state)
        # map raw_action (unbounded) through tanh to [-1,1], then scale to [low,high]
        tanh_action = torch.tanh(raw_action)
        action_scaled = (tanh_action + 1.0) / 2.0 * (self.action_high - self.action_low) + self.action_low
        return action_scaled.cpu().numpy().flatten(), float(logp.cpu().numpy().item()), float(value.cpu().numpy().item())

    def compute_gae(self, last_value=0.0):
        # compute advantages and returns from buffer transitions
        rewards = [t.reward for t in self.buffer.storage]
        values = [t.value for t in self.buffer.storage]
        dones = [t.done for t in self.buffer.storage]
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_adv = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[t])
            next_value = last_value if t == len(rewards)-1 else values[t+1]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            last_adv = delta + self.gamma * self.lam * mask * last_adv
            advantages[t] = last_adv
        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    def update(self):
        if len(self.buffer.storage) == 0:
            return

        # compute advantages & returns (last_value=0 for episode-terminated/on-policy)
        advantages, returns = self.compute_gae(last_value=0.0)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # prepare tensors
        states = torch.tensor(np.vstack([t.state for t in self.buffer.storage]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.vstack([ (t.action if t.action is not None and len(t.action)>0 else np.zeros((self.action_dim,))) for t in self.buffer.storage ]),
                               dtype=torch.float32, device=self.device)
        old_logps = torch.tensor(np.array([t.logp for t in self.buffer.storage], dtype=np.float32), dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advs_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        n_steps = len(self.buffer.storage)
        batch_size = self.minibatch_size

        for _ in range(self.epochs):
            # minibatch gradient steps
            for batch_idx in range(0, n_steps, batch_size):
                idxs = np.arange(batch_idx, min(batch_idx + batch_size, n_steps))
                b_states = states[idxs]
                b_actions = actions[idxs]
                b_old_logps = old_logps[idxs]
                b_returns = returns_t[idxs]
                b_advs = advs_t[idxs]

                # recompute distribution and log probs
                mean, std = self.actor(b_states)
                if self.action_dim == 0:
                    # nothing to update for actions
                    new_logps = torch.zeros_like(b_old_logps)
                    entropy = torch.zeros_like(new_logps).mean()
                else:
                    # to compute log_prob, we need raw_action such that after tanh and scaling we got b_actions
                    # simpler approach: sample raw from Normal(mean,std), but we need log_prob of the taken action.
                    # We can invert the scaling: convert b_actions back to tanh-space then to raw pre-tanh using atanh
                    # numerical stability: clamp to (-1+eps, 1-eps)
                    eps = 1e-6
                    # map actions to tanh_action space: tanh_action = (2*(a - low)/(high - low) - 1)
                    tanh_a = 2.0 * (b_actions - self.action_low) / (self.action_high - self.action_low) - 1.0
                    tanh_a = torch.clamp(tanh_a, -1 + eps, 1 - eps)
                    # atanh to get pre-tanh raw action
                    raw_action = 0.5 * (torch.log1p(tanh_a) - torch.log1p(-tanh_a))
                    # compute log_prob under Normal(mean,std)
                    dist = torch.distributions.Normal(mean, std)
                    # log_prob of raw action (sum across action dims)
                    logp_raw = dist.log_prob(raw_action).sum(dim=-1)
                    # correction for tanh squash (change of variables)
                    # see appendix of SAC/PPO with tanh: logp = logp_raw - sum(log(1 - tanh(a)^2) + eps)
                    log_det = torch.sum(torch.log(1.0 - torch.tanh(raw_action) ** 2 + 1e-6), dim=-1)
                    new_logps = logp_raw - log_det
                    entropy = dist.entropy().sum(dim=-1).mean()

                # ratio for PPO
                ratio = torch.exp(new_logps - b_old_logps)
                surr1 = ratio * b_advs
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advs
                policy_loss = -torch.min(surr1, surr2).mean()

                # value loss
                values_pred = self.critic(b_states)
                value_loss = nn.MSELoss()(values_pred, b_returns)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), max_norm=0.5)
                self.optimizer.step()

        # clear buffer after update
        self.buffer.clear()

# ------------------------------
# Training loop (collect-on-policy rollouts)
# ------------------------------
def train_ppo(inp_file,
              total_timesteps=200000,
              timesteps_per_batch=2048,
              max_episode_steps=200,
              update_epochs=10,
              minibatch_size=64,
              render=False):
    env = WaterNetworkEnv(inp_file)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    agent = PPOAgent(state_dim=env.state_dim,
                     action_dim=env.action_dim,
                     action_low=env.action_low,
                     action_high=env.action_high,
                     device=device,
                     epochs=update_epochs,
                     minibatch_size=minibatch_size)

    timesteps_collected = 0
    ep_rewards = []
    ep_pressures = []

    while timesteps_collected < total_timesteps:
        # collect rollout
        steps = 0
        while steps < timesteps_per_batch:
            state = env.reset()
            ep_reward = 0.0
            for ep_step in range(max_episode_steps):
                # select action
                action, logp, value = agent.select_action(state)
                # action may be empty if no PRVs
                next_state, reward, done, info = env.step(action if env.action_dim > 0 else None)
                # store transition
                agent.buffer.add(state, action if env.action_dim>0 else np.array([]), logp, reward, done, value)
                state = next_state
                ep_reward += reward
                steps += 1
                timesteps_collected += 1
                if steps >= timesteps_per_batch:
                    break
                if done:
                    break
            ep_rewards.append(ep_reward)
            ep_pressures.append(info.get('avg_pressure', 0.0))

        # update policy (on collected batch)
        agent.update()

        # logging
        avg_r = np.mean(ep_rewards[-10:]) if len(ep_rewards) >= 1 else 0.0
        avg_p = np.mean(ep_pressures[-10:]) if len(ep_pressures) >= 1 else 0.0
        print(f"[timesteps {timesteps_collected}/{total_timesteps}] recent_avg_reward={avg_r:.2f}, recent_avg_pressure={avg_p:.2f}")

    # final plots
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(ep_rewards, label='episode_rewards')
    plt.xlabel('episode')
    plt.ylabel('total reward')
    plt.title('Episode Rewards')
    plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(ep_pressures, label='avg_pressure')
    plt.xlabel('episode')
    plt.ylabel('avg pressure (m)')
    plt.title('Average Node Pressure')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------------------
# Run training
# ------------------------------
if __name__ == "__main__":
    inp_file = "/content/2PRV.inp"   # <- مسیر فایل .inp خودت را اینجا قرار بده
    # مثال پارامترها (می‌تونی تغییر بدی)
    train_ppo(inp_file,
              total_timesteps=50000,      # تعداد کل استپ‌های محیط برای آموزش
              timesteps_per_batch=2048,   # تعداد استپ‌هایی که پیش از یک به‌روزرسانی جمع می‌شوند
              max_episode_steps=100,      # حداکثر طول هر اپیزود
              update_epochs=10,
              minibatch_size=64)
