# bdq_lightning_fixed_full.py
import os
import random
import time
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wntr
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# ---------------------------
# SumTree + PER
# ---------------------------
class SumTree:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float32)
        self.data = np.zeros(self.capacity, dtype=object)
        self.write = 0
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
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
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

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = 1e-6

    def add(self, error, sample):
        p = (abs(error) + self.epsilon) ** self.alpha
        self.tree.add(p, sample)

    def sample(self, n, beta=0.4):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)
        priorities = np.array(priorities, dtype=np.float32)
        probs = priorities / (self.tree.total() + 1e-8)
        weights = (self.tree.n_entries * probs) ** (-beta)
        weights = weights / (weights.max() + 1e-8)
        return batch, idxs, weights

    def update(self, idxs, errors):
        for idx, err in zip(idxs, errors):
            p = (abs(err) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries

# ---------------------------
# Environment wrapper (EPANET)
# ---------------------------
class WaterEnvEPANET:
    def __init__(self, inp_file, min_pressure=40.0, max_pressure=60.0):
        self.inp_file = inp_file
        self.wn = wntr.network.WaterNetworkModel(inp_file)
        self.sim = wntr.sim.EpanetSimulator(self.wn)
        self.node_list = self.wn.junction_name_list
        self.prv_list = [name for name, v in self.wn.valves() if v.valve_type == 'PRV']
        self.n_prv = len(self.prv_list)
        self.state_dim = len(self.node_list)
        self.min_pressure = float(min_pressure)
        self.max_pressure = float(max_pressure)
        self.action_bins = None
        self.last_avg_pressure = None

    def set_action_bins(self, bins):
        self.action_bins = np.array(bins, dtype=np.float32)

    def reset(self):
        self.wn = wntr.network.WaterNetworkModel(self.inp_file)
        self.sim = wntr.sim.EpanetSimulator(self.wn)
        default = float(np.mean(self.action_bins)) if self.action_bins is not None else (self.min_pressure + self.max_pressure) / 2.0
        for valve_name in self.prv_list:
            valve = self.wn.get_link(valve_name)
            valve.initial_setting = float(default)
        results = self.sim.run_sim()
        pressures = results.node['pressure'].loc[:, self.node_list].iloc[0].values.astype(np.float32)
        self.last_avg_pressure = float(np.mean(pressures))
        return pressures

    def step(self, action_idxs):
        if self.n_prv > 0:
            if self.action_bins is None:
                raise RuntimeError("action_bins not set")
            for valve_name, idx in zip(self.prv_list, action_idxs):
                valve = self.wn.get_link(valve_name)
                valve.initial_setting = float(self.action_bins[int(idx)])
        results = self.sim.run_sim()
        pressures = results.node['pressure'].loc[:, self.node_list].iloc[0].values.astype(np.float32)
        avg_p = float(np.mean(pressures))
        self.last_avg_pressure = avg_p
        reward, done = self._reward(pressures)
        info = {'avg_pressure': avg_p}
        return pressures, float(reward), bool(done), info

    def _reward(self, pressures):
        p = np.array(pressures, dtype=np.float32)
        in_range = np.logical_and(p >= self.min_pressure, p <= self.max_pressure)
        count_in = np.sum(in_range)
        reward = float(count_in) * 10.0
        under = p[p < self.min_pressure]
        over = p[p > self.max_pressure]
        under_pen = float(np.sum(self.min_pressure - under)) if under.size > 0 else 0.0
        over_pen = float(np.sum(over - self.max_pressure)) if over.size > 0 else 0.0
        reward -= 2.0 * under_pen + 1.0 * over_pen
        reward -= 0.1 * float(np.std(p))
        reward /= max(1.0, float(len(p)))
        reward = float(np.clip(reward, -100.0, 100.0))
        done = False
        return reward, done

# ---------------------------
# BDQ Network
# ---------------------------
def orthogonal_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class BDQNetwork(nn.Module):
    def __init__(self, state_dim, n_branches, n_bins, hidden_sizes=(256,128)):
        super().__init__()
        last = state_dim
        trunk_layers = []
        for h in hidden_sizes:
            trunk_layers.append(nn.Linear(last, h))
            trunk_layers.append(nn.ReLU())
            last = h
        self.trunk = nn.Sequential(*trunk_layers)
        self.value_head = nn.Sequential(
            nn.Linear(last, last//2),
            nn.ReLU(),
            nn.Linear(last//2, 1)
        )
        self.adv_heads = nn.ModuleList()
        for _ in range(n_branches):
            self.adv_heads.append(nn.Sequential(
                nn.Linear(last, last//2),
                nn.ReLU(),
                nn.Linear(last//2, n_bins)
            ))
        self.apply(orthogonal_init)

    def forward(self, x):
        feat = self.trunk(x)
        value = self.value_head(feat).squeeze(-1)
        advs = []
        for branch in self.adv_heads:
            adv = branch(feat)
            advs.append(adv.unsqueeze(1))
        advantages = torch.cat(advs, dim=1)
        return value, advantages

# ---------------------------
# Lightning Module
# ---------------------------
Transition = namedtuple('Transition', ('state','action_idxs','reward','next_state','done'))

class BDQLightning(pl.LightningModule):
    def __init__(self,
                 inp_file,
                 n_bins=8,
                 min_pressure=40.0,
                 max_pressure=60.0,
                 lr=1e-4,
                 gamma=0.99,
                 batch_size=64,
                 replay_capacity=200000,
                 target_update_freq=2000,
                 eps_start=1.0,
                 eps_final=0.05,
                 eps_decay=200000,
                 alpha_per=0.6,
                 beta_start=0.4,
                 beta_frames=200000,
                 episode_len=24):
        super().__init__()
        self.save_hyperparameters()
        self._manual_epoch = 0

        self.env = WaterEnvEPANET(inp_file, min_pressure=min_pressure, max_pressure=max_pressure)
        self.action_bins = np.linspace(30.0, 100.0, num=n_bins)
        self.env.set_action_bins(self.action_bins)
        self.n_branches = max(1, self.env.n_prv)
        self.state_dim = self.env.state_dim
        self.n_bins = n_bins

        self.policy_net = BDQNetwork(self.state_dim, self.n_branches, n_bins)
        self.target_net = BDQNetwork(self.state_dim, self.n_branches, n_bins)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.replay = PrioritizedReplayBuffer(replay_capacity, alpha=alpha_per)
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.target_update_freq = target_update_freq

        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_decay = eps_decay
        self.frame_idx = 0

        self.beta_start = beta_start
        self.beta_frames = beta_frames

        self.learn_steps = 0
        self.episode_len = episode_len

        self.writer = SummaryWriter(log_dir=os.path.join("tb_logs", time.strftime("%Y%m%d-%H%M%S")))
        self.device_ = None

    def configure_optimizers(self):
        return optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def epsilon(self, frame):
        if frame >= self.eps_decay:
            return self.eps_final
        return self.eps_final + (self.eps_start - self.eps_final) * (1.0 - frame / float(self.eps_decay))

    def beta_by_frame(self, frame):
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * frame / float(self.beta_frames))

    def select_action(self, state_np, eps=None):
        state = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(self.device_)
        with torch.no_grad():
            val, adv = self.policy_net(state)
            val_exp = val.unsqueeze(1).expand(-1, self.n_branches).unsqueeze(-1)
            q = val_exp + (adv - adv.mean(dim=-1, keepdim=True))
            q = q.squeeze(0).cpu().numpy()
        if eps is None:
            eps = self.epsilon(self.frame_idx)
        action_idxs = []
        for b in range(self.n_branches):
            if random.random() < eps:
                action_idxs.append(random.randrange(self.n_bins))
            else:
                action_idxs.append(int(np.argmax(q[b])))
        return action_idxs

    def store_transition(self, state, action_idxs, reward, next_state, done):
        td_err = abs(reward) + 1e-3
        self.replay.add(td_err, (state.copy(), np.array(action_idxs, dtype=np.int64), float(reward), next_state.copy(), bool(done)))

    def learn_step(self):
        if len(self.replay) < max(1000, self.batch_size):
            return None

        beta = self.beta_by_frame(self.frame_idx)
        batch, idxs, weights = self.replay.sample(self.batch_size, beta=beta)
        states = np.vstack([b[0] for b in batch]).astype(np.float32)
        actions = np.vstack([b[1] for b in batch]).astype(np.int64)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.vstack([b[3] for b in batch]).astype(np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device_)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device_)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device_)

        # current Q-values
        vals, advs = self.policy_net(states_t)
        vals_exp = vals.unsqueeze(1).expand(-1, self.n_branches).unsqueeze(-1)
        q_all = vals_exp + (advs - advs.mean(dim=-1, keepdim=True))
        batch_size, n_branches, n_bins = q_all.shape
        actions_flat = actions.reshape(-1)
        q_all_flat = q_all.view(batch_size * n_branches, n_bins)
        q_taken = q_all_flat[torch.arange(batch_size * n_branches, device=self.device_), actions_flat]
        q_taken = q_taken.view(batch_size, n_branches).sum(dim=1)

        # Double DQN target
        with torch.no_grad():
            next_vals_policy, next_advs_policy = self.policy_net(next_states_t)
            next_vals_policy_exp = next_vals_policy.unsqueeze(1).expand(-1, self.n_branches).unsqueeze(-1)
            next_q_policy = next_vals_policy_exp + (next_advs_policy - next_advs_policy.mean(dim=-1, keepdim=True))
            next_actions = next_q_policy.argmax(dim=-1)

            next_vals_target, next_advs_target = self.target_net(next_states_t)
            next_vals_target_exp = next_vals_target.unsqueeze(1).expand(-1, self.n_branches).unsqueeze(-1)
            next_q_target = next_vals_target_exp + (next_advs_target - next_advs_target.mean(dim=-1, keepdim=True))
            next_q_target_flat = next_q_target.view(batch_size * n_branches, n_bins)
            next_actions_flat = next_actions.view(-1)
            next_q_sel = next_q_target_flat[torch.arange(batch_size * n_branches, device=self.device_), next_actions_flat]
            next_q_sel = next_q_sel.view(batch_size, n_branches).sum(dim=1)

            td_target = torch.tensor(rewards, dtype=torch.float32, device=self.device_) + \
                        (1.0 - torch.tensor(dones, dtype=torch.float32, device=self.device_)) * (self.gamma * next_q_sel)

        td_error = td_target - q_taken
        loss = (weights_t * td_error ** 2).mean()

        opt = self.optimizers()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        opt.step()

        abs_err = td_error.detach().abs().cpu().numpy()
        self.replay.update(idxs, abs_err.tolist())

        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item(), float(abs_err.mean())

    # -------------------------
    # Manual training loop
    # -------------------------
    def on_train_start(self):
        self.device_ = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.to(self.device_)
        print(f"[BDQLightning] training on device: {self.device_}")

    def train_during_epoch(self, frames_per_epoch=2048):
        frames = 0
        epoch_reward = 0.0
        while frames < frames_per_epoch:
            state = self.env.reset()
            ep_reward = 0.0
            for t in range(self.episode_len):
                action_idxs = self.select_action(state)
                next_state, reward, done, info = self.env.step(action_idxs)
                self.store_transition(state, action_idxs, reward, next_state, done)
                res = self.learn_step()
                state = next_state
                ep_reward += reward
                self.frame_idx += 1
                frames += 1
                if res is not None:
                    loss_val, td = res
                    if self.frame_idx % 500 == 0:
                        self.writer.add_scalar('train/step_loss', loss_val, self.frame_idx)
                        self.writer.add_scalar('train/step_td', td, self.frame_idx)
                if frames >= frames_per_epoch:
                    break
            epoch_reward += ep_reward
        avg_reward = epoch_reward / max(1, frames // self.episode_len)
        self.writer.add_scalar('train/epoch_reward', avg_reward, self._manual_epoch)
        return avg_reward

    def evaluate_policy(self, n_episodes=5):
        results = []
        for ep in range(n_episodes):
            state = self.env.reset()
            ep_reward = 0.0
            pressures_list = []
            for t in range(self.episode_len):
                action_idxs = self.select_action(state, eps=0.0)
                next_state, reward, done, info = self.env.step(action_idxs)
                ep_reward += reward
                pressures_list.append(info.get('avg_pressure', 0.0))
                state = next_state
            avg_p = np.mean(pressures_list)
            results.append({'episode': ep, 'total_reward': ep_reward, 'avg_pressure': avg_p})
        mean_reward = np.mean([r['total_reward'] for r in results])
        mean_p = np.mean([r['avg_pressure'] for r in results])
        self.writer.add_scalar('eval/mean_reward', mean_reward, self._manual_epoch)
        self.writer.add_scalar('eval/mean_pressure', mean_p, self._manual_epoch)
        return results

# ---------------------------
# Orchestration
# ---------------------------
def train_bdq_lightning(inp_file,
                        n_bins=8,
                        frames_per_epoch=2048,
                        episode_len=24,
                        batch_size=64,
                        replay_capacity=200000,
                        lr=1e-4,
                        gpus=1,
                        max_epochs=50,
                        save_dir='bdq_lightning_ckpt'):
    pl.seed_everything(42)
    os.makedirs(save_dir, exist_ok=True)

    model = BDQLightning(inp_file=inp_file,
                         n_bins=n_bins,
                         lr=lr,
                         batch_size=batch_size,
                         replay_capacity=replay_capacity,
                         episode_len=episode_len)

    ckpt_cb = ModelCheckpoint(dirpath=save_dir, filename='bdq-{epoch:02d}-{step}', save_top_k=-1, every_n_epochs=1)
    lrmon = LearningRateMonitor(logging_interval='step')

    if torch.cuda.is_available() and gpus > 0:
        trainer = pl.Trainer(accelerator='gpu', devices=gpus, max_epochs=max_epochs,
                             callbacks=[ckpt_cb, lrmon], log_every_n_steps=50,
                             default_root_dir=save_dir)
    else:
        trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=max_epochs,
                             callbacks=[ckpt_cb, lrmon], log_every_n_steps=50,
                             default_root_dir=save_dir)

    model.on_train_start()
    for ep in range(max_epochs):
        print(f"=== Starting epoch {ep+1}/{max_epochs} ===")
        model._manual_epoch = ep
        model.train()
        avg_reward = model.train_during_epoch(frames_per_epoch)
        eval_res = model.evaluate_policy(n_episodes=5)
        print(f"[Epoch {ep+1}] Avg Reward={avg_reward:.2f}, Eval mean_reward={np.mean([r['total_reward'] for r in eval_res]):.2f}, mean_pressure={np.mean([r['avg_pressure'] for r in eval_res]):.2f}")

        ckpt_path = os.path.join(save_dir, f'ckpt_epoch{ep+1}.pth')
        torch.save({
            'policy_state': model.policy_net.state_dict(),
            'target_state': model.target_net.state_dict(),
            'frame_idx': model.frame_idx
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    final_policy = os.path.join(save_dir, 'bdq_policy_final.pth')
    torch.save(model.policy_net.state_dict(), final_policy)
    print(f"Saved final policy: {final_policy}")

    return model

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    inp_file = r"/content/2PRV.inp"
    model = train_bdq_lightning(inp_file,
                                n_bins=8,
                                frames_per_epoch=2048,
                                episode_len=24,
                                batch_size=64,
                                replay_capacity=200000,
                                lr=1e-4,
                                gpus=1,
                                max_epochs=30,
                                save_dir='bdq_lightning_ckpt')
