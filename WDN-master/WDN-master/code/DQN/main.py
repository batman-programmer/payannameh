import itertools
import os
import random
from pathlib import Path
from typing import Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers
from tensorflow import keras
from dotenv import load_dotenv
import wntr
from wntr.network import Valve, ControlAction, controls, Junction, TimeSeries

# -------------------------
# Constants
# -------------------------
DEFAULT_ACTION_ZONE = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240)
REWARD_FUNCTION_1 = "just_average"
REWARD_FUNCTION_2 = "without_negative"

# -------------------------
# Environment
# -------------------------
class WaterNetworkEnv(gym.Env):
    MIN_PRESSURE = 40
    MID1_PRESSURE = 50
    MID2_PRESSURE = 67
    MAX_PRESSURE = 70
    PSI_UNIT = 1.4503773800722
    NEGATIVE_REWARD = -1000

    def __init__(self, inp_file, seed=42, bins=3, action_zone=(20, 30, 40, 50), do_log=True, node_demand_random_failure=0.3, reward_function_type=REWARD_FUNCTION_1):
        super(WaterNetworkEnv, self).__init__()
        self.seed = seed
        self.do_log = do_log
        self.reward_function_type = reward_function_type
        self.wn = wntr.network.WaterNetworkModel(inp_file)
        self.num_nodes = self.wn.num_nodes
        self.num_valves = self.wn.num_valves
        self.action_space = gym.spaces.Discrete(len(list(itertools.product(action_zone, repeat=self.num_valves))))
        self.actions_index = {i: item for i, item in enumerate(list(itertools.product(action_zone, repeat=self.num_valves)))}
        self.node_demand_random_failure = node_demand_random_failure
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_nodes,), dtype=np.float32, seed=seed)
        self.state = np.zeros(self.num_nodes)
        self.time = 0
        self.wn.options.time.duration = 3600 * 24
        self.wn.options.time.hydraulic_timestep = 3600
        self.sim = wntr.sim.WNTRSimulator(self.wn)

    def _change_valve_setting(self, setting, valve: Valve):
        name = valve.name
        control_name = f"valve_{name}_control"
        act = ControlAction(valve, 'setting', setting)
        cond = controls.SimTimeCondition(self.wn, '=', int(self.wn.sim_time))
        c = controls.Control(cond, act, name=control_name)
        try:
            self.wn.remove_control(control_name)
        except KeyError:
            pass
        self.wn.add_control(control_name, c)

    def perform_random_failure(self):
        prefix = "random_demand|id="
        for junc_id, junction in self.wn.junctions():
            junction: Junction
            for i, d_timeseries in enumerate(junction.demand_timeseries_list):
                d_timeseries: TimeSeries
                if d_timeseries.pattern_name.startswith(prefix):
                    del junction.demand_timeseries_list[i]
            if random.random() < self.node_demand_random_failure:
                base_demand = junction.base_demand or 0
                random_factor = random.random() * 20
                if self.do_log:
                    print(f"junction={junc_id}, random_factor={random_factor}, base_demand*factor={base_demand*random_factor}")
                junction.add_demand(base_demand * random_factor, f"{prefix}{junc_id}")

    def step(self, action_index: int):
        actions = self.actions_index[action_index]
        for i, valve in enumerate(self.wn.valves()):
            self._change_valve_setting(int(actions[i]), valve[1])
        self.time = self.wn.sim_time
        results = self.sim.run_sim()
        if len(results.node["pressure"].values) == 0:
            reward = 0
            done = False
            state = self.state
        else:
            state = results.node["pressure"].values[-1] * self.PSI_UNIT / 100
            reward = self._calculate_reward()
            done = False

        if self.do_log:
            print(f"STEP {self.time}: Reward={reward:.2f}, Done={done}, Pressure={state}, Actions={[self.PSI_UNIT*a for a in actions]}")

        self.state = state
        return state, reward, done, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.wn.reset_initial_values()
        self.time = 0
        self.state = np.zeros(self.num_nodes)
        return self.state

    def _pressure_cost(self, pressure: float):
        if self.MIN_PRESSURE <= pressure < self.MID1_PRESSURE:
            return 6 - (pressure / 20)
        elif self.MID1_PRESSURE <= pressure < self.MID2_PRESSURE:
            return 1
        elif self.MID2_PRESSURE <= pressure < self.MAX_PRESSURE:
            return 14 - (pressure / 5)
        else:
            return 0

    def _calculate_reward(self):
        if self.reward_function_type == REWARD_FUNCTION_2:
            return self.reward_function_without_negative_pressure()
        return self.reward_function_1()

    def reward_function_1(self):
        summation = 0
        for node in self.wn.nodes:
            node_pressure = self.wn.get_node(node).pressure * self.PSI_UNIT
            summation += node_pressure * self._pressure_cost(node_pressure)
        return summation / self.num_nodes

    def reward_function_without_negative_pressure(self):
        summation = 0
        for node in self.wn.nodes:
            node_pressure = self.wn.get_node(node).pressure * self.PSI_UNIT
            if node_pressure < 0:
                summation += self.NEGATIVE_REWARD
            else:
                summation += node_pressure * self._pressure_cost(node_pressure)
        return summation / self.num_nodes


# -------------------------
# DQN Agent
# -------------------------
class DQN_WaterNetwork:
    INDEX_COLORS = {1: "b", 2: "r", 3: "k", 4: "c", 5: "m", 6: "y", 7: "k", 8: "w"}

    def __init__(self, network_file, action_zone, seed=42, epsilon_greedy_frames=20000, epsilon_random_frames=10000,
                 gamma=0.99, epsilon_min=0.01, epsilon_max=1.0, batch_size=32, max_steps_per_episode=72,
                 model=None, iterations=500, do_log=False, random_failure=0, reward_function_type=REWARD_FUNCTION_1):
        self.network_file = network_file
        self.action_zone = action_zone
        self.do_log = do_log
        self.env = WaterNetworkEnv(network_file, seed=seed, action_zone=action_zone, do_log=do_log,
                                   node_demand_random_failure=random_failure, reward_function_type=reward_function_type)
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_interval = self.epsilon_max - self.epsilon_min
        self.epsilon_random_frames = epsilon_random_frames
        self.epsilon_greedy_frames = epsilon_greedy_frames
        self.batch_size = batch_size
        self.max_steps_per_episode = max_steps_per_episode
        self.max_iteration = iterations
        self.num_actions = self.env.action_space.n
        self.epsilon = self.epsilon_max
        self.ALL_EPISODE_REWARDS = []
        self.steps_rewards = []
        self.avg_pressures = {}
        self.node_hour_pressure = {node: {} for node in range(self.env.num_nodes)}
        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.rewards_history = []
        self.done_history = []
        self.episode_reward_history = []
        self.step_action_history_all = []
        if model is None:
            self.model = self.create_nn_model(self.env.num_nodes, self.env.num_valves)
            self.model_target = self.create_nn_model(self.env.num_nodes, self.env.num_valves)
        else:
            self.model = model
            self.model_target = keras.models.clone_model(model)

    def create_nn_model(self, num_nodes, num_valves):
        inputs = layers.Input(shape=(num_nodes,))
        x = layers.Dense(128, activation="relu")(inputs)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
        outputs = layers.Dense(self.env.action_space.n, activation="linear")(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    # -------------------------
    # Train function, epsilon-greedy, replay buffer
    # -------------------------
    def train(self, update_after_actions=7, update_target_network=10000, max_memory_length=100000, verbose=0):
        optimizer = keras.optimizers.Adam(learning_rate=0.025, clipnorm=1.0)
        loss_function = keras.losses.Huber()
        frame_count = 0
        running_reward = 0

        for i in range(self.max_iteration):
            state = np.array(self.env.reset())
            episode_reward = 0
            self.env.perform_random_failure()
            for timestep in range(1, self.max_steps_per_episode):
                frame_count += 1
                # epsilon-greedy
                if frame_count < self.epsilon_random_frames or np.random.rand() < self.epsilon:
                    action, done, reward, state_next = self.run_random_action_and_apply_to_env()
                else:
                    action, done, reward, state_next = self.run_greedy_and_apply_to_env(state)

                # decay epsilon
                self.epsilon -= self.epsilon_interval / self.epsilon_greedy_frames
                self.epsilon = max(self.epsilon, self.epsilon_min)

                # print for logging
                if self.do_log:
                    print(f"Frame {frame_count}: Action={action}, Reward={reward:.2f}, Done={done}, Epsilon={self.epsilon:.3f}")

                # record history
                self.action_history.append(action)
                self.step_action_history_all.append((frame_count, action))
                self.state_history.append(state)
                self.state_next_history.append(state_next)
                self.done_history.append(done)
                self.rewards_history.append(reward)
                state = state_next
                episode_reward += reward

                # update model
                if frame_count % update_after_actions == 0 and len(self.done_history) > self.batch_size:
                    indices = np.random.choice(range(len(self.done_history)), size=self.batch_size)
                    state_sample = np.array([self.state_history[i] for i in indices])
                    state_next_sample = np.array([self.state_next_history[i] for i in indices])
                    rewards_sample = [self.rewards_history[i] for i in indices]
                    action_sample = [self.action_history[i] for i in indices]
                    done_sample = tf.convert_to_tensor([float(self.done_history[i]) for i in indices])

                    future_rewards = self.model_target.predict(state_next_sample, verbose=verbose)
                    updated_q_values = rewards_sample + self.gamma * tf.reduce_max(future_rewards, axis=1)
                    updated_q_values = updated_q_values * (1 - done_sample) - done_sample
                    masks = tf.one_hot(action_sample, self.num_actions)

                    with tf.GradientTape() as tape:
                        q_values = self.model(state_sample)
                        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                        loss = loss_function(updated_q_values, q_action)
                    grads = tape.gradient(loss, self.model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                if frame_count % update_target_network == 0:
                    self.model_target.set_weights(self.model.get_weights())

                if len(self.rewards_history) > max_memory_length:
                    del self.rewards_history[:1]
                    del self.state_history[:1]
                    del self.state_next_history[:1]
                    del self.action_history[:1]
                    del self.done_history[:1]

                if done:
                    break

            self.episode_reward_history.append(episode_reward)
            self.ALL_EPISODE_REWARDS.append((episode_reward, frame_count))
            self.steps_rewards.append((i, episode_reward))
            if len(self.episode_reward_history) > 200:
                del self.episode_reward_history[:1]
            running_reward = np.mean(self.episode_reward_history)

        return self

    def run_random_action_and_apply_to_env(self):
        action = np.random.choice(self.num_actions)
        state_next, reward, done, _ = self.env.step(action)
        state_next = np.array(state_next)
        return action, done, reward, state_next

    def run_greedy_and_apply_to_env(self, state):
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = self.model(state_tensor, training=False)
        action = tf.argmax(action_probs[0]).numpy()
        state_next, reward, done, _ = self.env.step(action)
        state_next = np.array(state_next)
        return action, done, reward, state_next

    # -------------------------
    # Plot functions
    # -------------------------
    def plot_rewards(self, show=False):
        base_path = Path(f'./plt_results/{Path(self.network_file).stem}/rewards/')
        base_path.mkdir(parents=True, exist_ok=True)
        plt.plot([x[0] for x in self.steps_rewards], [y[1] for y in self.steps_rewards])
        plt.xlabel('episodes')
        plt.ylabel('rewards')
        plt.savefig(base_path.joinpath(f"{Path(self.network_file).stem}.jpg"))
        if show:
            plt.show()
        plt.clf()

    def plot_chosen_action(self, show=False):
        base_path = Path(f'./plt_results/{Path(self.network_file).stem}/actions/')
        base_path.mkdir(parents=True, exist_ok=True)
        step_action_history_all = [(item[0], self.env.actions_index.get(item[1])) for item in self.step_action_history_all]
        for i in range(self.env.num_valves):
            keys = [item[0] for item in step_action_history_all]
            values = [item[1][i] for item in step_action_history_all]
            plt.plot(keys, values, color=self.INDEX_COLORS.get(i))
        plt.xlabel('steps')
        plt.ylabel('actions')
        plt.savefig(base_path.joinpath("result.jpg"))
        if show:
            plt.show()
        plt.clf()


# -------------------------
# Main execution
# -------------------------
if __name__ == "__main__":
    load_dotenv(dotenv_path=Path(__file__).parent.joinpath(".env"))
    network_file = os.getenv('INP_FILE')
    dqn_water = DQN_WaterNetwork(network_file=network_file,
                                 iterations=2000,
                                 action_zone=DEFAULT_ACTION_ZONE,
                                 do_log=True,
                                 random_failure=1,
                                 reward_function_type=REWARD_FUNCTION_2)
    dqn_water.train()
    model_path = Path(__file__).parent.joinpath("models").joinpath(Path(network_file).stem)
    dqn_water.model.save(str(model_path) + "_randomness")
    dqn_water.plot_rewards(True)
    dqn_water.plot_chosen_action(True)
