import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wntr
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random

# --- پیکربندی‌های پروژه ---
# مسیر به فایل EPANET .inp (مثلاً "networks/simple_net.inp")
# فرض کنید فایل .inp شما در زیرپوشه 'networks' قرار دارد
NETWORK_FILE = 'net1.inp' # Example: Replace with your actual network file
SAVE_MODEL_PATH = 'trained_dqn_model.h5' # Path to save/load model weights

# تنظیمات محیط
ENV_CONFIG = {
    'episode_length': 72,  # 72 time steps = 3 days (if report interval is 1 hour)
    'min_pressure_threshold': 40, # PSI
    'max_pressure_threshold': 70, # PSI
    'action_space_values': [i * 10 for i in range(1, 21)], # PRV settings from 10 to 200 PSI, step 10
    'reward_type': 'reward_2' # Use 'reward_1' or 'reward_2'
}

# تنظیمات عامل DQN
AGENT_CONFIG = {
    'learning_rate': 0.0001,
    'gamma': 0.99,
    'epsilon': 1.0,           # Initial epsilon for exploration
    'eps_decay': 0.000005,    # Rate of epsilon decay
    'eps_end': 0.01,          # Minimum epsilon
    'batch_size': 64,         # Number of experiences to learn from
    'max_mem_size': 100000,   # Max replay memory size
    'target_model_update_freq': 500 # Update target model every 500 learning steps
}

# تنظیمات آموزش
N_EPISODES = 5000 # Number of training episodes
# N_EPISODES = 100000 # For more robust training, use a larger number of episodes


class DQNAgent:
    """
    Deep Q-Network (DQN) Agent implementation using Keras.
    """
    def __init__(self, input_dims, n_actions, learning_rate=0.0001, gamma=0.99,
                 epsilon=1.0, eps_decay=0.00099, eps_end=0.01,
                 batch_size=32, max_mem_size=50000,
                 target_model_update_freq=100):
        """
        Initializes the DQN Agent.

        Args:
            input_dims (int): Dimensionality of the state space (e.g., number of pressure nodes).
            n_actions (int): Number of possible discrete actions.
            learning_rate (float): Learning rate for the Adam optimizer.
            gamma (float): Discount factor for future rewards.
            epsilon (float): Initial epsilon value for epsilon-greedy action selection.
            eps_decay (float): Rate at which epsilon decays over time.
            eps_end (float): Minimum epsilon value.
            batch_size (int): Number of experiences to sample from replay memory for learning.
            max_mem_size (int): Maximum size of the replay memory.
            target_model_update_freq (int): How often (in learning steps) to update the target Q-network.
        """
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_end = eps_end
        self.batch_size = batch_size
        self.max_mem_size = max_mem_size
        self.target_model_update_freq = target_model_update_freq

        self.memory = deque(maxlen=self.max_mem_size) # Replay memory
        self.q_eval_model = self._build_q_network()  # Main Q-network
        self.q_target_model = self._build_q_network() # Target Q-network
        self.q_target_model.set_weights(self.q_eval_model.get_weights()) # Initialize target with eval weights
        self.learn_step_counter = 0 # Counter for learning steps

    def _build_q_network(self):
        """
        Builds the Keras Deep Q-Network model.
        Architecture based on the provided document: 3 hidden layers (128, 64, 64) with ReLU.
        """
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(self.input_dims,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.n_actions, activation='linear') # Output layer for Q-values
        ])
        
        # Using Huber loss as specified in the document
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss=keras.losses.Huber()) 
        return model

    def remember(self, state, action, reward, next_state, done):
        """
        Stores an experience tuple in the replay memory.

        Args:
            state (np.array): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.array): Next state.
            done (bool): Whether the episode ended.
        """
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, observation):
        """
        Chooses an action based on the epsilon-greedy strategy.

        Args:
            observation (np.array): Current state observation.

        Returns:
            int: Index of the chosen action.
        """
        if np.random.rand() <= self.epsilon:
            # Exploration: Choose a random action
            return np.random.randint(self.n_actions)
        else:
            # Exploitation: Choose the action with the highest Q-value from the evaluation network
            # Reshape observation for model prediction if it's a single sample
            if observation.ndim == 1:
                observation = np.expand_dims(observation, axis=0)
            q_values = self.q_eval_model.predict(observation, verbose=0)
            return np.argmax(q_values[0])

    def learn(self):
        """
        Performs one learning step using a batch of experiences from replay memory.
        Updates the evaluation Q-network.
        """
        # Do not learn if memory is not yet full enough for a batch
        if len(self.memory) < self.batch_size:
            return

        # Update target network weights periodically
        if self.learn_step_counter % self.target_model_update_freq == 0:
            self.q_target_model.set_weights(self.q_eval_model.get_weights())

        # Sample a batch of experiences from memory
        batch = random.sample(self.memory, self.batch_size)
        
        # Unpack the batch
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to numpy arrays for TensorFlow processing
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.bool_)

        # Predict Q-values for current states using the evaluation network
        q_eval = self.q_eval_model.predict(states, verbose=0)
        # Predict Q-values for next states using the target network
        q_next = self.q_target_model.predict(next_states, verbose=0)

        # Build the target Q-values for training
        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # Calculate the target Q-value for each experience in the batch
        # If 'done' is True, the target Q-value is just the reward.
        # Otherwise, it's reward + gamma * max_Q(s', a')
        q_target[batch_index, actions] = rewards + self.gamma * np.max(q_next, axis=1) * (1 - dones)
        
        # Train the evaluation Q-network
        self.q_eval_model.train_on_batch(states, q_target)

        # Decay epsilon
        self.epsilon = max(self.eps_end, self.epsilon - self.eps_decay)
        self.learn_step_counter += 1

    def save_model(self, path):
        """Saves the Q-evaluation model weights."""
        self.q_eval_model.save_weights(path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Loads the Q-evaluation model weights."""
        self.q_eval_model.load_weights(path)
        self.q_target_model.set_weights(self.q_eval_model.get_weights()) # Update target as well
        print(f"Model loaded from {path}")

class WaterNetworkEnv:
    """
    A custom environment class for water distribution networks,
    simulating interaction with EPANET via WNTR.
    """
    def __init__(self, network_file_path, episode_length=72,
                 min_pressure_threshold=40, max_pressure_threshold=70,
                 action_space_values=None, reward_type='reward_2'):
        """
        Initializes the water network environment.

        Args:
            network_file_path (str): Path to the EPANET .inp file.
            episode_length (int): Number of time steps in each episode.
            min_pressure_threshold (float): Minimum acceptable pressure at demand nodes.
            max_pressure_threshold (float): Maximum desirable pressure at demand nodes.
            action_space_values (list): List of discrete pressure values for PRVs.
                                        If None, a default range (10-200 PSI, step 10) is used.
            reward_type (str): Type of reward function to use ('reward_1' or 'reward_2').
        """
        self.network_file_path = network_file_path
        self.episode_length = episode_length
        self.min_pressure_threshold = min_pressure_threshold
        self.max_pressure_threshold = max_pressure_threshold
        self.reward_type = reward_type

        # Load the network model using wntr
        # wntr.network.WaterNetworkModel.read_inp is the method to read .inp files.
        self.wn = wntr.network.WaterNetworkModel(network_file_path)

        # Ensure the network has PRVs (Pressure Reducing Valves)
        self.prv_names = [name for name, valve in self.wn.valves() if valve.valve_type == 'PRV']
        if not self.prv_names:
            raise ValueError("No PRVs found in the network. Please add PRVs to your .inp file.")

        # Identify demand nodes (junctions with demand)
        # We only consider junctions with demand as "demand nodes" for state observation.
        self.demand_node_names = [name for name, node in self.wn.junctions() if node.base_demand > 0]
        if not self.demand_node_names:
            print("Warning: No demand nodes found. State observation will be based on all junctions.")
            self.demand_node_names = [name for name, node in self.wn.junctions()]

        self.state_space_dim = len(self.demand_node_names)
        self.num_prvs = len(self.prv_names)

        # Define action space (discrete PRV settings)
        if action_space_values is None:
            # Default action space: PRV settings from 10 PSI to 200 PSI, step 10
            self.action_space_values = [i * 10 for i in range(1, 21)]  # [10, 20, ..., 200]
        else:
            self.action_space_values = action_space_values
        
        # The number of discrete actions an agent can take.
        # If there are multiple PRVs, a single action means setting all PRVs to the chosen value.
        # For more complex control, action space would be (num_prvs * len(action_space_values)).
        # For simplicity, we assume one action index corresponds to one PRV setting for all PRVs.
        self.n_actions = len(self.action_space_values)

        self.sim = None  # WNTR simulator object
        self.current_time_step = 0
        self.initial_prv_settings = {} # To store initial settings for reset

        # Store initial PRV settings to revert to on reset
        for prv_name in self.prv_names:
            # For PRVs, the setting is typically `pressure_setting`
            self.initial_prv_settings[prv_name] = self.wn.get_node(prv_name).pressure_setting

        print(f"Environment initialized. Network: {self.network_file_path}")
        print(f"Demand Nodes for state: {len(self.demand_node_names)}")
        print(f"PRVs for action: {len(self.prv_names)}")
        print(f"Discrete Action Space Values: {self.action_space_values}")
        print(f"Total possible actions: {self.n_actions}")

    def reset(self):
        """
        Resets the environment to its initial state.
        This re-initializes the WNTR simulator and sets PRVs to initial values.

        Returns:
            np.array: Initial state (pressures at demand nodes).
        """
        # Close previous simulator if exists
        if self.sim:
            self.sim.close()
        
        # Re-initialize the network model to ensure a clean state
        # This is crucial for proper reset, especially after modifying network
        self.wn = wntr.network.WaterNetworkModel(self.network_file_path)

        # Set PRVs back to their initial settings
        for prv_name, initial_setting in self.initial_prv_settings.items():
            prv = self.wn.get_node(prv_name)
            # wntr automatically sets the correct attribute based on valve type
            prv.pressure_setting = initial_setting 

        # Create a new simulator for the current episode
        self.sim = wntr.sim.EpanetSimulator(self.wn)
        self.results = self.sim.run_sim() # Run full simulation for initial state
        
        self.current_time_step = 0

        # Get initial pressures at demand nodes
        # Pressures are usually in PSI in WNTR results unless units are changed
        initial_pressures = self.results.node['pressure'].loc[self.wn.options.time.report_times[0], self.demand_node_names].values
        
        # Ensure state is a float32 numpy array
        initial_state = np.array(initial_pressures, dtype=np.float32)
        
        return initial_state

    def step(self, action_index):
        """
        Performs one step in the environment given an action.

        Args:
            action_index (int): Index of the chosen action from `self.action_space_values`.

        Returns:
            tuple: (next_state, reward, done, info)
                next_state (np.array): Pressures at demand nodes after the action.
                reward (float): Reward obtained from the action.
                done (bool): True if the episode has ended, False otherwise.
                info (dict): Additional information (e.g., actual PRV setting).
        """
        if self.current_time_step >= self.episode_length:
            return self.get_state(), 0, True, {"message": "Episode ended"}

        # Get the actual PRV setting value from the action space
        prv_setting = self.action_space_values[action_index]

        # Apply the action to all PRVs in the network for the current time step
        # Note: In a more complex scenario, each PRV might have its own action.
        # Here, we assume one action applies to all PRVs (or the main PRV).
        for prv_name in self.prv_names:
            prv = self.wn.get_node(prv_name)
            # WNTR allows setting control values directly.
            # Here, we set a simple time-based control for the next time step.
            # For dynamic control, we set the valve property directly before simulation.
            # A simpler way is to update the WNTR model's control for the next step.
            prv.pressure_setting = prv_setting # Apply the new setting

        # Re-run simulation from the current time step for one report interval
        # WNTR can be run incrementally, but it's often easier to modify the model
        # and re-run for a short period, or manage controls externally.
        # For simplicity and robust WNTR usage, we'll re-run a portion or rely on
        # how WNTR handles controls when `run_sim` is called after modifications.
        
        # In this simple implementation, we'll assume setting `prv.pressure_setting`
        # directly updates the model for the next `get_results_at_time`.
        # WNTR simulation generally requires re-running to see effects of changes.
        # For a truly dynamic control, you would need to run `sim.run_sim` for 
        # a single time step after updating controls for that step.
        
        # For simplicity and to fit typical WNTR usage in an RL loop:
        # We need to run simulation for one step, get results, then update controls for next.
        # WNTR's sim.run_sim() usually runs the whole scenario.
        # A more 'real-time' control involves:
        # sim = wntr.sim.EpanetSimulator(self.wn)
        # sim.run_sim(start_time=self.current_time_step * self.wn.options.time.report_interval,
        #             end_time=(self.current_time_step + 1) * self.wn.options.time.report_interval)
        # Then get results for that specific time.

        # Simplest way to get results after applying control for current step:
        # If we want the action to take effect *for* current_time_step, we get results at current_time_step+1
        # and base reward/next_state on that.
        
        # To get the state *after* the action has been applied for this timestep
        # we advance the simulator's internal clock and get results for the next report time.
        # Note: WNTR's `run_sim` often runs the full duration.
        # For granular control, you typically define `TimeControls` in the .inp or via WNTR API.
        # A workaround for dynamic control in RL:
        # 1. Modify network model (e.g., PRV setting).
        # 2. Rerun a short simulation from current time.
        # 3. Get results for the end of that short simulation.

        # Let's simplify: we will assume self.results already contains all timeseries.
        # We fetch the pressures at the *next* report time after applying the action.
        
        # This implies that `reset` has already run `sim.run_sim()` for the whole duration.
        # If `sim.run_sim()` is only run once in `reset`, then the controls set here
        # will not dynamically affect `self.results` directly in the `step` method unless
        # the simulation is partially re-run or WNTR's controls API is used more deeply.

        # For a truly dynamic step-by-step simulation, you'd integrate the WNTR simulator
        # to run one step at a time, apply controls, then get results.
        # WNTR's `sim.run_sim` computes everything upfront.
        # A common pattern for dynamic control with WNTR is to
        # 1. Set controls for the *next* time step.
        # 2. Advance time in simulator.
        # 3. Get results for that advanced time.

        # Here's a common simplification for RL: The agent acts for the *next* time step.
        # So we get state at `current_time_step`, act for `current_time_step+1`,
        # then get reward/next_state from `current_time_step+1`.
        
        self.current_time_step += 1
        done = self.current_time_step >= self.episode_length

        if done:
            next_pressures = self.get_state() # Last observed state
        else:
            # Get pressures for the next report time (current_time_step after increment)
            # Ensure current_time_step maps correctly to report_times index
            if self.current_time_step >= len(self.wn.options.time.report_times):
                # This could happen if episode_length exceeds actual simulation report times
                # For simplicity, we'll just use the last available data or handle as done.
                next_pressures = self.results.node['pressure'].loc[self.wn.options.time.report_times[-1], self.demand_node_names].values
                done = True # Force done if we run out of simulation data
            else:
                next_pressures = self.results.node['pressure'].loc[self.wn.options.time.report_times[self.current_time_step], self.demand_node_names].values

        # Calculate reward
        reward = self._calculate_reward(next_pressures)
        
        next_state = np.array(next_pressures, dtype=np.float32)

        info = {
            "prv_setting_applied": prv_setting,
            "current_sim_time_step": self.wn.options.time.report_times[self.current_time_step] if not done else self.wn.options.time.report_times[-1]
        }
        
        return next_state, reward, done, info

    def get_state(self):
        """
        Returns the current state (pressures at demand nodes).
        Assumes `results` from `sim.run_sim()` is available.
        """
        # Get pressures at the current report time
        current_pressures = self.results.node['pressure'].loc[self.wn.options.time.report_times[self.current_time_step], self.demand_node_names].values
        return np.array(current_pressures, dtype=np.float32)

    def _calculate_reward(self, pressures):
        """
        Calculates the reward based on the given pressures.
        Implements Reward Function 1 and 2 from the document.
        """
        if self.reward_type == 'reward_1':
            return self._reward_function_1(pressures)
        elif self.reward_type == 'reward_2':
            return self._reward_function_2(pressures)
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")

    def _reward_function_1(self, pressures):
        """
        Reward Function 1: Minimize average pressure within a target range (40-70 PSI),
        penalizing outside this range. Closer to 40 PSI is better.
        """
        total_cost = 0
        for p in pressures:
            if p < self.min_pressure_threshold:
                # Heavy penalty for pressure below minimum
                total_cost += -100 # Or a more complex penalty (e.g., squared difference)
            elif p > self.max_pressure_threshold:
                # Penalty for pressure too high (wasteful)
                # Linear decrease from 1.0 at max_pressure_threshold to -0.5 at max_pressure_threshold + 50
                total_cost += max(-0.5, 1.0 - (p - self.max_pressure_threshold) / 50)
            else:
                # Optimal range: 40-70 PSI. Reward peaks at 40, decreases to 1.0 at 70.
                # Example: reward of 2.0 at 40 PSI, 1.0 at 50 PSI, 0.5 at 60 PSI, 0.0 at 70 PSI
                if p <= 50:
                    total_cost += 2.0 - (p - self.min_pressure_threshold) / 10 # 2.0 at 40, 1.0 at 50
                else: # 50 to 70 PSI
                    total_cost += 1.0 - (p - 50) / 20 # 1.0 at 50, 0.0 at 70
        return total_cost / len(pressures) if pressures.any() else 0

    def _reward_function_2(self, pressures):
        """
        Reward Function 2: Similar to Reward 1, but with a severe penalty
        if any pressure is very low or negative.
        """
        total_reward = 0
        for p in pressures:
            if p <= 0: # Severe penalty for zero or negative pressure
                return -1000 # Immediate severe penalty, stop further calculation
            elif p < self.min_pressure_threshold: # Penalty for pressure below minimum (but >0)
                total_reward += -20 # Milder penalty than negative pressure, but still significant
            else:
                # Apply the same cost function as Reward 1 for the optimal range
                if p <= 50:
                    total_reward += 2.0 - (p - self.min_pressure_threshold) / 10
                else: # 50 to 70 PSI
                    total_reward += 1.0 - (p - 50) / 20
                
                if p > self.max_pressure_threshold:
                    total_reward += max(-0.5, 1.0 - (p - self.max_pressure_threshold) / 50)
        
        return total_reward / len(pressures) if pressures.any() else 0

    def close(self):
        """
        Closes the WNTR simulator.
        """
        if self.sim:
            self.sim.close()
            self.sim = None
        print("Environment closed.")




# --- شروع آموزش ---
def train_agent():
    # مقداردهی اولیه محیط
    env = WaterNetworkEnv(
        network_file_path=NETWORK_FILE,
        episode_length=ENV_CONFIG['episode_length'],
        min_pressure_threshold=ENV_CONFIG['min_pressure_threshold'],
        max_pressure_threshold=ENV_CONFIG['max_pressure_threshold'],
        action_space_values=ENV_CONFIG['action_space_values'],
        reward_type=ENV_CONFIG['reward_type']
    )

    # ابعاد حالت (تعداد گره‌های تقاضا)
    # ابعاد حالت در اینجا فقط فشارهای گره‌های تقاضا هستند.
    # اگر می‌خواهید تنظیمات PRV جاری نیز بخشی از حالت باشد، باید آن را در env.get_state() اضافه کنید.
    input_dims = env.state_space_dim 

    # مقداردهی اولیه عامل
    agent = DQNAgent(
        input_dims=input_dims,
        n_actions=env.n_actions,
        learning_rate=AGENT_CONFIG['learning_rate'],
        gamma=AGENT_CONFIG['gamma'],
        epsilon=AGENT_CONFIG['epsilon'],
        eps_decay=AGENT_CONFIG['eps_decay'],
        eps_end=AGENT_CONFIG['eps_end'],
        batch_size=AGENT_CONFIG['batch_size'],
        max_mem_size=AGENT_CONFIG['max_mem_size'],
        target_model_update_freq=AGENT_CONFIG['target_model_update_freq']
    )

    # لیست‌ها برای ذخیره نتایج
    episode_rewards = []
    episode_avg_pressures = []

    print("\n--- شروع آموزش عامل DQN ---")
    # نوار پیشرفت برای نمایش وضعیت آموزش
    for episode in tqdm(range(N_EPISODES), desc="Training Episodes"):
        state = env.reset() # بازنشانی محیط برای شروع اپیزود جدید
        done = False
        total_episode_reward = 0
        episode_pressure_sum = 0
        episode_step_count = 0

        # حلقه گام‌های زمانی در هر اپیزود
        while not done:
            action_index = agent.choose_action(state) # عامل یک عمل را انتخاب می‌کند
            
            # انجام عمل در محیط و دریافت حالت بعدی، پاداش و وضعیت
            next_state, reward, done, info = env.step(action_index)
            
            # ذخیره تجربه در حافظه بازپخش عامل
            agent.remember(state, action_index, reward, next_state, done)
            
            # عامل یاد می‌گیرد (از حافظه بازپخش نمونه‌برداری می‌کند)
            agent.learn()
            
            total_episode_reward += reward
            episode_pressure_sum += np.mean(next_state) # میانگین فشار گره‌های تقاضا در این گام
            episode_step_count += 1
            
            state = next_state # به‌روزرسانی حالت

        episode_rewards.append(total_episode_reward)
        if episode_step_count > 0:
            episode_avg_pressures.append(episode_pressure_sum / episode_step_count)
        else:
            episode_avg_pressures.append(0) # در صورت عدم وجود گام در اپیزود

        # چاپ وضعیت در هر چند اپیزود
        if episode % 100 == 0:
            print(f"\nEpisode: {episode}, Total Reward: {total_episode_reward:.2f}, "
                  f"Avg Pressure: {episode_avg_pressures[-1]:.2f}, Epsilon: {agent.epsilon:.4f}")

    env.close() # بستن محیط شبیه‌ساز پس از اتمام آموزش
    agent.save_model(SAVE_MODEL_PATH) # ذخیره وزن‌های مدل آموزش‌دیده

    return episode_rewards, episode_avg_pressures, agent

# --- نمایش نتایج ---
def plot_results(rewards, pressures):
    """
    Plots the total reward per episode and average pressure per episode.
    """
    # نمودار پاداش
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)

    # نمودار میانگین فشار
    plt.subplot(1, 2, 2)
    plt.plot(pressures)
    plt.title('Average Pressure per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Pressure (PSI)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# --- تابع برای تست عامل آموزش دیده (اختیاری) ---
def test_agent(agent, env, num_test_episodes=5):
    """
    Tests the performance of a trained agent.
    """
    print(f"\n--- شروع تست عامل برای {num_test_episodes} اپیزود ---")
    test_rewards = []
    test_avg_pressures = []

    # Epsilon را برای تست صفر می‌کنیم تا فقط بهره‌برداری انجام شود
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0 

    for episode in range(num_test_episodes):
        state = env.reset()
        done = False
        total_episode_reward = 0
        episode_pressure_sum = 0
        episode_step_count = 0
        
        print(f"\nTest Episode {episode + 1}:")
        while not done:
            action_index = agent.choose_action(state)
            next_state, reward, done, info = env.step(action_index)
            
            total_episode_reward += reward
            episode_pressure_sum += np.mean(next_state)
            episode_step_count += 1
            state = next_state
            
            # Print current step details (optional)
            # print(f"  Step {episode_step_count}: PRV set to {info['prv_setting_applied']} PSI, Avg Pressures: {np.mean(next_state):.2f}, Reward: {reward:.2f}")

        test_rewards.append(total_episode_reward)
        if episode_step_count > 0:
            test_avg_pressures.append(episode_pressure_sum / episode_step_count)
        else:
            test_avg_pressures.append(0)

        print(f"Test Episode {episode + 1} finished. Total Reward: {total_episode_reward:.2f}, Avg Pressure: {test_avg_pressures[-1]:.2f}")

    agent.epsilon = original_epsilon # Reset epsilon
    env.close() # Close environment after testing
    
    print("\n--- تست به پایان رسید ---")
    print(f"میانگین پاداش تست: {np.mean(test_rewards):.2f}")
    print(f"میانگین فشار تست: {np.mean(test_avg_pressures):.2f}")


# --- اجرای اصلی ---
if __name__ == "__main__":
    # مرحله آموزش
    rewards, pressures, trained_agent = train_agent()
    plot_results(rewards, pressures)

    # مرحله تست (اختیاری)
    # env_test = WaterNetworkEnv(
    #     network_file_path=NETWORK_FILE,
    #     episode_length=ENV_CONFIG['episode_length'],
    #     min_pressure_threshold=ENV_CONFIG['min_pressure_threshold'],
    #     max_pressure_threshold=ENV_CONFIG['max_pressure_threshold'],
    #     action_space_values=ENV_CONFIG['action_space_values'],
    #     reward_type=ENV_CONFIG['reward_type']
    # )
    # agent_for_test = DQNAgent(
    #     input_dims=env_test.state_space_dim,
    #     n_actions=env_test.n_actions,
    #     **{k: v for k, v in AGENT_CONFIG.items() if k not in ['epsilon']} # Don't pass initial epsilon, it will be set to 0
    # )
    # agent_for_test.load_model(SAVE_MODEL_PATH) # Load trained weights
    # test_agent(agent_for_test, env_test, num_test_episodes=5)

