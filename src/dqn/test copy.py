import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque
import random
import os
import wntr
import tensorflow as tf
from tensorflow import keras

# --- 1. محیط EPANET (water_env.py) ---
class WaterNetworkEnv:
    """
    کلاس محیط شبکه آب، مشابه محیط های OpenAI Gym، با استفاده از wntr.
    این کلاس مسئول بارگذاری مدل شبکه آب از یک فایل .inp، انجام شبیه سازی های هیدرولیکی،
    و محاسبه وضعیت و پاداش برای عامل یادگیری تقویتی است.
    """
    def __init__(self, inp_file_path, episode_length=72, reward_function_type="REWARD_FUNCTION_2"):
        """
        مقداردهی اولیه محیط با مسیر فایل INP.
        :param inp_file_path: مسیر فایل EPANET .inp.
        :param episode_length: طول یک اپیزود بر حسب گام های زمانی.
        :param reward_function_type: نوع تابع پاداش (REWARD_FUNCTION_1 یا REWARD_FUNCTION_2).
        """
        if not os.path.exists(inp_file_path):
            raise FileNotFoundError(f"فایل INP در مسیر مشخص شده یافت نشد: {inp_file_path}")

        self.inp_file_path = inp_file_path
        self.episode_length = episode_length
        self.reward_function_type = reward_function_type
        self.current_step_in_episode = 0

        self._load_network_model()
        self._identify_network_elements()

        # طبق متن پایان نامه (بخش 4-2 فضای عمل)، بازه [10, 200] با فواصل 10 واحدی است.
        self.prv_action_values = [i for i in range(10, 201, 10)]

        self.action_space_size = len(self.prv_action_values) ** len(self.prv_names) if len(self.prv_names) > 0 else 1
        self.observation_space_shape = (self.num_demand_nodes,)

        self.min_demand_pressure = 40  # PSI

        self.sim = None
        self.results = None

    def _load_network_model(self):
        """بارگذاری مدل شبکه آب از مسیر فایل INP."""
        try:
            self.wn = wntr.network.WaterNetworkModel(self.inp_file_path)
        except wntr.api.EN_API_ERROR as e:
            raise ValueError(f"Error loading EPANET file: {e}")

    def _identify_network_elements(self):
        """
        شناسایی PRV ها و گره های تقاضا (Junctions).
        این تابع برای تطابق با تغییرات API در wntr به روز شده است.
        """
        self.prv_names = []
        for valve_name, valve_obj in self.wn.valves():
            if valve_obj.valve_type == 'PRV':
                self.prv_names.append(valve_name)

        if not self.prv_names:
            print("هشدار: شبکه .inp هیچ شیر PRV ای ندارد. لطفاً مطمئن شوید که شیر PRV در فایل INP تعریف شده است.")

        self.demand_node_names = [name for name, node in self.wn.nodes.items() if isinstance(node, wntr.network.Junction)]
        if not self.demand_node_names:
             raise ValueError("شبکه .inp هیچ گره تقاضایی (Junction) ندارد.")
        self.num_demand_nodes = len(self.demand_node_names)

    def reset(self):
        """
        محیط را به حالت اولیه بازنشانی می کند.
        :return: وضعیت اولیه (فشارهای گره های تقاضا).
        """
        self.current_step_in_episode = 0
        self._load_network_model()
        self._identify_network_elements()

        try:
            temp_duration_backup = self.wn.options.time.duration
            temp_hydraulic_timestep_backup = self.wn.options.time.hydraulic_timestep
            self.wn.options.time.duration = self.wn.options.time.hydraulic_timestep

            sim_temp = wntr.sim.EpanetSimulator(self.wn)
            initial_results = sim_temp.run_sim()

            self.wn.options.time.duration = temp_duration_backup
            self.wn.options.time.hydraulic_timestep = temp_hydraulic_timestep_backup

            current_pressures = initial_results.node['pressure'].loc[initial_results.node['pressure'].index[0], self.demand_node_names].values
        except Exception as e:
            print(f"خطا در اجرای شبیه سازی wntr در reset: {e}")
            return np.zeros(self.num_demand_nodes, dtype=np.float32)

        return current_pressures.astype(np.float32)

    def step(self, action_index):
        """
        یک عمل را در محیط انجام می دهد و وضعیت بعدی، پاداش و پایان اپیزود را برمی گرداند.
        """
        self.current_step_in_episode += 1
        done = self.current_step_in_episode >= self.episode_length

        if not self.prv_names:
            prv_settings_for_action = []
        else:
            prv_settings_for_action = self._map_action_index_to_prv_settings(action_index)

        for i, prv_name in enumerate(self.prv_names):
            prv_value = prv_settings_for_action[i]
            self.wn.get_link(prv_name).initial_setting = float(prv_value)

        try:
            temp_duration_backup = self.wn.options.time.duration
            temp_hydraulic_timestep_backup = self.wn.options.time.hydraulic_timestep
            self.wn.options.time.duration = self.wn.options.time.hydraulic_timestep

            sim_temp = wntr.sim.epanet.EpanetSimulator(self.wn)
            step_results = sim_temp.run_sim()

            self.wn.options.time.duration = temp_duration_backup
            self.wn.options.time.hydraulic_timestep = temp_hydraulic_timestep_backup

            next_state = step_results.node['pressure'].iloc[-1][self.demand_node_names].values
        except Exception as e:
            print(f"خطا در اجرای شبیه سازی wntr در گام {self.current_step_in_episode}: {e}")
            reward = -10000
            next_state = np.zeros(self.num_demand_nodes, dtype=np.float32)
            done = True
            return next_state, reward, done, {}

        next_state = next_state.astype(np.float32)
        reward = self._calculate_reward(next_state)
        info = {}
        return next_state, reward, done, info

    def _map_action_index_to_prv_settings(self, action_index):
        prv_settings = []
        num_values_per_prv = len(self.prv_action_values)

        temp_action_index = action_index
        for _ in range(len(self.prv_names)):
            value_index = temp_action_index % num_values_per_prv
            prv_settings.insert(0, self.prv_action_values[value_index])
            temp_action_index //= num_values_per_prv
        return prv_settings

    def _calculate_reward(self, node_pressures):
        def cost_function(pressure):
            if pressure < self.min_demand_pressure:
                return -1000
            elif pressure >= self.min_demand_pressure and pressure <= 40:
                if self.min_demand_pressure == 40:
                    return 1.0
                if (40 - self.min_demand_pressure) > 0:
                    return 0.5 + 0.5 * (40 - pressure) / (40 - self.min_demand_pressure)
                else:
                    return 1.0 if pressure == 40 else 0.0
            elif pressure > 40 and pressure <= 70:
                return 1.0 - (pressure - 40) / 30.0
            else:
                return max(-5.0, 0.0 - (pressure - 70) / 10.0 * 2.0)

        if self.reward_function_type == "REWARD_FUNCTION_1":
            avg_pressure = np.mean(node_pressures)
            return cost_function(avg_pressure)

        elif self.reward_function_type == "REWARD_FUNCTION_2":
            for pressure in node_pressures:
                if pressure < self.min_demand_pressure:
                    return -1000
            avg_pressure = np.mean(node_pressures)
            return cost_function(avg_pressure)
        else:
            raise ValueError("نوع تابع پاداش نامعتبر است. از 'REWARD_FUNCTION_1' یا 'REWARD_FUNCTION_2' استفاده کنید.")

# --- 2. پیاده سازی عامل DQN (dqn_agent.py) ---
class DQNAgent:
    """
    عامل Deep Q-Network (DQN) برای کنترل فشار شبکه آب.
    """
    def __init__(self, observation_space_shape, action_space_size, params):
        self.observation_space_shape = observation_space_shape
        self.action_space_size = action_space_size

        self.learning_rate = params.get('learning_rate', 0.0001)
        self.gamma = params.get('gamma', 0.99)
        self.batch_size = params.get('batch_size', 32)
        self.epsilon = params.get('epsilon', 1.0)
        self.epsilon_decay = params.get('epsilon_decay', 0.9999)
        self.epsilon_min = params.get('epsilon_min', 0.01)
        self.max_mem_size = params.get('max_mem_size', 50000)
        self.target_update_frequency = params.get('target_update_frequency', 1000)

        self.memory = deque(maxlen=self.max_mem_size)
        self.t_step = 0

        self.q_network = self._build_model()
        self.target_q_network = self._build_model()
        self.target_q_network.set_weights(self.q_network.get_weights())

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Input(shape=self.observation_space_shape),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.action_space_size, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss=keras.losses.Huber())
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_space_size)

        state_tensor = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
        q_values = self.q_network.predict(state_tensor, verbose=0)[0]
        return np.argmax(q_values)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        if self.t_step % self.target_update_frequency == 0:
            self.target_q_network.set_weights(self.q_network.get_weights())

        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([exp[0] for exp in minibatch])
        actions = np.array([exp[1] for exp in minibatch])
        rewards = np.array([exp[2] for exp in minibatch])
        next_states = np.array([exp[3] for exp in minibatch])
        dones = np.array([exp[4] for exp in minibatch])

        current_q_values = self.q_network.predict(states, verbose=0)
        target_q_values_next_state = self.target_q_network.predict(next_states, verbose=0)

        target_q_values = np.copy(current_q_values)
        batch_indices = np.arange(self.batch_size, dtype=np.int32)
        
        max_next_q = np.max(target_q_values_next_state, axis=1)
        target_q_for_actions = rewards + self.gamma * max_next_q * (1 - dones)
        target_q_values[batch_indices, actions] = target_q_for_actions

        self.q_network.train_on_batch(states, target_q_values)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.t_step += 1

    def save_models(self, path="dqn_models"):
        if not os.path.exists(path):
            os.makedirs(path)
        self.q_network.save_weights(os.path.join(path, "q_network.h5"))
        self.target_q_network.save_weights(os.path.join(path, "target_q_network.h5"))
        print(f"مدل های DQN در {path} ذخیره شدند.")

    def load_models(self, path="dqn_models"):
        try:
            self.q_network.load_weights(os.path.join(path, "q_network.h5"))
            self.target_q_network.load_weights(os.path.join(path, "target_q_network.h5"))
            print(f"مدل های DQN از {path} بارگذاری شدند.")
        except Exception as e:
            print(f"خطا در بارگذاری مدل ها: {e}. مطمئن شوید مسیر و فایل ها صحیح هستند.")


# --- 3. اسکریپت اصلی آموزش (main.py) ---
def train_dqn_agent(env, agent, n_total_steps):
    episode_rewards = []
    all_avg_pressures_during_training = []

    current_episode_reward = 0
    state = env.reset()
    episode_count = 0

    print(f"شروع آموزش برای {n_total_steps} گام زمانی...")
    with tqdm(total=n_total_steps, desc="Training Steps") as pbar:
        for step in range(n_total_steps):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            current_episode_reward += reward
            all_avg_pressures_during_training.append(np.mean(state))

            if done:
                episode_rewards.append(current_episode_reward)
                episode_count += 1
                pbar.set_postfix({
                    'Episode': episode_count,
                    'Reward': f"{current_episode_reward:.2f}",
                    'Epsilon': f"{agent.epsilon:.4f}"
                })
                state = env.reset()
                current_episode_reward = 0
            
            pbar.update(1)

    if current_episode_reward != 0:
        episode_rewards.append(current_episode_reward)
        episode_count += 1

    print("\nآموزش به پایان رسید.")
    return episode_rewards, all_avg_pressures_during_training

def plot_results(rewards_per_episode, avg_pressures_history, episode_length):
    plt.figure(figsize=(14, 6))

    # نمودار پاداش
    plt.subplot(1, 2, 1)
    plt.plot(rewards_per_episode, label='Total Reward per Episode')
    plt.title('پاداش کل در هر اپیزود')
    plt.xlabel('اپیزود')
    plt.ylabel('پاداش کل')
    plt.grid(True)
    plt.legend()
    
    # نمودار فشار میانگین
    plt.subplot(1, 2, 2)
    # برای نمایش نمودار فشار بر اساس اپیزود، باید داده ها را به صورت میانگین در هر اپیزود گروه بندی کنیم.
    # این کد فشار میانگین را در طول تمام گام ها نشان می دهد.
    num_episodes = len(rewards_per_episode)
    # از یک فیلتر میانگین متحرک برای صاف کردن نمودار فشار استفاده می کنیم.
    window_size = 500 # می توانید این مقدار را تغییر دهید.
    avg_pressures_smoothed = np.convolve(avg_pressures_history, np.ones(window_size)/window_size, mode='valid')
    plt.plot(avg_pressures_smoothed, label='Avg Pressure (Smoothed)')
    plt.title('فشار میانگین در طول زمان')
    plt.xlabel(f'گام زمانی ({window_size} گام)')
    plt.ylabel('فشار میانگین (PSI)')
    plt.grid(True)
    plt.axhline(y=40, color='r', linestyle='--', label='Min Pressure Threshold')
    plt.axhline(y=70, color='g', linestyle='--', label='Max Pressure Threshold')
    plt.legend()

    plt.suptitle('نتایج آموزش عامل یادگیری تقویتی')
    plt.tight_layout()
    plt.savefig('training_results.png')
    print("\nنمودارهای نتایج آموزش در فایل 'training_results.png' ذخیره شدند.")

# اجرای اصلی برنامه
if __name__ == "__main__":
  
    
    inp_file_path = 'net1.inp'

    dqn_params = {
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'batch_size': 32,
        'epsilon': 1.0,
        'epsilon_decay': 0.9999,
        'epsilon_min': 0.01,
        'max_mem_size': 50000,
        'target_update_frequency': 1000
    }

    n_total_training_steps = 50000
    episode_length = 72

    try:
        env = WaterNetworkEnv(inp_file_path=inp_file_path,
                              episode_length=episode_length,
                              reward_function_type="REWARD_FUNCTION_2")
        agent = DQNAgent(env.observation_space_shape, env.action_space_size, dqn_params)

        print(f"شکل فضای حالت: {env.observation_space_shape}")
        print(f"اندازه فضای عمل: {env.action_space_size}")
        print(f"تعداد PRV ها: {len(env.prv_names)}")
        print(f"مقادیر مجاز PRV: {env.prv_action_values}")
        print(f"تعداد گره های تقاضا (Junctions): {env.num_demand_nodes}")
        
        rewards_per_episode, avg_pressures_history = train_dqn_agent(env, agent, n_total_training_steps)
        agent.save_models()
        plot_results(rewards_per_episode, avg_pressures_history, episode_length)

    except FileNotFoundError as fnfe:
        print(f"خطا: {fnfe}")
    except ValueError as ve:
        print(f"خطا در مقداردهی اولیه محیط: {ve}")
    except Exception as e:
        print(f"یک خطای غیرمنتظره رخ داد: {e}")
    finally:
        if os.path.exists(temp_inp_file):
            os.remove(temp_inp_file)
            print(f"فایل موقت '{temp_inp_file}' حذف شد.")

