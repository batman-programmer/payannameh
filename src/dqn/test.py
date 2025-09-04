import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
from collections import namedtuple
import matplotlib.pyplot as plt

# استفاده از GPU در صورت موجود بودن
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print("GPU is available and being used.")
    except RuntimeError as e:
        print(e)
else:
    print("GPU not available, using CPU.")

# تعریف یک Tuple برای نگهداری تجربه‌ها (حالت، عمل، پاداش، حالت بعدی، وضعیت پایان)
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class PrioritizedReplayBuffer:
    """
    بافر بازپخش اولویت‌بندی شده (PER) با استفاده از آرایه‌های NumPy.
    این بافر تجربه‌ها را بر اساس اهمیت و خطای زمانی (TD-error) ذخیره می‌کند.
    """
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.alpha = alpha  # ضریب تعیین میزان استفاده از اولویت

    def push(self, *args):
        """
        یک تجربه جدید را به بافر اضافه می‌کند.
        به آن بالاترین اولویت را می‌دهد تا حداقل یک بار نمونه‌برداری شود.
        """
        self.buffer[self.position] = Transition(*args)
        self.priorities[self.position] = self.priorities.max() if len(self) > 0 else 1.0
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        نمونه‌ای از تجربه‌ها را بر اساس اولویت‌شان برمی‌گرداند.
        همچنین وزن‌های مهم بودن نمونه (IS-weights) را برای کاهش سوگیری محاسبه می‌کند.
        """
        if len(self) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        # تبدیل اولویت‌ها به یک توزیع احتمال
        probabilities = priorities ** self.alpha
        sum_probabilities = probabilities.sum()
        if sum_probabilities == 0:
            probabilities = np.ones_like(probabilities) / len(probabilities)
        else:
            probabilities /= sum_probabilities

        # نمونه‌گیری بر اساس توزیع احتمال
        indices = np.random.choice(len(self), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        # محاسبه IS-weights
        total = len(self)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, tf.convert_to_tensor(weights, dtype=tf.float32)

    def update_priorities(self, indices, errors):
        """
        اولویت‌های تجربه‌ها را بر اساس خطاهای جدید به‌روز می‌کند.
        """
        for idx, error in zip(indices, errors):
            self.priorities[idx] = np.abs(error) + 1e-5

    def __len__(self):
        if self.buffer[self.position] is None:
            return self.position
        else:
            return self.capacity

def create_q_network(state_dim, action_dim):
    """
    شبکه عصبی برای تخمین ارزش Q (Q-value) را ایجاد می‌کند.
    """
    model = keras.Sequential([
        layers.Dense(128, activation="relu", input_shape=(state_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(action_dim, activation="linear")
    ])
    return model

class SafeDQN:
    """
    عامل Safe DQN که شامل یک بافر بازپخش اولویت‌بندی شده (PER)
    و یک لایه ایمنی برای جلوگیری از اعمال خطرناک است.
    """
    def __init__(self, state_dim, action_dim, safety_layer_dim, lr=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, buffer_capacity=10000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # شبکه Q و شبکه هدف
        self.q_network = create_q_network(state_dim, action_dim)
        self.target_q_network = create_q_network(state_dim, action_dim)
        self.target_q_network.set_weights(self.q_network.get_weights())

        # بافر PER
        self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity)
        
        # بهینه‌ساز
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # پارامترهای لایه ایمنی
        self.safety_layer_dim = safety_layer_dim
        self.safety_penalty_factor = 1.5

    def act(self, state):
        """
        عمل را بر اساس سیاست اپسیلون-حریص (epsilon-greedy) انتخاب می‌کند.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
            state_tensor = tf.expand_dims(state_tensor, 0)
            q_values = self.q_network(state_tensor)
            return tf.argmax(q_values[0]).numpy()

    def learn(self, batch_size, beta=0.4):
        """
        یادگیری را بر اساس نمونه‌های بافر انجام می‌دهد.
        """
        if len(self.replay_buffer) < batch_size:
            return

        transitions, indices, weights = self.replay_buffer.sample(batch_size, beta)
        batch = Transition(*zip(*transitions))

        # تبدیل به تنسور
        state_batch = tf.convert_to_tensor(np.array(batch.state), dtype=tf.float32)
        action_batch = tf.convert_to_tensor(np.array(batch.action), dtype=tf.int32)
        reward_batch = tf.convert_to_tensor(np.array(batch.reward), dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(np.array(batch.next_state), dtype=tf.float32)
        done_batch = tf.convert_to_tensor(np.array(batch.done), dtype=tf.float32)

        with tf.GradientTape() as tape:
            # محاسبه Q-value فعلی
            q_values_current = tf.gather_nd(self.q_network(state_batch), tf.stack([tf.range(batch_size), action_batch], axis=1))

            # محاسبه Q-value هدف
            next_q_values = tf.reduce_max(self.target_q_network(next_state_batch), axis=1)
            expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
            
            # محاسبه خطاهای TD
            td_errors = expected_q_values - q_values_current
            
            # محاسبه پاداش ایمنی (به عنوان مثال، فشار زیر یک آستانه)
            # این بخش باید بر اساس شرایط ایمنی واقعی در محیط شما پیاده‌سازی شود.
            # برای مثال، فرض می‌کنیم متغیر فشار در اولین بعد از حالت (state) قرار دارد.
            # safety_check_indices = tf.where(state_batch[:, 0] < min_pressure_threshold)
            # safety_penalty = tf.cast(tf.size(safety_check_indices), dtype=tf.float32) * self.safety_penalty_factor

            # برای سادگی، یک جریمه نمونه را اعمال می‌کنیم
            safety_penalty = 0.0

            # ترکیب loss با جریمه ایمنی و وزن‌های IS
            loss = tf.reduce_mean(tf.square(td_errors) * weights) + safety_penalty

        # به روز رسانی اولویت‌های بافر
        self.replay_buffer.update_priorities(indices, td_errors.numpy())
        
        # بهینه‌سازی
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

    def update_target_network(self):
        """
        وزن‌های شبکه هدف را با وزن‌های شبکه Q همگام می‌کند.
        """
        self.target_q_network.set_weights(self.q_network.get_weights())
        
    def decay_epsilon(self):
        """
        مقدار اپسیلون را برای کاهش کاوش تصادفی کاهش می‌دهد.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

class DRLEpanetEnv:
    """
    کلاس محیط (Environment) برای شبیه‌سازی شبکه توزیع آب با استفاده از EPANET.
    **نکته:** شما باید این کلاس را بر اساس مدل شبکه خود (فایل .inp) تکمیل کنید.
    این یک مثال مفهومی است.
    """
    def __init__(self, epanet_inp_file, state_dim, action_dim):
        # اینجا باید EPANET را با استفاده از کتابخانه‌ای مانند epanet-python بارگذاری کنید
        # self.epanet_network = ...
        self.state_dim = state_dim
        self.action_dim = action_dim

    def reset(self):
        """
        محیط را به حالت اولیه بازنشانی می‌کند.
        """
        initial_state = np.random.rand(self.state_dim) # حالت اولیه تصادفی به عنوان مثال
        return initial_state

    def step(self, action):
        """
        یک عمل را در محیط انجام می‌دهد و حالت بعدی، پاداش و وضعیت پایان را برمی‌گرداند.
        """
        # اینجا باید شبیه‌سازی EPANET را یک گام جلو ببرید.
        # state, reward, done, info = self.epanet_network.step(action)
        
        # مقادیر نمونه برای توضیح
        next_state = np.random.rand(self.state_dim)
        reward = random.uniform(0, 1) # پاداش به عنوان مثال
        done = random.random() < 0.05
        info = {}
        
        return next_state, reward, done, info

def plot_results(episode_rewards):
    """
    نمودار پاداش کل در هر اپیزود را رسم می‌کند.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # پارامترهای آموزش
    STATE_DIM = 20 # تعداد متغیرهای حالت (مثلاً فشار، جریان، سطح آب)
    ACTION_DIM = 5 # تعداد اعمال ممکن (مثلاً باز/بسته کردن پمپ‌ها)
    SAFETY_LAYER_DIM = 1 # ابعاد متغیرهای ایمنی در حالت
    
    BATCH_SIZE = 64
    TOTAL_EPISODES = 200
    SYNC_TARGET_NETWORK_INTERVAL = 10 # هر چند گام شبکه هدف به‌روز شود

    # ایجاد محیط و عامل
    # این فایل EPANET فقط یک Placeholder است.
    env = DRLEpanetEnv(epanet_inp_file='net1.inp', state_dim=STATE_DIM, action_dim=ACTION_DIM)
    agent = SafeDQN(STATE_DIM, ACTION_DIM, SAFETY_LAYER_DIM)

    episode_rewards = [] # لیست برای نگهداری پاداش کل هر اپیزود

    # حلقه اصلی آموزش
    for episode in range(TOTAL_EPISODES):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            # افزودن تجربه به بافر با اولویت موقت بالا
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # یادگیری از بافر
            if len(agent.replay_buffer) > BATCH_SIZE:
                agent.learn(BATCH_SIZE)
            
            # به‌روزرسانی شبکه هدف
            if step % SYNC_TARGET_NETWORK_INTERVAL == 0:
                agent.update_target_network()

            state = next_state
            episode_reward += reward
            step += 1
            
        # کاهش اپسیلون برای کاهش کاوش
        agent.decay_epsilon()
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{TOTAL_EPISODES}, Total Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

    # رسم نمودار نتایج پس از پایان آموزش
    plot_results(episode_rewards)
