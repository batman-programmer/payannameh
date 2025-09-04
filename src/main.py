import wntr
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

class WaterNetworkEnv(gym.Env):
    """
    محیط Gym برای کنترل PRVها در شبکه EPANET با wntr
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, inp_file, min_pressure=25, max_pressure=40):
        super().__init__()
        self.inp_file = inp_file
        self.min_pressure = min_pressure
        self.max_pressure = max_pressure
        self._load_network()

    def _load_network(self):
        """بارگذاری شبکه و شناسایی PRVها و junctionها"""
        self.wn = wntr.network.WaterNetworkModel(self.inp_file)

        self.prv_links = []
        for name, link in self.wn.links():
            # کلاس لینک PRV: wntr.network.controls.PRValve
            if link.__class__.__name__ == "PRValve":
                self.prv_links.append(name)

        print("PRVهای شبکه:", self.prv_links)

        if len(self.prv_links) == 0:
            raise ValueError("هیچ PRV در شبکه پیدا نشد!")

        # نگهداری ایندکس PRVها برای دسترسی راحت
        self.prv_indices = {name: i for i, name in enumerate(self.prv_links)}

        # شناسایی junctionها
        self.monitor_nodes = list(self.wn.junction_name_list)

        # فضای عمل و حالت
        self.action_space = spaces.Box(
            low=np.array([10.0]*len(self.prv_links)),
            high=np.array([50.0]*len(self.prv_links)),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=100.0,
            shape=(len(self.monitor_nodes),),
            dtype=np.float32
        )

        # آخرین اکشن برای جریمه تغییر زیاد
        self.last_action = np.zeros(len(self.prv_links), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        """Reset محیط با شبیه‌سازی اولیه"""
        super().reset(seed=seed)  # اگر بخواهیم از seed استفاده کنیم
        self._load_network()
        pressures = self._simulate_network()
        info = {}  # You can add relevant info here if needed
        return pressures, info

    def _simulate_network(self):
        """اجرای شبیه‌سازی هیدرولیکی با wntr"""
        sim = wntr.sim.WNTRSimulator(self.wn)
        results = sim.run_sim()
        pressures = results.node['pressure'].iloc[0][self.monitor_nodes].values.astype(np.float32)
        return pressures

    def step(self, action):
        # اعمال اکشن‌ها روی PRVها
        action = np.clip(action, self.action_space.low, self.action_space.high)
        for name, a in zip(self.prv_links, action):
            self.wn.get_link(name).settings = a

        # شبیه‌سازی شبکه
        pressures = self._simulate_network()

        # پاداش: فشار در محدوده مطلوب
        reward = 0
        for p in pressures:
            if self.min_pressure <= p <= self.max_pressure:
                reward += 1.0
            else:
                reward -= abs(p - np.clip(p, self.min_pressure, self.max_pressure))*0.1

        # جریمه تغییر زیاد PRV
        reward -= 0.01 * np.sum(np.abs(action - self.last_action))
        self.last_action = action.copy()

        done = False
        info = {}  # You can add relevant info here if needed
        return pressures, reward, done, False, info # Updated return signature for gymnasium

    def render(self, mode="human"):
        pressures = self._simulate_network()
        print(f"فشار گره‌ها: {pressures}")
        print("تنظیمات PRVها:")
        for name, idx in self.prv_indices.items():
            setting = self.wn.get_link(name).settings
            print(f"{name} (index {idx}) = {setting}")


if __name__ == "__main__":
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    import logging

    # -----------------------------
    # تنظیم لاگر
    # -----------------------------
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # -----------------------------
    # Callback برای چاپ لاگ پیشرفته
    # -----------------------------
    class AdvancedLoggerCallback(BaseCallback):
        def __init__(self, env, verbose=0):
            super().__init__(verbose)
            self.env = env
            self.prev_prv_values = None

        def _on_step(self) -> bool:
            obs = self.locals['new_obs']
            actions = self.locals['actions']
            reward = self.locals['rewards']

            pressures = obs  # فرض: obs = آرایه فشار گره‌ها
            mean_pressure = np.mean(pressures)
            out_of_bounds = np.sum((pressures < self.env.min_pressure) | (pressures > self.env.max_pressure))

            # تغییرات PRV نسبت به مرحله قبل
            if self.prev_prv_values is not None:
                prv_change = np.array(actions) - np.array(self.prev_prv_values)
            else:
                prv_change = np.zeros(len(actions))

            self.prev_prv_values = actions

            logger.info(f"Step {self.n_calls}: mean_pressure={mean_pressure:.2f}, "
                        f"out_of_bounds={out_of_bounds}, PRV_change={prv_change}, reward={reward}")

            return True

    # -----------------------------
    # فایل شبکه و محیط
    # -----------------------------
    inp_file = "networks\\2PRV.inp"
    env = WaterNetworkEnv(inp_file, min_pressure=5, max_pressure=9)

    obs = env.reset()
    logger.info(f"مشاهده اولیه شبکه: {obs}")
    logger.info(f"PRVهای شبکه: {env.prv_links}")

    # -----------------------------
    # Callback ذخیره مدل
    # -----------------------------
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="./models/", name_prefix="ppo_wntr")
    advanced_logger = AdvancedLoggerCallback(env)

    # -----------------------------
    # تعریف مدل PPO بهینه
    # -----------------------------
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0001,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        verbose=0
    )

    # -----------------------------
    # آموزش مدل با لاگ پیشرفته
    # -----------------------------
    logger.info("شروع آموزش مدل PPO روی شبکه WNTR")
    model.learn(total_timesteps=20000, callback=[checkpoint_callback, advanced_logger])
    logger.info("آموزش پایان یافت")

    model.save("ppo_wntr_prv_advanced_log")

    # -----------------------------
    # تست مدل با لاگ پیشرفته
    # -----------------------------
    obs = env.reset()
    done = False
    step_count = 0
    prev_prv_values = [0.0]*len(env.prv_links)  # مقدار اولیه PRV
    logger.info("شروع تست مدل آموزش دیده")
    while not done and step_count < 20:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        pressures = obs
        mean_pressure = np.mean(pressures)
        out_of_bounds = np.sum((pressures < env.min_pressure) | (pressures > env.max_pressure))
        prv_change = np.array(action) - np.array(prev_prv_values)
        prev_prv_values = action

        logger.info(f"Test Step {step_count}: mean_pressure={mean_pressure:.2f}, "
                    f"out_of_bounds={out_of_bounds}, PRV_change={prv_change}, reward={reward}")

        env.render()
        step_count += 1
    logger.info("تست مدل پایان یافت")