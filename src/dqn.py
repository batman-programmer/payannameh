import gymnasium as gym
from gymnasium import spaces
import numpy as np
import wntr

class WaterNetworkEnv(gym.Env):
    """محیط Gym برای کنترل PRVها در شبکه EPANET با WNTR"""
    
    metadata = {"render.modes": ["human"]}

    def __init__(self, inp_file, min_pressure=25, max_pressure=40):
        super().__init__()
        self.inp_file = inp_file
        self.min_pressure = min_pressure
        self.max_pressure = max_pressure
        self._load_network()
        self.last_action = np.zeros(len(self.prv_links), dtype=np.float32)
        self.last_pressures = None

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

    def reset(self, *, seed=None, options=None):
        """Reset محیط با شبیه‌سازی اولیه"""
        super().reset(seed=seed)
        pressures = self._simulate_hydraulic()
        self.last_action = np.zeros(len(self.prv_links), dtype=np.float32)
        self.last_pressures = pressures
        info = {}
        return pressures, info

    def _simulate_hydraulic(self):
        """اجرای شبیه‌سازی هیدرولیکی واقعی"""
        sim = wntr.sim.WNTRSimulator(self.wn)
        results = sim.run_sim()
        pressures = results.node['pressure'].iloc[0][self.monitor_nodes].values.astype(np.float32)
        self.last_pressures = pressures
        return pressures

    def step(self, action):
        # اعمال اکشن‌ها روی PRVها
        action = np.clip(action, self.action_space.low, self.action_space.high)
        for name, a in zip(self.prv_links, action):
            self.wn.get_link(name).settings = a

        # شبیه‌سازی هیدرولیکی
        pressures = self._simulate_hydraulic()

        # پاداش: فشار در محدوده مطلوب
        reward = np.sum(
            np.where(
                (pressures >= self.min_pressure) & (pressures <= self.max_pressure),
                1.0,
                -0.1 * np.abs(pressures - np.clip(pressures, self.min_pressure, self.max_pressure))
            )
        )

        # جریمه تغییر زیاد PRV
        reward -= 0.01 * np.sum(np.abs(action - self.last_action))
        self.last_action = action.copy()

        done = False
        info = {}
        return pressures, reward, done, False, info

    def render(self, mode="human"):
        if self.last_pressures is None:
            print("شبکه هنوز شبیه‌سازی نشده است.")
            return
        print(f"فشار گره‌ها: {self.last_pressures}")
        print("تنظیمات PRVها:")
        for name, idx in self.prv_indices.items():
            setting = self.wn.get_link(name).setting
            print(f"{name} (index {idx}) = {setting}")


# ===============================
# نمونه اجرا با اکشن تصادفی
# ===============================
if __name__ == "__main__":
    inp_file = r"networks\test.inp"  # مسیر فایل شبکه EPANET
    env = WaterNetworkEnv(inp_file)

    # ریست کردن محیط
    state, info = env.reset()
    env.render()

    # اجرای چند مرحله با اکشن تصادفی
    for step in range(3):
        action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
        print(f"\nمرحله {step+1}:")
        print(f"اکشن اعمال شده: {action}")
        print(f"پاداش: {reward}")
        env.render()
