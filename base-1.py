import wntr
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# 1. بارگذاری شبکه شیر دار
# ------------------------------
wn = wntr.network.WaterNetworkModel(r'networks\WDN-\networks\net1_withValve.inp')
sim = wntr.sim.WNTRSimulator(wn)

# ------------------------------
# 2. تنظیم initial_setting برای شیرهای PRV
# ------------------------------
for valve_name, valve in wn.valves():
    valve.initial_setting = 50  # فشار هدف
print("شیرهای PRV روی فشار هدف 50 متر تنظیم شدند.\n")

# ------------------------------
# 3. تنظیم شبیه‌سازی
# ------------------------------
wn.options.time.duration = 24*3600       # 24 ساعت
wn.options.time.hydraulic_timestep = 3600  # 1 ساعت

# ------------------------------
# 4. اجرای شبیه‌سازی
# ------------------------------
print("شروع شبیه‌سازی شبکه...")
results = sim.run_sim()
print("شبیه‌سازی کامل شد!\n")

# ------------------------------
# 5. گرفتن فشار نودها
# ------------------------------
pressure = results.node['pressure']  # DataFrame: index = زمان (ثانیه)، ستون = نام نودها

# ------------------------------
# 6. چاپ متنی فشار همه نودها برای 24 ساعت
# ------------------------------
print("فشار همه نودها در طول 24 ساعت (ساعت: فشار نودها):")
for hour in range(24):
    pressures_hour = pressure.iloc[hour]
    print(f"ساعت {hour+1}:")
    for node in pressure.columns:
        print(f"  نود {node}: {pressures_hour[node]:.2f} متر")
    print("-----------------------------")

# ------------------------------
# 7. رسم نمودار فشار نودها
# ------------------------------
plt.figure(figsize=(12,6))
for node in pressure.columns:
    plt.plot(pressure.index / 3600, pressure[node], label=str(node))
plt.xlabel('Hour')
plt.ylabel('Pressure (m)')
plt.title('Valve-Controlled Pressure - Method Reference 1')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()
