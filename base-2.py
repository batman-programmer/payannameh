import wntr
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# 1. بارگذاری شبکه شیر دار
# ------------------------------
wn = wntr.network.WaterNetworkModel(r'networks\WDN-\networks\net1_withValve.inp')
sim = wntr.sim.WNTRSimulator(wn)

# ------------------------------
# 2. تنظیمات اولیه شیرها (می‌توان ساعت به ساعت تغییر داد)
# ------------------------------
print("شروع شبیه‌سازی روش مرجع 2 (کنترل ساعت به ساعت)...")

# لیست فشار هدف هر ساعت (مثال: تغییرات فرضی)
target_pressures = [50 + 5*((hour%4)-2) for hour in range(24)]  # نوسان ±10 متر

# ذخیره فشار نودها برای همه ساعت‌ها
pressure_all_hours = []

for hour, target in enumerate(target_pressures):
    # تنظیم initial_setting همه شیرهای PRV برای این ساعت
    for valve_name, valve in wn.valves():
        valve.initial_setting = target
    
    # شبیه‌سازی 1 ساعت
    wn.options.time.duration = (hour+1)*3600
    wn.options.time.hydraulic_timestep = 3600
    results = sim.run_sim()
    
    # گرفتن فشار نودهای ساعت فعلی (آخرین رکورد)
    pressure_hour = results.node['pressure'].iloc[-1]
    pressure_all_hours.append(pressure_hour)
    
    print(f"ساعت {hour+1}: فشار هدف شیرها {target} متر")
    for node in pressure_hour.index:
        print(f"  نود {node}: {pressure_hour[node]:.2f} متر")
    print("-----------------------------")

# ------------------------------
# 3. تبدیل خروجی به DataFrame
# ------------------------------
pressure_df = pd.DataFrame(pressure_all_hours)
pressure_df.index = range(1,25)  # ساعت 1 تا 24

# ------------------------------
# 4. رسم نمودار فشار همه نودها
# ------------------------------
plt.figure(figsize=(12,6))
for node in pressure_df.columns:
    plt.plot(pressure_df.index, pressure_df[node], label=str(node))
plt.xlabel('Hour')
plt.ylabel('Pressure (m)')
plt.title('Valve-Controlled Pressure - Method Reference 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()
