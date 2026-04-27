import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. 读取我们刚保存的 CSV 数据
df = pd.read_csv("rov_trajectory_comparison.csv")

# 2. 创建 3D 图表 (用于对比轨迹)
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title("3D Trajectory Tracking: PPO vs DE-Planner")
# 画出绿色参考线 (虚线)
ax1.plot(df['ref_x'], df['ref_y'], df['ref_z'], color='green', linestyle='--', linewidth=2, label='DE Reference Path')
# 画出红色实际线 (实线)
ax1.plot(df['x'], df['y'], df['z'], color='red', linewidth=2, label='PPO Actual Path')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_zlabel('Depth Z (m)')
ax1.legend()

# 3. 创建 2D 折线图 (用于量化误差)
ax2 = fig.add_subplot(122)
ax2.set_title("Tracking Error over Time (Steps)")
ax2.plot(df['step'], df['tracking_error'], color='blue', linewidth=1.5)
ax2.set_xlabel('Simulation Steps')
ax2.set_ylabel('Distance Error (m)')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.axhline(y=df['tracking_error'].mean(), color='orange', linestyle='-', label=f"Mean Error: {df['tracking_error'].mean():.2f}m")
ax2.legend()

plt.tight_layout()
plt.show()