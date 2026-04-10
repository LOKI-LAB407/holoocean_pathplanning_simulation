import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 设置全局字体格式 (符合学术论文标准)
plt.rcParams['font.sans-serif'] = ['Arial'] # 英文论文常用 Arial
plt.rcParams['axes.unicode_minus'] = False

def plot_3d_trajectory(csv_file):
    print(f"正在读取数据: {csv_file}...")
    df = pd.read_csv(csv_file)

    # 2. 创建 3D 图布
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 3. 绘制 ROV 主轨迹 (蓝色实线)
    ax.plot(df['rov_x'], df['rov_y'], df['rov_z'], 
            label='ROV Trajectory', color='#1f77b4', linewidth=2.5, zorder=3)

    # 4. 绘制起点和终点
    start_x, start_y, start_z = df['rov_x'].iloc[0], df['rov_y'].iloc[0], df['rov_z'].iloc[0]
    # 假设你的终点是 (15.0, 15.0, -8.0)，根据实际情况修改
    target_x, target_y, target_z = 15.0, 15.0, -8.0 
    
    ax.scatter(start_x, start_y, start_z, color='green', s=120, label='Start', marker='o', edgecolors='white')
    ax.scatter(target_x, target_y, target_z, color='red', s=200, label='Target', marker='*', edgecolors='white')

    # 5. 绘制静态障碍物 (如果是之前设定的两块固定礁石)
    # 格式: (x, y, z, 半径)
    static_obstacles = [(5.0, 5.0, -5.0, 3.0), (10.0, 10.0, -8.0, 4.0)]
    for (ox, oy, oz, r) in static_obstacles:
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = r * np.cos(u) * np.sin(v) + ox
        y = r * np.sin(u) * np.sin(v) + oy
        z = r * np.cos(v) + oz
        # 画半透明的黄色球体代表石头
        ax.plot_surface(x, y, z, color='yellow', alpha=0.2, edgecolor='orange', linewidth=0.5)

    # 6. 绘制动态障碍物轨迹 (如果 CSV 里有它们的数据)
    obs_cols = [c for c in df.columns if c.startswith('obs_') and c.endswith('_x')]
    for i in range(len(obs_cols)):
        # 提取第 i 个移动障碍物的轨迹
        obs_x = df[f'obs_{i}_x']
        obs_y = df[f'obs_{i}_y']
        obs_z = df[f'obs_{i}_z']
        # 用红色虚线画出移动障碍物的行动路径
        ax.plot(obs_x, obs_y, obs_z, linestyle='--', color='salmon', alpha=0.8, linewidth=1.5, label=f'Moving Obs {i+1} Path')

    # 7. 学术化坐标轴排版
    ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_zlabel('Depth Z (m)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('ROV 3D Evasion Trajectory under Deep RL Control', fontsize=16, fontweight='bold', pad=20)
    
    # 翻转 Z 轴 (因为水下通常是负数，让水面在上面)
    ax.set_zlim(-15, 0)
    
    # 图例设置
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9, edgecolor='black')
    
    # 调整视角 (仰角 25 度，方位角 -45 度，你可以自己微调找到最霸气的角度)
    ax.view_init(elev=25, azim=-45)

    # 8. 导出 300 DPI 超高清图片！
    output_img = 'rov_trajectory_300dpi.png'
    plt.tight_layout()
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"✅ 学术高清图已生成: {output_img}")
    
    # 展示交互窗口
    plt.show()

if __name__ == "__main__":
    plot_3d_trajectory("rov_trajectory_data.csv")