import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 学术论文全局绘图格式设置
# ==========================================
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'


def plot_2_5d_ocean_current():
    # ==========================================
    # 2. 仿真空间网格设置
    # ==========================================
    map_size = 25.0
    grid_res = 150  # 网格分辨率
    x = np.linspace(-map_size, map_size, grid_res)
    y = np.linspace(-map_size, map_size, grid_res)
    X, Y = np.meshgrid(x, y)

    # ==========================================
    # 3. 严格对齐训练代码参数的 6 个漩涡
    # 物理约束：eta在[3,8], xi在[3,8], z0在[-35,-2], sigma_z在[3,15]
    # ==========================================
    vortices = [
        {'x0': 10.0, 'y0': 10.0, 'z0': -4.0, 'eta': 7.5, 'xi': 6.0, 'sigma_z': 6.0},  # V1: 浅水强涡
        {'x0': -12.0, 'y0': 8.0, 'z0': -15.0, 'eta': -6.5, 'xi': 4.5, 'sigma_z': 10.0},  # V2: 中层逆涡
        {'x0': -8.0, 'y0': -15.0, 'z0': -30.0, 'eta': 6.0, 'xi': 5.0, 'sigma_z': 12.0},  # V3: 深水大涡
        {'x0': 15.0, 'y0': -12.0, 'z0': -8.0, 'eta': -8.0, 'xi': 7.0, 'sigma_z': 5.0},  # V4: 浅水烈涡
        {'x0': 0.0, 'y0': 5.0, 'z0': -22.0, 'eta': 4.0, 'xi': 3.5, 'sigma_z': 8.0},  # V5: 中深层干扰
        {'x0': 5.0, 'y0': -5.0, 'z0': -3.0, 'eta': -4.5, 'xi': 7.5, 'sigma_z': 4.0}  # V6: 水面宽涡
    ]

    # 切片深度设定：浅水、中水、深水
    z_levels = [-5.0, -15.0, -30.0]
    MAX_CURRENT = 0.65  # 物理限速锁

    # 创建 1x3 的并排画板
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=300)
    fig.suptitle('2.5D Ocean Current Layered Slices\n(6 Vortices, Max Speed Clamped to 0.65 m/s)',
                 fontsize=18, fontweight='bold', y=1.05)

    for idx, z in enumerate(z_levels):
        ax = axes[idx]
        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        # 向量化计算 6 个漩涡在当前深度的叠加流场
        for v in vortices:
            dx = X - v['x0']
            dy = Y - v['y0']
            r_sq = np.clip(dx ** 2 + dy ** 2, 1e-4, None)

            decay_factor = (1.0 - np.exp(-r_sq / (v['xi'] ** 2))) / (2 * np.pi * r_sq)
            z_decay = np.exp(-((z - v['z0']) ** 2) / (v['sigma_z'] ** 2))

            U += -v['eta'] * dy * decay_factor * z_decay
            V += v['eta'] * dx * decay_factor * z_decay

        # 全局底流漂移叠加
        depth_ratio = np.clip(1.0 - abs(z) / 40.0, 0.0, 1.0)
        global_drift_x = 0.05 * depth_ratio
        global_drift_y = 0.03 * depth_ratio
        U += global_drift_x
        V += global_drift_y

        # 绝对物理安全锁计算
        Speed = np.sqrt(U ** 2 + V ** 2)
        exceed_mask = Speed > MAX_CURRENT
        if np.any(exceed_mask):
            scale = MAX_CURRENT / Speed[exceed_mask]
            U[exceed_mask] *= scale
            V[exceed_mask] *= scale
            Speed[exceed_mask] = MAX_CURRENT

        # 1. 绘制速度热力图底色 (统一色标范围 0 到 0.7)
        contour = ax.contourf(X, Y, Speed, levels=np.linspace(0, 0.7, 30), cmap='YlGnBu', extend='max')

        # 2. 绘制流线图
        lw = 1.0 + 1.5 * (Speed / MAX_CURRENT)  # 根据流速动态调整线粗
        ax.streamplot(X, Y, U, V, color='k', linewidth=lw, density=1.2, arrowsize=1.2, arrowstyle='->')

        # 3. 智能标注漩涡中心
        for v_idx, v in enumerate(vortices):
            # 计算该漩涡在当前水层的衰减比例
            v_z_decay = np.exp(-((z - v['z0']) ** 2) / (v['sigma_z'] ** 2))

            # 只有当该漩涡对当前层产生 >10% 的影响力时，才绘制它的星星
            if v_z_decay > 0.1:
                ax.scatter(v['x0'], v['y0'], color='red', marker='*', s=200 * v_z_decay,
                           edgecolors='white', alpha=v_z_decay, zorder=5)
                # 标注编号和真实的 Z0 深度
                ax.text(v['x0'] + 1, v['y0'] + 1, f"V{v_idx + 1}({v['z0']}m)", color='darkred',
                        fontsize=10, alpha=v_z_decay, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=v_z_decay * 0.7, edgecolor='none', boxstyle='round,pad=0.1'))

        # 4. 子图格式化
        ax.set_title(f'Depth Z = {z} m\n(Max V = {np.max(Speed):.2f} m/s)', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position (m)', fontsize=13, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Y Position (m)', fontsize=13, fontweight='bold')
        ax.set_xlim(-map_size, map_size)
        ax.set_ylim(-map_size, map_size)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_aspect('equal')  # 保证 X 和 Y 轴比例为 1:1，圆不会变成椭圆

    # 5. 添加全局 Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(contour, cax=cbar_ax)
    cbar.set_label('Current Velocity Magnitude (m/s)', fontsize=14, fontweight='bold')
    cbar.set_ticks(np.arange(0, 0.8, 0.1))

    # 保存与显示
    plt.subplots_adjust(right=0.9, wspace=0.15)
    output_filename = 'ocean_current_2_5d_slices_aligned.png'
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"✅ 2.5D 海流切片图已生成，已保存为: {output_filename}")

    plt.show()


if __name__ == "__main__":
    plot_2_5d_ocean_current()