import numpy as np
import matplotlib.pyplot as plt

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')

# 1. Yaw Bonus & Attitude Bonus (Gaussian Curves)
x_yaw = np.linspace(-60, 60, 500)
y_yaw_bonus = np.exp(-(x_yaw**2) / 400.0)

x_tilt = np.linspace(0, 45, 500)
y_attitude_bonus = np.exp(-(x_tilt**2) / 300.0)
y_tilt_penalty = np.where(x_tilt > 25.0, -(x_tilt - 25.0) * 0.2, 0)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x_yaw, y_yaw_bonus, 'b-', label='Yaw Bonus (Gaussian)')
plt.axvline(0, color='r', linestyle='--', alpha=0.3)
plt.title('Yaw Alignment Bonus')
plt.xlabel('Yaw Error (degrees)')
plt.ylabel('Reward Value')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_tilt, y_attitude_bonus, 'g-', label='Attitude Bonus')
plt.plot(x_tilt, y_tilt_penalty, 'r--', label='Tilt Penalty (>25 deg)')
plt.title('Attitude & Stability Reward')
plt.xlabel('Tilt (Roll/Pitch) (degrees)')
plt.ylabel('Reward Value')
plt.legend()
plt.tight_layout()
plt.savefig('yaw_attitude_rewards.png')

# 2. Sonar Obstacle Penalty (Soft + Hard Seamless)
def sonar_reward(d):
    safe_dist = 3.0
    danger_dist = 1.2
    max_soft = 2.5
    if d < danger_dist:
        return -(max_soft + 30.0 * (danger_dist - d) / danger_dist)
    elif d < safe_dist:
        ratio = (safe_dist - d) / (safe_dist - danger_dist)
        return -max_soft * (ratio ** 2)
    else:
        return 0

d_sonar = np.linspace(0.1, 5.0, 500)
r_sonar = [sonar_reward(d) for d in d_sonar]

plt.figure(figsize=(10, 6))
plt.plot(d_sonar, r_sonar, 'r-', linewidth=2)
plt.axvline(1.2, color='k', linestyle=':', label='Danger Dist (1.2m)')
plt.axvline(3.0, color='g', linestyle=':', label='Safe Dist (3.0m)')
plt.fill_between(d_sonar, r_sonar, color='red', alpha=0.1)
plt.title('Sonar Obstacle Penalty (Continuous Soft-to-Hard)')
plt.xlabel('Distance to Obstacle (m)')
plt.ylabel('Penalty Value')
plt.legend()
plt.grid(True)
plt.savefig('sonar_penalty.png')

# 3. Success Performance Rewards (Linear Funnels)
x_perf = np.linspace(0, 1.2, 500)
y_rot_perf = np.clip(1.0 - x_perf / 0.5, 0.0, 1.0) * 100
y_speed_perf = np.clip(1.0 - x_perf / 1.0, 0.0, 1.0) * 100

plt.figure(figsize=(10, 6))
plt.plot(x_perf, y_rot_perf, label='Rotation Perf (Max 100, Scale 0.5)')
plt.plot(x_perf, y_speed_perf, label='Speed Perf (Max 100, Scale 1.0)')
plt.title('Terminal Performance Bonuses (Funnel Effect)')
plt.xlabel('Error / Speed Magnitude')
plt.ylabel('Extra Reward')
plt.legend()
plt.savefig('performance_funnels.png')

# 4. Moving Multiplier & Yaw Penalty Factor
x_progress = np.linspace(-0.001, 0.004, 500)
y_moving_mult = np.clip(x_progress / 0.002, 0.0, 1.0)

x_sonar_factor = np.linspace(1.0, 5.0, 500)
y_yaw_penalty_factor = np.clip(1.5 - 1.3 * (4.0 - x_sonar_factor) / (4.0 - 1.2), 0.2, 1.5)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x_progress, y_moving_mult, 'orange')
plt.title('Moving Multiplier (Anti-Idling)')
plt.xlabel('Progress (m/step)')
plt.ylabel('Multiplier (0 to 1)')

plt.subplot(1, 2, 2)
plt.plot(x_sonar_factor, y_yaw_penalty_factor, 'purple')
plt.title('Dynamic Yaw Penalty Factor')
plt.xlabel('Sonar Distance (m)')
plt.ylabel('Penalty Multiplier (Scale for Yaw Change)')
plt.tight_layout()
plt.savefig('multipliers_factors.png')