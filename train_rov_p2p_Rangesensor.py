import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import holoocean
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# ==========================================
# 0. 保护 C 盘与引擎路径配置
# ==========================================
os.environ["HOLOOCEAN_PATH"] = "D:\\HoloOceanEngine"
os.environ["HOLOOCEAN_SYSTEM_PATH"] = "D:\\HoloOceanEngine"

def get_rov_mixing_matrix():
    """ROV 8个底层推进器与 6个高层自由度的推力映射矩阵"""
    M = np.zeros((8, 6))
    M[0, :] = np.array([0,  0,    1,   -1,    1,    0])  
    M[1, :] = np.array([0,  0,    1,    1,    1,    0])  
    M[2, :] = np.array([0,  0,    1,   -1,   -1,    0])  
    M[3, :] = np.array([0,  0,    1,    1,   -1,    0])  
    M[4, :] = np.array([1, -1,    0,    0,    0,   -1])  
    M[5, :] = np.array([1,  1,    0,    0,    0,    1])  
    M[6, :] = np.array([1,  1,    0,    0,    0,   -1])  
    M[7, :] = np.array([1, -1,    0,    0,    0,    1])  
    return M

# ==========================================
# 1. HoloOcean 蓝图配置 (注意：这里确实没有 RangeSensor)
# ==========================================
rov_config = {
    "name": "ROV_P2P_Dynamic_Task",
    "world": "OpenWater",
    "package_name": "Ocean", 
    "main_agent": "my_rov",
    "frames_per_sec": 30, 
    "agents": [
        {
            "agent_name": "my_rov",
            "agent_type": "HoveringAUV",
            "control_scheme": 0, 
            "location": [0.0, 0.0, -5.0], 
            "sensors": [
                {"sensor_type": "LocationSensor"}, 
                {"sensor_type": "VelocitySensor"}, 
                {"sensor_type": "RotationSensor"}  
                # 🚫 为什么这里没有 RangeSensor？
                # 因为我们要检测的移动障碍物是用代码画出来的虚拟物品，没有物理体积。
                # 所以我们要在 Python 里用纯数学方法(self._get_simulated_sonar)来模拟声呐射线！
            ]
        }
    ]
}

# ==========================================
# 2. 核心环境 Wrapper：含虚拟声呐与移动障碍物
# ==========================================
class ROVP2PDynamicWrapper(gym.Env):
    def __init__(self, config):
        super(ROVP2PDynamicWrapper, self).__init__()
        self.holo_env = holoocean.make(scenario_cfg=config)
        
        # 动作空间：6个自由度的期望推力
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        
        # 🌟 状态空间：9 (相对坐标+相对速度+姿态) + 5 (虚拟声呐测距) = 14维
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
        
        # 环境与物理参数初始化
        self.tam_inverse = get_rov_mixing_matrix()
        self.max_steps = 2000  
        self.current_step = 0
        self.max_tilt_rad = 1.0        # 最大允许倾角(翻车红线)
        self.water_surface_z = -0.5    # 水面高度(越界红线)
        self.water_bottom_z = -40.0    # 海底深度(越界红线)
        
        # 记录缓存变量
        self.target_pos = np.zeros(3)
        self.prev_waypoint = np.zeros(3)
        self.previous_distance = 0.0
        self.previous_action = np.zeros(6) 
        self.current_disturbance = np.zeros(8) 
        self.max_current_power = 5.0   # 洋流强度

        # ========================================
        # 📡 工业级多波束前视声呐 (FLS) 硬件参数
        # ========================================
        self.num_sonar_rays = 5      
        self.sonar_max_range = 30.0  # 真实量程：30米
        self.sonar_min_range = 0.5   # 硬件盲区：0.5米以内无法成像
        
        # 水平视角 120° (离散化为 5 根波束)
        self.sonar_angles = np.deg2rad([-60, -30, 0, 30, 60]) 
        
        # 垂直波束角 20° (极其重要：决定了声呐的上下扫描厚度)
        self.vertical_fov = np.deg2rad(20)

        self.clean_sonar_ranges = np.ones(self.num_sonar_rays) * self.sonar_max_range
    # ----------------------------------------
    # 辅助功能 1：生成移动障碍物 (在 reset 中调用)
    # ----------------------------------------
    def _generate_dynamic_obstacles(self):
        """在地图中生成几条随机游动的 '虚拟机械鱼'"""
        self.dynamic_obs = []
        num_obstacles = np.random.randint(3, 6) # 每回合随机生成 3 到 5 个
        
        for _ in range(num_obstacles):
            # 把障碍物散布在起点和终点之间的必经之路上
            pos = self.target_pos * np.random.uniform(0.2, 0.8) 
            pos += np.random.uniform(-5.0, 5.0, 3) 
            pos[2] = np.clip(pos[2], self.water_bottom_z + 2, self.water_surface_z - 2)
            
            # 给它们赋予随机的巡游速度 (0.3 ~ 0.8 m/s)
            vel = np.random.uniform(-1.0, 1.0, 3)
            vel[2] *= 0.2 # 垂直方向运动慢一点
            vel = (vel / np.linalg.norm(vel)) * np.random.uniform(0.3, 0.8)
            
            radius = np.random.uniform(0.5, 1.5) # 大小随机
            
            # 存入列表
            self.dynamic_obs.append({'pos': pos, 'vel': vel, 'radius': radius})

    # ----------------------------------------
    # 辅助功能 2：更新障碍物位置 (在 step 中调用)
    # ----------------------------------------
    def _update_dynamic_obstacles(self):
        """计算障碍物的移动，并在画面中渲染出来"""
        dt = 1.0 / 30.0 # 假设仿真频率是 30Hz
        try:
            for obs in self.dynamic_obs:
                obs['pos'] += obs['vel'] * dt # 距离 = 速度 * 时间
                
                # 如果碰到空气墙（30米边界），就反弹回去
                for axis in range(3):
                    if obs['pos'][axis] > 30.0 or obs['pos'][axis] < -30.0:
                        obs['vel'][axis] *= -1
                
                # 🌟 引擎画图：在 3D 画面中画出橙色方块代表这些障碍物
                self.holo_env.draw_box(
                    center=obs['pos'].tolist(), 
                    extent=[obs['radius'], obs['radius'], obs['radius']], 
                    color=[255, 165, 0], 
                    lifetime=0.1
                )
        except Exception:
            pass

    # ----------------------------------------
    # 辅助功能 3：📡 纯数学虚拟声呐探测 (极其关键)
    # ----------------------------------------
    def _get_simulated_sonar(self, current_pos, roll, pitch, yaw):
        """
        基于 3D 姿态矩阵与圆锥波束近似的工业级声呐模拟器
        """
        ranges = np.ones(self.num_sonar_rays) * self.sonar_max_range
        
        # 1. 构建 ROV 机体到世界坐标系的 3D 旋转矩阵 (ZYX 顺序)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        
        R_yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        R_roll = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        R_body2world = R_yaw @ R_pitch @ R_roll # 总旋转矩阵

        # 计算垂直波束一半视角的正切值，用于后续计算波束厚度
        tan_half_vfov = np.tan(self.vertical_fov / 2.0)

        for i, angle in enumerate(self.sonar_angles):
            # 射线在机体坐标系下的方向 (假设声呐水平安装)
            ray_dir_body = np.array([np.cos(angle), np.sin(angle), 0.0])
            
            # 🌟 核心升级 1：射线随着 ROV 的抬头、翻滚一起在 3D 空间中旋转！
            ray_dir_world = R_body2world @ ray_dir_body

            for obs in self.dynamic_obs:
                # 障碍物圆心到 ROV 的向量
                V = current_pos - obs['pos'] 
                
                # 计算障碍物在射线方向上的投影距离
                t_approx = np.dot(-V, ray_dir_world)

                if t_approx > 0: # 障碍物在射线前方
                    # 🌟 核心升级 2：圆锥波束模拟 (Cone Approximation)
                    # 声呐波束越远越宽。我们在数学上等效为：把远处的障碍物“放大”。
                    # 这样即使障碍物中心不在射线直线上，只要被声呐的“厚度”扫到，也能检测出来！
                    r_eff = obs['radius'] + t_approx * tan_half_vfov

                    # 标准 3D 射线-球体相交算法 (Ray-Sphere Intersection)
                    a = 1.0 # ray_dir_world 是单位向量
                    b = 2.0 * np.dot(V, ray_dir_world)
                    c = np.dot(V, V) - r_eff**2

                    discriminant = b**2 - 4*c
                    if discriminant > 0: # 射线穿透了膨胀后的球体
                        # 求最近的交点距离
                        t = (-b - np.sqrt(discriminant)) / 2.0

                        # 🌟 核心升级 3：盲区与量程限制
                        if t < self.sonar_min_range:
                            dist = self.sonar_min_range # 落入盲区，返回极小值触发警报
                        else:
                            dist = t

                        if 0 < dist < ranges[i]:
                            ranges[i] = dist

        return ranges

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.previous_action = np.zeros(6) 
        
        # 随机生成本回合目标点
        rx = np.random.uniform(-20.0, 20.0)
        ry = np.random.uniform(-20.0, 20.0)
        rz = np.random.uniform(-15.0, -2.0)
        self.target_pos = np.array([rx, ry, rz])
        
        # 随机洋流
        self.current_disturbance = np.random.uniform(-self.max_current_power, self.max_current_power, size=(8,))
        
        obs_dict = self.holo_env.reset()
        self.current_obs_dict = obs_dict
        
        initial_pos = np.array(obs_dict["LocationSensor"])
        
        # 初始化用于计算 CTE 偏航惩罚的上一航点
        self.prev_waypoint = initial_pos.copy() 
        self.previous_distance = np.linalg.norm(self.target_pos - initial_pos)
        
        # 🐟 调用辅助功能：生成水下的移动障碍物
        self._generate_dynamic_obstacles()
        
        try:
            # 画出终点和基准引导线
            self.holo_env.draw_box(center=self.target_pos.tolist(), extent=[0.5, 0.5, 0.5], color=[255, 0, 0], lifetime=0)
            self.holo_env.draw_line(start=initial_pos.tolist(), end=self.target_pos.tolist(), color=[0, 255, 0], lifetime=0)
        except Exception:
            pass 
            
        return self._get_obs(obs_dict), {}

    def step(self, action):
        self.current_step += 1
        
        # 🐟 1. 刷新移动障碍物的位置并在屏幕上画出来
        self._update_dynamic_obstacles()
        
        # ========================================
        # 2. DVL 底层补偿逻辑
        # ========================================
        dvl_velocity_world = np.array(self.current_obs_dict["VelocitySensor"]) 
        roll_deg, pitch_deg, yaw_deg = self.current_obs_dict["RotationSensor"]
        roll, pitch, yaw_rad = np.deg2rad(roll_deg), np.deg2rad(pitch_deg), np.deg2rad(yaw_deg)

        cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
        vx_world, vy_world, vz_world = dvl_velocity_world[0], dvl_velocity_world[1], dvl_velocity_world[2]
        vx_body = vx_world * cos_y + vy_world * sin_y  
        vy_body = -vx_world * sin_y + vy_world * cos_y  
        vz_body = vz_world                              
        dvl_velocity_body = np.array([vx_body, vy_body, vz_body])
        
        dvl_compensation_6d = np.zeros(6)
        dvl_compensation_6d[0:3] = -dvl_velocity_body * 2.5 
        combined_action = action + dvl_compensation_6d / 50.0 
        
        # ========================================
        # 3. 动力分配与物理执行
        # ========================================
        thruster_commands = np.dot(self.tam_inverse, combined_action)
        max_thrust = np.max(np.abs(thruster_commands))
        if max_thrust > 1.0: thruster_commands = thruster_commands / max_thrust
            
        actual_physics_force = np.clip(thruster_commands * 50.0 + self.current_disturbance, -100.0, 100.0)
        obs_dict_new = self.holo_env.step(actual_physics_force.tolist())
        self.current_obs_dict = obs_dict_new  
        
        # ========================================
        # 4. 提取状态指标，为计算 Reward 做准备
        # ========================================
        current_pos = np.array(obs_dict_new["LocationSensor"])
        speed = np.linalg.norm(np.array(obs_dict_new["VelocitySensor"]))
        roll_new, pitch_new, yaw_new = np.deg2rad(obs_dict_new["RotationSensor"])
        
        dx = self.target_pos[0] - current_pos[0]
        dy = self.target_pos[1] - current_pos[1]
        target_yaw = np.arctan2(dy, dx)
        yaw_error = (target_yaw - yaw_new + np.pi) % (2 * np.pi) - np.pi
        distance_to_target = np.linalg.norm(self.target_pos - current_pos)
        

        # 将声呐结果存入缓存
        self.clean_sonar_ranges = self._get_simulated_sonar(current_pos, roll_new, pitch_new, yaw_new)
        min_sonar_dist = np.min(self.clean_sonar_ranges)
        # # 📡 调用虚拟声呐，获取当前 5 根射线的探测距离
        # current_sonar = self._get_simulated_sonar(current_pos, yaw_new)
        # min_sonar_dist = np.min(current_sonar)
        
        # ========================================
        # 5. 极度复杂的综合奖励函数体系 (Reward)
        # ========================================
        reward = 0.0
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # [A] 目标进度奖励
        progress = self.previous_distance - distance_to_target
        reward += progress * 10.0  
        self.previous_distance = distance_to_target 
        reward += np.cos(yaw_error) * 0.5 
        
        # [B] 姿态维稳奖励
        attitude_score = np.exp(- (roll_new**2 + pitch_new**2) / 0.1)
        roll_penalty = max(0.0, abs(np.rad2deg(roll_new)) - 5.0) * 0.5
        pitch_penalty = max(0.0, abs(np.rad2deg(pitch_new)) - 5.0) * 0.5
        reward += (attitude_score * 2.0) - (roll_penalty + pitch_penalty) * 1.0
        
        # [C] 严格限速惩罚 (保证慢速行驶，稳定姿态)
        target_speed_limit = 0.5 # 超过 0.5 m/s 就开始扣分
        if speed > target_speed_limit:
            reward -= (speed - target_speed_limit) * 5.0 
            
        reward -= np.sum(np.square(action)) * 0.05
        reward -= np.sum(np.square(action - self.previous_action)) * 0.1
        self.previous_action = np.copy(action)

        # [D] 🚨 声呐动态斥力墙与偏航橡皮筋
        safe_dist = 2.0 # 2米内拉响警报
        if min_sonar_dist < safe_dist:
            danger_ratio = (safe_dist - min_sonar_dist) / safe_dist
            reward -= 10.0 * (danger_ratio ** 2) # 二次抛物线爆炸扣分
            
        # 偏航惩罚 (CTE)：躲过障碍后，逼它回到原始航线上
        line_vec = self.target_pos - self.prev_waypoint
        line_length = np.linalg.norm(line_vec)
        if line_length > 0.1:
            rov_vec = current_pos - self.prev_waypoint
            cte_distance = np.linalg.norm(np.cross(line_vec, rov_vec)) / line_length
            if cte_distance > 1.0: # 偏离航线超过 1 米扣分
                reward -= (cte_distance - 1.0) * 2.0

        # ========================================
        # 6. 致命边界判定
        # ========================================
        if self.current_step > 10: 
            crash_threshold = 1.0 # ROV 自身大概半径
            if min_sonar_dist < crash_threshold:
                reward -= 1000.0
                terminated = True
                print(f"💥 [重启] 与移动障碍物发生物理碰撞！")
                
            elif abs(roll_new) > self.max_tilt_rad or abs(pitch_new) > self.max_tilt_rad:
                reward -= 1000.0
                terminated = True
                print(f"💀 [重启] 姿态翻车!")
                
            elif current_pos[2] > self.water_surface_z or current_pos[2] < self.water_bottom_z:
                reward -= 1000.0
                terminated = True
                print(f"💀 [重启] 冲出水面或沉入海底!")
                
            elif distance_to_target < 0.5:
                # 必须满足姿态平稳、速度慢，才算完美通关
                if abs(roll_new) < 0.2 and abs(pitch_new) < 0.2 and speed < 0.5:
                    reward += 1000.0 
                    print(f"✅ 完美到达！耗时 {self.current_step} 步")
                else:
                    reward += 500.0  
                terminated = True
            
        return self._get_obs(obs_dict_new), reward, terminated, truncated, {}

    def _get_obs(self, obs_dict):
        """将传感器数据拼接为 14 维的神经网络输入状态"""
        current_pos = np.array(obs_dict["LocationSensor"])
        velocity_world = np.array(obs_dict["VelocitySensor"])
        roll, pitch, yaw = np.deg2rad(obs_dict["RotationSensor"])

        dx = self.target_pos[0] - current_pos[0]
        dy = self.target_pos[1] - current_pos[1]
        dz = self.target_pos[2] - current_pos[2]
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)

        # 1. 相对目标坐标 (3维)
        rel_x_body = dx * cos_y + dy * sin_y   
        rel_y_body = -dx * sin_y + dy * cos_y  
        rel_z_body = dz                        
        relative_pos_body = np.array([rel_x_body, rel_y_body, rel_z_body])

        # 2. 机体相对速度 (3维)
        v_surge = velocity_world[0] * cos_y + velocity_world[1] * sin_y  
        v_sway = -velocity_world[0] * sin_y + velocity_world[1] * cos_y 
        v_heave = velocity_world[2]
        velocity_body = np.array([v_surge, v_sway, v_heave])

        # 3. 姿态与偏航误差 (3维)
        target_yaw = np.arctan2(dy, dx)
        yaw_error = (target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
        rotation_body = np.array([roll, pitch, yaw_error])

        # 4. 📡 核心：读取虚拟声呐距离，并归一化为 0 到 1 的小数 (5维)
        # 🌟 直接使用 step 中算好的干净数据，不用再算一遍了！
        # 如果是 reset 调用的，就实时算一次；如果是 step 调用的，直接用缓存
        if self.current_step == 0:
            sonar_ranges = self._get_simulated_sonar(current_pos, roll, pitch, yaw)
        else:
            sonar_ranges = self.clean_sonar_ranges.copy()
        # sonar_ranges = self._get_simulated_sonar(current_pos, roll, pitch, yaw)
        # 🌟 域随机化 (Sim-to-Real 的核心秘籍)
        # 1. 注入高斯环境噪声 (模拟水下杂波，正负 0.2 米的误差)
        noise = np.random.normal(0, 0.2, size=self.num_sonar_rays)
        sonar_ranges += noise
        
        # 2. 模拟偶发的传感器丢包或死机 (2%的概率突然读不到数据，返回最大量程)
        if np.random.rand() < 0.02:
            sonar_ranges = np.ones(self.num_sonar_rays) * self.sonar_max_range
            
        # 截断以防越界
        sonar_ranges = np.clip(sonar_ranges, 0.1, self.sonar_max_range)
        
        normalized_sonar = sonar_ranges / self.sonar_max_range

        # 拼接返回！共 14 维
        state = np.concatenate([relative_pos_body, velocity_body, rotation_body, normalized_sonar])
        return state.astype(np.float32)

    def close(self):
        pass

# ==========================================
# 3. 训练启动入口
# ==========================================
if __name__ == "__main__":
    print("🌟 正在初始化 [带声呐与移动避障] 的全新强化学习环境...")
    env = ROVP2PDynamicWrapper(rov_config)
    
    print("🧠 状态空间升维至 14，正在孕育全新的 PPO 模型...")
    # 从零开始训练新大脑！
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003, 
        tensorboard_log="./rov_tensorboard/"
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path='./rov_models/',
        name_prefix='ppo_rov_dynamic_v1' 
    )
    
    print("🔥 开启全境动态避障训练！")
    
    model.learn(
        total_timesteps=2000000, 
        callback=checkpoint_callback, 
        tb_log_name="PPO_Dynamic_Sonar_1" 
    )
    
    print("✅ 初始训练完成，保存模型...")
    model.save("ppo_rov_dynamic_v1_final")