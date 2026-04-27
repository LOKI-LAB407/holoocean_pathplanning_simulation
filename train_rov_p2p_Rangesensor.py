import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import holoocean
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

# ==========================================
# 🌊 工业级 3D 海流孪生模型 (Lamb Vortex + Gaussian Vertical Decay)
# ==========================================
class OceanCurrentSimulator:
    def __init__(self, num_vortices=0, map_size=25.0):
        self.num_vortices = num_vortices
        self.map_size = map_size
        
        # 🌟 调整 1：将洋流振幅降至 0.15，对齐 Edge 8 实机的抗流极限 (约 1.2 m/s)
        self.A = 0
        self.omega = 2 * np.pi / (12.4 * 3600) 
        self.phi_0 = 0.0 
        
        self.vortices = []
        
    def reset(self):
        self.vortices = []
        self.phi_0 = np.random.uniform(0, 2 * np.pi) 
        
        ## 生成多个涡旋，每个涡旋在水平面上有随机位置和强度，在垂直方向上有不同的衰减特性
        for _ in range(self.num_vortices):
            x0 = np.random.uniform(-self.map_size, self.map_size)
            y0 = np.random.uniform(-self.map_size, self.map_size)
            z0 = np.random.uniform(-35.0, -2.0) 
            sigma_z = np.random.uniform(3.0, 15.0) 
            eta = np.random.choice([1, -1]) * np.random.uniform(3.0, 8.0) 
            xi = np.random.uniform(4.0, 10.0) 
            
            self.vortices.append({
                'x0': x0, 'y0': y0, 'z0': z0, 
                'eta': eta, 'xi': xi, 'sigma_z': sigma_z
            })

    def get_current_velocity(self, x, y, z, t):
        c_fx_base = 0.0
        c_fy_base = 0.0
        
        for v in self.vortices:
            dx = x - v['x0']
            dy = y - v['y0']
            r_sq = dx**2 + dy**2
            
            if r_sq < 1e-4:
                decay_factor = 1.0 / (2 * np.pi * v['xi']**2)
            else:
                decay_factor = (1.0 - np.exp(-r_sq / (v['xi']**2))) / (2 * np.pi * r_sq)
            
            z_decay = np.exp(-((z - v['z0'])**2) / (v['sigma_z']**2))
            
            c_fx_base += -v['eta'] * dy * decay_factor * z_decay
            c_fy_base +=  v['eta'] * dx * decay_factor * z_decay

        time_factor = 1.0 + self.A * np.sin(self.omega * t + self.phi_0)
        depth_ratio = np.clip(1.0 - abs(z)/40.0, 0.0, 1.0)
        global_drift_x = 0.05 * time_factor * depth_ratio
        global_drift_y = 0.03 * time_factor * depth_ratio
        
        # 1. 先计算出原始的合成流速
        raw_vx = c_fx_base * time_factor + global_drift_x
        raw_vy = c_fy_base * time_factor + global_drift_y
        
        # 2. 🌟 绝对物理安全锁：计算水平总流速并限速
        MAX_CURRENT = 0.1  # 设定的洋流速度绝对上限 (m/s)
        horiz_speed = np.linalg.norm([raw_vx, raw_vy])
        
        if horiz_speed > MAX_CURRENT:
            # 等比例缩放，保证洋流的“方向”不变，只削弱“力度”
            scale = MAX_CURRENT / horiz_speed
            raw_vx *= scale
            raw_vy *= scale

        return np.array([raw_vx, raw_vy, 0.0])

# ==========================================
# 0. 保护 C 盘与引擎路径配置
# ==========================================
os.environ["HOLOOCEAN_PATH"] = "D:\\HoloOceanEngine"
os.environ["HOLOOCEAN_SYSTEM_PATH"] = "D:\\HoloOceanEngine"

def get_rov_mixing_matrix():
    # 矩阵维度: [8个推进器, 6个自由度(Surge, Sway, Heave, Roll, Pitch, Yaw)]
    M = np.zeros((8, 6))
    
    # ========================================================
    # 🔹 物理通道 1-4：水平推进器组 (完全根据 QGC 构型图)
    # 注意：QGC复选框已处理后部电机反转，所以 Surge (第一列) 全部是 1
    # ========================================================
    # M1 (右前): 往前推(+1)，往右偏(-1)，导致左转(-1)
    M[0, :] = np.array([ 1, -1,  0,  0,  0, -1])  
    
    # M2 (左前): 往前推(+1)，往左偏(+1)，导致右转(+1)
    M[1, :] = np.array([ 1,  1,  0,  0,  0,  1])  
    
    # M3 (右后): 往前推(+1)，往右偏(-1)，导致右转(+1)
    M[2, :] = np.array([ 1, -1,  0,  0,  0,  1])  
    
    # M4 (左后): 往前推(+1)，往左偏(+1)，导致左转(-1)
    M[3, :] = np.array([ 1,  1,  0,  0,  0, -1])  
    
    # ========================================================
    # 🔹 物理通道 5-8：垂直推进器组 (完全根据 QGC 构型图)
    # 注意：Heave(下潜/上升，第三列) 全部是 1
    # ========================================================
    # M5 (右侧前): 往上推(+1)，导致向左横滚(-1)，向下低头(+1)
    M[4, :] = np.array([ 0,  0,  1, -1,  1,  0])  
    
    # M6 (左侧前): 往上推(+1)，导致向右横滚(+1)，向下低头(+1)
    M[5, :] = np.array([ 0,  0,  1,  1,  1,  0])  
    
    # M7 (右侧后): 往上推(+1)，导致向左横滚(-1)，向上抬头(-1)
    M[6, :] = np.array([ 0,  0,  1, -1, -1,  0])  
    
    # M8 (左侧后): 往上推(+1)，导致向右横滚(+1)，向上抬头(-1)
    M[7, :] = np.array([ 0,  0,  1,  1, -1,  0])  

    return M

rov_config = {
    "name": "ROV_P2P_3DOcean",
    "world": "OpenWater",
    "package_name": "Ocean", 
    "main_agent": "my_rov",
    "frames_per_sec": 30, 
    "agents": [
        {
            "agent_name": "my_rov",
            "agent_type": "HoveringAUV",
            "control_scheme": 0, 
            ## 初始位置设置在水面下 5 米，远离边界，给它足够的空间来感知和学习  
            "location": [0.0, 0.0, -5.0], 
            "sensors": [
                {"sensor_type": "LocationSensor"}, 
                {"sensor_type": "VelocitySensor"}, 
                {"sensor_type": "RotationSensor"},
                {"sensor_type": "IMUSensor",
                 "socket": "IMUSocket"
                 }  
            ]
        }
    ]
}

# ==========================================
# 2. 核心环境 Wrapper
# ==========================================
class ROVP2PDynamicWrapper(gym.Env):
    def __init__(self, config):
        super(ROVP2PDynamicWrapper, self).__init__()
        self.holo_env = holoocean.make(scenario_cfg=config, show_viewport=False)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32)
        
        self.tam_inverse = get_rov_mixing_matrix()
        self.max_steps = 2000  
        self.current_step = 0
        self.max_tilt_rad = 1.0        
        self.water_surface_z = -0.5    
        self.water_bottom_z = -40.0    
        
        self.target_pos = np.zeros(3)
        self.prev_waypoint = np.zeros(3)
        self.previous_distance = 0.0
        self.previous_action = np.zeros(6) 
        self.current_disturbance = np.zeros(8) 

        self.buoyancy_bias = np.zeros(8)
        self.buoyancy_bias[4:8] = 3.0

        self.ocean_sim = OceanCurrentSimulator(num_vortices=0, map_size=25.0)

        # 🌟 11 射线 3D 全景声呐配置
        self.sonar_max_range = 30.0  
        self.sonar_min_range = 0.5   
        self.vertical_fov = np.deg2rad(15) # 减小 FOV，因为射线变密了

        self.sonar_ray_dirs = []
        # 1. 水平面 5 条
        for angle in [-60, -30, 0, 30, 60]:
            self.sonar_ray_dirs.append([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), 0.0])
        # 2. 向上俯仰 25度 的 3 条
        for angle in [-40, 0, 40]:
            self.sonar_ray_dirs.append([np.cos(np.deg2rad(angle))*np.cos(np.deg2rad(25)), 
                                        np.sin(np.deg2rad(angle))*np.cos(np.deg2rad(25)), np.sin(np.deg2rad(25))])
        # 3. 向下俯仰 25度 的 3 条
        for angle in [-40, 0, 40]:
            self.sonar_ray_dirs.append([np.cos(np.deg2rad(angle))*np.cos(np.deg2rad(-25)), 
                                        np.sin(np.deg2rad(angle))*np.cos(np.deg2rad(-25)), np.sin(np.deg2rad(-25))])

        
        self.num_sonar_rays = len(self.sonar_ray_dirs) # 变成 11 条
        self.clean_sonar_ranges = np.ones(self.num_sonar_rays) * self.sonar_max_range

        # 🌟 性能优化：提前计算并缓存 TAM 转置的伪逆矩阵
        self.pinv_M_T = np.linalg.pinv(self.tam_inverse.T)

    def _generate_dynamic_obstacles(self):
        self.dynamic_obs = []
        # num_obstacles = np.random.randint(6, 10) # 🌟 数量翻倍：变成鱼群
        num_obstacles = 0
        for _ in range(num_obstacles):
            pos = self.target_pos * np.random.uniform(0.2, 0.8) 
            pos += np.random.uniform(-8.0, 8.0, 3) 
            pos[2] = np.clip(pos[2], self.water_bottom_z + 2, self.water_surface_z - 2)
            
            vel = np.random.uniform(-1.0, 1.0, 3)
            vel[2] *= 0.1 
            vel = (vel / np.linalg.norm(vel)) * np.random.uniform(1.5, 2) # 速度较快
            radius = np.random.uniform(0.2, 0.4) # 🌟 尺寸缩小：0.2-0.4m 半径的小鱼

            self.dynamic_obs.append({'pos': pos, 'vel': vel, 'radius': radius})

    def _update_dynamic_obstacles(self):
        dt = 1.0 / 30.0 
        current_rov_pos = np.array(self.current_obs_dict["LocationSensor"])
  
        for obs in self.dynamic_obs:
           # 🌟 鱼群避让逻辑：如果离 ROV 小于 3 米，鱼会向反方向加速逃逸
            dist_to_rov = np.linalg.norm(obs['pos'] - current_rov_pos)
            if dist_to_rov < 3.0:
                escape_vec = obs['pos'] - current_rov_pos
                escape_vec[2] *= 0.1 # 避免鱼飞出水面
                # 🌟 修复：加入极小值保护，防止除以零引发 NaN 病毒
                norm = np.linalg.norm(escape_vec)
                if norm > 1e-4:
                    escape_dir = escape_vec / norm
                    obs['vel'] = escape_dir * 1.0

            obs['pos'] += obs['vel'] * dt 
            # ========== 替换原有的 for axis in range(3) 逻辑 ==========
            # 1. 处理 X, Y 轴水平边界
            for axis in range(2): 
                if obs['pos'][axis] > 30.0:
                    obs['vel'][axis] = -abs(obs['vel'][axis]) 
                elif obs['pos'][axis] < -30.0:
                    obs['vel'][axis] = abs(obs['vel'][axis])  

            # 2. 单独处理 Z 轴垂直边界 (防止鱼飞出水面或钻入地底)
            if obs['pos'][2] > self.water_surface_z - 1.0: # 离水面还有1米时就反弹
                obs['vel'][2] = -abs(obs['vel'][2])
            elif obs['pos'][2] < self.water_bottom_z + 1.0:
                obs['vel'][2] = abs(obs['vel'][2])
  

    def _get_simulated_sonar(self, current_pos, roll, pitch, yaw):
        ranges = np.ones(self.num_sonar_rays) * self.sonar_max_range
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        
        R_yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        R_roll = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        R_body2world = R_yaw @ R_pitch @ R_roll 

        tan_half_vfov = np.tan(self.vertical_fov / 2.0)
        
        all_targets = list(self.dynamic_obs)

        # 之前是 for i, angle in enumerate(self.sonar_angles):
        for i, ray_body in enumerate(self.sonar_ray_dirs):
            # 将机体坐标系下的 3D 射线方向转到世界坐标系
            ray_dir_world = R_body2world @ np.array(ray_body)

            for obs in all_targets:
                V = current_pos - obs['pos'] 
                t_approx = np.dot(-V, ray_dir_world)

                if t_approx > 0: 
                    r_eff = obs['radius'] + t_approx * tan_half_vfov
                    b = 2.0 * np.dot(V, ray_dir_world)
                    c = np.dot(V, V) - r_eff**2
                    discriminant = b**2 - 4*c
                    if discriminant > 0: 
                        t = (-b - np.sqrt(discriminant)) / 2.0
                        dist = self.sonar_min_range if t < self.sonar_min_range else t
                        if 0 < dist < ranges[i]: ranges[i] = dist

        return ranges

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.previous_action = np.zeros(6) 
        
        # 🌟 架构升级：适配 DE 算法的密集航点追踪
        # 假设 DE 规划的航点间隔在 3 到 8 米之间，我们就只在这个范围内训练！
        # ROV 出生在 [0, 0, -5.0]，我们在它周围随机刷出近距离目标
        
        # 先生成一个 3到 4米的随机距离
        target_dist = np.random.uniform(3.0, 4.0)
        
        # 随机生成一个水平方向的角度 (0 到 2pi)
        angle = np.random.uniform(0, 2 * np.pi)
        
        # 随机生成一个深度差异 
        dz = np.random.uniform(-7.0, -2.0)
        
        self.target_pos = np.array([
            target_dist * np.cos(angle),          # X 坐标
            target_dist * np.sin(angle),          # Y 坐标
            dz                                    # Z 坐标 (保证在水下安全范围内)
        ])
        
        self.ocean_sim.reset()
        obs_dict = self.holo_env.reset()
        self.current_obs_dict = obs_dict

        # ==========================================
        # 🌟 救命修复：初始化 previous_yaw，防止第一步崩溃！
        _, _, yaw_init = np.deg2rad(obs_dict["RotationSensor"])
        self.previous_yaw = yaw_init
        # ==========================================
        
        initial_pos = np.array(obs_dict["LocationSensor"])
        self.previous_distance = np.linalg.norm(self.target_pos - initial_pos)
        
        self._generate_dynamic_obstacles()
        
        # try:
        #     self.holo_env.draw_box(center=self.target_pos.tolist(), extent=[0.5]*3, color=[255, 0, 0], lifetime=0)
        #     self.holo_env.draw_line(start=initial_pos.tolist(), end=self.target_pos.tolist(), color=[0, 255, 0], lifetime=0)
        # except Exception: pass 
            
        return self._get_obs(obs_dict), {}

    def step(self, action):
        self.current_step += 1
        self._update_dynamic_obstacles()

        current_pos_old = np.array(self.current_obs_dict["LocationSensor"])
        
        dvl_velocity_world = np.array(self.current_obs_dict["VelocitySensor"]) 
        roll_rad, pitch_rad, yaw_rad = np.deg2rad(self.current_obs_dict["RotationSensor"])
        cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
        cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
        cr, sr = np.cos(roll_rad), np.sin(roll_rad)

        # 2. 构造 World -> Body 完整 3D 旋转矩阵 (ZYX 顺序的逆矩阵)
        R_yaw = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]])
        R_pitch = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
        R_roll = np.array([[1, 0, 0], [0, cr, sr], [0, -sr, cr]])
        R_world2body = R_roll @ R_pitch @ R_yaw

        # 3. 矩阵乘法：一步到位获取精确的 3D 机体速度
        vx_body, vy_body, vz_body = R_world2body @ dvl_velocity_world
        # 根据帧率计算当前的物理时间
        current_time = self.current_step * (1.0 / 30) 

        # 获取世界坐标系下的真实洋流速度
        v_current_world = self.ocean_sim.get_current_velocity(current_pos_old[0], current_pos_old[1], current_pos_old[2], current_time)
        # 洋流速度也同步转入机体坐标系（保持物理严谨）
        v_cx_body, v_cy_body, v_cz_body = R_world2body @ v_current_world
        
        rel_vx, rel_vy, rel_vz = vx_body - v_cx_body, vy_body - v_cy_body, vz_body - v_cz_body
        
        force_x_body, force_y_body, force_z_body = -20.0 * rel_vx * abs(rel_vx), -20.0 * rel_vy * abs(rel_vy),-30.0 * rel_vz * abs(rel_vz)
        force_6dof_body = np.array([force_x_body, force_y_body, force_z_body, 0.0, 0.0, force_y_body * -0.2])
        # 修复：使用 TAM 转置的伪逆来正确分配阻力
        # 直接使用预存的伪逆矩阵计算洋流干扰力
        self.current_disturbance = np.dot(self.pinv_M_T, force_6dof_body)

        thruster_commands = np.dot(self.tam_inverse, action)
        max_thrust = np.max(np.abs(thruster_commands))
        if max_thrust > 1.0: 
            thruster_commands = thruster_commands / max_thrust
            
        # 🌟 调整 2：推力乘数改为 65.0，绝对匹配 Edge 8 推进器硬件极限
        actual_physics_force = np.clip(
            thruster_commands * 65.0 + self.current_disturbance + self.buoyancy_bias, 
            -500.0, 500.0
        )
        
        obs_dict_new = self.holo_env.step(actual_physics_force.tolist())
        self.current_obs_dict = obs_dict_new  
        
        current_pos = np.array(obs_dict_new["LocationSensor"])
        speed = np.linalg.norm(np.array(obs_dict_new["VelocitySensor"]))
        roll_new, pitch_new, yaw_new = np.deg2rad(obs_dict_new["RotationSensor"])
        
        dx, dy = self.target_pos[0] - current_pos[0], self.target_pos[1] - current_pos[1]
        target_yaw = np.arctan2(dy, dx)
        yaw_error = (target_yaw - yaw_new + np.pi) % (2 * np.pi) - np.pi
        distance_to_target = np.linalg.norm(self.target_pos - current_pos)
        
        self.clean_sonar_ranges = self._get_simulated_sonar(current_pos, roll_new, pitch_new, yaw_new)
        min_sonar_dist = np.min(self.clean_sonar_ranges)
        
        # ========================================
        # 🌟 5. 终极完美版：奖励函数结算
        # ========================================
        reward = 0.0
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        ## 1. 前进奖励
        progress = self.previous_distance - distance_to_target
        # 修复：统一系数，用生存税来驱动时间效率
        reward += progress * 40.0   # 正负统一，消除尖点
        self.previous_distance = distance_to_target


        ## 生存税
        reward -= 0.25

        # 🌟 核心防挂机考核：基于真实物理速度 (0.2m/s ~ 1.0m/s) 的线性门控
        # progress 是单步(1/30秒)内向目标缩短的距离。乘以 30 即为真实的“有效接近速度(m/s)”
        closing_speed = max(progress, 0) * 30.0
        
        # 线性插值精算：
        # - 速度 <= 0.2 m/s：乘数为 0 (彻底掐断低速摸鱼赚姿态分的可能)
        # - 速度 >= 1.0 m/s：乘数为 1.0 (全速巡航，拿满所有绩效)
        # - 速度在 0.2 ~ 1.0 之间：平滑线性增长
        moving_multiplier = np.clip((closing_speed - 0.1) / (1.0 - 0.1), 0.0, 1.0)

        
        ## 2. 🌟 铁腕航向对齐 (防挂机版)
        yaw_error_deg_abs = abs(np.rad2deg(yaw_error))
        
        # 钟形曲线：0度=1.0分, 20度≈0.36分, 40度≈0.01分
        yaw_bonus = np.exp(- (yaw_error_deg_abs**2) / 2000.0)
        reward += yaw_bonus  * moving_multiplier  # 只有在有效前进时才奖励航向对齐
            
        yaw_change = abs(yaw_new - self.previous_yaw)
        if yaw_change > np.pi: 
            yaw_change = 2 * np.pi - yaw_change

        # 区分"无目的摇头"和"紧急规避转向"
        # 动态计算转向惩罚系数，在 1.2m 到 4.0m 之间平滑过渡 (0.2 到 1.5)
        yaw_penalty_factor = np.clip(1.5 - 1.3 * (4.0 - min_sonar_dist) / (4.0 - 1.0), 0.2, 1.5)
        reward -= yaw_change * yaw_penalty_factor

        self.previous_yaw = yaw_new
        
        ## 3. ⚖️ 高斯姿态奖励 (取消线性扣分)
        tilt = np.sqrt(roll_new**2 + pitch_new**2)
        tilt_deg = np.rad2deg(tilt)

        # 钟形曲线：0度=1.0分, 15度≈0.47分, 30度≈0.05分
        # 修复：只在有效前进时才给姿态奖励
        attitude_bonus = np.exp(-(tilt_deg**2) / 300.0)
        reward += attitude_bonus * moving_multiplier
                    
        # 危险倾角底线惩罚 (防止彻底翻车)
        # 🌟 修复：长途续航版 晚熟指数障碍惩罚 (Late-Blooming Exponential Barrier)
        # 40度时惩罚仅为 0.088，完美适应长步数累积；55度后爆发至 35+ 防止侧翻
        raw_barrier = 1e-8 * (np.exp(0.4 * tilt_deg) - 1.0)
        barrier_penalty = np.clip(raw_barrier, 0.0, 100.0)
        reward -= barrier_penalty
        # if tilt_deg > 40.0:
        #     reward -= (tilt_deg - 40.0) * 0.2

     
        # # ## 4. 速度惩罚
        # # 根据声呐最近距离动态限制速度
        # safe_speed = np.clip(min_sonar_dist / 2.5, 0.3, 1.5)  # 障碍越近，允许速度越低
        # if speed > safe_speed:
        #     reward -= (speed - safe_speed) * 8.0
        
        ## 5. 🌟 释放性能封印 (极低的动力惩罚，鼓励敢于满功率抗流)
        # horizontal_effort = action[0]**2 + action[1]**2
        # vertical_effort = action[2]**2 + action[3]**2 + action[4]**2
        # yaw_effort = action[5]**2 
        # 惩罚降到极低，让 ROV 放开手脚对抗大自然
        # reward -= (horizontal_effort * 0.002 + vertical_effort * 0.001 + yaw_effort * 0.01)
        # 动作平滑惩罚大幅降低，允许快速机动
        reward -= np.sum(np.square(action - self.previous_action)) * 0.04
        self.previous_action = np.copy(action)
        
        ## 6. 声呐障碍物软惩罚
        ## 替换原来的软惩罚
        # 🌟 柔化声呐惩罚
        safe_dist = 4.0
        danger_dist = 1.0  
        max_soft_penalty = 1.5 # 降低软惩罚的极值，鼓励它大胆靠近

        if min_sonar_dist < danger_dist:
            # 即使进入 1.0m，惩罚也不要那么狂暴，给鱼逃跑的时间
            reward -= max_soft_penalty + 5.0 * (danger_dist - min_sonar_dist) / danger_dist
        elif min_sonar_dist < safe_dist:
            danger_ratio = (safe_dist - min_sonar_dist) / (safe_dist - danger_dist)
            reward -= max_soft_penalty * (danger_ratio ** 2)
            
        # 🌟 新策略：减速等待，小角度绕行
        # 当遇到动态障碍物 (距离较近) 时，考核其是否进行了减速
        if min_sonar_dist < safe_dist: 
            safe_speed_limit = np.clip(min_sonar_dist * 0.4, 0.3, 1.5)
            
            # 构建一个边界平滑衰减系数
            # 距离 4.0m 时系数为 0，距离 1.0m 时系数为 1。完美实现平滑过渡
            zone_ratio = np.clip((safe_dist - min_sonar_dist) / (safe_dist - danger_dist), 0.0, 1.0)

            # 修复：完全删除 speed_yield_ratio 的正奖励部分
            # 超速才罚，低速不给奖，让进度奖励来驱动移动
            if speed > safe_speed_limit:
                reward -= (speed - safe_speed_limit) * 12.0 * zone_ratio


        # 修复：在奖励函数中增加软边界惩罚
        # 接近水面/水底时给予渐进惩罚
        # 在每步奖励计算中加入
        current_z = current_pos[2]

        # 水面危险区（-0.5 到 -2.0m 之间线性增强）
        if current_z > -2:
            surface_danger = np.clip((current_z + 2) / 1.5, 0.0, 1.0)   # 0 到 1
            reward -= surface_danger ** 2 * 8.0          # 最大 -8/步

        # 水底危险区
        if current_z < self.water_bottom_z + 2:
            bottom_danger = np.clip((self.water_bottom_z + 2 - current_z) / 2.0, 0.0, 1.0)
            reward -= bottom_danger ** 2 * 8.0
            # 删除 else 分支的 yield 奖励

        # ========================================
        # 🚨 8. 终极 AABB 刚体碰撞与越界检测
        # ========================================
        if self.current_step > 10: 
            rov_radius = 0.6 
            # 死亡的真实代价 = 基础罚款 + 把没交完的税一次性交齐！
            death_penalty = 1000

            # 鱼碰撞改为终止，消除无限叠加
            is_fish_crashed = False
            for fish in self.dynamic_obs:
                if np.linalg.norm(current_pos - fish['pos']) < (fish['radius'] + rov_radius):
                    is_fish_crashed = True
                    print(f"💥 碰撞了鱼！")
                    break

            if is_fish_crashed:
                reward -= death_penalty  # 同样按时间缩放
                terminated = True

            
            elif abs(roll_new) > self.max_tilt_rad or abs(pitch_new) > self.max_tilt_rad:
                reward -= death_penalty
                terminated = True
                print(f"💀 [重启] 姿态翻车!")
                
            elif current_pos[2] > self.water_surface_z or current_pos[2] < self.water_bottom_z \
                or abs(current_pos[0]) > 35.0 or abs(current_pos[1]) > 35.0:
                reward -= death_penalty
                terminated = True
                print(f"💀 [重启] 冲出水面、沉入海底或逃离了测试海域!")
                
            elif distance_to_target < 0.8:
                reward += 3000.0 # 基础到达奖金
                # 姿态越平，给得越多 (最高各给 100)
                roll_perf = np.clip(1.0 - abs(roll_new)/0.5, 0.0, 1.0)
                pitch_perf = np.clip(1.0 - abs(pitch_new)/0.5, 0.0, 1.0)
                reward += (roll_perf * 100.0 + pitch_perf * 100.0)

                # 速度越稳，给得越多 (最高给 100)
                speed_perf = np.clip(1.0 - speed/1.0, 0.0, 1.0)
                reward += speed_perf * 100.0
                print(f"✅ 完美到达！耗时 {self.current_step} 步")
                
                terminated = True
            
        return self._get_obs(obs_dict_new), reward, terminated, truncated, {}

    def _get_obs(self, obs_dict):
        # --- 1. 获取原始生数据 (Raw Data) ---
        current_pos = np.array(obs_dict["LocationSensor"])
        velocity_world = np.array(obs_dict["VelocitySensor"])
        roll, pitch, yaw = np.deg2rad(obs_dict["RotationSensor"])

        dx = self.target_pos[0] - current_pos[0]
        dy = self.target_pos[1] - current_pos[1] 
        dz = self.target_pos[2] - current_pos[2]
        # cos_y, sin_y = np.cos(yaw), np.sin(yaw)

        # 在 _get_obs 中使用完整的 3D 逆矩阵
        R_yaw = np.array([[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        R_pitch = np.array([[np.cos(pitch), 0, -np.sin(pitch)], [0, 1, 0], [np.sin(pitch), 0, np.cos(pitch)]])
        R_roll = np.array([[1, 0, 0], [0, np.cos(roll), np.sin(roll)], [0, -np.sin(roll), np.cos(roll)]])
        R_world2body = R_roll @ R_pitch @ R_yaw

        # 彻底修复：三维位置与速度的全耦合投影
        relative_pos_body = R_world2body @ np.array([dx, dy, dz])
        velocity_body = R_world2body @ velocity_world

        # 航向误差计算
        target_yaw = np.arctan2(dy, dx)
        yaw_error = (target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi

        # 声呐获取与噪声注入
        if self.current_step == 0:
            sonar_ranges = self._get_simulated_sonar(current_pos, roll, pitch, yaw)
        else:
            sonar_ranges = self.clean_sonar_ranges.copy()
        #无障碍物时输出纯净的 30.0m 常数
        # sonar_ranges += np.random.normal(0, 0.2, size=self.num_sonar_rays)
        sonar_ranges = np.clip(sonar_ranges, 0.1, self.sonar_max_range)

        # IMU 角速度获取
        imu_data = np.array(obs_dict["IMUSensor"])
        angular_velocity = imu_data[1] 

        # ==========================================
        # 🌟 2. 终极归一化处理 (Sim2Real 核心防御层)
        # ==========================================
        # 位置归一化：除以地图预估最大跨度 40.0m，并截断防止越界爆炸
        norm_relative_pos = np.clip(relative_pos_body / 40.0, -1.5, 1.5)
        
        # 速度归一化：除以 ROV 最大物理速度约 2.0m/s
        norm_velocity = np.clip(velocity_body / 2.0, -2.0, 2.0)
        
        # 姿态归一化与连续化：
        # roll/pitch 除以 pi 压入 [-1, 1]；yaw_error 拆解为 sin/cos 消除 -180到180度的悬崖
        norm_rotation = np.array([
            np.clip(roll / 1.0, -1.2, 1.2),     # 充分利用 [-1, 1] 区间
            np.clip(pitch / 1.0, -1.2, 1.2),              
            np.sin(yaw_error),          
            np.cos(yaw_error)           
        ])
        
        # 角ROV 极限自旋约 85度/秒 (1.5 rad/s)
        norm_angular_vel = np.clip(angular_velocity / 1.5, -2.0, 2.0)

        # 声呐归一化：已经是 0~1 之间的完美数值
        norm_sonar = sonar_ranges / self.sonar_max_range
        # 获取当前绝对深度并归一化 (水底约 -40m)
        norm_absolute_depth = np.clip(current_pos[2] / 40.0, -1.0, 0.0)
        # --- 3. 拼装为 25 维 Tensor 喂给神经网络 ---
        state = np.concatenate([
            norm_relative_pos,   # 3 维
            np.array([norm_absolute_depth]), # 🌟 新增 1 维
            norm_velocity,       # 3 维
            norm_rotation,       # 4 维 (原为 3 维)
            norm_angular_vel,    # 3 维
            norm_sonar           # 11 维
        ])
        
        # 严格控制数据类型为 float32，对接 ONNX 模型
        return state.astype(np.float32)

# ==========================================
# 3. 训练启动入口 (从零开始全新训练)
# ==========================================
if __name__ == "__main__":
    print("🌟 正在初始化 [Headless 无头模式] 的全新强化学习环境...")
    env = ROVP2PDynamicWrapper(rov_config)
    
    print("🧠 正在孕育全新的 SAC 模型 (V2 完美奖励版)...")
    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0001,
        buffer_size=500000,       # 更大的经验池，防止好样本被快速覆盖
        batch_size=512,           # 🌟 大批次，梯度更稳定
        ent_coef='auto',          
        gamma=0.99,               
        train_freq=1,             # 每步都学，但每步只学一次
        gradient_steps=1,         # 每次只更新 1 次，降低过拟合风险
        tensorboard_log="./rov_tensorboard/"
    )
    
    
    # 🌟 换一个新的前缀，防止覆盖你之前旧版本的权重，方便以后复盘对比
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path='./rov_models/',
        name_prefix='sac_rov_edge8_v2' ,
        save_replay_buffer=True
    )
    
    print("🔥 开启全新后台无头训练！目标：50万步！")
    
    model.learn(
        total_timesteps=600000, 
        callback=checkpoint_callback, 
        # 🌟 日志也换个新名字，这样在 TensorBoard 里会生成一条全新的曲线
        tb_log_name="SAC_Edge8_Sim2Real_V1" 
    )
    
    print("✅ 初始训练完成，保存最新模型...")
    model.save("sac_rov_edge8_no_current_no_fish_small_distance_v1")
    model.save_replay_buffer("sac_rov_v1_replay_buffer")