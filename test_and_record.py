import os
import cv2
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import holoocean
from stable_baselines3 import SAC 

# ==========================================
# 🌊 1. 3D 洋流模型 
# ==========================================
class OceanCurrentSimulator:
    def __init__(self, num_vortices=6, map_size=25.0, max_current=1, amplitude=0.0):
        self.num_vortices = num_vortices
        self.map_size = map_size
        self.max_current = max_current
        
        # 🌟 修复1：强制关闭潮汐随机波动，锁定最高倍率的死亡水流
        self.A = amplitude
        self.omega = 0.0 
        self.phi_0 = np.pi / 2  # 锁定 sin(90度) = 1，让 time_factor 永远满载
        
        self.vortices = []
        
    def reset(self):
        self.vortices = []
        
        # 🌟 修复2：定向打击！在它的必经之路 [6, 6] 正中央放一个巨型漩涡
        self.vortices.append({
            'x0': 6.0, 'y0': 6.0, 'z0': -8.0, 
            'eta': 35.0,        # 极度狂暴的旋转力矩
            'xi': 5.0, 
            'sigma_z': 15.0
        })
        
        # 其余的漩涡也强制缩拢在航线附近 (0 到 15 米区域)
        for _ in range(self.num_vortices - 1):
            x0 = np.random.uniform(0.0, 15.0)
            y0 = np.random.uniform(0.0, 15.0)
            z0 = np.random.uniform(-15.0, -2.0) 
            sigma_z = np.random.uniform(5.0, 15.0) 
            eta = np.random.choice([1, -1]) * np.random.uniform(15.0, 20.0) 
            xi = np.random.uniform(5.0, 10.0) 
            
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
        
        raw_vx = c_fx_base * time_factor + global_drift_x
        raw_vy = c_fy_base * time_factor + global_drift_y
        
        # 2. 🌟 绝对物理安全锁：使用类的动态极速限制
        horiz_speed = np.linalg.norm([raw_vx, raw_vy])
        if horiz_speed > self.max_current:
            scale = self.max_current / horiz_speed
            raw_vx *= scale
            raw_vy *= scale

        return np.array([raw_vx, raw_vy, 0.0])

# ==========================================
# 2. 底层物理环境 Wrapper (保持原有逻辑不变)
# ==========================================
os.environ["HOLOOCEAN_PATH"] = "D:\\HoloOceanEngine"
os.environ["HOLOOCEAN_SYSTEM_PATH"] = "D:\\HoloOceanEngine"

def get_rov_mixing_matrix():
    M = np.zeros((8, 6))
    M[0,:]=[ 1, -1,  0,  0,  0, -1]
    M[1,:]=[ 1,  1,  0,  0,  0,  1]
    M[2,:]=[ 1, -1,  0,  0,  0,  1]
    M[3,:]=[ 1,  1,  0,  0,  0, -1]
    M[4,:]=[ 0,  0,  1, -1,  1,  0]
    M[5,:]=[ 0,  0,  1,  1,  1,  0]
    M[6,:]=[ 0,  0,  1, -1, -1,  0]
    M[7,:]=[ 0,  0,  1,  1, -1,  0]
    return M

class ROVFullTestWrapper(gym.Env):
    def __init__(self):
        super().__init__()
        cfg = {
            "name": "ROV_Full_Test", "world": "OpenWater", "package_name": "Ocean", "main_agent": "my_rov",
            "frames_per_sec": 30, "agents": [{
                "agent_name": "my_rov", "agent_type": "HoveringAUV", "control_scheme": 0, "location": [0,0,-5],
                "sensors": [
                    {"sensor_type": "LocationSensor"}, 
                    {"sensor_type": "VelocitySensor"}, 
                    {"sensor_type": "RotationSensor"},
                    {"sensor_type": "IMUSensor"}, 
                    {
                        "sensor_type": "RGBCamera", 
                        "sensor_name": "FollowCamera", 
                        "location": [-3.5, 0.0, 1.5],
                        "rotation": [0.0, -15.0, 0.0],
                        "configuration": {
                            "CaptureWidth": 1280, 
                            "CaptureHeight": 720
                        }
                    }
                ]
            }]
        }
        self.holo_env = holoocean.make(scenario_cfg=cfg)
        self.action_space = spaces.Box(-1.0, 1.0, (6,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (25,), np.float32) 
        self.tam_inverse = get_rov_mixing_matrix()
        
        self.target_pos = np.zeros(3)
        self.buoyancy_bias = np.zeros(8)
        self.buoyancy_bias[4:8] = 3.0
        
        self.sonar_max_range = 30.0  
        self.sonar_min_range = 0.5   
        self.vertical_fov = np.deg2rad(15) 

        self.sonar_ray_dirs = []
        for angle in [-60, -30, 0, 30, 60]:
            self.sonar_ray_dirs.append([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), 0.0])
        for angle in [-40, 0, 40]:
            self.sonar_ray_dirs.append([np.cos(np.deg2rad(angle))*np.cos(np.deg2rad(25)), np.sin(np.deg2rad(angle))*np.cos(np.deg2rad(25)), np.sin(np.deg2rad(25))])
        for angle in [-40, 0, 40]:
            self.sonar_ray_dirs.append([np.cos(np.deg2rad(angle))*np.cos(np.deg2rad(-25)), np.sin(np.deg2rad(angle))*np.cos(np.deg2rad(-25)), np.sin(np.deg2rad(-25))])
        
        self.num_sonar_rays = len(self.sonar_ray_dirs)
        self.clean_sonar_ranges = np.ones(self.num_sonar_rays) * self.sonar_max_range
        # 🌟 核心修改
        self.ocean_sim = OceanCurrentSimulator(
            num_vortices=6,       
            map_size=25.0, 
            max_current=35,    # 3.5m/s 将产生 245N~250N 的阻力，几乎榨干推进器
            amplitude=0.0       # 关掉振幅随机
        )

    def reset_for_test(self, start_pos, final_target):
        self.current_step = 0
        self.target_pos = final_target
        self.ocean_sim.reset()
        
        obs_dict = self.holo_env.reset()
        self.current_obs_dict = obs_dict  
        self.holo_env.agents["my_rov"].teleport(start_pos) 
        
        return self._get_obs(obs_dict)

    def step(self, action):
        self.current_step += 1
        dt = 1/30.0
        obs_d = self.current_obs_dict 
        
        curr_p = obs_d["LocationSensor"]
        v_world = obs_d["VelocitySensor"]
        roll, pitch, yaw = np.deg2rad(obs_d["RotationSensor"])
        
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        R_yaw = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]])
        R_pitch = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
        R_roll = np.array([[1, 0, 0], [0, cr, sr], [0, -sr, cr]])
        R_world2body = R_roll @ R_pitch @ R_yaw
        
        vx_body, vy_body, vz_body = R_world2body @ v_world
        
        v_c = self.ocean_sim.get_current_velocity(curr_p[0], curr_p[1], curr_p[2], self.current_step*dt)
        v_cx_body, v_cy_body, v_cz_body = R_world2body @ v_c
        
        rel_vx = vx_body - v_cx_body
        rel_vy = vy_body - v_cy_body
        rel_vz = vz_body - v_cz_body
        
        force_x_body = -20.0 * rel_vx * abs(rel_vx)
        force_y_body = -20.0 * rel_vy * abs(rel_vy)
        force_z_body = -30.0 * rel_vz * abs(rel_vz)
        force_6dof_body = np.array([force_x_body, force_y_body, force_z_body, 0.0, 0.0, force_y_body * -0.2])
        
        pinv_M_T = np.linalg.pinv(self.tam_inverse.T)
        dist_8d = np.dot(pinv_M_T, force_6dof_body)
        
        thruster_commands = np.dot(self.tam_inverse, action)
        max_t = np.max(np.abs(thruster_commands))
        if max_t > 1.0: 
            thruster_commands /= max_t
            
        actual_physics_force = np.clip(
            thruster_commands * 65.0 + dist_8d + self.buoyancy_bias, 
            -500.0, 500.0
        )
        
        obs_dict_new = self.holo_env.step(actual_physics_force.tolist())
        self.current_obs_dict = obs_dict_new
        self.clean_sonar_ranges = np.ones(self.num_sonar_rays) * self.sonar_max_range
        
        return self._get_obs(obs_dict_new), obs_dict_new

    def _get_obs(self, obs_dict):
        current_pos = np.array(obs_dict["LocationSensor"])
        velocity_world = np.array(obs_dict["VelocitySensor"])
        roll, pitch, yaw = np.deg2rad(obs_dict["RotationSensor"])
        
        dx, dy, dz = self.target_pos[0] - current_pos[0], self.target_pos[1] - current_pos[1], self.target_pos[2] - current_pos[2]

        R_yaw = np.array([[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        R_pitch = np.array([[np.cos(pitch), 0, -np.sin(pitch)], [0, 1, 0], [np.sin(pitch), 0, np.cos(pitch)]])
        R_roll = np.array([[1, 0, 0], [0, np.cos(roll), np.sin(roll)], [0, -np.sin(roll), np.cos(roll)]])
        R_world2body = R_roll @ R_pitch @ R_yaw

        relative_pos_body = R_world2body @ np.array([dx, dy, dz])
        velocity_body = R_world2body @ velocity_world

        target_yaw = np.arctan2(dy, dx)
        yaw_error = (target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
        
        sonar_ranges = self.clean_sonar_ranges.copy()
        norm_sonar = sonar_ranges / self.sonar_max_range
        
        imu_data = np.array(obs_dict["IMUSensor"])
        angular_velocity = imu_data[1] 

        norm_relative_pos = np.clip(relative_pos_body / 40.0, -1.5, 1.5)
        norm_velocity = np.clip(velocity_body / 2.0, -2.0, 2.0)
        norm_rotation = np.array([
            np.clip(roll / 1.0, -1.2, 1.2),     
            np.clip(pitch / 1.0, -1.2, 1.2),   
            np.sin(yaw_error),          
            np.cos(yaw_error)           
        ])
        norm_angular_vel = np.clip(angular_velocity / 1.5, -2.0, 2.0)
        norm_absolute_depth = np.clip(current_pos[2] / 40.0, -1.0, 0.0)

        state = np.concatenate([
            norm_relative_pos, np.array([norm_absolute_depth]), norm_velocity, 
            norm_rotation, norm_angular_vel, norm_sonar           
        ])
        return state.astype(np.float32)

# ==========================================
# 🌟 3. 核心新增：参数化平滑曲线生成器
# ==========================================
def generate_smooth_curve(start_pos, end_pos, num_points=50):
    """
    生成一条起止点在训练范围内，且具有平滑侧向凸起的 3D 曲线
    """
    t = np.linspace(0, 1, num_points)
    path = np.zeros((num_points, 3))
    
    # X 和 Z 轴进行线性插值
    path[:, 0] = start_pos[0] + t * (end_pos[0] - start_pos[0])
    path[:, 2] = start_pos[2] + t * (end_pos[2] - start_pos[2])
    
    # Y 轴加入正弦波形，使其向侧面凸出 2.5 米，形成平滑的弧线
    path[:, 1] = start_pos[1] + t * (end_pos[1] - start_pos[1]) + np.sin(t * np.pi) * 2.5
    
    return path

# ==========================================
# 4. 运行控制塔
# ==========================================
if __name__ == "__main__":
    env = ROVFullTestWrapper()
    
    # 🌟 加载你刚刚用 30万步 锐化后的极限 0.8 米模型！
    print("🧠 正在加载 0.8米 极限特训后的巅峰模型...")
    model = SAC.load("sac_rov_edge8_extreme_current_no_fish_normal_dis.zip") 
    # print(model.policy)
    # 物理账本完全对齐训练环境
    START = np.array([0.0, 0.0, -5.0])  
    GOAL = np.array([12.0, 12.0, -12.0])    
    
    print(f"🎯 任务确立：训练域内极限追踪！从 {START} 潜航至 {GOAL}")
    
    # 抛弃 DEPlanner，直接生成绝对平滑的 50 个密集航点
    global_path = generate_smooth_curve(START, GOAL, num_points=50)
    
    if len(global_path) > 0:
        obs = env.reset_for_test(START, np.array(global_path[1]))
        wp_idx = 1
        
        # 画出平滑的绿色引导曲线
        for i in range(len(global_path)-1):
            env.holo_env.draw_line(global_path[i].tolist(), global_path[i+1].tolist(), color=[0, 255, 0], thickness=2.0, lifetime=0)
            
        print("🚀 纯净平滑弧线铺设完毕！SAC 开始执行高精度跟踪任务...")
        prev_p = np.copy(START)
        
        for step in range(3500):
            action, _ = model.predict(obs, deterministic=True)
            obs, raw_d = env.step(action)
            curr_p = raw_d["LocationSensor"]
            
            env.holo_env.draw_line(prev_p.tolist(), curr_p.tolist(), color=[255, 0, 0], thickness=4.0, lifetime=0)
            prev_p = np.copy(curr_p) 
            
            # 🌟 核心修改：极其严苛的 0.8 米航点切换逻辑
            # 只有当 ROV 贴近当前航点小于 0.8 米时，才允许切换到下一个航点！
            # 这完美触发了它在训练中 "distance_to_target < 0.8" 时拿 3000 分的冲刺记忆。
            target_hit_radius = 0.8
            if np.linalg.norm(curr_p - np.array(global_path[wp_idx])) < target_hit_radius and wp_idx < len(global_path)-1:
                wp_idx += 1
                env.target_pos = np.array(global_path[wp_idx])

            # --- 画面渲染 ---
            # --- 画面渲染与指标精算 ---
            frame = cv2.cvtColor(np.uint8(raw_d["FollowCamera"]), cv2.COLOR_RGBA2BGR)
            cv2.putText(frame, f"MISSION: SMOOTH CURVE", (40, 50), 2, 0.8, (0, 165, 255), 2)
            cv2.putText(frame, f"WAYPOINT: {wp_idx}/{len(global_path)-1}", (40, 90), 2, 0.8, (0, 255, 0), 2)

            # ==========================================
            # 🌟 核心新增：3D 横向跟踪误差 (Cross-Track Error) 精算
            # ==========================================
            # 1. 获取当前线段的起点 (上一个航点) 和终点 (当前目标航点)
            prev_wp = np.array(global_path[max(0, wp_idx-1)])
            curr_wp = np.array(global_path[wp_idx])
            
            # 2. 计算航线向量与 ROV 位移向量
            vec_path = curr_wp - prev_wp          # 航线的方向向量
            vec_rov = curr_p - prev_wp            # ROV 相对于线段起点的向量
            
            # 3. 向量叉乘计算正交距离
            path_length = np.linalg.norm(vec_path)
            if path_length > 1e-4:
                # 叉乘的模长除以底边长，即为平行四边形的高（ROV 到直线的绝对距离）
                cross_track_error = np.linalg.norm(np.cross(vec_path, vec_rov)) / path_length
            else:
                cross_track_error = np.linalg.norm(curr_p - curr_wp)
                
            # 4. 计算到当前航点的直线距离 (供参考比对)
            dist_to_wp = np.linalg.norm(curr_p - curr_wp)

            # 将精算结果打印到屏幕左侧 (红色代表越界警告色，白色代表安全)
            cte_color = (0, 0, 255) if cross_track_error > 0.5 else (255, 255, 255)
            cv2.putText(frame, f"CTE (ERR): {cross_track_error:.3f} m", (40, 130), 2, 0.8, cte_color, 2)
            cv2.putText(frame, f"DIST TO WP: {dist_to_wp:.2f} m", (40, 170), 2, 0.8, (255, 255, 0), 2)
            # ==========================================

            yaw_err_deg = np.rad2deg(obs[8]) 
            cv2.putText(frame, f"YAW ERR: {yaw_err_deg:+.1f} deg", (950, 50), 2, 0.8, (0, 255, 255), 2)

            if curr_p[2] > -0.5 or curr_p[2] < -39.0:
                cv2.putText(frame, "OUT OF BOUNDS!", (100, 360), 2, 4.0, (0, 0, 255), 8)
                cv2.imshow("ROV Curve Test", frame)
                cv2.waitKey(2000) 
                break

            cv2.imshow("ROV Curve Test", frame)
            
            # 终极胜利判定
            if wp_idx == len(global_path)-1 and np.linalg.norm(curr_p - GOAL) < 0.8:
                print("🏁 测试通过！ROV 完美贴合了 3D 弧线飞行！")
                break
            
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break