import os
import time
import cv2  # 📸 新增：导入 OpenCV 用于视频录制
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import holoocean
from stable_baselines3 import PPO
from rrt_planner import RRT3D # 导入我们刚写的规划器
import pandas as pd

# 1. 保护 C 盘
os.environ["HOLOOCEAN_PATH"] = "D:\\HoloOceanEngine"
os.environ["HOLOOCEAN_SYSTEM_PATH"] = "D:\\HoloOceanEngine"

# ==========================================
# 保持与训练完全一致的底层矩阵与配置
# ==========================================
def get_rov_mixing_matrix():
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

rov_config = {
    "name": "ROV_Evaluate",
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
                {"sensor_type": "RotationSensor"},
                # 📸 新增：在测试环境中挂载高清摄像机
                {
                    "sensor_type": "RGBCamera",
                    "socket": "CameraSocket", 
                    "configuration": {
                        "CaptureWidth": 512,  
                        "CaptureHeight": 512
                    }
                }
            ]
        }
    ]
}

# ==========================================
# 检验专用 Wrapper (加入了轨迹绘制功能)
# ==========================================
class ROVEvalWrapper(gym.Env):
    def __init__(self, config):
        super(ROVEvalWrapper, self).__init__()
        self.holo_env = holoocean.make(scenario_cfg=config)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.tam_inverse = get_rov_mixing_matrix()
        
        self.target_pos = np.zeros(3)
        self.last_pos = np.zeros(3)
        
        self.max_current_power = 0.0 
        self.current_disturbance = np.zeros(8)

    def reset(self, seed=None, options=None):
        self.current_step = 0
        
        obs_dict = self.holo_env.reset()
        self.current_obs_dict = obs_dict
        self.start_pos = np.array(obs_dict["LocationSensor"])
        self.final_target = np.array([15.0, 15.0, -8.0]) 
        
        self.obstacles = [
            (5.0, 5.0, -5.0, 3.0),   
            (10.0, 10.0, -8.0, 4.0)  
        ]
        
        if self.max_current_power == 0.0:
            print(f"\n🌊 本次测试：无洋流干扰，纯净环境！")
        else:
            print(f"\n🌊 本次测试：洋流干扰强度为 {self.max_current_power}！")
        self.current_disturbance = np.random.uniform(-self.max_current_power, self.max_current_power, size=(8,))
        
        try:
            self.holo_env.draw_box(center=self.final_target.tolist(), extent=[1.0, 1.0, 1.0], color=[255, 0, 0], lifetime=0)
            
            for obs in self.obstacles:
                self.holo_env.draw_box(
                    center=list(obs[:3]), 
                    extent=[obs[3], obs[3], obs[3]], 
                    color=[255, 255, 0], 
                    lifetime=0.1
                ) 
        except Exception as e:
            print(f"⚠️ 画石头警告: {e}")
            
        return self._get_obs(obs_dict), {}

    def step(self, action):
        self.current_step += 1
        
        dvl_velocity_world = np.array(self.current_obs_dict["VelocitySensor"]) 
        _, _, yaw_deg = self.current_obs_dict["RotationSensor"]
        yaw_rad = np.deg2rad(yaw_deg)

        cos_y = np.cos(yaw_rad)
        sin_y = np.sin(yaw_rad)
        
        vx_world, vy_world, vz_world = dvl_velocity_world[0], dvl_velocity_world[1], dvl_velocity_world[2]
        
        vx_body = vx_world * cos_y + vy_world * sin_y   
        vy_body = -vx_world * sin_y + vy_world * cos_y  
        vz_body = vz_world                              
        
        dvl_velocity_body = np.array([vx_body, vy_body, vz_body])
        
        k_dvl = 2.5 
        dvl_compensation_6d = np.zeros(6)
        dvl_compensation_6d[0:3] = -dvl_velocity_body * k_dvl 
        
        combined_action = action + dvl_compensation_6d / 50.0 
        
        thruster_commands = np.dot(self.tam_inverse, combined_action)
        max_thrust = np.max(np.abs(thruster_commands))
        if max_thrust > 1.0:
            thruster_commands = thruster_commands / max_thrust
            
        actual_physics_force = np.clip(thruster_commands * 50.0 + self.current_disturbance, -100.0, 100.0)
        
        obs_dict_new = self.holo_env.step(actual_physics_force.tolist())

        try:
            for obs in self.obstacles:
                self.holo_env.draw_box(
                    center=list(obs[:3]), 
                    extent=[obs[3], obs[3], obs[3]], 
                    color=[255, 255, 0], 
                    lifetime=0.2         
                )
        except Exception as e:
            print(f"⚠️ step中画石头警告: {e}")
        
        self.current_obs_dict = obs_dict_new
        obs = self._get_obs(obs_dict_new)
        
        current_pos = obs_dict_new["LocationSensor"]
        
        try:
            self.holo_env.draw_line(
                start=self.last_pos.tolist(), 
                end=current_pos.tolist(), 
                color=[0, 255, 0],  
                thickness=3.0, 
                lifetime=0          
            )
        except Exception:
            pass
            
        self.last_pos = current_pos 
        
        distance_to_target = np.linalg.norm(self.target_pos - current_pos)
        roll_deg, pitch_deg, _ = obs_dict_new["RotationSensor"]
        terminated = False
        
        if distance_to_target < 1.0:
            print(f"🚀 精准到达目标点！耗时 {self.current_step} 步")
            terminated = True
        elif distance_to_target > 40.0:
            print("💀 游得太远，迷失在深海。")
            terminated = True
        elif abs(roll_deg) > 60.0 or abs(pitch_deg) > 60.0:
            print(f"💀 姿态翻车! 倾角过大 (Roll: {roll_deg:.1f}°, Pitch: {pitch_deg:.1f}°)")
            terminated = True
            
        return obs, 0.0, terminated, False, {}
    
    def _get_obs(self, obs_dict):
        current_pos = np.array(obs_dict["LocationSensor"])
        velocity_world = np.array(obs_dict["VelocitySensor"])
        roll_deg, pitch_deg, yaw_deg = obs_dict["RotationSensor"]

        roll = np.deg2rad(roll_deg)
        pitch = np.deg2rad(pitch_deg)
        yaw = np.deg2rad(yaw_deg)

        dx = self.target_pos[0] - current_pos[0]
        dy = self.target_pos[1] - current_pos[1]
        dz = self.target_pos[2] - current_pos[2]

        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)

        rel_x_body = dx * cos_y + dy * sin_y   
        rel_y_body = -dx * sin_y + dy * cos_y  
        rel_z_body = dz                      
        relative_pos_body = np.array([rel_x_body, rel_y_body, rel_z_body])

        v_surge = velocity_world[0] * cos_y + velocity_world[1] * sin_y  
        v_sway = -velocity_world[0] * sin_y + velocity_world[1] * cos_y 
        v_heave = velocity_world[2]
        velocity_body = np.array([v_surge, v_sway, v_heave])

        target_yaw = np.arctan2(dy, dx)
        yaw_error = (target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
        rotation_body = np.array([roll, pitch, yaw_error])

        state = np.concatenate([relative_pos_body, velocity_body, rotation_body])
        return state.astype(np.float32)

# ==========================================
# 运行检验！
# ==========================================
import pandas as pd  # 📊 记得在文件最上方加上这句 pandas 导入！

# ==========================================
# 运行检验与双重记录（视频 + CSV数据）
# ==========================================
if __name__ == "__main__":
    env = ROVEvalWrapper(rov_config)
    model = PPO.load("ppo_rov_dynamic_v1_final") # 替换为你的最终模型名
    
    obs, _ = env.reset()
    
    # 1. 启动上帝视角的导航仪 (RRT)
    bounds = [(-20, 20), (-20, 20), (-15, 0)] 
    planner = RRT3D(env.start_pos, env.final_target, env.obstacles, bounds, step_size=2.0)
    
    global_path = planner.plan()
    
    if global_path is None:
        print("无法规划出安全路径，任务取消！")
        exit()
        
    print(f"✅ 规划成功！共生成 {len(global_path)} 个航点。开始执行底层追踪...")
    
    try:
        for i in range(len(global_path)-1):
            env.holo_env.draw_line(global_path[i].tolist(), global_path[i+1].tolist(), color=[0, 0, 255], thickness=2.0, lifetime=0)
    except Exception:
        pass

    # 🎥 2. OpenCV 视频录制初始化
    video_filename = "rov_evaluation_fpv.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    fps = 30.0
    width, height = 512, 512
    out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
    print(f"🔴 开始录制第一人称视频: {video_filename}")

    # =======================================================
    # 📊 新增位置 1：在进入循环前，初始化空列表用于装载数据
    # =======================================================
    trajectory_data = []
    print(f"🔴 开始记录 3D 轨迹数据...")

    # 3. 开始逐个追踪航点
    current_waypoint_index = 1 
    done = False
    
    while not done and current_waypoint_index < len(global_path):
        current_target = global_path[current_waypoint_index]
        env.target_pos = current_target 
        
        obs = env._get_obs(env.current_obs_dict)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        current_pos = env.current_obs_dict["LocationSensor"]
        
        # =======================================================
        # 📊 新增位置 2：在 step() 执行完，立刻把当前坐标存进字典
        # =======================================================
        record = {
            "step": env.current_step,
            "rov_x": current_pos[0],
            "rov_y": current_pos[1],
            "rov_z": current_pos[2]
        }
        
        # 如果测试环境中开启了动态障碍物，一并记录它们的位置
        if hasattr(env, 'dynamic_obs'):
            for idx, obs_data in enumerate(env.dynamic_obs):
                record[f"obs_{idx}_x"] = obs_data['pos'][0]
                record[f"obs_{idx}_y"] = obs_data['pos'][1]
                record[f"obs_{idx}_z"] = obs_data['pos'][2]
                record[f"obs_{idx}_r"] = obs_data['radius']
                
        trajectory_data.append(record)
        # =======================================================

        # 判断是否到达当前航点
        distance_to_waypoint = np.linalg.norm(current_target - current_pos)
        if distance_to_waypoint < 1.5: 
            print(f"到达航点 {current_waypoint_index}，切换下一个目标！")
            current_waypoint_index += 1
            
        # 静态礁石碰撞检测
        for obs_data in env.obstacles:
            if np.linalg.norm(current_pos - obs_data[:3]) < obs_data[3]:
                print("💥 灾难：偏离航线，撞击礁石！")
                done = True
                break
        
        # 📸 视频抽帧逻辑
        if "RGBCamera" in env.current_obs_dict:
            rgba_img = env.current_obs_dict["RGBCamera"]
            rgba_img = np.uint8(rgba_img)
            bgr_img = cv2.cvtColor(rgba_img, cv2.COLOR_RGBA2BGR)
            
            speed = np.linalg.norm(env.current_obs_dict["VelocitySensor"])
            cv2.putText(bgr_img, f"Speed: {speed:.2f} m/s", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(bgr_img, f"Target WP: {current_waypoint_index}/{len(global_path)-1}", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            out.write(bgr_img)
            cv2.imshow("ROV First-Person View", bgr_img)
            cv2.waitKey(1) 
                
        time.sleep(0.02) # 控制展示速度
        
    if current_waypoint_index >= len(global_path):
        print("🎉 完美避障！ROV 成功抵达最终目的地！")

    # 🛑 4. 循环结束，释放视频资源
    out.release()
    cv2.destroyAllWindows()
    print("🎬 视频录制结束！")

    # =======================================================
    # 📊 新增位置 3：一切结束后，将收集到的列表转为 CSV 文件
    # =======================================================
    if len(trajectory_data) > 0:
        df = pd.DataFrame(trajectory_data)
        csv_filename = "rov_trajectory_data.csv"
        df.to_csv(csv_filename, index=False)
        print(f"📄 轨迹数据已成功保存至: {csv_filename} (可用于生成论文 3D 轨迹图)")