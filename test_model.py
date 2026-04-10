import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import holoocean
from stable_baselines3 import PPO
from rrt_planner import RRT3D # 导入我们刚写的规划器
import time

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
                {"sensor_type": "RotationSensor"}  
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
        self.last_pos = np.zeros(3) # 用于记录上一帧位置以绘制轨迹
        
        # 检验时，你可以把洋流调大，看看它能不能抗住！这里先设为 0 看它的纯净轨迹
        # 0.0 表示没有洋流干扰，完全由模型自己控制
        self.max_current_power = 0.0 
        # 平静水面
        # self.max_current_power = 3.0 
        # 中等洋流
        # self.max_current_power = 6.0
        # 极端洋流
        # self.max_current_power = 10.0 
        self.current_disturbance = np.zeros(8)

    def reset(self, seed=None, options=None):
        self.current_step = 0
        
        # 1. 设定起点和终点
        obs_dict = self.holo_env.reset()
        self.current_obs_dict = obs_dict
        self.start_pos = np.array(obs_dict["LocationSensor"])
        self.final_target = np.array([15.0, 15.0, -8.0]) # 全局终点
        
        # 2. 生成虚拟障碍物 (x, y, z, radius)
        # 这里故意在直线路径上放两块大石头拦路！
        self.obstacles = [
            (5.0, 5.0, -5.0, 3.0),   # 坐标(5,5,-5)，半径3米的石头
            (10.0, 10.0, -8.0, 4.0)  # 坐标(10,10,-8)，半径4米的石头
        ]
        
        # 3. 开启洋流
        # self.current_disturbance = np.random.uniform(-10.0, 10.0, size=(8,))
                # 随机生成本回合的洋流干扰
        if self.max_current_power == 0.0:
            print(f"\n🌊 本次测试：无洋流干扰，纯净环境！")
        else:
            print(f"\n🌊 本次测试：洋流干扰强度为 {self.max_current_power}！")
        self.current_disturbance = np.random.uniform(-self.max_current_power, self.max_current_power, size=(8,))
        
        # 4. 在 HoloOcean 画面中渲染终点和石头 (方便你肉眼观察)
        try:
            # 画终点 (红色框)
            self.holo_env.draw_box(center=self.final_target.tolist(), extent=[1.0, 1.0, 1.0], color=[255, 0, 0], lifetime=0)
            
            # 画石头 (灰色球体/多边形，用线框代替)
            # 🌟 修复版：强行转换为 list，并涂上亮黄色！
            for obs in self.obstacles:
                # 注意这里的 list(obs[:3])
                self.holo_env.draw_box(
                    center=list(obs[:3]), 
                    extent=[obs[3], obs[3], obs[3]], 
                    color=[255, 255, 0], 
                    lifetime=0.1
                ) 
        except Exception as e:
            print(f"⚠️ 画石头警告: {e}")
            
        return self._get_obs(obs_dict), {}

    # def reset(self, seed=None, options=None):
    #     self.current_step = 0
    #     self.previous_action = np.zeros(6) 
        
    #     # 随机生成目标点
    #     rx = np.random.uniform(-20.0, 20.0)
    #     ry = np.random.uniform(-20.0, 20.0)
    #     rz = np.random.uniform(-15.0, -2.0)
    #     self.target_pos = np.array([rx, ry, rz])
        
    #     # 随机生成本回合的洋流干扰
    #     if self.max_current_power == 0.0:
    #         print(f"\n🌊 本次测试：无洋流干扰，纯净环境！")
    #     else:
    #         print(f"\n🌊 本次测试：洋流干扰强度为 {self.max_current_power}！")
    #     self.current_disturbance = np.random.uniform(-self.max_current_power, self.max_current_power, size=(8,))
        
    #     obs_dict = self.holo_env.reset()
        
    #     # 👇 关键修复 1：必须缓存初始状态，供 DVL 读取！
    #     self.current_obs_dict = obs_dict 
    #     self.last_pos = obs_dict["LocationSensor"]
        
    #     # 绘制目标点
    #     try:
    #         self.holo_env.draw_box(center=self.target_pos.tolist(), extent=[1.0, 1.0, 1.0], color=[255, 0, 0], lifetime=0)
    #     except Exception:
    #         pass 
            
    #     return self._get_obs(obs_dict), {}

    def step(self, action):
        self.current_step += 1
        
        # ==========================================
        # 👇 关键修复 2：必须把训练时的 DVL 补偿原封不动搬过来！
        # ==========================================
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
        
        # 组合 RL 意图与 DVL 补偿
        combined_action = action + dvl_compensation_6d / 50.0 
        
        # 推力分配与执行 (使用 combined_action)
        thruster_commands = np.dot(self.tam_inverse, combined_action)
        max_thrust = np.max(np.abs(thruster_commands))
        if max_thrust > 1.0:
            thruster_commands = thruster_commands / max_thrust
            
        actual_physics_force = np.clip(thruster_commands * 50.0 + self.current_disturbance, -100.0, 100.0)
        
        obs_dict_new = self.holo_env.step(actual_physics_force.tolist())

        # 🌟 强行每帧渲染石头，确保持久可见
        try:
            for obs in self.obstacles:
                # 👇 关键修复：加上 list()！
                self.holo_env.draw_box(
                    center=list(obs[:3]), 
                    extent=[obs[3], obs[3], obs[3]], 
                    color=[255, 255, 0], # 统一换成扎眼的黄色
                    lifetime=0.2         # 既然每帧(30Hz)都在画，存活 0.2 秒足够了
                )
        except Exception as e:
            # 永远不要在调试画图时用 pass，把错误逼出来！
            print(f"⚠️ step中画石头警告: {e}")
        
        # 👇 关键修复 3：更新缓存字典
        self.current_obs_dict = obs_dict_new
        obs = self._get_obs(obs_dict_new)
        
        current_pos = obs_dict_new["LocationSensor"]
        
        # ==========================================
        # 🌟 炫酷视觉效果：绘制 ROV 真实行驶轨迹 🌟
        # ==========================================
        try:
            self.holo_env.draw_line(
                start=self.last_pos.tolist(), 
                end=current_pos.tolist(), 
                color=[0, 255, 0],  # 荧光绿
                thickness=3.0,      
                lifetime=0          
            )
        except Exception:
            pass
            
        self.last_pos = current_pos 
        
        # 检验代码：加入翻车判断，否则它沉底了你还在傻等
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
        # 1. 获取世界坐标系下的原始数据
        current_pos = np.array(obs_dict["LocationSensor"])
        velocity_world = np.array(obs_dict["VelocitySensor"])
        roll_deg, pitch_deg, yaw_deg = obs_dict["RotationSensor"]

        roll = np.deg2rad(roll_deg)
        pitch = np.deg2rad(pitch_deg)
        yaw = np.deg2rad(yaw_deg)

        # 2. 相对位置转换：世界 -> 机体
        dx = self.target_pos[0] - current_pos[0]
        dy = self.target_pos[1] - current_pos[1]
        dz = self.target_pos[2] - current_pos[2]

        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)

        # 【核心数学：2D 旋转矩阵映射】
        # rel_x_body > 0 表示目标在机头正前方，< 0 在机尾
        # rel_y_body > 0 表示目标在机身右侧，< 0 在左侧
        rel_x_body = dx * cos_y + dy * sin_y   
        rel_y_body = -dx * sin_y + dy * cos_y  
        rel_z_body = dz                        # 深度一般不随偏航角旋转
        
        relative_pos_body = np.array([rel_x_body, rel_y_body, rel_z_body])

        # 3. 速度转换：世界 -> 机体 (告诉神经网络它当前真实的滑行方向)
        v_surge = velocity_world[0] * cos_y + velocity_world[1] * sin_y  # 向前滑行速度
        v_sway = -velocity_world[0] * sin_y + velocity_world[1] * cos_y # 侧向滑行速度
        v_heave = velocity_world[2]
        
        velocity_body = np.array([v_surge, v_sway, v_heave])

        # 4. 姿态转换：抛弃绝对 Yaw，改用相对目标的 Yaw 误差！
        target_yaw = np.arctan2(dy, dx)
        yaw_error = (target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
        
        # 现在，神经网络只知道自己倾斜了多少，以及距离目标方向偏了多少角度
        rotation_body = np.array([roll, pitch, yaw_error])

        # 5. 拼装成全新的 9 维机体视角 State
        # 此时的 State 维度依然是 9，完全不需要修改 __init__ 里的 observation_space！
        state = np.concatenate([relative_pos_body, velocity_body, rotation_body])
        
        return state.astype(np.float32)

# ==========================================
# 运行检验！
# ==========================================
if __name__ == "__main__":
    env = ROVEvalWrapper(rov_config)
    model = PPO.load("ppo_rov_p2p_v4") # 你的最强大脑
    
    obs, _ = env.reset()
    
    # 1. 启动上帝视角的导航仪 (RRT)
    bounds = [(-20, 20), (-20, 20), (-15, 0)] # 设定搜索空间边界
    planner = RRT3D(env.start_pos, env.final_target, env.obstacles, bounds, step_size=2.0)
    
    global_path = planner.plan()
    
    if global_path is None:
        print("无法规划出安全路径，任务取消！")
        exit()
        
    print(f"✅ 规划成功！共生成 {len(global_path)} 个航点。开始执行底层追踪...")
    
    # 在 3D 画面中用蓝线画出这条规划好的全局路径！
    try:
        for i in range(len(global_path)-1):
            env.holo_env.draw_line(global_path[i].tolist(), global_path[i+1].tolist(), color=[0, 0, 255], thickness=2.0, lifetime=0)
    except Exception:
        pass

    # 2. 开始逐个追踪航点
    current_waypoint_index = 1 # 索引 0 是起点，我们直接追索引 1
    done = False
    
    while not done and current_waypoint_index < len(global_path):
        # 设定当前的“胡萝卜”
        current_target = global_path[current_waypoint_index]
        
        # 将当前的胡萝卜坐标塞给环境 (环境的 _get_obs 会自动把它转成机体相对坐标喂给神经网络！)
        env.target_pos = current_target 
        
        # 重新获取观测值 (以新的胡萝卜为目标)
        obs = env._get_obs(env.current_obs_dict)
        
        # PPO 大脑输出抗流推力
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        # 判断是否吃到了当前的胡萝卜？
        current_pos = env.current_obs_dict["LocationSensor"]
        distance_to_waypoint = np.linalg.norm(current_target - current_pos)
        
        if distance_to_waypoint < 1.5: # 触碰半径：距离胡萝卜不到 1.5 米就算吃到
            print(f"到达航点 {current_waypoint_index}，切换下一个目标！")
            current_waypoint_index += 1
            
        # 碰撞检测：如果在追踪过程中撞到了石头，直接判定失败
        for obs_data in env.obstacles:
            if np.linalg.norm(current_pos - obs_data[:3]) < obs_data[3]:
                print("💥 灾难：偏离航线，撞击礁石！")
                done = True
                break
                
        time.sleep(0.02)
        
    if current_waypoint_index >= len(global_path):
        print("🎉 完美避障！ROV 成功抵达最终目的地！")
