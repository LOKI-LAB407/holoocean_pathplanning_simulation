import os
import numpy as np
import holoocean
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# 1. 保护 C 盘
os.environ["HOLOOCEAN_PATH"] = "D:\\HoloOceanEngine"
os.environ["HOLOOCEAN_SYSTEM_PATH"] = "D:\\HoloOceanEngine"

# (此处省略 get_rov_mixing_matrix()、rov_config 和 ROVP2PWrapper 的代码，
# 请直接把 train_rov_p2p.py 里这三个部分完全复制过来)
# ... [复制你的环境代码] ...

def get_rov_mixing_matrix():
    """
    基于 ArduSub 标准 8 推进器矢量构型 (BlueROV2 Heavy / Edge8) 的直接混合矩阵。
    行: 8 个底层推进器
    列: 6 个高层自由度期望指令 [Surge(前后), Sway(左右), Heave(上下), Roll(横滚), Pitch(俯仰), Yaw(偏航)]
    """
    M = np.zeros((8, 6))
    
    # ==========================================
    # 1. 垂直推进器组 (T1~T4) -> 专职负责 Z轴及姿态
    # 设定：+Heave(上浮), +Roll(右倾), +Pitch(抬头)
    # ==========================================
    #             [Surge, Sway, Heave, Roll, Pitch, Yaw]
    M[0, :] = np.array([0,  0,    1,   -1,    1,    0])  # T1: 垂直前右
    M[1, :] = np.array([0,  0,    1,    1,    1,    0])  # T2: 垂直前左
    M[2, :] = np.array([0,  0,    1,   -1,   -1,    0])  # T3: 垂直后右
    M[3, :] = np.array([0,  0,    1,    1,   -1,    0])  # T4: 垂直后左

    # ==========================================
    # 2. 水平交叉推进器组 (T5~T8) -> 专职负责 XY面移动
    # 设定：+Surge(前进), +Sway(右移), +Yaw(右转)
    # ==========================================
    #             [Surge, Sway, Heave, Roll, Pitch, Yaw]
    M[4, :] = np.array([1, -1,    0,    0,    0,   -1])  # T5: 水平前右
    M[5, :] = np.array([1,  1,    0,    0,    0,    1])  # T6: 水平前左
    M[6, :] = np.array([1,  1,    0,    0,    0,   -1])  # T7: 水平后右
    M[7, :] = np.array([1, -1,    0,    0,    0,    1])  # T8: 水平后左
    
    return M

# ==========================================
# 第一部分：HoloOcean 蓝图配置
# ==========================================
rov_config = {
    "name": "ROV_P2P_Task",
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
# 第二部分：自定义环境包装器 
# ==========================================
class ROVP2PWrapper(gym.Env):
    def __init__(self, config):
        super(ROVP2PWrapper, self).__init__()
        self.holo_env = holoocean.make(scenario_cfg=config)
        
        # 👇【修改 1】：动作空间降维！现在 PPO 只需要输出 6 个方向的期望力
        # [Surge(前后), Sway(左右), Heave(上下), Roll(横滚), Pitch(俯仰), Yaw(偏航)]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        
        # 👇【修改 2】：把刚才算好的逆矩阵存进大脑
        self.tam_inverse = get_rov_mixing_matrix()
        
        self.max_steps = 1000  
        self.current_step = 0
        self.max_tilt_rad = 1.0        
        self.water_surface_z = -0.5    
        self.water_bottom_z = -40.0    
        
        self.target_pos = np.zeros(3)
        self.previous_distance = 0.0
        self.previous_action = np.zeros(6) # 记得这里也要改成 6
        self.current_disturbance = np.zeros(8) 
        
        self.max_current_power = 10.0

    def reset(self, seed=None, options=None):
        self.current_step = 0
        
        # 👇 就是这里！把原来的 8 改成 6！
        self.previous_action = np.zeros(6) 
        
        # 1. 随机生成目标点 (X/Y 在周围 20 米内，Z 在水下 2 到 15 米之间)
        rx = np.random.uniform(-20.0, 20.0)
        ry = np.random.uniform(-20.0, 20.0)
        rz = np.random.uniform(-15.0, -2.0)
        self.target_pos = np.array([rx, ry, rz])
        
        # 2. 随机生成本回合的洋流 (如果 max_current_power 是 0，这里就是全 0)
        self.current_disturbance = np.random.uniform(-self.max_current_power, self.max_current_power, size=(8,))
        
        # 3. 让引擎画出目标点
        try:
            self.holo_env.draw_box(center=self.target_pos.tolist(), extent=[0.5, 0.5, 0.5], color=[255, 0, 0], lifetime=0)
            self.holo_env.draw_line(start=[0,0,0], end=self.target_pos.tolist(), color=[0, 255, 0], lifetime=0)
        except Exception as e:
            pass 
            
        obs_dict = self.holo_env.reset()

        # 👇 新增这一行：将初始状态保存下来，作为第一帧动作执行前的参考！
        self.current_obs_dict = obs_dict
        
        # 初始化上一帧的距离
        initial_pos = obs_dict["LocationSensor"]
        self.previous_distance = np.linalg.norm(self.target_pos - initial_pos)
        
        return self._get_obs(obs_dict), {}

    def step(self, action):
        self.current_step += 1
        
        # ==========================================
        # 1. DVL 数据提取与【绝对->相对坐标】转换！(修复致命 Bug)
        # ==========================================
        # 强制转为 numpy array 防止类型运算报错
        dvl_velocity_world = np.array(self.current_obs_dict["VelocitySensor"]) 
        _, _, yaw_deg = self.current_obs_dict["RotationSensor"]
        yaw_rad = np.deg2rad(yaw_deg)

        # 构建 2D 旋转矩阵 (将世界坐标系速度，旋转到机体局部坐标系)
        cos_y = np.cos(yaw_rad)
        sin_y = np.sin(yaw_rad)
        
        vx_world, vy_world, vz_world = dvl_velocity_world[0], dvl_velocity_world[1], dvl_velocity_world[2]
        
        # 核心数学转换！
        vx_body = vx_world * cos_y + vy_world * sin_y   # 机头方向的相对速度 (Surge)
        vy_body = -vx_world * sin_y + vy_world * cos_y  # 机身侧向的相对速度 (Sway)
        vz_body = vz_world                              # 垂直方向假设姿态平稳，无需过度旋转 (Heave)
        
        dvl_velocity_body = np.array([vx_body, vy_body, vz_body])
        
        # ==========================================
        # 2. DVL 反向推力补偿 (Feedforward) 
        # ==========================================
        k_dvl = 2.5 
        dvl_compensation_6d = np.zeros(6)
        # 现在，我们用机身坐标系的速度来反推，万无一失！
        dvl_compensation_6d[0:3] = -dvl_velocity_body * k_dvl 
        
        # 将 RL 的智力意图 与 物理基底的肌肉补偿 叠加
        combined_action = action + dvl_compensation_6d / 50.0 
        
        # ==========================================
        # 3. 推力分配与防饱和保护
        # ==========================================
        thruster_commands = np.dot(self.tam_inverse, combined_action)
        max_thrust = np.max(np.abs(thruster_commands))
        if max_thrust > 1.0:
            thruster_commands = thruster_commands / max_thrust
            
        actual_physics_force = np.clip(thruster_commands * 50.0 + self.current_disturbance, -100.0, 100.0)
        
        # ==========================================
        # 4. 执行力矩，获取全新状态并更新缓存！
        # ==========================================
        obs_dict_new = self.holo_env.step(actual_physics_force.tolist())
        self.current_obs_dict = obs_dict_new  # 👈 保证时间轴闭环的核心
        
        # ==========================================
        # 5. 提取用于计算 Reward 的状态
        # ==========================================
        current_pos = np.array(obs_dict_new["LocationSensor"])
        velocity_world = np.array(obs_dict_new["VelocitySensor"]) # Reward 依然需要在世界坐标下评判
        roll_deg, pitch_deg, yaw_deg = obs_dict_new["RotationSensor"]

        roll = np.deg2rad(roll_deg)
        pitch = np.deg2rad(pitch_deg)
        yaw = np.deg2rad(yaw_deg)
        speed = np.linalg.norm(velocity_world)
        
        # 相对目标的信息
        dx = self.target_pos[0] - current_pos[0]
        dy = self.target_pos[1] - current_pos[1]
        target_yaw = np.arctan2(dy, dx)
        yaw_error = (target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
        distance_to_target = np.linalg.norm(self.target_pos - current_pos)
        
        reward = 0.0
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # 【正常计算进度奖励】
        progress = self.previous_distance - distance_to_target
        reward += progress * 10.0  
        self.previous_distance = distance_to_target 

        # 【航向角打转惩罚】
        yaw_penalty_weight = 2.0
        reward -= (yaw_error**2) * yaw_penalty_weight

        # 【抗流平稳性奖励】
        velocity_unit = velocity_world / (speed + 1e-6)
        target_dir = (self.target_pos - current_pos) / (distance_to_target + 1e-6)
        alignment = np.dot(velocity_unit, target_dir)
        reward += max(0, alignment) * 0.5
        
        # 【姿态惩罚】
        # 1. 连续的高斯平滑奖励 (越接近水平，得分越高)
        # 当 roll/pitch 为 0 时，得到 1.0 的满分；偏离越大，分数呈指数衰减
        attitude_score = np.exp(- (roll**2 + pitch**2) / 0.1)
        
        # 2. 严厉的越界惩罚 (Soft Constraint)
        # 允许 ±5 度的微小摇晃，超过 5 度则开始线性扣分
        roll_penalty = max(0.0, abs(roll_deg) - 5.0) * 0.5
        pitch_penalty = max(0.0, abs(pitch_deg) - 5.0) * 0.5
        
        # 将姿态分数融入总奖励 (权重可根据你对姿态的要求自行微调)
        k_attitude_bonus = 2.0  # 保持水平的奖励系数
        k_attitude_penalty = 1.0 # 摇晃的惩罚系数
        
        reward += (attitude_score * k_attitude_bonus) - (roll_penalty + pitch_penalty) * k_attitude_penalty
        # =======================================
            
        # 【发呆惩罚】
        if distance_to_target > 1.0 and speed < 0.1:
            reward -= 0.5  
        else:
            reward -= 0.1  

        # 【动作惩罚】
        reward -= np.sum(np.square(action)) * 0.02
        action_diff = np.sum(np.square(action - self.previous_action))
        reward -= action_diff * 0.1
        self.previous_action = np.copy(action)
        
        # 【死亡红线判定】
        if self.current_step > 10: 
            if abs(roll) > self.max_tilt_rad or abs(pitch) > self.max_tilt_rad:
                reward -= 1000.0
                terminated = True
                print(f"💀 [重启原因] 姿态翻车! 倾角 (Roll: {roll_deg:.1f}°, Pitch: {pitch_deg:.1f}°)")
                
            elif current_pos[2] > self.water_surface_z:
                reward -= 1000.0
                terminated = True
                print(f"💀 [重启原因] 冲出水面! 当前 Z: {current_pos[2]:.2f}")
                
            elif current_pos[2] < self.water_bottom_z:
                reward -= 1000.0
                terminated = True
                print(f"💀 [重启原因] 沉入海底! 当前 Z: {current_pos[2]:.2f}")
                
            elif distance_to_target < 0.5:
                if abs(roll) < 0.2 and abs(pitch) < 0.2:
                    reward += 1000.0 
                    print(f"✅ 完美到达目标！耗时 {self.current_step} 步")
                else:
                    reward += 500.0  
                    print(f"⚠️ 翻滚着撞线，分数减半")
                terminated = True
            
        return self._get_obs(obs_dict_new), reward, terminated, truncated, {}

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

    def close(self):
        pass



###############################################################################################

if __name__ == "__main__":
    print("初始化强化学习环境...")
    env = ROVP2PWrapper(rov_config)
    
    # 🌟 续训核心秘籍：给环境加压！
    env.max_current_power = 3.0 
    
    # ==========================================
    # 核心：加载旧模型，覆盖新环境
    # ==========================================
    MODEL_PATH = "ppo_rov_p2p_v3_final"  # 你上一次保存的模型名字
    
    print(f"正在唤醒沉睡的 AI 大脑：{MODEL_PATH} ...")
    
    # 重新加载模型，并且强制绑定现在的加压环境 (env=env)
    # 降低学习率 (比如从 0.0003 降到 0.0001)，因为现在是“微调”阶段，步子迈大了容易扯着蛋，忘掉以前学的东西
    model = PPO.load(MODEL_PATH, env=env, custom_objects={"learning_rate": 0.0001}, tensorboard_log="./rov_tensorboard/")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=20000, 
        save_path='./rov_models/',
        name_prefix='ppo_rov_p2p_v4' # 换个名字防止覆盖旧的 checkpoint
    )
    
    print("🔥 开启抗流微调训练！")
    
    # reset_num_timesteps=False 极其重要！
    # 它会让 TensorBoard 的曲线接着之前的继续画，而不是从 0 重新开始！
    model.learn(
        total_timesteps=300000, 
        callback=checkpoint_callback, 
        reset_num_timesteps=False, 
        tb_log_name="PPO_Finetune_3" # 在 TensorBoard 里新建一条线
    )
    
    print("✅ 续训完成，保存进化版模型...")
    model.save("ppo_rov_p2p_v4_final")