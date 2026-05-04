# 文件名: train_phase5.py
from stable_baselines3 import SAC
from callback import RollingCheckpointCallback
from rov_env_new import ROVP2PDynamicWrapper, rov_config

if __name__ == "__main__":
    print("🌟 正在初始化 [Phase 6: 避障训练] 训练环境...")
    
    # 🌟 绝对对齐的配置字典！
    phase6_curriculum = {
        "target_dist_range": (8.0, 10.0), # 对齐 8~10米
        "dz_range": (-12.0, -2.0),        # 对齐深度
        "num_vortices": 3,                # 对齐 3 个涡旋
        "max_current": 0.4,              # 对齐 0.40 m/s 极速(约为1.5节)
        "num_dynamic_obs": 4,
        "is_static_obs": True,           # 🌟 加入静态障碍物，增加避障难度
        "amplitude": 0.1                 # 🌟 对齐潮汐振幅，极其关键！            
    }
    
    env = ROVP2PDynamicWrapper(rov_config, curriculum_config=phase6_curriculum)
    
    custom_objects = {
        "learning_rate": 1e-4,  
        "batch_size": 1024,
        "buffer_size": 100000,
        "ent_coef": "auto_0.05",   # 保持探索动力，防止局部最优
        
        # 🛡️ 核心安全锁：强制预热步数！
        # 必须 >= batch_size (1024)。模型将静默收集 1024 步数据后，才开始第一次计算。
        # "learning_starts": 1024
    }
    
    
    model_path = f"./rov_models/sac_rov_edge8_phase6_fish01_2575000_steps.zip"
    buffer_path = f"./rov_models/sac_rov_edge8_phase6_fish01_replay_buffer_2575000_steps.pkl"

    # 请确保这里的字符串是你真实要加载的模型名
    print("🧠 正在加载 Phase 6 巅峰模型，准备迎接洋流洗礼...")
    model = SAC.load(
        model_path, 
        env=env,
        custom_objects=custom_objects,
        tensorboard_log="./rov_tensorboard/"
    )

    print(f"📦 正在导入断点经验包 (读取几十万条数据可能需要一些时间和内存): {buffer_path} ...")
    model.load_replay_buffer(buffer_path)
    # ===================================================================================
    
    checkpoint_callback = RollingCheckpointCallback(
        save_freq=25000,
        save_path='./rov_models/',
        name_prefix='sac_rov_edge8_phase6_fish01',
        save_replay_buffer=True,
        keep_last=3,
        verbose=1
)
    
    print("🔥 开启 Phase 6 深海避障训练 (100万步)！")
    
    model.learn(
        total_timesteps=1000000, 
        callback=checkpoint_callback, 
        tb_log_name="SAC_Edge8_Phase6_Avoid_Fish_01", 
        reset_num_timesteps=False  # 继续累积步数，保持训练日志的连续性 
    )
    
    print("✅ Phase 6 训练完成，保存最新抗流模型...")
    model.save("sac_rov_edge8_phase6_ready")
    model.save_replay_buffer("sac_rov_phase6_replay_buffer")