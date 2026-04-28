# 文件名: train_phase5.py
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from rov_env import ROVP2PDynamicWrapper, rov_config

if __name__ == "__main__":
    print("🌟 正在初始化 [Phase 6: 避障训练] 训练环境...")
    
    # 🌟 绝对对齐的配置字典！
    phase5_curriculum = {
        "target_dist_range": (5.0, 8.0), # 对齐 8~5米
        "dz_range": (-15.0, -2.0),        # 对齐深度
        "num_vortices": 3,                # 对齐 3 个涡旋
        "max_current": 0.70,              # 对齐 0.70 m/s 极速(约为1.5节)
        "amplitude": 0.1,                 # 🌟 对齐潮汐振幅，极其关键！
        "num_dynamic_obs": 0              
    }
    
    env = ROVP2PDynamicWrapper(rov_config, curriculum_config=phase5_curriculum)
    
    custom_objects = {
        "learning_rate": 8e-5,  
        "batch_size": 1024,
        "buffer_size": 500000 
    }
    
    # 请确保这里的字符串是你真实要加载的模型名
    print("🧠 正在加载 Phase 6 巅峰模型，准备迎接洋流洗礼...")
    model = SAC.load(
        "sac_rov_edge8_mild_current_no_fish_normal_dis", 
        env=env,
        custom_objects=custom_objects,
        tensorboard_log="./rov_tensorboard/"
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=25000, 
        save_path='./rov_models/',
        name_prefix='sac_rov_edge8_phase6_fish01',
        save_replay_buffer=True # 保留保存账本的好习惯
    )
    
    print("🔥 开启 Phase 6 深海抗流训练 (80万步)！")
    
    model.learn(
        total_timesteps=800000, 
        callback=checkpoint_callback, 
        tb_log_name="SAC_Edge8_Phase6_Avoid_Fish_01", 
        reset_num_timesteps=False 
    )
    
    print("✅ Phase 6 训练完成，保存最新抗流模型...")
    model.save("sac_rov_edge8_phase6_ready")
    model.save_replay_buffer("sac_rov_phase6_replay_buffer")