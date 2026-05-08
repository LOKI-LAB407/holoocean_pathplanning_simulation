# 文件名: train_phase8.py
from stable_baselines3 import SAC
from callback import RollingCheckpointCallback
from rov_env_new import ROVP2PDynamicWrapper, rov_config


def run_training(total_timesteps=2000000, model_path=None, buffer_path=None):
    print("🌟 正在初始化 [Phase 8: 动态小鱼避障] 训练环境...")

    phase8_curriculum = {
        "target_dist_range": (20.0, 25.0),
        "dz_range": (-10.0, -2.0),
        "num_vortices": 3,
        "max_current": 0.5,
        "num_dynamic_obs": 1,
        "is_static_obs": False,
        "amplitude": 0.1,
        "obstacle_layout": "phase8"
    }

    env = ROVP2PDynamicWrapper(rov_config, curriculum_config=phase8_curriculum)

    custom_objects = {
        "learning_rate": 3e-5,
        "batch_size": 1024,
        "buffer_size": 500000
    }

    if model_path is None:
        model_path = "sac_rov_edge8_phase7_ready.zip"
    # if buffer_path is None:
    #     buffer_path = "./rov_models/sac_rov_phase7_replay_buffer.pkl"

    print("🧠 正在加载 Phase 7 泛化模型，启动动态避障训练...")
    model = SAC.load(
        model_path,
        env=env,
        custom_objects=custom_objects,
        tensorboard_log="./rov_tensorboard/"
    )

    # print(f"📦 正在导入断点经验包: {buffer_path} ...")
    # model.load_replay_buffer(buffer_path)

    checkpoint_callback = RollingCheckpointCallback(
        save_freq=50000,
        save_path='./rov_models/',
        name_prefix='sac_rov_edge8_phase8_dynamic_fish',
        save_replay_buffer=True,
        keep_last=3,
        verbose=1
    )

    print(f"🔥 开启 Phase 8 动态小鱼避障训练 ({total_timesteps} 步)！")
    print(f"   🗺️  距离: {phase8_curriculum['target_dist_range']}m")
    print(f"   🌊 洋流: {phase8_curriculum['max_current']} m/s, {phase8_curriculum['num_vortices']} 涡旋")
    print(f"   🐟 主线动态小鱼: 2-3 个 (r=0.1-0.25m, 逃逸速度 0.5m/s)")

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        tb_log_name="SAC_Edge8_Phase8_DynamicFish",
        reset_num_timesteps=False
    )

    final_model_path = "sac_rov_edge8_phase8_ready"
    final_buffer_path = "sac_rov_phase8_replay_buffer"
    print("✅ Phase 8 训练完成，保存最终模型...")
    model.save(final_model_path)
    model.save_replay_buffer(final_buffer_path)
    return final_model_path, final_buffer_path


if __name__ == "__main__":
    run_training()
