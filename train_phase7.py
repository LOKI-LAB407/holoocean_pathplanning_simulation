# 文件名: train_phase7.py
from stable_baselines3 import SAC
from callback import RollingCheckpointCallback
from rov_env_new import ROVP2PDynamicWrapper, rov_config


def run_training(total_timesteps=700000, model_path=None, buffer_path=None):
    print("🌟 正在初始化 [Phase 7: 泛化避障] 训练环境...")

    phase7_curriculum = {
        "target_dist_range": (20.0, 25.0),
        "dz_range": (-10.0, -2.0),
        "num_vortices": 3,
        "max_current": 0.5,
        "num_dynamic_obs": 1,
        "is_static_obs": False,     # 让漂移小鱼动起来
        "amplitude": 0.1,
        "obstacle_layout": "phase7"
    }

    env = ROVP2PDynamicWrapper(rov_config, curriculum_config=phase7_curriculum)

    custom_objects = {
        "learning_rate": 8e-5,
        "batch_size": 1024,
        "buffer_size": 500000
    }

    if model_path is None:
        model_path = "./rov_models/sac_rov_edge8_phase7_generalize_4025000_steps.zip"
    if buffer_path is None:
        buffer_path = "./rov_models/sac_rov_edge8_phase7_generalize_replay_buffer_4025000_steps.pkl"

    print("🧠 正在加载 Phase 6 模型，启动泛化避障训练...")
    model = SAC.load(
        model_path,
        env=env,
        custom_objects=custom_objects,
        tensorboard_log="./rov_tensorboard/"
    )

    print(f"📦 正在导入断点经验包: {buffer_path} ...")
    model.load_replay_buffer(buffer_path)

    checkpoint_callback = RollingCheckpointCallback(
        save_freq=50000,
        save_path='./rov_models/',
        name_prefix='sac_rov_edge8_phase7_generalize',
        save_replay_buffer=True,
        keep_last=3,
        verbose=1
    )

    print(f"🔥 开启 Phase 7 泛化避障训练 ({total_timesteps} 步)！")
    print(f"   🗺️  距离: {phase7_curriculum['target_dist_range']}m")
    print(f"   🌊 洋流: {phase7_curriculum['max_current']} m/s, {phase7_curriculum['num_vortices']} 涡旋")
    print(f"   🐟 主线小鱼: 2-3 个 (r=0.1-0.25m, 微速漂移 0-0.1m/s)")

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        tb_log_name="SAC_Edge8_Phase7_Generalize",
        reset_num_timesteps=False
    )

    final_model_path = "sac_rov_edge8_phase7_ready"
    final_buffer_path = "sac_rov_phase7_replay_buffer"
    print("✅ Phase 7 训练完成，保存最终模型...")
    model.save(final_model_path)
    model.save_replay_buffer(final_buffer_path)
    return final_model_path, final_buffer_path


if __name__ == "__main__":
    run_training()
