# 文件名: train_phase6_new.py
from stable_baselines3 import SAC
from callback import RollingCheckpointCallback
from rov_env_new import ROVP2PDynamicWrapper, rov_config


def run_training(total_timesteps=100000, model_path=None, buffer_path=None):
    print("🌟 正在初始化 [Phase 5: 续训练] 训练环境...")

    phase5_curriculum = {
        "target_dist_range": (8.0, 10.0),
        "dz_range": (-10.0, -2.0),
        "num_vortices": 3,
        "max_current": 0.7,
        "num_dynamic_obs": 0,
        # "is_static_obs": True,
        "amplitude": 0.1
    }

    env = ROVP2PDynamicWrapper(rov_config, curriculum_config=phase5_curriculum)

    custom_objects = {
        "learning_rate": 8e-5,
        "batch_size": 1024,
        "buffer_size": 500000
    }

    if model_path is None:
        model_path = "sac_rov_edge8_extreme_current_no_fish_normal_dis.zip"
    # if buffer_path is None:
    #     buffer_path = "./rov_models/sac_rov_edge8_phase5_fish01_replay_buffer_2925000_steps.pkl"

    print("🧠 正在加载 Phase 5 模型，准备迎接洋流洗礼...")
    model = SAC.load(
        model_path,
        env=env,
        custom_objects=custom_objects,
        tensorboard_log="./rov_tensorboard/"
    )

    # print(f"📦 正在导入断点经验包: {buffer_path} ...")
    # model.load_replay_buffer(buffer_path)

    checkpoint_callback = RollingCheckpointCallback(
        save_freq=25000,
        save_path='./rov_models/',
        name_prefix='sac_rov_edge8_phase5_fish01',
        save_replay_buffer=True,
        keep_last=3,
        verbose=1
    )

    print(f"🔥 开启 Phase 5 深海避障训练 ({total_timesteps} 步)！")

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        tb_log_name="SAC_Edge8_Phase5_ExtremeCurrent_07_0",
        reset_num_timesteps=False
    )

    final_model_path = "sac_rov_edge8_phase5_ready"
    final_buffer_path = "sac_rov_phase5_replay_buffer"
    print("✅ Phase 5 训练完成，保存最新模型...")
    model.save(final_model_path)
    model.save_replay_buffer(final_buffer_path)
    return final_model_path, final_buffer_path


if __name__ == "__main__":
    run_training()
