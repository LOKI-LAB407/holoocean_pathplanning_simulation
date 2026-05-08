import os
import time
import argparse
import numpy as np
import holoocean
from stable_baselines3 import SAC

os.environ["HOLOOCEAN_PATH"] = "D:\\HoloOceanEngine"
os.environ["HOLOOCEAN_SYSTEM_PATH"] = "D:\\HoloOceanEngine"

from rov_env_new import ROVP2PDynamicWrapper, rov_config

test_curriculum = {
    "target_dist_range": (14.0, 16.0),
    "dz_range": (-10.0, -2.0),
    "num_vortices": 3,
    "max_current": 0.4,
    "num_dynamic_obs": 1,
    "is_static_obs": True,
    "amplitude": 0.1,
    "show_viewport": True,
}


def draw_obstacle_box(env, obs_obj):
    c = obs_obj['pos']
    r = obs_obj['radius']
    corners = np.array([
        [c[0]-r, c[1]-r, c[2]-r], [c[0]-r, c[1]-r, c[2]+r],
        [c[0]-r, c[1]+r, c[2]-r], [c[0]-r, c[1]+r, c[2]+r],
        [c[0]+r, c[1]-r, c[2]-r], [c[0]+r, c[1]-r, c[2]+r],
        [c[0]+r, c[1]+r, c[2]-r], [c[0]+r, c[1]+r, c[2]+r],
    ])
    edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    for a, b in edges:
        env.holo_env.draw_line(corners[a].tolist(), corners[b].tolist(),
                               color=[0, 0, 255], thickness=1.5, lifetime=0)


def run_test(model_path, episodes=5):
    print(f"🧠 加载模型: {model_path}")
    model = SAC.load(model_path)
    print(f"🌊 初始化训练环境...")
    env = ROVP2PDynamicWrapper(rov_config, curriculum_config=test_curriculum)

    success = 0
    for ep in range(episodes):
        obs, _ = env.reset()
        target = env.target_pos.copy()
        start_pos = np.array(env.current_obs_dict["LocationSensor"])

        print(f"\n{'='*40}")
        print(f"  Episode {ep+1}: {np.round(start_pos,1)} → {np.round(target,1)}")
        print(f"  距离: {np.linalg.norm(target-start_pos):.1f}m")

        # 绘制起点→终点绿线
        env.holo_env.draw_line(start_pos.tolist(), target.tolist(),
                               color=[0, 255, 0], thickness=2.0, lifetime=0)
        # 绘制障碍物红色线框
        for o in env.dynamic_obs:
            draw_obstacle_box(env, o)
            print(f"  障碍: pos={np.round(o['pos'],1)}, r={o['radius']:.2f}m, "
                  f"距起点={np.linalg.norm(o['pos']-start_pos):.1f}m")

        prev_p = start_pos.copy()
        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            curr_p = np.array(env.current_obs_dict["LocationSensor"])
            # 绘制 ROV 红色轨迹
            env.holo_env.draw_line(prev_p.tolist(), curr_p.tolist(),
                                   color=[255, 0, 0], thickness=4.0, lifetime=0)
            prev_p = curr_p.copy()

            if terminated:
                dist = np.linalg.norm(curr_p - target)
                if dist < 0.8:
                    print(f"  ✅ 到达！{step+1} 步")
                    success += 1
                else:
                    print(f"  ❌ 碰撞/翻车！{step+1} 步, 距目标 {dist:.1f}m")
                    # 画个黄色十字标记碰撞点
                    r = 1.0
                    env.holo_env.draw_line(
                        [curr_p[0]-r, curr_p[1], curr_p[2]], [curr_p[0]+r, curr_p[1], curr_p[2]],
                        color=[255, 255, 0], thickness=6.0, lifetime=0)
                    env.holo_env.draw_line(
                        [curr_p[0], curr_p[1]-r, curr_p[2]], [curr_p[0], curr_p[1]+r, curr_p[2]],
                        color=[255, 255, 0], thickness=6.0, lifetime=0)
                break
        else:
            curr_p = np.array(env.current_obs_dict["LocationSensor"])
            print(f"  ⏰ 超时, 距目标 {np.linalg.norm(curr_p-target):.1f}m")

    print(f"\n{'='*40}")
    print(f"  结果: {success}/{episodes} 成功")
    print(f"{'='*40}")
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="sac_rov_edge8_phase6_ready.zip")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    run_test(args.model, episodes=args.episodes)
