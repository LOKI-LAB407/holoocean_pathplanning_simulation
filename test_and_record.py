import os
import sys
import time
import argparse
import cv2
import numpy as np
import holoocean
from stable_baselines3 import SAC
from rov_env_new import ROVP2PDynamicWrapper

# ==========================================
# 0. 引擎路径配置
# ==========================================
os.environ["HOLOOCEAN_PATH"] = "D:\\HoloOceanEngine"
os.environ["HOLOOCEAN_SYSTEM_PATH"] = "D:\\HoloOceanEngine"

# ==========================================
# 1. 测试配置 (训练配置 + 跟随相机)
# ==========================================
rov_test_config = {
    "name": "ROV_P2P_Test",
    "world": "OpenWater",
    "package_name": "Ocean",
    "main_agent": "my_rov",
    "frames_per_sec": 30,
    "agents": [{
        "agent_name": "my_rov",
        "agent_type": "HoveringAUV",
        "control_scheme": 0,
        "location": [0.0, 0.0, -6.0],
        "sensors": [
            {"sensor_type": "LocationSensor"},
            {"sensor_type": "VelocitySensor"},
            {"sensor_type": "RotationSensor"},
            {"sensor_type": "IMUSensor", "socket": "IMUSocket"},
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

# ==========================================
# 2. 测试课程配置 (对齐 train_phase6_new.py)
# ==========================================
test_curriculum = {
    "target_dist_range": (8.0, 10.0),
    "dz_range": (-10.0, -2.0),
    "num_vortices": 3,
    "max_current": 0.4,
    "num_dynamic_obs": 1,
    "is_static_obs": True,
    "amplitude": 0.1
}


class ROVTestWrapper(ROVP2PDynamicWrapper):
    """在训练环境基础上增加相机传感器和定点测试能力。"""

    def __init__(self, config, curriculum_config=None):
        super().__init__(config, curriculum_config)

    def reset_for_test(self, start_pos, target_pos):
        """定点测试重置：固定起点和目标航点，障碍物沿测试航线生成。"""
        self.current_step = 0
        self.previous_action = np.zeros(6)
        self.target_pos = np.array(target_pos, dtype=np.float64)

        self.ocean_sim.reset()
        obs_dict = self.holo_env.reset()

        self.holo_env.agents["my_rov"].teleport(start_pos)
        self.current_obs_dict = obs_dict

        _, _, yaw_init = np.deg2rad(obs_dict["RotationSensor"])
        self.previous_yaw = yaw_init

        initial_pos = np.array(obs_dict["LocationSensor"])
        self.previous_distance = np.linalg.norm(self.target_pos - initial_pos)

        self._generate_dynamic_obstacles()
        return self._get_obs(obs_dict)

    def switch_waypoint(self, new_target, current_pos):
        """切换航点并同步 previous_distance，避免 progress 奖励跳变。"""
        self.target_pos = np.array(new_target, dtype=np.float64)
        self.previous_distance = np.linalg.norm(self.target_pos - current_pos)


# ==========================================
# 3. 参数化平滑曲线生成器
# ==========================================
def generate_smooth_curve(start_pos, end_pos, num_points=50):
    t = np.linspace(0, 1, num_points)
    path = np.zeros((num_points, 3))
    path[:, 0] = start_pos[0] + t * (end_pos[0] - start_pos[0])
    path[:, 2] = start_pos[2] + t * (end_pos[2] - start_pos[2])
    path[:, 1] = start_pos[1] + t * (end_pos[1] - start_pos[1]) + np.sin(t * np.pi) * 2.5
    return path


# ==========================================
# 4. 测试核心逻辑
# ==========================================
def run_test(model_path, video_path=None, headless=False, start_pos=None, goal_pos=None):
    """
    运行航点跟踪 + 避障测试。

    Args:
        model_path: 模型文件路径
        video_path: 视频输出路径 (None 则自动生成)
        headless: True 时关闭 cv2.imshow，仅写视频文件
        start_pos: 起点 (None 则默认 [0, 0, -6])
        goal_pos:  终点 (None 则默认 [12, 12, -12])

    Returns:
        dict: 测试指标
    """
    if start_pos is None:
        start_pos = np.array([0.0, 0.0, -6.0])
    if goal_pos is None:
        goal_pos = np.array([12.0, 12.0, -12.0])
    if video_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_path = f"test_output_{timestamp}.avi"

    print("🌊 正在初始化测试环境 (对齐 Phase 6 训练配置)...")
    print(f"   洋流: {test_curriculum['max_current']} m/s, "
          f"涡旋: {test_curriculum['num_vortices']}, "
          f"障碍物: {test_curriculum['num_dynamic_obs']} 静态")
    env = ROVTestWrapper(rov_test_config, curriculum_config=test_curriculum)

    print(f"🧠 正在加载模型: {model_path} ...")
    model = SAC.load(model_path)

    print(f"🎯 任务：从 {start_pos} 潜航至 {goal_pos}")

    global_path = generate_smooth_curve(start_pos, goal_pos, num_points=50)
    obs = env.reset_for_test(start_pos, np.array(global_path[1]))
    wp_idx = 1
    target_hit_radius = 0.8

    # 绘制绿色引导曲线
    for i in range(len(global_path) - 1):
        env.holo_env.draw_line(
            global_path[i].tolist(), global_path[i + 1].tolist(),
            color=[0, 255, 0], thickness=2.0, lifetime=0
        )

    # 初始化视频录制
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (1280, 720))
    print(f"🎥 录制视频: {video_path}")

    print("🚀 平滑弧线铺设完毕，开始高精度跟踪 + 避障测试...")
    prev_p = np.copy(start_pos)
    final_reached = False
    termination_cause = "unknown"
    total_steps = 0
    cte_history = []
    min_sonar_history = []

    for step in range(3500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        curr_p = np.array(env.current_obs_dict["LocationSensor"])
        dist_to_wp = np.linalg.norm(curr_p - global_path[wp_idx])

        # 处理 episode 终止
        if terminated:
            if dist_to_wp < 1.2:
                if wp_idx < len(global_path) - 1:
                    wp_idx += 1
                    env.switch_waypoint(global_path[wp_idx], curr_p)
                else:
                    final_reached = True
                    termination_cause = "arrived"
                    total_steps = step
                    break
            else:
                termination_cause = "crash_or_oob"
                total_steps = step
                break

        # 航点切换
        if dist_to_wp < target_hit_radius and wp_idx < len(global_path) - 1:
            wp_idx += 1
            env.switch_waypoint(global_path[wp_idx], curr_p)

        # --- 画面渲染 ---
        frame = cv2.cvtColor(
            np.uint8(env.current_obs_dict["FollowCamera"]), cv2.COLOR_RGBA2BGR
        )

        env.holo_env.draw_line(
            prev_p.tolist(), curr_p.tolist(),
            color=[255, 0, 0], thickness=4.0, lifetime=0
        )
        prev_p = np.copy(curr_p)

        # 状态文字
        cv2.putText(frame, "MISSION: SMOOTH CURVE + AVOID", (40, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        cv2.putText(frame, f"WAYPOINT: {wp_idx}/{len(global_path) - 1}", (40, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # CTE
        prev_wp = np.array(global_path[max(0, wp_idx - 1)])
        curr_wp = np.array(global_path[wp_idx])
        vec_path = curr_wp - prev_wp
        vec_rov = curr_p - prev_wp
        path_length = np.linalg.norm(vec_path)
        if path_length > 1e-4:
            cross_track_error = np.linalg.norm(np.cross(vec_path, vec_rov)) / path_length
        else:
            cross_track_error = np.linalg.norm(curr_p - curr_wp)
        cte_history.append(cross_track_error)

        cte_color = (0, 0, 255) if cross_track_error > 0.5 else (255, 255, 255)
        cv2.putText(frame, f"CTE (ERR): {cross_track_error:.3f} m", (40, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, cte_color, 2)
        cv2.putText(frame, f"DIST TO WP: {dist_to_wp:.2f} m", (40, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # 航向误差
        dx = env.target_pos[0] - curr_p[0]
        dy = env.target_pos[1] - curr_p[1]
        target_yaw = np.arctan2(dy, dx)
        _, _, yaw = np.deg2rad(env.current_obs_dict["RotationSensor"])
        yaw_error = (target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
        yaw_err_deg = np.rad2deg(yaw_error)
        cv2.putText(frame, f"YAW ERR: {yaw_err_deg:+.1f} deg", (950, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 声呐最近距离
        min_sonar = np.min(env.clean_sonar_ranges)
        min_sonar_history.append(min_sonar)
        sonar_color = (0, 0, 255) if min_sonar < 3.0 else (255, 255, 255)
        cv2.putText(frame, f"SONAR MIN: {min_sonar:.2f} m", (950, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, sonar_color, 2)

        # 奖励值
        cv2.putText(frame, f"REWARD: {reward:+.2f}", (950, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # 越界检测
        if curr_p[2] > -0.5 or curr_p[2] < -39.0:
            cv2.putText(frame, "OUT OF BOUNDS!", (100, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 0, 255), 8)
            video_writer.write(frame)
            termination_cause = "out_of_bounds"
            total_steps = step
            break

        video_writer.write(frame)

        if not headless:
            cv2.imshow("ROV Curve Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                termination_cause = "user_quit"
                break

        # 终极胜利判定
        if wp_idx == len(global_path) - 1 and np.linalg.norm(curr_p - goal_pos) < 0.8:
            final_reached = True
            termination_cause = "arrived"
            total_steps = step
            break

    # 善后
    video_writer.release()
    if not headless:
        cv2.destroyAllWindows()

    # 汇总指标
    metrics = {
        "success": final_reached,
        "termination_cause": termination_cause,
        "total_steps": total_steps,
        "waypoints_completed": wp_idx,
        "total_waypoints": len(global_path) - 1,
        "avg_cte": float(np.mean(cte_history)) if cte_history else -1,
        "max_cte": float(np.max(cte_history)) if cte_history else -1,
        "avg_min_sonar": float(np.mean(min_sonar_history)) if min_sonar_history else 30.0,
        "min_sonar_overall": float(np.min(min_sonar_history)) if min_sonar_history else 30.0,
        "video_path": video_path,
        "model_path": model_path,
    }

    # 打印报告
    print("\n" + "=" * 55)
    print("              测 试 报 告")
    print("=" * 55)
    print(f"  模型:         {model_path}")
    print(f"  结果:         {'✅ 成功到达' if final_reached else '❌ 未完成 (' + termination_cause + ')'}")
    print(f"  耗时步数:     {total_steps}")
    print(f"  完成航点:     {wp_idx}/{len(global_path) - 1}")
    print(f"  平均 CTE:     {metrics['avg_cte']:.3f} m")
    print(f"  最大 CTE:     {metrics['max_cte']:.3f} m")
    print(f"  平均声呐:     {metrics['avg_min_sonar']:.2f} m")
    print(f"  最近障碍:     {metrics['min_sonar_overall']:.2f} m")
    print(f"  视频:         {video_path}")
    print("=" * 55)

    return metrics


# ==========================================
# 5. CLI 入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROV 航点跟踪 + 避障测试")
    parser.add_argument("--model", type=str, default="sac_rov_edge8_phase6_fish01_2925000_steps.zip",
                        help="模型文件路径")
    parser.add_argument("--video", type=str, default=None,
                        help="视频输出路径 (默认自动生成时间戳文件名)")
    parser.add_argument("--headless", action="store_true",
                        help="无头模式：不显示窗口，仅保存视频文件")
    args = parser.parse_args()

    run_test(model_path=args.model, video_path=args.video, headless=args.headless)
