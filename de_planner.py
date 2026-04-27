import numpy as np
from scipy.optimize import differential_evolution
from scipy.special import comb

class DEPlanner3D:
    def __init__(self, start, goal, obstacles, bounds, num_control_points=3, rov_safety_radius=1.5):
        """
        差分进化全局路径规划器 (DE + Bezier)
        :param start: 起点 (3,)
        :param goal: 终点 (3,)
        :param obstacles: 静态障碍物列表 [(x, y, z, r), ...]
        :param bounds: 地图边界 [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
        :param num_control_points: 贝塞尔曲线的中间控制点数量 (决定了曲线的灵活度)
        :param rov_safety_radius: ROV 的安全防撞半径
        """
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = obstacles
        self.bounds = bounds
        self.num_control_points = num_control_points
        self.rov_safety_radius = rov_safety_radius
        
        # 优化变量的边界: 每个控制点有 x, y, z 三个维度
        self.de_bounds = []
        for _ in range(self.num_control_points):
            self.de_bounds.extend([bounds[0], bounds[1], bounds[2]])
            
    def _bezier_curve(self, control_points, num_points=100):
        """
        根据控制点生成 N 阶贝塞尔曲线
        数学上保证了绝对的平滑，不会出现任何尖锐拐角
        """
        n = len(control_points) - 1
        t = np.linspace(0, 1, num_points)
        curve = np.zeros((num_points, 3))
        
        for i in range(n + 1):
            # Bernstein 多项式
            bernstein = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
            # 扩展维度以进行矩阵乘法
            curve += np.outer(bernstein, control_points[i])
            
        return curve

    def _fitness(self, x):
        """DE 算法的适应度函数 (目标函数)"""
        middle_points = x.reshape((self.num_control_points, 3))
        cps = np.vstack([self.start, middle_points, self.goal])
        
        path = self._bezier_curve(cps, num_points=50) 
        
        collision_penalty = 0.0
        for obs in self.obstacles:
            ox, oy, oz, r = obs
            obs_center = np.array([ox, oy, oz])
            
            # 🌟 核心修复 1：使用 L-infinity 范数 (最大绝对偏差) 
            # 这完美等价于 HoloOcean 中 draw_box 生成的 3D 立方体 AABB 碰撞盒！
            box_distances = np.max(np.abs(path - obs_center), axis=1)
            
            safe_dist = r + self.rov_safety_radius
            violations = safe_dist - box_distances
            violations[violations < 0] = 0 # 没撞到就是 0
            collision_penalty += np.sum(violations) * 10000.0 # 施加巨额碰撞罚分
            
        z_values = path[:, 2]
        z_min, z_max = self.bounds[2][0], self.bounds[2][1]
        out_of_bounds = np.sum(z_values < z_min) + np.sum(z_values > z_max)
        collision_penalty += out_of_bounds * 10000.0

        path_length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        return path_length + collision_penalty

    def plan(self):
        """执行 DE 优化并返回最终密集的航点"""
        print(f"🧬 启动差分进化 (DE) 算法全局优化...")
        
        result = differential_evolution(
            self._fitness, 
            bounds=self.de_bounds, 
            maxiter=1000,     
            popsize=15,       
            tol=0.01,         
            mutation=(0.5, 1.0), 
            recombination=0.7,   
            disp=False        
        )
        
        # 🌟 核心修复 2：加入 result.fun < 5000 的拦截机制
        # 如果代价函数 > 5000，说明 DE 实在找不到不穿模的路（障碍物彻底堵死）
        if result.success and result.fun < 5000.0:
            print(f"✅ DE 优化完成！最优路径代价值: {result.fun:.2f}")
            best_middle_points = result.x.reshape((self.num_control_points, 3))
            best_cps = np.vstack([self.start, best_middle_points, self.goal])
            
            total_dist = np.linalg.norm(self.start - self.goal)
            num_dense_points = max(50, int(total_dist * 2)) 
            
            final_path = self._bezier_curve(best_cps, num_points=num_dense_points)
            return final_path.tolist()
        else:
            print("❌ DE 算法未能找到绝对安全的路径（雷区可能已完全封死去路）！")
            return None