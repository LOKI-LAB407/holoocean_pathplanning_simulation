import numpy as np

class RRT3D:
    def __init__(self, start, goal, obstacles, bounds, step_size=2.0, max_iter=2000, rov_safety_radius=1.5, waypoint_interval=1.5):
        """
        :param rov_safety_radius: ROV 安全膨胀半径，防擦边
        :param waypoint_interval: 航点致密化间距（米），保证直线段也有足够的航点引导
        """
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = obstacles 
        self.bounds = bounds       
        self.step_size = step_size
        self.max_iter = max_iter
        self.rov_safety_radius = rov_safety_radius 
        self.waypoint_interval = waypoint_interval # 👈 新增参数：航点插值间距
        
        self.nodes = [{'pos': self.start, 'parent': -1}]

    def is_collision_free(self, p1, p2):
        """检查两点之间的连线是否撞到膨胀后的障碍物包围盒"""
        steps = int(np.linalg.norm(p2 - p1) / 0.5) + 2
        for i in range(steps):
            point = p1 + (p2 - p1) * (i / (steps - 1))
            px, py, pz = point
            
            for obs in self.obstacles:
                ox, oy, oz, r = obs
                min_x = ox - r - self.rov_safety_radius
                max_x = ox + r + self.rov_safety_radius
                min_y = oy - r - self.rov_safety_radius
                max_y = oy + r + self.rov_safety_radius
                min_z = oz - r - self.rov_safety_radius
                max_z = oz + r + self.rov_safety_radius
                
                if (min_x <= px <= max_x) and (min_y <= py <= max_y) and (min_z <= pz <= max_z):
                    return False
        return True

    def get_random_node(self):
        if np.random.rand() < 0.1:
            return self.goal
        else:
            x = np.random.uniform(self.bounds[0][0], self.bounds[0][1])
            y = np.random.uniform(self.bounds[1][0], self.bounds[1][1])
            z = np.random.uniform(self.bounds[2][0], self.bounds[2][1])
            return np.array([x, y, z])

    def prune_path(self, path):
        """贪婪剪枝：拉直 RRT 产生的锯齿路径"""
        if path is None or len(path) <= 2:
            return path
            
        print(f"✂️ 正在进行路径剪枝... (原节点数: {len(path)})")
        pruned_path = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            furthest_visible = i + 1
            for j in range(len(path) - 1, i, -1):
                if self.is_collision_free(path[i], path[j]):
                    furthest_visible = j
                    break
                    
            pruned_path.append(path[furthest_visible])
            i = furthest_visible
            
        print(f"✅ 剪枝完成！大转折点数: {len(pruned_path)}")
        return pruned_path

    def densify_path(self, path):
        """🌟 核心新增：在线段之间进行等距插值，撒下面包屑"""
        if path is None or len(path) < 2:
            return path
            
        dense_path = []
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i+1]
            dist = np.linalg.norm(p2 - p1)
            
            # 计算这一段直线需要被切分成几段
            num_segments = max(1, int(np.ceil(dist / self.waypoint_interval)))
            
            # 线性插值生成中间点
            for j in range(num_segments):
                interpolated_point = p1 + (p2 - p1) * (j / num_segments)
                dense_path.append(interpolated_point)
                
        # 加上最终的目标点
        dense_path.append(path[-1])
        
        print(f"📍 航点致密化完成！已生成连续追踪航点共 {len(dense_path)} 个")
        return dense_path

    def plan(self):
        print("🗺️ 正在生成全局避障路径...")
        for _ in range(self.max_iter):
            rand_pos = self.get_random_node()
            
            distances = [np.linalg.norm(node['pos'] - rand_pos) for node in self.nodes]
            nearest_idx = np.argmin(distances)
            nearest_node = self.nodes[nearest_idx]
            
            direction = rand_pos - nearest_node['pos']
            length = np.linalg.norm(direction)
            if length == 0: continue
            direction = direction / length
            
            new_pos = nearest_node['pos'] + direction * min(self.step_size, length)
            
            if self.is_collision_free(nearest_node['pos'], new_pos):
                self.nodes.append({'pos': new_pos, 'parent': nearest_idx})
                
                if np.linalg.norm(new_pos - self.goal) < self.step_size:
                    self.nodes.append({'pos': self.goal, 'parent': len(self.nodes) - 1})
                    
                    raw_path = self.extract_path()           # 1. 提取原始锯齿线
                    pruned_path = self.prune_path(raw_path)  # 2. 剪枝成大折线
                    final_path = self.densify_path(pruned_path) # 3. 致密化成面包屑轨迹
                    
                    return final_path
                    
        print("❌ RRT 规划失败：未能在最大迭代次数内找到路径。")
        return None

    def extract_path(self):
        path = []
        curr_idx = len(self.nodes) - 1
        while curr_idx != -1:
            path.append(self.nodes[curr_idx]['pos'])
            curr_idx = self.nodes[curr_idx]['parent']
        return path[::-1]