import os
import glob
from stable_baselines3.common.callbacks import BaseCallback

class RollingCheckpointCallback(BaseCallback):
    """
    保存时写入带步数的文件，同时自动删除旧文件，仅保留最近 N 个检查点。
    """
    def __init__(self, save_freq, save_path, name_prefix="model",
                 save_replay_buffer=True, keep_last=3, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.keep_last = keep_last
        
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq != 0:
            return True
        
        # 保存新文件
        model_path = os.path.join(
            self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.zip"
        )
        self.model.save(model_path)
        
        if self.save_replay_buffer and hasattr(self.model, "save_replay_buffer"):
            buffer_path = os.path.join(
                self.save_path,
                f"{self.name_prefix}_replay_buffer_{self.num_timesteps}_steps.pkl"
            )
            self.model.save_replay_buffer(buffer_path)
        
        # 清理旧文件
        self._cleanup_old_checkpoints()
        
        if self.verbose > 0:
            print(f"📀 已保存 checkpoint @ {self.num_timesteps} 步（保留最近 {self.keep_last} 份）")
        
        return True

    def _cleanup_old_checkpoints(self):
        """删除超出保留数量的旧模型和 buffer 文件"""
        # 收集所有模型文件
        model_pattern = os.path.join(self.save_path, f"{self.name_prefix}_*_steps.zip")
        model_files = sorted(glob.glob(model_pattern))
        
        # 删除多余的模型文件
        while len(model_files) > self.keep_last:
            old_file = model_files.pop(0)
            os.remove(old_file)
            if self.verbose > 0:
                print(f"🗑️  已删除旧模型: {os.path.basename(old_file)}")
        
        # 收集所有 buffer 文件
        buffer_pattern = os.path.join(
            self.save_path, f"{self.name_prefix}_replay_buffer_*_steps.pkl"
        )
        buffer_files = sorted(glob.glob(buffer_pattern))
        while len(buffer_files) > self.keep_last:
            old_file = buffer_files.pop(0)
            os.remove(old_file)