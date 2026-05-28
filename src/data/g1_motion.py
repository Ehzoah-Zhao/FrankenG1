import os
import numpy as np
import torch


class G1MotionLoader:
    """Loads pre-extracted G1 motion features from new_joint_vecs_v4/ directory."""

    name = "g1_joints"

    def __init__(
        self,
        base_dir: str,
        fps: float = 20.0,
        disable: bool = False,
        nfeats: int = 363,
        umin_s: float = 0.5,
        umax_s: float = 3.0,
        cache: bool = True,
    ):
        self.fps = fps
        self.base_dir = base_dir
        self.cache = cache
        self.motions = {}
        self.disable = disable
        self.nfeats = nfeats

        self.umin = max(1, int(self.fps * umin_s))
        self.umax = int(self.fps * umax_s)

    def __call__(self, path, start, end, drop_motion_perc=None, load_transition=False):
        if self.disable:
            return {"x": path, "length": int(self.fps * (end - start))}

        if self.cache:
            if path not in self.motions:
                motion_path = os.path.join(self.base_dir, path + ".npy")
                motion = np.load(motion_path)
                motion = torch.from_numpy(motion).to(torch.float)
                self.motions[path] = motion
            motion = self.motions[path]
        else:
            motion_path = os.path.join(self.base_dir, path + ".npy")
            motion = np.load(motion_path)
            motion = torch.from_numpy(motion).float()

        if load_transition:
            import random
            duration = random.randint(self.umin, min(self.umax, len(motion)))
            start_idx = random.randint(0, len(motion) - duration)
            motion = motion[start_idx : start_idx + duration]
        else:
            begin = int(start * self.fps)
            end_idx = int(end * self.fps) if end > 0 else len(motion)
            motion = motion[begin:end_idx]

            if drop_motion_perc is not None and drop_motion_perc > 0:
                max_frames_to_drop = int(len(motion) * drop_motion_perc)
                n_frames_to_drop = random.randint(0, max_frames_to_drop)
                n_frames_left = random.randint(0, n_frames_to_drop)
                n_frames_right = n_frames_to_drop - n_frames_left
                if n_frames_right > 0:
                    motion = motion[n_frames_left:-n_frames_right]
                else:
                    motion = motion[n_frames_left:]

        return {"x": motion, "length": len(motion)}
