"""Evaluate G1 FrankenBot model."""
from __future__ import annotations
import logging, os, sys, json
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from src.config import read_config

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYOPENGL_PLATFORM"] = "egl"
logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="eval_part", version_base="1.3")
def evaluate(c: DictConfig):
    logger.info("G1 Evaluation Script")
    import torch
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    cfg = read_config(c.run_dir)
    fps = cfg.data.motion_loader.fps

    split = c.input_type
    dataset = instantiate(cfg.data, split=split)

    from src.tools.inference import load_diffusion, move_batch_to_device
    diffusion = load_diffusion(cfg, c.run_dir, c.ckpt, c.device)
    diffusion.guidance_param = c.guidance

    import numpy as np
    from tqdm import tqdm

    exp_name = f"{split}_{c.ckpt}_{c.num_runs}runs"
    if c.baseline != "none":
        exp_name += "_baseline_" + c.baseline
    exp_dir = os.path.join(c.run_dir, "generations", "t2m", exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    dataloader = instantiate(cfg.dataloader, dataset=dataset,
        collate_fn=dataset.collate_fn, shuffle=False, batch_size=c.num_runs)

    all_results = []
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating")):
        move_batch_to_device(batch, c.device)
        n_motions = batch['x'].shape[0]
        lengths = batch['length'].tolist()

        for run_idx in range(c.num_runs):
            torch.manual_seed(c.seed + run_idx * 1000 if c.seed >= 0 else run_idx * 1000)
            xstart = diffusion.sample_unconditional(batch, n_motions=n_motions)

            for idx in range(n_motions):
                length = lengths[idx]
                motion = xstart[idx, :length, :cfg.data.motion_loader.nfeats].cpu().numpy()
                key = f"{batch_idx:04d}_{idx:04d}_run{run_idx:02d}"
                np.save(os.path.join(exp_dir, f"{key}.npy"), motion)
                all_results.append({
                    "key": key,
                    "batch_idx": batch_idx,
                    "sample_idx": idx,
                    "run_idx": run_idx,
                    "length": int(length),
                })

    with open(os.path.join(exp_dir, "generation_index.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Generated {len(all_results)} motions to {exp_dir}")
    logger.info("Use src.eval.run to compute metrics (FID, R-Precision, etc.)")


if __name__ == "__main__":
    evaluate()
