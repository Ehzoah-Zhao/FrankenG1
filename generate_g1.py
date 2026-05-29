"""Generate G1 robot motions from part-level text prompts."""
from __future__ import annotations
import numpy as np
import logging, os, json, shutil
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from src.config import read_config
from src.tools.inference import load_diffusion, extract_motion_outputs, resolve_clip_dir, move_batch_to_device

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYOPENGL_PLATFORM"] = "egl"
logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="generate_g1", version_base="1.3")
def generate(c: DictConfig):
    logger.info("G1 Motion Generation Script")
    assert c.input_type in ("val", "user_input")
    assert c.baseline in ["none", "sinc", "sinc_lerp"]

    import torch

    if c.input_type == "val":
        n_samples = c.num_sample if hasattr(c, 'num_sample') else 5
        if c.fast:
            n_samples = min(n_samples, 2)
        exp_folder_name = f"{c.input_type}_samples_{n_samples}"
    else:
        exp_folder_name = os.path.splitext(os.path.basename(c.user_input_file))[0]

    if c.baseline != "none":
        exp_folder_name += "_baseline_" + c.baseline

    cfg = read_config(c.run_dir)
    fps = cfg.data.motion_loader.fps
    if hasattr(c, 'condition'):
        cfg.data.condition = c.condition
    elif not hasattr(cfg.data, 'condition'):
        cfg.data.condition = 't2m'

    if c.input_type == "val":
        split = c.input_type
        dataset = instantiate(cfg.data, split=split)
        n_motions = c.num_sample if hasattr(c, 'num_sample') else min(5, len(dataset))
        if c.fast:
            n_motions = min(n_motions, 2)

        dataloader = instantiate(cfg.dataloader, dataset=dataset,
            collate_fn=dataset.collate_fn, shuffle=True, batch_size=n_motions)
        batch = next(iter(dataloader))
        move_batch_to_device(batch, c.device)

        texts = batch.get('text', [])
        global_texts = batch.get('global_text', [None] * len(texts))
        lengths = batch['length'].tolist() if 'length' in batch else []

        infos = {
            "all_texts": texts,
            "all_lengths": lengths,
            "output_lengths": lengths,
            "split": split,
            "global_texts": global_texts,
        }
        logger.info(f"Loaded 1 batch from {split} dataset with {n_motions} samples")

    else:
        from src.tools.parse_user_input import parse_and_validate_user_input
        from src.data.text_part_utils import load_from_annotation_with_model

        logger.info("Loading user input")
        annotation_dict = parse_and_validate_user_input(
            filepath=c.user_input_file, cfg=cfg, fps=fps)
        text_encoder = instantiate(cfg.data.text_encoder)
        text_encoder.no_model = False
        text_encoder.rand_mask = False

        tx_emb_dict, text_dict = load_from_annotation_with_model(
            text_encoder=text_encoder,
            annotations=annotation_dict["annotations"],
            path=annotation_dict["path"],
            start=annotation_dict["start"],
            end=annotation_dict["end"])

        duration_frames = int((annotation_dict["end"] - annotation_dict["start"]) * fps)
        motion_dim = cfg.data.motion_loader.nfeats
        text_dim = tx_emb_dict['local']["x"].shape[1]

        motion = torch.zeros((duration_frames, motion_dim))
        from src.data.text_motion import align
        text_aligned = align(motion, tx_emb_dict['local']["x"])
        x = torch.cat([motion, text_aligned], dim=1)

        batch = {
            "x": x.unsqueeze(0).to(c.device),
            "motion_dim": motion_dim,
            "text_dim": text_dim,
            "text": [text_dict],
            "tx": {"local": {"x": tx_emb_dict['local']["x"].unsqueeze(0).to(c.device),
                              "mask": tx_emb_dict['local']["mask"].unsqueeze(0).to(c.device),
                              "length": tx_emb_dict['local']["length"]}},
            "length": torch.tensor([duration_frames]),
            "mask": torch.ones(1, duration_frames).bool(),
            "keyid": ["user_input_0"],
        }
        batch["tx"]["mask"] = torch.ones(1, dtype=torch.bool).to(c.device)
        batch["tx"]["length"] = 1
        if "x" in tx_emb_dict:
            batch["tx"]["x"] = tx_emb_dict["x"].unsqueeze(0).to(c.device)
        infos = {
            "all_texts": [text_dict],
            "all_lengths": [duration_frames],
            "output_lengths": [duration_frames],
            "split": "user_input",
            "global_texts": [tx_emb_dict.get("text", "")],
            "featsname": cfg.data.motion_loader.name if hasattr(cfg.data.motion_loader, 'name') else "g1_joints",
        }
        n_motions = 1

    # Load diffusion model
    diffusion = load_diffusion(cfg, c.run_dir, c.ckpt, c.device)
    diffusion.guidance_param = c.guidance

    # Generate
    logger.info("Generating motions...")
    n_motions = batch['x'].shape[0]
    xstart = diffusion.batch_forward(batch, infos).cpu()

    # Save outputs
    exp_dir = os.path.join(c.run_dir, "generations", "t2m", exp_folder_name)
    os.makedirs(exp_dir, exist_ok=True)

    logger.info(f"Saving {n_motions} motions to {exp_dir}")

    for idx in range(n_motions):
        length = infos["output_lengths"][idx]
        xstart_i = xstart[idx].cpu()
        npy_path = os.path.join(exp_dir, f"{idx:06d}.npy")
        output = xstart_i[:length, :cfg.data.motion_loader.nfeats].detach()
        np.save(npy_path, output.numpy())

        # Save text input
        text_input = infos["all_texts"][idx] if idx < len(infos["all_texts"]) else None
        input_text_path = npy_path.replace(".npy", "_input_text.txt")
        with open(input_text_path, "w") as f:
            if isinstance(text_input, dict):
                for part_name, segments in text_input.items():
                    if part_name.endswith("_length"):
                        continue
                    if not isinstance(segments, list):
                        continue
                    f.write(f"{part_name}:\n")
                    for seg in segments:
                        if isinstance(seg, tuple) and len(seg) == 2:
                            seg_text, seg_len = seg
                            f.write(f"  - [{seg_len} frames] {seg_text}\n")
                        elif isinstance(seg, dict) and "text" in seg:
                            f.write(f"  - [{seg.get('start', 0):.1f}s-{seg.get('end', 0):.1f}s] {seg['text']}\n")
                        else:
                            f.write(f"  - {seg}\n")
            elif text_input is not None:
                f.write(str(text_input))

    # Save infos
    with open(os.path.join(exp_dir, "infos.json"), "w") as f:
        json.dump({k: v for k, v in infos.items() if k != "all_texts"}, f, indent=2)

    logger.info(f"Done! Motions saved to {exp_dir}")
    logger.info("Use TextOp tracker to deploy these motions to the real G1 robot.")


if __name__ == "__main__":
    generate()
