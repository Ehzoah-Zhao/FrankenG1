"""Quick evaluation for FrankenMotion G1: FID, Diversity, and R-Precision proxy.

Uses raw 363-dim motion features for FID/Diversity.
For R-Precision, uses CCA to find a shared text-motion subspace,
then computes standard retrieval metrics.

Usage:
    python eval_g1_quick.py --gt_dir <gt_npy_dir> --gen_dir <gen_npy_dir> --anno_dir <annotations_dir>
"""

from __future__ import annotations
import argparse, json, os, sys, warnings
from pathlib import Path
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from scipy import linalg
from tqdm import tqdm
warnings.filterwarnings("ignore")

def calculate_fid(feats1, feats2):
    mu1 = np.mean(feats1, axis=0)
    sigma1 = np.cov(feats1, rowvar=False)
    mu2 = np.mean(feats2, axis=0)
    sigma2 = np.cov(feats2, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        eps = 1e-6
        covmean = linalg.sqrtm((sigma1+np.eye(sigma1.shape[0])*eps).dot(sigma2+np.eye(sigma2.shape[0])*eps))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2*np.trace(covmean)
    return float(max(fid, 0))

def calculate_diversity(feats, num_samples=300):
    n = feats.shape[0]
    if n < 2:
        return 0.0
    k = min(num_samples, n)
    idx1 = np.random.choice(n, k, replace=False)
    idx2 = np.random.choice(n, k, replace=False)
    return float(np.linalg.norm(feats[idx1] - feats[idx2], axis=1).mean())

def r_precision_cca(motion_feats, text_feats, n_components=64):
    n = motion_feats.shape[0]
    if n < n_components * 2:
        n_components = max(1, n // 4)
    train_idx = np.arange(0, n, 2)
    test_idx = np.arange(1, n, 2)
    if len(train_idx) < n_components or len(test_idx) < 1:
        train_idx = test_idx = np.arange(n)
    m_train = motion_feats[train_idx]
    t_train = text_feats[train_idx]
    m_train = m_train / (np.linalg.norm(m_train, axis=1, keepdims=True) + 1e-8)
    t_train = t_train / (np.linalg.norm(t_train, axis=1, keepdims=True) + 1e-8)
    cca = CCA(n_components=min(n_components, len(train_idx)-1), max_iter=1000)
    cca.fit(m_train, t_train)
    m_test = motion_feats[test_idx]
    t_test = text_feats[test_idx]
    m_test = m_test / (np.linalg.norm(m_test, axis=1, keepdims=True) + 1e-8)
    t_test = t_test / (np.linalg.norm(t_test, axis=1, keepdims=True) + 1e-8)
    m_proj, t_proj = cca.transform(m_test, t_test)
    sim = m_proj @ t_proj.T
    rankings = np.argsort(-sim, axis=1)
    results = {}
    for k in [1, 3, 5, 10]:
        correct = np.any(rankings[:, :k] == np.arange(len(test_idx))[:, None], axis=1)
        results[f"R@{k}"] = float(correct.mean())
    gt_ranks = np.argmax(rankings == np.arange(len(test_idx))[:, None], axis=1)
    results["MedR"] = float(np.median(gt_ranks) + 1)
    return results

def load_npy_dir(directory):
    data = {}
    if not os.path.isdir(directory):
        print(f"WARNING: dir not found: {directory}")
        return data
    for f in sorted(os.listdir(directory)):
        if f.endswith(".npy"):
            data[Path(f).stem] = np.load(os.path.join(directory, f))
    return data

def load_text_emb(anno_dir, key):
    p = os.path.join(anno_dir, f"{key}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        ann = json.load(f)
    emb = ann.get("sequence_caption_emb") or ann.get("caption_emb")
    return np.array(emb, dtype=np.float32) if emb is not None else None

def aggregate(motion, method="mean"):
    if motion.ndim == 1:
        return motion
    return motion.mean(axis=0) if method == "mean" else motion[-1]

def main():
    ap = argparse.ArgumentParser(description="Quick G1 evaluation")
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--anno_dir", default=None)
    ap.add_argument("--n_cca", type=int, default=64)
    ap.add_argument("--aggregate", default="mean", choices=["mean", "last"])
    args = ap.parse_args()

    print("=" * 60)
    print("FrankenMotion G1 Quick Evaluation")
    print("=" * 60)

    gt_data = load_npy_dir(args.gt_dir)
    gen_data = load_npy_dir(args.gen_dir)
    print(f"GT samples: {len(gt_data)}  |  Generated: {len(gen_data)}")

    common = sorted(set(gt_data) & set(gen_data))
    if not common:
        print("WARNING: no common keys, using independent sets")
        gt_keys = sorted(gt_data)
        gen_keys = sorted(gen_data)
    else:
        gt_keys = gen_keys = common
    n = len(common)
    print(f"Common samples: {n}")

    print("Aggregating features...")
    gt_feats = np.stack([aggregate(gt_data[k], args.aggregate) for k in gt_keys])
    gen_feats = np.stack([aggregate(gen_data[k], args.aggregate) for k in gen_keys])
    print(f"Feature dim: {gt_feats.shape[1]}")

    print()
    print("--- FID ---")
    fid = calculate_fid(gt_feats, gen_feats)
    print(f"FID (raw): {fid:.4f}")

    pca_dim = min(64, gt_feats.shape[1], n - 1)
    if pca_dim >= 2:
        pca = PCA(n_components=pca_dim)
        fid_pca = calculate_fid(pca.fit_transform(gt_feats), pca.transform(gen_feats))
        print(f"FID (PCA-{pca_dim}): {fid_pca:.4f}")

    print()
    print("--- Diversity ---")
    print(f"GT: {calculate_diversity(gt_feats):.4f}")
    print(f"Gen: {calculate_diversity(gen_feats):.4f}")

    print()
    print("--- R-Precision (CCA) ---")
    if args.anno_dir and os.path.isdir(args.anno_dir):
        text_embs = {}
        for k in tqdm(gt_keys, desc="Loading texts"):
            emb = load_text_emb(args.anno_dir, k)
            if emb is not None:
                text_embs[k] = emb
        common_t = sorted(set(gt_keys) & set(text_embs) & set(gen_keys))
        print(f"  Samples with text: {len(common_t)}")
        if len(common_t) >= 10:
            t_feats = np.stack([text_embs[k] for k in common_t])
            m_feats = np.stack([aggregate(gen_data[k], args.aggregate) for k in common_t])
            rp = r_precision_cca(m_feats, t_feats, args.n_cca)
            for metric, value in rp.items():
                print(f"  {metric}: {value:.4f}")
        else:
            print("  Need >= 10 samples with text annotations")
    else:
        print("  Skipped (no --anno_dir)")

    print()
    print("=" * 60)
    print(f"SUMMARY: FID={fid:.4f}  Div_Gen={calculate_diversity(gen_feats):.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
