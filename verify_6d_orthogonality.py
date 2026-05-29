"""验证生成结果的6D旋转正交性 —— 纯推理后验检查，不依赖训练。

用法:
    python verify_6d_orthogonality.py --input_dir outputs_g1/.../generations/t2m/val_samples_5/
    python verify_6d_orthogonality.py --input_dir test_outputs/ --single_file 000000.npy
"""
import argparse
import os
import numpy as np
import torch


def rotation_6d_to_matrix(d6):
    """与 feature_to_joints_v4.py 完全一致的 Gram-Schmidt 实现"""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + 1e-8)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def compute_6d_deviation(rot_data_6d):
    """测量6D向量偏离合法旋转矩阵前两列的程度。"""
    a1 = rot_data_6d[..., :3]
    a2 = rot_data_6d[..., 3:]

    # 1. 正交性偏差
    dot_product = (a1 * a2).sum(dim=-1).abs()
    ortho_error = dot_product.mean().item()

    # 2. 单位长度偏差
    norm_a1 = a1.norm(dim=-1)
    norm_a2 = a2.norm(dim=-1)
    norm_error = ((norm_a1 - 1.0).abs().mean() + (norm_a2 - 1.0).abs().mean()).item() / 2

    # 3. Gram-Schmidt 前后的 Frobenius 差异
    R_gs = rotation_6d_to_matrix(rot_data_6d)
    R_raw = torch.zeros_like(R_gs)
    R_raw[..., 0] = a1 / (a1.norm(dim=-1, keepdim=True) + 1e-8)
    R_raw[..., 1] = a2 / (a2.norm(dim=-1, keepdim=True) + 1e-8)
    R_raw[..., 2] = torch.cross(R_raw[..., 0], R_raw[..., 1], dim=-1)
    frob_diff = (R_gs - R_raw).norm(dim=(-2, -1)).mean().item()

    return ortho_error, norm_error, frob_diff


def verify_file(npy_path):
    """验证单个 .npy 文件中的 rot_data 部分"""
    data = torch.from_numpy(np.load(npy_path)).float()
    T, D = data.shape

    if (D - 3) % 12 != 0:
        print(f"  WARNING: dim {D} not matching V4 format, skipping")
        return None

    J = (D - 3) // 12

    idx4 = 4 + (J - 1) * 3
    idx5 = idx4 + J * 6
    rot_data = data[:, idx4:idx5].reshape(T, J, 6)

    ortho, norm, frob = compute_6d_deviation(rot_data)
    return ortho, norm, frob


def main():
    parser = argparse.ArgumentParser(description="验证6D旋转正交性")
    parser.add_argument("--input_dir", required=True, help="生成结果的目录")
    parser.add_argument("--single_file", default=None, help="只验证单个文件")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Frobenius差异告警阈值 (默认0.05)")
    args = parser.parse_args()

    if args.single_file:
        files = [os.path.join(args.input_dir, args.single_file)]
    else:
        files = sorted([
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir) if f.endswith('.npy')
        ])

    if not files:
        print("ERROR: no .npy files found")
        return

    print(f"Verifying {len(files)} files...\n")

    all_ortho, all_norm, all_frob = [], [], []

    for f in files:
        result = verify_file(f)
        if result is None:
            continue
        ortho, norm, frob = result
        all_ortho.append(ortho)
        all_norm.append(norm)
        all_frob.append(frob)

    if not all_ortho:
        print("ERROR: no valid files")
        return

    ortho_mean = np.mean(all_ortho)
    norm_mean = np.mean(all_norm)
    frob_mean = np.mean(all_frob)
    frob_max = np.max(all_frob)

    print("=" * 60)
    print("6D Rotation Quality Report")
    print("=" * 60)
    print(f"  Files:            {len(all_ortho)}")
    print(f"  Ortho deviation:  {ortho_mean:.6f}  (ideal: 0)")
    print(f"  Norm deviation:   {norm_mean:.6f}  (ideal: 0)")
    print(f"  GS Frobenius:     {frob_mean:.6f}  (mean)")
    print(f"  GS Frobenius:     {frob_max:.6f}  (max)")
    print()

    if frob_max < args.threshold:
        print(f"PASS: Frobenius < {args.threshold}, 6D rotation quality is good")
    elif frob_max < 0.1:
        print(f"OK: Frobenius in [{args.threshold}, 0.1), acceptable")
    else:
        print(f"WARN: Frobenius >= 0.1, consider switching to quaternion format")


if __name__ == "__main__":
    main()
