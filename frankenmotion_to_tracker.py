"""FrankenMotion -> G1 Tracker 适配器

把 FrankenMotion 生成的 363维特征 .npy 转成 TextOp tracker 可用的 .npz。

用法:
    python frankenmotion_to_tracker.py \
        --input generations/t2m/val_samples_3/000000.npy \
        --output my_motion.npz

    # 批量转换
    python frankenmotion_to_tracker.py \
        --input_dir generations/t2m/val_samples_3/ \
        --output_dir tracker_outputs/
"""
import argparse, os, sys, numpy as np
import torch
import mujoco
from scipy.spatial.transform import Rotation as R
from scipy import interpolate


# ===================== 从 MotionGPT_2 移植的数学工具 =====================

def qinv_np(q):
    a = q.copy()
    a[..., 1:] *= -1
    return a

def qmul_np(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], axis=-1)

def wxyz_to_xyzw(q): return q[..., [1, 2, 3, 0]]
def xyzw_to_wxyz(q): return q[..., [3, 0, 1, 2]]

def inv_rename_transform(pos, rot_wxyz):
    pos_zup = pos[..., [2, 0, 1]]
    rot_xyzw = wxyz_to_xyzw(rot_wxyz)
    rot_xyzw_zup = rot_xyzw[..., [2, 0, 1, 3]]
    rot_wxyz_zup = xyzw_to_wxyz(rot_xyzw_zup)
    return pos_zup, rot_wxyz_zup


# ===================== G1 关节名映射 =====================

BODY_NAME_MAP_INV = {
    1: "left_hip_pitch_link", 2: "left_hip_roll_link", 3: "left_hip_yaw_link",
    4: "left_knee_link", 5: "left_ankle_pitch_link", 6: "left_ankle_roll_link",
    7: "right_hip_pitch_link", 8: "right_hip_roll_link", 9: "right_hip_yaw_link",
    10: "right_knee_link", 11: "right_ankle_pitch_link", 12: "right_ankle_roll_link",
    13: "waist_yaw_link", 14: "waist_roll_link", 15: "waist_pitch_link",
    16: "left_shoulder_pitch_link", 17: "left_shoulder_roll_link", 18: "left_shoulder_yaw_link",
    19: "left_elbow_link", 20: "left_wrist_roll_link", 21: "left_wrist_pitch_link", 22: "left_wrist_yaw_link",
    23: "right_shoulder_pitch_link", 24: "right_shoulder_roll_link", 25: "right_shoulder_yaw_link",
    26: "right_elbow_link", 27: "right_wrist_roll_link", 28: "right_wrist_pitch_link", 29: "right_wrist_yaw_link",
}


# ===================== 特征恢复 (从 MotionGPT_2 feature_to_joints_v4.py) =====================

def qinv(q):
    a = q.clone()
    a[..., 1:] *= -1
    return a

def qmul(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)

def qrot(q, v):
    q_v = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    res = qmul(qmul(q, q_v), qinv(q))
    return res[..., 1:]

def rotation_6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)

def matrix_to_quaternion(matrix):
    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    w = torch.sqrt(torch.clamp(1.0 + trace, min=1e-8)) * 0.5
    scale = 0.25 / w
    scale = torch.clamp(scale, max=1e6)
    x = (matrix[..., 2, 1] - matrix[..., 1, 2]) * scale
    y = (matrix[..., 0, 2] - matrix[..., 2, 0]) * scale
    z = (matrix[..., 1, 0] - matrix[..., 0, 1]) * scale
    return torch.stack([w, x, y, z], dim=-1)

def yaw_to_quaternion(yaw):
    half_yaw = yaw / 2.0
    w = torch.cos(half_yaw)
    y = torch.sin(half_yaw)
    x = torch.zeros_like(w)
    z = torch.zeros_like(w)
    return torch.stack([w, x, y, z], dim=-1)

def recover_g1_motion_v4(features):
    B, T, D = features.shape
    assert (D - 3) % 12 == 0, f"Invalid feature dim {D}"
    J = (D - 3) // 12

    idx1, idx2, idx3 = 1, 3, 4
    idx4 = 4 + (J - 1) * 3
    idx5 = idx4 + J * 6

    yaw_vel = features[..., 0:idx1]
    root_vel_xz = features[..., idx1:idx2]
    root_y = features[..., idx2:idx3]
    ric_data = features[..., idx3:idx4].view(B, T, J - 1, 3)
    rot_data = features[..., idx4:idx5].view(B, T, J, 6)

    root_rots_yaw_only = torch.zeros((B, T, 4), device=features.device)
    root_pos = torch.zeros((B, T, 3), device=features.device)

    q_curr = torch.tensor([1.0, 0.0, 0.0, 0.0], device=features.device).repeat(B, 1)
    pos_curr = torch.zeros((B, 3), device=features.device)

    for t in range(T):
        if t > 0:
            q_delta = yaw_to_quaternion(yaw_vel[:, t - 1, 0])
            q_curr = qmul(q_curr, q_delta)
            q_curr = q_curr / q_curr.norm(dim=-1, keepdim=True)

            v_loc_xz = root_vel_xz[:, t - 1]
            v_loc_3d = torch.stack([v_loc_xz[:, 0], torch.zeros_like(v_loc_xz[:, 0]), v_loc_xz[:, 1]], dim=-1)
            q_prev = root_rots_yaw_only[:, t - 1]
            v_glob = qrot(q_prev, v_loc_3d)
            pos_curr[:, 0] += v_glob[:, 0]
            pos_curr[:, 2] += v_glob[:, 2]

        pos_curr[:, 1] = root_y[:, t, 0]
        root_rots_yaw_only[:, t] = q_curr.clone()
        root_pos[:, t] = pos_curr.clone()

    joint_rot_relative_mat = rotation_6d_to_matrix(rot_data)
    joint_rot_relative = matrix_to_quaternion(joint_rot_relative_mat)
    q_root_expanded = root_rots_yaw_only.unsqueeze(2).repeat(1, 1, J, 1)
    global_rotations = qmul(q_root_expanded, joint_rot_relative)

    global_positions = torch.zeros((B, T, J, 3), device=features.device)
    global_positions[:, :, 0, :] = root_pos
    q_root_expanded_pos = root_rots_yaw_only.unsqueeze(2).repeat(1, 1, J - 1, 1)
    local_pos = qrot(q_root_expanded_pos, ric_data)
    global_positions[:, :, 1:, :] = root_pos.unsqueeze(2) + local_pos

    return global_positions, global_rotations


# ===================== Tracker 计算 =====================

def compute_tracker_data(root_pos, root_rot, joints_global_q, xml_path):
    """逆FK + MuJoCo FK -> tracker qpos/qvel/ref_global"""
    print(f"  Loading MuJoCo XML: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)

    T = root_pos.shape[0]
    joint_angles = np.zeros((T, 29))

    print("  Computing inverse FK...")
    for i in range(29):
        body_idx = i + 1
        if body_idx not in BODY_NAME_MAP_INV:
            continue
        body_name = BODY_NAME_MAP_INV[body_idx]
        mj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if mj_body_id == -1:
            continue

        jnt_addr = model.body_jntadr[mj_body_id]
        if jnt_addr == -1:
            continue
        jnt_axis = model.jnt_axis[jnt_addr]

        parent_id = model.body_parentid[mj_body_id]
        q_child = joints_global_q[:, i, :]

        if parent_id == 0:
            q_parent = root_rot
        else:
            parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id)
            p_idx = next((k - 1 for k, v in BODY_NAME_MAP_INV.items() if v == parent_name), -1)
            q_parent = joints_global_q[:, p_idx, :] if p_idx != -1 else root_rot

        q_rel = qmul_np(qinv_np(q_parent), q_child)
        r = R.from_quat(wxyz_to_xyzw(q_rel))
        angle = np.sum(r.as_rotvec() * jnt_axis, axis=1)
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        joint_angles[:, i] = angle

    qpos = np.concatenate([root_pos, root_rot, joint_angles], axis=-1)

    print("  Upsampling 20Hz -> 60Hz...")
    ratio = 3
    T_new = (T - 1) * ratio + 1
    t_old = np.linspace(0, 1, T)
    t_new = np.linspace(0, 1, T_new)

    qpos_60 = np.zeros((T_new, 36))
    for i in list(range(3)) + list(range(7, 36)):
        qpos_60[:, i] = interpolate.interp1d(t_old, qpos[:, i], kind="linear")(t_new)

    slerp = R.from_quat(wxyz_to_xyzw(qpos[:, 3:7]))
    key_times = t_old.copy()
    key_rots = slerp
    rot_new = scipy_slerp(key_times, key_rots, t_new)
    qpos_60[:, 3:7] = xyzw_to_wxyz(rot_new)

    data = mujoco.MjData(model)
    ref_global_pos = np.zeros((T_new, model.nbody - 1, 3))
    ref_global_rot = np.zeros((T_new, model.nbody - 1, 4))

    for t in range(T_new):
        data.qpos[:] = qpos_60[t]
        mujoco.mj_kinematics(model, data)
        ref_global_pos[t] = data.xpos[1:].copy()
        ref_global_rot[t] = data.xquat[1:].copy()

    dt = 1.0 / 60.0
    qvel = np.zeros((T_new, 35))
    qvel[:, 0:3] = np.gradient(qpos_60[:, 0:3], dt, axis=0)
    qvel[:, 6:] = np.gradient(qpos_60[:, 7:], dt, axis=0)
    global_vel = np.gradient(ref_global_pos, dt, axis=0)

    return {
        "qpos": qpos_60,
        "qvel": qvel,
        "ref_dof_pos": qpos_60[:, 7:],
        "ref_global_translation": ref_global_pos,
        "ref_global_rotation_quat": ref_global_rot,
        "ref_global_velocity": global_vel,
        "fps": 60.0,
    }


def scipy_slerp(key_times, key_rots, query_times):
    """SciPy Slerp wrapper."""
    from scipy.spatial.transform import Slerp
    s = Slerp(key_times, key_rots)
    return s(query_times).as_quat()


# ===================== 主入口 =====================

def convert_single(npy_path, output_path, xml_path):
    print(f"FrankenMotion -> Tracker: {npy_path}")
    
    feat = torch.from_numpy(np.load(npy_path)).unsqueeze(0).float()

    print("  Recovering joint positions...")
    with torch.no_grad():
        global_pos_t, global_rot_t = recover_g1_motion_v4(feat)

    global_pos = global_pos_t[0].cpu().numpy()
    global_rot = global_rot_t[0].cpu().numpy()

    rp_y, rr_y = global_pos[:, 0, :], global_rot[:, 0, :]
    jg_y = global_rot[:, 1:, :]

    print("  Y-up -> Z-up...")
    rp_z, rr_z = inv_rename_transform(rp_y, rr_y)
    _, jg_z = inv_rename_transform(rp_y, jg_y)

    print("  Computing tracker data...")
    tracker_data = compute_tracker_data(rp_z, rr_z, jg_z, xml_path)

    np.savez(output_path, **tracker_data)
    print(f"  Saved: {output_path}")
    print(f"  Duration: {tracker_data['qpos'].shape[0] / 60:.1f}s @ 60Hz")
    return tracker_data


def main():
    parser = argparse.ArgumentParser(description="FrankenMotion -> G1 Tracker")
    parser.add_argument("--input", default=None, help="单个 .npy 文件")
    parser.add_argument("--output", default="tracker_output.npz", help="输出 .npz")
    parser.add_argument("--input_dir", default=None, help="批量转换目录")
    parser.add_argument("--output_dir", default="tracker_outputs", help="批量输出目录")
    parser.add_argument("--xml",
        default="F:/MotionGPT_2/tracker/source/textop_tracker/textop_tracker/assets/unitree_description/mjcf/g1_act_fixed.xml",
        help="G1 MuJoCo XML 路径")
    args = parser.parse_args()

    if args.input_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".npy"))
        for f in files:
            name = os.path.splitext(f)[0]
            convert_single(
                os.path.join(args.input_dir, f),
                os.path.join(args.output_dir, f"{name}.npz"),
                args.xml,
            )
    else:
        if not args.input:
            print("ERROR: need --input or --input_dir")
            return
        convert_single(args.input, args.output, args.xml)


if __name__ == "__main__":
    main()
