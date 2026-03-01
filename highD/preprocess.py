import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ==============================================================================
# 1. 설정 및 피처 맵 정의
# ==============================================================================
TARGET_HZ = 5.0  
T_H = 15         # 3.0s 관측
T_F = 25         # 5.0s 예측
MAX_NEIGHBORS = 8 # highD 8-slot 기반

# Ego Feature Indices (x, y, vx, vy, ax, ay)
EGO_FT = {
    'e1': [0, 1],                   # x, y
    'e12': [0, 1, 2, 3],            # x, y, vx, vy
    'e13': [0, 1, 4, 5],            # x, y, ax, ay
    'e123': [0, 1, 2, 3, 4, 5]      # x, y, vx, vy, ax, ay
}

# Neighbor Feature Indices (dx, dy, dvx, dvy, dax, day, lc_state, dx_time, gate)
NBR_FT = {
    'n1': [0, 1],                           # dx, dy
    'n2': [2, 3],                           # dvx, dvy
    'n3': [6, 7],                           # lc_state, dx_time
    'n4': [0, 1, 8],                        # dx, dy, gate
    'n5': [0, 1, 6, 7],                     # dx, dy, lc_state, dx_time
    'n6': [0, 1, 6, 8],                     # dx, dy, lc_state, gate
    'n7': [4, 5, 6, 7, 8],                  # dax, day, lc_state, dx_time, gate
    'n8': [0, 1, 2, 3, 4, 5, 8],            # dx, dy, dvx, dvy, dax, day, gate
    'n_all': [0, 1, 2, 3, 4, 5, 6, 7, 8]    # all features
}

# 실험 모드 정의: 'mode_name': [ego_key, nbr_key]
EXPERIMENT_MODE_MAP = {
    'baseline': ['e1', 'n1'],
    'baseline_v': ['e1', 'n2'],
    'exp1': ['e1', 'n5'],
    'exp2': ['e1', 'n7'],
    'exp3': ['e1', 'n8'],
    'exp4': ['e1', 'n_all'],
    'exp5': ['e1', 'n4'],
    'exp6': ['e1', 'n6'],
}

NEIGHBOR_COLS = [
    "precedingId", "followingId", "leftPrecedingId", "leftAlongsideId",
    "leftFollowingId", "rightPrecedingId", "rightAlongsideId", "rightFollowingId"
]

# ==============================================================================
# 2. 전처리 핵심 유틸리티 (Flip & Feature Logic)
# ==============================================================================

def get_neighbor_features(ego_hist, nb_window, args):
    rel_pos = nb_window[:, 1:3] - ego_hist[:, 1:3] 
    rel_vel = nb_window[:, 3:5] - ego_hist[:, 3:5] 
    rel_acc = nb_window[:, 5:7] - ego_hist[:, 5:7]

    dx, dy = rel_pos[:, 0], rel_pos[:, 1]
    dvx, dvy = rel_vel[:, 0], rel_vel[:, 1]
    
    # lc_state (vY > 0: Right moving)
    lc_state = np.zeros_like(dy)
    nb_vy = nb_window[:, 4]
    lc_state[(dy < -1.0) & (nb_vy > args.vy_eps)] = -1.0 # Left -> Ego
    lc_state[(dy > 1.0) & (nb_vy < -args.vy_eps)] = 1.0  # Right -> Ego

    # dx_time & gate
    denom = dvx.copy()
    denom[dvx >= 0] += args.eps_gate
    denom[dvx < 0] -= args.eps_gate
    dx_time = dx / denom
    gate = np.zeros_like(dx_time)
    gate[(-args.t_back < dx_time) & (dx_time < args.t_front)] = 1.0

    return np.stack([dx, dy, dvx, dvy, rel_acc[:,0], rel_acc[:,1], lc_state, dx_time, gate], axis=-1)

def process_recording(rec_id, raw_dir, args):
    try:
        df = pd.read_csv(raw_dir / f"{rec_id}_tracks.csv")
        tmeta = pd.read_csv(raw_dir / f"{rec_id}_tracksMeta.csv")
        rmeta = pd.read_csv(raw_dir / f"{rec_id}_recordingMeta.csv")
    except FileNotFoundError: return []

    # 좌표 보정 및 Flip (drivingDirection 1: Upper 통합)
    df["x"] = df["x"] + df["width"] / 2.0
    df["y"] = df["y"] + df["height"] / 2.0
    df = df.merge(tmeta[["id", "drivingDirection"]], on="id")
    
    if args.normalize_flip:
        mask = df["drivingDirection"] == 1
        up_m = [float(f) for f in str(rmeta.loc[0, "upperLaneMarkings"]).split(";") if f]
        lo_m = [float(f) for f in str(rmeta.loc[0, "lowerLaneMarkings"]).split(";") if f]
        C_y = up_m[-1] + lo_m[0]
        x_max = df["x"].max()
        df.loc[mask, "x"] = x_max - df.loc[mask, "x"]
        df.loc[mask, "y"] = C_y - df.loc[mask, "y"]
        for col in ["xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]:
            df.loc[mask, col] *= -1

    # Downsampling
    raw_fps = rmeta.loc[0, "frameRate"]
    stride = int(round(raw_fps / TARGET_HZ))
    df = df[df["frame"] % stride == 0].sort_values(["id", "frame"])

    # 피처 인덱스 설정
    ego_key, nbr_key = EXPERIMENT_MODE_MAP[args.experiment_mode]
    ego_idxs = EGO_FT[ego_key]
    nbr_idxs = NBR_FT[nbr_key]
    
    samples = []
    agents = {vid: g[["frame", "x", "y", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]].values 
              for vid, g in df.groupby("id")}
    
    for vid, data in agents.items():
        if len(data) < T_H + T_F: continue
        for i in range(0, len(data) - (T_H + T_F) + 1, int(TARGET_HZ)):
            window = data[i : i + T_H + T_F]
            ego_hist = window[:T_H]
            obs_fr = ego_hist[-1, 0]
            
            # Neighbor ID 조회
            nbr_ids = df[(df["id"] == vid) & (df["frame"] == obs_fr)][NEIGHBOR_COLS].values.flatten()
            
            # Tensors: LED Input Format [cite: 92, 108]
            # ego_in: (T_H, D_ego), nbr_in: (8, T_H, D_nbr)
            ego_tensor = ego_hist[:, 1:][:, ego_idxs]
            nbr_tensor = np.zeros((MAX_NEIGHBORS, T_H, len(nbr_idxs)), dtype=np.float32)
            
            for nb_idx, nb_id in enumerate(nbr_ids):
                if nb_id <= 0 or nb_id not in agents: continue
                nb_data = agents[nb_id]
                nb_win = nb_data[np.isin(nb_data[:, 0], ego_hist[:, 0])]
                if len(nb_win) < T_H: continue
                
                all_feats = get_neighbor_features(ego_hist, nb_win, args)
                nbr_tensor[nb_idx, :, :] = all_feats[:, nbr_idxs]
            
            samples.append({
                "ego_past": ego_tensor.astype(np.float32),
                "nbr_past": nbr_tensor.astype(np.float32),
                "target": window[T_H:, 1:3].astype(np.float32) # (T_F, 2) x,y
            })
    return samples

# ==============================================================================
# 3. 메인 실행부 (Split & Save)
# ==============================================================================

def process_wrapper(args_tuple):
    rid, raw_path, args = args_tuple
    return rid, process_recording(rid, raw_path, args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="highD/raw")
    parser.add_argument("--out_dir", type=str, default="highD")
    parser.add_argument("--experiment_mode", type=str, default="baseline", choices=EXPERIMENT_MODE_MAP.keys())
    parser.add_argument("--normalize_flip", action="store_true", default=True)
    parser.add_argument("--t_front", type=float, default=3.0)
    parser.add_argument("--t_back", type=float, default=5.0)
    parser.add_argument("--vy_eps", type=float, default=0.27)
    parser.add_argument("--eps_gate", type=float, default=0.1)
    args = parser.parse_args()

    raw_path = Path(args.raw_dir)
    rec_ids = sorted(set([f.name.split("_")[0] for f in raw_path.glob("*_tracks.csv")]))
    
    print(f"Mode: {args.experiment_mode} | Processing {len(rec_ids)} recordings...")

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_wrapper, [(rid, raw_path, args) for rid in rec_ids]), total=len(rec_ids)))

    all_samples = {rid: s for rid, s in results if s}
    
    # 7:1:2 Split Logic
    rec_list = sorted(all_samples.keys())
    np.random.seed(42)
    np.random.shuffle(rec_list)
    n = len(rec_list)
    train_ids = rec_list[:int(n*0.7)]
    val_ids = rec_list[int(n*0.7):int(n*0.8)]
    test_ids = rec_list[int(n*0.8):]

    # Save by Mode Folder
    mode_dir = Path(args.out_dir) / args.experiment_mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        split_data = [s for rid in ids for s in all_samples[rid]]
        if not split_data: continue
        
        # LED 모델에서 부르기 편하게 .npz로 묶어 저장 (또는 개별 .npy)
        np.savez_compressed(
            mode_dir / f"{split_name}.npz",
            ego_past=np.array([s["ego_past"] for s in split_data]),
            nbr_past=np.array([s["nbr_past"] for s in split_data]),
            target=np.array([s["target"] for s in split_data])
        )
        print(f"Saved {split_name}.npz with {len(split_data)} samples")

if __name__ == "__main__":
    main()