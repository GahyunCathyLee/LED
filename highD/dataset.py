import numpy as np
import torch
from torch.utils.data import Dataset

class HighDDataset(Dataset):
    def __init__(self, data_path):
        """
        HighD 데이터셋 로더: 전처리된 6차원 피처를 RAM에 직접 로드합니다.
        """
        print(f"[{data_path}] RAM에 데이터를 적재하는 중...")
        data = np.load(data_path)
        
        # (B, 9, T_H, 6) - Ego와 주변차 8대의 [rel_x, rel_y, vx, vy, ax, ay]
        self.past_traj = torch.from_numpy(data['past_traj']).float()
        # (B, T_F, 2) - 상대 좌표로 변환된 미래 경로
        self.fut_traj = torch.from_numpy(data['fut_traj']).float()
        # (B, 2) - 복원을 위한 마지막 관측 시점의 절대 좌표
        self.initial_pos = torch.from_numpy(data['initial_pos']).float()
        
        self.length = self.fut_traj.shape[0]
        print(f"[{data_path}] 적재 완료! (총 {self.length} 샘플)")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            'past_traj': self.past_traj[idx],
            'fut_traj': self.fut_traj[idx],
            'initial_pos': self.initial_pos[idx]
        }