import numpy as np
import torch
from torch.utils.data import Dataset

class HighDDataset(Dataset):
    def __init__(self, data_path):
        """
        HighD 데이터셋 로더 (In-Memory 캐싱)
        data_path: .npz 파일 경로 (e.g., 'highD/exp1/train.npz')
        """
        print(f"[{data_path}] RAM에 데이터를 적재하는 중...")
        data = np.load(data_path)
        
        # 데이터를 torch.Tensor로 변환하여 RAM에 상주
        self.ego_past = torch.from_numpy(data['ego_past']).float()   # (B, T_H, D_ego)
        self.nbr_past = torch.from_numpy(data['nbr_past']).float()   # (B, 8, T_H, D_nbr)
        self.target = torch.from_numpy(data['target']).float()       # (B, T_F, 2)
        
        self.length = self.target.shape[0]
        print(f"[{data_path}] 적재 완료! (총 {self.length} 샘플)")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 이미 RAM에 로드된 텐서를 인덱싱 (매우 빠름)
        return {
            'ego_past': self.ego_past[idx],
            'nbr_past': self.nbr_past[idx],
            'target': self.target[idx]
        }

def seq_collate(data):
    """
    기존 LED 코드의 collate_fn과 형식을 맞추기 위한 함수
    """
    batch = {
        'ego_past': torch.stack([item['ego_past'] for item in data]),
        'nbr_past': torch.stack([item['nbr_past'] for item in data]),
        'target': torch.stack([item['target'] for item in data]),
    }
    return batch