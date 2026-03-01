import os
import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# LED 핵심 모듈 (기존 파일 활용)
from utils.config import Config as LEDConfig
from utils.utils import prepare_seed, print_log, mkdir_if_missing
from models.model_led_initializer import LEDInitializer
from models.model_diffusion import TransformerDenoisingModel
from trainer.train_led_trajectory_augment_input import Trainer
from highD.dataset import HighDDataset

class AutomatedLEDTrainer(Trainer):
    def __init__(self, args):
        # 1. 파일 이름에서 실험 모드 추출 (예: exp1.yaml -> exp1)
        self.experiment_mode = Path(args.config).stem
        
        # 2. Config 로드 (Config 클래스 내부의 .yml 검색 로직을 .yaml로 수정 필요)
        self.cfg = LEDConfig(self.experiment_mode, args.info) 
        
        # 3. 모델 저장 경로 커스텀 설정
        self.checkpoint_dir = Path("ckpts") / self.experiment_mode
        mkdir_if_missing(str(self.checkpoint_dir))
        self.best_path = self.checkpoint_dir / "best.pt"

        # 4. 하드웨어 및 파라미터 자동 로드 (Config 우선)
        self.gpu = self.cfg.get('gpu', 0)
        self.device = torch.device(f'cuda:{self.gpu}' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available(): torch.cuda.set_device(self.gpu)
        
        # 5. 자동 데이터 경로 설정: highD/{experiment_mode}/
        data_root = Path("highD") / self.experiment_mode
        train_dset = HighDDataset(data_root / "train.npz")
        val_dset = HighDDataset(data_root / "val.npz")

        self.train_loader = DataLoader(train_dset, batch_size=self.cfg.train_batch_size, shuffle=True)
        self.test_loader = DataLoader(val_dset, batch_size=self.cfg.test_batch_size, shuffle=False)

        # 6. 모델 및 옵티마이저 (Ego 차원 자동 인식)
        ego_dim = train_dset.ego_past.shape[-1]
        self.model = TransformerDenoisingModel().to(self.device)
        self.model_initializer = LEDInitializer(
            t_h=self.cfg.past_frames, d_h=ego_dim, 
            t_f=self.cfg.future_frames, d_f=2, k_pred=20
        ).to(self.device)

        self.opt = torch.optim.AdamW(self.model_initializer.parameters(), lr=float(self.cfg.lr))
        self.scheduler_model = torch.optim.lr_scheduler.StepLR(self.opt, step_size=self.cfg.decay_step, gamma=self.cfg.decay_gamma)
        
        # 7. 로깅 설정
        self.writer = SummaryWriter(log_dir=str(Path("logs") / self.experiment_mode / datetime.now().strftime("%m%d-%H%M")))
        self.log = open(os.path.join(self.cfg.log_dir, 'log.txt'), 'a+')

        # Diffusion 초기화 (Trainer 상속 로직)
        self.n_steps = self.cfg.diffusion.steps
        self.betas = self.make_beta_schedule(
            schedule=self.cfg.diffusion.beta_schedule, n_timesteps=self.n_steps, 
            start=self.cfg.diffusion.beta_start, end=self.cfg.diffusion.beta_end).to(self.device)
        self._setup_params()

    def _setup_params(self):
        self.alphas = 1 - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, 0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)
        self.temporal_reweight = torch.FloatTensor([(self.cfg.future_frames + 1 - i) for i in range(1, self.cfg.future_frames + 1)]).to(self.device).view(1, 1, -1) / 10

    def fit(self):
        best_ade = float('inf')
        for epoch in range(1, self.cfg.num_epochs + 1):
            loss_dict = self._train_single_epoch(epoch)
            self.writer.add_scalar('Loss/Total', loss_dict['total'], epoch)
            
            if epoch % self.cfg.get('test_interval', 2) == 0:
                performance, _ = self._test_single_epoch()
                curr_ade = performance['ADE'][-1] / len(self.test_loader.dataset)
                self.writer.add_scalar('Metric/ADE', curr_ade, epoch)

                # Best 모델 저장: ckpts/{experiment_mode}/best.pt
                if curr_ade < best_ade:
                    best_ade = curr_ade
                    torch.save({
                        'epoch': epoch,
                        'model_initializer_dict': self.model_initializer.state_dict(),
                        'best_ade': best_ade,
                        'experiment_mode': self.experiment_mode
                    }, str(self.best_path))
                    print(f"⭐ Best Model Saved to {self.best_path} (ADE: {best_ade:.4f})")
            
            self.scheduler_model.step()

# ==============================================================================
# 3. Main 실행 루프
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to yaml file (e.g., cfg/exp1.yaml)')
    args = parser.parse_args()

    prepare_seed(42)
    trainer = AutomatedLEDTrainer(args)
    trainer.fit()

if __name__ == "__main__":
    main()