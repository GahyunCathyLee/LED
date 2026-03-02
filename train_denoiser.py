"""
train_denoiser.py  —  LED Stage 1: Denoiser 사전 학습
──────────────────────────────────────────────────────
논문 Section 4.4 기준:
    L_NE = ||ε - f_ε(Y^(γ+1), f_context(X, X_N), γ+1)||²
    γ ~ U{1, ..., Γ},  ε ~ N(0, I)

사용법:
    python train_denoiser.py --config configs/baseline.yaml

학습 완료 후 train.py를 실행하면 denoiser checkpoint를 자동으로 로드합니다.
checkpoint 경로: {train.ckpt_dir}/{exp.exp_tag}/denoiser/best.pt
"""

import argparse
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from highD.dataset import HighDDataset
from models.denoiser import (
    TransformerDenoisingModel,
    noise_estimation_loss,
    setup_diffusion,
)

NUM_NODE = 9


# ==============================================================================
# Config & Seed
# ==============================================================================

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ==============================================================================
# Data
# ==============================================================================

def get_dataloader(cfg, split='train'):
    data_path = (
        Path(cfg['data']['base_dir']) / cfg['exp']['feature_mode'] / f"{split}.npz"
    )
    dataset     = HighDDataset(data_path)
    num_workers = cfg['data']['num_workers']

    loader_kwargs = dict(
        batch_size  = cfg['data']['batch_size'],
        shuffle     = (split == 'train'),
        num_workers = num_workers,
        pin_memory  = True,
    )
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = cfg['data'].get('persistent_workers', True)
        loader_kwargs['prefetch_factor']    = 4

    print(f"  [{split}] {data_path}")
    return DataLoader(dataset, **loader_kwargs)


def data_preprocess(batch, device, T_H):
    """
    Returns
    -------
    past_traj : (B*N, T_H, 6)
    fut_traj  : (B, T_F, 2)
    traj_mask : (N, N)  — 모든 노드가 서로 참조 가능한 binary mask
    """
    past = batch['past_traj'].to(device)   # (B, N, T_H, 6)
    fut  = batch['fut_traj'].to(device)    # (B, T_F, 2)
    past_flat  = past.reshape(-1, T_H, 6) # (B*N, T_H, 6)
    traj_mask  = torch.ones((NUM_NODE, NUM_NODE), device=device)
    return past_flat, fut, traj_mask


# ==============================================================================
# Train / Validate
# ==============================================================================

def train_epoch(model, loader, optimizer, diffusion, T_H, device, use_amp):
    model.train()
    total_loss = 0.
    pbar = tqdm(loader, desc='Train', dynamic_ncols=True)

    for batch in pbar:
        past_traj, fut_traj, traj_mask = data_preprocess(batch, device, T_H)

        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
            loss = noise_estimation_loss(model, fut_traj, past_traj, traj_mask, diffusion)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(L=f"{loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, diffusion, T_H, device, use_amp):
    model.eval()
    total_loss = 0.
    pbar = tqdm(loader, desc='Valid', dynamic_ncols=True)

    for batch in pbar:
        past_traj, fut_traj, traj_mask = data_preprocess(batch, device, T_H)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
            loss = noise_estimation_loss(model, fut_traj, past_traj, traj_mask, diffusion)
        total_loss += loss.item()
        pbar.set_postfix(L=f"{loss.item():.4f}")

    return total_loss / len(loader)


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    args = parser.parse_args()

    cfg    = load_config(args.config)
    seed_everything(cfg['exp']['seed'])
    device = torch.device(cfg['exp']['device'] if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    use_amp = cfg.get('use_amp', True) and (device.type == 'cuda')

    feature_mode = cfg['exp']['feature_mode']
    exp_tag      = cfg['exp'].get('exp_tag', feature_mode)
    T_H = cfg['data']['history_frames']

    # Stage 1 전용 하이퍼파라미터: config에 'stage1' 섹션이 없으면 'train' 섹션을 사용
    s1 = cfg.get('stage1', cfg['train'])

    ckpt_dir  = Path(cfg['train']['ckpt_dir']) / exp_tag / 'denoiser'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / 'best.pt'

    print("=" * 55)
    print(f"  Stage 1 — Denoiser Pre-training")
    print(f"  Experiment : {feature_mode}  (tag: {exp_tag})")
    print(f"  T_H={T_H}  Device={device}  BF16={use_amp}")
    print(f"  Checkpoint → {best_path}")
    print("=" * 55)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TransformerDenoisingModel(t_h=T_H, d_h=6, num_node=NUM_NODE).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Denoiser params: {n_params:,}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = optim.Adam(
        model.parameters(),
        lr=s1.get('lr', 1e-4),
        weight_decay=s1.get('weight_decay', 0.0),
    )
    # val_loss가 patience epoch 동안 안 내려가면 lr을 절반으로
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-6,
    )

    # ── Diffusion setup ───────────────────────────────────────────────────────
    diffusion = setup_diffusion(cfg, device)
    print(f"  Diffusion steps: {diffusion['n_steps']}")

    # ── Logging & Data ────────────────────────────────────────────────────────
    log_dir = Path('logs') / feature_mode / 'denoiser' / datetime.now().strftime('%m%d-%H%M')
    writer  = SummaryWriter(log_dir=str(log_dir))
    print(f"  TensorBoard → {log_dir}\n")

    train_loader = get_dataloader(cfg, 'train')
    val_loader   = get_dataloader(cfg, 'val')
    print(f"  Train: {len(train_loader.dataset):,}  /  Val: {len(val_loader.dataset):,}\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    epochs   = s1.get('epochs', 100)
    best_val = float('inf')

    for epoch in range(1, epochs + 1):
        print(f"{'=' * 25}  Epoch {epoch}/{epochs}  {'=' * 25}")

        train_loss = train_epoch(model, train_loader, optimizer, diffusion, T_H, device, use_amp)
        val_loss   = validate(model, val_loader, diffusion, T_H, device, use_amp)
        scheduler.step(val_loss)

        lr_now = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch:3d} | Train {train_loss:.4f} | Val {val_loss:.4f} | LR {lr_now:.2e}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val',   val_loss,   epoch)
        writer.add_scalar('LR',         lr_now,     epoch)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'epoch':               epoch,
                'model_denoiser_dict': model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'val_loss':            val_loss,
                'feature_mode':        feature_mode,
            }, best_path)
            print(f"  >>> Best saved → {best_path}  (val_loss: {val_loss:.4f})")

        print()

    writer.close()
    print(f"\nStage 1 완료. Checkpoint: {best_path}")
    print("이제 config의 'train.denoiser_ckpt'에 위 경로를 추가하고 train.py를 실행하세요.")


if __name__ == '__main__':
    main()
