import argparse
import os
from unittest import loader
import yaml
import numpy as np
from highD.preprocess import T_H
import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import random
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.nn.attention import SDPBackend, sdpa_kernel

from models.model_led_initializer import LEDInitializer
from models.denoiser import (
    TransformerDenoisingModel,
    setup_diffusion,
    noise_estimation_loss,
    p_sample_loop_accelerate,
)
from highD.dataset import HighDDataset

def setup_ddp():
    """DDP 환경 초기화"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

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

def get_dataloader(cfg, split='train', is_ddp=True):
    data_path = (
        Path(cfg['data']['base_dir']) / cfg['exp']['feature_mode'] / f"{split}.npz"
    )
    dataset = HighDDataset(data_path)
    
    sampler = DistributedSampler(dataset) if is_ddp else None

    loader_kwargs = dict(
        batch_size=cfg['data']['batch_size'],
        shuffle=(sampler is None and split == 'train'),
        num_workers=cfg['data']['num_workers'],
        pin_memory=True,
        sampler=sampler,
        prefetch_factor=4,
        persistent_workers=True
    )

    print(f"  [{split}] Loading from {data_path} ...")
    return DataLoader(dataset, **loader_kwargs)


def data_preprocess(batch, device, cfg):
    past_traj = batch['past_traj'].to(device)     # (B, 9, T_H, 6)
    fut_traj = batch['fut_traj'].to(device)       # (B, T_F, 2)
    initial_pos = batch['initial_pos'].to(device) # (B, 2)
    
    B = past_traj.shape[0] # 512
    N = past_traj.shape[1] # 9
    T_H = cfg['data']['history_frames']

    past_traj_flat = past_traj.reshape(-1, T_H, 6)

    traj_mask = torch.ones((N, N), device=device) 

    return B, traj_mask, past_traj_flat, fut_traj, initial_pos


# ==============================================================================
# Stage 1: Denoiser 사전 학습
# ==============================================================================

def run_stage1(model_denoiser, train_loader, val_loader,
               diffusion, cfg, device, denoiser_ckpt_path, writer):
    """
    논문 Section 4.4 Stage 1: L_NE = ||ε - f_ε(Y^(γ+1), context, γ+1)||²
    denoiser_ckpt_path 가 없을 때만 호출됩니다.
    학습이 끝나면 best checkpoint를 denoiser_ckpt_path 에 저장합니다.
    """
    s1 = cfg.get('stage1', cfg['train'])
    epochs = s1.get('epochs', 100)

    optimizer = optim.Adam(
        model_denoiser.parameters(),
        lr=s1.get('lr', 1e-4),
        weight_decay=s1.get('weight_decay', 0.0),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-6,
    )

    denoiser_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float('inf')

    use_amp = cfg.get('use_amp', True) and (device.type == 'cuda')

    print(f"\n{'=' * 20}  Stage 1: Denoiser Pre-training  {'=' * 20}")
    print(f"  epochs={epochs}  lr={optimizer.param_groups[0]['lr']:.2e}  BF16={use_amp}")
    print(f"  checkpoint → {denoiser_ckpt_path}\n")

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model_denoiser.train()
        total = 0.
        pbar  = tqdm(train_loader, desc=f'[S1] Train', dynamic_ncols=True)
        for batch in pbar:
            B, traj_mask, past_traj, fut_traj, _ = data_preprocess(batch, device, cfg)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                loss = noise_estimation_loss(model_denoiser, fut_traj, past_traj, traj_mask, diffusion)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_denoiser.parameters(), 1.)
            optimizer.step()
            total += loss.item()
            pbar.set_postfix(L=f"{loss.item():.4f}")
        train_loss = total / len(train_loader)

        # ── Validate ───────────────────────────────────────────────────────
        model_denoiser.eval()
        total = 0.
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'[S1] Valid', dynamic_ncols=True):
                B, traj_mask, past_traj, fut_traj, _ = data_preprocess(batch, device, cfg)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                    total += noise_estimation_loss(
                        model_denoiser, fut_traj, past_traj, traj_mask, diffusion
                    ).item()
        val_loss = total / len(val_loader)
        scheduler.step(val_loss)

        print(f"[S1] Epoch {epoch:3d}/{epochs} | Train {train_loss:.4f} | Val {val_loss:.4f}")
        writer.add_scalar('Stage1/Loss/train', train_loss, epoch)
        writer.add_scalar('Stage1/Loss/val',   val_loss,   epoch)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'epoch':               epoch,
                'model_denoiser_dict': model_denoiser.state_dict(),
                'val_loss':            val_loss,
            }, denoiser_ckpt_path)
            print(f"  >>> Best denoiser saved (val_loss={val_loss:.4f})")

    print(f"\nStage 1 완료. Best val_loss={best_val:.4f}\n")


# ==============================================================================
# Model utilities
# ==============================================================================

def count_parameters(model_initializer, model_denoiser):
    init_p = sum(p.numel() for p in model_initializer.parameters() if p.requires_grad)
    deno_p = sum(p.numel() for p in model_denoiser.parameters())
    total  = init_p + deno_p
    mem    = total * 4 / (1024 ** 2)
    print("=" * 50)
    print("Model Size Info")
    print(f"  Initializer (trainable) : {init_p:,}")
    print(f"  Denoiser    (frozen)    : {deno_p:,}")
    print(f"  Total Parameters        : {total:,}")
    print(f"  Model Memory Size       : {mem:.2f} MB")
    print("=" * 50)
    print()


# ==============================================================================
# Train / Validate
# ==============================================================================

def train_epoch(model_initializer, model_denoiser, loader, optimizer, 
                diffusion, temporal_reweight, cfg, device, local_rank, epoch):
    model_initializer.train()
    model_denoiser.eval()
    
    # DDP 환경에서 epoch마다 셔플링이 잘 되도록 설정
    loader.sampler.set_epoch(epoch)
    
    num_node = cfg['model']['num_node']
    use_amp = cfg.get('use_amp', True)
    
    total_loss = 0.
    pbar = tqdm(loader, desc=f'Train (GPU {local_rank})', disable=(local_rank != 0))

    for batch in pbar:
        B, traj_mask, past_traj, fut_traj, _ = data_preprocess(batch, device, cfg)
        ego_idx = torch.arange(B, device=device) * num_node

        # 1. Initializer (BF16 Autocast)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
            sample_pred, mean_est, var_est = model_initializer(past_traj, traj_mask)
            # sample_pred : (B*num_node, K=20, T_F, 2)
            # mean_est    : (B*num_node, T_F, 2)
            # var_est     : (B*num_node, 1)
            sample_pred = (
                torch.exp(var_est / 2)[..., None, None] * sample_pred
                / sample_pred.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
            )
            loc = sample_pred + mean_est[:, None]   # (B*num_node, K, T_F, 2)

        # ── Denoiser inference: frozen → float32 (BF16+head_dim=256 Flash Attn 오류 방지)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                generated_y = p_sample_loop_accelerate(
                    model_denoiser, diffusion, past_traj, traj_mask, loc.detach(),
                    num_tau=cfg.get('diffusion', {}).get('num_tau', 5) # 논문 권장 τ=5 [cite: 268]
                )

        # ── Ego-only predictions & Loss ──────────────────────────────────────
        ego_loc = loc[ego_idx]                     # (B, K, T_F, 2) — BF16, gradient 있음
        ego_gen = generated_y[ego_idx]             # (B, K, T_F, 2) — float32, uncertainty 보정용
        ego_var = var_est[ego_idx].squeeze(-1)     # (B,) — BF16
        fut_k   = fut_traj.unsqueeze(1)            # (B, 1, T_F, 2)

        loss_dist = (
            (ego_loc - fut_k).norm(p=2, dim=-1) * temporal_reweight
        ).mean(dim=-1).min(dim=1)[0].mean()

        norms    = (ego_gen - fut_k).norm(p=2, dim=-1).mean(dim=(1, 2))  # (B,) detached — OK
        loss_unc = (torch.exp(-ego_var) * norms + ego_var).mean()

        loss = loss_dist * 50 + loss_unc

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_initializer.parameters(), 1.)
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model_initializer, model_denoiser, loader, diffusion, cfg, device):
    model_initializer.eval()
    model_denoiser.eval()
    num_node = cfg['model']['num_node']
    use_amp  = cfg.get('use_amp', True) and (device.type == 'cuda')

    ade_sum = fde_sum = 0.
    samples = 0

    pbar = tqdm(loader, desc='Valid', dynamic_ncols=True)
    with torch.no_grad():
        for batch in pbar:
            B, traj_mask, past_traj, fut_traj, _ = data_preprocess(batch, device, cfg)
            ego_idx = torch.arange(B, device=device) * num_node

            # Initializer (BF16 autocast)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                sample_pred, mean_est, var_est = model_initializer(past_traj, traj_mask)
                sample_pred = (
                    torch.exp(var_est / 2)[..., None, None] * sample_pred
                    / sample_pred.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
                )
                loc = sample_pred + mean_est[:, None]

            # Denoiser inference: float32 (BF16+head_dim=256 Flash Attn 오류 방지)
            pred_traj = p_sample_loop_accelerate(
                model_denoiser, diffusion, past_traj, traj_mask, loc.float(),
                num_tau=cfg.get('diffusion', {}).get('num_tau', 5),
            )  # (B*num_node, K, T_F, 2)

            ego_pred  = pred_traj[ego_idx]                          # (B, K, T_F, 2)
            fut_k     = fut_traj.unsqueeze(1).expand_as(ego_pred)   # (B, K, T_F, 2)
            distances = torch.norm(fut_k - ego_pred, dim=-1)        # (B, K, T_F)

            ade_sum  += distances.mean(dim=-1).min(dim=-1)[0].sum().item()
            fde_sum  += distances[:, :, -1].min(dim=-1)[0].sum().item()
            samples  += B

            pbar.set_postfix(ADE=f"{ade_sum / samples:.4f}")

    return ade_sum / samples, fde_sum / samples


# ==============================================================================
# Main
# ==============================================================================

def main():
    # 1. DDP 설정
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    args = parser.parse_args()

    cfg    = load_config(args.config)
    seed_everything(cfg['exp']['seed'])

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    feature_mode = cfg['exp']['feature_mode']
    exp_tag      = cfg['exp'].get('exp_tag', feature_mode)
    T_H = cfg['data']['history_frames']
    T_F = cfg['data']['future_frames']

    # ── Checkpoint 경로 (exp_tag 하위로 통일) ─────────────────────────────────
    ckpt_dir        = Path(cfg['train']['ckpt_dir']) / exp_tag
    denoiser_ckpt   = ckpt_dir / 'denoiser' / 'best.pt'
    best_path       = ckpt_dir / 'best.pt'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print(f"  Experiment : {feature_mode}  (tag: {exp_tag})")
    print(f"  T_H={T_H}  T_F={T_F}  Device={device}")
    print(f"  Checkpoint dir: {ckpt_dir}")
    print("=" * 55)

    # ── Models ────────────────────────────────────────────────────────────────
    model_initializer = LEDInitializer(t_h=T_H, d_h=6, t_f=T_F, d_f=2, k_pred=20).to(device)
    model_denoiser = TransformerDenoisingModel(t_h=T_H, d_h=6).to(device)
    
    model_initializer = DDP(model_initializer, device_ids=[local_rank])

    # ── Diffusion setup (Stage 1/2 공통) ──────────────────────────────────────
    diffusion = setup_diffusion(cfg, device)

    # ── Logging (Stage 1/2 공통 writer) ───────────────────────────────────────
    log_dir = Path('logs') / exp_tag / datetime.now().strftime('%m%d-%H%M')
    writer  = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard log -> {log_dir}")

    # ── Data (Stage 1/2 공통 loader) ──────────────────────────────────────────
    train_loader = get_dataloader(cfg, 'train', is_ddp=True)
    val_loader = get_dataloader(cfg, 'val', is_ddp=False)
    print(f"Train: {len(train_loader.dataset):,}  /  Val: {len(val_loader.dataset):,}\n")

    # ── Stage 1: Denoiser 사전 학습 (checkpoint 없을 때만 실행) ───────────────
    if not denoiser_ckpt.exists():
        run_stage1(model_denoiser, train_loader, val_loader,
                   diffusion, cfg, device, denoiser_ckpt, writer)

    ckpt = torch.load(denoiser_ckpt, map_location=device)
    model_denoiser.load_state_dict(ckpt['model_denoiser_dict'])
    print(f"Denoiser loaded: {denoiser_ckpt}  (epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f})")

    # Denoiser는 Stage 2에서 완전히 frozen
    model_denoiser.eval()
    for p in model_denoiser.parameters():
        p.requires_grad_(False)

    # ── Stage 2 준비 ──────────────────────────────────────────────────────────
    temporal_reweight = (
        torch.FloatTensor([(T_F + 1 - i) for i in range(1, T_F + 1)])
        .to(device).view(1, 1, -1) / 10
    )

    optimizer = optim.AdamW(
        model_initializer.parameters(),
        lr=cfg['train']['lr'],
        weight_decay=cfg['train']['weight_decay'],
        fused=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['train']['epochs'],
        eta_min=cfg['train'].get('lr_min', 1e-6),
    )

    start_epoch  = 1
    best_val_ade = float('inf')

    count_parameters(model_initializer, model_denoiser)

    if cfg.get('compile', True) and hasattr(torch, 'compile'):
        compile_mode      = cfg.get('compile_mode', 'reduce-overhead')
        model_initializer = torch.compile(model_initializer, mode=compile_mode)
        model_denoiser    = torch.compile(model_denoiser,    mode=compile_mode)

    # ── Stage 2 Training loop ─────────────────────────────────────────────────
    print(f"\n{'=' * 20}  Stage 2: Initializer Training  {'=' * 20}\n")
    epochs = cfg['train']['epochs']
    for epoch in range(start_epoch, epochs + 1):
        print(f"{'=' * 30}  Epoch {epoch}/{epochs}  {'=' * 30}")

        avg_loss = train_epoch(
            model_initializer, model_denoiser, train_loader,
            optimizer, diffusion, temporal_reweight, cfg, device, local_rank, epoch
        )
        scheduler.step()
        
        if local_rank == 0:  # DDP에서는 rank 0만 검증 및 체크포인트 저장
            val_ade, val_fde = validate(
                model_initializer, model_denoiser, val_loader, diffusion, cfg, device,
            )

            print(
                f"Epoch {epoch:3d} | "
                f"Loss {avg_loss:.4f} | "
                f"Val ADE {val_ade:.4f} | Val FDE {val_fde:.4f}"
            )

            writer.add_scalar('Loss/train',       avg_loss, epoch)
            writer.add_scalar('Val/ADE',          val_ade,    epoch)
            writer.add_scalar('Val/FDE',          val_fde,    epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

            if val_ade < best_val_ade:
                best_val_ade = val_ade
                torch.save({
                    'epoch':                  epoch,
                    'model_initializer_dict': model_initializer.state_dict(),
                    'optimizer_state_dict':   optimizer.state_dict(),
                    'scheduler_state_dict':   scheduler.state_dict(),
                    'val_ade':                val_ade,
                    'val_fde':                val_fde,
                    'feature_mode':           feature_mode,
                }, best_path)
                print(f"Best model saved to {best_path}  (ADE: {val_ade:.4f})")

            print("-" * 50)
            print()

    writer.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()