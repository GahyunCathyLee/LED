import argparse
import os
import yaml
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import random

from models.model_led_initializer import LEDInitializer
from models.model_diffusion import TransformerDenoisingModel
from highD.dataset import HighDDataset

NUM_TAU = 5  # accelerated diffusion sampling steps


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
    dataset = HighDDataset(data_path)
    num_workers = cfg['data']['num_workers']
    shuffle = (split == 'train')

    loader_kwargs = dict(
        batch_size=cfg['data']['batch_size'],
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = cfg['data'].get('persistent_workers', True)
        loader_kwargs['prefetch_factor'] = 4

    print(f"  [{split}] Loading from {data_path} ...")
    return DataLoader(dataset, **loader_kwargs)


def data_preprocess(batch, device, cfg):
    """
    Convert HighDDataset batch → LED model inputs.

    Input shapes (from HighDDataset):
        ego_past  : (B, T_H, 2)          absolute x, y
        nbr_past  : (B, 8, T_H, D_nbr)   relative neighbor features
        target    : (B, T_F, 2)           absolute future x, y

    Returns
    -------
    B           : int
    traj_mask   : (B * num_node, B * num_node)   block-diagonal
    past_traj   : (B * num_node, T_H, 6)         6-dim augmented features
    fut_traj    : (B, T_F, 2)                     ego-only, relative to last obs
    initial_pos : (B, 1, 2)                       last observed ego position
    """
    num_node = cfg['model']['num_node']   # 9  (ego + 8 neighbors)
    T_H      = cfg['data']['history_frames']

    ego_past = batch['ego_past'].to(device)   # (B, T_H, 2)
    nbr_past = batch['nbr_past'].to(device)   # (B, 8, T_H, D_nbr)
    target   = batch['target'].to(device)     # (B, T_F, 2)
    B = ego_past.shape[0]

    # ── Ego feature augmentation: rel-pos + vel + acc  →  6-dim ──────────────
    initial_pos = ego_past[:, -1:, :]                        # (B, 1, 2)
    ego_rel     = ego_past - initial_pos                      # (B, T_H, 2)
    ego_vel     = torch.zeros_like(ego_rel)
    ego_vel[:, :-1] = ego_rel[:, 1:] - ego_rel[:, :-1]      # forward diff
    ego_acc     = torch.zeros_like(ego_vel)
    ego_acc[:, :-1] = ego_vel[:, 1:] - ego_vel[:, :-1]
    ego_feat = torch.cat([ego_rel, ego_vel, ego_acc], dim=-1)  # (B, T_H, 6)

    # ── Neighbor features: pad / truncate to 6-dim ───────────────────────────
    D_nbr = nbr_past.shape[-1]
    if D_nbr < 6:
        pad      = torch.zeros(B, 8, T_H, 6 - D_nbr, device=device)
        nbr_feat = torch.cat([nbr_past, pad], dim=-1)          # (B, 8, T_H, 6)
    else:
        nbr_feat = nbr_past[..., :6]                           # (B, 8, T_H, 6)

    # ── Stack ego + neighbors  →  (B * num_node, T_H, 6) ────────────────────
    all_feat  = torch.cat([ego_feat.unsqueeze(1), nbr_feat], dim=1)  # (B, 9, T_H, 6)
    past_traj = all_feat.reshape(B * num_node, T_H, 6)

    # ── Block-diagonal social mask ────────────────────────────────────────────
    traj_mask = torch.kron(
        torch.eye(B, device=device),
        torch.ones(num_node, num_node, device=device),
    )

    # ── Future trajectory: ego only, relative to last observed position ───────
    fut_traj = target - initial_pos   # (B, T_F, 2)

    return B, traj_mask, past_traj, fut_traj, initial_pos


# ==============================================================================
# Diffusion utilities
# ==============================================================================

def make_beta_schedule(schedule='linear', n_timesteps=100, start=1e-5, end=1e-2):
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == 'quad':
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == 'sigmoid':
        betas = torch.linspace(-6, 6, n_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    return betas


def setup_diffusion(cfg, device):
    diff_cfg = cfg.get('diffusion', {})
    n_steps  = diff_cfg.get('steps', 100)
    betas    = make_beta_schedule(
        schedule=diff_cfg.get('beta_schedule', 'linear'),
        n_timesteps=n_steps,
        start=diff_cfg.get('beta_start', 1e-5),
        end=diff_cfg.get('beta_end',   1e-2),
    ).to(device)
    alphas      = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    return {
        'n_steps':                  n_steps,
        'betas':                    betas,
        'alphas':                   alphas,
        'alphas_bar_sqrt':          torch.sqrt(alphas_prod),
        'one_minus_alphas_bar_sqrt': torch.sqrt(1 - alphas_prod),
    }


def _extract(values, t, x):
    """Gather diffusion coefficients at timestep t and reshape for broadcasting."""
    out = torch.gather(values, 0, t.to(values.device))
    return out.reshape([t.shape[0]] + [1] * (len(x.shape) - 1))


def _p_sample_accelerate(model_denoiser, diffusion, x, mask, cur_y, t):
    """Single reverse-diffusion step (accelerated, very small noise)."""
    t_tensor  = torch.tensor([t], device=x.device)
    eps_factor = (
        (1 - _extract(diffusion['alphas'], t_tensor, cur_y))
        / _extract(diffusion['one_minus_alphas_bar_sqrt'], t_tensor, cur_y)
    )
    beta      = _extract(diffusion['betas'], t_tensor.repeat(x.shape[0]), cur_y)
    eps_theta = model_denoiser.generate_accelerate(cur_y, beta, x, mask)
    mean      = (
        (1 / _extract(diffusion['alphas'], t_tensor, cur_y).sqrt())
        * (cur_y - eps_factor * eps_theta)
    )
    sigma_t = _extract(diffusion['betas'], t_tensor, cur_y).sqrt()
    return mean + sigma_t * torch.randn_like(cur_y) * 0.00001


def _p_sample_loop_accelerate(model_denoiser, diffusion, x, mask, loc):
    """
    Batch-parallel accelerated sampling over K=20 predictions.
    loc: (B*num_node, K=20, T_F, 2)
    """
    cur_y  = loc[:, :10]
    for i in reversed(range(NUM_TAU)):
        cur_y  = _p_sample_accelerate(model_denoiser, diffusion, x, mask, cur_y,  i)
    cur_y_ = loc[:, 10:]
    for i in reversed(range(NUM_TAU)):
        cur_y_ = _p_sample_accelerate(model_denoiser, diffusion, x, mask, cur_y_, i)
    return torch.cat((cur_y_, cur_y), dim=1)   # (B*num_node, K=20, T_F, 2)


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
                diffusion, temporal_reweight, cfg, device):
    model_initializer.train()
    model_denoiser.eval()   # denoiser not updated; eval avoids stochastic dropout
    num_node = cfg['model']['num_node']

    total_loss = total_dist = total_unc = 0.
    pbar = tqdm(loader, desc='Train', dynamic_ncols=True)

    for batch in pbar:
        B, traj_mask, past_traj, fut_traj, _ = data_preprocess(batch, device, cfg)

        # Ego node indices within the stacked (B*num_node) tensor: 0, 9, 18, ...
        ego_idx = torch.arange(B, device=device) * num_node

        # ── Initializer ──────────────────────────────────────────────────────
        sample_pred, mean_est, var_est = model_initializer(past_traj, traj_mask)
        # sample_pred : (B*num_node, K=20, T_F, 2)
        # mean_est    : (B*num_node, T_F, 2)
        # var_est     : (B*num_node, 1)

        sample_pred = (
            torch.exp(var_est / 2)[..., None, None] * sample_pred
            / sample_pred.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
        )
        loc = sample_pred + mean_est[:, None]   # (B*num_node, K, T_F, 2)

        # ── Accelerated diffusion sampling ────────────────────────────────────
        with torch.no_grad(): # Denoiser 샘플링 시 gradient 계산 차단
            generated_y = _p_sample_loop_accelerate(
                model_denoiser, diffusion, past_traj, traj_mask, loc
            )

        # ── Ego-only predictions ──────────────────────────────────────────────
        ego_gen = generated_y[ego_idx]             # (B, K, T_F, 2)
        ego_var = var_est[ego_idx].squeeze(-1)     # (B,)
        fut_k   = fut_traj.unsqueeze(1)            # (B, 1, T_F, 2)

        # ── Loss ─────────────────────────────────────────────────────────────
        loss_dist = (
            (ego_gen - fut_k).norm(p=2, dim=-1) * temporal_reweight
        ).mean(dim=-1).min(dim=1)[0].mean()

        norms    = (ego_gen - fut_k).norm(p=2, dim=-1).mean(dim=(1, 2))  # (B,)
        loss_unc = (torch.exp(-ego_var) * norms + ego_var).mean()

        loss = loss_dist * 50 + loss_unc

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_initializer.parameters(), 1.)
        optimizer.step()

        total_loss += loss.item()
        total_dist += loss_dist.item() * 50
        total_unc  += loss_unc.item()
        pbar.set_postfix(L=f"{loss.item():.4f}", D=f"{loss_dist.item():.4f}")

    n = len(loader)
    return total_loss / n, total_dist / n, total_unc / n


def validate(model_initializer, model_denoiser, loader, diffusion, cfg, device):
    model_initializer.eval()
    model_denoiser.eval()
    num_node = cfg['model']['num_node']

    ade_sum = fde_sum = 0.
    samples = 0

    pbar = tqdm(loader, desc='Valid', dynamic_ncols=True)
    with torch.no_grad():
        for batch in pbar:
            B, traj_mask, past_traj, fut_traj, _ = data_preprocess(batch, device, cfg)
            ego_idx = torch.arange(B, device=device) * num_node

            sample_pred, mean_est, var_est = model_initializer(past_traj, traj_mask)
            sample_pred = (
                torch.exp(var_est / 2)[..., None, None] * sample_pred
                / sample_pred.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
            )
            loc = sample_pred + mean_est[:, None]

            pred_traj = _p_sample_loop_accelerate(
                model_denoiser, diffusion, past_traj, traj_mask, loc
            )  # (B*num_node, K, T_F, 2)

            ego_pred = pred_traj[ego_idx]                           # (B, K, T_F, 2)
            fut_k    = fut_traj.unsqueeze(1).repeat(1, 20, 1, 1)   # (B, K, T_F, 2)

            distances = torch.norm(fut_k - ego_pred, dim=-1)       # (B, K, T_F)
            ade_sum  += distances.mean(dim=-1).min(dim=-1)[0].sum().item()
            fde_sum  += distances[:, :, -1].min(dim=-1)[0].sum().item()
            samples  += B

            pbar.set_postfix(ADE=f"{ade_sum / samples:.4f}")

    return ade_sum / samples, fde_sum / samples


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

    feature_mode = cfg['exp']['feature_mode']
    T_H = cfg['data']['history_frames']
    T_F = cfg['data']['future_frames']

    ckpt_dir  = Path(cfg['train']['ckpt_dir']) / feature_mode
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / 'best.pt'

    print("=" * 50)
    print(f"  Experiment : {feature_mode}")
    print(f"  T_H={T_H}  T_F={T_F}  Device={device}")
    print("=" * 50)

    # ── Models ────────────────────────────────────────────────────────────────
    model_initializer = LEDInitializer(
        t_h=T_H, d_h=6, t_f=T_F, d_f=2, k_pred=20
    ).to(device)
    model_denoiser = TransformerDenoisingModel(t_h=T_H, d_h=6).to(device)

    optimizer = optim.AdamW(
        model_initializer.parameters(),
        lr=cfg['train']['lr'],
        weight_decay=cfg['train']['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['train']['epochs'],
        eta_min=cfg['train'].get('lr_min', 1e-6),
    )

    # ── Diffusion setup ───────────────────────────────────────────────────────
    diffusion = setup_diffusion(cfg, device)
    temporal_reweight = (
        torch.FloatTensor([(T_F + 1 - i) for i in range(1, T_F + 1)])
        .to(device).view(1, 1, -1) / 10
    )

    start_epoch  = 1
    best_val_ade = float('inf')

    count_parameters(model_initializer, model_denoiser)

    if cfg.get('compile', True) and hasattr(torch, 'compile'):
        model_initializer = torch.compile(model_initializer)
        model_denoiser    = torch.compile(model_denoiser)

    # ── Logging & data ────────────────────────────────────────────────────────
    log_dir = Path('logs') / feature_mode / datetime.now().strftime('%m%d-%H%M')
    writer  = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard log -> {log_dir}")

    train_loader = get_dataloader(cfg, 'train')
    val_loader   = get_dataloader(cfg, 'val')
    print(f"Train: {len(train_loader.dataset):,}  /  Val: {len(val_loader.dataset):,}\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    epochs = cfg['train']['epochs']
    for epoch in range(start_epoch, epochs + 1):
        print(f"{'=' * 30}  Epoch {epoch}/{epochs}  {'=' * 30}")

        train_loss, dist_loss, unc_loss = train_epoch(
            model_initializer, model_denoiser, train_loader,
            optimizer, diffusion, temporal_reweight, cfg, device,
        )
        scheduler.step()
        val_ade, val_fde = validate(
            model_initializer, model_denoiser, val_loader, diffusion, cfg, device,
        )

        print(
            f"Epoch {epoch:3d} | "
            f"Loss {train_loss:.4f} (dist={dist_loss:.4f}, unc={unc_loss:.4f}) | "
            f"Val ADE {val_ade:.4f} | Val FDE {val_fde:.4f}"
        )

        writer.add_scalar('Loss/train',       train_loss, epoch)
        writer.add_scalar('Loss/dist',        dist_loss,  epoch)
        writer.add_scalar('Loss/uncertainty', unc_loss,   epoch)
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


if __name__ == '__main__':
    main()
