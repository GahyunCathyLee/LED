import argparse
import math
import yaml
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.attention import SDPBackend, sdpa_kernel

from models.model_led_initializer import LEDInitializer
from models.denoiser import (
    TransformerDenoisingModel,
    setup_diffusion,
    p_sample_loop_accelerate,
)
from highD.dataset import HighDDataset


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def strip_prefix(state_dict):
    """DDP / torch.compile 래핑으로 생긴 키 접두어 제거."""
    prefixes = ('_orig_mod.module.', '_orig_mod.', 'module.')
    new_sd = {}
    for k, v in state_dict.items():
        for prefix in prefixes:
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        new_sd[k] = v
    return new_sd


def data_preprocess(batch, device, cfg):
    past_traj  = batch['past_traj'].to(device)      # (B, 9, T_H, 6)
    fut_traj   = batch['fut_traj'].to(device)        # (B, T_F, 2)
    initial_pos = batch['initial_pos'].to(device)    # (B, 2)

    B   = past_traj.shape[0]
    N   = past_traj.shape[1]
    T_H = cfg['data']['history_frames']

    past_traj_flat = past_traj.reshape(-1, T_H, 6)
    traj_mask      = torch.ones((N, N), device=device)

    return B, traj_mask, past_traj_flat, fut_traj, initial_pos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--ckpt',   type=str, required=True, help='Path to initializer checkpoint')
    args = parser.parse_args()

    cfg = load_config(args.config)

    device_str = cfg['exp'].get('device', 'cuda:0')
    device     = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    T_H          = cfg['data']['history_frames']
    T_F          = cfg['data']['future_frames']
    num_node     = cfg['model']['num_node']
    feature_mode = cfg['exp']['feature_mode']
    fps          = T_F // 5   # 5Hz × 5s = 25 frames → fps=5
    use_amp      = cfg.get('use_amp', True) and (device.type == 'cuda')
    num_tau      = cfg.get('diffusion', {}).get('num_tau', 5)

    # ── Checkpoint 경로 ──────────────────────────────────────────────────────
    ckpt_path         = Path(args.ckpt)
    denoiser_ckpt_path = ckpt_path.parent / 'denoiser' / 'best.pt'

    # ── 모델 생성 ─────────────────────────────────────────────────────────────
    model_initializer = LEDInitializer(t_h=T_H, d_h=6, t_f=T_F, d_f=2, k_pred=20).to(device)
    model_denoiser    = TransformerDenoisingModel(t_h=T_H, d_h=6).to(device)

    # Denoiser 로드
    deno_ckpt = torch.load(denoiser_ckpt_path, map_location=device, weights_only=True)
    model_denoiser.load_state_dict(strip_prefix(deno_ckpt['model_denoiser_dict']))

    # Initializer 로드
    init_ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model_initializer.load_state_dict(strip_prefix(init_ckpt['model_initializer_dict']))
    print(f"\n✅ Checkpoint Loaded: {ckpt_path}")

    model_initializer.eval()
    model_denoiser.eval()
    for p in model_denoiser.parameters():
        p.requires_grad_(False)

    # ── 데이터 ────────────────────────────────────────────────────────────────
    data_path = Path(cfg['data']['base_dir']) / feature_mode / 'test.npz'
    print(f"\n[test] 데이터를 RAM에 로드 중: {data_path}")
    dataset = HighDDataset(data_path)
    loader  = DataLoader(
        dataset,
        batch_size=cfg['data']['batch_size'],
        shuffle=False,
        num_workers=cfg['data']['num_workers'],
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=cfg['data'].get('persistent_workers', True),
    )
    print(f"✅ {len(dataset):,}개의 샘플이 RAM 적재 완료되었습니다.")

    # ── Diffusion 설정 ────────────────────────────────────────────────────────
    diffusion = setup_diffusion(cfg, device)

    # ── 평가 ──────────────────────────────────────────────────────────────────
    print(f"\n🔍 평가 시작 (Samples: {len(dataset):,})")

    ade_sum    = 0.
    fde_sum    = 0.
    sq_sum_all = 0.                      # RMSE (전체 프레임)
    sq_sum_t   = torch.zeros(T_F)        # RMSE@T (프레임별)
    samples    = 0

    pbar = tqdm(loader, desc='Eval', dynamic_ncols=True)
    with torch.no_grad():
        for batch in pbar:
            B, traj_mask, past_traj, fut_traj, _ = data_preprocess(batch, device, cfg)
            ego_idx = torch.arange(B, device=device) * num_node

            # Initializer
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                sample_pred, mean_est, var_est = model_initializer(past_traj, traj_mask)
                sample_pred = (
                    torch.exp(var_est / 2)[..., None, None] * sample_pred
                    / sample_pred.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
                )
                loc = sample_pred + mean_est[:, None]   # (B*N, K, T_F, 2)

            # Denoiser (역방향 diffusion)
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                pred_traj = p_sample_loop_accelerate(
                    model_denoiser, diffusion, past_traj, traj_mask, loc.float(),
                    num_tau=num_tau,
                )   # (B*num_node, K, T_F, 2)

            ego_pred  = pred_traj[ego_idx]                           # (B, K, T_F, 2)
            fut_k     = fut_traj.unsqueeze(1).expand_as(ego_pred)    # (B, K, T_F, 2)
            distances = torch.norm(fut_k - ego_pred, dim=-1)         # (B, K, T_F)

            # ADE / FDE: min over K
            ade_sum += distances.mean(dim=-1).min(dim=1)[0].sum().item()
            fde_sum += distances[:, :, -1].min(dim=1)[0].sum().item()

            # RMSE: minADE 궤적 선택 후 프레임별 오차
            best_k   = distances.mean(dim=-1).argmin(dim=1)                     # (B,)
            best_dist = distances[torch.arange(B, device=device), best_k, :]    # (B, T_F)

            sq_sum_all += best_dist.pow(2).sum().item()
            sq_sum_t   += best_dist.pow(2).sum(dim=0).cpu()
            samples    += B

    # ── 최종 지표 계산 ────────────────────────────────────────────────────────
    ade  = ade_sum / samples
    fde  = fde_sum / samples
    rmse = math.sqrt(sq_sum_all / (samples * T_F))   # 5초 평균 RMSE

    rmse_per_t = (sq_sum_t / samples).sqrt()          # (T_F,)
    rmse_at_s  = [rmse_per_t[fps * s - 1].item() for s in range(1, 6)]

    # ── 출력 ──────────────────────────────────────────────────────────────────
    print()
    print("=" * 50)
    print("📊 Final Results (m)")
    print(f"  • ADE  : {ade:.4f}")
    print(f"  • FDE  : {fde:.4f}")
    print(f"  • RMSE : {rmse:.4f}")
    print("-" * 50)
    for s, r in enumerate(rmse_at_s, start=1):
        print(f"  • RMSE@{s}s  : {r:.4f}")
    print("=" * 50)


if __name__ == '__main__':
    main()
