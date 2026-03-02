import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import PositionalEncoding, ConcatSquashLinear, social_transformer


# ==============================================================================
# Diffusion schedule utilities  (Stage 1 & 2 공통)
# ==============================================================================

def make_beta_schedule(schedule='linear', n_timesteps=100, start=1e-5, end=1e-2):
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == 'quad':
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == 'sigmoid':
        betas = torch.linspace(-6, 6, n_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    return betas


def setup_diffusion(cfg, device):
    diff_cfg = cfg.get('diffusion', {})
    n_steps  = diff_cfg.get('steps', 100)
    betas    = make_beta_schedule(
        schedule   = diff_cfg.get('beta_schedule', 'linear'),
        n_timesteps= n_steps,
        start      = diff_cfg.get('beta_start', 1e-5),
        end        = diff_cfg.get('beta_end',   1e-2),
    ).to(device)
    alphas      = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    return {
        'n_steps':                   n_steps,
        'betas':                     betas,
        'alphas':                    alphas,
        'alphas_bar_sqrt':           torch.sqrt(alphas_prod),
        'one_minus_alphas_bar_sqrt': torch.sqrt(1 - alphas_prod),
    }


# ==============================================================================
# Forward diffusion  (Stage 1)
# ==============================================================================

def q_sample(y_0, t, diffusion, noise=None):
    """
    y_0: (B, T_F, 2)  — 깨끗한 ego 미래 경로
    t:   (B,)         — 정수 timestep 인덱스
    Returns: (y_noisy, noise), 각각 (B, T_F, 2)
    """
    if noise is None:
        noise = torch.randn_like(y_0)
    sqrt_alpha_bar = diffusion['alphas_bar_sqrt'][t].view(-1, 1, 1)        # (B, 1, 1)
    sqrt_one_minus = diffusion['one_minus_alphas_bar_sqrt'][t].view(-1, 1, 1)  # (B, 1, 1)
    y_noisy = sqrt_alpha_bar * y_0 + sqrt_one_minus * noise
    return y_noisy, noise


# ==============================================================================
# Model
# ==============================================================================

class TransformerDenoisingModel(nn.Module):
    """
    TransformerDenoisingModel

    Stage 1 — forward(y_noisy, beta, past_traj, mask)
        ego 궤적 1개에 대한 noise 예측.  y_noisy: (B, T_F, 2)

    Stage 2 — encode_context() 1회 호출 후 generate_accelerate() 반복 호출.
        K 샘플을 한 번에 병렬 denoising.  x: (B*N, K, T_F, 2)
    """

    def __init__(self, context_dim=256, tf_layer=2, t_h=10, d_h=6, num_node=9):
        super().__init__()
        self.num_node = num_node
        self.encoder_context = social_transformer(t_h=t_h, d_h=d_h)
        self.pos_emb   = PositionalEncoding(d_model=2 * context_dim, dropout=0.1, max_len=100)
        self.concat1   = ConcatSquashLinear(2,                context_dim * 2, context_dim + 3)
        self.layer     = nn.TransformerEncoderLayer(
            d_model=2 * context_dim, nhead=2,
            dim_feedforward=2 * context_dim, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=tf_layer)
        self.concat3   = ConcatSquashLinear(context_dim * 2, context_dim,      context_dim + 3)
        self.concat4   = ConcatSquashLinear(context_dim,     context_dim // 2, context_dim + 3)
        self.linear    = ConcatSquashLinear(context_dim // 2, 2,               context_dim + 3)

    # ------------------------------------------------------------------
    # Shared: context encoding
    # ------------------------------------------------------------------

    def encode_context(self, past_traj, mask):
        """
        past_traj: (B*N, T_H, 6)
        mask:      (N, N)  — 0/1 binary mask (not pre-filled)
        Returns:   (B*N, 1, 256)
        """
        mask_f = mask.float() \
            .masked_fill(mask == 0, float('-inf')) \
            .masked_fill(mask == 1, 0.0)
        return self.encoder_context(past_traj, mask_f)   # (B*N, 1, 256)

    # ------------------------------------------------------------------
    # Stage 1: noise prediction for a single ego trajectory
    # ------------------------------------------------------------------

    def forward(self, y_noisy, beta, past_traj, mask):
        """
        Stage 1 noise prediction.

        y_noisy:   (B, T_F, 2)    — 노이즈가 추가된 ego 미래 경로
        beta:      (B,)            — 해당 timestep의 beta 값
        past_traj: (B*N, T_H, 6)  — 전체 노드의 과거 경로
        mask:      (N, N)
        Returns:   (B, T_F, 2)    — 예측된 noise ε̂
        """
        B = y_noisy.size(0)

        # Social context → ego 노드만 선택
        ctx_all = self.encode_context(past_traj, mask)  # (B*N, 1, 256)
        ctx_ego = ctx_all[::self.num_node]              # (B,   1, 256)

        beta     = beta.view(B, 1, 1)
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb  = torch.cat([time_emb, ctx_ego], dim=-1)                       # (B, 1, 259)

        x = self.concat1(ctx_emb, y_noisy)   # (B, T_F, 512)
        x = self.pos_emb(x)
        trans = self.transformer_encoder(x)
        trans = self.concat3(ctx_emb, trans)  # (B, T_F, 256)
        trans = self.concat4(ctx_emb, trans)  # (B, T_F, 128)
        return self.linear(ctx_emb, trans)    # (B, T_F, 2)

    # ------------------------------------------------------------------
    # Stage 2: K-sample parallel denoising
    # ------------------------------------------------------------------

    def generate_accelerate(self, x, beta, context_emb):
        """
        Stage 2 denoising step (K 샘플 병렬).

        x:           (B*N, K, T_F, 2)   — 현재 noisy 궤적
        beta:        (B*N,)              — 현재 timestep의 beta 값
        context_emb: (B*N, 1, 256)      — encode_context() 결과 (루프 밖에서 1회 계산)
        Returns:     (B*N, K, T_F, 2)   — 예측된 noise ε̂
        """
        B_N      = x.size(0)
        n_samples= x.size(1)   # K
        t_f      = x.size(2)   # T_F

        beta     = beta.view(B_N, 1, 1)
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B*N, 1, 3)
        time_emb = time_emb.to(context_emb.dtype)
        
        ctx_emb  = torch.cat([time_emb, context_emb], dim=-1)                   # (B*N, 1, 259)
        ctx_emb  = ctx_emb.repeat(1, n_samples, 1).unsqueeze(2)                 # (B*N, K, 1, 259)

        x = self.concat1.batch_generate(ctx_emb, x)            # (B*N, K, T_F, 512)
        x_flat   = x.view(B_N * n_samples, t_f, 512)
        emb_flat = self.pos_emb(x_flat)
        trans_flat = self.transformer_encoder(emb_flat)         # (B*N*K, T_F, 512)
        trans = trans_flat.view(B_N, n_samples, t_f, 512)

        trans = self.concat3.batch_generate(ctx_emb, trans)     # (B*N, K, T_F, 256)
        trans = self.concat4.batch_generate(ctx_emb, trans)     # (B*N, K, T_F, 128)
        return self.linear.batch_generate(ctx_emb, trans)       # (B*N, K, T_F, 2)


# ==============================================================================
# Stage 1: Training loss  L_NE
# ==============================================================================

def noise_estimation_loss(model, y_0, past_traj, mask, diffusion):
    """
    y_0:       (B, T_F, 2)   — 깨끗한 ego 미래 경로
    past_traj: (B*N, T_H, 6)
    mask:      (N, N)
    """
    B  = y_0.size(0)
    t  = torch.randint(0, diffusion['n_steps'], (B,), device=y_0.device)

    y_noisy, noise = q_sample(y_0, t, diffusion)   # 둘 다 (B, T_F, 2)
    beta_t   = diffusion['betas'][t]               # (B,)
    eps_pred = model(y_noisy, beta_t, past_traj, mask)  # (B, T_F, 2)

    return F.mse_loss(eps_pred, noise)


# ==============================================================================
# Stage 2: Accelerated reverse diffusion
# ==============================================================================

def p_sample_accelerate(model, diffusion, cur_y, context_emb, t):
    """
    역방향 diffusion 1-step.

    cur_y:       (B*N, K, T_F, 2)
    context_emb: (B*N, 1, 256)    — 루프 밖에서 미리 계산된 context
    t:           int               — 현재 timestep
    Returns:     (B*N, K, T_F, 2)
    """
    B_N    = cur_y.size(0)
    device = cur_y.device

    alpha_t              = diffusion['alphas'][t]
    one_minus_ab_sqrt_t  = diffusion['one_minus_alphas_bar_sqrt'][t]
    beta_t_scalar        = diffusion['betas'][t]

    eps_factor = (1 - alpha_t) / one_minus_ab_sqrt_t             # scalar

    beta_t = beta_t_scalar.expand(B_N)                           # (B*N,)
    eps_theta = model.generate_accelerate(cur_y, beta_t, context_emb)

    mean    = (1 / alpha_t.sqrt()) * (cur_y - eps_factor * eps_theta)
    sigma_t = beta_t_scalar.sqrt()
    return mean + sigma_t * torch.randn_like(cur_y) * 0.00001


def p_sample_loop_accelerate(model, diffusion, past_traj, mask, loc, num_tau=5):
    """
    전체 τ-step 역방향 diffusion.

    past_traj: (B*N, T_H, 6)
    mask:      (N, N)
    loc:       (B*N, K, T_F, 2)  — Initializer 출력 (prior)
    Returns:   (B*N, K, T_F, 2)  — denoised 궤적

    """
    context_emb = model.encode_context(past_traj, mask)  # (B*N, 1, 256)

    cur_y = loc.clone()
    for i in reversed(range(num_tau)):
        cur_y = p_sample_accelerate(model, diffusion, cur_y, context_emb, i)
    return cur_y
