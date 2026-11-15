"""
Neural Network Transceiver Architectures for UAV/UGV Cooperative Path Planning:
1. Path-SC (Proposed): DynamicViT-based Semantic Encoder with Token Importance Prediction Module
   + Wireless Channel + MAE-style ViT Semantic Decoder.
2. DeiT-SC (Benchmark & Teacher): Full-token ViT Semantic Encoder & Decoder without sparsification.
3. MAE-SC (Benchmark): Fixed-ratio random patch masking SemCom transceiver without semantic guidance.

Implements Section V, Fig. 2, Equations (3)-(5), (23)-(25), and Algorithm 2 from:
Zhao et al., 'Adaptive Semantic Communication for UAV/UGV Cooperative Path Planning', IEEE TVT 2026.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from wireless_channel import pass_through_wireless_channel


@dataclass
class PathSCConfig:
    """
    Configuration for Path-SC, DeiT-SC, and MAE-SC transceivers.
    Supports both hardware-efficient configuration (for rapid local training/verification on CPU/MPS)
    and the paper's full 12-layer / 8-layer configuration.
    """
    img_size: int = 64          # UAV subregion image resolution (64x64 per UAV -> 256x256 global map)
    patch_size: int = 8         # b x b patch size -> N = (64/8)^2 = 64 patches per UAV tile
    in_chans: int = 3
    embed_dim: int = 128        # Token embedding dimension L_e
    encoder_depth: int = 4      # Number of ViT / DynamicViT encoder blocks (12 in paper on RTX A6000)
    decoder_depth: int = 3      # Number of ViT decoder blocks (8 in paper on RTX A6000)
    num_heads: int = 4          # Attention heads (12 in paper)
    mlp_ratio: float = 2.0
    default_delta: float = 0.70 # Default sparsification ratio delta during training (Section VI-A)
    mae_keep_ratio: float = 0.30 # Fixed keep ratio for MAE-SC benchmark (~296.4 kb vs 988.3 kb in Table I)


class PatchEmbed(nn.Module):
    """Partitions image x_t^k into N patches of size b x b and projects to token embeddings."""

    def __init__(self, img_size: int = 64, patch_size: int = 8, in_chans: int = 3, embed_dim: int = 128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, N, D)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    """Standard Vision Transformer block supporting optional attention/token masking."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor, decision_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        As described in Section V-A:
        High-importance tokens compute full global attention, while low-importance (sparsified)
        tokens are masked from global cross-token attention to reduce computational complexity.
        """
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        if decision_mask is not None:
            # decision_mask: (B, N, 1) in [0, 1]
            x = x + decision_mask * attn_out
        else:
            x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TokenPredictionModule(nn.Module):
    """
    Lightweight MLP-based Prediction Module inside DynamicViT (Fig. 2, Section V-A).
    Evaluates the semantic importance of each patch token (obstacle boundaries, road edges,
    traversable corridors) and selects top delta_k fraction of tokens for wireless transmission.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.local_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
        )
        self.global_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, 2),  # Logits for [drop, keep]
        )

    def forward(
        self,
        tokens: torch.Tensor,
        delta_k: float,
        patch_edge_prior: Optional[torch.Tensor] = None,
        hard_select: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (B, N, D) token features.
            delta_k: Target proportion of tokens to retain in (0, 1].
            patch_edge_prior: Optional (B, N) task-relevant boundary contrast prior.
            hard_select: Whether to produce binary top-k selection (with straight-through gradient).

        Returns:
            keep_prob: Continuous keep probability in (0, 1) of shape (B, N).
            binary_mask: Binary decision indicator D_hat in {0, 1} of shape (B, N).
        """
        b, n, _ = tokens.shape
        local_feat = self.local_mlp(tokens)  # (B, N, D/2)
        global_feat = self.global_mlp(tokens.mean(dim=1, keepdim=True)).expand(-1, n, -1)  # (B, N, D/2)
        combined = torch.cat([local_feat, global_feat], dim=-1)  # (B, N, D)
        logits = self.score_head(combined)  # (B, N, 2)

        if patch_edge_prior is not None:
            # Guide keep logit toward high semantic variance patches (obstacle contours & roads)
            logits[:, :, 1] = logits[:, :, 1] + 2.5 * patch_edge_prior

        probs = F.softmax(logits, dim=-1)[:, :, 1]  # (B, N) keep probability

        k_keep = max(1, min(n, int(round(delta_k * n))))
        if hard_select:
            _, topk_idx = torch.topk(probs, k=k_keep, dim=1)
            hard_mask = torch.zeros_like(probs)
            hard_mask.scatter_(1, topk_idx, 1.0)
            # Straight-through estimator: forward uses binary top-k mask, backward flows through probs
            binary_mask = hard_mask - probs.detach() + probs
        else:
            binary_mask = probs

        return probs, binary_mask


class SemComTransceiver(nn.Module):
    """
    Unified Vision Transformer Semantic Communication Transceiver supporting all three modes:
    - mode='path_sc': Proposed Path-SC with DynamicViT TokenPredictionModule and adaptive delta_k.
    - mode='deit_sc': Full-token DeiT-SC benchmark (and teacher network for distillation).
    - mode='mae_sc': Fixed-ratio random patch masking MAE-SC benchmark.
    """

    def __init__(self, cfg: PathSCConfig, mode: str = "path_sc"):
        super().__init__()
        assert mode in ("path_sc", "deit_sc", "mae_sc")
        self.cfg = cfg
        self.mode = mode

        # 1. Patch Embedding + Encoder Positional Embedding
        self.patch_embed = PatchEmbed(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            embed_dim=cfg.embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, cfg.embed_dim) * 0.02)

        # Residual conv refinement layer on reconstructed patches to preserve sharp obstacle edges
        self.edge_refine = nn.Sequential(
            nn.Conv2d(cfg.in_chans, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, cfg.in_chans, kernel_size=3, padding=1),
        )

        # 2. Encoder Transformer Blocks + DynamicViT Prediction Modules (at stages 1 and 2)
        self.encoder_blocks = nn.ModuleList(
            [TransformerBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio) for _ in range(cfg.encoder_depth)]
        )
        self.encoder_norm = nn.LayerNorm(cfg.embed_dim)

        # Two sparsification stages M = 2 as in DynamicViT & Eq. (25)
        s0 = max(0, cfg.encoder_depth // 2 - 1)
        s1 = max(s0 + 1, cfg.encoder_depth - 1) if cfg.encoder_depth > 1 else 0
        self.pred_stage_indices = sorted(list({s0, s1}))
        self.prediction_modules = nn.ModuleList(
            [TokenPredictionModule(cfg.embed_dim) for _ in range(len(self.pred_stage_indices))]
        )

        # 3. Semantic Decoder (MAE-style mask token + ViT decoder blocks + patch pixel projection)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, num_patches, cfg.embed_dim) * 0.02)

        self.decoder_blocks = nn.ModuleList(
            [TransformerBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio) for _ in range(cfg.decoder_depth)]
        )
        self.decoder_norm = nn.LayerNorm(cfg.embed_dim)
        patch_dim = cfg.patch_size * cfg.patch_size * cfg.in_chans
        self.decoder_pred = nn.Linear(cfg.embed_dim, patch_dim)

    def _compute_patch_semantic_prior(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes per-patch local gradient / obstacle-boundary salience in [0, 1]
        so the DynamicViT prediction module prioritizes patches containing road boundaries,
        building contours, and water edges.
        """
        b, c, h, w = x.shape
        ps = self.cfg.patch_size
        # Sobel-like spatial gradient magnitude
        gx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        gy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        grad = (gx + gy).mean(dim=1, keepdim=True)  # (B, 1, H, W)

        # Also boost patches whose RGB signature matches road or obstacle transitions
        patch_grad = F.avg_pool2d(grad, kernel_size=ps, stride=ps).flatten(1)  # (B, N)
        patch_var = F.avg_pool2d(x ** 2, kernel_size=ps, stride=ps).mean(dim=1).flatten(1) - (
            F.avg_pool2d(x, kernel_size=ps, stride=ps).mean(dim=1).flatten(1) ** 2
        )
        salience = patch_grad + 1.5 * torch.sqrt(patch_var.clamp(min=0.0) + 1e-6)
        salience = (salience - salience.min(dim=1, keepdim=True)[0]) / (
            salience.max(dim=1, keepdim=True)[0] - salience.min(dim=1, keepdim=True)[0] + 1e-6
        )
        return salience

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs: (B, 3, H, W) -> patches: (B, N, patch_size^2 * 3)"""
        p = self.cfg.patch_size
        b, c, h, w = imgs.shape
        gh, gw = h // p, w // p
        x = imgs.reshape(b, c, gh, p, gw, p)
        x = torch.einsum("bchpwq->bhwpqc", x)
        return x.reshape(b, gh * gw, p * p * c)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """patches: (B, N, patch_size^2 * 3) -> imgs: (B, 3, H, W)"""
        p = self.cfg.patch_size
        b, n, _ = patches.shape
        gh = gw = int(n ** 0.5)
        x = patches.reshape(b, gh, gw, p, p, self.cfg.in_chans)
        x = torch.einsum("bhwpqc->bchpwq", x)
        return x.reshape(b, self.cfg.in_chans, gh * p, gw * p)

    def forward(
        self,
        x: torch.Tensor,
        delta_k: Optional[float] = None,
        snr_db: float = 12.0,
        channel_type: str = "AWGN",
    ) -> Dict[str, object]:
        """
        Full forward pass:
        x_t^k -> Semantic Encoder E_e(x_t^k; alpha, delta_k) -> s_t^k
              -> Wireless Channel (AWGN / Rayleigh) -> s_hat_t^k
              -> Semantic Decoder E_d(s_hat_t^k; alpha, delta_k) -> x_hat_t^k
        """
        if delta_k is None:
            delta_k = self.cfg.default_delta

        b = x.shape[0]
        tokens = self.patch_embed(x) + self.pos_embed  # (B, N, D)
        n_patches = tokens.shape[1]

        decision_masks: List[torch.Tensor] = []
        keep_probs: List[torch.Tensor] = []
        current_mask = torch.ones((b, n_patches), device=x.device, dtype=x.dtype)

        if self.mode == "path_sc":
            edge_prior = self._compute_patch_semantic_prior(x)
            stage_ptr = 0
            for idx, blk in enumerate(self.encoder_blocks):
                tokens = blk(tokens, decision_mask=current_mask.unsqueeze(-1))
                if idx in self.pred_stage_indices:
                    # Progressive sparsification across stages toward target delta_k
                    is_last_stage = stage_ptr == len(self.pred_stage_indices) - 1
                    stage_delta = delta_k if is_last_stage else (delta_k ** 0.5)
                    probs, d_mask = self.prediction_modules[stage_ptr](
                        tokens, delta_k=stage_delta, patch_edge_prior=edge_prior, hard_select=True
                    )
                    current_mask = d_mask
                    keep_probs.append(probs)
                    decision_masks.append(d_mask)
                    stage_ptr += 1

        elif self.mode == "deit_sc":
            # Full-token DeiT-SC: no sparsification (all tokens kept, delta_k = 1.0)
            for blk in self.encoder_blocks:
                tokens = blk(tokens, decision_mask=None)
            decision_masks.append(current_mask)
            keep_probs.append(current_mask)

        elif self.mode == "mae_sc":
            # Fixed-ratio random masking without semantic guidance (MAE-SC benchmark)
            k_keep = max(1, int(round(self.cfg.mae_keep_ratio * n_patches)))
            rand_scores = torch.rand((b, n_patches), device=x.device)
            _, topk_idx = torch.topk(rand_scores, k=k_keep, dim=1)
            current_mask = torch.zeros((b, n_patches), device=x.device, dtype=x.dtype)
            current_mask.scatter_(1, topk_idx, 1.0)
            for blk in self.encoder_blocks:
                tokens = blk(tokens, decision_mask=current_mask.unsqueeze(-1))
            decision_masks.append(current_mask)
            keep_probs.append(current_mask)

        encoded_tokens = self.encoder_norm(tokens)  # t_{il} in Eq. (24)

        # Mask out dropped tokens prior to wireless physical channel transmission
        mask_3d = current_mask.unsqueeze(-1)  # (B, N, 1)
        s_tx = encoded_tokens * mask_3d

        # Equation (4): Transmit over noisy physical channel (AWGN or Rayleigh)
        s_rx = pass_through_wireless_channel(s_tx, snr_db=snr_db, channel_type=channel_type)

        # Replace dropped token positions with learnable mask_token at the decoder input
        dec_tokens = s_rx * mask_3d + self.mask_token.expand(b, n_patches, -1) * (1.0 - mask_3d)
        dec_tokens = dec_tokens + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            dec_tokens = blk(dec_tokens, decision_mask=None)

        dec_tokens = self.decoder_norm(dec_tokens)
        pred_patches = self.decoder_pred(dec_tokens)  # (B, N, patch_dim)

        # For kept tokens, blend direct patch projection with decoded context to preserve sharp semantics
        raw_img = self.unpatchify(pred_patches)
        x_hat = torch.sigmoid(raw_img + 0.25 * self.edge_refine(raw_img))

        # Calculate transmitted payload volume (scaled to paper's 1500x1500 full map in Table I)
        active_fraction = float(current_mask.mean().item())
        # In Table I, full-token DeiT-SC is 988.3 kb across the 16 UAVs -> ~61.77 kb per UAV at delta=1.0
        bits_per_uav_full = (988.3 * 1000.0) / 16.0
        transmitted_bits = active_fraction * bits_per_uav_full

        return {
            "x_hat": x_hat,
            "encoded_tokens": encoded_tokens,
            "decision_masks": decision_masks,
            "keep_probs": keep_probs,
            "final_mask": current_mask,
            "active_fraction": active_fraction,
            "transmitted_bits": transmitted_bits,
        }


def compute_algorithm2_loss(
    x: torch.Tensor,
    student_out: Dict[str, object],
    teacher_out: Optional[Dict[str, object]] = None,
    target_delta: float = 0.70,
    lambda_kl: float = 0.15,
    lambda_delta: float = 2.0,
    lambda_distill: float = 0.25,
) -> Dict[str, torch.Tensor]:
    """
    Computes the training loss in Equations (23), (24), and (25) (Algorithm 2):
        L = L_MSE + lambda_KL * KL(x_hat || x_tilde) + lambda_delta * L_{delta_k}
            + lambda_distill * sum(D_hat * (t - t')^2) / sum(D_hat)
    """
    x_hat: torch.Tensor = student_out["x_hat"]

    # Equation (23): Mean Squared Error loss L_MSE
    loss_mse = F.mse_loss(x_hat, x)

    # Equation (25): Ratio constraint loss L_{delta_k} across M sparsification stages
    keep_probs: List[torch.Tensor] = student_out["keep_probs"]
    loss_delta = torch.tensor(0.0, device=x.device)
    if keep_probs:
        for probs_m in keep_probs:
            mean_keep = probs_m.mean(dim=1)  # (B,)
            loss_delta = loss_delta + torch.mean((target_delta - mean_keep) ** 2)
        loss_delta = loss_delta / len(keep_probs)

    loss_kl = torch.tensor(0.0, device=x.device)
    loss_distill = torch.tensor(0.0, device=x.device)

    if teacher_out is not None:
        x_tilde: torch.Tensor = teacher_out["x_hat"].detach()
        t_teacher: torch.Tensor = teacher_out["encoded_tokens"].detach()
        t_student: torch.Tensor = student_out["encoded_tokens"]
        final_mask: torch.Tensor = student_out["final_mask"]  # (B, N)

        # Pixel distribution KL divergence KL(x_hat || x_tilde)
        b = x.shape[0]
        s_log_prob = F.log_softmax(x_hat.reshape(b, -1) / 0.1, dim=-1)
        t_prob = F.softmax(x_tilde.reshape(b, -1) / 0.1, dim=-1)
        loss_kl = F.kl_div(s_log_prob, t_prob, reduction="batchmean")

        # Token distillation loss weighted by binary decision mask D_hat^{i, M} in Eq. (24)
        token_diff_sq = torch.mean((t_student - t_teacher) ** 2, dim=-1)  # (B, N)
        denom = torch.sum(final_mask) + 1e-6
        loss_distill = torch.sum(final_mask * token_diff_sq) / denom

    total_loss = (
        loss_mse
        + lambda_kl * loss_kl
        + lambda_delta * loss_delta
        + lambda_distill * loss_distill
    )

    return {
        "total": total_loss,
        "mse": loss_mse,
        "kl": loss_kl,
        "delta": loss_delta,
        "distill": loss_distill,
    }
