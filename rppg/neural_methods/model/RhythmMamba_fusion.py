"""
RhythmMamba_fusion — Quality-Weighted Spatial Fusion + Pure Temporal Mamba

核心设计原则 (GPT-5.6 Sol 建议):
  "不要重排 token。改成每帧多区域 → 可靠性/质量加权融合成 1 个 per-frame token
   → 原始纯时间 RhythmMamba backbone"

与 quality_sort 的关键区别:
  quality_sort:  region tokens → soft sort → Mamba over (K*T) → pool → temporal Mamba
                 ↑ 排序可能破坏时间连续性
  fusion (NEW):  region tokens → per-frame quality fusion → 1 token/frame → pure temporal Mamba
                 ↑ 动态性在空间融合权重, 不是扫描顺序

优势:
  1. 时间连续性完整保持 (不重排)
  2. 计算更轻 (T vs K*T tokens in Mamba)
  3. 直接兼容 RhythmMamba 原始 backbone (Block_mamba)
  4. Quality-guided spatial selection before temporal modeling
"""
import torch
from torch import nn
import torch.nn.functional as F
import torch.fft
from functools import partial
from timm.models.layers import trunc_normal_, lecun_normal_
from timm.models.layers import DropPath
import math
from einops import rearrange
from mamba_ssm.modules.mamba_simple import Mamba


# ==============================================================================
# Re-used from RhythmMamba original
# ==============================================================================

class Fusion_Stem(nn.Module):
    def __init__(self, apha=0.5, belta=0.5, dim=24):
        super().__init__()
        self.stem11 = nn.Sequential(
            nn.Conv2d(3, dim // 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim // 2, eps=1e-05, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False))
        self.stem12 = nn.Sequential(
            nn.Conv2d(12, dim // 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim // 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False))
        self.stem21 = nn.Sequential(
            nn.Conv2d(dim // 2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False))
        self.stem22 = nn.Sequential(
            nn.Conv2d(dim // 2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False))
        self.apha, self.belta = apha, belta

    def forward(self, x):
        N, D, C, H, W = x.shape
        x1 = torch.cat([x[:, :1], x[:, :1], x[:, :D - 2]], 1)
        x2 = torch.cat([x[:, :1], x[:, :D - 1]], 1)
        x3 = x
        x4 = torch.cat([x[:, 1:], x[:, D - 1:]], 1)
        x5 = torch.cat([x[:, 2:], x[:, D - 1:], x[:, D - 1:]], 1)
        x_diff = self.stem12(torch.cat([x2 - x1, x3 - x2, x4 - x3, x5 - x4], 2).view(N * D, 12, H, W))
        x3 = x3.contiguous().view(N * D, C, H, W)
        x = self.stem11(x3)
        x_path1 = self.apha * x + self.belta * x_diff
        x = self.apha * self.stem21(x_path1) + self.belta * self.stem22(x_diff)
        return x


class Attention_mask(nn.Module):
    def forward(self, x):
        xsum = x.sum(dim=3, keepdim=True).sum(dim=4, keepdim=True)
        return x / xsum * x.shape[3] * x.shape[4] * 0.5


class Frequencydomain_FFN(nn.Module):
    def __init__(self, dim, mlp_ratio):
        super().__init__()
        self.scale = 0.02
        inner = dim * mlp_ratio
        self.r = nn.Parameter(self.scale * torch.randn(inner, inner))
        self.i = nn.Parameter(self.scale * torch.randn(inner, inner))
        self.rb = nn.Parameter(self.scale * torch.randn(inner))
        self.ib = nn.Parameter(self.scale * torch.randn(inner))
        self.fc1 = nn.Sequential(
            nn.Conv1d(dim, inner, 1, bias=False),
            nn.BatchNorm1d(inner), nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Conv1d(inner, dim, 1, bias=False),
            nn.BatchNorm1d(dim))

    def forward(self, x):
        B, N, C = x.shape
        x = self.fc1(x.transpose(1, 2)).transpose(1, 2)
        x_fre = torch.fft.fft(x, dim=1, norm='ortho')
        x_real = F.relu(torch.einsum('bnc,cc->bnc', x_fre.real, self.r) -
                        torch.einsum('bnc,cc->bnc', x_fre.imag, self.i) + self.rb)
        x_imag = F.relu(torch.einsum('bnc,cc->bnc', x_fre.imag, self.r) +
                        torch.einsum('bnc,cc->bnc', x_fre.real, self.i) + self.ib)
        x_fre = torch.stack([x_real, x_imag], dim=-1).float()
        x_fre = torch.view_as_complex(x_fre)
        x = torch.fft.ifft(x_fre, dim=1, norm="ortho").to(torch.float32)
        return self.fc2(x.transpose(1, 2)).transpose(1, 2)


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=48, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        return self.mamba(self.norm(x))


class Block_mamba(nn.Module):
    """Original RhythmMamba temporal block with multi-temporal parallelization."""
    def __init__(self, dim, mlp_ratio, drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = MambaLayer(dim)
        self.mlp = Frequencydomain_FFN(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None: m.bias.data.zero_()

    def forward(self, x):
        B, D, C = x.size()
        path, segment = 3, 4
        tt = D // segment
        x_r = x.repeat(segment, 1, 1)
        x_o = x_r.clone()
        for i in range(1, segment):
            x_o[i * B:(i + 1) * B, :D - i * tt, :] = x_r[i * B:(i + 1) * B, i * tt:, :]
        x_o = self.attn(x_o)
        for i in range(1, segment):
            for j in range(i):
                x_o[0:B, tt * i: tt * (i + 1), :] += x_o[B * (j + 1):B * (j + 2), tt * (i - j - 1): tt * (i - j), :]
            x_o[0:B, tt * i: tt * (i + 1), :] /= (i + 1)
        x = x + self.drop_path(self.norm1(x_o[0:B]))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ==============================================================================
# NEW: Region Quality Estimator (same as G1 in dynamic_scan)
# ==============================================================================

class RegionQualityEstimator(nn.Module):
    """Time-varying per-region quality from region tokens.

    Input:  (B, T, K, C) region tokens
    Output: (B, T, K) quality scores in [0, 1]

    Uses temporal convolution to capture local instability,
    followed by a lightweight MLP head.
    """
    def __init__(self, dim, temporal_window=5):
        super().__init__()
        self.temporal_window = temporal_window
        self.quality_conv = nn.Sequential(
            nn.Conv1d(dim, dim // 2, kernel_size=temporal_window,
                      padding=temporal_window // 2),
            nn.BatchNorm1d(dim // 2),
            nn.ReLU(inplace=True),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(dim // 2, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """x: (B, T, K, C) -> q: (B, T, K)"""
        B, T, K, C = x.shape
        x_r = rearrange(x, 'b t k c -> (b k) c t')
        feat = self.quality_conv(x_r)
        feat = rearrange(feat, '(b k) c t -> b t k c', b=B, k=K)
        q = self.quality_head(feat).squeeze(-1)
        return q


# ==============================================================================
# NEW: Quality-Weighted Per-Frame Fusion (the core innovation)
# ==============================================================================

class QualityWeightedFusion(nn.Module):
    """Per-frame quality-weighted spatial fusion.

    For each time step t:
      token(t) = sum_k softmax(q(t,k)) * region_token(t,k)

    This preserves temporal continuity: there is no reordering,
    each frame t maps to exactly one fused token.

    Input:  (B, T, K, C) region tokens
    Output: (B, T, C) fused per-frame tokens
    """
    def __init__(self, dim, num_regions, temperature=1.0):
        super().__init__()
        self.K = num_regions
        self.dim = dim
        # Learnable temperature for quality sharpness
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))
        # Small learnable bias per region (initial prior)
        self.region_bias = nn.Parameter(torch.zeros(num_regions))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, quality):
        """x: (B, T, K, C), quality: (B, T, K) -> out: (B, T, C)"""
        B, T, K, C = x.shape
        temp = torch.exp(self.log_temperature).clamp(min=0.5, max=10.0)

        # Compute fusion weights: softmax over regions per time step
        # Higher quality → higher weight
        logits = quality / temp + self.region_bias.view(1, 1, K)
        weights = torch.softmax(logits, dim=-1)  # (B, T, K)

        # Normalize region tokens
        x_normed = self.norm(x)

        # Weighted sum: (B,T,K,C) * (B,T,K,1) -> (B,T,C)
        out = (x_normed * weights.unsqueeze(-1)).sum(dim=2)
        return out, weights


# ============================================================================
# Optional: Quality statistics for analysis/logging
# ============================================================================

def compute_quality_stats(q):
    """q: (B,T,K) -> stats dict."""
    with torch.no_grad():
        K = q.shape[-1]
        q_norm = q - q.max(dim=-1, keepdim=True)[0]
        prob = torch.softmax(q_norm, dim=-1)
        entropy = -(prob * torch.log(prob + 1e-8)).sum(-1) / math.log(K)
        top2 = torch.topk(q, k=2, dim=-1)[0]
        gap = top2[..., 0] - top2[..., 1]
        return {
            'entropy_mean': entropy.mean().item(),
            'gap_mean': gap.mean().item(),
            'q_max_mean': q.max(-1)[0].mean().item(),
            'q_min_mean': q.min(-1)[0].mean().item(),
        }


# ============================================================================
# Helper init
# ============================================================================

def _init_weights(module, n_layer, initializer_range=0.02,
                  rescale_prenorm_residual=True, n_residuals_per_layer=1):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None: nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        lecun_normal_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias); nn.init.ones_(m.weight)


# ============================================================================
# Main: RhythmMamba_fusion
# ============================================================================

class RhythmMamba_fusion(nn.Module):
    """RhythmMamba with Quality-Weighted Per-Frame Spatial Fusion.

    Architecture:
      Video [B,D,3,H,W]
        → Fusion_Stem + stem3 + attention_mask  [B,C,T',H',W']
        → AdaptivePool2d → (grid×grid) regions  [B,T',K,C]
        → RegionQualityEstimator → q(B,T',K)
        → QualityWeightedFusion → (B,T',C)  per-frame tokens
        → Block_mamba × 24  (pure temporal, RhythmMamba original)
        → Upsample + Conv1d → rPPG [B,T]

    Key differences from quality_sort (dynamic_scan):
      - No region reordering / soft permutation
      - Mamba sees T tokens, not K*T tokens
      - Spatial selection happens in fusion weights, not scan order

    Key differences from RhythmMamba original:
      - No spatial GAP (replaced by quality-weighted fusion)
      - Quality weights are time-varying and learned
    """
    def __init__(self, depth=24, embed_dim=96, mlp_ratio=2, grid_size=3,
                 drop_rate=0., drop_path_rate=0.1,
                 initializer_cfg=None, device=None, dtype=None, **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs)
        super().__init__()

        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.num_regions = grid_size * grid_size

        # ---- RhythmMamba stem ----
        self.Fusion_Stem = Fusion_Stem(dim=embed_dim // 4)
        self.attn_mask = Attention_mask()
        self.stem3 = nn.Sequential(
            nn.Conv3d(embed_dim // 4, embed_dim,
                      kernel_size=(2, 5, 5), stride=(2, 1, 1),
                      padding=(0, 2, 2)),
            nn.BatchNorm3d(embed_dim),
        )

        # ---- NEW: Quality-guided fusion ----
        self.quality_estimator = RegionQualityEstimator(embed_dim)
        self.fusion = QualityWeightedFusion(embed_dim, self.num_regions)

        # ---- Original RhythmMamba temporal backbone ----
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr
        self.blocks = nn.ModuleList([
            Block_mamba(dim=embed_dim, mlp_ratio=mlp_ratio,
                       drop_path=inter_dpr[i])
            for i in range(depth)
        ])

        # ---- Output ----
        self.upsample = nn.Upsample(scale_factor=2)
        self.ConvBlockLast = nn.Conv1d(embed_dim, 1, kernel_size=1, stride=1, padding=0)

        # ---- Init ----
        self.apply(segm_init_weights)
        self.apply(partial(_init_weights, n_layer=depth,
                          **(initializer_cfg if initializer_cfg is not None else {})))

    def forward(self, x, return_quality=False):
        B, D, C, H, W = x.shape

        # ---- Stem (same as RhythmMamba) ----
        x = self.Fusion_Stem(x)
        x = x.view(B, D, self.embed_dim // 4, H // 8, W // 8).permute(0, 2, 1, 3, 4)
        x = self.stem3(x)  # [B, C, T', H', W']

        # Attention mask
        mask = torch.sigmoid(x)
        mask = self.attn_mask(mask)
        x = x * mask  # [B, C, T', H', W']

        # ---- KEY MODIFICATION: quality-weighted fusion instead of GAP ----
        B2, C2, T2, H2, W2 = x.shape

        # Spatial → region tokens via adaptive pooling
        x_reshape = rearrange(x, 'b c t h w -> (b t) c h w')
        x_regions = F.adaptive_avg_pool2d(x_reshape, (self.grid_size, self.grid_size))
        x_regions = rearrange(x_regions, '(b t) c h w -> b t (h w) c',
                             b=B2, t=T2)  # [B, T, K, C]

        # Estimate quality
        quality = self.quality_estimator(x_regions)  # [B, T, K]

        # Fuse: per-frame quality-weighted sum
        x, fusion_weights = self.fusion(x_regions, quality)  # [B, T, C]

        # ---- Original temporal Mamba backbone ----
        for blk in self.blocks:
            x = blk(x)  # [B, T, C]

        # ---- Output head ----
        rPPG = x.permute(0, 2, 1)  # [B, C, T]
        rPPG = self.upsample(rPPG)
        rPPG = self.ConvBlockLast(rPPG).squeeze(1)  # [B, T]

        if return_quality:
            return rPPG, quality, fusion_weights
        return rPPG
