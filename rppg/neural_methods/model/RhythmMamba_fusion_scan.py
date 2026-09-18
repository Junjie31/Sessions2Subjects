"""
RhythmMamba_fusion_scan — Multi-Mode + Multi-Quality SSM Scan for rPPG.

Mamba modulation modes (V1-V13): scalar, strong, mlp, b_gate, delta_b, exp,
  sigmoid, per_dim, cd_modulate, d_gate, power, additive, layer_init

Quality signal modes (new in Phase 3):
  "fusion"       : weighted avg of spatial region quality (default)
  "freq"         : spectral purity of fused tokens via FFT
  "temporal_grad": fusion quality × (1 - |∇q|)
  "learned"      : Conv1d on fused tokens → learnable quality
  "hybrid"       : 0.5*fusion + 0.5*freq
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

from mamba_ssm_scan.modules.mamba_simple_scan import QualityScanMamba
from neural_methods.model.RhythmMamba_fusion import (
    Fusion_Stem, Attention_mask, Frequencydomain_FFN,
    RegionQualityEstimator, QualityWeightedFusion,
    segm_init_weights, _init_weights,
)


# ==============================================================================
# Alternative Quality Estimators
# ==============================================================================

class FreqQualityHead(nn.Module):
    """Per-frame quality from spectral purity of fused tokens."""
    def __init__(self, dim):
        super().__init__()
        # Input: C (fused) + C (fft mag) = 2*C channels
        self.conv = nn.Sequential(
            nn.Conv1d(dim * 2, dim // 2, 5, padding=2), nn.BatchNorm1d(dim // 2),
            nn.ReLU(), nn.Conv1d(dim // 2, 1, 1), nn.Sigmoid())

    def forward(self, x):
        """x: (B, T, C) → q: (B, T)"""
        x_t = x.transpose(1, 2)  # (B, C, T)
        x_f = torch.fft.rfft(x_t, dim=-1).abs()
        x_f = F.interpolate(x_f, size=x_t.shape[-1], mode='linear')
        feat = torch.cat([x_t, x_f], dim=1)  # (B, 2*C, T)
        return self.conv(feat).squeeze(1)


class LearnedQualityHead(nn.Module):
    """Direct quality prediction from fused tokens — no spatial prior."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim//4, 7, padding=3), nn.BatchNorm1d(dim//4),
            nn.ReLU(), nn.Conv1d(dim//4, dim//8, 5, padding=2),
            nn.BatchNorm1d(dim//8), nn.ReLU(),
            nn.Conv1d(dim//8, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x.transpose(1, 2)).squeeze(1)


# ==============================================================================
# Scan Mamba Layers
# ==============================================================================

class ScanMambaLayer(nn.Module):
    def __init__(self, dim, d_state=48, d_conv=4, expand=2,
                 modulation_mode="scalar", quality_scale_init=0.1,
                 osc_learn_omega=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = QualityScanMamba(
            d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand,
            modulation_mode=modulation_mode, quality_scale_init=quality_scale_init,
            osc_learn_omega=osc_learn_omega)

    def forward(self, x, quality=None):
        return self.mamba(self.norm(x), quality=quality)


class ScanBlockMamba(nn.Module):
    def __init__(self, dim, mlp_ratio, drop_path=0.,
                 modulation_mode="scalar", quality_scale_init=0.1,
                 osc_learn_omega=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = ScanMambaLayer(
            dim, modulation_mode=modulation_mode, quality_scale_init=quality_scale_init,
            osc_learn_omega=osc_learn_omega)
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

    def forward(self, x, quality=None):
        B, D, C = x.size()
        segment = 4; tt = D // segment
        x_r = x.repeat(segment, 1, 1); x_o = x_r.clone()
        for i in range(1, segment):
            x_o[i * B:(i + 1) * B, :D - i * tt, :] = x_r[i * B:(i + 1) * B, i * tt:, :]
        q_for_attn = quality.repeat(segment, 1) if quality is not None else None
        x_o = self.attn(x_o, quality=q_for_attn)
        for i in range(1, segment):
            for j in range(i):
                x_o[0:B, tt * i: tt * (i + 1), :] += x_o[B * (j + 1):B * (j + 2), tt * (i - j - 1): tt * (i - j), :]
            x_o[0:B, tt * i: tt * (i + 1), :] /= (i + 1)
        x = x + self.drop_path(self.norm1(x_o[0:B]))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ==============================================================================
# MAIN MODEL
# ==============================================================================

class RhythmMamba_fusion_scan(nn.Module):
    """Multi-mode scan model with alternative quality signals."""

    def __init__(self, depth=24, embed_dim=96, mlp_ratio=2, grid_size=3,
                 drop_rate=0., drop_path_rate=0.1,
                 modulation_mode="scalar", quality_scale_init=0.1,
                 quality_mode="fusion",
                 osc_learn_omega=True,  # for osc/osc_real modes
                 initializer_cfg=None, device=None, dtype=None, **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs); super().__init__()
        self.embed_dim = embed_dim; self.grid_size = grid_size
        self.num_regions = grid_size * grid_size
        self.modulation_mode = modulation_mode
        self.quality_mode = quality_mode
        # Map pfold/osc_pfold/phase/cycle variants to backbone + fold config
        _fold_cfg = {
            "pfold": ("none", True, True),
            "pfold_phase": ("none", True, False),
            "pfold_cycle": ("none", False, True),
            "osc_pfold": ("osc_real", True, True),
        }
        if modulation_mode in _fold_cfg:
            backbone_mode, self._pf_phase, self._pf_cycle = _fold_cfg[modulation_mode]
        else:
            backbone_mode = modulation_mode
            self._pf_phase = self._pf_cycle = False
        self.backbone_mode = backbone_mode

        # Stem
        self.Fusion_Stem = Fusion_Stem(dim=embed_dim // 4)
        self.attn_mask = Attention_mask()
        self.stem3 = nn.Sequential(
            nn.Conv3d(embed_dim // 4, embed_dim,
                      kernel_size=(2, 5, 5), stride=(2, 1, 1), padding=(0, 2, 2)),
            nn.BatchNorm3d(embed_dim))

        # Spatial quality (always needed for fusion weights)
        self.quality_estimator = RegionQualityEstimator(embed_dim)
        self.fusion = QualityWeightedFusion(embed_dim, self.num_regions)

        # Alternative quality heads
        if quality_mode == "freq":
            self.freq_quality = FreqQualityHead(embed_dim)
        elif quality_mode == "learned":
            self.learned_quality = LearnedQualityHead(embed_dim)
        elif quality_mode == "hybrid":
            self.freq_quality = FreqQualityHead(embed_dim)
        # temporal_grad doesn't need extra params

        # Temporal Mamba blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr
        # layer_init: decreasing α from 0.3 → 0.01 across depth
        self.blocks = nn.ModuleList([
            ScanBlockMamba(
                dim=embed_dim, mlp_ratio=mlp_ratio, drop_path=inter_dpr[i],
                modulation_mode=backbone_mode,
                quality_scale_init=(0.3 - 0.29 * i / max(depth-1,1)) if backbone_mode == "layer_init" else quality_scale_init,
                osc_learn_omega=osc_learn_omega,
            )
            for i in range(depth)
        ])

        # Period-fold scan branch (geometry prior, orthogonal to oscillatory-A)
        # pfold = period-fold only; osc_pfold = oscillatory-A backbone + fold scan
        self.pfold = None
        if modulation_mode in ("pfold", "pfold_phase", "pfold_cycle", "osc_pfold"):
            from neural_methods.model.period_fold_scan import PeriodFoldScan
            self.pfold = PeriodFoldScan(dim=embed_dim, P0=32,
                                        phase_scan=self._pf_phase, cycle_scan=self._pf_cycle, shared=True)

        # Output
        self.upsample = nn.Upsample(scale_factor=2)
        self.ConvBlockLast = nn.Conv1d(embed_dim, 1, kernel_size=1, stride=1, padding=0)
        self.apply(segm_init_weights)
        self.apply(partial(_init_weights, n_layer=depth,
                          **(initializer_cfg if initializer_cfg is not None else {})))

    def _compute_frame_quality(self, spatial_q, fusion_weights, fused_tokens):
        """Compute per-frame quality based on quality_mode."""
        q_spatial = (fusion_weights * spatial_q).sum(dim=-1)  # (B, T)

        if self.quality_mode == "fusion":
            return q_spatial

        elif self.quality_mode == "freq":
            q_freq = self.freq_quality(fused_tokens)  # (B, T)
            return q_freq

        elif self.quality_mode == "temporal_grad":
            # Penalize frames where spatial quality changes rapidly
            grad = torch.abs(q_spatial[:, 1:] - q_spatial[:, :-1])
            grad = F.pad(grad, (1, 0), value=float(grad.mean()))
            instability = grad / (grad.max(dim=1, keepdim=True)[0] + 1e-8)
            return q_spatial * (1.0 - 0.5 * instability)

        elif self.quality_mode == "learned":
            return self.learned_quality(fused_tokens)

        elif self.quality_mode == "hybrid":
            q_freq = self.freq_quality(fused_tokens)
            return 0.5 * q_spatial + 0.5 * q_freq

        return q_spatial

    def forward(self, x, return_debug=False, external_quality=None):
        """external_quality: (B,T) pre-computed per-frame quality. Used for oracle."""
        B, D, C, H, W = x.shape
        x = self.Fusion_Stem(x)
        x = x.view(B, D, self.embed_dim // 4, H // 8, W // 8).permute(0, 2, 1, 3, 4)
        x = self.stem3(x)
        mask = torch.sigmoid(x); mask = self.attn_mask(mask); x = x * mask

        B2, C2, T2, H2, W2 = x.shape
        xr = F.adaptive_avg_pool2d(rearrange(x, 'b c t h w -> (b t) c h w'), (self.grid_size, self.grid_size))
        x_regions = rearrange(xr, '(b t) c h w -> b t (h w) c', b=B2, t=T2)

        quality = self.quality_estimator(x_regions)
        x_fused, fusion_weights = self.fusion(x_regions, quality)
        frame_quality = (external_quality if external_quality is not None
                        else self._compute_frame_quality(quality, fusion_weights, x_fused))

        # Period-fold scan branch (geometry prior)
        if self.pfold is not None:
            x_fused = self.pfold(x_fused)

        for blk in self.blocks:
            x_fused = blk(x_fused, quality=frame_quality)

        rPPG = x_fused.permute(0, 2, 1)
        rPPG = self.upsample(rPPG)
        rPPG = self.ConvBlockLast(rPPG).squeeze(1)

        if return_debug:
            return rPPG, quality, fusion_weights, frame_quality
        return rPPG
