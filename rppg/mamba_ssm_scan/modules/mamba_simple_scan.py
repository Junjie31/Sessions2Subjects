"""
Quality-Guided Mamba — Extended Modulation Modes for rPPG.

V1-V8  (Phase 1): scalar, strong, mlp, b_gate, delta_b, exp, sigmoid, per_dim
V9-V13 (Phase 3): cd_modulate, d_gate, power, additive, layer_init
V16-V18 (Phase 5, NEW — 机制性突破, 非 1+αq 标量增益族):
  V16 "qbc_gate" : per-state 双向门控 B 和 C —— 干预"选择性"本身。
                   B' = B * g_in(q,x),  C' = C * g_out(q,x)  (每个 state 独立门)
                   低质量帧 -> 抑制写入+读出状态 = 让 SSM "跳过坏帧"。
  V17 "biconsist": 双向一致性加权。正向扫 + 反向扫, 用两条轨迹的一致性做动态
                   可信度 —— 内生 quality, 不依赖外部 q 信号质量, 契合准周期脉搏对称性。
  V18 "qtime"    : 保留 q 的完整时间分辨率 (不 mean 掉 L) + warmup 门控,
                   低质量早期帧不污染初始状态 (修复 tgrad_q 的意图)。

Route 1 (振荡型 A — 机制上与 Δ/B/C 调制完全正交, 改的是"状态转移"本身):
  V19 "osc"      : 复数振荡 A = −exp(a) + i·ω, 每个 state 一个可学习振荡频率 ω。
                   让 SSM 的记忆本身具有周期性, 天生偏好心率带 (0.7–4Hz) 动力学。
                   直接用 selective_scan_fn 的复数分支 (需 kernel 支持复数)。
  V20 "osc_real" : 用 2x2 旋转矩阵把复数振荡拆成两路实数 state 模拟, 数学等价,
                   完全不依赖复数 kernel —— 保证在任何 selective_scan_fn 上都能跑。
                   这是 osc 的 fallback / 默认推荐实现。

All modes inject signals into selective_scan_fn. No CUDA kernel modified.
biconsist 跑两次 scan (fwd + flipped), 开销 ~2x, 仍用融合 kernel。
osc_real 把 d_state 翻倍 (每个振荡子占 2 个实 state), 开销约 1x-1.3x。
"""
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat
from mamba_ssm_scan.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from mamba_ssm_scan.modules.mamba_simple import Mamba as _OriginalMamba

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None


# 新增的机制性模式
_V2_MODES = ("qbc_gate", "biconsist", "qtime")
_OSC_MODES = ("osc", "osc_real", "osc_damped", "osc_bi")   # Route 1: 振荡型 A
_PLOCK_MODES = ("plock", "osc_plock")   # Route 2: 相位对齐 Δ (采样节奏耦合心率)
_MSCALE_MODES = ("mscale", "osc_mscale")  # Route 3: 多尺度扩张扫描
# 需要振荡动力学 (ω) 参数的模式 — 含叠加变体
_OSC_DYN_MODES = _OSC_MODES + ("osc_plock", "osc_mscale", "osc_locked")
_LOCKED_MODES = ("osc_locked",)  # Route 4: ω 锁定到观测心率


class QualityMamba(_OriginalMamba):
    """Multi-mode quality-guided Mamba — 13 旧模式 + 3 新机制模式。"""

    def __init__(
        self, d_model, d_state=16, d_conv=4, expand=2,
        dt_rank="auto", dt_min=0.001, dt_max=0.1,
        dt_init="random", dt_scale=1.0, dt_init_floor=1e-4,
        conv_bias=True, bias=False, use_fast_path=True,
        layer_idx=None, device=None, dtype=None,
        modulation_mode: str = "scalar",
        quality_scale_init: float = 0.1,
        b_scale_init: float = 0.1,
        mlp_hidden: int = 16,
        # ---- 新模式超参 ----
        qbc_hidden: int = 8,
        bic_beta_init: float = 2.0,
        bic_share: bool = True,
        qtime_warmup: int = 8,
        # ---- Route 1 振荡 A 超参 ----
        osc_fps: float = 30.0,        # 视频帧率, 用于把 ω 初始化到心率带
        osc_hr_lo: float = 0.7,       # 心率带下界 (Hz) ~42 bpm
        osc_hr_hi: float = 4.0,       # 心率带上界 (Hz) ~240 bpm
        osc_learn_omega: bool = True, # ω 是否可学习 (False=固定频率组)
    ):
        super().__init__(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
            dt_rank=dt_rank, dt_min=dt_min, dt_max=dt_max, dt_init=dt_init,
            dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            conv_bias=conv_bias, bias=bias,
            use_fast_path=False, layer_idx=layer_idx, device=device, dtype=dtype,
        )
        self.modulation_mode = modulation_mode
        self.d_inner_val = int(self.expand * d_model)
        self.d_state_val = d_state

        # ================= 旧模式参数 (V1-V13) =================
        # ---- scalar / strong / delta_b / layer_init / power: Δ scale ----
        if modulation_mode in ("scalar", "strong", "delta_b", "layer_init", "power"):
            self.quality_scale = nn.Parameter(torch.tensor(quality_scale_init))
        else:
            self.quality_scale = None

        if modulation_mode == "mlp":
            self.q_mlp = nn.Sequential(
                nn.Linear(1, mlp_hidden), nn.SiLU(),
                nn.Linear(mlp_hidden, mlp_hidden), nn.SiLU(),
                nn.Linear(mlp_hidden, 1),
            )
            for m in self.q_mlp:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01); nn.init.zeros_(m.bias)
        else:
            self.q_mlp = None

        if modulation_mode in ("b_gate", "delta_b"):
            self.b_scale = nn.Parameter(torch.tensor(b_scale_init))
        else:
            self.b_scale = None

        if modulation_mode == "exp":
            self.quality_scale = nn.Parameter(torch.tensor(quality_scale_init))

        if modulation_mode == "sigmoid":
            self.gate_scale = nn.Parameter(torch.tensor(5.0))
            self.gate_threshold = nn.Parameter(torch.tensor(0.5))
            self.gate_output_scale = nn.Parameter(torch.tensor(2.0))
        else:
            self.gate_scale = None; self.gate_threshold = None; self.gate_output_scale = None

        if modulation_mode == "per_dim":
            self.quality_scales = nn.Parameter(torch.randn(self.d_inner_val) * 0.1 + 0.1)
        else:
            self.quality_scales = None

        if modulation_mode == "cd_modulate":
            self.c_scale = nn.Parameter(torch.tensor(0.1))
            self.d_quality_scale = nn.Parameter(torch.tensor(0.1))
        elif modulation_mode == "d_gate":
            self.d_quality_scale = nn.Parameter(torch.tensor(0.5))
            self.c_scale = None
        else:
            self.c_scale = None; self.d_quality_scale = None

        if modulation_mode == "power":
            self.power_exp = nn.Parameter(torch.tensor(1.5))
        else:
            self.power_exp = None

        if modulation_mode == "additive":
            self.add_scale = nn.Parameter(torch.tensor(0.1))
            self.add_bias = nn.Parameter(torch.tensor(0.0))
        else:
            self.add_scale = None; self.add_bias = None

        # ================= 新模式参数 (V16-V18) =================
        # ---- V16 qbc_gate ----
        if modulation_mode == "qbc_gate":
            self.gin = nn.Sequential(
                nn.Linear(2, qbc_hidden), nn.SiLU(),
                nn.Linear(qbc_hidden, d_state),
            )
            self.gout = nn.Sequential(
                nn.Linear(2, qbc_hidden), nn.SiLU(),
                nn.Linear(qbc_hidden, d_state),
            )
            for mlp in (self.gin, self.gout):
                nn.init.normal_(mlp[0].weight, std=1e-2); nn.init.zeros_(mlp[0].bias)
                nn.init.normal_(mlp[2].weight, std=1e-2)
                nn.init.constant_(mlp[2].bias, 3.0)   # sigmoid(3)≈0.95 门初始近开
            # 修正1: 初始化为 -4 -> sigmoid(-4)≈0.018, warm-start 近恒等
            self.gate_residual = nn.Parameter(torch.tensor(-4.0))

        # ---- V17 biconsist ----
        if modulation_mode == "biconsist":
            self.bic_beta = nn.Parameter(torch.tensor(float(bic_beta_init)))
            self.bic_mix = nn.Parameter(torch.tensor(0.0))   # sigmoid(0)=0.5 均衡起步
            self.bic_share = bic_share

        # ---- V18 qtime ----
        if modulation_mode == "qtime":
            self.qtime_warmup = qtime_warmup
            self.qtime_scale = nn.Parameter(torch.tensor(0.1))
            # 修正2: 初始 linspace(1,2) -> sigmoid 后 0.73~0.88, 从"轻微折扣"开始学
            self.qtime_warm_gate = nn.Parameter(torch.linspace(1.0, 2.0, qtime_warmup))

        # ---- V19/V20 osc / osc_real / osc_damped: 振荡型 A ----
        if modulation_mode in _OSC_DYN_MODES:
            self.osc_fps = osc_fps
            self.osc_learn_omega = osc_learn_omega
            f_init = torch.linspace(osc_hr_lo, osc_hr_hi, self.d_state_val)
            omega_init = 2.0 * math.pi * f_init

            if modulation_mode in ("osc", "osc_real", "osc_bi", "osc_plock", "osc_mscale"):
                if osc_learn_omega:
                    self.omega_log = nn.Parameter(torch.log(omega_init))
                else:
                    self.register_buffer("omega_fixed", omega_init)

            if modulation_mode in ("osc_real", "osc_bi", "osc_plock", "osc_mscale", "osc_locked"):
                self.osc_imag_proj = nn.Linear(self.d_inner_val, self.d_state_val, bias=False)
                nn.init.zeros_(self.osc_imag_proj.weight)
                self.osc_out_scale = nn.Parameter(torch.tensor(1.0))

            # V25 osc_locked: ω 锁定到观测心率, 可学习谐波偏移
            if modulation_mode == "osc_locked":
                # 谐波系数: 覆盖基频 + 谐波 (1x, 2x, 3x ...), 可学习小偏移
                self.harmonic_offset = nn.Parameter(
                    torch.linspace(0.9, 2.4, self.d_state_val))
                # 用 linspace 存 ω 初始化 (供 warm-start 参考, 实际 forward 会被观测频率覆盖)
                if osc_learn_omega:
                    self.omega_log = nn.Parameter(torch.log(omega_init))
                else:
                    self.register_buffer("omega_fixed", omega_init)

            # V21 osc_damped v2: direct f + Q parameterization (AVOID softplus dead zone)
            if modulation_mode == "osc_damped":
                self.osc_imag_proj = nn.Linear(self.d_inner_val, self.d_state_val, bias=False)
                nn.init.zeros_(self.osc_imag_proj.weight)
                self.osc_out_scale = nn.Parameter(torch.tensor(1.0))
                # f_hz ∈ [0.5, 5.0] Hz via sigmoid, with jitter for gradient direction.
                # 关键: inverse sigmoid 要求 f ∈ (0.5, 5.0) 开区间, 否则 log 负值 -> NaN.
                # 之前 clamp(0.35, 6.0) 超出边界导致 Epoch0 即 NaN 爆炸.
                f_jitter = f_init + torch.randn(self.d_state_val) * 0.3
                f_jitter = f_jitter.clamp(min=0.6, max=4.5)   # 严格落在 (0.5, 5.0) 内部
                self.f_logit = nn.Parameter(torch.log(
                    (f_jitter - 0.5) / (5.0 - f_jitter)))     # 无 +1e-8, 保证边界安全
                # Q ∈ [1, 10] via sigmoid, initial broad sampling
                Q_init = torch.linspace(1.0, 8.0, self.d_state_val) + torch.randn(self.d_state_val) * 1.0
                Q_init = Q_init.clamp(1.1, 9.9)
                self.Q_logit = nn.Parameter(torch.log(
                    (Q_init - 1.0) / (10.0 - Q_init)))
                # Diagnostic: track if params moved
                self.register_buffer("_init_f", f_jitter.clone())
                self.register_buffer("_init_Q", Q_init.clone())
                self.register_buffer("_train_steps", torch.tensor(0))

        # ---- V23 plock / osc_plock: 相位对齐 Δ (采样节奏耦合心率) ----
        if modulation_mode in _PLOCK_MODES:
            # γ coupling strength, init 0 -> warm-start = identity
            self.plock_scale = nn.Parameter(torch.tensor(0.0))
            self.plock_window = 32   # sliding FFT window (frames)
            self.plock_hop = 8       # hop between windows

        # ---- V24 mscale / osc_mscale: 多尺度扩张扫描 ----
        if modulation_mode in _MSCALE_MODES:
            self.mscale_dilations = (1, 2, 4)
            # learnable fusion weights, init: dilation=1 dominant, others near 0
            n_scale = len(self.mscale_dilations)
            self.mscale_logits = nn.Parameter(torch.tensor([3.0] + [0.0] * (n_scale - 1)))

    # ==================================================================
    #  旧模式的调制函数 (保持与你原实现一致)
    # ==================================================================
    def _compute_modulation(self, quality, dt_shape):
        B, D, L = dt_shape
        if quality is None:
            return torch.ones(B, 1, L, dtype=torch.float32, device=self.A_log.device)
        if quality.dim() == 3:
            quality = quality.mean(dim=-1)
        q = quality.float().unsqueeze(1)  # (B, 1, L)
        mode = self.modulation_mode

        if mode in ("scalar", "strong", "layer_init"):
            mod = 1.0 + self.quality_scale * q
        elif mode == "mlp":
            q_in = q.squeeze(1).unsqueeze(-1)
            delta_q = self.q_mlp(q_in).squeeze(-1).unsqueeze(1)
            mod = 1.0 + delta_q
        elif mode == "exp":
            mod = torch.exp(self.quality_scale * q)
        elif mode == "sigmoid":
            gate = torch.sigmoid(self.gate_scale * (q - self.gate_threshold))
            mod = 1.0 + self.gate_output_scale * gate
        elif mode == "per_dim":
            scales = self.quality_scales.view(1, D, 1)
            mod = 1.0 + scales * q
            return mod.clamp(0.05, 10.0)
        elif mode == "power":
            p = self.power_exp.clamp(0.5, 5.0)
            mod = 1.0 + self.quality_scale * (q ** p)
        elif mode == "additive":
            bias_q = self.add_bias + 5.0 * q
            mod = 1.0 + self.add_scale * torch.tanh(bias_q)
        elif mode in ("b_gate",):
            return torch.ones(B, 1, L, dtype=torch.float32, device=q.device)
        else:  # delta_b etc
            mod = 1.0 + (self.quality_scale if self.quality_scale is not None else 0.1) * q
        return mod.clamp(0.05, 10.0)

    def _modulate_delta(self, dt, quality=None):
        if quality is None:
            return dt
        mod = self._compute_modulation(quality, dt.shape)
        return dt * mod.to(dtype=dt.dtype)

    def _modulate_B(self, B, quality, d_state):
        if quality is None or self.b_scale is None:
            return B
        if quality.dim() == 3:
            quality = quality.mean(dim=-1)
        q = quality.float().unsqueeze(1)
        mod = (1.0 + self.b_scale * q).clamp(0.05, 10.0)
        return B * mod

    def _modulate_CD(self, C, quality):
        if quality is None:
            return C
        if quality.dim() == 3:
            quality = quality.mean(dim=-1)
        q = quality.float()
        if self.modulation_mode == "cd_modulate":
            c_mod = (1.0 + self.c_scale * q.unsqueeze(1)).clamp(0.1, 5.0)
            return C * c_mod
        return C

    # ==================================================================
    #  新模式的辅助函数 (V16-V18)
    # ==================================================================
    @staticmethod
    def _chan_summary(x):
        """x: (B, d_inner, L) -> (B, L, 1). 用 L2 能量做通道摘要, 反映信号强弱。"""
        s = x.float().pow(2).mean(dim=1, keepdim=True).sqrt()          # (B,1,L)
        s = (s - s.mean(dim=-1, keepdim=True)) / (s.std(dim=-1, keepdim=True) + 1e-5)
        return rearrange(s, "b 1 l -> b l 1")

    # ---------- Route 4: osc_locked 观测心率估计 ----------
    def _estimate_obs_hr(self, x, seqlen):
        """Estimate observed HR frequency (Hz) via FFT peak of channel-averaged signal.

        x: (B, d_inner, L). Returns (B,) in Hz.
        """
        B, D, L = x.shape
        xm = x.mean(dim=1)                       # (B, L)
        xm = xm - xm.mean(dim=-1, keepdim=True)
        win = min(64, L)
        seg = xm[:, :win] * torch.hann_window(win, device=x.device)
        S = torch.fft.rfft(seg, dim=-1).abs()    # (B, win//2+1)
        S = S[:, 1:]                              # drop DC
        k = S.argmax(dim=-1).float() + 1.0        # (B,)
        # bin k -> freq = k / win * fs_effective. fs_effective ≈ 15 (30fps/stride2)
        f_obs = k / win * 15.0
        return f_obs.clamp(min=0.7, max=4.0)

    # ---------- Route 2: 相位对齐 Δ (plock) ----------
    def _estimate_instant_freq(self, x, seqlen):
        """Estimate instantaneous dominant frequency (bin index) from x (B,D,L) -> (B,L).

        Sliding-window FFT; dominant bin in HR band (period 4-24 frames).
        Returns relative freq (bin index) — actual Hz needs fs, but for the
        *ratio* f_ref/f_hr the fs cancels out.
        """
        B, D, L = x.shape
        xm = x.mean(dim=1)                       # (B, L) channel-averaged
        xm = xm - xm.mean(dim=-1, keepdim=True)  # de-mean

        win = min(self.plock_window, L)
        hop = max(1, self.plock_hop)
        if L < win:
            # degenerate: constant freq
            return torch.ones(B, L, device=x.device)

        n_win = (L - win) // hop + 1
        hann = torch.hann_window(win, device=x.device)
        bins = []
        for i in range(n_win):
            seg = xm[:, i * hop : i * hop + win] * hann          # (B, win)
            S = torch.fft.rfft(seg, dim=-1).abs()                # (B, win//2+1)
            S = S[:, 1:]                                          # drop DC
            k = S.argmax(dim=-1).float() + 1.0                    # (B,) dominant bin
            bins.append(k)
        bins = torch.stack(bins, dim=-1)                          # (B, n_win)

        # interpolate to full L (linear)
        f_hr = F.interpolate(bins.unsqueeze(1), size=L, mode="linear",
                             align_corners=True).squeeze(1)       # (B, L)
        return f_hr.clamp(min=1.0)

    # ---------- Route 3: 多尺度扩张扫描 (mscale) ----------
    def _mscale_scan(self, x, dt, B, C, z, seqlen):
        """G-way dilated scan: each dilation downsamples-scans-upsamples, then fuse.

        dilation d: sample every d frames, Mamba-scan, upsample back.
        Learnable fusion weights; init favors dilation=1 (near single-scale).
        osc_mscale uses oscillatory-A dynamics; mscale uses standard real A.
        """
        Bsz, d_inner, L = x.shape
        dilations = self.mscale_dilations
        weights = torch.softmax(self.mscale_logits, dim=0)        # (G,)
        use_osc = (self.modulation_mode == "osc_mscale")

        def scan_one(xd, dtd, Bd, Cd, zd, Ld):
            if use_osc:
                return self._osc_real_scan(xd, dtd, Bd, Cd, zd, Ld)
            else:
                A = -torch.exp(self.A_log.float())
                return selective_scan_fn(
                    xd, dtd, A, Bd, Cd, self.D.float(), z=zd,
                    delta_bias=self.dt_proj.bias.float(), delta_softplus=True)

        y_sum = torch.zeros_like(x)
        for gi, d in enumerate(dilations):
            if d == 1:
                yd = scan_one(x, dt, B, C, z, L)
            else:
                idx = torch.arange(0, L, d, device=x.device)
                xd = x[:, :, idx]; dtd = dt[:, :, idx]
                Bd = B[:, :, idx]; Cd = C[:, :, idx]; zd = z[:, :, idx]
                yd = scan_one(xd, dtd, Bd, Cd, zd, len(idx))
                yd = F.interpolate(yd, size=L, mode="linear", align_corners=True)
            y_sum = y_sum + weights[gi] * yd

        return y_sum

    def _qbc_gates(self, quality, x, seqlen):
        """生成 per-state 输入门 g_in 和输出门 g_out, 均 (B, d_state, L)。"""
        if quality is not None and quality.dim() == 3:
            quality = quality.mean(dim=-1)
        if quality is None:
            q = torch.zeros(x.shape[0], seqlen, 1, device=x.device)
        else:
            q = quality.float().unsqueeze(-1)                          # (B,L,1)
        feat = torch.cat([q, self._chan_summary(x)], dim=-1)          # (B,L,2)
        g_in = torch.sigmoid(self.gin(feat))                          # (B,L,d_state)
        g_out = torch.sigmoid(self.gout(feat))
        a = torch.sigmoid(self.gate_residual)                         # 学习注入强度, 初始≈0.018
        g_in = 1.0 + a * (g_in - 1.0)
        g_out = 1.0 + a * (g_out - 1.0)
        return (rearrange(g_in, "b l n -> b n l"),
                rearrange(g_out, "b l n -> b n l"))

    # ---------- Route 1: 振荡 A 的实数旋转扫描 (osc_real) ----------
    def _osc_real_scan(self, x, dt, B, C, z, seqlen):
        """用 2x2 旋转矩阵实现复数振荡 SSM (纯 PyTorch, 不依赖复数 kernel)。

        每个 state 维护一个复数隐状态 h = (hr, hi):
            [hr; hi]_t = e^{-Δλ} · R(Δω) · [hr; hi]_{t-1} + [Δ·B·u ; Δ·B·u_imag]
            y_t += C · hr_t   (取实部读出)
        R(θ) 是旋转矩阵。λ 复用父类 A_log(对 state 取均值到每 state 一个标量衰减),
        ω 为可学习振荡频率。u_imag 由 osc_imag_proj 从 x 投影 (初始 0)。

        形状: x,z:(B,d_inner,L)  dt:(B,d_inner,L)  B,C:(B,d_state,L)
        返回 y:(B,d_inner,L)
        """
        Bsz, d_inner, L = x.shape
        N = self.d_state_val
        # osc_damped v2: derive f_hz, Q from sigmoid parameterization
        if self.modulation_mode == "osc_damped":
            f_hz = 0.5 + 4.5 * torch.sigmoid(self.f_logit)  # [0.5, 5.0] Hz
            Q_val = 1.0 + 9.0 * torch.sigmoid(self.Q_logit)  # [1, 10]
            omega = 2.0 * math.pi * f_hz
            lam = omega / (2.0 * Q_val + 1e-8)
            if self.training:
                self._train_steps += 1
        # osc_locked: ω 锁定到观测心率主频 × 谐波偏移
        elif self.modulation_mode == "osc_locked":
            lam = torch.exp(self.A_log.float()).mean(dim=0)  # (N,)
            f_obs = self._estimate_obs_hr(x, L)               # (B,) observed HR freq (Hz)
            # 观测频率 -> rad/s, 广播到每个 state 的谐波
            omega = 2.0 * math.pi * f_obs.mean() * self.harmonic_offset.clamp(0.5, 4.0)  # (N,)
            # 若 batch 内多个样本, 用 batch 平均 (简化: 单 ω 组)
        else:
            lam = torch.exp(self.A_log.float()).mean(dim=0)  # (N,)
            omega = torch.exp(self.omega_log) if self.osc_learn_omega else self.omega_fixed  # (N,)
        # dt 对 d_inner 取均值 -> 每帧每 state 一个步长 (B,L,1)?  这里用 per (B,L) 标量步长
        dt_s = F.softplus(dt + self.dt_proj.bias.float()[None, :, None]).mean(dim=1)  # (B,L)
        # 虚部输入投影 (初始 0 -> 近原始行为)
        u_imag = self.osc_imag_proj(rearrange(x, "b d l -> b l d"))   # (B,L,N)
        u_imag = rearrange(u_imag, "b l n -> b n l")                  # (B,N,L)
        # 实部输入 = B·(x 的通道能量) —— 用 B 直接作为输入门, x 汇聚成每帧强度
        xin = x.mean(dim=1, keepdim=True)                            # (B,1,L) 输入强度
        u_real = B * xin                                             # (B,N,L)

        hr = x.new_zeros(Bsz, N)
        hi = x.new_zeros(Bsz, N)
        ys = []
        for t in range(L):
            d = dt_s[:, t].unsqueeze(-1)                             # (B,1)
            decay = torch.exp(-d * lam[None, :])                     # (B,N)
            th = d * omega[None, :]                                  # (B,N)
            cos, sin = torch.cos(th), torch.sin(th)
            hr_new = decay * (cos * hr - sin * hi) + d * u_real[:, :, t]
            hi_new = decay * (sin * hr + cos * hi) + d * u_imag[:, :, t]
            hr, hi = hr_new, hi_new
            y_t = (C[:, :, t] * hr).sum(dim=1)                       # (B,) 读实部
            ys.append(y_t)
        y = torch.stack(ys, dim=-1)                                  # (B,L)
        # 广播回 d_inner 并加 D 跳连 + 门控 z
        y = y.unsqueeze(1) * self.osc_out_scale                      # (B,1,L)
        y = y + x * self.D.float()[None, :, None]                    # 跳连
        y = y * F.silu(z)                                            # 门控
        return y

    # ==================================================================
    #  forward
    # ==================================================================
    def forward(self, hidden_states, quality=None, inference_params=None):
        batch, seqlen, dim = hidden_states.shape
        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l", l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())
        has_injection = (quality is not None)

        # ---- fast path: 无注入且非新模式时走原融合核 ----
        if (not has_injection and self.use_fast_path and causal_conv1d_fn is not None
                and self.modulation_mode not in _V2_MODES
                and self.modulation_mode not in _OSC_MODES):
            return mamba_inner_fn(
                xz, self.conv1d.weight, self.conv1d.bias,
                self.x_proj.weight, self.dt_proj.weight,
                self.out_proj.weight, self.out_proj.bias,
                A, None, None, self.D.float(),
                delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
            )

        x, z = xz.chunk(2, dim=1)
        if conv_state is not None:
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            x = causal_conv1d_fn(
                x=x, weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias, activation=self.activation,
            )

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        mode = self.modulation_mode

        # ================= V23 plock / osc_plock: 相位对齐 Δ =================
        # Δ_t ∝ 1/f_hr(t): 心率快→Δ小(密采样), 心率慢→Δ大(稀采样),
        # 归一化掉心率变异, 让状态在相位域均匀推进。
        # 与死掉的 Δ 调制区别: 不是乘外部 quality 增益, 而是按瞬时频率对齐采样节奏。
        if mode in _PLOCK_MODES:
            f_hr = self._estimate_instant_freq(x, seqlen)        # (B, L) dominant freq (bin)
            f_ref = f_hr.median(dim=-1, keepdim=True)[0].clamp(min=1.0)  # (B, 1)
            r = f_ref / f_hr.clamp(min=1.0) - 1.0                 # (B, L): >0 when slow HR
            mod = 1.0 + self.plock_scale * r                     # (B, L), γ init 0
            dt = dt * mod.unsqueeze(1).to(dt.dtype)               # broadcast over D

        # ================= V24 mscale / osc_mscale: 多尺度扩张扫描 =================
        if mode in _MSCALE_MODES:
            y = self._mscale_scan(x, dt, B, C, z, seqlen)
            return self.out_proj(rearrange(y, "b d l -> b l d"))

        # ================= V16 qbc_gate =================
        if mode == "qbc_gate":
            g_in, g_out = self._qbc_gates(quality, x, seqlen)         # (B,d_state,L)
            B = B * g_in.to(B.dtype)      # 控制"写入状态"多少
            C = C * g_out.to(C.dtype)     # 控制"从状态读出"多少
            y = selective_scan_fn(
                x, dt, A, B, C, self.D.float(), z=z,
                delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
            )
            return self.out_proj(rearrange(y, "b d l -> b l d"))

        # ================= V17 biconsist =================
        if mode == "biconsist":
            y_f = selective_scan_fn(
                x, dt, A, B, C, self.D.float(), z=z,
                delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
            )
            xr, dtr = x.flip(-1), dt.flip(-1)
            Br, Cr, zr = B.flip(-1), C.flip(-1), z.flip(-1)
            y_b = selective_scan_fn(
                xr, dtr, A, Br, Cr, self.D.float(), z=zr,
                delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
            ).flip(-1)
            diff = (y_f - y_b).abs().mean(dim=1, keepdim=True)        # (B,1,L)
            diff = diff / (diff.mean(dim=-1, keepdim=True) + 1e-5)
            w = torch.exp(-F.softplus(self.bic_beta) * diff)         # 一致性权重 (0,1]
            mix = torch.sigmoid(self.bic_mix)
            y = (mix * y_f + (1.0 - mix) * y_b) * w
            return self.out_proj(rearrange(y, "b d l -> b l d"))

        # ================= V18 qtime =================
        if mode == "qtime":
            if quality is not None:
                if quality.dim() == 3:
                    quality = quality.mean(dim=-1)
                q = quality.float().unsqueeze(1)                      # (B,1,L) 保留完整时间分辨率
                mod = (1.0 + self.qtime_scale * q).clamp(0.05, 10.0)
                dt = dt * mod.to(dt.dtype)
                w = torch.ones(seqlen, device=dt.device, dtype=dt.dtype)
                k = min(self.qtime_warmup, seqlen)
                w[:k] = torch.sigmoid(self.qtime_warm_gate[:k]).to(dt.dtype)
                dt = dt * w.view(1, 1, -1)
            y = selective_scan_fn(
                x, dt, A, B, C, self.D.float(), z=z,
                delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
            )
            return self.out_proj(rearrange(y, "b d l -> b l d"))

        # ================= V19 osc (复数 kernel) =================
        if mode == "osc":
            # 复数振荡 A = −exp(a) + i·ω。selective_scan_fn 支持复数 A 时直接传入。
            omega = torch.exp(self.omega_log) if self.osc_learn_omega else self.omega_fixed
            # A_c: (d_inner, d_state) 复数。实部 = -exp(A_log)(衰减), 虚部 = ω (广播到 d_inner)
            A_real = -torch.exp(self.A_log.float())                   # (d_inner, d_state)
            A_imag = omega[None, :].expand_as(A_real)                 # (d_inner, d_state)
            A_c = torch.complex(A_real, A_imag)
            # 复数 SSM 要求 B,C 也为复数; 虚部初始 0
            B_c = torch.complex(B, torch.zeros_like(B))
            C_c = torch.complex(C, torch.zeros_like(C))
            try:
                y = selective_scan_fn(
                    x, dt, A_c, B_c, C_c, self.D.float(), z=z,
                    delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
                )
            except (RuntimeError, TypeError) as e:
                # kernel 不支持复数 -> 自动回退到 osc_real (需预先实例化对应层;
                # 实际使用建议直接用 modulation_mode="osc_real")
                raise RuntimeError(
                    "selective_scan_fn 不支持复数 A。请改用 modulation_mode='osc_real' "
                    "(纯实数旋转实现, 无需复数 kernel)。原错误: " + str(e))
            return self.out_proj(rearrange(y, "b d l -> b l d"))

        # ================= V20 osc_real (2x2 旋转, 默认推荐) =================
        if mode in ("osc_real", "osc_plock", "osc_locked"):
            y = self._osc_real_scan(x, dt, B, C, z, seqlen)          # (B,d_inner,L)
            return self.out_proj(rearrange(y, "b d l -> b l d"))

        # ================= V21 osc_damped (独立 λ + Q 因子) =================
        if mode == "osc_damped":
            y = self._osc_real_scan(x, dt, B, C, z, seqlen)
            return self.out_proj(rearrange(y, "b d l -> b l d"))

        # ================= V22 osc_bi: osc_real bidirectional average =================
        # Forward scan + backward scan (flip input), average the two outputs.
        # Shares ALL weights with osc_real. Param count identical.
        # No external quality signal. No consistency weighting.
        if mode == "osc_bi":
            y_fwd = self._osc_real_scan(x, dt, B, C, z, seqlen)       # (B,d_inner,L)
            # Flip: x -> sequence-reversed input
            x_r = x.flip(-1); dt_r = dt.flip(-1)
            B_r = B.flip(-1); C_r = C.flip(-1); z_r = z.flip(-1)
            y_bwd = self._osc_real_scan(x_r, dt_r, B_r, C_r, z_r, seqlen)  # (B,d_inner,L)
            y_bwd = y_bwd.flip(-1)  # flip back to original time order
            y = 0.5 * y_fwd + 0.5 * y_bwd
            return self.out_proj(rearrange(y, "b d l -> b l d"))

        # ================= 旧模式 (V1-V13) =================
        if has_injection:
            B = self._modulate_B(B, quality, self.d_state)
            dt = self._modulate_delta(dt, quality)
            if self.modulation_mode in ("cd_modulate", "d_gate"):
                C = self._modulate_CD(C, quality)

        y = selective_scan_fn(
            x, dt, A, B, C, self.D.float(), z=z,
            delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
            return_last_state=ssm_state is not None,
        )
        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
        return self.out_proj(rearrange(y, "b d l -> b l d"))


    def export_q_factors(self):
        """For osc_damped mode: return per-state (ω_hz, λ, Q) as numpy arrays."""
        if self.modulation_mode != "osc_damped":
            return None
        import numpy as np
        f_hz = (0.5 + 4.5 * torch.sigmoid(self.f_logit)).detach().cpu().numpy()
        Q_val = (1.0 + 9.0 * torch.sigmoid(self.Q_logit)).detach().cpu().numpy()
        omega_hz = f_hz
        lam = omega_hz / (2.0 * Q_val + 1e-8)
        return {"omega_hz": omega_hz, "lam": lam, "Q": Q_val,
                "f_hz_init": self._init_f.cpu().numpy() if hasattr(self, '_init_f') else None,
                "Q_init": self._init_Q.cpu().numpy() if hasattr(self, '_init_Q') else None,
                "train_steps": self._train_steps.item()}

    def get_physics_param_group(self):
        """Returns (names, params) for physics parameters that need higher LR."""
        if self.modulation_mode != "osc_damped":
            return [], []
        phys_params = []
        phys_names = []
        for name in ["f_logit", "Q_logit", "osc_imag_proj.weight", "osc_out_scale"]:
            if hasattr(self, name):
                p = getattr(self, name)
                if isinstance(p, nn.Parameter):
                    phys_params.append(p); phys_names.append(name)
        return phys_names, phys_params

class QualityScanMamba(QualityMamba):
    """Default quality scan Mamba (与原接口一致)。"""
    def __init__(self, d_model, modulation_mode="scalar", quality_scale_init=0.1, **kwargs):
        kwargs.setdefault("modulation_mode", modulation_mode)
        kwargs.setdefault("quality_scale_init", quality_scale_init)
        super().__init__(d_model, **kwargs)
