# -*- coding: utf-8 -*-
"""MSRS-ABiLSTM 故障识别网络（双 frontend：谱前端 v1 / VMD 旧前端）。

设计依据：版本3 §4.4（网络与前端 v1 配方）、§0.2（冻结前端配方）、§2 L3（对齐模式语义）。

数据流（自上而下）：
    输入 (B, C, D)   C=通道数（logspec/cepstrum=1，vmd=K=8），D=特征维（256/236/512）
      │
    ① ConvFrontEnd —— 3 级 SMCBlock(32,64,128) + 池化 → (B,128,8)
    │    spectral_v1：MaxPool(2)→MaxPool(4)→AdaptiveMaxPool1d(8)，固定 8 步
    │      （logspec256 下与 (2,4,4) 逐点等价；cepstrum236 末级自适应吸收到 8，不补零）
    │    legacy_vmd ：MaxPool(4)×3（512→128→32→8），仅 vmd 分支用，等构复现旧前端
      │
    ② ResidualShrinkageUnit × 2   通道软阈值去噪
    ③ SE 通道注意力（可选）
    ④ BiLSTM（单层双向，hidden=128） (B,128,8)→(B,8,256)
    ⑤ AdditiveAttention  → 上下文向量 (B,256) + 注意力权重 (B,8)（频带重要性，版本3 §4.4）
    ⑥ 分类头 FC(256→64)→ReLU→Dropout(0.5)→FC(64→4)

对齐损失 conditional_mmd_loss：global / conditional（probs>=0.5 硬掩膜）/ confidence（softmax 软加权），
语义见版本3 §2 L3。首级池化必须 /2 的物理依据见版本3 §4.4（BPFO bin4 / BPFI bin6，/4 会并入同池）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

FRONTEND_NAMES = ("spectral_v1", "legacy_vmd")


# ---------------------------------------------------------------------------
# ① SMC 多尺度卷积块【LiConvFormer】
# ---------------------------------------------------------------------------
class SMCBlock(nn.Module):
    """并联多尺度卷积（k=3/7/15）+ 1×1 融合。输入 (B,C_in,L)→(B,C_out,L)，same padding。

    通道数变化时分支 groups=1（普通多尺度卷积，非 depthwise）。参数依据版本3 §0.2/§4.4。
    """

    def __init__(self, c_in, c_out, kernels=(3, 7, 15)):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Conv1d(c_in, c_out, k, padding=k // 2,
                      groups=c_in if c_in == c_out else 1)
            for k in kernels
        )
        self.pw = nn.Conv1d(c_out * len(kernels), c_out, 1)
        self.bn = nn.BatchNorm1d(c_out)

    def forward(self, x):
        out = torch.cat([b(x) for b in self.branches], dim=1)
        return F.relu(self.bn(self.pw(out)))


class ConvFrontEnd(nn.Module):
    """3 级 SMCBlock(32,64,128) + 池化，把输入压到 (B,128,8)（版本3 §4.4）。

    frontend='spectral_v1'：MaxPool(2)→MaxPool(4)→AdaptiveMaxPool1d(8)，固定 8 步。
        - logspec256：/2=128 → /4=32 → 自适应 8（与 (2,4,4) 逐点等价）。
        - cepstrum236：118 → 29 → 自适应 8（不补零，避免假频段，版本3 §4.4/§P3.1）。
        - 首级 /2 保住 BPFO(bin4)/BPFI(bin6) 可分（版本3 §4.4 物理依据）。
    frontend='legacy_vmd'：MaxPool(4)×3（512→128→32→8），仅 vmd 分支等构复现旧前端。
    """

    def __init__(self, c_in, frontend="spectral_v1", channels=(32, 64, 128)):
        super().__init__()
        if frontend not in FRONTEND_NAMES:
            raise ValueError(f"未知 frontend: {frontend}")
        self.frontend = frontend
        self.channels = tuple(channels)
        blocks, pools = [], []
        c_prev = c_in
        for i, c in enumerate(channels):
            blocks.append(SMCBlock(c_prev, c))
            if frontend == "spectral_v1":
                if i == 0:
                    pools.append(nn.MaxPool1d(2))
                elif i == 1:
                    pools.append(nn.MaxPool1d(4))
                else:
                    pools.append(nn.AdaptiveMaxPool1d(8))
            else:  # legacy_vmd
                pools.append(nn.MaxPool1d(4))
            c_prev = c
        self.blocks = nn.ModuleList(blocks)
        self.pools = nn.ModuleList(pools)
        self.out_ch = c_prev

    def forward(self, x):
        for b, p in zip(self.blocks, self.pools):
            x = p(b(x))
        return x


# ---------------------------------------------------------------------------
# ② RSU 残差收缩单元【DRSN】
# ---------------------------------------------------------------------------
class ResidualShrinkageUnit(nn.Module):
    """残差块 + 通道自适应软阈值去噪（x' = sign(x)·relu(|x|-τ)，τ 由特征自生成）。

    对迁移场景：源/目标域噪声特性不同，自适应阈值让网络按输入决定压制哪些通道。
    """

    def __init__(self, c, hidden=16, scale=1.0, threshold_mode="signed_mean"):
        super().__init__()
        self.scale = scale
        self.threshold_mode = threshold_mode
        self.conv1 = nn.Conv1d(c, c, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(c)
        self.conv2 = nn.Conv1d(c, c, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(c)
        self.fc = nn.Sequential(
            nn.Linear(c, hidden), nn.ReLU(), nn.Linear(hidden, c), nn.Sigmoid())

    def forward(self, x):
        r = F.relu(self.bn1(self.conv1(x)))
        r = self.bn2(self.conv2(r))
        if self.threshold_mode == "signed_mean":
            stat = r.mean(dim=2)
        elif self.threshold_mode == "mean_abs":
            stat = r.abs().mean(dim=2)
        elif self.threshold_mode == "rms":
            stat = r.square().mean(dim=2).sqrt()
        else:
            raise ValueError(f"未知 threshold_mode: {self.threshold_mode}")
        tau = self.fc(stat) * self.scale
        r = torch.sign(r) * F.relu(r.abs() - tau.unsqueeze(-1))
        return F.relu(x + r)


# ---------------------------------------------------------------------------
# ⑤ 加性注意力
# ---------------------------------------------------------------------------
class AdditiveAttention(nn.Module):
    """Bahdanau 加性注意力：对 8 个序列步 h_t 加权聚合。

    e_t=v^T tanh(W h_t+b)；α=softmax(e)；c=Σα_t·h_t。谱前端下 α 解释为频带重要性
    （版本3 §4.4；cepstrum 序列轴中点不连续，可视化须标注，版本3 §P3.1）。
    """

    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, 1, bias=False)

    def forward(self, h):
        e = self.v(torch.tanh(self.W(h))).squeeze(-1)
        alpha = torch.softmax(e, dim=1)
        c = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)
        return c, alpha


# ---------------------------------------------------------------------------
# 完整网络
# ---------------------------------------------------------------------------
class MSRSABiLSTM(nn.Module):
    """多尺度残差收缩-注意力双向 LSTM（双 frontend）。

    构造参数（版本3 §4.4/§4.5）：
      n_channels : 输入通道数（logspec/cepstrum=1，vmd=K=8）
      frontend   : 'spectral_v1'（谱表征）| 'legacy_vmd'（vmd 分支等构复现）
      n_classes  : 4；lstm_hidden=128；lstm_layers=1；use_se；threshold_mode
    forward(x) → (logits,alpha)；forward(x,return_feat=True) → (logits,alpha,feat(B,256))。
    """

    def __init__(self, n_channels=1, n_classes=4, frontend="spectral_v1",
                 lstm_hidden=128, lstm_layers=1, use_se=True,
                 threshold_mode="signed_mean"):
        super().__init__()
        self.frontend_name = frontend
        self.n_channels = int(n_channels)
        self.frontend = ConvFrontEnd(c_in=n_channels, frontend=frontend)
        c = self.frontend.out_ch
        self.rsu = nn.Sequential(
            ResidualShrinkageUnit(c, threshold_mode=threshold_mode),
            ResidualShrinkageUnit(c, threshold_mode=threshold_mode),
        )
        self.use_se = use_se
        if use_se:
            self.se = nn.Sequential(
                nn.Linear(c, c // 4), nn.ReLU(), nn.Linear(c // 4, c), nn.Sigmoid())
        self.lstm = nn.LSTM(input_size=c, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.attn = AdditiveAttention(lstm_hidden * 2)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, n_classes))

    def forward(self, x, return_feat=False):
        f = self.frontend(x)
        f = self.rsu(f)
        if self.use_se:
            w = self.se(f.mean(dim=2)).unsqueeze(-1)
            f = f * w
        seq = f.permute(0, 2, 1)
        h, _ = self.lstm(seq)
        c, alpha = self.attn(h)
        logits = self.head(c)
        return (logits, alpha, c) if return_feat else (logits, alpha)


# ---------------------------------------------------------------------------
# 迁移损失：多核 MMD
# ---------------------------------------------------------------------------
def gaussian_kernel(x, y, sigma=(1, 5, 10)):
    """多核 RBF 核矩阵（3 个带宽 1/5/10 等权组合）。返回 (m,n) 核矩阵。"""
    m, n = x.size(0), y.size(0)
    xx, yy, xy = (x * x).sum(1), (y * y).sum(1), x @ y.T
    dist = xx.unsqueeze(1) + yy.unsqueeze(0) - 2 * xy
    return sum(torch.exp(-dist / (2 * s)) for s in sigma) / len(sigma)


def mmd_loss(f_s, f_t, estimator="biased"):
    """最大均值差异 MMD²=E[k(s,s')]+E[k(t,t')]-2E[k(s,t)]。默认 biased（有限 batch 非负）。"""
    if estimator not in ("biased", "unbiased"):
        raise ValueError(f"未知 MMD estimator: {estimator}")
    m, n = f_s.size(0), f_t.size(0)
    if estimator == "unbiased" and (m < 2 or n < 2):
        estimator = "biased"
    if estimator == "biased":
        k_ss = gaussian_kernel(f_s, f_s).mean()
        k_tt = gaussian_kernel(f_t, f_t).mean()
        k_st = gaussian_kernel(f_s, f_t).mean()
        return (k_ss + k_tt - 2 * k_st).clamp_min(0)
    k_ss = gaussian_kernel(f_s, f_s).sum() - f_s.size(0)
    k_tt = gaussian_kernel(f_t, f_t).sum() - f_t.size(0)
    k_st = gaussian_kernel(f_s, f_t).sum()
    return (k_ss / (m * (m - 1)) + k_tt / (n * (n - 1)) - 2 * k_st / (m * n)).clamp_min(0)


def conditional_mmd_loss(f_s, y_s, f_t, logits_t, mode="conditional", estimator="biased"):
    """按源域标签与目标域伪标签对齐（版本3 §2 L3）。

    mode='global'：整体 MMD；'conditional'：probs[:,c]>=0.5 硬掩膜逐类对齐；
    'confidence'：softmax 概率软加权逐类对齐。
    """
    if mode == "global":
        return mmd_loss(f_s, f_t, estimator=estimator)
    probs = torch.softmax(logits_t, dim=1)
    n_classes = logits_t.size(1)
    terms, masses = [], []
    for c in range(n_classes):
        src_mask = y_s == c
        if src_mask.sum() < 2:
            continue
        fs_c = f_s[src_mask]
        if mode == "conditional" and len(f_t) < 2:
            continue
        if mode == "conditional":
            tgt_mask = probs[:, c].detach() >= 0.5
            if tgt_mask.sum() < 2:
                continue
            ft_c = f_t[tgt_mask]
            terms.append(mmd_loss(fs_c, ft_c, estimator=estimator))
            masses.append(float(tgt_mask.sum()))
        elif mode == "confidence":
            w = probs[:, c].detach()
            if float(w.sum()) < 1e-6:
                continue
            ws = torch.full((len(fs_c),), 1.0 / len(fs_c), device=f_s.device)
            wt = w / w.sum()
            k_ss = gaussian_kernel(fs_c, fs_c)
            k_tt = gaussian_kernel(f_t, f_t)
            k_st = gaussian_kernel(fs_c, f_t)
            if estimator == "biased":
                term = (ws[:, None] * ws[None, :] * k_ss).sum()
                term = term + (wt[:, None] * wt[None, :] * k_tt).sum()
                term = term - 2 * (ws[:, None] * wt[None, :] * k_st).sum()
            elif estimator == "unbiased":
                eye_s = torch.eye(len(ws), device=ws.device, dtype=ws.dtype)
                eye_t = torch.eye(len(wt), device=wt.device, dtype=wt.dtype)
                den_s = 1 - ws.square().sum()
                den_t = 1 - wt.square().sum()
                if den_s.item() <= 1e-6 or den_t.item() <= 1e-6:
                    continue
                ss = ((ws[:, None] * ws[None, :] * k_ss * (1 - eye_s)).sum() / den_s)
                tt = ((wt[:, None] * wt[None, :] * k_tt * (1 - eye_t)).sum() / den_t)
                st = (ws[:, None] * wt[None, :] * k_st).sum()
                term = ss + tt - 2 * st
            else:
                raise ValueError(f"未知 MMD estimator: {estimator}")
            terms.append(term.clamp_min(0))
            masses.append(float(w.sum()))
        else:
            raise ValueError(f"未知对齐模式: {mode}")
    if not terms:
        return mmd_loss(f_s, f_t, estimator=estimator)
    weights = f_s.new_tensor(masses)
    return torch.stack(terms).mul(weights / weights.sum()).sum()


def _self_check():
    """验证双 frontend 前向、logspec256 与 (2,4,4) 等价、注意力归一、MMD 三模式、小 batch。"""
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # spectral_v1：logspec (1,256)、cepstrum (1,236)
    for n_ch, dim in ((1, 256), (1, 236)):
        m = MSRSABiLSTM(n_channels=n_ch, frontend="spectral_v1").to(dev)
        x = torch.randn(4, n_ch, dim, device=dev)
        logits, alpha = m(x)
        assert logits.shape == (4, 4) and alpha.shape == (4, 8), (dim, alpha.shape)
        assert torch.allclose(alpha.sum(1), torch.ones(4, device=dev), atol=1e-5)

    # legacy_vmd：(8,512)
    m = MSRSABiLSTM(n_channels=8, frontend="legacy_vmd").to(dev)
    x = torch.randn(4, 8, 512, device=dev)
    logits, alpha, feat = m(x, return_feat=True)
    assert feat.shape == (4, 256) and alpha.shape == (4, 8)

    # spectral_v1 在 logspec256 下 (2,4)+自适应8 与固定 (2,4,4) 逐点等价
    fe = ConvFrontEnd(c_in=1, frontend="spectral_v1").eval()
    x = torch.randn(2, 1, 256)
    with torch.no_grad():
        # 手工 (2,4,4)：把末级 AdaptiveMaxPool1d(8) 换成 MaxPool1d(4)，输入 256→128→32→8
        y_adaptive = fe(x)
        # 复制权重到固定池化版本
        fe_fixed = ConvFrontEnd(c_in=1, frontend="spectral_v1")
        fe_fixed.load_state_dict(fe.state_dict())
        fe_fixed.pools[2] = nn.MaxPool1d(4)
        fe_fixed.eval()
        y_fixed = fe_fixed(x)
    assert torch.allclose(y_adaptive, y_fixed, atol=1e-6), \
        "logspec256 下 AdaptiveMaxPool1d(8) 应与 MaxPool1d(4) 等价"

    # MMD 三模式 × 两估计器有限值 + 小 batch 保护
    fs, ft = torch.randn(16, 32), torch.randn(16, 32) + 0.5
    ys = torch.arange(16) % 4
    lt = torch.randn(16, 4)
    for mode in ("global", "conditional", "confidence"):
        for est in ("unbiased", "biased"):
            assert torch.isfinite(conditional_mmd_loss(fs, ys, ft, lt, mode=mode, estimator=est))
            assert torch.isfinite(conditional_mmd_loss(
                torch.randn(1, 32), torch.zeros(1, dtype=torch.long),
                torch.randn(1, 32), torch.randn(1, 4), mode=mode, estimator=est))
    print("ALL MODEL CHECKS PASSED")


if __name__ == "__main__":
    _self_check()
