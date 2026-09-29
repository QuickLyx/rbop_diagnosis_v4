# -*- coding: utf-8 -*-
"""VMD 变分模态分解——T1 vmd/vmdstats 表征的底层信号算法。

设计依据：版本3 §4.3（vmd 表征）、§1.1（VMD 波形对线性模型不可读，需 CNN 重建）。
数值实现从旧库 `../rbop_diagnosis/vmd.py` 原样移植，须过版本3 §4.2 的 VMD 缓存等价审计
（抽样文件与旧库 npz allclose rtol=1e-5）。

数据流：分段信号 (L,) → 逐段归一化（preprocessing.normalize_segment）→ vmd(x,K=8) → (K,L)。
"""
import numpy as np


def vmd(x, K=8, alpha=2000.0, tau=0.0, tol=1e-7, max_iter=500):
    """变分模态分解（Dragomiretskiy & Zosso 2014 的精简 NumPy 实现）。

    在 rfft 正频率半谱上做交替方向乘子更新：
      u_k_hat ← (X - Σ_{j≠k} u_j_hat) / (1 + α·(f - ω_k)²)
      ω_k ← Σ f·|u_k_hat|² / Σ|u_k_hat|²
    收敛时 Σu_k ≈ x（窄带部分精确重构，宽带噪声留在残差里，天然降噪）。

    参数：x (n,) 信号；K 模态数（=vmd 表征通道数）；alpha 带宽惩罚（官方默认 2000）；
    tau 对偶噪声容忍（主流程 0）；tol 收敛阈值；max_iter 最大迭代。返回 (K, n) 模态矩阵。

    注意：行序不代表中心频率高低、不保证升序。需要有序通道时用
    `preprocessing.reorder_vmd_modes`（vmd 表征主线默认排序，见版本3 §4.3、E16）。
    """
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    X = np.fft.rfft(x)                 # 正频率半谱
    freqs = np.fft.rfftfreq(N)         # 归一化频率 [0, 0.5]
    u_hat = np.zeros((K, len(X)), dtype=complex)
    omega = np.arange(1, K + 1) / (2 * K)  # 中心频率均匀初始化
    sum_u = np.zeros_like(X)           # Σ u_j_hat 的滚动和
    for _ in range(max_iter):
        u_hat_old = u_hat.copy()
        for k in range(K):
            sum_k = sum_u - u_hat[k]   # 其他模态之和
            u_hat[k] = (X - sum_k) / (1.0 + alpha * (freqs - omega[k]) ** 2)
            p = np.abs(u_hat[k]) ** 2
            omega[k] = float(np.sum(freqs * p) / (np.sum(p) + 1e-16))
            sum_u = sum_k + u_hat[k]
        if np.sum(np.abs(u_hat - u_hat_old) ** 2) / max(np.sum(np.abs(u_hat_old) ** 2), 1e-16) < tol:
            break
    return np.array([np.fft.irfft(u_hat[k], n=N) for k in range(K)])


if __name__ == "__main__":
    fs = 12000
    t = np.arange(4096) / fs
    rng = np.random.default_rng(0)
    f_fault = 107.36  # BPFO @1797rpm
    impulses = np.zeros_like(t)
    for k in range(int(t[-1] * f_fault) + 1):
        idx = int(k / f_fault * fs)
        if idx < len(impulses):
            decay = np.exp(-300 * t[: len(t) - idx]) * np.sin(2 * np.pi * 3000 * t[: len(t) - idx])
            impulses[idx:] += decay
    x = impulses + 0.3 * rng.standard_normal(len(t))
    u = vmd(x, K=4)
    assert u.shape == (4, len(x)) and np.isfinite(u).all()
    resid = x - u.sum(0)
    print(f"[VMD] 4 modes, residual energy ratio = {np.sum(resid**2)/np.sum(x**2):.4f}")
    assert np.sum(resid**2) / np.sum(x**2) < 0.35, "VMD residual too large"
    print("ALL VMD CHECKS PASSED")
