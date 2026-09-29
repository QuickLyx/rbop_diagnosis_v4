# -*- coding: utf-8 -*-
"""model.py 测试：双 frontend 前向、注意力归一、MMD 三模式、小 batch（版本3 §4.7）。"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model as M  # noqa: E402


class TestForward(unittest.TestCase):
    def test_spectral_v1_logspec_cepstrum(self):
        for n_ch, dim in ((1, 256), (1, 236)):
            m = M.MSRSABiLSTM(n_channels=n_ch, frontend="spectral_v1")
            logits, alpha = m(torch.randn(4, n_ch, dim))
            self.assertEqual(logits.shape, (4, 4))
            self.assertEqual(alpha.shape, (4, 8))
            self.assertTrue(torch.allclose(alpha.sum(1), torch.ones(4), atol=1e-5))

    def test_legacy_vmd(self):
        m = M.MSRSABiLSTM(n_channels=8, frontend="legacy_vmd")
        logits, alpha, feat = m(torch.randn(4, 8, 512), return_feat=True)
        self.assertEqual(feat.shape, (4, 256))
        self.assertEqual(alpha.shape, (4, 8))

    def test_spectral_v1_equivalence_on_256(self):
        """logspec256 下 AdaptiveMaxPool1d(8) 与 MaxPool1d(4) 逐点等价（版本3 §4.4）。"""
        import torch.nn as nn
        fe = M.ConvFrontEnd(c_in=1, frontend="spectral_v1").eval()
        x = torch.randn(2, 1, 256)
        with torch.no_grad():
            y_adaptive = fe(x)
            fe_fixed = M.ConvFrontEnd(c_in=1, frontend="spectral_v1")
            fe_fixed.load_state_dict(fe.state_dict())
            fe_fixed.pools[2] = nn.MaxPool1d(4)
            fe_fixed.eval()
            y_fixed = fe_fixed(x)
        self.assertTrue(torch.allclose(y_adaptive, y_fixed, atol=1e-6))

    def test_seven_batchnorms(self):
        """全网络 7 个 BatchNorm1d（前端 3 + RSU 2×2；版本3 §5 P0.1）。"""
        import torch.nn as nn
        m = M.MSRSABiLSTM(n_channels=1, frontend="spectral_v1")
        n_bn = sum(1 for mod in m.modules() if isinstance(mod, nn.BatchNorm1d))
        self.assertEqual(n_bn, 7)


class TestMMD(unittest.TestCase):
    def test_modes_finite(self):
        fs, ft = torch.randn(16, 32), torch.randn(16, 32) + 0.5
        ys = torch.arange(16) % 4
        lt = torch.randn(16, 4)
        for mode in ("global", "conditional", "confidence"):
            for est in ("unbiased", "biased"):
                self.assertTrue(torch.isfinite(
                    M.conditional_mmd_loss(fs, ys, ft, lt, mode=mode, estimator=est)))

    def test_small_batch(self):
        for mode in ("global", "conditional", "confidence"):
            for est in ("unbiased", "biased"):
                self.assertTrue(torch.isfinite(M.conditional_mmd_loss(
                    torch.randn(1, 32), torch.zeros(1, dtype=torch.long),
                    torch.randn(1, 32), torch.randn(1, 4), mode=mode, estimator=est)))

    def test_mmd_same_lt_diff(self):
        a = torch.randn(64, 32)
        b = torch.randn(64, 32)
        c = torch.randn(64, 32) + 3.0
        self.assertLess(M.mmd_loss(a, b).item(), M.mmd_loss(a, c).item())

    def test_gaussian_kernel_default_is_fixed_bandwidth(self):
        """默认 sigma=(1,5,10) 固定多核，与显式传入逐位一致（冻结配方，勿改）。"""
        x, y = torch.randn(8, 16), torch.randn(8, 16)
        k_def = M.gaussian_kernel(x, y)
        k_fix = M.gaussian_kernel(x, y, sigma=(1, 5, 10))
        self.assertTrue(torch.allclose(k_def, k_fix))
        self.assertEqual(k_def.shape, (8, 8))
        self.assertTrue(bool((k_def > 0).all() and (k_def <= 1).all()))

    def test_gaussian_kernel_median_option_and_reject(self):
        """sigma='median' 作对照入口可用；未知选项报错。"""
        x, y = torch.randn(8, 16), torch.randn(8, 16)
        k_med = M.gaussian_kernel(x, y, sigma="median")
        self.assertEqual(k_med.shape, (8, 8))
        self.assertTrue(torch.isfinite(k_med).all())
        # 对角线 k(x,x)=exp(0)=1（unbiased 减 count 的前提）
        diag = M.gaussian_kernel(x, x, sigma="median")
        self.assertTrue(torch.allclose(torch.diag(diag), torch.ones(8), atol=1e-6))
        with self.assertRaises(ValueError):
            M.gaussian_kernel(x, y, sigma="rbf")


if __name__ == "__main__":
    unittest.main(verbosity=2)
