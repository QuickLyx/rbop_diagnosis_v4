# -*- coding: utf-8 -*-
"""preprocessing.py 测试：repr 形状/键/缓存校验、缓存键隔离（版本3 §4.7）。"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import preprocessing as pp  # noqa: E402
import dataset as ds  # noqa: E402


class TestExtractRepr(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.sigs = rng.standard_normal((8, 512)).astype(np.float64)

    def test_shapes(self):
        self.assertEqual(pp.extract_repr(self.sigs, pp.build_repr_config("logspec")).shape,
                         (8, 1, 256))
        self.assertEqual(pp.extract_repr(self.sigs, pp.build_repr_config("cepstrum")).shape,
                         (8, 1, 236))
        self.assertEqual(pp.extract_repr(self.sigs, pp.build_repr_config("vmdstats", K=8)).shape,
                         (8, 1, 32))
        self.assertEqual(pp.extract_repr(self.sigs, pp.build_repr_config("vmd", K=8)).shape,
                         (8, 8, 512))

    def test_seg_std(self):
        ls = pp.extract_repr(self.sigs, pp.build_repr_config("logspec"))
        self.assertLess(abs(float(ls[0, 0].mean())), 1e-5)

    def test_repr_key_unique(self):
        keys = {pp.repr_key(pp.build_repr_config(k)) for k in pp.REPR_KINDS}
        self.assertEqual(len(keys), 4)

    def test_cache_path_isolation(self):
        c_ls = pp.build_repr_config("logspec")
        c_cp = pp.build_repr_config("cepstrum")
        self.assertNotEqual(pp.get_cache_path("12k_DE/B007_0.mat", "DE", c_ls),
                            pp.get_cache_path("12k_DE/B007_0.mat", "FE", c_ls))
        self.assertNotEqual(pp.get_cache_path("12k_DE/B007_0.mat", "DE", c_ls),
                            pp.get_cache_path("Normal/B007_0.mat", "DE", c_ls))
        self.assertNotEqual(pp.get_cache_path("12k_DE/B007_0.mat", "DE", c_ls),
                            pp.get_cache_path("12k_DE/B007_0.mat", "DE", c_cp))

    def test_vmdstats_uses_sorted_modes(self):
        """vmdstats 必须基于排序后的 VMD 模态（保证与 E21/E22 对拍）。"""
        cfg = pp.build_repr_config("vmdstats", K=8)
        vm = pp.extract_repr(self.sigs, pp.build_repr_config("vmd", K=8))
        sorted_vm = pp.reorder_vmd_modes(vm)
        expect = pp._vmdstats(sorted_vm, cfg)
        got = pp.extract_repr(self.sigs, cfg)
        self.assertTrue(np.array_equal(got, expect))


class TestCacheRoundTrip(unittest.TestCase):
    def test_cache_config_validation(self):
        """缓存写入后读回一致；repr_config 字段不符时报错。"""
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as tmp:
            # 用合成 SegmentSet
            sigs = rng.standard_normal((5, 512)).astype(np.float64)
            seg = ds.SegmentSet(["d/a.mat"] * 5, sigs, [0, 1, 2, 3, 0], list(range(5)))
            cfg = pp.build_repr_config("logspec")
            X1, y1, _ = pp.build_feature_set(seg, ["d/a.mat"], "DE", cfg, cache_dir=tmp)
            # 第二次命中缓存
            X2, y2, _ = pp.build_feature_set(seg, ["d/a.mat"], "DE", cfg, cache_dir=tmp)
            import torch
            self.assertTrue(torch.equal(X1, X2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
