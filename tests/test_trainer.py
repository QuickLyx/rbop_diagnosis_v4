# -*- coding: utf-8 -*-
"""trainer.py 测试：checkpoint round-trip 与 schema 拒绝、零标签零 MMD、AdaBN、记录级（版本3 §4.7）。"""
import os
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import trainer as tr  # noqa: E402
import preprocessing as pp  # noqa: E402
from model import MSRSABiLSTM  # noqa: E402


def _ck_kwargs():
    return dict(stage="source", source=None, target=None, seg_len=512,
                input_repr="logspec", repr_config=pp.build_repr_config("logspec"),
                n_channels=1, input_dim=256, n_classes=4,
                class_names=["Normal", "Ball", "InnerRace", "OuterRace"],
                recurrent_layers=1, threshold_mode="signed_mean", use_se=True,
                frontend={"name": "spectral_v1", "channels": [32, 64, 128],
                          "adaptive_out": 8})


class TestCheckpoint(unittest.TestCase):
    def test_round_trip_and_reject(self):
        tr.seed_everything(0)
        model = MSRSABiLSTM(n_channels=1, frontend="spectral_v1").to(tr.DEV)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "source.pt")
            tr.save_checkpoint(path, model, **_ck_kwargs())
            conf = tr.load_checkpoint_config(path)
            self.assertEqual(conf["schema_version"], 1)
            self.assertEqual(conf["input_repr"], "logspec")
            blob = tr.load_checkpoint(path, model)
            self.assertEqual(blob["frontend"]["name"], "spectral_v1")
            # 篡改 schema_version 应被拒
            bad = torch.load(path, weights_only=False)
            bad["schema_version"] = 2
            torch.save(bad, path)
            with self.assertRaises(SystemExit):
                tr.load_checkpoint(path, model)


class TestTrainPaths(unittest.TestCase):
    def test_zero_label_zero_mmd(self):
        tr.seed_everything(0)
        model = MSRSABiLSTM(n_channels=1, frontend="spectral_v1").to(tr.DEV)
        X = torch.randn(16, 1, 256)
        y = torch.arange(16) % 4
        # lam=0、无标签：应正常跑完
        tr.train_transfer(model, X, y, None, None, None, epochs=1, bs=8, lam=0.0)

    def test_record_level_metrics(self):
        tr.seed_everything(0)
        model = MSRSABiLSTM(n_channels=1, frontend="spectral_v1").to(tr.DEV)
        X = torch.randn(16, 1, 256)
        y = torch.arange(16) % 4
        rid = np.arange(16) // 4
        res = tr.evaluate_details(model, X, y, record_ids=rid)
        self.assertIn("record_accuracy_hard", res)
        self.assertIn("record_accuracy_soft", res)
        self.assertEqual(res["record_n_total"], 4)

    def test_adabn_nonneg_var(self):
        tr.seed_everything(0)
        model = MSRSABiLSTM(n_channels=1, frontend="spectral_v1").to(tr.DEV)
        X = torch.randn(20, 1, 256)
        tr.adabn_recompute(model, X, scope="all", chunk=8)
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d):
                self.assertTrue(torch.all(m.running_var >= 0))

    def test_adabn_frontend_scope(self):
        tr.seed_everything(0)
        model = MSRSABiLSTM(n_channels=1, frontend="spectral_v1").to(tr.DEV)
        X = torch.randn(20, 1, 256)
        # 记录前端 BN 与非前端 BN 的初始 running_mean
        before = {n: m.running_mean.clone() for n, m in model.named_modules()
                  if isinstance(m, nn.BatchNorm1d)}
        tr.adabn_recompute(model, X, scope="frontend", chunk=8)
        for n, m in model.named_modules():
            if isinstance(m, nn.BatchNorm1d):
                changed = not torch.equal(before[n], m.running_mean)
                if n.startswith("frontend"):
                    self.assertTrue(changed, f"前端 BN {n} 应被重估")


if __name__ == "__main__":
    unittest.main(verbosity=2)
