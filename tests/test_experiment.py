# -*- coding: utf-8 -*-
"""experiment.py 测试：run 名唯一、结果字段完整、目录不覆盖、分模式隔离（版本3 §4.7）。"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import experiment as ex  # noqa: E402


class TestRunNaming(unittest.TestCase):
    def test_paired_name_with_repr_and_tag(self):
        name = ex.default_run_name("T1", 0, "transductive", 0.99, 0.3, "semi_supervised",
                                   split_mode="paired", label_pool_ratio=0.5,
                                   seg_len=512, repr_kind="logspec", tag="pilot")
        self.assertEqual(
            name,
            "T1_s0_paired_transductive_poolr0.5_lr0.99_lam0.3_semi_supervised_L512_logspec_pilot")

    def test_natural_name_no_pool(self):
        name = ex.default_run_name("T1", 1, "transductive", 0.1, 0.0, "target_only",
                                   split_mode="natural", seg_len=512, repr_kind="vmd")
        self.assertEqual(name, "T1_s1_natural_transductive_lr0.1_lam0_target_only_L512_vmd")


class TestResultSchema(unittest.TestCase):
    def test_fields_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = ex.append_result({"run": "x", "task": "T1", "split_mode": "paired",
                                     "transfer_acc": 0.9}, runs_dir=tmp)
            rows = json.load(open(path, encoding="utf-8"))
            self.assertEqual(set(rows[0]), set(ex.RESULT_FIELDS))

    def test_split_mode_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            ex.append_result({"run": "a", "split_mode": "paired"}, runs_dir=tmp)
            with self.assertRaises(SystemExit):
                ex.append_result({"run": "b", "split_mode": "natural"}, runs_dir=tmp)


class TestDirNoOverwrite(unittest.TestCase):
    def test_reject_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            ex.create_run_directory("R1", tmp)
            with self.assertRaises(SystemExit):
                ex.create_run_directory("R1", tmp)


class _Args:
    """构造最小 args 桩，只为测 _resolve_arm 与 _maybe_adabn。"""
    lr_transfer = 1e-4
    bn_adapt = False
    bn_adapt_scope = "all"
    bn_adapt_passes = 1


class TestResolveArm(unittest.TestCase):
    """P2.2 臂映射（版本3 §5 P2.2）。"""

    def test_arm_mapping(self):
        a = _Args()
        self.assertEqual(ex._resolve_arm(None, a),
                         {"freeze_backbone": False, "source_ce": True,
                          "source_ce_weight": 1.0, "lr_transfer": 1e-4})
        self.assertTrue(ex._resolve_arm("B", a)["freeze_backbone"])
        self.assertFalse(ex._resolve_arm("C", a)["source_ce"])
        self.assertEqual(ex._resolve_arm("D0.1", a)["source_ce_weight"], 0.1)
        self.assertEqual(ex._resolve_arm("D0.3", a)["source_ce_weight"], 0.3)
        self.assertEqual(ex._resolve_arm("E", a)["lr_transfer"], 1e-3)


class TestMaybeAdabn(unittest.TestCase):
    """AdaBN 主通路开关：未启用返回 (False,None,None)（版本3 §5 P0.1）。"""

    def test_off_returns_none(self):
        a = _Args()
        a.bn_adapt = False
        self.assertEqual(ex._maybe_adabn(a, None, {"X_unlab": None}),
                         (False, None, None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
