# -*- coding: utf-8 -*-
"""train.py 参数校验测试：AdaBN 主通路开关与协议约束（版本3 §5 P0.1）。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train  # noqa: E402


class TestBnAdaptValidation(unittest.TestCase):
    """--bn_adapt 只支持 transductive，且与 target_only 互斥。"""

    def setUp(self):
        self.ap = train.build_parser()

    def test_transductive_ok(self):
        args = self.ap.parse_args(["--bn_adapt"])
        train.validate_args(args)
        self.assertTrue(args.bn_adapt)
        self.assertEqual(args.bn_adapt_scope, "all")
        self.assertEqual(args.bn_adapt_passes, 1)

    def test_inductive_rejected(self):
        args = self.ap.parse_args(["--bn_adapt", "--protocol", "inductive"])
        with self.assertRaises(SystemExit):
            train.validate_args(args)

    def test_target_only_rejected(self):
        args = self.ap.parse_args(["--bn_adapt", "--control", "target_only",
                                   "--label_ratio", "0.5"])
        with self.assertRaises(SystemExit):
            train.validate_args(args)

    def test_nonpositive_passes_rejected(self):
        args = self.ap.parse_args(["--bn_adapt", "--bn_adapt_passes", "0"])
        with self.assertRaises(SystemExit):
            train.validate_args(args)

    def test_off_by_default(self):
        args = self.ap.parse_args([])
        train.validate_args(args)
        self.assertFalse(args.bn_adapt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
