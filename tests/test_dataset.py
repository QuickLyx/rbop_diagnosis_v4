# -*- coding: utf-8 -*-
"""dataset.py 测试：段统计、paired 语义、划分等价（版本3 §4.2、§4.7）。

划分等价是 W0 硬门槛：v4 复算全 128 run 的 labeled/eval/label_pool/unused 键，
必须与旧库 `../rbop_diagnosis/runs/paired/results.json` 0 不一致。旧库结果不存在时
（如脱离原始工作目录）该用例自动跳过，其余不变量仍受测试保护。
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataset as ds  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(ROOT, "Data")
OLD_RESULTS = os.path.join(ROOT, "..", "rbop_diagnosis", "runs", "paired", "results.json")
_HAS_DATA = os.path.isdir(os.path.join(DATA_ROOT, "12k_DE"))


@unittest.skipUnless(_HAS_DATA, "缺少 Data/ 目录")
class TestEnumerateAndSegments(unittest.TestCase):
    """文件枚举与分段的基本不变量（版本3 §4.2 不变量 1–3）。"""

    @classmethod
    def setUpClass(cls):
        cls.records = ds.enumerate_t1_files(DATA_ROOT)

    def test_56_records_four_classes(self):
        self.assertEqual(len(self.records), 56)
        by_key = {r["file_key"]: r for r in self.records}
        counts = ds._count_by_label([r["file_key"] for r in self.records], by_key)
        self.assertEqual(counts, {0: 4, 1: 12, 2: 12, 3: 28})

    def test_no_028_and_pairing(self):
        for r in self.records:
            self.assertFalse(ds.is_028_file(r["file_key"]))

    def test_de_segments_13191(self):
        seg = ds.build_t1_segments(DATA_ROOT, "DE", 512, records=self.records)
        self.assertEqual(len(seg), 13191)
        self.assertEqual(seg.fs, 12000)
        self.assertEqual(seg.signals.shape[1], 512)

    def test_fe_segments_present_and_paired(self):
        seg = ds.build_t1_segments(DATA_ROOT, "FE", 512, records=self.records)
        self.assertGreater(len(seg), 0)
        # 每个文件键在 DE/FE 都应有段
        de = ds.build_t1_segments(DATA_ROOT, "DE", 512, records=self.records)
        self.assertEqual(set(de.file_keys), set(seg.file_keys))


@unittest.skipUnless(_HAS_DATA, "缺少 Data/ 目录")
class TestPairedSemantics(unittest.TestCase):
    """paired 两级划分语义（版本3 §4.2 不变量 7）。"""

    @classmethod
    def setUpClass(cls):
        cls.records = ds.enumerate_t1_files(DATA_ROOT)

    def test_pool_complement_is_eval(self):
        split = ds.split_target_files(
            self.records, label_ratio=0.99, protocol="transductive",
            seed=100003, split_mode="paired", label_pool_ratio=0.5)
        pool = set(split["label_pool"])
        ev = set(split["eval"])
        self.assertFalse(pool & ev, "标签池与评估集不得重叠")
        self.assertEqual(pool | ev, {r["file_key"] for r in self.records})

    def test_lr099_takes_full_pool(self):
        split = ds.split_target_files(
            self.records, label_ratio=0.99, protocol="transductive",
            seed=100003, split_mode="paired", label_pool_ratio=0.5)
        self.assertEqual(set(split["labeled"]), set(split["label_pool"]))
        self.assertEqual(split["unused"], [])

    def test_partial_ratio_has_unused(self):
        split = ds.split_target_files(
            self.records, label_ratio=0.3, protocol="transductive",
            seed=100003, split_mode="paired", label_pool_ratio=0.5)
        # 标签文件 + unused 应恰好构成标签池
        self.assertEqual(set(split["labeled"]) | set(split["unused"]),
                         set(split["label_pool"]))
        self.assertFalse(set(split["labeled"]) & set(split["eval"]))

    def test_natural_rejects_pool_ratio(self):
        with self.assertRaises(ValueError):
            ds.split_target_files(self.records, 0.1, "transductive", seed=0,
                                  label_pool_ratio=0.5)

    def test_labels_never_leak(self):
        split = ds.split_target_files(
            self.records, label_ratio=0.6, protocol="transductive",
            seed=100003, split_mode="paired", label_pool_ratio=0.5)
        self.assertFalse(set(split["labeled"]) & set(split["eval"]))


@unittest.skipUnless(_HAS_DATA and os.path.exists(OLD_RESULTS),
                     "缺少 Data/ 或旧库 runs/paired/results.json")
class TestSplitEquivalence128Run(unittest.TestCase):
    """W0 硬门槛：全 128 run 划分键与旧库 0 不一致（版本3 §4.2 划分等价）。"""

    def test_all_128_runs_equivalent(self):
        rows = json.load(open(OLD_RESULTS, encoding="utf-8"))
        records = ds.enumerate_t1_files(DATA_ROOT)
        self.assertEqual(len(rows), 128, "旧库 paired run 数应为 128")
        mismatch = []
        for r in rows:
            split = ds.split_target_files(
                records, label_ratio=r["label_ratio"], protocol="transductive",
                seed=r["seed"] + 100003, split_mode="paired",
                label_pool_ratio=r["label_pool_ratio"])
            for field, key in (("target_labeled_keys", "labeled"),
                               ("target_eval_keys", "eval"),
                               ("target_label_pool_keys", "label_pool"),
                               ("target_unused_keys", "unused")):
                if sorted(r.get(field) or []) != sorted(split.get(key) or []):
                    mismatch.append((r["run"], field))
        self.assertEqual(mismatch, [], f"划分不一致：{mismatch[:5]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
