# -*- coding: utf-8 -*-
"""T1 结果汇总：配对差值 + Holm step-down + 按类分层 cluster bootstrap。

设计依据：版本3 §7（同 seed 配对差值、Holm、cluster bootstrap）、§P1.3、§8（验收）。
用 row.get 容忍缺省字段（版本3 §4.5）。natural/paired 分读，不混表。
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_rows(split_mode):
    path = os.path.join(ROOT, "runs", split_mode, "results.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def paired_diff(rows_a, rows_b, key="transfer_acc"):
    """同 seed 配对差值（右-左）；按 seed 对齐（版本3 §7）。返回 (diffs, seeds)。"""
    by_seed_a = {r["seed"]: r for r in rows_a}
    by_seed_b = {r["seed"]: r for r in rows_b}
    seeds = sorted(set(by_seed_a) & set(by_seed_b))
    diffs = [by_seed_b[s].get(key) - by_seed_a[s].get(key) for s in seeds]
    return np.array(diffs, dtype=float), seeds


def mean_sd(values):
    arr = np.asarray(values, dtype=float)
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return float(arr.mean()), sd


def paired_t_p(diffs):
    """配对 t 检验的双侧 p（探索性，n=4/8）。"""
    from scipy import stats
    if len(diffs) < 2 or np.allclose(diffs, diffs[0]):
        return float("nan")
    t, p = stats.ttest_1samp(diffs, 0.0)
    return float(p)


def holm(pvals, names):
    """Holm step-down 校正，返回 [(name, p, p_adj, reject@0.05)]（版本3 §7）。"""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return [(names[i], pvals[i], adj[i], adj[i] < 0.05) for i in range(m)]


def cluster_bootstrap_ci(rows, key="transfer_acc", n_boot=2000, seed=0):
    """按类分层的按记录 cluster bootstrap CI（版本3 §P1.3）。

    这里按 seed 级指标做简化 bootstrap（seed 作为聚类单元），给出均值 95% CI。
    需要向"总体记录"外推时使用；Normal 仅 2 条、离散度大需标注。
    """
    vals = np.array([r.get(key) for r in rows if r.get(key) is not None], dtype=float)
    if len(vals) < 2:
        return None
    rng = np.random.default_rng(seed)
    boots = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def summarize_grid(rows, group_keys=("input_repr", "control", "label_ratio", "lam")):
    """按配置分组打印段级/记录级均值±SD（版本3 §8）。"""
    groups = {}
    for r in rows:
        k = tuple(r.get(g) for g in group_keys)
        groups.setdefault(k, []).append(r)
    print(f"\n{'配置':<52} n  段级均值±SD      记录硬均值   macro-F1")
    for k in sorted(groups, key=lambda x: tuple(str(v) for v in x)):
        items = groups[k]
        acc_m, acc_s = mean_sd([r.get("transfer_acc") for r in items
                                if r.get("transfer_acc") is not None])
        rec = [r.get("record_accuracy_hard") for r in items
               if r.get("record_accuracy_hard") is not None]
        rec_m = mean_sd(rec)[0] if rec else float("nan")
        f1 = [r.get("macro_f1") for r in items if r.get("macro_f1") is not None]
        f1_m = mean_sd(f1)[0] if f1 else float("nan")
        label = "|".join(str(v) for v in k)
        print(f"{label:<52} {len(items):<2} {acc_m:.4f}±{acc_s:.4f}   "
              f"{rec_m:.4f}      {f1_m:.4f}")


def _self_check():
    """验证配对差值、Holm、bootstrap 的基本正确性。"""
    a = [{"seed": s, "transfer_acc": 0.80 + 0.01 * s} for s in range(4)]
    b = [{"seed": s, "transfer_acc": 0.85 + 0.01 * s} for s in range(4)]
    diffs, seeds = paired_diff(a, b)
    assert np.allclose(diffs, 0.05) and seeds == [0, 1, 2, 3]
    res = holm([0.01, 0.04, 0.03], ["x", "y", "z"])
    assert len(res) == 3 and res[0][2] >= res[0][1]
    ci = cluster_bootstrap_ci(b)
    assert ci is not None and ci[0] <= ci[1]
    print("ALL SUMMARIZE CHECKS PASSED")


def main():
    ap = argparse.ArgumentParser(description="T1 结果汇总（配对差值/Holm/bootstrap）")
    ap.add_argument("--split_mode", choices=["natural", "paired"], default="paired")
    ap.add_argument("--self_check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        _self_check()
        return
    rows = load_rows(args.split_mode)
    if not rows:
        print(f"[空] runs/{args.split_mode}/results.json 无数据")
        return
    print(f"[汇总] split_mode={args.split_mode}  共 {len(rows)} run")
    summarize_grid(rows)


if __name__ == "__main__":
    main()
