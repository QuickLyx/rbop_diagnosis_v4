# -*- coding: utf-8 -*-
"""AdaBN 评估：对已有 run 的 source/transfer ckpt 用无标签目标输入重估 BN，成对报告。

设计依据：版本3 §5 P0.1（分块精确统计）、§8 M2（AdaBN 配对增量）。
transductive 使用评估记录的 FE 无标签输入（版本3 §0）。

用法：
  python scripts/adabn_eval.py --run runs/paired/<run_name> --ckpt source.pt \
      --scope all --stats_source adapt --passes 1 3 10
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import dataset as ds  # noqa: E402
import preprocessing as pp  # noqa: E402
import trainer as tr  # noqa: E402
from model import MSRSABiLSTM  # noqa: E402


def build_parser():
    ap = argparse.ArgumentParser(description="AdaBN 目标域 BN 重估评估（版本3 §5 P0.1）")
    ap.add_argument("--run", required=True, help="run 目录（含 ckpt 与 run 配置）")
    ap.add_argument("--ckpt", default="source.pt", choices=["source.pt", "transfer.pt"])
    ap.add_argument("--scope", nargs="+", choices=["all", "frontend"], default=["all"])
    ap.add_argument("--stats_source", nargs="+", choices=["adapt", "all_target"],
                    default=["adapt"])
    ap.add_argument("--passes", type=int, nargs="+", default=[1, 3, 10])
    ap.add_argument("--data_root", default=pp.DATA_DIR)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--chunk", type=int, default=2048)
    return ap


def _load_run_config(run_dir, ckpt_name):
    """从 ckpt 元数据恢复表征/frontend 配置。"""
    path = os.path.join(run_dir, ckpt_name)
    if not os.path.exists(path):
        raise SystemExit(f"找不到 ckpt: {path}")
    return tr.load_checkpoint_config(path), path


def main():
    args = build_parser().parse_args()
    conf, ckpt_path = _load_run_config(args.run, args.ckpt)
    cfg = conf["repr_config"]
    repr_kind = conf["input_repr"]
    frontend = conf["frontend"]["name"]
    # 从 run 名解析 seed/pool/lr 以复现划分
    import json
    results = None
    for sm in ("paired", "natural"):
        rp = os.path.join(ROOT, "runs", sm, "results.json")
        if os.path.exists(rp):
            rows = json.load(open(rp, encoding="utf-8"))
            for r in rows:
                if os.path.join(ROOT, "runs", sm, r["run"]) == os.path.abspath(args.run) \
                   or r["run"] == os.path.basename(args.run):
                    results = r
                    break
    if results is None:
        raise SystemExit("无法在 results.json 找到该 run 的划分配置")

    records = ds.enumerate_t1_files(args.data_root)
    source_keys = sorted(r["file_key"] for r in records)
    target_set = ds.build_t1_segments(args.data_root, "FE", conf["seg_len"],
                                      file_keys=source_keys, records=records)
    cache_dir = args.cache_dir or pp.CACHE_DIR
    Xt, yt, rows_idx = pp.build_feature_set(target_set, source_keys, "FE", cfg,
                                            cache_dir=cache_dir)
    eval_keys = results["target_eval_keys"]
    eval_rows = target_set.rows(eval_keys)
    Xt_eval, yt_eval = pp.take_rows(Xt, yt, rows_idx, eval_rows)
    # 记录级 id
    key_to_idx = {k: i for i, k in enumerate(eval_keys)}
    eval_rid = []
    for k in eval_keys:
        span = target_set.span(k)
        eval_rid.extend([key_to_idx[k]] * (span.stop - span.start))
    eval_rid = np.array(eval_rid)

    # 统计来源
    adapt_rows = target_set.rows(results["target_adapt_keys"])
    X_adapt = pp.take_rows(Xt, yt, rows_idx, adapt_rows)[0]
    X_all_target = Xt

    def fresh_model():
        m = MSRSABiLSTM(n_channels=conf["n_channels"], n_classes=conf["n_classes"],
                        frontend=frontend, lstm_layers=conf["recurrent_layers"],
                        use_se=conf["use_se"], threshold_mode=conf["threshold_mode"]).to(tr.DEV)
        tr.load_checkpoint(ckpt_path, m)
        return m

    # 重估前
    base = tr.evaluate_details(fresh_model(), Xt_eval, yt_eval, record_ids=eval_rid)
    print(f"# AdaBN 评估  run={os.path.basename(args.run)} ckpt={args.ckpt} "
          f"repr={repr_kind} frontend={frontend}")
    print(f"[重估前] 段级={base['accuracy']:.4f} 记录硬={base['record_accuracy_hard']:.4f}")

    for scope in args.scope:
        for src in args.stats_source:
            X_stat = X_adapt if src == "adapt" else X_all_target
            for passes in args.passes:
                m = fresh_model()
                for _ in range(passes):
                    tr.adabn_recompute(m, X_stat, scope=scope, chunk=args.chunk)
                res = tr.evaluate_details(m, Xt_eval, yt_eval, record_ids=eval_rid)
                print(f"[重估后] scope={scope} stats={src} passes={passes:<2} "
                      f"段级={res['accuracy']:.4f} (Δ{res['accuracy']-base['accuracy']:+.4f}) "
                      f"记录硬={res['record_accuracy_hard']:.4f}")


if __name__ == "__main__":
    main()
