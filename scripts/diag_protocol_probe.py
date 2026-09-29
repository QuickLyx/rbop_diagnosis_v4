# -*- coding: utf-8 -*-
"""E22 协议匹配探针：paired 划分口径下线性探针的段级/记录级上限（E21 的 v4 复现）。

设计依据：版本3 §1.2（E21/E22）、§5 P0.0（E22 复现判据）、§4.3（共享提取函数）。

关键（版本3 §4.3、执行清单 S4）：
- 表征提取**直接调用 preprocessing.extract_repr**，与网络共用同一实现，禁止第二份。
- 与 train.py 共用 dataset.split_target_files（含 seed+100003 规则），因此每个 seed 的
  有标签/评估文件集合与同 seed 的网络 run 完全一致；评估文件的 FE 标签从不进入训练。
- 只用 CPU、逻辑回归（C=0.3、StandardScaler），不训练网络。

对照（每 seed 每预算）：
  target-only / target-only+balanced / source+target / zero-label（不对齐 + 全FE统计 + eval统计一阶对齐）

判据：E22 与 E21 同 seed/预算对拍，段级差 ≤1pp、逐 seed 同号（版本3 §5 P0.0）。

用法：
  python scripts/diag_protocol_probe.py --split_mode paired --label_pool_ratio 0.5 \
      --label_ratios 0.3 0.6 0.99 --seeds 0 1 2 3 --reprs logspec cepstrum vmdstats \
      --keys_out docs/evidence/E22_protocol_probe_keys.json
  # 只导出全网格划分键、不跑探针（版本3 §12 --keys_only）：
  python scripts/diag_protocol_probe.py --split_mode paired --label_pool_ratio 0.3 \
      --label_ratios 0 0.3 0.6 0.99 --seeds 0 1 2 3 --reprs logspec --keys_only \
      --keys_out docs/evidence/E22_keys_pool0.3.json
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import dataset as ds  # noqa: E402
import preprocessing as pp  # noqa: E402

REPRS = ("logspec", "cepstrum", "vmdstats")


def build_features(args):
    """按文件顺序构建 DE / FE 表征和逐段文件号（版本3 §4.3，复用 preprocessing）。

    返回 `(records, feats, fids)`：feats[repr][channel] 为 (N_seg, dim)；
    fids[channel] 为 (N_seg,) 段→文件下标，行序与 enumerate_t1_files 一致。
    """
    records = ds.enumerate_t1_files(args.data_root)
    keys = [r["file_key"] for r in records]
    out = {name: {} for name in args.reprs}
    fids = {}

    def fid_of(seg):
        return np.concatenate([np.full(seg.span(k).stop - seg.span(k).start, i)
                               for i, k in enumerate(keys)])

    for channel in ("DE", "FE"):
        seg = ds.build_t1_segments(args.data_root, channel, args.L,
                                   file_keys=keys, records=records)
        fids[channel] = fid_of(seg)
        for name in args.reprs:
            cfg = pp.build_repr_config(name, seg_len=args.L, K=args.K, nbin=args.nbin)
            # extract_repr 返回 (N,C,D)；探针用展平的 (N, C*D)
            feat = pp.extract_repr(seg.signals, cfg)
            out[name][channel] = feat.reshape(feat.shape[0], -1).astype(np.float32)
    return records, out, fids


def record_votes(pred, prob, true, fid):
    """返回记录级硬投票与软投票正确率（记录等权，版本3 §P1.1）。"""
    hard, soft = [], []
    for i in np.unique(fid):
        mask = fid == i
        counts = np.bincount(pred[mask], minlength=prob.shape[1])
        hard.append(int(counts.argmax()) == int(true[mask][0]))
        soft.append(int(prob[mask].mean(0).argmax()) == int(true[mask][0]))
    return float(np.mean(hard)), float(np.mean(soft))


def probe(train_X, train_y, eval_X, eval_y, eval_fid, args, class_weight=None,
          align=None):
    """训练线性探针并返回段级/记录级指标；align=(mu_s,sd_s,mu_t,sd_t) 时先做一阶对齐。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    from sklearn.preprocessing import StandardScaler

    if align is not None:
        mu_s, sd_s, mu_t, sd_t = align
        eval_X = (eval_X - mu_t) / sd_t * sd_s + mu_s
    scaler = StandardScaler().fit(train_X)
    clf = LogisticRegression(max_iter=args.max_iter, C=args.C, class_weight=class_weight,
                             random_state=0).fit(scaler.transform(train_X), train_y)
    prob = clf.predict_proba(scaler.transform(eval_X))
    pred = prob.argmax(1)
    hard, soft = record_votes(pred, prob, eval_y, eval_fid)
    cm = confusion_matrix(eval_y, pred, labels=range(len(ds.LABEL_NAMES)))
    recall = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(4), where=cm.sum(1) != 0)
    return {"acc": float(accuracy_score(eval_y, pred)),
            "f1": float(f1_score(eval_y, pred, average="macro", zero_division=0)),
            "recall": recall.tolist(), "hard": hard, "soft": soft}


def fmt_metrics(name, m):
    return (f"  {name:<26} 段级acc={m['acc']:.4f} f1={m['f1']:.4f} "
            f"硬={m['hard']:.4f} 软={m['soft']:.4f}  "
            + " ".join(f"{ds.LABEL_NAMES[i][:3]}={m['recall'][i]:.2f}"
                       for i in range(len(ds.LABEL_NAMES))))


def mean_sd(values):
    """均值±样本SD；单元素时 SD 记 0。"""
    arr = np.asarray(values, dtype=float)
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return float(arr.mean()), sd


def collect_keys(args):
    """生成全网格划分键（版本3 §5 P0.0 全网格 keys JSON）。"""
    records = ds.enumerate_t1_files(args.data_root)
    key_blob = {}
    for ratio in args.label_ratios:
        for seed in args.seeds:
            split = ds.split_target_files(
                records, label_ratio=ratio, protocol="transductive",
                seed=seed + 100003, split_mode=args.split_mode,
                label_pool_ratio=(args.label_pool_ratio
                                  if args.split_mode == "paired" else None))
            key_blob[f"{args.split_mode}|pool{args.label_pool_ratio}|lr{ratio}|s{seed}"] = {
                "labeled": sorted(split["labeled"]),
                "eval": sorted(split["eval"]),
                "label_pool": sorted(split.get("label_pool") or []),
                "unused": sorted(split.get("unused") or []),
            }
    return key_blob


def main():
    ap = argparse.ArgumentParser(description="E22 协议匹配线性探针（CPU；E21 的 v4 复现）")
    ap.add_argument("--data_root", default=pp.DATA_DIR)
    ap.add_argument("--cache_dir", default=pp.CACHE_DIR)
    ap.add_argument("--split_mode", choices=["paired", "natural"], default="paired")
    ap.add_argument("--label_pool_ratio", type=float, default=0.5)
    ap.add_argument("--label_ratios", type=float, nargs="+", default=[0.3, 0.6, 0.99])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--reprs", nargs="+", choices=list(REPRS),
                    default=["logspec", "cepstrum", "vmdstats"])
    ap.add_argument("--L", type=int, default=512, help="段长（不重叠）")
    ap.add_argument("--K", type=int, default=8, help="VMD 模态数（仅 vmdstats 用）")
    ap.add_argument("--nbin", type=int, default=256, help="log 谱维度")
    ap.add_argument("--C", type=float, default=0.3, help="逻辑回归 C")
    ap.add_argument("--max_iter", type=int, default=800)
    ap.add_argument("--keys_out", default=None, help="划分文件键 JSON 输出路径")
    ap.add_argument("--keys_only", action="store_true",
                    help="只导出全网格划分键、不跑探针（版本3 §12）")
    args = ap.parse_args()
    if args.split_mode == "paired" and args.label_pool_ratio is None:
        raise SystemExit("paired 必须提供 --label_pool_ratio")

    # --keys_only：只导出键
    if args.keys_only:
        key_blob = collect_keys(args)
        out = args.keys_out or "docs/evidence/E22_keys.json"
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(key_blob, handle, ensure_ascii=False, indent=2)
        print(f"[keys_only] 全网格划分键 {len(key_blob)} 条已写入 {out}")
        return

    print("# 命令: python " + " ".join(sys.argv))
    print("=" * 108)
    print("E22 协议匹配探针（paired 划分与 train.py 完全一致；CPU；逻辑回归；复用 preprocessing）")
    print(f"  数据根目录={args.data_root}  L={args.L} 不重叠  C={args.C}  "
          f"split_mode={args.split_mode}  label_pool_ratio={args.label_pool_ratio}")
    print("  记录级：硬=段 argmax 取多数；软=段概率均值后 argmax；两者均为记录等权")
    print("=" * 108)

    records, feats, fids = build_features(args)
    file_labels = np.array([r["label"] for r in records], dtype=np.int64)
    file_index = {r["file_key"]: i for i, r in enumerate(records)}
    key_blob = {}

    for repr_name in args.reprs:
        print(f"\n[表征] {repr_name}  dim={feats[repr_name]['DE'].shape[1]}")
        summary = {}
        for ratio in args.label_ratios:
            for seed in args.seeds:
                split = ds.split_target_files(
                    records, label_ratio=ratio, protocol="transductive",
                    seed=seed + 100003, split_mode=args.split_mode,
                    label_pool_ratio=(args.label_pool_ratio
                                      if args.split_mode == "paired" else None))
                lab_files = np.array([file_index[k] for k in split["labeled"]], dtype=int)
                ev_files = np.array([file_index[k] for k in split["eval"]], dtype=int)
                key_blob[f"{args.split_mode}|pool{args.label_pool_ratio}|lr{ratio}|s{seed}"] = {
                    "labeled": sorted(split["labeled"]), "eval": sorted(split["eval"])}

                X_de = feats[repr_name]["DE"]
                X_fe = feats[repr_name]["FE"]
                y_de = file_labels[fids["DE"]]
                y_fe = file_labels[fids["FE"]]
                tr_mask = np.isin(fids["FE"], lab_files) if len(lab_files) else np.zeros(len(y_fe), bool)
                ev_mask = np.isin(fids["FE"], ev_files)
                ev_fid = fids["FE"][ev_mask]
                print(f"--- {repr_name} seed={seed} label_ratio={ratio}："
                      f"有标签 {len(lab_files)} 文件/{int(tr_mask.sum())} 段，"
                      f"评估 {len(ev_files)} 文件/{int(ev_mask.sum())} 段")

                if tr_mask.sum() > 0:
                    m = probe(X_fe[tr_mask], y_fe[tr_mask], X_fe[ev_mask], y_fe[ev_mask],
                              ev_fid, args)
                    print(fmt_metrics("target-only", m))
                    summary.setdefault(("target_only", ratio), []).append(m)

                    m = probe(X_fe[tr_mask], y_fe[tr_mask], X_fe[ev_mask], y_fe[ev_mask],
                              ev_fid, args, class_weight="balanced")
                    print(fmt_metrics("target-only + balanced", m))
                    summary.setdefault(("target_only_balanced", ratio), []).append(m)

                    X_st = np.concatenate([X_fe[tr_mask], X_de])
                    y_st = np.concatenate([y_fe[tr_mask], y_de])
                    m = probe(X_st, y_st, X_fe[ev_mask], y_fe[ev_mask], ev_fid, args)
                    print(fmt_metrics("source+target", m))
                    summary.setdefault(("source_target", ratio), []).append(m)

                mu_s, sd_s = X_de.mean(0), X_de.std(0) + 1e-9
                m = probe(X_de, y_de, X_fe[ev_mask], y_fe[ev_mask], ev_fid, args)
                print(fmt_metrics("zero-label 不对齐", m))
                summary.setdefault(("zero_label_raw", ratio), []).append(m)
                for tag, rows in (("全FE统计", None), ("eval统计", ev_mask)):
                    ref = X_fe if rows is None else X_fe[rows]
                    mu_t, sd_t = ref.mean(0), ref.std(0) + 1e-9
                    m = probe(X_de, y_de, X_fe[ev_mask], y_fe[ev_mask], ev_fid, args,
                              align=(mu_s, sd_s, mu_t, sd_t))
                    print(fmt_metrics(f"zero-label 一阶对齐({tag})", m))
                    name = ("zero_label_align_all" if tag == "全FE统计"
                            else "zero_label_align_eval")
                    summary.setdefault((name, ratio), []).append(m)

        print(f"\n--- {repr_name} 汇总（{len(args.seeds)} seed 均值±样本SD）")
        for (variant, ratio), items in sorted(summary.items()):
            acc = np.array([m["acc"] for m in items])
            f1 = np.array([m["f1"] for m in items])
            hard = np.array([m["hard"] for m in items])
            soft = np.array([m["soft"] for m in items])
            rec = np.array([m["recall"] for m in items])
            acc_m, acc_s = mean_sd(acc)
            hard_m, _ = mean_sd(hard)
            soft_m, _ = mean_sd(soft)
            parts = []
            for i in range(len(ds.LABEL_NAMES)):
                rec_m, rec_s = mean_sd(rec[:, i])
                parts.append(f"{ds.LABEL_NAMES[i][:3]}={rec_m:.2f}±{rec_s:.2f}")
            print(f"  {variant:<20} lr={ratio:<5} 段级 {acc_m:.4f}±{acc_s:.4f}  "
                  f"f1 {f1.mean():.4f}  记录硬 {hard_m:.4f}  记录软 {soft_m:.4f}  "
                  + " ".join(parts))

    if args.keys_out:
        os.makedirs(os.path.dirname(args.keys_out) or ".", exist_ok=True)
        with open(args.keys_out, "w", encoding="utf-8") as handle:
            json.dump(key_blob, handle, ensure_ascii=False, indent=2)
        print(f"\n[产物] 划分文件键已写入 {args.keys_out}")


if __name__ == "__main__":
    main()
