# -*- coding: utf-8 -*-
"""注意力权重可视化：谱前端下 α 解释为频带重要性（版本3 §4.4/§P3.1）。

谱前端 v1 把频轴压到 8 步，注意力权重 α (B,8) 对应 8 个频带的重要性。
cepstrum 的 236 维是低/高倒频拼接，序列轴中点不连续，可视化与文字须写明（版本3 §P3.1）。

用法：
  python visualize_attention.py --run runs/paired/<run_name> --ckpt transfer.pt --out attn.png
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataset as ds
import preprocessing as pp
import trainer as tr
from model import MSRSABiLSTM


def build_parser():
    ap = argparse.ArgumentParser(description="T1 注意力频带重要性可视化")
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="transfer.pt")
    ap.add_argument("--out", default="attention.png")
    ap.add_argument("--data_root", default=pp.DATA_DIR)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--n_samples", type=int, default=200)
    return ap


def main():
    args = build_parser().parse_args()
    ckpt_path = os.path.join(args.run, args.ckpt)
    conf = tr.load_checkpoint_config(ckpt_path)
    cfg = conf["repr_config"]
    frontend = conf["frontend"]["name"]
    is_cepstrum = conf["input_repr"] == "cepstrum"

    records = ds.enumerate_t1_files(args.data_root)
    keys = sorted(r["file_key"] for r in records)
    seg = ds.build_t1_segments(args.data_root, "FE", conf["seg_len"],
                               file_keys=keys, records=records)
    X, y, _ = pp.build_feature_set(seg, keys, "FE", cfg,
                                   cache_dir=args.cache_dir or pp.CACHE_DIR)

    model = MSRSABiLSTM(n_channels=conf["n_channels"], n_classes=conf["n_classes"],
                        frontend=frontend, lstm_layers=conf["recurrent_layers"],
                        use_se=conf["use_se"], threshold_mode=conf["threshold_mode"]).to(tr.DEV)
    tr.load_checkpoint(ckpt_path, model)
    model.eval()

    import torch
    n = min(args.n_samples, len(X))
    alphas, labels = [], []
    with torch.no_grad():
        for start in range(0, n, 256):
            _, a = model(X[start:start + 256].to(tr.DEV))
            alphas.append(a.cpu().numpy())
            labels.append(y[start:start + 256].numpy())
    alpha = np.concatenate(alphas)      # (n,8)
    label = np.concatenate(labels)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        for c in range(len(ds.LABEL_NAMES)):
            mask = label == c
            if mask.sum() > 0:
                ax.plot(range(8), alpha[mask].mean(0), marker="o", label=ds.LABEL_NAMES[c])
        title = "频带重要性（谱前端 8 步）"
        if is_cepstrum:
            title += "；注意：cepstrum 低/高倒频拼接，序列轴中点不连续"
        ax.set_title(title)
        ax.set_xlabel("序列步（频带）")
        ax.set_ylabel("平均注意力权重")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out, dpi=120)
        print(f"[产物] 注意力图已保存 {args.out}")
    except ImportError:
        print("[跳过绘图] 未安装 matplotlib；各类平均注意力权重：")
        for c in range(len(ds.LABEL_NAMES)):
            mask = label == c
            if mask.sum() > 0:
                print(f"  {ds.LABEL_NAMES[c]}: {np.round(alpha[mask].mean(0), 3)}")


if __name__ == "__main__":
    main()
