# -*- coding: utf-8 -*-
"""T1 命令行入口：参数解析、协议校验、一次实验调用。

设计依据：版本3 §4.6（CLI 与 run 命名）、§0.2（冻结配置）、§5（对照与臂）。
唯一任务 T1：12k_DE/Normal 的 DE→FE 同文件跨测点四分类。非 vmd 表征强制 K=1、sort_vmd=False。
"""
import argparse
import os

import dataset as ds
import experiment as ex
import preprocessing as pp
from trainer import DEV


def build_parser():
    ap = argparse.ArgumentParser(description="T1 同文件跨测点 DE→FE 故障诊断（v4）")
    ap.add_argument("--task", choices=["T1"], default="T1")
    ap.add_argument("--data_root", default=pp.DATA_DIR)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--seg_len", type=int, default=512)
    # 表征（版本3 §4.6）
    ap.add_argument("--repr", choices=list(pp.REPR_KINDS), default="vmd",
                    help="输入表征；非 vmd 时 K 固定 1、sort_vmd 强制 False")
    ap.add_argument("--K", type=int, default=8, help="VMD 模态数（仅 vmd/vmdstats 用）")
    ap.add_argument("--nbin", type=int, default=256, help="logspec 维度")
    ap.add_argument("--sort_vmd", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--frontend", choices=["spectral_v1", "legacy_vmd"], default=None,
                    help="显式覆盖前端；中间臂用 logspec + legacy_vmd 做归因（版本3 §4.4）")
    # 协议与划分
    ap.add_argument("--protocol", choices=["transductive", "inductive"], default="transductive")
    ap.add_argument("--split_mode", choices=["natural", "paired"], default="natural")
    ap.add_argument("--label_pool_ratio", type=float, default=None)
    ap.add_argument("--label_ratio", type=float, default=0.1)
    ap.add_argument("--control", choices=["semi_supervised", "zero_label", "pure_mmd",
                                          "no_transfer", "target_only"],
                    default="semi_supervised")
    ap.add_argument("--arm", choices=["A", "B", "C", "D0.1", "D0.3", "E"], default=None,
                    help="P2.2 五臂（版本3 §5 P2.2）；None=常规")
    ap.add_argument("--lam", type=float, default=0.3, help="MMD 权重 λ")
    ap.add_argument("--class_weight", action="store_true")
    # 超参（冻结配置，版本3 §0.2/§P2.1）
    ap.add_argument("--source_epochs", type=int, default=30)
    ap.add_argument("--transfer_epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr_source", type=float, default=1e-3)
    ap.add_argument("--lr_transfer", type=float, default=1e-4)
    ap.add_argument("--mmd_estimator", choices=["unbiased", "biased"], default="biased")
    ap.add_argument("--alignment_mode", choices=["global", "conditional", "confidence"],
                    default="confidence")
    ap.add_argument("--recurrent_layers", type=int, default=1)
    ap.add_argument("--threshold_mode", choices=["signed_mean", "mean_abs", "rms"],
                    default="signed_mean")
    ap.add_argument("--use_se", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs_dir", default=None)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--tag", default=None)
    return ap


def validate_args(args):
    if args.task != "T1":
        raise SystemExit("当前只实现 T1。")
    for name in ("seg_len", "K", "source_epochs", "transfer_epochs", "bs", "recurrent_layers"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name} 必须为正数")
    for name in ("lr_source", "lr_transfer"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name} 必须为正数")
    if not 0 <= args.label_ratio < 1:
        raise SystemExit("--label_ratio 必须在 [0, 1) 内")
    # 非 vmd 表征：强制 K=1、sort_vmd=False（版本3 §4.6）
    if args.repr not in ("vmd", "vmdstats"):
        args.K = 1
        args.sort_vmd = False
    if args.split_mode == "paired":
        if args.protocol != "transductive":
            raise SystemExit("paired 只支持 transductive 协议")
        if args.label_pool_ratio is None:
            raise SystemExit("paired 必须提供 --label_pool_ratio")
        if not 0 < args.label_pool_ratio < 1:
            raise SystemExit("--label_pool_ratio 必须在 (0, 1) 内")
    elif args.label_pool_ratio is not None:
        raise SystemExit("natural 不接受 --label_pool_ratio")
    if args.runs_dir is None:
        args.runs_dir = os.path.join(ex.RUNS_DIR, args.split_mode)
    if args.lam < 0:
        raise SystemExit("--lam 必须非负")
    if args.control == "target_only" and args.label_ratio == 0:
        raise SystemExit("target_only 必须使用 label_ratio > 0")
    if args.control == "zero_label" and (args.label_ratio != 0 or args.lam != 0):
        raise SystemExit("zero_label 要求 --label_ratio 0 且 --lam 0")
    if args.control == "pure_mmd" and (args.label_ratio != 0 or args.lam == 0):
        raise SystemExit("pure_mmd 要求 --label_ratio 0 且 --lam 非零")
    if args.arm is not None and args.control != "semi_supervised":
        raise SystemExit("P2.2 臂只在 semi_supervised 下使用")
    if not os.path.isdir(args.data_root):
        raise SystemExit(f"数据根目录不存在：{args.data_root}")


def main():
    args = build_parser().parse_args()
    validate_args(args)
    print(f"device={DEV} task=T1 repr={args.repr} 类别={ds.LABEL_NAMES}")
    record = ex.run_experiment(args)
    print(f"RESULT: {record['run']} baseline={record['baseline_acc']} "
          f"transfer={record['transfer_acc']} macro-F1={record['macro_f1']} "
          f"记录硬={record['record_accuracy_hard']}")


if __name__ == "__main__":
    main()
