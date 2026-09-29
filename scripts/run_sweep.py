# -*- coding: utf-8 -*-
"""T1 对照网格：按 seed/协议/lr/λ/对照/表征/臂串行调用 train.py。

设计依据：版本3 §4.6（run 命名、--reprs/--alignment_modes/--arms/--tag）、§6（实验清单）。
natural/paired 严格分写；已存在 run 由 train.py 拒绝并计为跳过（版本3 §4.6 禁止删 run 重跑）。
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import experiment as ex  # noqa: E402


def build_parser():
    ap = argparse.ArgumentParser(description="T1 文件级对照批量实验（v4）")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--protocols", nargs="+", default=["transductive"],
                    choices=["transductive", "inductive"])
    ap.add_argument("--split_mode", choices=["natural", "paired"], default="natural")
    ap.add_argument("--label_pool_ratio", type=float, default=None)
    ap.add_argument("--label_ratios", type=float, nargs="+", default=[0.1])
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.3, 1.0])
    ap.add_argument("--controls", nargs="+", default=["semi_supervised"],
                    choices=["semi_supervised", "zero_label", "pure_mmd",
                             "no_transfer", "target_only"])
    ap.add_argument("--reprs", nargs="+", default=["vmd"],
                    choices=["vmd", "logspec", "cepstrum", "vmdstats"])
    ap.add_argument("--frontend", choices=["spectral_v1", "legacy_vmd"], default=None,
                    help="显式覆盖前端（中间臂：logspec + legacy_vmd）")
    ap.add_argument("--alignment_modes", nargs="+", default=[None],
                    help="global/conditional/confidence；默认沿用 train.py 默认")
    ap.add_argument("--arms", nargs="+", default=[None],
                    choices=["A", "B", "C", "D0.1", "D0.3", "E", None])
    ap.add_argument("--seg_lens", type=int, nargs="+", default=[512])
    ap.add_argument("--source_epochs", type=int, default=30)
    ap.add_argument("--transfer_epochs", type=int, default=30)
    ap.add_argument("--runs_dir", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--log", default=None)
    return ap


def legal_combo(control, label_ratio, lam):
    if not 0 <= label_ratio < 1 or lam < 0:
        return False
    if control == "pure_mmd":
        return label_ratio == 0 and lam > 0
    if control == "zero_label":
        return label_ratio == 0 and lam == 0
    if control == "target_only":
        return label_ratio > 0
    return True


def validate_args(args):
    if args.split_mode == "paired":
        if args.label_pool_ratio is None:
            raise SystemExit("paired 扫描必须提供 --label_pool_ratio")
        if not 0 < args.label_pool_ratio < 1:
            raise SystemExit("--label_pool_ratio 必须在 (0, 1) 内")
        if any(p != "transductive" for p in args.protocols):
            raise SystemExit("paired 扫描只支持 transductive")
    elif args.label_pool_ratio is not None:
        raise SystemExit("natural 扫描不接受 --label_pool_ratio")


def build_command(args, seed, protocol, ratio, lam, control, seg_len, repr_kind,
                  align, arm):
    run_name = ex.default_run_name(
        "T1", seed, protocol, ratio, lam, control, split_mode=args.split_mode,
        label_pool_ratio=args.label_pool_ratio, seg_len=seg_len, repr_kind=repr_kind,
        tag=(f"{args.tag}_{arm}" if (args.tag and arm) else (arm or args.tag)),
        align=align)
    runs_dir = args.runs_dir or os.path.join(ROOT, "runs", args.split_mode)
    cmd = [sys.executable, "-W", "ignore", os.path.join(ROOT, "train.py"),
           "--task", "T1", "--seed", str(seed), "--protocol", protocol,
           "--split_mode", args.split_mode, "--repr", repr_kind,
           "--label_ratio", str(ratio), "--lam", str(lam), "--control", control,
           "--seg_len", str(seg_len), "--source_epochs", str(args.source_epochs),
           "--transfer_epochs", str(args.transfer_epochs),
           "--runs_dir", runs_dir, "--run_name", run_name]
    if args.label_pool_ratio is not None:
        cmd += ["--label_pool_ratio", str(args.label_pool_ratio)]
    if align:
        cmd += ["--alignment_mode", align]
    if arm:
        cmd += ["--arm", arm]
    if getattr(args, "frontend", None):
        cmd += ["--frontend", args.frontend]
    return cmd


def main():
    args = build_parser().parse_args()
    validate_args(args)
    if args.runs_dir is None:
        args.runs_dir = os.path.join(ROOT, "runs", args.split_mode)
    os.makedirs(args.runs_dir, exist_ok=True)
    log_path = args.log or os.path.join(args.runs_dir, "sweep.log")
    combos = [(seed, protocol, ratio, lam, control, seg_len, repr_kind, align, arm)
              for seg_len in args.seg_lens for repr_kind in args.reprs
              for seed in args.seeds for protocol in args.protocols
              for ratio in args.label_ratios for control in args.controls
              for lam in args.lams for align in args.alignment_modes
              for arm in args.arms if legal_combo(control, ratio, lam)]
    print(f"共 {len(combos)} 个 T1 实验，日志：{log_path}", flush=True)
    ok = skipped = failed = 0
    started = time.time()
    with open(log_path, "a", encoding="utf-8") as log:
        for number, combo in enumerate(combos, 1):
            cmd = build_command(args, *combo)
            header = f"\n##### [{number}/{len(combos)}] {combo} #####"
            print(header, flush=True)
            log.write(header + "\n" + " ".join(cmd) + "\n")
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace")
            log.write(proc.stdout + proc.stderr)
            log.flush()
            out = proc.stdout + proc.stderr
            if proc.returncode == 0:
                ok += 1
                print("  完成", flush=True)
            elif "run 目录已存在" in out:
                skipped += 1
                print("  跳过：run 目录已存在", flush=True)
            else:
                failed += 1
                print(f"  失败（退出码 {proc.returncode}）", flush=True)
    print(f"完成 {ok}，跳过 {skipped}，失败 {failed}，耗时 {time.time() - started:.0f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
