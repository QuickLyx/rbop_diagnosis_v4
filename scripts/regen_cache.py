# -*- coding: utf-8 -*-
"""按 repr_config 缓存键预生成每文件/每通道的表征缓存（版本3 §4.1/§4.3）。

用法：
  python scripts/regen_cache.py --reprs logspec cepstrum --channels DE FE
  python scripts/regen_cache.py --reprs vmd --K 8   # VMD 缓存（慢）
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import dataset as ds  # noqa: E402
import preprocessing as pp  # noqa: E402


def build_parser():
    ap = argparse.ArgumentParser(description="重建 T1 文件/通道级表征缓存")
    ap.add_argument("--data_root", default=pp.DATA_DIR)
    ap.add_argument("--cache_dir", default=pp.CACHE_DIR)
    ap.add_argument("--channels", nargs="+", choices=["DE", "FE"], default=["DE", "FE"])
    ap.add_argument("--reprs", nargs="+", default=["logspec"],
                    choices=list(pp.REPR_KINDS))
    ap.add_argument("--seg_lens", type=int, nargs="+", default=[512])
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--nbin", type=int, default=256)
    ap.add_argument("--force", action="store_true", help="覆盖已有缓存")
    return ap


def main():
    args = build_parser().parse_args()
    records = ds.enumerate_t1_files(args.data_root)
    keys = sorted(r["file_key"] for r in records)
    started = time.time()
    total = 0
    for seg_len in args.seg_lens:
        for repr_kind in args.reprs:
            cfg = pp.build_repr_config(repr_kind, seg_len=seg_len, K=args.K, nbin=args.nbin)
            for channel in args.channels:
                segments = ds.build_t1_segments(args.data_root, channel, seg_len,
                                                file_keys=keys, records=records)
                pp.build_feature_set(segments, keys, channel, cfg,
                                     cache_dir=args.cache_dir, use_cache=not args.force)
                paths = [pp.get_cache_path(k, channel, cfg, args.cache_dir) for k in keys]
                missing = [p for p in paths if not os.path.exists(p)]
                if missing:
                    raise SystemExit(f"缓存重建后仍缺失: {missing[:3]}")
                total += len(paths)
                print(f"{repr_kind} {channel} L={seg_len}: {len(paths)} 文件就绪", flush=True)
    print(f"全部完成: {total} 个缓存，耗时 {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
