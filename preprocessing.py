# -*- coding: utf-8 -*-
"""T1 表征构建：分段信号 → 表征张量，唯一提取实现 + repr_config 冻结 + 缓存。

设计依据：版本3 §4.3（repr_config 冻结 + 共享提取函数）、§0.1#5（表征参数必须冻结进
缓存键与 checkpoint）、§4.2（缓存等价审计）。

核心原则（版本3 §4.3）：
- 探针脚本（scripts/diag_protocol_probe.py）与网络（experiment.py）**共用本模块**的提取函数，
  禁止两处各写一份，否则 E22 对拍会漂移。
- 四表征：
  - vmd      ：normalize_segment → VMD(K=8) → (K,L)；读出后可 sort（sort 不入缓存内容，
               便于与旧库缓存 allclose 对拍，见 §4.3）。
  - logspec  ：Hanning 窗 → rfft(L) → |·|+1e-9 → linspace(0,L//2,nbin) int 采样 → log
               → 段内跨维标准化（mean/std，eps 1e-9）→ (1,nbin)，nbin=256。
  - cepstrum ：同 logspec 幅谱（nbin=L）→ log → irfft → 取低倒频 2..119 + 高倒频 394..511
               （236 维）→ 段内标准化 → (1,236)。
  - vmdstats ：每 VMD 模态 4 统计量（log 能量、峭度、谱质心、高频占比）→ (1,32)。
- repr_config 整体序列化进缓存键、npz 内容与 checkpoint；读缓存时逐字段校验。

缓存键（版本3 §4.3）：cache/<dataset>/<channel>/<stem>_L<seg_len>_<repr_key>.npz，
repr_key 含完整参数（如 logspec_hanning_nbin256_log1e-9_std1）。

数据流形状：SegmentSet.signals (N,L) → extract_repr → (N, C, D)（vmd 为 (N,K,L)，
其余为 (N,1,dim)）；build_feature_set 返回 (torch (N,C,D), y (N,), rows (N,))。
"""
import os
from pathlib import Path

import numpy as np
import torch

from vmd import vmd as vmd_decompose

#: 相对仓库根目录的缓存目录。
CACHE_DIR = "cache"
#: T1 数据根目录。
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")

REPR_KINDS = ("vmd", "logspec", "cepstrum", "vmdstats")


# ---------------------------------------------------------------------------
# repr_config：冻结的完整表征参数
# ---------------------------------------------------------------------------
def build_repr_config(repr_kind, seg_len=512, K=8, nbin=256, sort_vmd=True):
    """构造并冻结一个表征的完整参数字典（版本3 §4.3、§0.1#5）。

    返回的 dict 会整体进入缓存键、npz 与 checkpoint。字段随表征类型不同：
    - 公共：repr_kind, seg_len, log_eps, std_eps, seg_std
    - logspec：window=hanning, nbin
    - cepstrum：window=hanning, cep_nbin=seg_len, low_range=(2,120), high_tail=118
    - vmd/vmdstats：K, sort_vmd（仅 vmd 读出排序）, vmd_alpha
    不变量：`n_channels` 与 `input_dim` 由本 config 唯一决定，模型据此构造与校验。
    """
    if repr_kind not in REPR_KINDS:
        raise ValueError(f"未知 repr_kind: {repr_kind}")
    cfg = {
        "repr_kind": repr_kind,
        "seg_len": int(seg_len),
        "log_eps": 1e-9,
        "std_eps": 1e-9,
        "seg_std": True,   # 段内跨维标准化（版本3 §3：段内 z-score 必须保留）
    }
    if repr_kind == "logspec":
        cfg.update({"window": "hanning", "nbin": int(nbin),
                    "n_channels": 1, "input_dim": int(nbin)})
    elif repr_kind == "cepstrum":
        # 低倒频 2..119（118 个）+ 高倒频 (cep_nbin-118)..(cep_nbin-1)（118 个）= 236
        cep_nbin = int(seg_len)
        cfg.update({"window": "hanning", "cep_nbin": cep_nbin, "cep_n": 120,
                    "n_channels": 1, "input_dim": 236})
    elif repr_kind == "vmd":
        cfg.update({"window": "hanning", "K": int(K), "sort_vmd": bool(sort_vmd),
                    "vmd_alpha": 2000.0, "seg_std": True,
                    "n_channels": int(K), "input_dim": int(seg_len)})
    elif repr_kind == "vmdstats":
        cfg.update({"window": "hanning", "K": int(K), "vmd_alpha": 2000.0,
                    "n_channels": 1, "input_dim": 32})
    return cfg


def repr_key(cfg):
    """由 repr_config 派生稳定的缓存键字符串（不含 seg_len，seg_len 单列文件名）。"""
    kind = cfg["repr_kind"]
    if kind == "logspec":
        return f"logspec_{cfg['window']}_nbin{cfg['nbin']}_log{cfg['log_eps']:g}_std1"
    if kind == "cepstrum":
        return (f"cepstrum_{cfg['window']}_nbin{cfg['cep_nbin']}_n{cfg['cep_n']}"
                f"_log{cfg['log_eps']:g}_std1")
    if kind == "vmd":
        return f"vmd_K{cfg['K']}_a{cfg['vmd_alpha']:g}"
    if kind == "vmdstats":
        return f"vmdstats_K{cfg['K']}_a{cfg['vmd_alpha']:g}"
    raise ValueError(f"未知 repr_kind: {kind}")


# ---------------------------------------------------------------------------
# 底层算子（口径与旧库 diag_protocol_probe.py 逐字节一致，保证 E22 对拍）
# ---------------------------------------------------------------------------
def normalize_segment(x):
    """逐段 z-score 归一化，输入/输出形状均为 `(L,)`（vmd 输入用，eps 1e-8 与旧库一致）。"""
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + 1e-8)


def _shape_std(Z, eps=1e-9):
    """逐样本（段）跨维度标准化，(N,D)→(N,D)。口径同旧库 shape_std（eps 1e-9）。"""
    Z = np.asarray(Z, dtype=np.float64)
    return (Z - Z.mean(1, keepdims=True)) / (Z.std(1, keepdims=True) + eps)


def _amp_spec(W, nbin, L, log_eps=1e-9):
    """Hanning 窗幅谱，再等距重采样到 nbin 个 bin（含重复抽取，int 截断）。

    口径与旧库 amp_spec 一致：rfft(W*hanning(L)) 取模 + log_eps；pick=linspace(0,A_bins-1,nbin).astype(int)。
    """
    A = np.abs(np.fft.rfft(W * np.hanning(L), axis=1)) + log_eps
    pick = np.linspace(0, A.shape[1] - 1, nbin).astype(int)
    return A[:, pick]


def _logspec(signals, cfg):
    """256 维段内标准化 log 幅谱，(N,L)→(N,1,nbin)。"""
    L = cfg["seg_len"]
    feat = _shape_std(np.log(_amp_spec(signals, cfg["nbin"], L, cfg["log_eps"])),
                      cfg["std_eps"]).astype(np.float32)
    return feat[:, None, :]


def _cepstrum(signals, cfg):
    """236 维段内标准化倒谱，(N,L)→(N,1,236)。低倒频 2..119 + 高倒频末尾 118 个。"""
    L = cfg["seg_len"]
    n = cfg["cep_n"]
    C = np.fft.irfft(np.log(_amp_spec(signals, cfg["cep_nbin"], L, cfg["log_eps"])), axis=1)
    keep = np.concatenate([np.arange(2, n), np.arange(C.shape[1] - n + 2, C.shape[1])])
    feat = _shape_std(C[:, keep], cfg["std_eps"]).astype(np.float32)
    return feat[:, None, :]


def _vmd_tensor(signals, cfg):
    """VMD 分解，(N,L)→(N,K,L) float32。输入先降到 float32 再归一化（与旧库缓存口径一致）。"""
    K = cfg["K"]
    sigs = np.asarray(signals, dtype=np.float32)
    return np.stack([vmd_decompose(normalize_segment(s), K=K).astype(np.float32)
                     for s in sigs])


def reorder_vmd_modes(X):
    """按每个样本的谱质心升序重排 `(N, K, L)` 的 VMD 通道（版本3 §4.3 vmd 读出排序）。"""
    X = np.asarray(X)
    power = np.abs(np.fft.rfft(X, axis=-1)) ** 2
    freq = np.fft.rfftfreq(X.shape[-1])
    centroid = (power * freq).sum(axis=-1) / (power.sum(axis=-1) + 1e-12)
    order = np.argsort(centroid, axis=1)
    return np.take_along_axis(X, order[..., None], axis=1)


def _vmdstats(vmd_tensor, cfg):
    """每 VMD 模态压成 4 统计量，(N,K,L)→(N,1,32)。口径同旧库 vmdstats_features。"""
    L = cfg["seg_len"]
    X = np.asarray(vmd_tensor, dtype=np.float32)
    s = X / (X.std(axis=2, keepdims=True) + 1e-9)
    loge = np.log10((X ** 2).mean(axis=2) + 1e-12)
    kurt = (s ** 4).mean(axis=2)
    P = np.abs(np.fft.rfft(X * np.hanning(L), axis=2)) ** 2 + 1e-12
    f = np.fft.rfftfreq(L)
    cen = (P * f).sum(axis=2) / P.sum(axis=2)
    tilt = P[:, :, f > f.mean()].sum(axis=2) / P.sum(axis=2)
    stats = np.concatenate([_shape_std(loge), kurt.astype(np.float64),
                            cen.astype(np.float64) / 1000.0,
                            tilt.astype(np.float64)], axis=1)
    return stats.astype(np.float32)[:, None, :]


def extract_repr(signals, cfg):
    """唯一提取入口：分段信号 (N,L) → 表征张量 (N,C,D)（版本3 §4.3）。

    vmd 返回 (N,K,L)（未排序，排序在 build_feature_set 读出后按 cfg['sort_vmd'] 施加）；
    logspec (N,1,nbin)；cepstrum (N,1,236)；vmdstats (N,1,32)。
    """
    kind = cfg["repr_kind"]
    signals = np.asarray(signals)
    if signals.shape[1] != cfg["seg_len"]:
        raise ValueError(f"段长 {signals.shape[1]} 与 seg_len={cfg['seg_len']} 不符")
    if kind == "logspec":
        return _logspec(signals, cfg)
    if kind == "cepstrum":
        return _cepstrum(signals, cfg)
    if kind == "vmd":
        return _vmd_tensor(signals, cfg)
    if kind == "vmdstats":
        # 统计量按模态排序后计算：每模态 4 统计量的 32 维向量是模态有序的，必须先按谱质心
        # 排序才能与旧库探针（build_feature_set sort_vmd=True 后取 stats）逐字节一致（E22 对拍）。
        return _vmdstats(reorder_vmd_modes(_vmd_tensor(signals, cfg)), cfg)
    raise ValueError(f"未知 repr_kind: {kind}")


# ---------------------------------------------------------------------------
# 缓存
# ---------------------------------------------------------------------------
def _split_file_key(file_key):
    """拆分 `dataset/filename` 文件键；兼容 Windows 反斜杠。"""
    normalized = str(file_key).replace("\\", "/").strip("/")
    parts = normalized.split("/")
    if len(parts) >= 2:
        return "/".join(parts[:-1]), parts[-1]
    return "legacy", parts[0]


def get_cache_path(file_key, channel, cfg, cache_dir=CACHE_DIR):
    """返回单文件单通道缓存路径 cache/<dataset>/<channel>/<stem>_L<seg_len>_<repr_key>.npz。"""
    if channel not in ("DE", "FE"):
        raise ValueError(f"未知缓存通道: {channel}")
    dataset, filename = _split_file_key(file_key)
    stem = Path(filename).stem
    return os.path.join(cache_dir, dataset, channel,
                        f"{stem}_L{int(cfg['seg_len'])}_{repr_key(cfg)}.npz")


def _config_matches(cached_cfg, cfg):
    """逐字段校验缓存的 repr_config 与当前一致（版本3 §4.3）。"""
    for key, want in cfg.items():
        got = cached_cfg.get(key)
        # numpy 标量/字符串统一比较
        if isinstance(want, bool):
            if bool(got) != want:
                return False, key
        elif isinstance(want, (int, float)):
            if got is None or float(got) != float(want):
                return False, key
        else:
            if str(got) != str(want):
                return False, key
    return True, None


def _load_cached(cache_file, labels, segment_ids, cfg, channel, cache_key):
    """读取并校验单文件缓存，防止只按行数误命中。校验 shape/标签/段号/通道/键/repr_config。"""
    import json
    with np.load(cache_file, allow_pickle=False) as data:
        X = np.asarray(data["X"])
        y = np.asarray(data["y"])
        cached_segments = np.asarray(data["segment_id"])
        cached_channel = str(data["channel"].item()) if "channel" in data else None
        cached_key = str(data["file_key"].item()) if "file_key" in data else None
        cached_cfg = json.loads(str(data["repr_config"].item())) if "repr_config" in data else {}
    exp_shape = (len(labels), cfg["n_channels"], cfg["input_dim"])
    if X.shape != exp_shape:
        raise RuntimeError(
            f"缓存 {cache_file} 形状 {X.shape} 与当前 {exp_shape} 不符；请删除后重建")
    if not np.array_equal(y, labels) or not np.array_equal(cached_segments, segment_ids):
        raise RuntimeError(f"缓存 {cache_file} 的标签或 segment_id 与当前文件不符")
    if cached_channel is not None and cached_channel != channel:
        raise RuntimeError(f"缓存 {cache_file} 通道为 {cached_channel}，当前请求 {channel}")
    if cached_key is not None and cached_key != cache_key:
        raise RuntimeError(f"缓存 {cache_file} 文件键为 {cached_key}，当前请求 {cache_key}")
    ok, bad = _config_matches(cached_cfg, cfg)
    if not ok:
        raise RuntimeError(f"缓存 {cache_file} 的 repr_config 字段 {bad} 与当前不符")
    return X, y


def build_feature_set(segments, file_keys, channel, cfg, cache_dir=CACHE_DIR,
                      use_cache=True):
    """构建指定文件键和通道的表征张量（版本3 §4.3）。

    参数：segments SegmentSet；file_keys 文件键序列；channel DE/FE；cfg repr_config。
    返回 (X torch (N,C,D) float32, y torch (N,) long, rows (N,) 分段行号)。
    vmd 表征在读出所有文件后按 cfg['sort_vmd'] 排序（排序不写回缓存，见 §4.3）。
    """
    import json
    if channel not in ("DE", "FE"):
        raise ValueError(f"未知通道: {channel}")
    if segments.signals.shape[1] != cfg["seg_len"]:
        raise ValueError(f"分段长度 {segments.signals.shape[1]} 与 seg_len={cfg['seg_len']} 不符")
    keys = ([str(file_keys)] if isinstance(file_keys, (str, bytes))
            else [str(key) for key in file_keys])
    cfg_json = json.dumps(cfg, ensure_ascii=False, sort_keys=True)

    X_parts, y_parts = [], []
    for cache_key in keys:
        span = segments.span(cache_key)
        sub_labels = segments.labels[span]
        sub_segments = segments.segment_ids[span]
        cache_file = get_cache_path(cache_key, channel, cfg, cache_dir)
        X = y = None
        if use_cache and os.path.exists(cache_file):
            X, y = _load_cached(cache_file, sub_labels, sub_segments, cfg,
                                channel, cache_key)
        if X is None:
            X = extract_repr(segments.signals[span], cfg)
            y = sub_labels.copy()
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            np.savez_compressed(cache_file, X=X, y=y, segment_id=sub_segments,
                                channel=np.asarray(channel),
                                file_key=np.asarray(cache_key),
                                repr_config=np.asarray(cfg_json))
        X_parts.append(X)
        y_parts.append(np.asarray(y, dtype=np.int64))

    if not X_parts:
        empty = torch.empty((0, cfg["n_channels"], cfg["input_dim"]), dtype=torch.float32)
        return empty, torch.empty((0,), dtype=torch.long), np.empty(0, dtype=np.int64)
    X_all = np.concatenate(X_parts, axis=0)
    if cfg["repr_kind"] == "vmd" and cfg.get("sort_vmd", True):
        X_all = reorder_vmd_modes(X_all)
    y_all = np.concatenate(y_parts, axis=0)
    return (torch.from_numpy(X_all.astype(np.float32, copy=False)),
            torch.from_numpy(y_all.astype(np.int64, copy=False)),
            segments.rows(keys))


def take_rows(X, y, index, positions):
    """按分段行号切取张量，保留 `positions` 顺序并支持非升序。"""
    positions = np.asarray(positions, dtype=np.int64)
    if len(positions) == 0:
        return X[:0], y[:0]
    row_map = {int(position): row for row, position in enumerate(index)}
    missing = [int(position) for position in positions if int(position) not in row_map]
    if missing:
        raise ValueError(f"positions 含不属于该特征集的索引: {missing[:3]}")
    rows = torch.as_tensor([row_map[int(position)] for position in positions], dtype=torch.long)
    return X[rows], y[rows]


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def _self_check():
    """验证四表征形状、repr_config 派生键、排序、缓存键隔离与非升序索引映射。"""
    rng = np.random.default_rng(0)
    sigs = rng.standard_normal((6, 512)).astype(np.float64)

    ls = extract_repr(sigs, build_repr_config("logspec"))
    assert ls.shape == (6, 1, 256), ls.shape
    cp = extract_repr(sigs, build_repr_config("cepstrum"))
    assert cp.shape == (6, 1, 236), cp.shape
    vs = extract_repr(sigs, build_repr_config("vmdstats", K=8))
    assert vs.shape == (6, 1, 32), vs.shape
    vm = extract_repr(sigs, build_repr_config("vmd", K=8))
    assert vm.shape == (6, 8, 512), vm.shape

    # 段内标准化：logspec 每段均值≈0
    assert abs(ls[0, 0].mean()) < 1e-5

    # repr_config 派生键唯一
    assert repr_key(build_repr_config("logspec")) == "logspec_hanning_nbin256_log1e-09_std1"
    assert repr_key(build_repr_config("vmd", K=8)) == "vmd_K8_a2000"

    # 排序后谱质心升序
    Xs = reorder_vmd_modes(vm)
    power = np.abs(np.fft.rfft(Xs, axis=-1)) ** 2
    freq = np.fft.rfftfreq(512)
    centroid = (power * freq).sum(-1) / (power.sum(-1) + 1e-12)
    assert (np.diff(centroid, axis=1) >= -1e-9).all()

    # 缓存键隔离：通道、来源、表征不同则路径不同
    c = build_repr_config("logspec")
    assert get_cache_path("12k_DE/B007_0.mat", "DE", c) != \
           get_cache_path("12k_DE/B007_0.mat", "FE", c)
    assert get_cache_path("12k_DE/B007_0.mat", "DE", c) != \
           get_cache_path("Normal/B007_0.mat", "DE", c)
    assert get_cache_path("12k_DE/B007_0.mat", "DE", build_repr_config("logspec")) != \
           get_cache_path("12k_DE/B007_0.mat", "DE", build_repr_config("cepstrum"))

    # 非升序 take_rows
    Xt = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    yt = torch.arange(10)
    sel_X, sel_y = take_rows(Xt, yt, np.array([18, 0, 12]), [12, 0])
    assert torch.equal(sel_X, Xt[[2, 1]]) and torch.equal(sel_y, yt[[2, 1]])
    print("ALL PREPROCESSING CHECKS PASSED")


if __name__ == "__main__":
    _self_check()
