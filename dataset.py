# -*- coding: utf-8 -*-
"""CWRU 数据入口：来源/通道读取、降采样、分段与文件级划分。

设计依据：版本3 §4.2（数据层不变量与迁移审计）。本模块位于训练流程最前端，
T1 只通过 `enumerate_t1_files`、`build_t1_segments`、`split_target_files` 进入数据。

关键不变量（版本3 §4.2）：
- `12k_DE` 排除全部 028，`Normal` 全保留；每文件必须同时含 DE/FE；共 56 记录、
  四类 4/12/12/28。
- Normal 原始 48 kHz 先降采样到 12 kHz 再切段；`seg_len=512`、不重叠；DE 共 13191 段。
- paired 语义：先按 `label_pool_ratio` 划固定标签池，补集为固定评估集，再按池内
  `label_ratio` 取有标签文件；未取用文件记 `unused`。
- 划分数值必须与旧库逐键等价：`split_target_files(records, ..., seed+100003)`（seed 偏移
  在 experiment.prepare_domains 施加）。因此本模块的划分算法（随机流用法、排序、取整规则）
  刻意与旧库保持逐字节一致，改动即会破坏 128 run 划分等价审计。

数据流形状：`.mat` → `read_t1_signal` (N,) → `build_t1_segments` → SegmentSet
（signals (M,seg_len)、labels (M,)、segment_ids (M,)）。
"""
from pathlib import Path
import re

import numpy as np
import scipy.io
from scipy.signal import resample_poly

#: 文件名故障部位前缀到整数标签的映射。
LABEL_MAP = {"N": 0, "B": 1, "IR": 2, "OR": 3}
#: 固定类别顺序，必须与模型输出维度一致。
LABEL_NAMES = ["Normal", "Ball", "InnerRace", "OuterRace"]
#: T1 使用的来源及原始采样率。
DATASET_FS = {"12k_DE": 12000, "Normal": 48000}
T1_DATASETS = ("12k_DE", "Normal")
T1_CHANNELS = ("DE", "FE")


# ---------------------------------------------------------------------------
# .mat 读取与键名统一
# ---------------------------------------------------------------------------
def remove_dic_items(dic):
    """就地移除 `loadmat` 添加的元数据键。"""
    for values in dic.values():
        for key in ("__header__", "__version__", "__globals__"):
            values.pop(key, None)


def rename_keys(dic):
    """把带编号的 CWRU 通道键统一为 `DE_time`/`FE_time`/`BA_time`/`RPM`。

    多候选键的保留规则与历史数据副本一致：按 `loadmat` 键序遍历，后出现的同类键
    覆盖前者。`check_multikey_regression` 用回归检查锁死这一依赖，防止换数据打包顺序
    后静默换通道。
    """
    for values in dic.values():
        for key in list(values.keys()):
            if not isinstance(key, str) or ("_time" not in key and "RPM" not in key):
                continue
            if "DE_time" in key:
                values["DE_time"] = values.pop(key)
            elif "BA_time" in key:
                values["BA_time"] = values.pop(key)
            elif "FE_time" in key:
                values["FE_time"] = values.pop(key)
            elif "RPM" in key:
                values["RPM"] = values.pop(key)


def _clean_mat(path):
    """读取一个 `.mat` 并完成元数据清理和通道键统一。"""
    raw = scipy.io.loadmat(path)
    remove_dic_items({"_": raw})
    rename_keys({"_": raw})
    return raw


def read_channel_signal(path, channel):
    """读取单个文件的指定通道信号，返回 `(N,)` float64。channel ∈ {DE, FE}。"""
    if channel not in T1_CHANNELS:
        raise ValueError(f"未知通道 {channel}，T1 只支持 DE / FE")
    raw = _clean_mat(path)
    key = f"{channel}_time"
    if key not in raw:
        raise ValueError(f"文件 {path} 缺少 {key}，T1 要求每个文件同时具备 DE/FE")
    signal = np.asarray(raw[key]).ravel()
    if signal.size == 0:
        raise ValueError(f"文件 {path} 的 {key} 为空")
    return signal.astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# 文件枚举与 T1 数据入口
# ---------------------------------------------------------------------------
def parse_label(filename):
    """从文件名解析 `N`/`B`/`IR`/`OR` 标签。"""
    filename = str(filename)
    if "IR" in filename:
        return "IR"
    if "OR" in filename:
        return "OR"
    if "B" in filename:
        return "B"
    if "Normal" in filename:
        return "N"
    raise ValueError(f"无法识别的文件名: {filename}")


def parse_load(filename):
    """解析文件名末尾 `_数字` 载荷；T1 只记录它，不按载荷划分。"""
    return int(str(filename).rsplit("_", 1)[-1].split(".")[0])


def is_028_file(filename):
    """判断故障文件是否属于需排除的 NTN 028 档（版本3 §4.2 不变量 1）。"""
    stem = Path(str(filename)).stem
    return bool(re.search(r"(?:B|IR|OR)028(?:@|_|$)", stem))


def make_file_key(dataset, filename):
    """生成携带来源目录的稳定文件键，如 `12k_DE/B007_0.mat`。"""
    return f"{dataset}/{Path(str(filename)).name}"


def enumerate_t1_files(data_root="Data", require_channels=True):
    """枚举 T1 的 56 个候选文件并返回文件元数据列表（版本3 §4.2 不变量 1、3）。

    `12k_DE` 过滤 028，`Normal` 全部保留；两类来源分别记录原始采样率。
    `require_channels=True` 时立即检查 DE 与 FE 都存在，缺失通道直接报错。
    """
    root = Path(data_root)
    if root.name in T1_DATASETS:
        root = root.parent
    records = []
    for dataset in T1_DATASETS:
        folder = root / dataset
        if not folder.is_dir():
            raise FileNotFoundError(f"缺少 T1 数据目录: {folder}")
        for path in sorted(folder.glob("*.mat")):
            if dataset == "12k_DE" and is_028_file(path.name):
                continue
            label_letter = parse_label(path.name)
            record = {
                "file_key": make_file_key(dataset, path.name),
                "dataset": dataset,
                "filename": path.name,
                "path": str(path),
                "label_letter": label_letter,
                "label": LABEL_MAP[label_letter],
                "load": parse_load(path.name),
                "raw_fs": DATASET_FS[dataset],
            }
            if require_channels:
                missing = [ch for ch in T1_CHANNELS
                           if f"{ch}_time" not in _clean_mat(path)]
                if missing:
                    raise ValueError(
                        f"T1 文件 {record['file_key']} 缺少通道 {missing}；"
                        "T1 要求 DE/FE 成对存在")
            records.append(record)
    if not records:
        raise ValueError(f"{root} 中没有可用的 T1 文件")
    return records


def check_multikey_regression(data_root="Data"):
    """核验多键文件仍按既定键序保留对应信号（防重打包静默换通道）。

    检查 `Normal_2.mat` 的 DE/FE；文件不存在（如合成测试目录）时返回空结果。
    以原始键序最后一个候选作为历史规则的期望，顺序变化即失败。
    """
    path = Path(data_root) / "Normal" / "Normal_2.mat"
    if not path.exists():
        return {}
    raw = scipy.io.loadmat(path)
    result = {}
    for channel in T1_CHANNELS:
        candidates = [k for k in raw if isinstance(k, str) and f"{channel}_time" in k]
        if len(candidates) < 2:
            continue
        expected = np.asarray(raw[candidates[-1]]).ravel()
        actual = read_channel_signal(path, channel)
        if not np.array_equal(actual, expected):
            raise ValueError(
                f"{path.name} 的 {channel} 多键读取回归失败："
                "保留键不再是当前数据副本的最后一个候选")
        result[channel] = candidates[-1]
    return result


def read_t1_signal(record, channel, target_fs=12000):
    """读取 T1 文件并把 48 kHz 来源先降采样到统一的 12 kHz（版本3 §4.2 不变量 2）。

    返回 `(N,)` float64。降采样发生在分段和表征之前，确保每个 512 点窗口代表相同
    物理时间尺度。
    """
    signal = read_channel_signal(record["path"], channel)
    raw_fs = int(record["raw_fs"])
    if raw_fs != target_fs:
        if raw_fs % target_fs != 0:
            raise ValueError(f"不支持的采样率转换: {raw_fs} -> {target_fs}")
        signal = resample_poly(signal, target_fs, raw_fs)
    return np.asarray(signal, dtype=np.float64)


# ---------------------------------------------------------------------------
# T1 分段集合
# ---------------------------------------------------------------------------
class SegmentSet:
    """T1 分段集合：按文件键连续排列的段信号、标签和 segment_id。

    每个文件占一段连续行区间，行号即身份：`rows(file_keys)` 给出子集行号，
    `span(file_key)` 给出单文件的连续区间。
    """

    def __init__(self, file_keys, signals, labels, segment_ids, fs=12000):
        self.file_keys = [str(key) for key in file_keys]
        self.signals = np.asarray(signals, dtype=np.float64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.segment_ids = np.asarray(segment_ids, dtype=np.int64)
        self.fs = int(fs)
        n_rows = len(self.labels)
        if not (len(self.file_keys) == n_rows == len(self.segment_ids)
                == len(self.signals)):
            raise ValueError(
                "SegmentSet 的 file_keys/labels/segment_ids/signals 长度必须一致")
        self._spans = {}
        for row, key in enumerate(self.file_keys):
            if key in self._spans:
                self._spans[key][1] = row + 1
            else:
                self._spans[key] = [row, row + 1]

    def __len__(self):
        return len(self.labels)

    def span(self, file_key):
        """返回单个文件的连续行区间 `slice`；文件无段或缺失时报错。"""
        span = self._spans.get(str(file_key))
        if span is None:
            raise KeyError(f"分段集合缺少文件键: {file_key}")
        return slice(span[0], span[1])

    def rows(self, file_keys):
        """返回文件键对应的行号，按给定顺序和段序排列。"""
        keys = list(file_keys)
        if not keys:
            return np.empty(0, dtype=np.int64)
        spans = [self.span(key) for key in keys]
        return np.concatenate([np.arange(span.start, span.stop, dtype=np.int64)
                               for span in spans])


def build_t1_segments(data_root="Data", channel="DE", segment_length=512,
                      file_keys=None, records=None):
    """按来源和通道构建 T1 分段集合（版本3 §4.2 不变量 2、3）。

    每行是一个不重叠 `(segment_length,)` 窗口，采样率统一为 12000（48 kHz 来源已先降采样）。
    `file_keys` 为可选子集，默认全部 T1 文件；`records` 可复用 `enumerate_t1_files` 输出。
    """
    if channel not in T1_CHANNELS:
        raise ValueError(f"未知通道 {channel}，T1 只支持 DE / FE")
    if segment_length <= 0:
        raise ValueError("segment_length 必须为正数")
    records = list(records or enumerate_t1_files(data_root))
    lookup = {record["file_key"]: record for record in records}
    keys = ([record["file_key"] for record in records] if file_keys is None
            else list(file_keys))
    unknown = [key for key in keys if key not in lookup]
    if unknown:
        raise KeyError(f"未知 T1 文件键: {unknown[:3]}")

    row_keys, labels, segment_ids, segments = [], [], [], []
    for key in keys:
        record = lookup[key]
        signal = read_t1_signal(record, channel)
        n_seg = len(signal) // segment_length
        for segment_id in range(n_seg):
            start = segment_id * segment_length
            row_keys.append(key)
            labels.append(record["label"])
            segment_ids.append(segment_id)
            segments.append(signal[start:start + segment_length])
    signals = (np.stack(segments) if segments
               else np.empty((0, segment_length), dtype=np.float64))
    return SegmentSet(row_keys, signals, labels, segment_ids)


# ---------------------------------------------------------------------------
# 文件级划分
# ---------------------------------------------------------------------------
def _records_by_label(records):
    groups = {}
    for record in records:
        label = int(record["label"] if "label" in record else LABEL_MAP[record["label_letter"]])
        groups.setdefault(label, []).append(record)
    return groups


def _count_by_label(keys, by_key):
    """统计文件键在四个类别上的数量。"""
    counts = {label: 0 for label in range(len(LABEL_NAMES))}
    for key in keys:
        counts[int(by_key[key]["label"])] += 1
    return counts


def split_files_by_class(records, ratio=0.1, seed=0, require_holdout=True):
    """按类别随机选择文件，返回 `(selected_keys, remaining_keys)`。

    每类先按文件键排序再用独立可复现随机流打乱；`ratio > 0` 时至少选一个文件。
    `require_holdout=True`（默认）要求每类至少留一个文件给评估，否则报错；固定评估集的
    标签预算实验传 `False`（候选文件可全部选作有标签文件，评估集另行固定）。

    随机流用法（`np.random.default_rng(seed).permutation` + 取整 `ceil`）刻意与旧库一致，
    以保证 128 run 划分等价（版本3 §4.2 划分等价审计）。
    """
    if not 0 <= ratio < 1:
        raise ValueError(f"文件比例必须在 [0, 1) 内，当前 {ratio}")
    rng = np.random.default_rng(seed)
    selected, remaining = [], []
    for label in sorted(_records_by_label(records)):
        group = sorted(_records_by_label(records)[label], key=lambda r: r["file_key"])
        order = rng.permutation(len(group))
        n_selected = 0 if ratio == 0 else max(1, int(np.ceil(ratio * len(group))))
        if require_holdout and n_selected >= len(group) and ratio > 0:
            raise ValueError(f"类别 {LABEL_NAMES[label]} 只有 {len(group)} 个文件，"
                             "当前 label_ratio 无法同时保留训练和评估文件")
        selected.extend(group[i]["file_key"] for i in order[:n_selected])
        remaining.extend(group[i]["file_key"] for i in order[n_selected:])
    return sorted(selected), sorted(remaining)


def split_target_files(records, label_ratio=0.1, protocol="transductive", seed=0,
                       adapt_ratio=None, reserved_keys=None, fixed_eval_keys=None,
                       candidate_keys=None, eval_keys=None, split_mode="natural",
                       label_pool_ratio=None):
    """按类别和文件切分目标 FE 的有标签、适配、评估集合（版本3 §4.2 不变量 4–8）。

    `split_mode="paired"`：先按 `label_pool_ratio` 从全部目标文件划出固定标签池，补集为固定
    评估集，再按池内 `label_ratio` 取有标签文件；池内未取用文件记 `unused`，不进入任何路径。
    `reserved_keys`/`fixed_eval_keys`/`candidate_keys` 为 natural 的历史标签预算接口。
    """
    if protocol not in ("transductive", "inductive"):
        raise ValueError(f"未知协议: {protocol}")
    if not 0 <= label_ratio < 1:
        raise ValueError(f"label_ratio 必须在 [0, 1) 内，当前 {label_ratio}")
    if split_mode not in ("natural", "paired"):
        raise ValueError(f"未知 split_mode: {split_mode}")
    by_key = {record["file_key"]: record for record in records}
    all_keys = set(by_key)
    if split_mode == "paired":
        return _split_paired(records, by_key, label_ratio, protocol, seed,
                             label_pool_ratio, reserved_keys, fixed_eval_keys,
                             candidate_keys, eval_keys)
    if label_pool_ratio is not None:
        raise ValueError("natural 不接受 label_pool_ratio")
    reserved = set(reserved_keys or [])
    fixed_eval = set(fixed_eval_keys if fixed_eval_keys is not None else (eval_keys or []))
    unknown = (reserved | fixed_eval | set(candidate_keys or [])) - all_keys
    if unknown:
        raise KeyError(f"目标划分包含未知文件键: {sorted(unknown)[:3]}")
    if reserved & fixed_eval:
        raise ValueError("reserved_keys 与 fixed_eval_keys 不得重叠")

    unused = []
    if fixed_eval:
        if protocol != "transductive":
            raise ValueError("fixed_eval_keys 当前只支持 transductive 标签预算实验")
        candidate = set(candidate_keys or (all_keys - fixed_eval - reserved))
        if candidate & fixed_eval or candidate & reserved:
            raise ValueError("candidate_keys 不得与评估/保留文件重叠")
        candidate_records = [by_key[key] for key in sorted(candidate)]
        labeled, unused = split_files_by_class(candidate_records, label_ratio, seed,
                                              require_holdout=False)
        rest = sorted(fixed_eval)
        return _natural_split(labeled, rest, list(rest), sorted(reserved),
                              sorted(unused), seed, label_ratio, 1.0)

    selectable = [record for record in records if record["file_key"] not in reserved]
    labeled, rest = split_files_by_class(selectable, label_ratio, seed)
    if protocol == "transductive":
        return _natural_split(labeled, rest, list(rest), sorted(reserved), [],
                              seed, label_ratio, 1.0)
    if adapt_ratio is None:
        adapt_ratio = 0.5
    if not 0 < adapt_ratio < 1:
        raise ValueError("inductive 的 adapt_ratio 必须在 (0, 1) 内")
    rng = np.random.default_rng(seed + 1)
    adapt, eval_keys_out = [], []
    for label in sorted(_records_by_label(selectable)):
        candidates = [key for key in rest if by_key[key]["label"] == label]
        if len(candidates) < 2:
            raise ValueError(f"inductive 下类别 {LABEL_NAMES[label]} 的剩余文件少于 2 个")
        candidates = list(np.asarray(sorted(candidates))[rng.permutation(len(candidates))])
        n_adapt = max(1, min(int(np.floor(len(candidates) * adapt_ratio)),
                            len(candidates) - 1))
        adapt.extend(candidates[:n_adapt])
        eval_keys_out.extend(candidates[n_adapt:])
    if not eval_keys_out:
        raise ValueError("inductive 划分没有评估文件")
    return _natural_split(sorted(labeled), sorted(adapt), sorted(eval_keys_out),
                          sorted(reserved), [], seed, label_ratio, adapt_ratio)


def _natural_split(labeled, adapt, eval_keys, reserved, unused, seed, label_ratio,
                   adapt_ratio):
    """natural 划分结果；池字段留空，不能伪装成 paired。"""
    return {"labeled": labeled, "adapt": adapt, "eval": eval_keys,
            "reserved": reserved, "unused": unused, "seed": seed,
            "label_ratio": label_ratio, "adapt_ratio": adapt_ratio,
            "split_mode": "natural", "split_source": "all_target_files",
            "label_pool_ratio": None, "label_pool": []}


def _split_paired(records, by_key, label_ratio, protocol, seed,
                  label_pool_ratio, reserved_keys, fixed_eval_keys, candidate_keys,
                  eval_keys):
    """固定标签池的两级划分，只支持 transductive（版本3 §4.2 不变量 7）。

    注意内部 `split_files_by_class(..., seed, ...)` 的 seed 就是外部传入值（已含 +100003），
    pool 与 labeled 两级都用同一 seed，与旧库逐字节一致，保证划分等价审计通过。
    """
    if protocol != "transductive":
        raise ValueError("paired 只支持 transductive 协议")
    if label_pool_ratio is None:
        raise ValueError("paired 必须提供 label_pool_ratio")
    if not 0 < float(label_pool_ratio) < 1:
        raise ValueError(f"label_pool_ratio 必须在 (0, 1) 内，当前 {label_pool_ratio}")
    legacy = set(reserved_keys or []) | set(fixed_eval_keys or []) \
        | set(candidate_keys or []) | set(eval_keys or [])
    if legacy:
        raise ValueError("paired 不能与 reserved_keys/fixed_eval_keys/candidate_keys 混用")
    present = set(_records_by_label(records))
    missing = [LABEL_NAMES[label] for label in range(len(LABEL_NAMES))
               if label not in present]
    if missing:
        raise ValueError(f"paired 缺少类别: {missing}")

    pool, evaluation = split_files_by_class(records, float(label_pool_ratio), seed,
                                            require_holdout=True)
    pool_counts = _count_by_label(pool, by_key)
    eval_counts = _count_by_label(evaluation, by_key)
    if any(pool_counts[label] == 0 or eval_counts[label] == 0
           for label in range(len(LABEL_NAMES))):
        raise ValueError("paired 要求每类标签池和评估集都非空")

    pool_records = [by_key[key] for key in pool]
    labeled, unused = split_files_by_class(pool_records, label_ratio, seed,
                                           require_holdout=False)
    if label_ratio == 0.99 and set(labeled) != set(pool):
        raise ValueError("label_ratio=0.99 没有取满当前标签池")
    return {"labeled": labeled, "adapt": list(evaluation), "eval": list(evaluation),
            "reserved": [], "unused": unused, "seed": seed,
            "label_ratio": label_ratio, "adapt_ratio": 1.0,
            "split_mode": "paired", "split_source": "label_pool",
            "label_pool_ratio": float(label_pool_ratio), "label_pool": list(pool)}


def make_development_split(records, ratio=0.2, seed=0):
    """生成每类固定文件级 development split，不改变正式源域全集。

    只用于预先选择固定轮数/学习率；正式 T1 运行不把 `dev` 从源域训练中永久扣除。
    """
    if not 0 < ratio < 1:
        raise ValueError("development ratio 必须在 (0, 1) 内")
    rng = np.random.default_rng(seed)
    train, dev = [], []
    groups = _records_by_label(records)
    for label in sorted(groups):
        group = sorted(groups[label], key=lambda r: r["file_key"])
        order = rng.permutation(len(group))
        n_dev = max(1, min(int(np.ceil(ratio * len(group))), len(group) - 1))
        dev.extend(group[i]["file_key"] for i in order[:n_dev])
        train.extend(group[i]["file_key"] for i in order[n_dev:])
    return {"train": sorted(train), "dev": sorted(dev), "seed": seed, "ratio": ratio}


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def _self_check():
    """验证 T1 文件入口、分段集合和文件级划分的基本不变量（版本3 §4.2）。"""
    data_root = Path(__file__).parent / "Data"
    if not (data_root / "12k_DE").exists():
        print(f"[跳过] 数据目录不存在: {data_root}")
        return
    records = enumerate_t1_files(data_root)
    assert len(records) == 56, len(records)
    counts = _count_by_label([r["file_key"] for r in records],
                             {r["file_key"]: r for r in records})
    assert counts == {0: 4, 1: 12, 2: 12, 3: 28}, counts
    check_multikey_regression(data_root)
    segments = build_t1_segments(data_root, "DE", 512, records=records)
    assert segments.fs == 12000
    assert len(segments) == 13191, len(segments)
    assert not any(is_028_file(key) for key in segments.file_keys)
    key0 = records[0]["file_key"]
    span = segments.span(key0)
    assert np.array_equal(segments.rows([key0]),
                          np.arange(span.start, span.stop, dtype=np.int64))
    try:
        segments.span("12k_DE/不存在.mat")
        raise AssertionError("未知文件键必须报错")
    except KeyError:
        pass
    split = split_target_files(records, 0.1, "transductive", seed=0)
    assert set(split["adapt"]) == set(split["eval"])
    assert not set(split["labeled"]) & set(split["eval"])
    # paired 语义：池补集为评估集，未取用记 unused
    paired = split_target_files(records, 0.99, "transductive", seed=100003,
                                split_mode="paired", label_pool_ratio=0.5)
    assert set(paired["labeled"]) == set(paired["label_pool"])
    assert not set(paired["label_pool"]) & set(paired["eval"])
    print(f"[T1] files={len(records)} 四类={counts} segments={len(segments)}")
    print("ALL DATASET CHECKS PASSED")


if __name__ == "__main__":
    _self_check()
