# -*- coding: utf-8 -*-
"""T1 单次实验编排：枚举、表征缓存、固定轮数训练、评估（含记录级）、结果记录。

设计依据：版本3 §4.5（results.json 全字段 + 记录级即时计算）、§4.6（run 命名）、
§5 P0.0/P2.1/P2.2（对照与臂）、§P1（记录级读出）。

唯一任务：`12k_DE/Normal: DE -> FE`（同文件跨测点四分类）。源域=全 56 文件 DE；
目标 FE 按文件级划分。评估标签不进训练；所有对照共享同 seed 目标文件键。
"""
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime

import torch

import dataset as ds
import preprocessing as pp
import trainer as tr
from model import MSRSABiLSTM

RUNS_DIR = "runs"
RESULTS_FILE = "results.json"
SOURCE_DOMAIN = {"datasets": ["12k_DE", "Normal"], "channel": "DE", "fs": 12000}
TARGET_DOMAIN = {"datasets": ["12k_DE", "Normal"], "channel": "FE", "fs": 12000}

#: 结果字段固定（版本3 §4.5）。summarize.py 用 row.get 容忍缺省。
RESULT_FIELDS = [
    # 运行元数据
    "run", "schema_version", "task", "protocol", "control", "seed", "split_mode",
    "split_source", "label_ratio", "label_pool_ratio", "adapt_ratio",
    "source_domain", "target_domain",
    "source_file_keys", "target_label_pool_keys", "target_labeled_keys",
    "target_unused_keys", "target_adapt_keys", "target_eval_keys", "dev_file_keys",
    "n_source_files", "n_target_files", "n_label_pool_files", "n_labeled_files",
    "n_unused_files", "n_adapt_files", "n_eval_files",
    "n_source", "n_labeled", "n_adapt", "n_eval",
    # 表征与模型
    "input_repr", "repr_config", "input_dim", "n_channels", "seg_len", "K", "sort_vmd",
    "recurrent_layers", "threshold_mode", "use_se", "frontend", "bs", "lr_source",
    "lr_transfer", "source_epochs", "transfer_epochs", "lam", "alignment_mode",
    "mmd_estimator", "class_weight", "arm",
    # 适配与读出
    "bn_adapt_on", "bn_adapt_scope", "bn_adapt_passes",
    "record_accuracy_hard", "record_accuracy_soft", "record_macro_f1",
    "record_class_recall", "per_file_accuracy", "per_file_support", "per_file_pred",
    # 指标
    "source_val_acc", "baseline_acc", "transfer_acc", "macro_f1", "class_recall",
    "confusion_matrix", "elapsed_s",
]


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
        return len(text)

    def flush(self):
        for s in self.streams:
            s.flush()


@contextmanager
def configure_logging(log_path):
    """把 print 复制到 UTF-8 日志文件。"""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    handle = open(log_path, "w", encoding="utf-8")
    original = sys.stdout
    sys.stdout = _Tee(original, handle)
    try:
        yield handle
    finally:
        sys.stdout = original
        handle.close()


def _fmt_number(value):
    return f"{float(value):g}"


def default_run_name(task="T1", seed=0, protocol="transductive", label_ratio=0.1,
                     lam=0.3, control="semi_supervised", split_mode="natural",
                     label_pool_ratio=None, seg_len=512, repr_kind="vmd", tag=None,
                     align=None):
    """生成 run 名（版本3 §4.6）：
    T1_s{seed}_{split_mode}_{protocol}[_poolr{p}]_lr{r}_lam{l}_{control}_L{seg}_{repr}[_align{mode}][_{tag}]
    align 非默认（confidence）时加后缀，避免同一网格下 global/conditional 同名冲突。
    """
    name = f"{task}_s{seed}_{split_mode}_{protocol}"
    if split_mode == "paired":
        name += f"_poolr{_fmt_number(label_pool_ratio)}"
    name += f"_lr{_fmt_number(label_ratio)}_lam{_fmt_number(lam)}_{control}"
    name += f"_L{seg_len}_{repr_kind}"
    if align and align != "confidence":
        name += f"_align{align}"
    if tag:
        name += f"_{tag}"
    return name


def create_run_directory(run_name, runs_dir=RUNS_DIR):
    """创建 run 目录；同名存在时拒绝覆盖（版本3 §4.6 禁止删 run 重跑）。"""
    path = os.path.join(runs_dir, run_name)
    if os.path.exists(path):
        raise SystemExit(f"run 目录已存在：{path}\n请更换 --run_name，禁止覆盖旧结果。")
    os.makedirs(path)
    return path


def append_result(record, results_path=None, runs_dir=None):
    """按划分模式追加结果；natural/paired 严格分写不混表（版本3 §4.6）。"""
    split_mode = record.get("split_mode", "natural")
    if split_mode not in ("natural", "paired"):
        raise ValueError(f"未知结果划分模式: {split_mode}")
    if results_path is None:
        runs_dir = runs_dir or os.path.join(RUNS_DIR, split_mode)
        results_path = os.path.join(runs_dir, RESULTS_FILE)
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    rows = []
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as handle:
            rows = json.load(handle)
        if not isinstance(rows, list):
            raise SystemExit(f"{results_path} 不是 JSON 数组，拒绝覆盖")
        existing_modes = {row.get("split_mode") for row in rows}
        if existing_modes and existing_modes != {split_mode}:
            raise SystemExit(f"{results_path} 已包含其他 split_mode，拒绝混写")
    rows.append({key: record.get(key) for key in RESULT_FIELDS})
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    return results_path


def _cache_dir(args):
    return getattr(args, "cache_dir", None) or pp.CACHE_DIR


def _repr_config(args):
    """按 CLI 参数构造 repr_config；非 vmd 强制 K=1、sort_vmd=False（版本3 §4.6）。

    is_vmd 时 K 才有意义（VMD 模态数）；非 vmd 表征下 build_repr_config 不读 K，
    统一用单一局部变量 `k` 传参，并对外返回写入 checkpoint/结果的 K=1。
    """
    kind = args.repr
    is_vmd = kind in ("vmd", "vmdstats")
    k = args.K if is_vmd else 1          # 对外返回与入 checkpoint 的 K；非 vmd 固定 1
    sort_vmd = bool(args.sort_vmd) if kind == "vmd" else False
    cfg = pp.build_repr_config(kind, seg_len=args.seg_len, K=k,
                               nbin=getattr(args, "nbin", 256), sort_vmd=sort_vmd)
    return cfg, k, sort_vmd


def _frontend_name(args):
    """选 frontend（版本3 §4.4）：默认 vmd→legacy_vmd、其余→spectral_v1；
    --frontend 可显式覆盖（中间臂：logspec + legacy_vmd，用于拆分表征/前端收益）。"""
    override = getattr(args, "frontend", None)
    if override:
        return override
    return "legacy_vmd" if args.repr == "vmd" else "spectral_v1"


def _frontend_spec(name):
    """由 frontend 名生成结构描述（入 checkpoint/结果，版本3 §4.5）。"""
    if name == "legacy_vmd":
        return {"name": "legacy_vmd", "channels": [32, 64, 128], "pools": [4, 4, 4],
                "adaptive_out": None}
    return {"name": "spectral_v1", "channels": [32, 64, 128], "pools": [2, 4],
            "adaptive_out": 8}


def prepare_domains(args, records=None):
    """构建 T1 源/目标张量并完成文件级划分（版本3 §4.2/§4.3）。"""
    data_root = getattr(args, "data_root", pp.DATA_DIR)
    records = list(records or ds.enumerate_t1_files(data_root))
    ds.check_multikey_regression(data_root)
    source_keys = sorted(record["file_key"] for record in records)
    target_split = ds.split_target_files(
        records, label_ratio=float(args.label_ratio), protocol=args.protocol,
        seed=int(args.seed) + 100003, adapt_ratio=getattr(args, "adapt_ratio", None),
        split_mode=getattr(args, "split_mode", "natural"),
        label_pool_ratio=getattr(args, "label_pool_ratio", None))

    cfg, K, sort_vmd = _repr_config(args)
    source_set = ds.build_t1_segments(data_root, "DE", args.seg_len,
                                      file_keys=source_keys, records=records)
    target_set = ds.build_t1_segments(data_root, "FE", args.seg_len,
                                      file_keys=source_keys, records=records)
    use_cache = not getattr(args, "no_cache", False)
    cache_dir = _cache_dir(args)
    Xs_tr, ys_tr, _ = pp.build_feature_set(
        source_set, source_keys, "DE", cfg, cache_dir=cache_dir, use_cache=use_cache)
    Xt, yt, target_rows = pp.build_feature_set(
        target_set, source_keys, "FE", cfg, cache_dir=cache_dir, use_cache=use_cache)

    eval_rows = target_set.rows(target_split["eval"])
    if len(eval_rows) == 0:
        raise SystemExit("目标评估文件为空；请降低 label_ratio 或检查文件级划分")
    X_lab, y_lab = pp.take_rows(Xt, yt, target_rows,
                                target_set.rows(target_split["labeled"])) \
        if target_split["labeled"] else (Xt[:0], yt[:0])
    X_unlab = pp.take_rows(Xt, yt, target_rows,
                           target_set.rows(target_split["adapt"]))[0]
    Xt_eval, yt_eval = pp.take_rows(Xt, yt, target_rows, eval_rows)
    # 评估段→记录下标（供记录级读出，版本3 §P1.1）
    eval_keys = list(target_split["eval"])
    key_to_idx = {k: i for i, k in enumerate(eval_keys)}
    eval_record_ids = []
    for k in eval_keys:
        span = target_set.span(k)
        eval_record_ids.extend([key_to_idx[k]] * (span.stop - span.start))

    return {
        "Xs_tr": Xs_tr, "ys_tr": ys_tr, "X_lab": X_lab, "y_lab": y_lab,
        "X_unlab": X_unlab, "Xt_eval": Xt_eval, "yt_eval": yt_eval,
        "eval_record_ids": eval_record_ids, "eval_keys": eval_keys,
        "source_file_keys": source_keys,
        "target_label_pool_keys": target_split.get("label_pool", []),
        "target_labeled_keys": target_split["labeled"],
        "target_unused_keys": target_split.get("unused", []),
        "target_adapt_keys": target_split["adapt"],
        "target_eval_keys": target_split["eval"],
        "adapt_ratio": target_split["adapt_ratio"],
        "split_mode": target_split.get("split_mode", "natural"),
        "split_source": target_split.get("split_source", "all_target_files"),
        "label_pool_ratio": target_split.get("label_pool_ratio"),
        "dev_file_keys": [],
        "source_domain": dict(SOURCE_DOMAIN), "target_domain": dict(TARGET_DOMAIN),
        "repr_config": cfg, "K": K, "sort_vmd": sort_vmd,
        "n_channels": cfg["n_channels"], "input_dim": cfg["input_dim"],
    }


def build_model(args, parts):
    """按表征/覆盖构造模型与 frontend（版本3 §4.4）。"""
    return MSRSABiLSTM(
        n_channels=parts["n_channels"], n_classes=len(ds.LABEL_NAMES),
        frontend=_frontend_name(args), lstm_layers=args.recurrent_layers,
        use_se=args.use_se, threshold_mode=args.threshold_mode).to(tr.DEV)


def compute_class_weights(args, labels):
    """按源域全部训练样本计算 inverse-frequency 类别权重（默认 off）。"""
    if not getattr(args, "class_weight", False):
        return None
    counts = torch.bincount(labels, minlength=len(ds.LABEL_NAMES)).float()
    total = counts.sum()
    return (total / (len(ds.LABEL_NAMES) * counts.clamp(min=1.0))).to(tr.DEV)


def _ck_kwargs(args, parts, stage):
    """生成 checkpoint 元数据（schema_version=1，版本3 §4.5）。"""
    return dict(
        stage=stage, source=parts["source_domain"], target=parts["target_domain"],
        seg_len=args.seg_len, input_repr=args.repr, repr_config=parts["repr_config"],
        n_channels=parts["n_channels"], input_dim=parts["input_dim"],
        n_classes=len(ds.LABEL_NAMES), class_names=ds.LABEL_NAMES,
        recurrent_layers=args.recurrent_layers, threshold_mode=args.threshold_mode,
        use_se=args.use_se, frontend=_frontend_spec(_frontend_name(args)),
        K=parts["K"], sort_vmd=parts["sort_vmd"])


def _resolve_arm(arm, args):
    """把 P2.2 臂映射为 train_transfer 的参数（版本3 §5 P2.2）。

    返回 dict：freeze_backbone / source_ce / source_ce_weight / lr_transfer。
      B 臂：冻结 backbone 只训分类头；C 臂：去源 CE；D0.1/D0.3：源 CE 梯度尺度；
      E 臂：迁移学习率括 1e-3。None=常规。
    """
    return {
        "freeze_backbone": arm == "B",
        "source_ce": arm != "C",
        "source_ce_weight": {"D0.1": 0.1, "D0.3": 0.3}.get(arm, 1.0),
        "lr_transfer": 1e-3 if arm == "E" else args.lr_transfer,
    }


def _maybe_adabn(args, model, parts):
    """若启用 AdaBN，用目标无标签输入重估 BN 统计（版本3 §5 P0.1）。

    统计源：transductive 下的适配集 X_unlab（即评估文件的 FE 无标签输入）。
    返回 (bn_adapt_on, scope, passes) 供结果记录；未启用时返回 (False, None, None)。
    """
    if not getattr(args, "bn_adapt", False):
        return False, None, None
    scope = getattr(args, "bn_adapt_scope", "all")
    passes = int(getattr(args, "bn_adapt_passes", 1))
    X_stat = parts["X_unlab"]
    if X_stat is None or len(X_stat) == 0:
        raise SystemExit("AdaBN 需要非空目标无标签适配集")
    print(f"[AdaBN] scope={scope} passes={passes} 统计源=adapt(n={len(X_stat)})")
    for _ in range(passes):
        tr.adabn_recompute(model, X_stat, scope=scope)
    return True, scope, passes


def run_target_only(args, parts, run_dir):
    """target_only 对照：只用目标有标签文件从零训练（版本3 §5）。

    返回 (eval_res, baseline_acc=None, bn_state=(False,None,None))。
    target_only 无源域/适配语义，不适用 AdaBN（已在 validate_args 拦截）。
    """
    if len(parts["X_lab"]) == 0:
        raise SystemExit("target_only 必须有 label_ratio > 0 的目标有标签文件")
    model = build_model(args, parts)
    target_weights = compute_class_weights(args, parts["y_lab"])
    print(f"[目标从零训练] epochs={args.source_epochs} bs={args.bs}")
    tr.train_source(model, parts["X_lab"], parts["y_lab"],
                    epochs=args.source_epochs, bs=args.bs, lr=args.lr_source,
                    seed=args.seed, class_weights=target_weights)
    tr.save_checkpoint(os.path.join(run_dir, "target_only.pt"), model,
                       **_ck_kwargs(args, parts, "target_only"))
    eval_res = tr.evaluate_details(model, parts["Xt_eval"], parts["yt_eval"],
                                   record_ids=parts["eval_record_ids"])
    return eval_res, None, (False, None, None)


def run_transfer_pipeline(args, parts, run_dir, control, arm, class_weights):
    """源域固定训练 → 基线 → （可选）迁移微调 → （可选）AdaBN → 评估。

    返回 (eval_res, baseline_acc, bn_state=(on,scope,passes))。
    """
    model = build_model(args, parts)
    print(f"[阶段一] 源域固定训练 epochs={args.source_epochs} bs={args.bs} "
          f"lr={args.lr_source} class_weight={'on' if class_weights is not None else 'off'}")
    tr.train_source(model, parts["Xs_tr"], parts["ys_tr"],
                    epochs=args.source_epochs, bs=args.bs, lr=args.lr_source,
                    seed=args.seed, class_weights=class_weights)
    tr.save_checkpoint(os.path.join(run_dir, "source.pt"), model,
                       **_ck_kwargs(args, parts, "source"))
    base = tr.evaluate_details(model, parts["Xt_eval"], parts["yt_eval"])
    baseline_acc = round(base["accuracy"], 6)
    print(f"[基线] 不迁移 target FE acc={baseline_acc:.4f}")

    if control == "no_transfer":
        print("[阶段二] no_transfer：跳过目标微调")
    else:
        use_labels = control not in ("zero_label", "pure_mmd")
        has_lab = use_labels and len(parts["X_lab"]) > 0
        X_lab = parts["X_lab"] if has_lab else None
        y_lab = parts["y_lab"] if has_lab else None
        arm_cfg = _resolve_arm(arm, args)
        print(f"[阶段二] control={control} arm={arm} epochs={args.transfer_epochs} "
              f"lr={arm_cfg['lr_transfer']} lam={args.lam} "
              f"freeze={arm_cfg['freeze_backbone']} source_ce={arm_cfg['source_ce']} "
              f"src_w={arm_cfg['source_ce_weight']}")
        tr.train_transfer(
            model, parts["Xs_tr"], parts["ys_tr"], X_lab, y_lab,
            parts["X_unlab"] if args.lam != 0 else None,
            epochs=args.transfer_epochs, bs=args.bs, lr=arm_cfg["lr_transfer"],
            lam=args.lam, mmd_estimator=args.mmd_estimator,
            alignment_mode=args.alignment_mode, seed=args.seed,
            class_weights=class_weights, freeze_backbone=arm_cfg["freeze_backbone"],
            source_ce=arm_cfg["source_ce"], source_ce_weight=arm_cfg["source_ce_weight"])
    tr.save_checkpoint(os.path.join(run_dir, "transfer.pt"), model,
                       **_ck_kwargs(args, parts, "transfer"))
    # AdaBN（评估前用目标无标签输入重估 BN，版本3 §5 P0.1）
    bn_state = _maybe_adabn(args, model, parts)
    eval_res = tr.evaluate_details(model, parts["Xt_eval"], parts["yt_eval"],
                                   record_ids=parts["eval_record_ids"])
    print(f"[最终评估] acc={eval_res['accuracy']:.4f} macro-F1={eval_res['macro_f1']:.4f} "
          f"记录硬={eval_res['record_accuracy_hard']:.4f} 记录软={eval_res['record_accuracy_soft']:.4f}")
    return eval_res, baseline_acc, bn_state


def run_experiment(args):
    """执行一次 T1 实验并追加结果记录（含记录级即时计算）。"""
    started = time.time()
    control = getattr(args, "control", "semi_supervised")
    split_mode = getattr(args, "split_mode", "natural")
    run_name = args.run_name or default_run_name(
        "T1", args.seed, args.protocol, args.label_ratio, args.lam, control,
        split_mode=split_mode, label_pool_ratio=getattr(args, "label_pool_ratio", None),
        seg_len=args.seg_len, repr_kind=args.repr, tag=getattr(args, "tag", None),
        align=getattr(args, "alignment_mode", None))
    run_dir = create_run_directory(run_name, args.runs_dir)
    log_path = os.path.join(run_dir, "run.log")
    command = "python train.py " + " ".join(sys.argv[1:])
    arm = getattr(args, "arm", None)

    with configure_logging(log_path):
        print(f"# 命令: {command}")
        print(f"# 开始: {datetime.now().isoformat(timespec='seconds')}")
        print(f"# task=T1 repr={args.repr} protocol={args.protocol} split_mode={split_mode} "
              f"seed={args.seed} control={control} arm={arm} device={tr.DEV}")
        tr.seed_everything(args.seed)
        parts = prepare_domains(args)
        print(f"[表征] {args.repr} n_channels={parts['n_channels']} "
              f"input_dim={parts['input_dim']} frontend={_frontend_spec(_frontend_name(args))['name']}")
        print(f"[划分] mode={parts['split_mode']} label_ratio={args.label_ratio} "
              f"pool={parts['label_pool_ratio']}")
        print(f"[文件] labeled={len(parts['target_labeled_keys'])} "
              f"unused={len(parts['target_unused_keys'])} "
              f"eval={len(parts['target_eval_keys'])}")
        print(f"[样本] source={len(parts['Xs_tr'])} labeled={len(parts['X_lab'])} "
              f"adapt={len(parts['X_unlab'])} eval={len(parts['Xt_eval'])}")

        class_weights = compute_class_weights(args, parts["ys_tr"])

        if control == "target_only":
            eval_res, baseline_acc, bn_state = run_target_only(
                args, parts, run_dir)
        else:
            eval_res, baseline_acc, bn_state = run_transfer_pipeline(
                args, parts, run_dir, control, arm, class_weights)
        bn_adapt_on, bn_adapt_scope, bn_adapt_passes = bn_state

        per_file = eval_res.get("per_file_detail", {})
        record = {
            "run": run_name, "schema_version": tr.SCHEMA_VERSION, "task": "T1",
            "protocol": args.protocol, "control": control, "seed": args.seed,
            "split_mode": parts["split_mode"], "split_source": parts["split_source"],
            "label_ratio": args.label_ratio, "label_pool_ratio": parts["label_pool_ratio"],
            "adapt_ratio": parts["adapt_ratio"],
            "source_domain": parts["source_domain"], "target_domain": parts["target_domain"],
            "source_file_keys": parts["source_file_keys"],
            "target_label_pool_keys": parts["target_label_pool_keys"],
            "target_labeled_keys": parts["target_labeled_keys"],
            "target_unused_keys": parts["target_unused_keys"],
            "target_adapt_keys": parts["target_adapt_keys"],
            "target_eval_keys": parts["target_eval_keys"], "dev_file_keys": [],
            "n_source_files": len(parts["source_file_keys"]),
            "n_target_files": len(parts["source_file_keys"]),
            "n_label_pool_files": len(parts["target_label_pool_keys"]),
            "n_labeled_files": len(parts["target_labeled_keys"]),
            "n_unused_files": len(parts["target_unused_keys"]),
            "n_adapt_files": len(parts["target_adapt_keys"]),
            "n_eval_files": len(parts["target_eval_keys"]),
            "n_source": len(parts["Xs_tr"]), "n_labeled": len(parts["X_lab"]),
            "n_adapt": len(parts["X_unlab"]), "n_eval": len(parts["Xt_eval"]),
            "input_repr": args.repr, "repr_config": parts["repr_config"],
            "input_dim": parts["input_dim"], "n_channels": parts["n_channels"],
            "seg_len": args.seg_len, "K": parts["K"], "sort_vmd": parts["sort_vmd"],
            "recurrent_layers": args.recurrent_layers, "threshold_mode": args.threshold_mode,
            "use_se": bool(args.use_se), "frontend": _frontend_spec(_frontend_name(args)),
            "bs": args.bs, "lr_source": args.lr_source, "lr_transfer": args.lr_transfer,
            "source_epochs": args.source_epochs, "transfer_epochs": args.transfer_epochs,
            "lam": args.lam, "alignment_mode": args.alignment_mode,
            "mmd_estimator": args.mmd_estimator,
            "class_weight": "on" if class_weights is not None else "off", "arm": arm,
            "bn_adapt_on": bn_adapt_on, "bn_adapt_scope": bn_adapt_scope,
            "bn_adapt_passes": bn_adapt_passes,
            "record_accuracy_hard": eval_res.get("record_accuracy_hard"),
            "record_accuracy_soft": eval_res.get("record_accuracy_soft"),
            "record_macro_f1": eval_res.get("record_macro_f1"),
            "record_class_recall": eval_res.get("record_class_recall"),
            "per_file_accuracy": {k: v["acc_hard"] for k, v in per_file.items()},
            "per_file_support": {k: v["support"] for k, v in per_file.items()},
            "per_file_pred": {k: v["pred_hard"] for k, v in per_file.items()},
            "source_val_acc": None, "baseline_acc": baseline_acc,
            "transfer_acc": round(eval_res["accuracy"], 6),
            "macro_f1": round(eval_res["macro_f1"], 6),
            "class_recall": [round(v, 6) for v in eval_res["recall"]],
            "confusion_matrix": eval_res["confusion_matrix"].tolist(),
            "elapsed_s": round(time.time() - started, 1),
        }
        print(f"[产物] {log_path} / {run_dir}")

    results_path = append_result(record, runs_dir=args.runs_dir)
    print(f"[汇总] 已追加到 {results_path}")
    return record


def _self_check():
    """验证日志、同名拒绝、结果字段集合、run 命名。"""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        name = default_run_name("T1", 0, "transductive", 0.99, 0.3, "semi_supervised",
                                split_mode="paired", label_pool_ratio=0.5, seg_len=512,
                                repr_kind="logspec", tag="pilot")
        assert name == "T1_s0_paired_transductive_poolr0.5_lr0.99_lam0.3_semi_supervised_L512_logspec_pilot", name
        name2 = default_run_name("T1", 0, "transductive", 0.99, 0.3, "semi_supervised",
                                 split_mode="paired", label_pool_ratio=0.5, seg_len=512,
                                 repr_kind="logspec", align="conditional")
        assert name2.endswith("_L512_logspec_alignconditional"), name2
        run_dir = create_run_directory(name, tmp)
        with configure_logging(os.path.join(run_dir, "run.log")):
            print("日志")
        assert open(os.path.join(run_dir, "run.log"), encoding="utf-8").read().strip() == "日志"
        try:
            create_run_directory(name, tmp)
            raise AssertionError("同名 run 必须拒绝")
        except SystemExit:
            pass
        path = append_result({"run": "x", "task": "T1", "split_mode": "paired",
                              "transfer_acc": 0.9}, runs_dir=tmp)
        rows = json.load(open(path, encoding="utf-8"))
        assert set(rows[0]) == set(RESULT_FIELDS)
    print("ALL EXPERIMENT CHECKS PASSED")


if __name__ == "__main__":
    _self_check()
