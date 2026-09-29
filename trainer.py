# -*- coding: utf-8 -*-
"""训练、评估和 checkpoint（schema_version=1）。

设计依据：版本3 §4.5（checkpoint/结果 schema，v4 起点无向后兼容）、§5 P0.0（固定轮数、
不做早停）、§P1.1（记录级读出即时计算）、§5 P0.1（AdaBN 分块精确统计）。

要点：
- checkpoint 用 `schema_version:1`，含 `repr_config` 与 `frontend`；不读旧库 format_version=2。
- `evaluate_details` 额外返回逐段概率 + 记录索引，供 experiment 即时算记录级指标（版本3 §P1.1）。
- `adabn_recompute` 用分块精确统计重估 BatchNorm running stats（版本3 §5 P0.1，防 4GB OOM）。
"""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dataset import LABEL_NAMES
from model import conditional_mmd_loss

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SCHEMA_VERSION = 1
_DEFAULT = object()


def seed_everything(seed):
    """固定 Python、NumPy、PyTorch 和 cuDNN 随机性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_dataloader(X, y=None, bs=64, shuffle=False, seed=0):
    """构造 `(x,y)` 或只含 `x` 的 DataLoader；输入形状 (N,C,D)。"""
    if X is None or len(X) == 0:
        raise ValueError("不能为零样本张量创建 DataLoader")
    generator = torch.Generator().manual_seed(seed)
    data = TensorDataset(X) if y is None else TensorDataset(X, y)
    return DataLoader(data, batch_size=bs, shuffle=shuffle, generator=generator)


def _next_batch(iterator, loader, device=DEV):
    """取下一个 batch，耗尽后循环重启，并搬到目标设备。"""
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    if isinstance(batch, (list, tuple)):
        moved = tuple(item.to(device) for item in batch)
        return (moved[0] if len(moved) == 1 else moved), iterator
    return batch.to(device), iterator


# ---------------------------------------------------------------------------
# 评估（含逐段概率 + 记录索引，供记录级读出）
# ---------------------------------------------------------------------------
def evaluate_details(model, X, y, bs=256, record_ids=None):
    """返回段级指标 + 逐段概率 + 记录索引（版本3 §P1.1 记录级即时计算）。

    返回 dict：accuracy/macro_f1/recall/confusion_matrix/support（段级）+ probs (N,4)/
    pred (N,)/true (N,)。若传 record_ids（(N,) 段→记录下标），额外返回记录级
    record_accuracy_hard/soft、record_macro_f1、record_class_recall、per-record 明细。
    """
    from sklearn.metrics import confusion_matrix, f1_score

    if X is None or y is None or len(X) == 0:
        raise ValueError("评估集为空，无法计算指标")
    model.eval()
    probs_all = []
    with torch.no_grad():
        for start in range(0, len(X), bs):
            logits = model(X[start:start + bs].to(DEV))[0]
            probs_all.append(torch.softmax(logits, dim=1).cpu())
    probs = torch.cat(probs_all).numpy()
    pred = probs.argmax(1)
    true = y.detach().cpu().numpy()
    labels = np.arange(len(LABEL_NAMES))
    cm = confusion_matrix(true, pred, labels=labels)
    support = cm.sum(axis=1)
    recall = np.divide(np.diag(cm), support, out=np.zeros(len(labels), dtype=float),
                       where=support != 0)
    out = {
        "accuracy": float((pred == true).mean()),
        "macro_f1": float(f1_score(true, pred, labels=labels, average="macro",
                                    zero_division=0)),
        "recall": recall.tolist(),
        "confusion_matrix": cm,
        "support": support.tolist(),
        "probs": probs, "pred": pred, "true": true,
    }
    if record_ids is not None:
        out.update(_record_level_metrics(probs, pred, true, np.asarray(record_ids)))
    return out


def _record_level_metrics(probs, pred, true, record_ids):
    """记录等权的硬/软投票指标（版本3 §P1.1/P1.2）。硬软分别，不混淆。"""
    from sklearn.metrics import f1_score

    rec_hard_pred, rec_soft_pred, rec_true = [], [], []
    per_file = {}
    for rid in np.unique(record_ids):
        mask = record_ids == rid
        counts = np.bincount(pred[mask], minlength=probs.shape[1])
        hard = int(counts.argmax())
        soft = int(probs[mask].mean(0).argmax())
        t = int(true[mask][0])
        rec_hard_pred.append(hard)
        rec_soft_pred.append(soft)
        rec_true.append(t)
        per_file[int(rid)] = {"acc_hard": int(hard == t), "support": int(mask.sum()),
                              "pred_hard": hard, "pred_soft": soft, "true": t}
    rec_hard_pred = np.array(rec_hard_pred)
    rec_soft_pred = np.array(rec_soft_pred)
    rec_true = np.array(rec_true)
    labels = np.arange(probs.shape[1])
    # 记录级低于 0.5 的记录数（版本3 §P1.2）——记录级单条命中为 0/1，这里统计错误记录数
    n_wrong_hard = int((rec_hard_pred != rec_true).sum())
    return {
        "record_accuracy_hard": float((rec_hard_pred == rec_true).mean()),
        "record_accuracy_soft": float((rec_soft_pred == rec_true).mean()),
        "record_macro_f1": float(f1_score(rec_true, rec_hard_pred, labels=labels,
                                          average="macro", zero_division=0)),
        "record_n_wrong_hard": n_wrong_hard,
        "record_n_total": int(len(rec_true)),
        "per_file_detail": per_file,
    }


def _as_weight(class_weights):
    """把类别权重转为设备上的 float32 张量。"""
    if class_weights is None:
        return None
    return torch.as_tensor(class_weights, dtype=torch.float32, device=DEV)


def train_source(model, X_tr, y_tr, epochs=30, bs=64, lr=1e-3, seed=0,
                 class_weights=None):
    """按固定轮数训练源域 CE（不做早停，版本3 §5 P0.0）。"""
    if X_tr is None or len(X_tr) == 0:
        raise ValueError("源域训练集为空")
    weight = _as_weight(class_weights)
    loader = make_dataloader(X_tr, y_tr, bs=bs, shuffle=True, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        started = time.time()
        for xb, yb in loader:
            xb, yb = xb.to(DEV), yb.to(DEV)
            loss = F.cross_entropy(model(xb)[0], yb, weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"ep{epoch + 1:02d} loss={total / len(loader):.4f} "
              f"({time.time() - started:.1f}s)", flush=True)


def train_transfer(model, X_s, y_s, X_lab=None, y_lab=None, X_unlab=None,
                   epochs=30, bs=64, lr=1e-4, lam=0.3,
                   mmd_estimator="biased", alignment_mode="confidence", seed=0,
                   class_weights=None, freeze_backbone=False, source_ce=True,
                   source_ce_weight=1.0):
    """训练 `source_ce_weight·CE(source) + CE(target labeled) + lam·MMD`。

    支持 P2.2 五臂（版本3 §5 P2.2）：
      freeze_backbone=True → 只训分类头（B 臂）；
      source_ce=False → 去源 CE、保留 MMD 与 backbone 训练（C 臂，MMD 仍需源域前向取 feat_s/y_s）；
      source_ce_weight∈{0.1,0.3} → D 臂（源 CE 梯度尺度）。
    返回 (ce_src, ce_tgt, mmd) 三个批均损失。
    """
    if X_s is None or len(X_s) == 0:
        raise ValueError("源域训练集为空")
    semi = X_lab is not None and y_lab is not None and len(X_lab) > 0
    use_mmd = float(lam) != 0.0
    if use_mmd and (X_unlab is None or len(X_unlab) == 0):
        raise ValueError("lam 非零时必须提供非空目标无标签适配集")
    weight = _as_weight(class_weights)

    if freeze_backbone:
        # B 臂：冻结除分类头外的所有参数
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("head")
    params = [p for p in model.parameters() if p.requires_grad]

    source_loader = make_dataloader(X_s, y_s, bs=bs, shuffle=True, seed=seed)
    unlab_loader = (make_dataloader(X_unlab, None, bs=bs, shuffle=True, seed=seed + 1)
                    if use_mmd else None)
    lab_loader = (make_dataloader(X_lab, y_lab, bs=bs, shuffle=True, seed=seed + 2)
                  if semi else None)
    it_unlab = iter(unlab_loader) if unlab_loader is not None else None
    it_lab = iter(lab_loader) if lab_loader is not None else None
    opt = torch.optim.Adam(params, lr=lr)
    stats = [0.0, 0.0, 0.0]

    for epoch in range(epochs):
        model.train()
        total = [0.0, 0.0, 0.0]
        started = time.time()
        for xb, yb in source_loader:
            xb, yb = xb.to(DEV), yb.to(DEV)
            logits_s, _, feat_s = model(xb, return_feat=True)
            loss = xb.new_zeros(())
            if source_ce:
                ce_src = F.cross_entropy(logits_s, yb, weight=weight)
                loss = loss + source_ce_weight * ce_src
                total[0] += ce_src.item()
            if semi:
                (x_lab, y_lab_batch), it_lab = _next_batch(it_lab, lab_loader)
                ce_tgt = F.cross_entropy(model(x_lab)[0], y_lab_batch, weight=weight)
                loss = loss + ce_tgt
                total[1] += ce_tgt.item()
            if use_mmd:
                x_target, it_unlab = _next_batch(it_unlab, unlab_loader)
                logits_t, _, feat_t = model(x_target, return_feat=True)
                mmd = conditional_mmd_loss(feat_s, yb, feat_t, logits_t,
                                           mode=alignment_mode, estimator=mmd_estimator)
                loss = loss + float(lam) * mmd
                total[2] += mmd.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        stats = [value / len(source_loader) for value in total]
        print(f"ep{epoch + 1:02d} ce_src={stats[0]:.4f} ce_tgt={stats[1]:.4f} "
              f"mmd={stats[2]:.4f} ({time.time() - started:.1f}s)", flush=True)

    if freeze_backbone:  # 复原 requires_grad
        for p in model.parameters():
            p.requires_grad = True
    return stats[0], stats[1], stats[2]


# ---------------------------------------------------------------------------
# AdaBN：分块精确统计重估（版本3 §5 P0.1）
# ---------------------------------------------------------------------------
def adabn_recompute(model, X_unlab, scope="all", chunk=2048):
    """用无标签目标输入重估 BatchNorm running stats（分块精确统计，防 4GB OOM）。

    规程（版本3 §5 P0.1）：model.train()+no_grad 前向，手工按样本数加权累计精确均值/方差
    写入 running_mean/running_var；scope='all' 重估全部 7 个 BN，'frontend' 只重估前端 3 个。
    禁止逐记录统计。结束后 model.eval()。
    """
    bns = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.BatchNorm1d):
            if scope == "all" or (scope == "frontend" and name.startswith("frontend")):
                bns.append((name, mod))
    if not bns:
        return model
    # 累计器：sum 和 sumsq，按每个 BN 的通道维
    acc = {name: None for name, _ in bns}
    handles = []
    captured = {}

    def make_hook(name):
        def hook(module, inp, out):
            x = inp[0].detach()  # (B,C,L)
            captured[name] = (x.sum(dim=(0, 2)), (x * x).sum(dim=(0, 2)),
                              x.shape[0] * x.shape[2])
        return hook

    for name, mod in bns:
        handles.append(mod.register_forward_hook(make_hook(name)))

    model.train()
    counts = {name: 0 for name, _ in bns}
    with torch.no_grad():
        for start in range(0, len(X_unlab), chunk):
            captured.clear()
            model(X_unlab[start:start + chunk].to(DEV))
            for name, _ in bns:
                s, sq, cnt = captured[name]
                if acc[name] is None:
                    acc[name] = [s.clone(), sq.clone(), cnt]
                else:
                    acc[name][0] += s
                    acc[name][1] += sq
                    acc[name][2] += cnt
    for h in handles:
        h.remove()
    for name, mod in bns:
        s, sq, n = acc[name]
        mean = s / n
        var = sq / n - mean * mean
        mod.running_mean.copy_(mean)
        mod.running_var.copy_(var.clamp_min(0))
        if mod.num_batches_tracked is not None:
            mod.num_batches_tracked.zero_()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Checkpoint（schema_version=1）
# ---------------------------------------------------------------------------
def _checkpoint_config(*, stage, source, target, seg_len, input_repr, repr_config,
                       n_channels, input_dim, n_classes, class_names, recurrent_layers,
                       threshold_mode, use_se, frontend, K=1, sort_vmd=False, task="T1"):
    """生成 checkpoint 中除权重外的完整配置（版本3 §4.5）。"""
    if class_names is None:
        class_names = list(LABEL_NAMES)
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": str(stage), "task": str(task),
        "source": source, "target": target,
        "seg_len": int(seg_len),
        "input_repr": str(input_repr),
        "repr_config": dict(repr_config),
        "n_channels": int(n_channels), "input_dim": int(input_dim),
        "K": int(K), "sort_vmd": bool(sort_vmd),
        "n_classes": int(n_classes), "class_names": list(class_names),
        "recurrent_layers": int(recurrent_layers),
        "threshold_mode": str(threshold_mode), "use_se": bool(use_se),
        "frontend": dict(frontend),
    }


def save_checkpoint(path, model, **cfg):
    """保存 schema_version=1 的权重和元数据。cfg 见 _checkpoint_config 参数。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    config = _checkpoint_config(**cfg)
    torch.save({"state_dict": model.state_dict(), **config}, path)


def load_checkpoint(path, model, *, strict_fields=("schema_version", "task",
                    "input_repr", "n_channels", "input_dim", "n_classes")):
    """按 schema_version 校验关键字段后加载权重（版本3 §4.5：不读旧库 format_version=2）。"""
    blob = torch.load(path, map_location=DEV, weights_only=False)
    if not isinstance(blob, dict) or "state_dict" not in blob:
        raise SystemExit(f"{path} 不是本项目的 checkpoint 格式（缺 state_dict）")
    if blob.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit(f"{path} schema_version={blob.get('schema_version')} "
                         f"与当前 {SCHEMA_VERSION} 不符；v4 不读旧库 checkpoint")
    model.load_state_dict(blob["state_dict"])
    return blob


def load_checkpoint_config(path):
    """读取 checkpoint 元数据，不加载权重。"""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or "state_dict" not in blob:
        raise SystemExit(f"{path} 不是本项目的 checkpoint 格式（缺 state_dict）")
    return {k: v for k, v in blob.items() if k != "state_dict"}


def _self_check():
    """最小自检：固定轮数、零标签零 MMD、schema round-trip 与拒绝、AdaBN、记录级。"""
    import tempfile
    from model import MSRSABiLSTM
    import preprocessing as pp

    seed_everything(0)
    X = torch.randn(16, 1, 256)
    y = torch.arange(16) % 4
    rid = np.arange(16) // 4  # 4 条记录
    model = MSRSABiLSTM(n_channels=1, frontend="spectral_v1").to(DEV)
    cfg = pp.build_repr_config("logspec")
    ck = dict(stage="source", source=None, target=None, seg_len=512,
              input_repr="logspec", repr_config=cfg, n_channels=1, input_dim=256,
              n_classes=4, class_names=list(LABEL_NAMES), recurrent_layers=1,
              threshold_mode="signed_mean", use_se=True,
              frontend={"name": "spectral_v1", "channels": [32, 64, 128],
                        "adaptive_out": 8})
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "source.pt")
        save_checkpoint(path, model, **ck)
        conf = load_checkpoint_config(path)
        assert conf["schema_version"] == 1 and conf["input_repr"] == "logspec"
        blob = load_checkpoint(path, model)
        assert blob["frontend"]["name"] == "spectral_v1"
        # 篡改 schema_version 应被拒
        bad = torch.load(path, weights_only=False)
        bad["schema_version"] = 2
        torch.save(bad, path)
        try:
            load_checkpoint(path, model)
            raise AssertionError("schema_version 不符必须拒绝")
        except SystemExit:
            pass

    # 零标签零 MMD 路径
    train_transfer(model, X, y, None, None, None, epochs=1, bs=8, lam=0.0)
    # 记录级评估
    res = evaluate_details(model, X, y, record_ids=rid)
    assert "record_accuracy_hard" in res and res["record_n_total"] == 4
    # AdaBN 分块精确统计不崩、running_var 非负
    adabn_recompute(model, X, scope="all", chunk=8)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            assert torch.all(m.running_var >= 0)
    print("ALL TRAINER CHECKS PASSED")


if __name__ == "__main__":
    _self_check()
