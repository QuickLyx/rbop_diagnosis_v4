# rbop_diagnosis_v4 —— T1 表征、对齐与读出改造（重写仓库）

> 本仓库是 `../rbop_diagnosis`（旧库）的**重写版**。旧库只作为数据、历史证据
> （E12–E21）与结果锚定来源，**不作为执行依据**。一切设计以
> `docs/PLAN_T1_表征对齐与读出改造_版本3.md`（下称"版本3"）与
> `docs/EXEC_PLAN_T1_可执行执行清单.md`（执行清单）为准。

## 1. 任务定义（T1）

- **任务**：CWRU `12k_DE/Normal` 数据集，**同一批文件**从驱动端（DE）迁移到风扇端（FE），
  四分类（Normal / Ball / InnerRace / OuterRace）。即"同文件跨测点 DE→FE"。
- **达标口径（95%）**：**段级 accuracy**，在 **paired 协议 / pool0.5 / 固定标签池 /
  per-seed 评估集（池的补集）** 上 ≥ 0.95。记录级为**辅助/部署语义指标**，不作为达标依据。
  禁止在段级/记录级之间切换口径择优。
- **协议边界**：transductive 允许使用评估文件的 FE **无标签输入**做适配（AdaBN/对齐/自训练）；
  inductive 的适配/评估互斥。评估文件的 FE **标签**从不进入训练。

## 2. 技术路线（一句话）

谱表征（logspec/cepstrum）+ 谱前端 v1（保频结构池化）+ 域对齐（AdaBN / MMD 对齐模式 / 自训练）
+ 记录级读出。详见版本3 §2 归因与 §5 改动方案。

## 3. 当前里程碑

严格串行推进 W0 → W4（版本3 §4.8、执行清单 §C）：

| 里程碑 | 内容 | 状态 |
|---|---|---|
| W0 | 数据与划分（Data 复制、dataset.py、划分等价 128 run、段统计 13191） | ✅ 完成（见 docs/evidence/W0_data_split_audit.txt） |
| W1 | 表征与探针（preprocessing.py 唯一提取、E22 复现、VMD 缓存等价） | ✅ 完成（E22 对拍 E21 = 0.0pp） |
| W2 | 训练管线（model.py 双 frontend、trainer/experiment/train、VMD 主格基线） | ✅ 完成（seed0 锤定差 0.00pp） |
| W3 | 读出与汇总（记录级即时产出、summarize.py、注意力可视化） | ✅ 完成（33 测试+6 自检全通过） |
| W4 | 进入实验（P2.1 pilot → M1 → P0.1/P0.2 → 网格 → P2.2 → P2.3 → M3） | ✅ 完成（M3 通过：8 seed 段级 0.9904，8/8 ≥0.95） |

## 4. 目录结构

```
rbop_diagnosis_v4/
  Data/                     # 从旧库整体复制（657 MB，56 文件 / 13191 DE 段）
  cache/                    # v4 自建；缓存键含完整 repr_config
  dataset.py                # 枚举、T1 不变量、划分（split_target_files）、SegmentSet
  preprocessing.py          # repr 提取（vmd/logspec/cepstrum/vmdstats）+ 缓存 + repr_config
  vmd.py                    # VMD 实现（从旧库移植）
  model.py                  # 双 frontend（spectral_v1 / legacy_vmd）+ RSU/SE + BiLSTM + 注意力 + MMD
  trainer.py                # train_source / train_transfer / evaluate_details / checkpoint(schema_version=1)
  experiment.py             # run 命名 / prepare_domains / run_experiment / 结果写入（含记录级）
  train.py                  # CLI
  visualize_attention.py    # 频带重要性可视化
  scripts/                  # run_sweep / diag_protocol_probe / adabn_eval / summarize / regen_cache
  tests/                    # 见 §4.7 of 版本3
  docs/                     # 协议、计划、evidence（E12–E21 引用旧库，E22/E23 v4 新增）
  runs/{natural,paired}/    # results.json + run 子目录
```

## 5. 数据来源

- `Data/` 从 `../rbop_diagnosis/Data` 整体复制（657 MB，含 `12k_DE/Normal`）。
- **56 条记录**：`12k_DE` 排除全部 028、`Normal` 全保留；四类 4/12/12/28；每条同时含 DE/FE。
- `seg_len=512`、不重叠；Normal 原始 48 kHz 先降采样到 12 kHz 再切段；DE 共 **13191 段**。
- `cache/` 不复制，v4 自建并与旧库缓存抽样对拍（版本3 §4.2 迁移审计）。

## 6. 环境与常用命令

```powershell
$PY = "E:/tool/anaconda/envs/fault-detect/python.exe"
$env:PYTHONUTF8 = "1"
Set-Location E:/文献/近两三年的故障诊断文献/rbop_diagnosis_v4

& $PY dataset.py            # 数据自检
& $PY preprocessing.py      # 表征自检
& $PY model.py              # 模型自检
& $PY trainer.py            # 训练器自检
& $PY experiment.py         # 编排自检
& $PY -m unittest discover -s tests -v
```

完整复现命令骨架见版本3 §12（`--reprs/--alignment_modes/--keys_only/--arms/--pseudo_*` 为 v4 新增 CLI）。

AdaBN 已接入主通路（train.py）：`--bn_adapt [--bn_adapt_scope all|frontend] [--bn_adapt_passes N]`，
评估前用目标无标签适配集重估 BN running stats，结果写入 results.json 的 `bn_adapt_on/scope/passes`。
只支持 transductive，且与 `--control target_only` 互斥。旁路脚本 `scripts/adabn_eval.py` 仍可对既有 ckpt 做成对扫描。

## 7. run 命名与 results.json schema 摘要

- run 名：`T1_s{seed}_{split_mode}_{protocol}[_poolr{p}]_lr{r}_lam{l}_{control}_L{seg}_{repr}[_{tag}]`
  （`lr` 指 label_ratio、`lam` 指 MMD 权重，均**不是学习率**）。
- `runs/natural/` 与 `runs/paired/` 严格分写，**不得混表**；禁止删 run 重跑。
- results.json 每条含：运行元数据 + 完整文件键 + 表征与模型（`input_repr/repr_config/frontend/...`）
  + 适配与读出（`bn_adapt_*` 由 `--bn_adapt` 主通路填充、记录级 `record_accuracy_hard/soft`、
  `record_class_recall`（记录级逐类召回）等）+ 指标。详见版本3 §4.5。
- checkpoint：`schema_version:1`，含 `repr_config` 与 `frontend`；不读旧库 `format_version=2`。

## 8. seed 规则

- **开发 seed = 0–3**：pilot、网格、A–E、自训练选型只用这四个。
- **确认 seed = 100–103**（划分种子 100103–100106）：M3 前**任何指标都不看**，
  只在冻结后的流水线上跑，与开发 4 seed 合并报告 8 seed。

## 9. 证据索引

| 编号 | 内容 | 位置 |
|---|---|---|
| E12–E20 | 旧库历史证据（数据/物理/探针/网络） | `../rbop_diagnosis/docs/evidence/` |
| E21 | paired 协议匹配探针（三表征） | 同上 |
| E22（v4 新增） | E21 的 v4 复现 + 全网格划分键 | `docs/evidence/E22_protocol_probe_v4.*` |
| E23（v4 新增） | 迁移审计：旧主格网络 run 的 v4 重跑锚定 | `docs/evidence/E23_migration_anchor.txt` |

## 10. 结果概览（2026-09-29 完成）

主格（logspec + spectral_v1，paired/pool0.5/lr0.99/λ0.3/semi/confidence）：

| seeds | 段级 accuracy | macro-F1 | 记录级硬投票 |
|---|---|---|---|
| 开发 0–3 | 0.9872 ± 0.0078 | 0.9894 | 0.9929 |
| 确认 100–103 | 0.9935 ± 0.0050 | 0.9920 | 1.0000 |
| **合并 8 seed** | **0.9904 ± 0.0070**（95% CI [0.9845, 0.9962]，8/8 ≥0.95） | 0.9920 | 0.9955 |

相对 v4 VMD 基线（0.8147）开发 4 seed 配对差值 +0.1725（4/4 正）。
关键归因：表征收益 +13.25pp（logspec vs vmd）> 前端收益 +4.01pp。
证据：`docs/evidence/M3_final_result.txt`、`M1_pilot_result.txt`、`P21_grid_result.txt`、
`P22_arms_result.txt`、`P02_align_summary.md`、`P01_adabn_summary.md`、
`E22_protocol_probe_v4.txt`、`E23_migration_anchor.txt`。

## 11. 不做的事

- 不实现 T2/T3（无可运行入口）。
- 不兼容旧库 checkpoint（`format_version=2`）；旧权重由旧库读取。
- 不做段级随机划分（存在相邻段跨集泄漏）；不为达标切回段级随机 train/test。
- 不引用旧网络数字作为 v4 结果；网络级结论必须由 v4 协议重跑。
