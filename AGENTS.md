# AGENTS.md —— rbop_diagnosis_v4 仓库级代理指令

> 本文件是 v4 仓库的代理执行规则。设计依据以 `docs/PLAN_T1_表征对齐与读出改造_版本3.md`
> （版本3）与 `docs/EXEC_PLAN_T1_可执行执行清单.md`（执行清单）为准。旧库
> `../rbop_diagnosis` 只读，作历史证据与结果锚定，**不得原样搬运其 README/AGENTS/注释**。

## 1. 语言规则

- 所有输出、解释、总结一律**简体中文**。
- 代码注释、docstring、commit message 用中文。
- 返回给用户的字符串（error message / 日志 / UI 文案）用中文。
- 标识符、类型、变量名、函数名保持**英文**（语法要求），不做中文命名。

## 2. 范围

- 只维护 **T1**（同文件跨测点 DE→FE 四分类）。不实现 T2/T3。
- 达标口径唯一：**paired / pool0.5 / lr0.99 / λ=0.3 / semi 的段级 accuracy ≥0.95**（开发 seeds 0–3）。
  记录级为辅助。禁止在段级/记录级之间切口径择优。

## 3. 环境与工作目录

- Python：`E:/tool/anaconda/envs/fault-detect/python.exe`，`PYTHONUTF8=1`。
- 工作目录：`E:/文献/近两三年的故障诊断文献/rbop_diagnosis_v4`。
- 单卡（GTX 1650 Ti，4GB）串行执行，不做并行抢占。

## 4. T1 不变量全集（必须逐条实现并用测试锁死）

数据层（版本3 §4.2）：
1. `12k_DE` 排除全部 028；`Normal` 单独枚举；每文件必须同时有 DE/FE。
2. Normal 原始 48 kHz 先降采样到 12 kHz，再切段。
3. 56 个文件、四类 4/12/12/28；`seg_len=512`、不重叠；DE 共 **13191 段**。
4. `--label_ratio` 是**每类目标文件比例**，不是段比例；目标评估标签不得进入训练。
5. transductive 允许评估文件的 FE **输入**做无标签适配；inductive 的适配/评估互斥。
6. 同文件 DE 可以进源域（源域 = 全 56 文件 DE）。
7. paired 语义：先按 `label_pool_ratio` 划固定标签池，池的补集为固定评估集；
   再按池内 `label_ratio` 取有标签文件；池内未取用文件记 `unused`（不进入任何路径）。
8. 划分数值必须与旧库一致：`split_target_files(records, ..., seed+100003)`，逐键比对。

schema 与命名（版本3 §4.5/§4.6）：
- checkpoint `schema_version:1`，`repr_config` 全量入键与 checkpoint；`frontend` 名入 checkpoint。
- 非 vmd 表征时 `--K` 固定 1、`--sort_vmd` 强制 False。
- run 名：`T1_s{seed}_{split_mode}_{protocol}[_poolr{p}]_lr{r}_lam{l}_{control}_L{seg}_{repr}[_{tag}]`；
  `lr`=label_ratio、`lam`=MMD 权重，均不是学习率。
- `runs/natural/` 与 `runs/paired/` 严格分写；禁止删 run 重跑；禁止把自然划分结果复制到 paired。

seed 规则（版本3 §7）：
- 开发 seed 0–3；确认 seed 100–103（划分种子 100103–100106），**开发阶段一律不使用、不看指标**。

## 5. 迁移审计门槛（W0–W2 验收数字，版本3 §4.2）

| 项 | 通过标准 |
|---|---|
| 划分等价 | seeds 0–3 × pool{0.3,0.5} × lr{0,0.3,0.6,0.99} 的键与旧库全 128 run **0 不一致** |
| 段统计等价 | 各文件段数、总段数 **13191** 完全一致 |
| logspec/cepstrum 等价 | E22 探针与 E21 同 seed/预算对拍，段级差 **≤1pp、逐 seed 同号** |
| VMD 缓存等价 | 抽样 3–5 文件与旧库 npz 形状/标签/段号一致、`allclose(rtol=1e-5)` |
| 网络重跑锚定 | 主格 seed0 段级/记录级与旧 run **≤2pp**（差异需书面解释） |

**未过门槛不得进入实验（W4）。**

## 6. 代码注释 / docstring 规范

- 一律中文；标识符/类型/文件名保持英文。
- 每个模块头部 docstring 说明**设计依据**（指向版本3 章节/证据编号）与数据流形状，不写复读代码的废话。
- 与旧库冲突的注释**不得原样带入**，典型：
  - checkpoint `format_version=2` → v4 是 `schema_version=1`；
  - `K` 的 VMD 通道语义 → 非 vmd 表征 `K` 无意义（固定 1）；
  - 三级 /4 池化 → 谱前端 v1 用 (2,4)+自适应8；仅 legacy_vmd 保留 (4,4,4)；
  - `sort_vmd` 只适用于 vmd 表征；
  - 非 vmd 表征的 K 语义。
- 函数注释写明参数形状与不变量（如 `repr_config` 必需字段、`AdaptiveMaxPool1d(8)` 的用途）。
- 每个可执行模块保留 `_self_check()` + `__main__` 入口，便于 CI/本地自检。

## 7. 文档同步规则与归档边界

- 任何 schema、CLI、目录规则或不变量变更，必须在**同一提交**内同步
  `README.md`/`AGENTS.md`/`tests`，并在 `docs/evidence/` 落盘对应验收结果。文档滞后视为未完成。
- 旧库 `../rbop_diagnosis` **只读**：只作历史证据（E12–E21）与结果锚定来源，不改、不引用其路径进 v4 代码。
- 每个里程碑（W0–W4）验收结果写入 `docs/evidence/`，不允许跳过里程碑。
