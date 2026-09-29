# T1 表征、对齐与读出改造方案（版本3，目标代码库 rbop_diagnosis_v4）

## 0. 文档定位、前提与 v3 相对 v2 的变更

本文取代 `../rbop_diagnosis/docs/PLAN_T1_表征对齐与读出改造_版本2.md`（下称 v2），
作为**新代码库 `rbop_diagnosis_v4`** 的执行依据。v1/v2 保留作历史对照，不再作为执行依据。

**前提（用户已确认）**：代码在根目录 `rbop_diagnosis_v4/` 重新实现，旧库只提供
数据、证据文本与历史结果；v3 据此对 v2 做以下结构性调整：

- v2 §5 的"v2 checkpoint 兼容读取、旧 run 命名兼容、2026-09-26 批次边界"等**遗留约束全部消除**；
  改为在 v4 中**从零设计 schema**，并增加把旧证据/旧结果用作锚点的**迁移审计**（§4.2、§4.5）；
- 旧库 `docs/evidence/E12–E21` 仍是数据层与探针客观实的唯一历史来源（正文以相对路径引用）；
  凡网络级结论必须由 v4 协议重跑，禁止引用旧网络数字作为 v4 结果；
- 95% 目标口径不变：**paired 协议、pool0.5、固定标签池、per-seed 评估集（池的补集）上的段级
  accuracy ≥0.95**；记录级为辅助与部署语义指标，不作为达标依据，禁止在段级/记录级之间切换口径择优。

### 0.1 v3 相对 v2 的变更清单（均为已核验的事实修正或设计补强）

| # | v2 内容 | v3 处理 |
|---|---|---|
| 1 | 称 E21 划分键"与 128 个网络 run 逐键核对一致" | **证据不落地**：`E21_protocol_probe_keys.json` 只含 12 个划分（pool0.5 × lr∈{0.3,0.6,0.99} × 4 seed），只能核对 48 个 run。已复算全部 128 run 的划分与 `runs/paired/results.json` 0 不一致；v3 要求 v4 生成**全网格划分键**（E22）并把口径写准 |
| 2 | 称 E21“硬投票与 softmax 均值投票数值一致” | **错误**:倒谱 target-only lr0.99 为 0.9554 vs 0.9464;倒谱 source+target lr0.3 为 0.8750 vs 0.8482;VMD 统计量 source+target lr0.99 为 0.8125 vs 0.8214。v3 按实际口径改写 |
| 3 | AdaBN"momentum=None 时与批顺序、分组无关" | **过强**：`momentum=None` 是批统计的累积平均，末尾不满批等权、其组成随 shuffle 变化；v3 改为"分块/整批显式精确统计"规程（§5 P0.1） |
| 4 | P2.2 判据称 B 臂支持"源 CE 有害" | **归因错位**：B 同时去掉 backbone 微调与特征适配；源 CE 净效应只能由 C 臂给出。v3 重写判据（§5 P2.2） |
| 5 | 表征只以 `repr_kind` 标识 | **不安全**：logspec/cepstrum 的具体参数（窗、nbin、对数偏移、段内标准化、倒谱取段）会改变结果；v3 冻结完整 `repr_config` 并进入缓存键/checkpoint（§4.3、§4.5） |
| 6 | P3.1“应减小或去掉频轴池化”（未定量） | v3 给出可验证配方并已冻结：**首级池化 stride ≤ 2，谱分支 (2,4)+自适应 8 → 8 步**，BPFO/BPFI 的 bin 数学见 §4.4 |
| 7 | 48 run 的 VMD 对齐网格排在 pilot 之前 | 与 v2 自己“结论表征特定、必须在最终表征上复跑”矛盾；v3 把对齐网格**移到选定表征**上跑，VMD 网格降级为条件项（§5 P0.2、§6）；**已确认** |
| 8 | M3“关键格扩到 8 seed” | 存在选择-评估污染风险；v3 **预留 seed 100–103（新编号）为确认样本**，开发阶段（含网格）只用 0–3，且要求披露配置选择路径（§7、§8） |
| 9 | 浅层谱探针只作机制参照 | v3 把 **logspec/cepstrum + 逻辑回归（含可选伪标签自训练）列为正式基线**，与网络同格同 seed 报告（§5 P2.1、§9） |
| 10 | λ=0、能量通道、自训练细节未定 | v3 把 semi λ=0（新表征）纳入 pilot 波次；能量通道列为可选消融；自训练阈值/类别配额/轮数预登记（§5 P2.1/P2.3、P3.4） |

### 0.2 已冻结决策（用户已确认，执行时不再更改）

| 项 | 决策 |
|---|---|
| 前端 v1 | 谱分支 **SMCBlock(32,64,128) + MaxPool(2) → MaxPool(4) → AdaptiveMaxPool(8)**，固定 8 步；logspec 256 下与 (2,4,4) 等价；VMD 分支保持旧前端 (4,4,4) 等构复现 |
| 对齐网格 | VMD 网格降级为条件项；默认只在最终表征上跑 `{global, conditional}` 主格（confidence 复用 pilot） |
| 确认 seed | **100–103（新编号；划分种子 100103–100106）**，开发阶段一律不使用 |
| 数据 | `Data/` 整体复制进 v4（657 MB）；**cache 不复制**，v4 自建并与旧库缓存抽样对拍 |
| schema | v4 起点 `schema_version: 1`；不读旧库 checkpoint；`repr_config` 全量入缓存键与 checkpoint |
| 表征 | logspec 首选、cepstrum 并列（pilot 都跑）；vmdstats 仅最后备选；vmd 仅作锚定与历史对照 |

---

## 1. 决定性证据（历史来源，数字已逐条复核）

### 1.1 E12–E20（旧库 `../rbop_diagnosis/docs/evidence/`，复核一致）

| 测什么 | 结果 | 出处 |
|---|---|---|
| 源域模型在 DE 全 56 记录 | acc **0.9993** | E18 |
| 同一网络在 DE 文件级 5 折 CV（从零训练） | **0.9817 ± 0.0202** | E13 |
| 同一网络在 FE 文件级 5 折 CV（从零训练） | **0.8782 ± 0.1367**，逐折 0.633/0.825/0.992/0.955/0.986 | E13 |
| 冻结 DE 特征上 FE 的线性探针上限 | seed0 **0.7080**、seed1 **0.6724** | E12 |
| log 谱 256 段级探针（FE 域内、文件级 5 折） | 段级 **0.9564 ± 0.0205**，F1 0.9611；记录硬投票 **0.982**；每记录均值特征 **1.000（56/56）** | E14 |
| 窗口长度对照 | L=512/1024/2048 均 **1.000**，L=4096 为 **0.985** | E14 |
| 固定评估集标签预算（每类 1/2/3 条标签记录） | 段级 0.5192/0.6135/0.6757；OuterRace 召回 0.29/0.34/**0.42** | E14 |
| 零标签对齐方法族（探针） | 一阶对齐 / CORAL / 类别条件 / 条件+原型：logspec **0.786** / 0.706 / 0.472 / 0.416；cepstrum 0.767 / 0.569 / 0.441 / 0.360；VMD 0.273 / **0.384** / 0.258 / 0.249 | E17 |
| 表征矩阵（域内 CV / 零标签迁移） | VMD 展平 0.402 / 0.276；logspec 0.962 / 0.588；cepstrum 0.955 / 0.640 | E16 |
| z-score 与 sort_vmd 消融 | 保留幅值：域内 0.9667 vs 0.9624，零标签 0.4675 vs 0.5883；排序 vs 不排序 0.402/0.394（展平）、0.830/0.807（统计量） | E16 |
| RSU 阈值未退化 | tau 跨样本 CV 0.374/0.267；逐样本极差中位 0.666/0.570；近饱和 4.3%/2.3% | E17 |
| 失败形态 | Ball 的 12 条 FE 记录逐条准确率 **全部 0.000**（DE 上 10 条 1.000、B014_3=0.992、B021_3=0.971、Ball 类 DE 0.997） | E18 |
| 物理频率判据 | 谐波峰/底噪比 Ball DE 136/FE 161、IR 607/418、OR 116/105；免训练"最强故障族"判据 DE 0.519 / FE 0.462；RMS（g）DE 0.060/0.139/0.296/0.347、FE 0.072/0.091/0.174/0.185 | E19 |
| 错分流向（主格 4 seed 聚合） | InnerRace→OuterRace **14.7%（n=839）**；主格段级 **0.8147 ± 0.0801**、F1 0.7812；target-only 0.8308；no-transfer 0.4727（pool0.5） | E20 |

**两条轴必须分开**：E13/E14 是 FE 域内可分性；E16/E17/E21 是 DE→FE 可迁移性。

**L2 计数（按 `runs/paired/results.json` 复算）**：同池同 seed 同预算下 target-only 段级
acc 高于 semi 的比例——λ=0：**18/24**（macro-F1 亦 18/24）；λ=0.3：**12/24**（14/24）；
λ=1.0：**10/24**（13/24）；主格上 target-only 0.8308 > semi λ=0.3 0.8147。

### 1.2 E21（协议匹配探针，旧库已落盘；v4 需以 E22 复现）

脚本 `../rbop_diagnosis/scripts/diag_protocol_probe.py`，证据
`../rbop_diagnosis/docs/evidence/E21_protocol_probe.txt`。与 `train.py` 共用
`split_target_files(paired, pool, label_ratio, seed+100003)`；训练只用标签池文件的目标标签
与全部 56 文件 DE 段；评估文件的 FE 标签从未进入训练；CPU 逻辑回归（C=0.3、StandardScaler）。

**lr=0.99（4 seed，pool0.5）**：

| 变体 | logspec 256 | cepstrum 236 | vmdstats 32 |
|---|---:|---:|---:|
| FE 有标签文件 only | 0.8994 ± 0.0454（F1 0.9140，硬 0.9375） | 0.8997 ± 0.0360（F1 0.9133，硬 **0.9554**、软 0.9464） | 0.7719 ± 0.0575 |
| 同上 + DE 全 13191 段 | **0.9202 ± 0.0204（F1 0.9307，硬 1.0000）** | **0.9255 ± 0.0258（F1 0.9345，硬 1.0000）** | 0.7276 ± 0.0527（硬 0.8125、软 0.8214） |
| 零标签不对齐 | 0.5893 ± 0.0216 | 0.7043 ± 0.0066 | 0.4790 ± 0.0215 |
| 零标签 + 一阶对齐（全 FE / eval 统计） | 0.7539 / 0.7559 | 0.7270 / 0.7310 | 0.4173 / 0.3964 |

低预算（段级均值，logspec）：lr=0.3 / 0.6 / 0.99 的 FE-only 为 0.6739/0.8417/0.8994，
+DE 为 0.7890/0.8583/0.9202（记录硬 0.8661/0.9018/1.0000）。
逐类召回区间（+DE 变体）：logspec Ball 0.83–0.93、IR 0.90–0.96、OR 0.90–0.96；
cepstrum Ball 0.85–0.93、IR 0.88–0.95、OR 0.90–0.97；FE-only 的 IR 在 seed2 低至 0.58。

**三条结论**：

1. 同协议同标签下，浅层谱模型已达段级 ~0.90–0.93、记录级 0.94–1.00，**显著高于现行网络**
   （0.8147），表征路线收益空间真实存在；
2. **L2 是表征特定的**：VMD 上线性格局与谱表征不同，P2.2 必须在选定表征上重做；
3. **95%（段级）尚未被浅层模型闭合**（含源域 0.92），剩余 2.5–3pp 需要"网络非线性 + transductive
   无标签适配 + 保谱结构前端"三者叠加——此句是**待检验假设**，不是预期值。

**E21 的局限（如实披露）**：单一 C、未调参、线性模型、n=28 记录；keys JSON 只覆盖 12 个划分
（见 §0.1#1）；除 logspec 外硬/软投票不等价（见 §0.1#2）；一阶对齐在"全 FE 统计"下使用了
池文件的无标签 FE 输入（transductive，符合 T1 不变量）。

---

## 2. 归因

**L1（有直接证据）输入表征压制了可分性。**
- 域内轴：VMD 时域输入 + 完整网络 = 0.878（E13）；logspec + LR = 0.962（E14/E16）；
  VMD 展平探针仅 0.402。VMD 波形对线性模型几乎不可读，必须靠 CNN 重建，而重建方向在 DE 上学习。
- 跨域轴：零标签 DE→FE 不对齐 0.276（VMD）vs 0.588（logspec）；一阶对齐后 0.273 vs 0.786
  （E17，探针口径）；E21 同协议 0.90–0.93 vs 网络 0.81–0.83。
- E12 的 0.708/0.672 是**冻结网络表示**的上限证据，不是原始信号上限。

**L2（观察成立、机制未定、表征特定）源域分支未被证明有正贡献。**
观察见 §1.1 的 18/24、12/24、10/24 计数；但限定在 VMD 表征；E21 显示谱表征 + DE 在 4/4 seed
记录级不低于 FE-only（各变体 112/112 条记录正确）。源域 CE 是有害偏置还是有益正则，
由 §5 P2.2 的匹配实验判定（C 臂是唯一干净检验，见 §5）。

**L3（仅零标签探针口径成立；网络级 `conditional` 从未跑过）对齐方法族之间有实质差距。**
E17 见 §1.1；网络代码语义已核实：`conditional` 按 `probs[:,c]>=0.5` 硬掩膜、
`confidence` 用 softmax 概率软加权。最优对齐方法随表征变化，因此对齐结论必须在最终表征上复跑。

---

## 3. 已排除的解释（不要重复调查；沿用 v2，数字复核一致）

| 假设 | 判据 | 结论 |
|---|---|---|
| 网络容量不足 | DE 文件级 CV 0.982（E13） | 否 |
| VMD 丢失 FE 故障谱峰 | E19 谐波峰/底噪比 | 否 |
| 免训练频率规则可达 95% | DE 0.519 / FE 0.462（E19） | 否 |
| 512 点分辨率不足 | L=512/1024/2048 均 1.000（E14） | 否，不要改 `seg_len` |
| 逐段 z-score 有害 | 零标签迁移 −12pp（E16） | 否，保留段内标准化 |
| `reorder_vmd_modes` 有害 | E16 +1~2pp | 否，VMD 路径保留排序 |
| RSU 阈值退化 | E17 | 否 |
| 按记录归一化能闭合域差 | 0.956→0.506/0.502（E14） | 否，且有害 |
| 域内可分性不足是瓶颈 | E21 浅层 0.90–0.93 vs 网络 0.81 | 否；瓶颈在管线 |

**按记录归一化有害的两层机制**：段内标准化后的 log 谱幅值已被除掉，被破坏的是记录间的
平均谱形差异；对未归一化特征还会抹掉幅值判据（E19）。**只能把整个目标域当作一个分布做全局对齐。**

---

## 4. v4 代码库设计（替代 v2 §5 的遗留兼容条目）

### 4.1 目录与模块

```
rbop_diagnosis_v4/
  Data/                     # 已确认：从旧库整体复制（657 MB，见 §4.2 数据迁移）
  cache/                    # v4 自建（不复制旧缓存）；键含完整 repr_config
  dataset.py                # 枚举、T1 不变量、划分（split_target_files）、SegmentSet
  preprocessing.py          # repr 提取（vmd/logspec/cepstrum/vmdstats）+ 缓存 + repr_config
  vmd.py                    # VMD 实现（从旧库移植；数值需过 §4.2 审计）
  model.py                  # FrontEnd v1 + RSU/SE + BiLSTM + AdditiveAttention + 分类头 + MMD
  trainer.py                # train_source / train_transfer / evaluate_details / checkpoint
  experiment.py             # default_run_name / prepare_domains / run_experiment / 结果写入
  train.py                  # CLI
  visualize_attention.py    # 频带重要性可视化
  scripts/
    run_sweep.py            # 含 --alignment_modes、--reprs、--tag
    diag_protocol_probe.py  # E21 同协议探针（E22 复现 + 全网格划分键）
    adabn_eval.py           # AdaBN 评估（含分块精确统计）
    summarize.py            # 汇总 + 配对差值 + Holm + cluster bootstrap + --self-check
    regen_cache.py          # --reprs 参数
  tests/                    # 见 §4.7
  docs/                     # 协议、计划、evidence（含从旧库复制/引用的 E12–E21）
  runs/{natural,paired}/    # results.json + run 子目录，schema 见 §4.5
```

### 4.2 数据层不变量与迁移审计

**数据迁移（已确认：整体复制）**：v4 不引用旧库路径，`Data/` 从 `../rbop_diagnosis/Data` 整体
复制（657 MB；含 `12k_DE/12k_FE/48k_DE/Normal` 共 161 个 .mat）。

1. 复制：`robocopy ..\rbop_diagnosis\Data Data /E`（或等价方式）；
2. 清单校验：`enumerate_t1_files` 必须得到 **56 条记录**（12k_DE 排除全部 028、Normal 全保留），
   且每条记录同时含 `DE_time` 与 `FE_time`；
3. 生成 `docs/data_manifest.json`（文件键 + 字节数），与旧库清单比对；
4. **缓存不在复制范围**：v4 自建；与旧库缓存的等价性由下方审计表抽样验证。

**不变量（必须逐条实现并用测试锁死）**：

1. `12k_DE` 排除全部 028；`Normal` 单独枚举；每文件必须同时有 DE/FE；
2. Normal 原始 48 kHz 先降采样到 12 kHz，再切段；
3. 56 个文件、四类 4/12/12/28；`seg_len=512`、不重叠；DE 共 13191 段；
4. `--label_ratio` 是**每类目标文件比例**，不是段比例；目标评估标签不得进入训练；
5. transductive 允许评估文件的 FE **输入**做无标签适配；inductive 的适配/评估互斥；
6. 同文件 DE 可以进源域（源域 = 全 56 文件 DE）；
7. paired 语义：先按 `label_pool_ratio` 划固定标签池，**池的补集为固定评估集**，
   再按池内 `label_ratio` 取有标签文件，池内未取用文件记 `unused`（不进入任何路径）；
8. 划分数值必须与旧库一致：`split_target_files(records, ..., seed+100003)`，逐键比对。

**迁移审计（进入实验前的硬门槛）**：

| 项 | 方法 | 通过标准 |
|---|---|---|
| 划分等价 | v4 生成 seeds 0–3 × pool{0.3,0.5} × lr{0,0.3,0.6,0.99} 的 labeled/eval 键，与旧库 `../rbop_diagnosis/runs/paired/results.json` 全 128 run 比对 | 0 不一致 |
| 段统计等价 | 各文件段数、总段数 13191 | 完全一致 |
| logspec/cepstrum 等价 | 复用 E22 探针：v4 特征跑同协议 LR，与 E21 同 seed/预算对拍 | 段级差 ≤1pp，逐 seed 同号 |
| VMD 缓存等价 | 抽样 3–5 文件，与旧库 `../rbop_diagnosis/cache/` 对应 npz 比对 | 形状/标签/段号一致、数值 allclose（rtol=1e-5） |
| 网络重跑锚定 | VMD、主格（pool0.5/lr0.99/λ=0.3）、seed0（属 §5 P0.0 的 4 run 基线） | 段级/记录级与旧 run 差 ≤2pp，差异需书面解释 |

### 4.3 表征层：repr_config 冻结 + 共享提取函数

`preprocessing.py` 提供唯一实现，探针脚本与网络共用（禁止两处各写一份）：

- `vmd`：`normalize_segment`（逐段 z-score）→ VMD(K=8) → 读出后 `sort_vmd=True`（排序不入缓存内容，与旧库口径一致，便于缓存等价性比对）；
- `logspec`：Hanning 窗 → `rfft`(512) → 幅值 + 1e-9 → `linspace(0,256,256)` 取样（int 截断）
  → `log` → **段内跨维标准化**（mean/std，eps 1e-9）→ float32；(1,256)；
- `cepstrum`：同 logspec 幅谱（nbin=L=512）→ `log` → `irfft` → 取 **低倒频 2..119 + 高倒频
  394..511（236 维）** → 段内标准化；(1,236)；
- `vmdstats`：每模态 4 统计量（log 能量、峭度、谱质心、高频占比），(1,32)。

`repr_config` 必须整体序列化（窗型、nbin、对数偏移、标准化、倒谱取段、K/sort 等）：

- 缓存键：`cache/<dataset>/<channel>/<stem>_L<seg_len>_<repr_key>.npz`，其中 `repr_key`
  含完整参数（如 `logspec_hanning_nbin256_log1e-9_std1`）；npz 内同时存 `repr_config`，
  读取时逐字段校验；
- checkpoint/结果同样存 `repr_config`（§4.5）。

### 4.4 网络与前端 v1 配方（P2.1 的组成部分）

**结构（已冻结，谱分支 v1）**：`c_in=1`（可选 2，见 P3.4）→ SMCBlock(32, 64, 128) +
**MaxPool1d(2) → MaxPool1d(4) → AdaptiveMaxPool1d(8)** → (B,128,8) → RSU×2 → SE →
BiLSTM(1×128, 双向) → 加性注意力 → 256 维特征 → 分类头(256→64→ReLU→Dropout 0.5→4)。

- logspec 256：/2 得 128 → /4 得 32 → 末级自适应等价 /4 → 8 步；
- cepstrum 236：118 → 29 → 自适应到 8（**不要补零硬凑 256**，会引入假频段）；
- 末级用 `AdaptiveMaxPool1d(8)`（而不是固定 /4）是为了让两个并列候选共享同一架构与 8 步注意力；
  对 logspec 256 它与 (2,4,4) 逐点等价；
- **vmd 分支保持旧前端 (4,4,4)**（输入 (B,8,512) → 128→32→8）：迁移锚定要求等构复现，
  VMD 数据流一律不改；frontend 类型写入 checkpoint 与结果以区分。

**为什么首级池化必须是 /2**（12 kHz、512 点、Δf=23.4375 Hz；rfft 0 索引 bin；系数取 E19 脚本
`COEF={"BPFO": 3.5847, "BPFI": 5.4265}`，fr 在全部工况为 28.7–30.0 Hz，下列 bin 指数不变）：

| 故障频率 | 频率 | bin | 首级 /2（池 2） | 首级 /4（池 4） |
|---|---:|---:|---:|---|
| BPFO | 103–107.5 Hz（3.5847·fr） | 4 | 池 2 | 池 1 |
| BPFI | 156–163 Hz（5.4265·fr） | 6 | 池 3 | 池 1 |

即首级 /4 会把二者并入同一池（v2 的担忧成立）；谱前端 v1（(2,4)+自适应 8，logspec 256 下
等价于 (2,4,4)）在保住二者可分的同时把序列长度
压到 8，与旧网络的 LSTM/注意力形状一致，改动面最小。池化后频轴不再是时间轴：
注意力权重解释为**频带重要性**（可视化路径同步更新）。

**归因中间臂（必须跑）**：新表征 + 旧前端（`c_in=1`、池化 (4,4,4)，序列长度 4）——用于把
"表征收益"与"前端收益"分开；该臂首级 /4 会把 BPFO/BPFI 并入同一池（事实），若其明显更差，则
P3.1 的必要性被实测确认；若与主配方接近，则说明首级池化不是主因。该臂不与主配方混表。

### 4.5 checkpoint 与结果 schema（v4 起点，无向后兼容负担）

**checkpoint（`schema_version: 1`）**：`schema_version, stage, task, source, target, seg_len,
input_repr, repr_config, n_channels, input_dim, K（仅 vmd 语义）, sort_vmd, n_classes,
class_names, recurrent_layers, threshold_mode, use_se, frontend（name=spectral_v1|legacy_vmd, channels, pools, adaptive_out=8）`。
`load_checkpoint` 按 `schema_version` 匹配校验；不提供读旧库 `format_version=2` 的路径
（旧权重由旧库读取，不在 v4 流程内）。

**results.json 每条记录**（`summarize.py` 以 `row.get` 容忍缺省字段）：

- 运行元数据：`run, schema_version, task, protocol, control, seed, split_mode, split_source,
  label_ratio, label_pool_ratio, adapt_ratio, source_domain, target_domain` 与完整文件键
  （`source_file_keys, target_label_pool_keys, target_labeled_keys, target_unused_keys,
  target_adapt_keys, target_eval_keys`）+ 各种 `n_*`；
- 表征与模型：`input_repr, repr_config, input_dim, n_channels, seg_len, K, sort_vmd,
  recurrent_layers, threshold_mode, use_se, frontend, bs, lr_source, lr_transfer,
  source_epochs, transfer_epochs, lam, alignment_mode, mmd_estimator, class_weight`；
- 适配与读出：`bn_adapt_on, bn_adapt_scope, bn_adapt_passes`、
  记录级 `record_accuracy_hard/soft, record_macro_f1, record_class_recall,
  per_file_accuracy/support/pred`；
- 指标：`source_val_acc, baseline_acc, transfer_acc, macro_f1, class_recall, confusion_matrix,
  elapsed_s`。
- 记录级指标在**每次 run 结束时即时计算**（`evaluate_details` 返回逐段概率 + 记录索引），
  不再事后重评旧 checkpoint。

### 4.6 CLI 与 run 命名规则

- `train.py --repr {vmd,logspec,cepstrum,vmdstats}`；非 vmd 时 `--K` 固定 1、`--sort_vmd` 强制 False；
  v4 用 `n_channels/input_dim` 构造与校验模型，防止同形状不同语义混用；
- `run_sweep.py --alignment_modes`、`--reprs`；run 名统一：
  `T1_s{seed}_{split_mode}_{protocol}[_poolr{p}]_lr{r}_lam{l}_{control}_L{seg}_{repr}[_{tag}]`
  （`lr` 指 label_ratio、`lam` 指 MMD 权重，均不是学习率）；
- 新批次以 `repr`/`am`/`adb`/`pl` 等 tag 进 run 名，`runs/natural/` 与 `runs/paired/` 严格分写，
  **不得混表**；禁止删除已有 run 后重跑，禁止把自然划分结果复制到 paired。

### 4.7 测试与自检

- `tests/test_dataset.py`：划分等价（§4.2）、段统计、缓存键隔离、paired 语义；
- `tests/test_preprocessing.py`：repr 形状/键/缓存校验、与固定 fixture 的数值一致；
- `tests/test_model.py`：(1,256)/(8,512) 前向、注意力归一、MMD 三模式与估计器有限值、小 batch 保护；
- `tests/test_trainer.py`：checkpoint round-trip 与 schema 拒绝、零标签零 MMD 路径；
- `tests/test_experiment.py`：run 名唯一、结果字段完整、目录不覆盖、分模式隔离；
- 自检：`python model.py`、`dataset.py`、`preprocessing.py`、`trainer.py`、`experiment.py`、
  `scripts/summarize.py --self-check`，每次新 run 批次前跑全套。

### 4.8 v4 落地路线（实现里程碑）

| 里程碑 | 交付物 | 验收门槛 |
|---|---|---|
| W0 数据与划分 | Data 复制 + `docs/data_manifest.json`；`dataset.py`（枚举/划分/SegmentSet）；`tests/test_dataset.py`；**`README.md`/`AGENTS.md` 按 §4.9 重写（初版已随 v3 生成，W0 核对定稿）** | 56 记录且 DE/FE 成对；128 run 划分键 0 不一致；DE 共 13191 段 |
| W1 表征与探针 | `preprocessing.py` + `regen_cache.py` + `scripts/diag_protocol_probe.py`；E22 证据 | E22 对拍 E21 ≤1pp 且方向一致；VMD 缓存抽样与旧库 allclose |
| W2 训练管线 | `model.py`（谱 v1 + VMD 旧前端）、`trainer.py`、`experiment.py`、`train.py`、`tests`；VMD 主格基线 4 run | seed0 锚定与旧 run ≤2pp；结果 schema 字段齐备 |
| W3 读出与汇总 | 记录级指标随 run 产出；`summarize.py`（配对/Holm/bootstrap）；`visualize_attention.py` | 全套 tests + 自检通过 |
| W4 进入实验 | §5 P2.1 pilot（20 run）+ 浅层基线 | 过 M1 才铺网格；未过按 M1 分支处理 |

严格按 W0→W4 推进；每个里程碑必须同步更新 `README.md`/`AGENTS.md` 与新增代码注释（§4.9），验收结果写入 `docs/evidence/`，不允许跳过。

### 4.9 文档与注释规范（README / AGENTS / 代码注释必须按本方案重写）

**原则**：v4 是重写仓库，**不允许原样搬运旧库的 README/AGENTS/注释**；所有文字以本方案
（含 §0.2 冻结决策）为准，旧库只作历史证据与对照来源。

1. **`README.md`（仓库门面）**必须包含：
   - 任务定义（T1：12k DE→FE 同文件跨测点四分类）、95% 段级口径与 transductive/paired 协议边界；
   - 技术路线一句话（谱表征 + 谱前端 v1 + 对齐/AdaBN/自训练 + 记录级读出）与当前里程碑 W0–W4；
   - 目录结构、数据来源（复制自旧库；56 文件 / 13191 段）、环境与最常用命令；
   - run 命名与 `results.json` schema 摘要；开发 seed 0–3 与**确认 seed 100–103 开发阶段禁用**的规则；
   - 证据索引（旧库 E12–E21、v4 的 E22/E23）与"不做的事"（T2/T3、旧 checkpoint 兼容、段级随机划分）。
2. **`AGENTS.md`（仓库级代理指令）**必须包含：
   - 语言规则（输出/注释/文档中文；标识符英文）；范围（只维护 T1）；
   - 环境与工作目录；T1 不变量全集（§4.2 不变量 + §4.5 schema + §4.6 命名规则 + §7 seed 规则）；
   - 迁移审计门槛（W0–W2 验收数字）与"未过门槛不得进入实验"；
   - 代码注释/docstring 规范（下条）；文档同步规则与归档边界（旧库只读）。
3. **代码注释/docstring**：
   - 一律中文；标识符/类型/文件名保持英文；
   - 每个模块头部 docstring 说明**设计依据**（指向本方案章节/证据编号）与数据流形状，不写复读代码的废话；
   - 与旧库冲突的注释**不得原样带入**（典型：checkpoint `format_version=2`、`K`=VMD 通道语义、
     三级 /4 池化、`sort_vmd` 的适用范围、非 VMD 表征的 K 语义）；
   - 函数注释写明参数形状与不变量（如 `repr_config` 必需字段、`AdaptiveMaxPool1d(8)` 的用途）；
   - 每个可执行模块保留 `_self_check()` + `__main__` 入口，便于 CI/本地自检。
4. **文档同步规则**：任何 schema、CLI、目录规则或不变量变更，必须在同一提交内同步
   `README.md`/`AGENTS.md`/`tests`，并在 `docs/evidence/` 落盘对应验收结果；文档滞后视为未完成。

---

## 5. 改动方案（按 v3 重排）

### P0（基建 + 近零成本）

**P0.0 schema/表征/读出基建 + E22 复现。**

- 落地 §4 全部设计；跑迁移审计（§4.2）；
- 跑 **v4 自跑的 VMD 主格基线**（pool0.5/lr0.99/λ=0.3/semi，seeds 0–3，4 run）：M1/M2/M3 的
  paired 差值一律相对这条同代码基线计算，历史 0.8147 只作锚定核对（两者差应 ≤2pp）；其中 seed0
  同时充当迁移审计的网络锚定；
- 复现 E21 为 **E22**：三表征、pool0.5/lr{0.3,0.6,0.99}、seeds 0–3，并**补全 pool0.3 与
  lr=0 的划分键**（全网格 keys JSON）；
- 判据：E22 与 E21 同 seed/预算对拍，段级差 ≤1pp、方向一致；否则先修实现。

**P0.1 目标域 BatchNorm 统计重估（AdaBN）。**

- **规程**：`model.train()` + `torch.no_grad()`，对无标签目标输入前向；用**分块（如每块 2048）
  手工累计精确统计量**（或整块不整除时手工按样本数加权）写入 running stats；结束后 `model.eval()`；
  记录 `num_batches_tracked` 与块组织。禁止逐记录统计的推理。
- 因子设计（4 个小因子，秒级/项）：
  - checkpoint：① source（无迁移）② transfer；
  - 范围：全部 7 个 BatchNorm1d / 仅前端 3 个；
  - 统计来源：仅 adapt 文件（paired 下 = 评估文件 28 条）/ 全 56 文件 FE 输入；
  - 遍数：1 / 3 / 10。
- 披露：`bn_adapt_on: eval_files|all_target_inputs`；transductive 使用评估输入的 FE 无标签输入。
- 判据：与同 run "重估前"成对报告，4 seed 全列方向；**不得把 E17 的 0.786 当预期值**。

**P0.2 对齐模式对照（在选定表征上跑）。**

- 时机调整：排在 P2.1 pilot 之后，直接在新表征上跑 `mode ∈ {global, conditional}` × 主格
  （pool0.5/lr0.99/λ=0.3）× seeds 0–3 = **8 run**；可选加 pool0.5/lr0.6 = 8 run。
  3 模式家族由此闭合：`confidence` 基线直接复用 pilot 中同配置的 semi run，不需要另跑。
- VMD 网格降级：仅当论文需要 VMD 历史机制对照时，补 `conditional` 主格 4 run（不铺 48 run）。
- 不重跑 λ=0：`lam=0` 时 `use_mmd=False`，`alignment_mode` 不参与计算。
- 判据：同 seed 配对差值；家族预注册（3 模式两两对比 × 单元数）并 Holm 校正；L3 的最终结论以
  本网格为准，跑完前只能写"探针层假设"。

### P1（记录级读出与统计，随 P0.0 一并落地）

- **P1.1**：`record_accuracy_hard = mean_records(majority(argmax 段预测) == 真值)`；
  `record_accuracy_soft = mean_records(argmax(mean 段概率) == 真值)`；均记录等权；不混淆两者
  （E21 已证倒谱/VMD 上硬软不同）。记录级不单独作主指标。
- **P1.2**：报告误差在记录上的聚集（低于 0.5 的记录数、分位数）。
- **P1.3**：`summarize.py` 支持按类分层的 cluster bootstrap（Normal 仅 2 条、离散度大，需标注），
  仅在需要向"总体记录"外推时使用。

### P2（决定能否接近/达到 95%）

**P2.1 输入表征换成谱形 + 前端协同（pilot 与网格）。**

- 选型以 E22（复现后的 E21）为准：logspec 与 cepstrum **并列候选，pilot 都要跑**
  （不再用"3pp 差才回退倒谱"的串行规则）；VMD 统计量仅作最后备选。
- **pilot（主格 pool0.5/lr0.99/λ=0.3，seeds 0–3）**：
  1. logspec + 谱前端 v1（(2,4)+自适应 8）：4 run；
  2. cepstrum + 谱前端 v1（末级自适应吸收 236→8）：4 run；
  3. 中间臂（logspec + 旧前端 (4,4,4)）：4 run，用于归因；
  4. target-only（新表征，lam=0）：4 run（logspec）；
  5. **semi λ=0（新表征）：4 run**（L2 复判的最短路径）；
  6. no-transfer + AdaBN：复用 pilot 的 `source.pt`，不新增训练。
- **浅层基线（正式报告行）**：logspec/cepstrum + LR，同格同 seed（CPU，分钟级）；
  可选对浅层模型做伪标签自训练（与 P2.3 同规程），作为"最便宜的 95% 竞争者"。
- pilot 过 M1 后再铺 `lr ∈ {0.3,0.6,0.99}` 与 pool0.3 的必要格（12–24 run）。
- 参数量与训练超参：bs/Adam/30 轮/MMD biased/`class_weight=off`/`use_se=True`/
  `threshold_mode=signed_mean` 沿用冻结配置；**按谱输入重扫的 `lr_transfer` 必须作为独立
  预登记消融臂**，不得混入主格。

**P2.2 源域分支作用的对照实验（新表征上 A–E；VMD 仅补 A/B/C 可选）。**

五臂（匹配源 loader 207 步/epoch、同 bs、同 30 轮、目标 batch 数与采样一致）：

| 臂 | 内容 | 实现要点 |
|---|---|---|
| A | 源 CE + 目标 CE + λ·MMD | 基线 |
| B | 冻结 backbone，只训分类头 | 去源监督 + 去特征适配的**整体参照**；对应 E12 的冻结表示上限 |
| C | 去源 CE，保留 MMD 与 backbone 训练 | **源 CE 净效应的唯一干净检验**（MMD 仍需源域前向取 `feat_s`/`y_s`） |
| D | 源 CE loss 系数 0.1/0.3 | 步数不变、仅梯度尺度变化（等效学习率） |
| E | `lr_transfer` 1e-4 → 1e-3 + warmup + 梯度裁剪 | 探索性，不并入正式结论 |

**判据（修正 v2 的归因映射）**：
- C > A ⇒ 源 CE 有害（论文结论只能由此给出）；
- B ≈ A ⇒ 冻结特征不劣于联合微调（与"源 CE 有害"是两回事）；
- B 最差 ⇒ 源域监督+微调整体有益；
- 三种结果都写进论文，n=4 以效应方向与配对差值为主。

**P2.3 自训练/伪标签（M2 未达 95% 时启动；规程预登记）。**

- 伪标签来源：当前模型对 adapt 输入（paired 下 = 评估文件 FE）的段级概率；
- 阈值 τ∈{0.9,0.95}；**按类别均衡配额**（预登记默认：每类最多取 `min(该类阈值以上段数, 1000)`；
  adapt 集各类段数约 Normal 355 / Ball 1424 / IR 1425 / OR 3332（seed0），因此该规则等价于
  Normal 全取、其余三类各封顶 1000，防 OR 占 50% 把 accuracy 注水）；
- 混合：目标 CE + `w_pl`·伪标签 CE，`w_pl∈{0.3,1.0}`；1–2 轮；固定步数；
- 每轮记录伪标签类别分布与记录级一致性；
- 披露：使用评估记录的无标签输入与自身伪标签，transductive 适配；
- 判据：同 seed 配对差值 + macro-F1/Ball 召回，防止只涨 accuracy。

### P3（条件性）

**P3.1 前端协同**：已并入 §4.4 与 P2.1（默认谱前端 v1：MaxPool(2)→MaxPool(4)→自适应 8；中间臂归因）。cepstrum 的 236 维是
低/高倒频拼接，序列轴中点不连续，可视化与文字须写明。

**P3.2 类别再平衡 / logit adjustment（独立消融）**：先验只能来自标签池或均匀，禁止用评估集
真实比例；开启后必须同时报告 accuracy 与 macro-F1 的 trade-off；Ball 召回单独报
（E19 显示 Ball 物理弱性）。

**P3.3 多分辨率与集成（可选，仅冲 95%）**：同一 `seg_len` 下 256/512 点窗 log 谱双通道；
多种子/多窗推理集成（transductive 允许，作为独立消融并披露）。

**P3.4（可选）能量通道消融**：输入 `(B,2,256)` = 段内标准化 log 谱 + 逐段 log-RMS 通道。
E19 显示幅值有判别力、E16 显示它会伤零标签迁移，因此只作独立消融，不进主格。

---

## 6. 实验清单与成本（GTX 1650 Ti；以旧库实测 semi ≈8.2 min/run 为基准，±30%）

| 阶段 | run 数 | 说明 | 估计 |
|---|---:|---|---|
| P0.0 迁移审计 + E22 + VMD 主格基线 | 4（含锚定 run） | CPU 探针 + 4 复跑 | 1–1.5 h |
| P0.1 AdaBN | 0 新训练 | 2 ckpt × 2 范围 × 2 统计来源 × 3 遍，秒级 | <0.5 h |
| P2.1 pilot（含 λ=0） | 20 | logspec/cepstrum/中间臂/target-only/λ=0 | 2.5–3 h |
| P0.2 对齐（新表征主格） | 8（+8 可选） | global/conditional × 4 seed | 1–2 h |
| P2.1 网格扩展 | 12–24 | lr 与 pool0.3 必要格 | 1.5–3 h |
| P2.2 五臂（新表征；VMD 补 12 可选） | 20（+12） | A–E × 4 seed | 3–4.5 h |
| P2.3 自训练 | 8–12 | τ × w_pl 的必要组合 | 1–1.5 h |
| M3 确认（seeds 100–103） | 4–8 | 冻结流水线（开发阶段禁用） | 0.5–1 h |
| P3.3 集成（可选） | 8–16 | 独立消融 | 1–2 h |
| **合计** | **≈75–135**（含可选项） | | **≈10–18 h GPU** |

执行顺序：P0.0（含 VMD 主格基线 4 run）→ P2.1 pilot（过 M1 后）→ P0.1/P0.2（新表征）→
P2.1 网格 → P2.2 → P2.3（按需）→ M3 确认 → P3.3（可选）。单卡串行，不做并行抢占。

---

## 7. 统计与确认协议

1. **开发 seed = 0–3**；**确认 seed = 100–103（新编号；划分种子 100103–100106）**，开发阶段
   （pilot、网格、A–E、自训练选型）一律不看 100–103 的任何指标；M3 只在冻结后的流水线上跑
   100–103，并与开发 4 seed 合并报告 8 seed 联合结果。
2. 主判据一律为**同 seed 配对差值**（右侧−左侧）；报告逐 seed、均值 ± 样本 SD、seed 级 t 的
   名义 95% CI；n=4/8 时 p 值只作探索性辅助。
3. 多重比较：预先声明家族（如 P0.2 的 3 模式两两对比 × 单元数；P2.2 的 `{B,C,D0.1,D0.3,E} vs A`
   共 5 个对比），家族内 Holm step-down；**不做事后挑单元**。
4. 评估集组成不确定性：按类分层的按记录 cluster bootstrap（P1.3），随最终表格报告。
5. **配置选择路径披露**：正文列明"看过哪些格、按什么规则选的主配方"，并声明确认 seed 未被用于选择。

---

## 8. 验收判据（95% = 段级；三级里程碑）

**主验收格**：`paired / pool0.5 / label_ratio=0.99 / λ=0.3 / semi_supervised`（开发 seeds 0–3）。
**次验收格**：pool0.3/lr0.99 与 pool0.5/lr0.6（稳健性，不用于唯一达标论证）。
**主格历史基线（旧库，仅作锚定）**：段级 0.8147 ± 0.0801（F1 0.7812；四类召回 N 0.9271 /
Ball 0.7537 / IR 0.7979 / OR 0.8355）。**M1/M2/M3 的 paired 差值一律相对 v4 自跑的 VMD 主格
基线（§5 P0.0，4 run）计算。**

### M1（表征落地；P2.1 pilot 后）

- 主格段级均值 **≥0.90**，且相对 v4 VMD 基线的配对差值均值 **≥+0.05**、**4/4 seed 为正**
  （历史 0.8147 仅作锚定核对）；
- macro-F1 **≥0.88**；Ball / IR 召回不低于 **v4 VMD 基线同格 4 seed 均值**
  （历史参考值 0.7537 / 0.7979）；
- **管线健康检查**：网络段级 ≥ E22 浅层探针（同表征、FE-only、同划分）− 3pp；若明显低于探针，
  先修前端/优化再铺网格；
- 0.85–0.90：先看中间臂归因（前端 vs 表征）再决定；<0.85：表征路线降级，回到对齐与读出（P0/P1），
  记录失败证据。

### M2（对齐与适配叠加；P0.1/P0.2/P2.3 后）

- 段级均值 **≥0.93**；记录级作辅助参考（E22 同协议浅层 0.9375–1.000，n=28，不作门槛）；
- 分别报告 AdaBN、对齐模式、自训练的**各自配对增量**，避免收益归因混淆；
- 0.93–0.95：进入 P3.3 并把差距量化写清。

### M3（目标 95%）

- 主格段级 **≥0.95**（开发 4 seed 均值），且**在确认 seeds 100–103 上重跑冻结流水线后仍 ≥0.95**
  （8 seed 联合报告逐 seed、均值±SD、seed 级 t 的 95% CI）；
- 同时报告 macro-F1 与四类召回；若 accuracy 达标而 Ball/IR 明显恶化，必须说明达标来自类别不均衡
  （OR 占评估段约 50%）；
- M2 达成但 0.93–0.95：按"未达 95%"如实报告，给出与浅层天花板（E22）的差距分解，
  **不得挑单 seed 最高值或改口径**。

**降级分支（预登记）**：若 P2.1 全网格后主格段级 <0.90，表征路线不成立，论文改按
"域适应机制研究"表述，以 P0.2/P2.2/P1 结果为主要贡献；不得回头修改协议或评估集。

---

## 9. 论文表述边界（必须写进正文）

- **95% 口径**：段级 accuracy，paired、固定标签池、per-seed 评估集；记录级为部署语义指标；
  达标声明只能来自正式 paired 重跑，不能来自探针。
- **浅层谱方法是正式基线**（表征、标签、划分与网络一致），必须与网络并列报告；
  E22 的 0.90–0.93 是浅层参考，不是网络预期。
- 段级与记录级并列报告，记录级必须报 4–8 seed 均值±SD；硬/软投票口径分别标注。
- DANN/CDAN/对抗式方法未测，不排除；若主张域适应贡献，至少补一个 GRL/DANN 或 Deep-CORAL/自训练强基线。
- 不要为达标切回段级随机划分（CWRU 文献 95–99% 多为段级随机 train/test，存在相邻段跨集泄漏）；
  比较时必须声明口径差异。
- transductive 下 AdaBN、自训练、对齐使用评估记录的 FE 无标签输入，inductive 分支不可引用。
- E21 的"谱表征下源域分支记录级占优"只适用于**同文件 DE→FE**设定，不得外推为跨文件/跨轴承泛化。

---

## 10. 风险登记与对策

| 风险 | 触发信号 | 对策 |
|---|---|---|
| 网络赢不过浅层探针（管线问题） | M1 管线健康检查不通过 | 中间臂归因；先冻结前端配方、再调 lr；必要时先用探针级方法打底报告 |
| 95% 不可达（线性/浅层边界） | M2 卡在 0.93–0.95 | 自训练（类别均衡）+ 多分辨率/集成；仍不达按降级分支如实报告 |
| 选择-评估污染 | 网格与确认共用 seed | 预留 seeds 100–103；披露选择路径（§7） |
| 表征/缓存漂移 | E22 对拍 >1pp | 修共享提取函数；repr_config 全量入键 |
| 4GB 显存 AdaBN 整批前向 OOM | 显存峰值 | 分块累计精确统计（§5 P0.1） |
| cepstrum 序列轴不连续被误读 | 可视化/正文 | 明确低/高倒频拼接语义 |

---

## 11. 证据索引

| 编号 | 内容 | 位置 |
|---|---|---|
| E12–E20 | 旧库历史证据（数据/物理/探针/网络），§1.1 已复核 | `../rbop_diagnosis/docs/evidence/` |
| E21 | paired 协议匹配探针（三表征）；keys 覆盖 12 划分 | 同上 |
| **E22（v4 新增）** | E21 的 v4 复现 + 全网格划分键（pool0.3/0.5、lr 0–0.99、seeds 0–3） | `docs/evidence/E22_protocol_probe_v4.*` |
| **E23（v4 新增）** | 迁移审计：旧主格网络 run 的 v4 重跑锚定对拍 | `docs/evidence/E23_migration_anchor.txt` |

---

## 12. 复现命令骨架（目标形态；`--reprs/--alignment_modes/--keys_only/--arms/--pseudo_*`
为 v4 新增 CLI，落地前不可直接照搬旧库参数）

```powershell
$PY = "E:/tool/anaconda/envs/fault-detect/python.exe"
$env:PYTHONUTF8 = "1"
Set-Location E:/文献/近两三年的故障诊断文献/rbop_diagnosis_v4

# P0.0 数据迁移（已确认：整体复制 + 清单校验）
robocopy ..\rbop_diagnosis\Data Data /E /NFL /NDL /NJH /NJS
& $PY -W ignore -m unittest tests.test_dataset -v

# P0.0 迁移审计 + E22 复现（CPU）
& $PY -m unittest discover -s tests -v
& $PY -W ignore scripts/diag_protocol_probe.py --split_mode paired --label_pool_ratio 0.5 `
  --label_ratios 0.3 0.6 0.99 --seeds 0 1 2 3 --reprs logspec cepstrum vmdstats `
  --keys_out docs/evidence/E22_protocol_probe_keys.json
& $PY -W ignore scripts/diag_protocol_probe.py --split_mode paired --label_pool_ratio 0.3 `
  --label_ratios 0.3 0.6 0.99 --seeds 0 1 2 3 --reprs logspec cepstrum --keys_only

# P0.0 v4 VMD 主格基线（M1/M2/M3 的配对参照；seed0 兼作迁移锚定）
& $PY -W ignore scripts/run_sweep.py --split_mode paired --label_pool_ratio 0.5 `
  --seeds 0 1 2 3 --protocols transductive --label_ratios 0.99 `
  --controls semi_supervised --lams 0.3 --reprs vmd `
  --tag baseline --source_epochs 30 --transfer_epochs 30

# P2.1 pilot（新表征 + 新前端；主格）
& $PY -W ignore scripts/run_sweep.py --split_mode paired --label_pool_ratio 0.5 `
  --seeds 0 1 2 3 --protocols transductive --label_ratios 0.99 `
  --controls semi_supervised --lams 0.3 --reprs logspec cepstrum `
  --tag pilot --source_epochs 30 --transfer_epochs 30

# target-only（lam 必须为 0：该对照不产生 MMD，写 lam=0.3 会污染结果字段）
& $PY -W ignore scripts/run_sweep.py --split_mode paired --label_pool_ratio 0.5 `
  --seeds 0 1 2 3 --protocols transductive --label_ratios 0.99 `
  --controls target_only --lams 0 --reprs logspec `
  --tag pilot_to --source_epochs 30 --transfer_epochs 30

# P0.1 AdaBN（对 pilot 的 source/transfer ckpt；分块精确统计）
& $PY -W ignore scripts/adabn_eval.py --run runs/paired/<run_name> --ckpt source.pt `
  --scope all --stats_source adapt --passes 1 3 10

# P0.2 对齐（新表征主格）
& $PY -W ignore scripts/run_sweep.py --split_mode paired --label_pool_ratio 0.5 `
  --seeds 0 1 2 3 --protocols transductive --label_ratios 0.99 `
  --controls semi_supervised --lams 0.3 --alignment_modes global conditional `
  --reprs logspec --tag am --source_epochs 30 --transfer_epochs 30

# P2.2 五臂（A–E；--arm 由 trainer 支持）
& $PY -W ignore scripts/run_sweep.py --split_mode paired --label_pool_ratio 0.5 `
  --seeds 0 1 2 3 --protocols transductive --label_ratios 0.99 `
  --controls semi_supervised --lams 0.3 --reprs logspec --arms A B C D E --tag arms

# P2.3 自训练（预登记组合）
& $PY -W ignore scripts/run_sweep.py --split_mode paired --label_pool_ratio 0.5 `
  --seeds 0 1 2 3 --protocols transductive --label_ratios 0.99 --controls semi_supervised `
  --lams 0.3 --reprs logspec --pseudo_tau 0.9 0.95 --pseudo_w 0.3 1.0 --tag pl

# 汇总（配对差值、Holm、记录级、cluster bootstrap）
& $PY -W ignore scripts/summarize.py --split_mode paired --families p02 p22 p23 --holm

# M3 确认（冻结流水线；确认 seeds 100–103，开发阶段禁止使用）
& $PY -W ignore scripts/run_sweep.py --split_mode paired --label_pool_ratio 0.5 `
  --seeds 100 101 102 103 --protocols transductive --label_ratios 0.99 `
  --controls semi_supervised --lams 0.3 --reprs logspec `
  --tag confirm --source_epochs 30 --transfer_epochs 30
```
