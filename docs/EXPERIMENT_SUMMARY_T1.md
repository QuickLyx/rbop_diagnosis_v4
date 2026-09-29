# T1 实验总结报告 —— 同文件跨测点 DE→FE 故障诊断（rbop_diagnosis_v4）

> 生成日期：2026-09-29
> 依据：`docs/PLAN_T1_表征对齐与读出改造_版本3.md`（版本3）、`docs/EXEC_PLAN_T1_可执行执行清单.md`（执行清单）、
> `docs/evidence/` 全部验收证据、`runs/paired/results.json`（68 run）。
> 口径：一切数字以 v4 自跑证据为准；旧库 `../rbop_diagnosis` 仅作历史锚定，不作结果引用。

---

## 0. 一句话结论

在 CWRU `12k_DE/Normal` 数据集上，对**同一批文件从驱动端（DE）迁移到风扇端（FE）**的四分类任务，
采用 **logspec 谱表征 + 谱前端 v1 + 源域 CE / MMD 对齐 / 半监督标签适配** 的主格流水线，
在达标口径（paired / pool0.5 / lr0.99 / λ=0.3 / semi 的**段级 accuracy**）上取得
**合并 8 seed 均值 0.9904 ± 0.0070（95% CI [0.9845, 0.9962]，8/8 seed ≥ 0.95）**，
**达标 95% 目标**，相对 v4 VMD 基线（0.8147）配对提升 **+17.25pp**。

---

## 1. 任务与达标口径

- **任务（T1）**：CWRU `12k_DE/Normal`，**同一批文件**从 DE 迁移到 FE，四分类
  （Normal / Ball / InnerRace / OuterRace）。即"同文件跨测点 DE→FE"。
- **数据**：56 条记录（`12k_DE` 排除全部 028、`Normal` 全保留；四类 4/12/12/28），
  每条同时含 DE/FE；`seg_len=512` 不重叠；Normal 由 48 kHz 降采样到 12 kHz 后切段；DE 共 **13191 段**。
- **达标口径（唯一）**：**段级 accuracy ≥ 0.95**，在 paired 协议 / pool0.5 / 固定标签池 /
  per-seed 评估集（池的补集）上。记录级为辅助/部署语义指标，**不作为达标依据**，
  且**禁止在段级/记录级之间切换口径择优**。
- **协议边界**：transductive 允许使用评估文件的 FE **无标签输入**做适配（AdaBN / 对齐 / 自训练）；
  评估文件的 FE **标签**从不进入训练。

---

## 2. 技术路线

**谱表征（logspec/cepstrum）+ 谱前端 v1（保频结构池化）+ 域对齐（AdaBN / MMD / 自训练）+ 记录级读出。**

主格最终配置：

| 组件 | 取值 |
|---|---|
| 表征 input_repr | `logspec`（nbin=256，hanning 窗，段内标准化 seg_std=True，input_dim=256） |
| 前端 frontend | `spectral_v1`（MaxPool(2)→MaxPool(4)→AdaptiveMaxPool1d(8)，保频结构池化） |
| 骨干 | RSU/SE + BiLSTM + 注意力读出 |
| 对齐 alignment_mode | `confidence`（置信度加权 MMD） |
| MMD | biased 估计，权重 λ=0.3 |
| 适配 control | `semi_supervised`（源 CE + MMD 对齐 + 目标有标签 CE） |
| 标签预算 | label_ratio=0.99（每类目标文件比例），label_pool_ratio=0.5 |
| 训练 | 30（源）+ 30（迁移）epoch，bs=64，lr_source=1e-3，lr_transfer=1e-4 |
| 读出 | 段级 + 记录级硬/软投票（等权） |

---

## 3. 里程碑推进（W0 → W4 严格串行）

| 里程碑 | 内容 | 关键验收数字 | 状态 |
|---|---|---|---|
| **W0** | 数据与划分 | 56 记录 DE/FE 成对；128 run 划分键与旧库 **0 不一致**；DE 共 **13191 段** | ✅ |
| **W1** | 表征与探针 | E22 对拍 E21 段级差 **0.000pp**（≤1pp 门槛）；VMD 缓存抽样 allclose maxdiff=0.0 | ✅ |
| **W2** | 训练管线 | seed0 主格锚定与旧 run 差 **0.00pp**（≤2pp 门槛）；results.json 69 字段齐备 | ✅ |
| **W3** | 读出与汇总 | 33 测试 + 6 模块自检全通过；记录级读出随 run 即时产出 | ✅ |
| **W4** | 进入实验 | M1→P0.1→P0.2→P2.1→P2.2→M3 全通过；M3 段级 8 seed 0.9904，8/8 ≥0.95 | ✅ |

### 迁移审计门槛（W0–W2 验收，全部通过）

| 项 | 通过标准 | 实测 |
|---|---|---|
| 划分等价 | 全 128 run 键 0 不一致 | 0 不一致 ✓ |
| 段统计等价 | 总段数 13191 | 13191 ✓ |
| logspec/cepstrum 等价 | E22 对拍 E21 ≤1pp 且逐 seed 同号 | 最大差 0.000pp ✓ |
| VMD 缓存等价 | 抽样与旧库 allclose(rtol=1e-5) | maxdiff=0.0 ✓ |
| 网络重跑锚定 | 主格 seed0 与旧 run ≤2pp | 0.00pp / 均值锚定差 0.005pp ✓ |

---

## 4. W4 实验全流程与结果

### 4.1 M1 —— 表征落地（P2.1 pilot，20 run，开发 seeds 0–3）

| 配置 | n | 段级均值±SD | macro-F1 |
|---|---|---|---|
| **logspec + spectral_v1（主候选）** | 4 | **0.9872 ± 0.0078** | 0.9894 |
| cepstrum + spectral_v1（并列候选） | 4 | 0.9445 ± 0.0621 | 0.9583 |
| 中间臂 logspec + legacy_vmd | 4 | 0.9471 ± 0.0623 | 0.9597 |
| target-only logspec λ=0 | 4 | 0.8868 ± 0.0795 | 0.9057 |
| semi λ=0 logspec | 4 | 0.9362 ± 0.0309 | 0.9455 |
| VMD 基线（锚定） | 4 | 0.8147 ± 0.0801 | 0.7812 |

**归因分解（核心科学结论）**：
- **表征收益（logspec − vmd，同前端）：+13.25pp** —— 主导贡献。
- 前端收益（spectral_v1 − legacy_vmd，同 logspec）：+4.01pp —— 正贡献但非主因（主要在 seed0 +12pp）。
- L2 复判（semi λ=0 − target-only，谱表征）：+4.95pp —— **源域分支在新表征下有益**，
  与 VMD 表征下相反，证实"源域分支作用表征特定"的预测。

M1 全部判据通过（段级≥0.90、配对差≥+0.05、4/4 正、macro-F1≥0.88、各类召回达标、网络显著高于浅层探针）。

### 4.2 P0.1 —— AdaBN 因子网格（0 新训练，复用 pilot source.pt）

- 对 **no-transfer** 模型：AdaBN 全程为正（4/4 seed），最佳 frontend/all_target/p10 = 0.6973（**+7.35pp**）；
  全量单遍即收敛（分块精确统计 = 单遍全量）。
- **AdaBN 单独上限 ~0.70**，远低于主候选 0.9872 —— BN 统计平移不足以解决 DE→FE 的类别结构差异，
  须结合标签/MMD 迁移。
- 对**已迁移** transfer.pt：全部 12 因子 Δ ∈ [−0.12pp, +0.20pp]，**中性**——主格流水线已通过
  CE+MMD+标签适配吸收 BN 统计差异。

### 4.3 P0.2 —— 对齐模式网格（L3 结论）

| mode | n | 段级均值±SD |
|---|---|---|
| confidence | 4 | 0.9872 ± 0.0078 |
| conditional | 4 | 0.9865 ± 0.0077 |
| global | 4 | 0.9763 ± 0.0177 |

高标签预算 + 强判别表征下，三模式差异 <1pp（confidence ≈ conditional > global），
Holm 校正后无显著差异。主格选 **confidence**（与 conditional 仅差 0.07pp）。

### 4.4 P2.1 —— 标签预算 × 标签池网格

| 配置 | n | 段级均值±SD |
|---|---|---|
| pool0.5 × lr0.3 | 4 | 0.8470 ± 0.0855 |
| pool0.5 × lr0.6 | 4 | 0.9530 ± 0.0659 |
| pool0.5 × lr0.99（主格） | 4 | 0.9872 ± 0.0078 |
| pool0.3 × lr0.99 | 4 | 0.8960 ± 0.0920 |

段级随标签预算单调上升（0.847→0.953→0.987），**lr0.6 已达 0.95**；低预算下 seed 间方差大
（有标签文件集合偶发偏难/偏易），与浅层探针趋势一致。

### 4.5 P2.2 —— 五臂消融（相对 A 基线配对差）

| 臂 | 定义 | n | 段级均值±SD | 相对 A Δ均值 | Holm 显著性 |
|---|---|---|---|---|---|
| A | 基线（主格） | 4 | 0.9872 ± 0.0078 | +0.0000 | — |
| B | 冻结 backbone 只训头 | 4 | 0.8012 ± 0.0234 | **−18.60pp** | 显著（最差） |
| C | 去源域 CE 保留 MMD | 4 | 0.9730 ± 0.0123 | **−1.42pp** | 显著（差于 A） |
| D0.1 | 源 CE 系数 0.1 | 4 | 0.9872 ± 0.0058 | +0.00pp | 不显著 |
| D0.3 | 源 CE 系数 0.3 | 4 | 0.9842 ± 0.0052 | −0.30pp | 不显著 |
| E | lr_transfer=1e-3（十倍） | 4 | 0.9791 ± 0.0157 | −0.80pp | 不显著 |

**结论**：
1. **B 显著最差（−18.6pp）**：目标域适配必须训练 backbone，仅平移 BN 或只训头无法解决跨测点差异。
2. **C 显著差于 A（−1.42pp）**：**新表征下源域 CE 有益**（与 VMD 表征历史结论相反）——"源域分支作用表征特定"的又一证据。
3. D 臂（源 CE 系数 0.1–1.0）稳健，无需精调。
4. E（lr_transfer 十倍）略差但不显著，lr_transfer=1e-4 合理。

### 4.6 M3 —— 确认 seeds 100–103（冻结流水线，最终达标）

冻结代码后在从未参与选型的确认 seed 上复核：

| seeds | 段级 accuracy | macro-F1 | 记录级硬投票 |
|---|---|---|---|
| 开发 0–3 | [0.9763, 0.9870, 0.9914, 0.9940]，均值 **0.9872 ± 0.0078** | 0.9894 | 0.9929 |
| 确认 100–103 | [0.9862, 0.9955, 0.9949, 0.9976]，均值 **0.9935 ± 0.0050** | 0.9920 | 1.0000 |
| **合并 8 seed** | **0.9904 ± 0.0070**，95% CI [0.9845, 0.9962]，min 0.9763，**8/8 ≥0.95** | 0.9920 | 0.9955 |

- 四类召回（8 seed）：Normal 1.0000 / Ball 0.9959 / IR 0.9973 / OR 0.9838。
- 达标校验：开发均值 0.9872 ≥ 0.95 ✓；确认均值 0.9935 ≥ 0.95 ✓；8 seed 95% CI 下界 0.9845 ≥ 0.95 ✓。

---

## 5. 核心科学结论

1. **表征是主导贡献（+13.25pp ≫ 前端 +4.01pp）**：从 VMD 换到 logspec 谱表征是达标的关键，
   谱前端 v1 的保频结构池化提供次级增益。
2. **源域分支作用是"表征特定"的**：在谱表征下源域 CE 有益（去掉掉 −1.42pp），
   与 VMD 表征下"源域分支有害"的历史结论相反——L2 假设得到 M1（+4.95pp）与 P2.2（C<A）双重确证。
3. **BN 统计平移不充分**：AdaBN 单独上限仅 ~0.70，远低于完整迁移的 0.99；
   在已迁移模型上 AdaBN 收益被主格流水线覆盖（中性）。须结合标签 + MMD 适配。
4. **目标域必须训练 backbone**：冻结 backbone 只训头崩塌 −18.6pp，跨测点差异是特征层级而非仅分类层。
5. **高标签预算下对齐模式选择不敏感**：confidence ≈ conditional > global，差异 <1pp（伪标签质量高）。
6. **标签预算单调有效**：lr0.6 即达 0.95，lr0.99 达 0.987。

---

## 6. 方法学与可信度保障

- **严格里程碑门槛**：W0–W2 五项迁移审计全部通过（划分键 0 不一致、段数 13191、
  E22 对拍 0.000pp、VMD 缓存 allclose、seed0 锚定 0.00pp），未过门槛不进入实验。
- **开发/确认 seed 隔离**：确认 seeds 100–103 在 M3 之前从未参与任何配置选择、调参或指标观察，
  避免选型泄漏。
- **口径不择优**：全程以段级 accuracy 为唯一达标依据，记录级仅作辅助报告，不在两者间切换择优。
- **统计规范**：配对差值 + Holm step-down + cluster bootstrap；n=4 的臂间比较标注为探索性。
- **可复现**：33 测试 + 6 模块自检全通过；results.json 69 字段完整；checkpoint schema_version=1
  含 repr_config/frontend；natural/paired 严格分写，禁止删 run 重跑。

---

## 7. 证据索引

| 结论 | 证据文件 |
|---|---|
| 最终达标（8 seed 0.9904） | `docs/evidence/M3_final_result.txt` |
| 表征落地 + 归因分解 | `docs/evidence/M1_pilot_result.txt` |
| AdaBN 网格 | `docs/evidence/P01_adabn_summary.md`、`P01_adabn_grid.txt`、`P01_adabn_transfer.txt` |
| 对齐模式网格 | `docs/evidence/P02_align_summary.md` |
| 标签预算网格 | `docs/evidence/P21_grid_result.txt` |
| 五臂消融 | `docs/evidence/P22_arms_result.txt` |
| 迁移锚定 | `docs/evidence/E23_migration_anchor.txt` |
| 协议探针复现 | `docs/evidence/E22_protocol_probe_v4.txt` |
| W0–W3 里程碑验收 | `docs/evidence/W0_data_split_audit.txt`、`W1_repr_probe_audit.txt`、`W2W3_pipeline_audit.txt` |
| W4 汇总表 | `docs/evidence/W4_summary_table.txt` |
| 全部 run 明细（68 run） | `runs/paired/results.json` |

---

## 8. 边界与不做的事

- 仅实现 **T1**（同文件跨测点 DE→FE 四分类），不实现 T2/T3。
- 不兼容旧库 checkpoint（`format_version=2`）；不引用旧网络数字作为 v4 结果。
- 不做段级随机划分（存在相邻段跨集泄漏）；不为达标切回段级随机 train/test。
- 记录级硬投票虽已达 0.9955（确认 seed 达 1.0000），但**不作为达标口径**，仅作部署语义参考。
