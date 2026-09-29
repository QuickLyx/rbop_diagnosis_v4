# T1 可依次执行的实施清单（Execution Plan，目标库 rbop_diagnosis_v4）

> 本清单是 `PLAN_T1_表征对齐与读出改造_版本3.md`（下称"版本3"）的**执行分解**。
> 所有设计依据、口径、证据、判据均以版本3为准，本文只负责把版本3拆成
> **可依次执行、可勾选、可验收**的步骤。凡涉及数字/口径/判据处，正文以
> `见版本3 §X` 方式回指，不重复论证。
> 证据来源：旧库 `../rbop_diagnosis/docs/evidence/E12–E21`（版本3 §11）。

---

## A. 对版本3的方案审查结论（是否还有问题）

审查方法：逐条核对版本3自称"已复核修正"的断言与旧库代码/证据实际是否一致。

### A.1 已逐条核实为**正确**的关键断言

| 版本3断言 | 核实结果 | 依据 |
|---|---|---|
| `conditional` 用 `probs[:,c]>=0.5` 硬掩膜、`confidence` 用 softmax 软加权（§2 L3） | 属实 | `../rbop_diagnosis/model.py:301-319` `conditional_mmd_loss` |
| 旧前端为 3 级 `MaxPool1d(4)`（512→128→32→8），需改谱前端 v1、VMD 保留 (4,4,4)（§0.2/§4.4） | 属实 | `../rbop_diagnosis/model.py:80-86` `ConvFrontEnd` |
| SMCBlock 通道 (32,64,128)（§0.2/§4.4） | 属实 | `../rbop_diagnosis/model.py:80` |
| 全网络 **7 个 BatchNorm1d**（前端 3 + RSU×2×2=4），AdaBN"全部7/仅前端3"（§5 P0.1） | 属实 | `model.py:65`（SMC 各 1）、`model.py:124,126`（RSU 各 2） |
| 划分种子 `seed+100003`，确认 seed 100–103 → 划分种子 100103–100106（§0.2/§7） | 属实 | `../rbop_diagnosis/experiment.py:144`、`dataset.py:401` 内部再 +1 |
| E21 keys JSON 只覆盖 12 划分（§0.1#1） | 属实 | `../rbop_diagnosis/docs/evidence/E21_protocol_probe_keys.json` 存在且口径如述 |
| 旧库 `diag_protocol_probe.py` 支持 `--reprs/--keys_out`，但无 `--keys_only/--arms/--pseudo_*` | 属实（后者为 v4 新增，版本3 §12 已注明） | `../rbop_diagnosis/scripts/diag_protocol_probe.py:177-190` |
| 旧库无 AdaBN/自训练/多 frontend 实现（均需 v4 新建） | 属实 | 旧库 `model.py`/`trainer.py` 无相关符号 |
| E12–E21 证据文件齐全 | 属实 | `../rbop_diagnosis/docs/evidence/` 全部存在 |

### A.2 审查结论

- **技术层面**：未发现需要修正的事实性/口径性错误。版本3 已把 v2 的
  归因错位（P2.2 的 C 臂是唯一干净检验）、硬/软投票不等价、AdaBN momentum、
  E21 keys 覆盖不全、表征参数未冻结等问题全部改正，且与旧库代码一致。
- **可执行性层面**：版本3 是**设计说明书**，粒度停在里程碑（W0–W4）与阶段
  （P0–P3），缺少一份**可依次勾选的操作级 checklist**。本文补齐这一层，
  不改变任何设计决策。
- **唯一需要执行者注意的两处执行风险（非方案错误，属落地时的一致性要求）**：
  1. **CLI 契约先行**：版本3 §12 的 `--keys_only/--arms/--pseudo_tau/--pseudo_w/
     --alignment_modes/--reprs/--tag/--source_epochs/--transfer_epochs` 中，多数为
     v4 新增。必须在 W1/W2 实现脚本时**先冻结 CLI 契约**（写进 `README.md`），
     否则 §12 的复现命令骨架无法直接跑。已在 S 步骤中显式列为交付物。
  2. **共享提取函数唯一化**：版本3 §4.3 要求探针与网络共用一份 `preprocessing.py`
     提取实现。落地时必须让 `diag_protocol_probe.py` 直接 import `preprocessing`，
     禁止复制第二份。已在 S3/S4 验收门槛中固化。

> 结论：**版本3 无需修改即可作为设计依据**；本文据其生成可依次执行的 plan。

---

## B. 执行总原则（继承版本3，执行时不再更改）

- 严格串行 W0 → W4，里程碑不通过不进入下一步（版本3 §4.8）。
- 开发只用 seeds 0–3；确认 seeds 100–103 在 M3 前**任何指标都不看**（版本3 §7）。
- `runs/natural/` 与 `runs/paired/` 严格分写，不得混表，不得删 run 重跑（版本3 §4.6）。
- 段级 accuracy 为唯一达标口径；记录级为辅助（版本3 §0 前提、§8）。
- 每个里程碑同步更新 `README.md`/`AGENTS.md`/`tests` 并把验收结果落 `docs/evidence/`
  （版本3 §4.8、§4.9）；文档滞后视为未完成。
- 环境：`E:/tool/anaconda/envs/fault-detect/python.exe`，`PYTHONUTF8=1`，
  工作目录 `rbop_diagnosis_v4/`（版本3 §12）。

---

## C. 可依次执行的步骤（每步含：产出 / 依据 / 验收门槛）

> 勾选规则：一步的"验收门槛"全部满足才可勾选并进入下一步。

### 里程碑 W0 —— 数据与划分（依据版本3 §4.1/§4.2/§4.8/§4.9）

- [x] **S0. 建立仓库骨架与文档初版**
  - 产出：按版本3 §4.1 建 `dataset.py/preprocessing.py/vmd.py/model.py/trainer.py/
    experiment.py/train.py/visualize_attention.py/scripts/*/tests/*/docs/*/runs/*` 空骨架；
    按版本3 §4.9 写 `README.md`（任务/口径/协议/目录/命名/schema 摘要/seed 规则/证据索引/"不做的事"）
    与 `AGENTS.md`（语言规则/范围/不变量全集/审计门槛/注释规范/文档同步）。
  - 验收：`README.md`/`AGENTS.md` 覆盖 §4.9 全部必备条目；目录结构与 §4.1 一致。

- [x] **S1. 数据迁移 + 清单校验**（版本3 §4.2 数据迁移 1–3）
  - 产出：`robocopy ..\rbop_diagnosis\Data Data /E` 复制 657 MB；`docs/data_manifest.json`
    （文件键+字节数），与旧库清单比对。
  - 验收：`enumerate_t1_files` 得 **56 记录**（12k_DE 排除全部 028、Normal 全保留），
    每记录同时含 `DE_time`/`FE_time`；四类 4/12/12/28；`docs/data_manifest.json` 生成。

- [x] **S2. `dataset.py` + 划分等价 + 段统计**（版本3 §4.2 不变量 1–8）
  - 产出：`enumerate_t1_files`、`split_target_files`（含 paired 语义、`seed+100003`）、
    `SegmentSet`；`tests/test_dataset.py`（划分等价、段统计、缓存键隔离、paired 语义）。
  - 验收（迁移审计 §4.2 表前两行）：
    - 划分等价：v4 生成 seeds 0–3 × pool{0.3,0.5} × lr{0,0.3,0.6,0.99} 的 labeled/eval 键，
      与旧库 `../rbop_diagnosis/runs/paired/results.json` **全 128 run 比对 0 不一致**；
    - 段统计：各文件段数、总段数 **13191** 完全一致；`seg_len=512` 不重叠；
      Normal 48k→12k 降采样后切段；
    - `tests/test_dataset.py` 通过。

**W0 门槛（版本3 §4.8 行 W0）**：56 记录且 DE/FE 成对；128 run 划分键 0 不一致；DE 共 13191 段。

---

### 里程碑 W1 —— 表征与探针（依据版本3 §4.3/§4.2 审计/§5 P0.0）

- [x] **S3. `preprocessing.py`（唯一提取实现）+ `repr_config` 冻结 + 缓存**（版本3 §4.3）
  - 产出：`vmd/logspec/cepstrum/vmdstats` 四表征提取（参数严格按 §4.3：logspec Hanning+rfft512+
    256 采样+log+段内标准化；cepstrum 低倒频 2..119 + 高倒频 394..511 共 236 维；vmdstats 32 维）；
    缓存键 `cache/<dataset>/<channel>/<stem>_L512_<repr_key>.npz` 且 npz 内存 `repr_config`
    并逐字段校验；`regen_cache.py --reprs`；`tests/test_preprocessing.py`（形状/键/缓存校验/fixture 数值）。
  - 验收：四表征形状 (1,256)/(1,236)/(1,32)/(8,512) 正确；`repr_config` 全量入键与 npz；测试通过。

- [x] **S4. `scripts/diag_protocol_probe.py` → E22 复现 + 全网格划分键**（版本3 §1.2、§5 P0.0）
  - 产出：探针脚本**直接 import `preprocessing`**（禁止第二份实现）；新增 `--keys_only`；
    生成 `docs/evidence/E22_protocol_probe_v4.txt` 与全网格 `E22_protocol_probe_keys.json`
    （pool{0.3,0.5} × lr{0,0.3,0.6,0.99} × seeds 0–3）。
  - 验收（版本3 §4.2 审计 logspec/cepstrum 行 + §5 P0.0 判据）：
    E22 与 E21 同 seed/预算对拍 **段级差 ≤1pp、逐 seed 同号**；否则先修实现再继续。

- [x] **S5. VMD 缓存等价性抽样对拍**（版本3 §4.2 审计 VMD 行）
  - 产出：抽样 3–5 文件，v4 `cache/` 与旧库 `../rbop_diagnosis/cache/` 对应 npz 比对脚本/记录。
  - 验收：形状/标签/段号一致、数值 `allclose(rtol=1e-5)`。

**W1 门槛（版本3 §4.8 行 W1）**：E22 对拍 E21 ≤1pp 且方向一致；VMD 缓存抽样与旧库 allclose。

---

### 里程碑 W2 —— 训练管线（依据版本3 §4.4/§4.5/§4.6/§4.7、§5 P0.0）

- [x] **S6. `model.py`（双 frontend）**（版本3 §4.4/§4.5）
  - 产出：`ConvFrontEnd` 支持 `spectral_v1`（MaxPool(2)→MaxPool(4)→AdaptiveMaxPool1d(8)，固定 8 步）
    与 `legacy_vmd`（(4,4,4)）；`c_in=1`（可选 2）/`c_in=8`；frontend 类型入 checkpoint；
    `conditional_mmd_loss` 三模式沿用；`tests/test_model.py`（(1,256)/(1,236)/(8,512) 前向、
    注意力归一、MMD 三模式与估计器有限值、小 batch 保护）。
  - 验收：logspec256 下 spectral_v1 与 (2,4,4) 逐点等价（§4.4）；cepstrum236 走自适应到 8 步不补零；
    测试通过。

- [x] **S7. `trainer.py` + `experiment.py` + `train.py` + schema**（版本3 §4.5/§4.6/§4.7）
  - 产出：`train_source/train_transfer/evaluate_details/checkpoint`；checkpoint `schema_version:1`
    含 `repr_config/frontend/...`（§4.5）；results.json 全字段（§4.5 三组字段 + 记录级即时计算）；
    AdaBN 分块精确统计 hook（§5 P0.1 规程，供 W3/P0.1 调用）；CLI 契约冻结并写入 `README.md`
    （`--repr/--K/--sort_vmd/--alignment_modes/--reprs/--tag/--source_epochs/--transfer_epochs`
    及 run 命名规则 §4.6）；`tests/test_trainer.py`/`tests/test_experiment.py`。
  - 验收：checkpoint round-trip + schema 拒绝；零标签零 MMD 路径；run 名唯一、字段完整、目录不覆盖、
    natural/paired 分模式隔离；测试通过。

- [x] **S8. v4 自跑 VMD 主格基线（4 run，seed0 兼作迁移锚定）**（版本3 §5 P0.0、§4.2 审计末行）
  - 产出：`run_sweep.py` 跑 paired/pool0.5/lr0.99/λ=0.3/semi、reprs=vmd、seeds 0–3、tag=baseline；
    `docs/evidence/E23_migration_anchor.txt`。
  - 验收：seed0 段级/记录级与旧主格 run **差 ≤2pp**（差异需书面解释）；结果 schema 字段齐备。
    该 4 run 为 M1/M2/M3 一切 paired 配对差值的参照基线（历史 0.8147 仅作锚定核对）。

**W2 门槛（版本3 §4.8 行 W2）**：seed0 锚定与旧 run ≤2pp；结果 schema 字段齐备。

---

### 里程碑 W3 —— 读出与汇总（依据版本3 §4.5/§4.7/§P1）

- [x] **S9. 记录级读出（随 run 即时产出）**（版本3 §P1.1/P1.2）
  - 产出：`record_accuracy_hard/soft`（等权、硬软分别）、误差在记录上的聚集统计（<0.5 记录数/分位数）；
    `evaluate_details` 返回逐段概率 + 记录索引。
  - 验收：硬/软口径分别落字段（§P1.1 已证倒谱/VMD 硬软不同，不得混淆）。

- [x] **S10. `summarize.py` + `visualize_attention.py`**（版本3 §4.7/§P1.3/§7）
  - 产出：配对差值、Holm step-down、按类分层 cluster bootstrap、`--self-check`；
    注意力→频带重要性可视化（cepstrum 需标注低/高倒频拼接、序列轴中点不连续，§P3.1）。
  - 验收：全套 `tests/` + 各模块 `_self_check()` 与 `summarize.py --self-check` 通过（版本3 §4.7）。

**W3 门槛（版本3 §4.8 行 W3）**：全套 tests + 自检通过。

---

### 里程碑 W4 —— 进入实验（依据版本3 §5/§6/§8，严格按执行顺序）

> 执行顺序（版本3 §6 末）：P2.1 pilot → 判 M1 → P0.1/P0.2 → P2.1 网格 → P2.2 → P2.3（按需）→ M3 → P3.3（可选）。

- [x] **S11. P2.1 pilot（20 run）+ 浅层基线**（版本3 §5 P2.1）
  - 产出：logspec/cepstrum + 谱前端 v1（各 4）、中间臂 logspec+旧前端(4,4,4)（4）、
    target-only λ=0（4）、semi λ=0（4）；no-transfer+AdaBN 复用 pilot `source.pt`；
    浅层 logspec/cepstrum+LR 正式基线（同格同 seed，CPU）。
  - **判 M1（版本3 §8 M1）**：主格段级均值 ≥0.90 且相对 v4 VMD 基线配对差值均值 ≥+0.05 且 4/4 seed 为正；
    macro-F1 ≥0.88；Ball/IR 召回不低于 v4 VMD 基线同格 4 seed 均值；网络段级 ≥ E22 同表征 FE-only 探针 −3pp。
    - 0.85–0.90：先看中间臂归因再决定；<0.85：表征路线降级，回 P0/P1 并记录失败证据（§8 降级分支）。

- [x] **S12. P0.1 AdaBN（0 新训练）**（版本3 §5 P0.1）
  - 产出：分块精确统计；因子 = {source/transfer ckpt} × {全部7 BN / 仅前端3} × {仅 adapt / 全 56 FE 输入} × {1/3/10 遍}；
    `bn_adapt_on` 披露。
  - 验收：与"重估前"成对报告、4 seed 全列方向；不得把 E17 的 0.786 当预期值。

- [x] **S13. P0.2 对齐（新表征主格，8 run + 可选 8）**（版本3 §5 P0.2）
  - 产出：`mode ∈ {global, conditional}` × 主格 × seeds 0–3；`confidence` 复用 pilot 同配置 semi run；
    λ=0 不重跑。可选 VMD `conditional` 主格 4 run（历史机制对照，不铺 48 run）。
  - 验收：同 seed 配对差值；家族预注册 + Holm；L3 最终结论以本网格为准。

- [x] **S14. P2.1 网格扩展（12–24 run）**（版本3 §5 P2.1）
  - 产出：过 M1 后铺 lr∈{0.3,0.6,0.99} 与 pool0.3 必要格；`lr_transfer` 重扫作为独立预登记消融臂（不入主格）。
  - 验收：结果按格落盘、不混主格。

- [x] **S15. P2.2 五臂 A–E（20 run，VMD 补 A/B/C 可选 12）**（版本3 §5 P2.2）
  - 产出：A 基线 / B 冻结 backbone / C 去源 CE 保 MMD / D 源 CE 系数 0.1,0.3 / E lr_transfer 探索；
    `--arms` 由 trainer 支持。
  - 验收（版本3 §5 P2.2 判据）：C>A ⇒ 源 CE 有害（论文结论唯一来源）；B≈A / B 最差 各自解释；
    家族 `{B,C,D0.1,D0.3,E} vs A` Holm 校正；三种结果都写论文。

- [ ] **S16. P2.3 自训练（按需，8–12 run；M2 未达 95% 才启动）**（版本3 §5 P2.3）
  - 产出：τ∈{0.9,0.95} × w_pl∈{0.3,1.0}；类别均衡配额（每类封顶 min(阈值以上段数,1000)）；1–2 轮固定步数；
    每轮记录伪标签类别分布与记录级一致性。
  - 验收：同 seed 配对差值 + macro-F1/Ball 召回（防只涨 accuracy）。
  - **判 M2（版本3 §8 M2）**：段级均值 ≥0.93；分别报告 AdaBN/对齐/自训练各自配对增量。
    0.93–0.95 → 进 S17（P3.3）并量化差距。

- [ ] **S17. P3.3 多分辨率/集成（可选，仅冲 95%）**（版本3 §5 P3.3）
  - 产出：256/512 双通道 log 谱；多种子/多窗 transductive 推理集成（独立消融并披露）。
  - 验收：作为独立消融报告，不改口径。

- [x] **S18. M3 确认（seeds 100–103，冻结流水线）**（版本3 §7/§8 M3）
  - 产出：在**冻结后**流水线上跑 100–103，与开发 4 seed 合并 8 seed 报告；披露配置选择路径、
    声明确认 seed 未参与选择。
  - **判 M3（版本3 §8 M3）**：主格段级 ≥0.95（开发 4 seed 均值）**且**确认 seeds 100–103 重跑仍 ≥0.95；
    8 seed 报逐 seed、均值±SD、seed 级 t 的 95% CI；同报 macro-F1 与四类召回。
    - 0.93–0.95：按"未达 95%"如实报告，给出与浅层天花板（E22）的差距分解，不得挑单 seed/改口径（§8）。

---

## D. 全局验收与降级分支（依据版本3 §8/§10）

- **降级分支（预登记，§8）**：P2.1 全网格后主格段级 <0.90 ⇒ 表征路线不成立，论文改按
  "域适应机制研究"表述，以 P0.2/P2.2/P1 为主贡献；不得回改协议或评估集。
- **风险对策**：见版本3 §10（管线赢不过探针→中间臂归因；95% 不可达→自训练/集成/降级；
  选择-评估污染→预留 100–103；表征漂移→repr_config 全量入键；OOM→分块精确统计；
  cepstrum 轴不连续→明确拼接语义）。

## E. 复现命令

严格照版本3 §12 的 PowerShell 骨架逐段执行（`--keys_only/--arms/--pseudo_*/--alignment_modes`
等为 v4 新增 CLI，须先在 S7 冻结契约后使用）。本文不重复命令，避免与 §12 漂移。

## F. 证据与文档索引

- 历史证据 E12–E21：`../rbop_diagnosis/docs/evidence/`（版本3 §11、§1）。
- v4 新增 E22（探针复现+全网格键）、E23（迁移锚定）：`docs/evidence/`（版本3 §11）。
- 设计依据全文：`docs/PLAN_T1_表征对齐与读出改造_版本3.md`。


---

## D. 执行状态（2026-09-29 完成）

- **S0–S15、S18 全部完成**；S16（P2.3 自训练）、S17（P3.3 多分辨率）为按需/可选，
  在 M2/M3 已达 95% 的条件下**按版本3 §6 顺序跳过**（版本3 §5 P2.3/§5 P3.3 触发条件均未用上）。
- 关键验收（详见 `docs/evidence/`）：
  - W0：128 run 划分等价 0 不一致；DE 13191 段；56 文件字节对拍一致。
  - W1：E22 对拍 E21 段级差 0.000pp（10 个锚点全等）；VMD 缓存抽样 allclose maxdiff 0.0。
  - W2：v4 VMD 基线逐 seed 与旧库 0.00pp（锚定差 0.005pp）；schema_version=1。
  - M1：主候选 logspec+spectral_v1 段级 0.9872（≥0.90），相对 VMD 配对 +0.1725（4/4 正）。
  - P0.1：AdaBN 对 no-transfer +7.35pp（0.6238→0.6973）；对 transfer.pt 中性（≤0.2pp）。
  - P0.2：confidence 0.9872 ≈ conditional 0.9865 > global 0.9763（差异 <1.1pp）。
  - P2.1：lr0.3/0.6/0.99 = 0.8470/0.9530/0.9872；pool0.3 满预算 0.8960。
  - P2.2：B 0.8012（显著最差）、C 0.9730（显著差于 A，源 CE 在新表征下**有益**）、
    D0.1 0.9872、D0.3 0.9842、E 0.9791（后三与 A 不可区分）。
  - M3：开发 4 seed 0.9872、确认 100–103 0.9935、合并 8 seed **0.9904**（8/8 ≥0.95）。
- 落地注意事项（A.2 两条）落实情况：CLI 契约已写入 `README.md` §6/§7；
  探针与网络共用 `preprocessing.extract_repr`（无第二份实现，E22 对拍 0.000pp 佐证）。
