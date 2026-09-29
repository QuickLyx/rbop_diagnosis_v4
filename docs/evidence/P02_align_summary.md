P0.2 对齐模式网格结果（新表征主格，L3 结论）
================================================================
依据：版本3 §5 P0.2 / §2 L3；执行清单 S13。
配置：主候选 logspec + spectral_v1，paired/pool0.5/lr0.99/λ0.3/semi，seeds 0-3，30+30 epoch。
confidence 复用 pilot 同配置 run（λ=0.3）；global/conditional 为本次新跑 8 run
（conditional 因默认命名不含对齐模式，加 `_alignconditional` 后缀；结果按 results.json
的 alignment_mode 字段汇总，不依赖名字）。

[对齐模式 4 seed 结果]
mode         n  段级均值±SD      逐 seed
confidence   4  0.9872±0.0078   [0.9763, 0.9870, 0.9914, 0.9940]
conditional  4  0.9865±0.0077   [0.9783, 0.9819, 0.9913, 0.9946]
global       4  0.9763±0.0177   [0.9850, 0.9482, 0.9857, 0.9863]

[同 seed 配对差值]
conditional − confidence: [−0.0020, +0.0051, +0.0001, −0.0006] 均值 −0.0007
global − confidence:      [−0.0087, +0.0388, +0.0057, +0.0077] 均值 +0.0089（seed1 外均小）

[结论（L3）]
- 在本主格（高标签预算 lr0.99 + 强判别表征）下，三种对齐模式差异 <1pp：
  confidence ≈ conditional > global（均值差 0.07pp / 1.09pp），家族预注册 + Holm 校正后
  无显著差异（n=4 探索性；正式判据按 §7）。
- 与 VMD 表征历史结论一致方向（conditional 硬掩膜理论更精确），但高预算下伪标签质量高，
  模式选择敏感度低；L3 最终结论以本网格为准（不再引用旧库机制数字）。
- 失败/边界情况：conditional 的 probs>=0.5 硬掩膜在训练早期类样本不足时回退到 global
  MMD（model.conditional_mmd_loss 内保护），未出现 NaN。
================================================================
