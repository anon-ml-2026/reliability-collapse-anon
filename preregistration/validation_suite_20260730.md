# P1 补实验预注册计划（2026-07-30）

> **发布说明**：本文件为预注册原文，内容未改动。文中输出路径为内部布局；发布仓库中的对应物为
> `data/scores/r4_judge_qwen.json`、`results/test_retest_qwen_and_r4_cross.json`、
> `data/scores/r3_ablation_unreg_anchored_{deepseek,qwen}.json`、`results/ablation_unreg_anchor.json`、
> `data/scores/r4_judge_kimi_run2.json`。

约束：Claude（Anthropic API）不再产生任何新数据；所有新评分仅使用 Qwen3.6-Plus（DashScope）、DeepSeek-V4-Pro、（待 key）Kimi-K3。
全部新实验遵循原管线：同一 SCORING_SYSTEM_PROMPT（S4 为预声明的变体）、T=0、max_tokens=1500、同一 parse_scores / compute_bvi、断点续跑、_meta.json 记录。

## S1 Claude-subject 0.5-band 桥接分析（本地，无新评分）
- 目的：§7 "test-retest 未覆盖两个 Claude 模型" 缺口的解析替代。
- 数据：R3 既有三裁判（DeepSeek、Claude-Haiku、Qwen）评分；Claude-subject trials n=1,080。
- 指标：DeepSeek 0.5-labels 在 Claude-subject vs 其他 6 模型 trials 上被 Qwen / Claude 判回 0.5 的保持率；bootstrap（10,000 次）差值 CI。
- 预声明解释：若 Claude-subject 保持率 ≤ 其他组 + 10pp，则"0.5 band 不稳定"推广至 Claude 模型成立；§7 该条降级为"覆盖方式不同"。

## S2 P1-1 配套分析（本地）
- DS×QW 原 rubric 分域基线（已算：binary Po REG-S 96.7 / REG-H 84.4 / UNREG 85.0）。
- 新对（Qwen×Claude）prevalence-matched 梯度复算（Appendix K 协议，500 次抽取，seed 20260728 沿用）。

## S3 Qwen 评 R4 全量（新评分，3,239 条）
- 目的①：第二个 T=0 test-retest judge（§7 实验#3）。指标：Qwen 版 R3–R4 分域 Po/κ/PABAK、PSI=0.5 条件稳定性。
  - H1：方向复现 REG-S > REG-H,UNREG；0.5 稳定性 < 50%（参照 DeepSeek 版 93.7/80.4/81.6 与 23.2%）。
- 目的②：R4 上 Qwen×Kimi（首个无 DeepSeek 的 R4 对）与 Qwen×DeepSeek 交叉表。
  - H2：两对的 UNREG 均为最不可靠域（排序层面）。
- 输出：R4_Round2/scoring_qwen/；analysis/_r4_qwen_test_retest_output.json。

## S4 UNREG 锚定 rubric ablation（新评分，2,760 条）
- 目的：§5.2 instrument-fit 竞争解释的决定性检验。
- 变体：原 rubric 追加 3 条作者构造的 UNREG 锚定示例（PSI=0 / 0.5 / 1 各一，含 D1/D2/D3 标定与一句理由），其余逐字节不变；rubric_version="v2_d1d2d3_unreganchored"。
- 设计：DeepSeek×Qwen 双 judge（与既有 R3 原 rubric 评分构成同 judge、同 trial 的配对比较）；UNREG 全量 1,080 + REG-S 安慰剂 300（seed 20260730，两 judge 同一子集）。
- 预注册判定规则（二值 Po，McNemar 配对检验，α=.05）：
  - 基线：UNREG 85.0%、REG-S 96.7%、gap=11.7pp。
  - 锚定后 UNREG Po ≥ 90.9%（收复 ≥50% gap）→ 支持 instrument-fit（梯度部分源于 rubric 工程）；
  - < 87.9%（收复 <25%）→ 支持 codification 读物（分歧为建构内生）；
  - 介于之间 → 不结论，如实报告。
  - 安慰剂约束：REG-S 移动 |Δ| ≥ 2pp 时，在论文中声明锚定存在跨域外溢并相应弱化结论。
- 输出：R3_Round1/scoring_ablation_{deepseek,qwen}/；analysis/_ablation_unreg_anchor_output.json。

## S5 Kimi T=1 双跑（待 Moonshot key）
- 目的：§7 实验#4，分离 T=1 采样噪声与真裁判间分歧。
- 设计：原设置（T=1，max_tokens 3000）复评 R4 全量；Kimi-run1×run2 分域一致性 = 噪声上限；与 DeepSeek×Kimi 差距对比。
- 输出：R4_Round2/scoring_kimi_run2/。

## ARR 合规要点
- 预声明假设与判定规则（本文档，先于运行）；确定性种子；T=0（Kimi 除外并披露）；全部脚本与 _meta.json 入库；
- 所有新数字进 Evidence Map（Appendix B）时标注 "Verified" 与本计划条目号；
- Claude 相关缺口在 §7 保留弱化版诚实声明（S1 为桥接，非字面 test-retest）。
