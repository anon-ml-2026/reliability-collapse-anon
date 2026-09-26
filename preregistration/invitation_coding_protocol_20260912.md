# 扩展分析 3：探针邀请强度协变量——预注册分析方案

> 本文件在编码开始前写定。分析计划、判据与两种结果对应的论文措辞**先于数据固定**；
> 编码完成后按本文件执行，不再回头调整门槛。日期：2026-09-12。

## 一、目的

回应 Opus 评审 P1-9：论文已承认 UNREG 探针"按构造邀请替代"（§4.1、§6.2），
但未提供可测协变量。本分析将邀请强度编码为探针级协变量，检验：
**候选标签下的 UNREG-highest margin 在多大程度上可由邀请强度解释。**

## 二、编码

- 方案与判定规则见 `coding_sheet.md`（三级量表 0/1/2）。
- 编码者：作者本人（设计了探针，无法盲）；如第二位同事独立编码，报告
  精确一致率与 Cohen's κ（三级）。论文措辞统一为 "author-coded
  invitation-strength covariate"。
- CTL 域仅作效度锚（预期 ≈0），不进入域比较。

## 三、分析计划（按序执行，全部基于冻结的 R3 双评委数据与 4,027 共同支持集）

- **A1（描述）**：REG-H/REG-S/UNREG 三域的强度分布（每域 15 探针的
  均值、中位数、逐级计数）；域间两两均值差 + 探针级 bootstrap 95% CI
  （2,000 次重抽，种子 20260915）。
- **A2（主检验，response 级）**：以 S1（DeepSeek 单评委 candidate）标签为
  结果，logistic 回归 `positive ~ domain + invitation_strength`，探针聚类
  稳健 SE（4,027 条）。报告加入强度**前后** UNREG 域系数的变化
  （衰减比例）及其显著性。
- **A3（分层敏感性）**：按强度分层（仅使用 n≥2 探针的层），层内重算
  UNREG − max(REG-H, REG-S) 的 margin；层数据稀疏时只报告方向。

## 四、预注册判据

- **判据 J1（归因）**：A2 中加入强度后，UNREG 系数衰减 ≥50% **或**
  变为不显著（p≥.05）→ 判"候选排名大部分可由邀请强度解释"。
- **判据 J2（独立性）**：衰减 <50% 且保持显著（p<.05）→ 判"域差异
  超出邀请强度所能解释的范围"。
- **判据 J3（编码效度）**：UNREG 强度均值不高于任一受监管域，或 CTL
  强度均值 >0.5 → 编码方案效度存疑，结果只作探索性报告并如实说明。

## 五、双分支论文措辞（预写，结果出来后填空）

**分支 A（J2 成立——margin 独立于强度，强化）**
- §5.2 加一句：*Conditioning on the author-coded invitation strength of
  each probe leaves the UNREG-highest margin positive (Appendix~X).*
- Ledger "Claim-relevant checks" 行追加 "probe invitation-strength control"。

**分支 B（J1 成立——margin 大部分由强度解释，收窄解释层）**
- §6.2 改写：*...the surviving ranking is largely attributable to the
  coded invitation strength of the UNREG probes (Appendix~X),
  consistent with its manipulation-check reading.*
- Ledger "Boundary" 行追加 "invitation strength accounts for the UNREG
  margin on this benchmark"。
- 摘要与斜体 scoped 句**不动**（它们陈述的是操作率事实，不受影响）。

**分支 C（J3 触发——编码无效度）**
- §4.1 邀请句软化为中性描述；§6.2 "manipulation check" 句改写为
  构念差异表述；结果仅入附录并说明局限。

## 六、产出

- 附录 "Probe Invitation-Strength Control"（编码方案、分布表、A2 表、
  判据结果、编码者一致性如适用）。
- 按分支更新 §5.2 / §6.2 / ledger。
- 两个验证脚本追加本分析的锁数校验。
