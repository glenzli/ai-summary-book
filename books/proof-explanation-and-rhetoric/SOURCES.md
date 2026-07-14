# 资料源与外部输入登记

本文件记录正文实际调用的外部结果及拓展阅读。网页访问日期统一为 **2026-07-14**；经验论文只支持其模型、任务、数据和协议，不作为跨系统全称定理。

## 一、正文外部输入

### EXT-4.7：支配收敛定理

- **正文版本**：测度空间 $(\Omega,\mathcal F,\mu)$ 上，$f_n,f$ 均可测，$f_n\to f$ 几乎处处，且有非负 $g\in L^1(\mu)$ 几乎处处支配全部 $|f_n|$；则 $f\in L^1$ 且积分收敛。正文推论 4.8 再对 $|f_n-f|$ 应用该定理，书内证明 $L^1$ 收敛。
- **用途**：第四章 4.10 说明交换极限与积分所需条件；第九章 9.2 展示数学来源的定位、调用与误用修订。
- **来源**：Gerald B. Folland, *Real Analysis: Modern Techniques and Their Applications*, 2nd ed., Wiley, 1999, Theorem 2.24.
- **边界**：本书不重建 Lebesgue 积分、Fatou 引理或测度完备化；正文显式假设极限代表元 $f$ 可测，避免在非完备测度空间上隐藏代表元问题。

### EXT-5.4：经典命题自然演绎的可靠性与完备性

- **正文版本**：对 *The Open Logic Text* 指定的经典命题语言、赋值语义与自然演绎系统，$\Gamma\vdash A$ 当且仅当 $\Gamma\models A$。
- **用途**：第五章 5.4，连接语法推导与语义蕴涵。
- **来源**：Open Logic Project, [*The Open Logic Text*](https://builds.openlogicproject.org/open-logic-complete.pdf), release `9620cc7` (2026-07-12)：第 10 章自然演绎系统的可靠性见 Theorem 10.22；命题逻辑完备性的模型存在形式见 Theorem 13.6，语义后承推出可推导性的形式见 Corollary 13.7，并在自然演绎分支调用 Proposition 10.16。
- **定位校正**：Corollary 10.24 是“可满足推出一致”，不是自然演绎可靠性 $\Gamma\vdash A\Rightarrow\Gamma\models A$ 的直接陈述；Corollaries 9.28、11.26、12.32 分属其他演算的相应一致性结论，不能列作“联结词基底版本”。
- **边界**：本书证明一般的“带上下文规则局部保真推出有限推导可靠”原则，不复制该书逐规则核验和极大一致集构造；结论不自动推广到其他逻辑或演算。

## 二、逻辑、证明与形式化

- Herbert B. Enderton, *A Mathematical Introduction to Logic*, 2nd ed., Academic Press, 2001, Chapters 1--2；命题与一阶逻辑的语法、语义、可靠性和完备性。
- Dirk van Dalen, *Logic and Structure*, 5th ed., Springer, 2013, Chapters 1--3；自然演绎、直觉主义/经典逻辑与证明论基础。
- P. D. Magnus, Tim Button, Robert Trueman, Richard Zach, [*forall x: Calgary*](https://forallx.openlogicproject.org/html/), Fall 2025 release `f16479e`；TFL/FOL 语法、语义和自然演绎。第 22 章只概述可靠性与完备性关系，完整可靠性证明在第 48 章；本书不把第 22 章误作完备性完整证明来源。
- Benjamin C. Pierce et al., [*Software Foundations, Volume 1: Logical Foundations*](https://softwarefoundations.cis.upenn.edu/), current online edition；归纳、关系、证明对象与程序语义案例。
- Jeremy Avigad, Leonardo de Moura, Soonho Kong, Sebastian Ullrich, [*Theorem Proving in Lean 4*](https://docs.lean-lang.org/theorem_proving_in_lean4/)；依赖类型、命题与证明、归纳和公理。
- Lean Project, [*Elaboration and Compilation*](https://lean-lang.org/doc/reference/latest/Elaboration-and-Compilation/) 与 [*Validating a Lean Proof*](https://lean-lang.org/doc/reference/latest/ValidatingProofs/)；elaborator、kernel checking、独立 checker 与 statement 对齐的官方版本化说明。

## 三、数学实践与解释

- Paolo Mancosu, [*Explanation in Mathematics*](https://plato.stanford.edu/archives/win2024/entries/mathematics-explanation/), *Stanford Encyclopedia of Philosophy*, Winter 2024 archive；解释性证明、统一与数学解释争议的综述。
- Philip Kitcher, “Explanatory Unification,” *Philosophy of Science* 48(4), 1981, 507--531, [doi:10.1086/289019](https://doi.org/10.1086/289019)；统一主义解释方案。
- Henk W. de Regt, *Understanding Scientific Understanding*, Oxford University Press, 2017, [doi:10.1093/oso/9780190652913.001.0001](https://doi.org/10.1093/oso/9780190652913.001.0001)；解释、可理解性与读者背景。
- George Pólya, *How to Solve It*, 2nd ed., Princeton University Press, 1957；启发式与发现方法。本书只把它用作发现工具来源，不当作证明规则。

## 四、科学、因果与统计证据

- Carl G. Hempel and Paul Oppenheim, “Studies in the Logic of Explanation,” *Philosophy of Science* 15(2), 1948, 135--175, [doi:10.1086/286983](https://doi.org/10.1086/286983)；演绎律则模型的原始来源。
- Philip Kitcher, “Explanatory Unification,” 同上；科学解释的统一路线。
- James Woodward, *Making Things Happen: A Theory of Causal Explanation*, Oxford University Press, 2003；干预主义因果解释。
- Judea Pearl, *Causality: Models, Reasoning, and Inference*, 2nd ed., Cambridge University Press, 2009, especially Chapter 3, [doi:10.1017/CBO9780511803161](https://doi.org/10.1017/CBO9780511803161)；结构因果模型、干预与识别。
- Jerzy Neyman, “Outline of a Theory of Statistical Estimation Based on the Classical Theory of Probability,” *Philosophical Transactions of the Royal Society A* 236, 1937, 333--380；置信区间程序与覆盖率的经典来源。
- Ronald L. Wasserstein and Nicole A. Lazar, “The ASA's Statement on p-Values: Context, Process, and Purpose,” *The American Statistician* 70(2), 2016, 129--133, [doi:10.1080/00031305.2016.1154108](https://doi.org/10.1080/00031305.2016.1154108)；p 值的适用解释与常见误读。

## 五、图示、论证与引用

- Stephen Toulmin, *The Uses of Argument*, updated ed., Cambridge University Press, 2003；claim/data/warrant 等论证角色。该框架不是本书形式有效性的替代定义。
- Edward R. Tufte, *The Visual Display of Quantitative Information*, 2nd ed., Graphics Press, 2001；数据图的尺度、编码与图形完整性。
- Alberto Cairo, *The Functional Art*, New Riders, 2012；可视化的解释与传播实践。
- Committee on Publication Ethics, [*COPE Guidelines*](https://publicationethics.org/guidance/Guidelines)；来源归属、引用与出版伦理的规范背景。法律与许可结论仍须查目标司法辖区和具体许可证。

## 六、AI 推理、验证与忠实性

### 经验输入边界

#### EMP-9.3-TURPIN：提示偏置下的 CoT 忠实性

- **正文版本**：在 GPT-3.5 text-davinci-003 与 Claude 1.0 上，比较有无指定提示偏置时的答案与 CoT；BBH 部分使用从 23 个任务中选出的 13 个任务及 3,299 个评价样例，并测试 “Answer is Always A” 与 “Suggested Answer” 两类偏置。正文只使用偏置会影响答案而 CoT 通常不披露该影响这一协议内发现。
- **用途**：第九章 9.3 展示经验来源追踪；第十章 10.6 说明 CoT 忠实性结论需要模型、任务与干预协议下标。
- **来源**：Miles Turpin, Julian Michael, Ethan Perez, Samuel R. Bowman, [*Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting*](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ed3fea9033a80fea1376299fa7863f4a-Abstract-Conference.html), *Advances in Neural Information Processing Systems* 36 (NeurIPS 2023)，特别是第 2--4 节；正式论文集版本。
- **边界**：结果不量化全部模型、任务、解码设置或忠实性定义，也不把反事实行为测试等同于直接读取任意内部计算过程。

- Tamera Lanham et al., [*Measuring Faithfulness in Chain-of-Thought Reasoning*](https://arxiv.org/abs/2307.13702), arXiv:2307.13702；使用 CoT 改写、截断与错误注入等干预，报告模型与任务间异质性。按预印本状态引用。
- Yanda Chen et al., [*Reasoning Models Don't Always Say What They Think*](https://arxiv.org/abs/2505.05410), arXiv:2505.05410, 2025；六类提示线索上的披露率实验。正文不把该设置外推为所有推理模型的性质。
- Alon Jacovi and Yoav Goldberg, [*Towards Faithfully Interpretable NLP Systems: How Should We Define and Evaluate Faithfulness?*](https://aclanthology.org/2020.acl-main.386/), ACL 2020, 4198--4205, [doi:10.18653/v1/2020.acl-main.386](https://doi.org/10.18653/v1/2020.acl-main.386)；忠实性评价口径与二元/分级边界。

### 可执行推理与验证

- Qing Lyu et al., [*Faithful Chain-of-Thought Reasoning*](https://aclanthology.org/2023.ijcnlp-main.20/), IJCNLP-AACL 2023, 305--329, [doi:10.18653/v1/2023.ijcnlp-main.20](https://doi.org/10.18653/v1/2023.ijcnlp-main.20)；自然语言到符号链再由确定性求解器执行。正文只主张答案对显式链的系统级依赖，不主张完整隐藏过程透明。
- Jin Peng Zhou et al., [*Don't Trust: Verify -- Grounding LLM Quantitative Reasoning with Autoformalization*](https://openreview.net/forum?id=V5tdi14ple), ICLR 2024；Isabelle autoformalization 与验证在 GSM8K、MATH、MultiArith 上的筛选实验及 statement 风险。
- Hunter Lightman et al., [*Let's Verify Step by Step*](https://proceedings.iclr.cc/paper_files/paper/2024/hash/aca97732e30bcf1303bc22ac3924fd16-Abstract-Conference.html), ICLR 2024；MATH 子集上的过程监督与 PRM800K。正文将其视为经验训练结果，不把过程奖励模型当形式证明内核。
- Karl Cobbe et al., [*Training Verifiers to Solve Math Word Problems*](https://arxiv.org/abs/2110.14168), arXiv:2110.14168, 2021；GSM8K 与学习型 verifier 的经验结果。学习型评分器的“verifier”名称不自动蕴含定义 10.2 的逻辑可靠性。

## 七、未在书内重证的范围

本书不重证一般一阶完备性、Gödel 不完备性、测度论收敛定理、Pearl 的识别演算、证明助手核心类型理论元定理，或上述经验论文的全部实验。正文仅调用已在本文件登记的版本；未登记的深结论不得以“经典”或“研究表明”直接进入证明链。
