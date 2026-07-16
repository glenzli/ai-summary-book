# 概率、随机过程与信息论：从测度到熵

**状态：内容本体收口版。** 本目录是一套中文 Markdown 教材，目标是在一个统一的测度论框架中给出概率、随机过程与信息论的共同语言。正文只在本目录内自含展开；大型测度论、遍历论与编码定理作为外部输入登记在 [SOURCES.md](SOURCES.md) 与 [THEOREM_INDEX.md](THEOREM_INDEX.md)。

本书的中心问题是：给定一个概率空间、一个时间索引族和一个可观测函数族，哪些量只是分布的函数，哪些量依赖路径结构，哪些量度量信息损失、可压缩性或可传输性。答案必须同时说明对象类型、可测性、极限方式和证明边界。

## 章节构成

1. [序章：范围、层次与证明责任](00_preface_measure_entropy.md)
2. [$\sigma$-代数、测度与可测结构](01_sets_sigma_algebras_and_measures.md)
3. [概率空间、随机变量与分布](02_probability_spaces_random_variables_and_laws.md)
4. [Lebesgue 积分、期望与不等式](03_lebesgue_integration_and_expectation.md)
5. [独立性、乘积与随机核](04_independence_products_and_kernels.md)
6. [条件期望、滤过与鞅](05_conditioning_and_martingales.md)
7. [收敛方式与极限定理](06_convergence_and_limit_theorems.md)
8. [随机过程与 Markov 链](07_stochastic_processes_and_markov_chains.md)
9. [熵、散度与互信息](08_entropy_divergence_and_information.md)
10. [信道、编码与信息界](09_information_channels_and_coding.md)
11. [熵率、遍历输入与模型接口](10_ergodic_asymptotic_information_and_models.md)

附录：

- [附录 A：测度扩张、正则条件分布与外部输入](A_measure_extension_and_regular_conditional_probability.md)
- [附录 B：实分析、凸性与极限工具](B_real_analysis_and_convexity_tools.md)
- [附录 C：有限字母表计算表](C_finite_alphabet_calculation_tables.md)

配套文件：

- [写作规范](SKILL.md)
- [符号表](NOTATION.md)
- [资料源与外部输入](SOURCES.md)
- [定理索引](THEOREM_INDEX.md)
- [依赖图](DEPENDENCY_GRAPH.md)
- [内容闭合审计](CONTENT_CLOSURE_AUDIT.md)
- [习题解答](SOLUTIONS.md)

## 严格性约定

正文中的非平凡断言只允许四种终态：书内定理、外部输入定理、逐步推导或明确的模型假设。测度扩张、Radon--Nikodym、Fubini--Tonelli、Kolmogorov 扩张、正则条件分布、强大数律、中心极限定理、Birkhoff 遍历定理、Shannon--McMillan--Breiman 定理和 Shannon 编码定理不在正文中重证；它们被精确列入外部输入。

本书默认读者熟悉集合、函数、实数序列和有限维线性代数。需要的测度论接口在第一至三章给出，未证明的大型存在定理均在附录 A 标明。
