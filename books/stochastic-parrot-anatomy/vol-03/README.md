# 卷三 概率、复现与评测

本卷把概率论与复现理论组织成一条十六章的问题链：概率对象进入随机算法与语言模型接口，再进入相同关系、状态、浮点、并发、训练/推理复现、统计复制、评测与可执行合同。概率篇使用 `P` locator，复现篇使用 `R` locator，跨卷审计接口使用 `S` locator。

1. [概率、随机性与复现问题](ch01_probability_randomness_reproducibility.md)
2. [概率空间与可测性](ch02_probability_spaces_measurability.md)
3. [随机变量、分布与积分](ch03_random_variables_distribution_integration.md)
4. [独立性、随机核与条件信息](ch04_independence_kernels_conditioning.md)
5. [收敛、极限定理与有限样本界](ch05_convergence_limit_finite_sample.md)
6. [熵、评分、校准与决策](ch06_entropy_scoring_calibration_decision.md)
7. [随机算法与语言模型概率](ch07_randomized_algorithms_lm_probability.md)
8. [观察、干预与概率语言边界](ch08_observation_intervention_probability_language.md)
9. [相同关系与观察投影](ch09_sameness_observation_projection.md)
10. [状态、RNG 与 checkpoint](ch10_state_rng_checkpoint.md)
11. [浮点、并发与分布式执行](ch11_floating_point_concurrency.md)
12. [训练与推理复现](ch12_training_inference_reproducibility.md)
13. [数据、环境、provenance 与科学复制](ch13_data_environment_provenance_replication.md)
14. [统计复制与等效判决](ch14_statistical_replication_equivalence.md)
15. [基准、能力与风险决策](ch15_benchmarks_capability_risk.md)
16. [首分叉诊断与可执行合同](ch16_first_divergence_executable_contract.md)

卷内配套按材料类型统一保存，并在文件内部保留 `P`/`R` 分区：[来源](SOURCES.md)、[定理与主张责任表](RESPONSIBILITY_LEDGER.md)、[符号与术语](REFERENCE.md)、[习题解答](SOLUTIONS.md)。本卷文件解答 `P`/`R` 习题；编号为 `S` 的跨卷综合题统一在[全书解答](../SOLUTIONS.md)中处理。

长证明只在有实质增量时进入附录：[附录 D](../appendices/app-d_probability_decision_kernel.md)保留一般 Jensen、两两独立弱大数律、有限条件期望、有限 KL 数据处理、随机源实现和 ATE 识别；[附录 F](../appendices/app-f_reproducibility_numerical_kernel.md)保留合同可判定性、程序等价不可判定性和首分叉三分。其余课程证明以本卷正文为唯一位置。
