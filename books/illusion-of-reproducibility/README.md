# 可复现性的幻觉

**状态/定位：已收口的计算与科学复现教材；可执行合同、来源表、解答和闭合审计的当前入口见 [STATUS.md](STATUS.md)。**

## 从同一个 seed 到同一个科学结论

**作者：Dr. Stochastic Parrot**

这是一部关于计算与科学复现的中文教材。标题中的“幻觉”不是说复现不重要，而是说人们常把一种复现误认成另一种：同一脚本跑出相同字节，不保证理论结论稳健；不同 GPU 出现最低位差异，也不必推翻实验；设置同一个 seed，更不等于固定了程序的全部状态。

本书从“什么叫相同”开始，把精确等价、容差接受、统计相容和科学支持分开，继而建立程序状态、伪随机流、IEEE 754 误差、并发偏序、机器学习训练、推理服务、内容身份、制品谱系和统计复制的统一合同：合法证据返回三值判决，schema 不合法返回结构错误。所有充分性与必要性都相对于明确实现、观察和允许环境陈述。

## 阅读方式

正文假定读者熟悉集合与函数、基础概率统计、有限和及归纳法，并能阅读简短程序执行
轨迹；所需的数值分析、并发语义和统计检验条件会在使用处给出。IEEE 754、语言与
框架行为、机构术语及精确抽样分布作为标明版本的外部输入，不要求读者预先掌握其
完整规范。

各章从一项可追踪的失败或计算进入。第一至四章建立“相同”、状态、舍入与调度；
第五至七章把它们用于训练、推理和制品；第八、九章处理跨领域词典与统计复制；
第十、十一章把诊断结果收束为可执行合同。若只处理一次工程事故，可从第十章进入，
再沿交叉引用回查所需模型；若要建立完整复现协议，宜按目录顺读。

## 目录

1. [序章：复现不是一个无参数谓词](00_preface_and_scope.md)
2. [相同的五种含义](01_relations_of_sameness.md)
3. [状态、随机源与确定性执行](02_state_rng_and_execution.md)
4. [浮点算术、误差与运算次序](03_floating_point_and_error.md)
5. [偏序、调度与分布式执行](04_parallel_and_distributed.md)
6. [训练复现的条件与证据](05_training_reproducibility.md)
7. [推理、采样与服务端漂移](06_inference_reproducibility.md)
8. [内容身份、环境与制品谱系](07_data_environment_and_artifacts.md)
9. [ACM、NASEM 与 VIM 的不同词典](08_scientific_reproduction.md)
10. [统计相容、等效与多重比较](09_statistical_replication.md)
11. [失败模式、首分叉与诊断](10_failure_modes.md)
12. [可执行复现合同与判定边界](11_reproducibility_contract.md)

配套：[状态说明](STATUS.md)、[术语表](GLOSSARY.md)、[来源](SOURCES.md)、[主张责任表](CLAIM_LEDGER.md)、[习题解答](SOLUTIONS.md)、[闭合审计](CLOSURE_AUDIT.md)。
