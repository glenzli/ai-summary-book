# 定理索引与证明状态

本索引登记正文全部 `T*` 与 `EI-*` 标签。`P` 表示正文给出完整书内证明；`E` 表示本书只调用 [SOURCES.md](SOURCES.md) 固定的外部结果。局部编号引理和命题保留在相应章节，不另造全书标签。

| 标签 | 名称 | 文件 | 状态 | 直接证明依赖 |
| --- | --- | --- | --- | --- |
| T0.1 | 对象层级的非自动传递原则 | `00_preface_scope_and_metatheory.md` | P | 定义 0.1--0.4，三个反例 |
| T1.1 | 有限配置的一步函数可计算 | `01_effective_procedures_and_machines.md` | P | 引理 1.4，定义 1.5 |
| EI-1 | 通用解释器与经典模型等价 | `01_effective_procedures_and_machines.md`、`SOURCES.md` | E | S1--S3、S14 |
| T2.1 | 对角停机集合不可判定 | `02_undecidability_and_reductions.md` | P | 定义 1.6，命题 1.8 |
| T2.2 | many-one 归约的传递性 | `02_undecidability_and_reductions.md` | P | 定义 2.4 |
| T2.3 | Rice 定理 | `02_undecidability_and_reductions.md` | P | T2.1，命题 2.5，EI-1，EI-2 |
| EI-2 | 可接受程序系统的 s-m-n 参数定理 | `02_undecidability_and_reductions.md`、`SOURCES.md` | E | S3 |
| T3.1 | 替换在 alpha 商上良定义 | `03_lambda_calculus_and_combinatory_computation.md` | P | 引理 3.4，捕获避免替换定义 |
| T3.2 | beta 正规形唯一性 | `03_lambda_calculus_and_combinatory_computation.md` | P | EI-3 |
| EI-3 | 无类型 lambda 演算的 Church--Rosser 合流性 | `03_lambda_calculus_and_combinatory_computation.md`、`SOURCES.md` | E | S4 |
| T4.1 | STLC weakening | `04_simple_types_and_normalization.md` | P | 定义 4.2、4.4 |
| T4.2 | STLC 替换定理 | `04_simple_types_and_normalization.md` | P | T4.1，T3.1 |
| T4.3 | STLC preservation | `04_simple_types_and_normalization.md` | P | T4.2，引理 4.2 |
| T4.4 | STLC progress | `04_simple_types_and_normalization.md` | P | 引理 4.4 |
| T4.5 | STLC 多步类型安全 | `04_simple_types_and_normalization.md` | P | T4.3，T4.4 |
| EI-4 | 纯 STLC 强正规化 | `04_simple_types_and_normalization.md`、`SOURCES.md` | E | S5、S6 |
| T5.1 | MLTT 弱化与依赖替换 | `05_dependent_types_and_constructive_logic.md` | P | 定义 5.1--5.10 的同时推导归纳 |
| T5.2 | MLTT regularity | `05_dependent_types_and_constructive_logic.md` | P | T5.1 |
| T5.3 | 自然演绎规则的类型保持翻译 | `05_dependent_types_and_constructive_logic.md` | P | 定义 5.6--5.9，T5.1 |
| EI-5 | 固定 MLTT 核心的弱头正规化与 Nat canonicity | `05_dependent_types_and_constructive_logic.md`、`SOURCES.md` | E | S7；S8 仅作补充 |
| T6.1 | System F preservation | `06_polymorphism_recursion_and_effects.md` | P | 引理 6.4--6.6 |
| T6.2 | Kleisli 合成的单位律与结合律 | `06_polymorphism_recursion_and_effects.md` | P | 定义 6.10 的三条 monad 律 |
| EI-6 | 纯 System F 关系参数性 | `06_polymorphism_recursion_and_effects.md`、`SOURCES.md` | E | S9、S10 |
| T7.1 | while 命令终止时大小步等价 | `07_operational_semantics_and_abstract_machines.md` | P | 引理 7.4、7.5 |
| T7.2 | 唯一分解蕴含小步确定性 | `07_operational_semantics_and_abstract_machines.md` | P | 定义 7.7 |
| T7.3 | CEK 机与 CBV 求值等价 | `07_operational_semantics_and_abstract_machines.md` | P | T7.2，引理 7.11、7.12 |
| EI-7 | 特定 SOS 格式的双模拟同余定理 | `07_operational_semantics_and_abstract_machines.md`、`SOURCES.md` | E | S12、S13 |
| T8.1 | Kleene 最小不动点定理 | `08_denotational_semantics_domains_and_fixed_points.md` | P | 定义 8.1、8.2 |
| T8.2 | 偏状态函数域是带底 omega-cpo | `08_denotational_semantics_domains_and_fixed_points.md` | P | 图包含序，递增图并 |
| T8.3 | 命令指称与大步语义双向一致 | `08_denotational_semantics_domains_and_fixed_points.md` | P | T8.1，T8.2，引理 8.5、8.6，第 7 章大步规则 |
| EI-8 | PCF computational adequacy 与游戏语义完全抽象 | `08_denotational_semantics_domains_and_fixed_points.md`、`SOURCES.md` | E | S17--S19 |
| T9.1 | 基本 Hoare 系统 soundness | `09_axiomatic_semantics_logics_and_verification.md` | P | 引理 9.4，定义 9.1--9.3 |
| EI-9 | Cook 相对完备性 | `09_axiomatic_semantics_logics_and_verification.md`、`SOURCES.md` | E | S20、S21 |
| T10.1 | 类型安全、指称观察可靠性与 Hoare soundness 彼此不蕴含 | `10_expressivity_full_abstraction_and_synthesis.md` | P | T4.5，T9.1，三个显式语言包 |
| T10.2 | 可接受语言的非平凡外延性质不可通判 | `10_expressivity_full_abstraction_and_synthesis.md` | P | 定义 10.6，T2.3 |

## 状态闭合

- `P` 条目只能依赖本表更早的 `P` 结果、正文局部引理或明确列出的 `E` 输入。
- `E` 条目的精确对象系统、假设、结论、版本和定位均由 [SOURCES.md](SOURCES.md) 承担。
- EI-4、EI-5、EI-6、EI-7、EI-8、EI-9 分别只支持正文声明的固定演算或格式，不向扩张系统自动迁移。
