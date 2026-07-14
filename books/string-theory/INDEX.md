# 总索引

本文档是全书统一索引入口。定理、习题、术语、归一化和资料源分别由专门索引维护；本文件负责把它们组织成可阅读路径。

## 1. 全书工具索引

| 需求 | 文件 |
|---|---|
| 阅读顺序与目录 | [README.md](README.md) |
| 符号约定 | [NOTATION.md](NOTATION.md) |
| 归一化、regulator 与渐近约定 | [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) |
| 压缩术语表 | [GLOSSARY.md](GLOSSARY.md) |
| 主线例题集 | [WORKED_EXAMPLES.md](WORKED_EXAMPLES.md) |
| 定理、命题、猜想状态 | [THEOREM_INDEX.md](THEOREM_INDEX.md) |
| 习题与解答索引 | [EXERCISE_INDEX.md](EXERCISE_INDEX.md) |
| 核心习题解答 | [SOLUTIONS.md](SOLUTIONS.md) |
| 资料源与原始输入代码 | [SOURCES.md](SOURCES.md) |
| 逐章资料源映射 | [REFERENCE_MAP.md](REFERENCE_MAP.md) |
| 主线证明链 | [MAINLINE_PROOF_CHAINS.md](MAINLINE_PROOF_CHAINS.md) |
| 内容收口审定 | [CONTENT_CLOSURE_AUDIT.md](CONTENT_CLOSURE_AUDIT.md) |

## 2. 主线概念索引

| 概念 | 主位置 | 支撑文件 |
|---|---|---|
| 作用量、变分、stress tensor | [第 1 章](01_relativistic_fields_and_sigma_models.md) | [附录 A](A_geometry_forms_curvature.md) |
| Nambu-Goto 与 Polyakov 作用量 | [第 2 章](02_classical_bosonic_string.md) | [归一化表](NORMALIZATION_TABLE.md) |
| CFT、OPE、Virasoro | [第 3 章](03_worldsheet_cft.md) | [附录 B](B_cft_formulae_virasoro.md) |
| 玻色弦量子化与质量公式 | [第 4 章](04_canonical_quantization_bosonic_string.md) | [归一化表](NORMALIZATION_TABLE.md) |
| Ghost、BRST、moduli measure | [第 5 章](05_path_integral_brv_quantization.md) | [主线证明链](MAINLINE_PROOF_CHAINS.md) |
| 顶点算子和散射振幅 | [第 6 章](06_vertex_operators_and_string_scattering.md) | [习题索引](EXERCISE_INDEX.md) |
| 紧化、T-duality、D-branes | [第 7 章](07_compactification_t_duality_and_dbranes.md) | [归一化表](NORMALIZATION_TABLE.md) |
| RNS、GSO、picture number | [第 8 章](08_rns_superstrings_and_gso.md) | [附录 E](E_spinors_supersymmetry.md) |
| Type II 与 GS 形式 | [第 9 章](09_green_schwarz_type_ii.md) | [资料源映射](REFERENCE_MAP.md) |
| Heterotic、lattice、anomaly | [第 10 章](10_heterotic_strings_and_anomalies.md) | [附录 C](C_lie_algebras_anomalies.md) |
| 低能有效作用 | [第 11 章](11_low_energy_effective_actions.md) | [归一化表](NORMALIZATION_TABLE.md) |
| DBI、WZ、worldvolume theory | [第 12 章](12_dbrane_effective_theory.md) | [术语表](GLOSSARY.md) |
| Calabi-Yau 与四维有效理论 | [第 13 章](13_calabi_yau_compactification.md) | [附录 A](A_geometry_forms_curvature.md) |
| S-duality、U-duality、M-theory | [第 14 章](14_dualities_and_m_theory.md) | [定理索引](THEOREM_INDEX.md) |
| 高 genus 扰动论 | [第 15 章](15_riemann_surfaces_and_perturbation_theory.md) | [附录 D](D_riemann_surfaces_spin_theta.md) |
| Topological strings 与 mirror symmetry | [第 16 章](16_topological_strings_and_mirror_symmetry.md) | [资料源](SOURCES.md) |
| Black branes 与 entropy | [第 17 章](17_black_branes_and_entropy.md) | [习题解答](SOLUTIONS.md) |
| AdS/CFT | [第 18 章](18_ads_cft.md) | [定理索引](THEOREM_INDEX.md) |
| Flux compactification 与 landscape | [第 19 章](19_flux_compactification_and_landscape.md) | [资料源映射](REFERENCE_MAP.md) |
| D-brane/QFT、指标、曲线计数与模性接口 | [第 20 章](20_interfaces_with_qft_geometry_number_theory.md) | [资料源](SOURCES.md) |

## 3. 定理状态索引

全书非平凡陈述统一登记在 [THEOREM_INDEX.md](THEOREM_INDEX.md)。状态含义：

- `P`：正文给出证明。
- `S`：正文给出标准物理推导说明。
- `E`：外部输入定理。
- `C`：物理猜想或对偶性原则。

阅读时若某陈述用于后续章节，应优先查看该索引确认其证明状态。

## 4. 习题索引

全部 70 道正文习题都在 [EXERCISE_INDEX.md](EXERCISE_INDEX.md) 中登记，并在 [SOLUTIONS.md](SOLUTIONS.md) 中给出核心解答。若以后新增习题，必须同步更新这两个文件。

## 5. 严格性边界速查

| 需要区分的层级 | 主位置 | 台账 |
|---|---|---|
| Polyakov metric equation、conformal-gauge constraint、离壳域与壳上物理态 | [第 2 章](02_classical_bosonic_string.md)、[第 4 章](04_canonical_quantization_bosonic_string.md) | 第 4 章定义 4.4B--C、4.8；[定理索引](THEOREM_INDEX.md) 2.6、2.9A、4.9 |
| 有限能 operator domain、Virasoro/ghost central term、BRST complex | [第 3--5 章](03_worldsheet_cft.md) | [归一化表](NORMALIZATION_TABLE.md) 第 6、9 节 |
| Polyakov/FP path integral 与 determinant regulator | [第 5 章](05_path_integral_brv_quantization.md) | [资料源](SOURCES.md) `POLY81` |
| Reduced amplitude、绝对 normalization、解析延拓与 factorization | [第 6 章](06_vertex_operators_and_string_scattering.md) | [归一化表](NORMALIZATION_TABLE.md) 第 8、9 节 |
| GSO 单弦投影、modular invariance、target anomaly | [第 8 章](08_rns_superstrings_and_gso.md)、[第 10 章](10_heterotic_strings_and_anomalies.md) | [定理索引](THEOREM_INDEX.md) 8.16、8.16A、10.11 |
| Exact compact-boson T-duality、Buscher input、几何 D-brane 模型 | [第 7 章](07_compactification_t_duality_and_dbranes.md) | [资料源](SOURCES.md) `BUS87/88`、`NAR86` |
| $g_s$ loop、$\alpha'$ derivative 与低能 truncation | [第 11 章](11_low_energy_effective_actions.md) | [归一化表](NORMALIZATION_TABLE.md) 第 9 节 |
| Calabi--Yau 定义、holonomy、moduli、KK truncation | [第 13 章](13_calabi_yau_compactification.md) | [资料源映射](REFERENCE_MAP.md) |
| D1-D5 charge convention、普通 Cardy、K3 elliptic genus/fixed-index Jacobi 渐近、index/绝对简并边界与 quantum entropy | [第 17 章](17_black_branes_and_entropy.md) | [定理索引](THEOREM_INDEX.md) 17.7--17.12A、[资料源](SOURCES.md) `SW99`、`DMVV97`、`EZ85/DMZ12` |
| D3 supergravity input、AdS PDE calculation、GKPW conjecture | [第 18 章](18_ads_cft.md) | [定理索引](THEOREM_INDEX.md) 18.1--18.12 |
| 拉伸弦/Higgs 字典、Dirac index、A-model 生成函数与 elliptic-genus 模性 | [第 20 章](20_interfaces_with_qft_geometry_number_theory.md) | [定理索引](THEOREM_INDEX.md) 20.2--20.17 |
