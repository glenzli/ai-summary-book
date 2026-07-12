# String Theory：从世界面量子场论到对偶性

作者：Dr. Stochastic Parrot
状态：严格教材第二版，已达到教材内容层面收口，进入出版化细化阶段
主资料源：Green-Schwarz-Witten, Polchinski, Becker-Becker-Schwarz, Zwiebach, Di Francesco-Mathieu-Senechal, Ginsparg, Kiritsis, Johnson, Blumenhagen-Lust-Theisen, Hori et al., Maldacena, Witten

本书目标是写成一部数学化、专业化、成体系的 string theory 教材，而不是科普导览。正文从相对论场论、世界面作用量和二维共形场论开始，逐步进入玻色弦量子化、BRST、散射振幅、D-branes、超弦、heterotic string、低能有效作用、紧化、对偶性、AdS/CFT、拓扑弦和镜像对称。

## 写作约束

本书写作约束见 [SKILL.md](SKILL.md)。后续写作必须遵守：

- 定义先于物理图像；每个关键公式必须声明时空签名、世界面签名、$\alpha'$ 归一化和正规序约定。
- 非平凡命题必须给出证明、标准物理推导说明，或标注为“外部输入定理”。
- 世界面 CFT、BRST、ghost、modular invariance、anomaly cancellation、supersymmetry 和 duality 不得只用口号描述。
- 低能极限、有效作用和散射振幅必须说明所处近似阶数：tree level、genus expansion、$\alpha'$ expansion 或 string coupling expansion。
- 资料源必须能在 [SOURCES.md](SOURCES.md) 中追溯。

总索引见 [INDEX.md](INDEX.md)，压缩术语表见 [GLOSSARY.md](GLOSSARY.md)。符号约定见 [NOTATION.md](NOTATION.md)，归一化总表见 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md)，主线例题集见 [WORKED_EXAMPLES.md](WORKED_EXAMPLES.md)，数学审查记录见 [MATH_REVIEW.md](MATH_REVIEW.md)，收口标准见 [CLOSURE_STATUS.md](CLOSURE_STATUS.md)。正式教材完整性标准见 [FORMAL_TEXTBOOK_COMPLETENESS.md](FORMAL_TEXTBOOK_COMPLETENESS.md)，内容收口审定见 [CONTENT_CLOSURE_AUDIT.md](CONTENT_CLOSURE_AUDIT.md)。主要定理状态见 [THEOREM_INDEX.md](THEOREM_INDEX.md)，章节依赖见 [DEPENDENCY_GRAPH.md](DEPENDENCY_GRAPH.md)，主线证明链见 [MAINLINE_PROOF_CHAINS.md](MAINLINE_PROOF_CHAINS.md)，概念审定见 [CONCEPTUAL_AUDIT.md](CONCEPTUAL_AUDIT.md)，章节收口表见 [CHAPTER_CLOSURE_AUDIT.md](CHAPTER_CLOSURE_AUDIT.md)，习题覆盖见 [EXERCISE_COVERAGE.md](EXERCISE_COVERAGE.md)，习题索引见 [EXERCISE_INDEX.md](EXERCISE_INDEX.md)，编号审定见 [NUMBERING_CROSSREF_AUDIT.md](NUMBERING_CROSSREF_AUDIT.md)，逐章资料源映射见 [REFERENCE_MAP.md](REFERENCE_MAP.md)，核心习题解答见 [SOLUTIONS.md](SOLUTIONS.md)。

## 总目录

### 第一部分：世界面与玻色弦

0. [序章：范围、严格性和主线](00_preface_and_scope.md)
1. [第一章：相对论场论、作用量和 sigma model 语言](01_relativistic_fields_and_sigma_models.md)
2. [第二章：经典玻色弦、Nambu-Goto 与 Polyakov 作用量](02_classical_bosonic_string.md)
3. [第三章：二维共形场论和 OPE 语言](03_worldsheet_cft.md)
4. [第四章：玻色弦的正则量子化](04_canonical_quantization_bosonic_string.md)
5. [第五章：路径积分、ghost 和 BRST 量子化](05_path_integral_brv_quantization.md)
6. [第六章：顶点算子和弦散射振幅](06_vertex_operators_and_string_scattering.md)
7. [第七章：紧化、T-duality 和 D-branes](07_compactification_t_duality_and_dbranes.md)

### 第二部分：超弦、异常和低能理论

8. [第八章：RNS 超弦和 GSO 投影](08_rns_superstrings_and_gso.md)
9. [第九章：Green-Schwarz 形式、type II strings 和 spacetime supersymmetry](09_green_schwarz_type_ii.md)
10. [第十章：heterotic strings、current algebra 和 anomaly cancellation](10_heterotic_strings_and_anomalies.md)
11. [第十一章：低能有效作用、supergravity 和 alpha-prime 修正](11_low_energy_effective_actions.md)
12. [第十二章：D-brane 有效理论、DBI 作用量和 Wess-Zumino 耦合](12_dbrane_effective_theory.md)

### 第三部分：紧化、对偶性和几何

13. [第十三章：Calabi-Yau 紧化、模空间和四维有效理论](13_calabi_yau_compactification.md)
14. [第十四章：S-duality、U-duality、M-theory 和 brane web](14_dualities_and_m_theory.md)
15. [第十五章：Riemann surfaces、moduli of curves 和高 genus 扰动论](15_riemann_surfaces_and_perturbation_theory.md)
16. [第十六章：topological strings、A/B model 和 mirror symmetry](16_topological_strings_and_mirror_symmetry.md)

### 第四部分：非微扰结构和应用

17. [第十七章：black branes、BPS states 和黑洞熵](17_black_branes_and_entropy.md)
18. [第十八章：AdS/CFT 的精确定式和基本检验](18_ads_cft.md)
19. [第十九章：flux compactifications、moduli stabilization 和 landscape 边界](19_flux_compactification_and_landscape.md)
20. [第二十章：string theory 与量子场论、几何和数论的接口](20_interfaces_with_qft_geometry_number_theory.md)

### 附录

- [附录 A：微分几何、纤维丛和曲率约定](A_geometry_forms_curvature.md)
- [附录 B：二维 CFT 公式表和 Virasoro 表示论](B_cft_formulae_virasoro.md)
- [附录 C：Lie algebras、Kac-Moody algebras 和 anomaly polynomials](C_lie_algebras_anomalies.md)
- [附录 D：Riemann surfaces、spin structures 和 theta functions](D_riemann_surfaces_spin_theta.md)
- [附录 E：supersymmetry、spinors 和 Clifford algebras](E_spinors_supersymmetry.md)

## 当前教材化状态

- 已固定主线：世界面 CFT、玻色弦量子化、BRST、超弦、D-branes、紧化、对偶性、AdS/CFT。
- 已建立写作约束、符号表、术语表、归一化表、依赖图、资料源、定理索引、习题索引和主线例题集。
- 已完成第 0 至 20 章和附录 A-E，并将第 6 至 20 章扩写为主线教材章或收束章。
- 当前版本已达到教材内容层面收口；后续不是扩张新主线，而是补例题、习题、局部证明、附录公式表和出版化排版。
