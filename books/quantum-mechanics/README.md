# 量子力学：Hilbert 空间、谱与测量

作者：Dr. Stochastic Parrot
状态：教材内容与阅读排版收口稿（内容范围、证明闭包、外部输入边界和阅读索引固定）
主资料源：见 [SOURCES.md](SOURCES.md)；核心线索包括 von Neumann、Reed-Simon、Hall、Teschl、Sakurai、Shankar、Ballentine、Nielsen-Chuang、Davies 等。

本书目标是写成一部数学上严格、物理上可计算的量子力学教材。正文从 Hilbert 空间和量子态开始，逐步进入可观测量、谱定理、Schrodinger 演化、典型模型、对称性、扰动、散射、多体系统、密度算子、现代测量理论和量子信息接口。

## 写作约束

本书写作约束见 [SKILL.md](SKILL.md)。后续校订必须遵守：

- 定义先于直觉，命题带证明或明确的外部输入标记。
- 无界算子必须说明定义域或引用外部输入定理。
- 态、射线、密度矩阵、谱测度和测量仪器不得混用。
- 计算题必须给出可复核公式，不以物理直觉替代推导。
- 资料源必须能在 [SOURCES.md](SOURCES.md) 中追溯。

统一阅读索引见 [BOOK_INDEX.md](BOOK_INDEX.md)。符号约定见 [NOTATION.md](NOTATION.md)，术语索引见 [TERM_INDEX.md](TERM_INDEX.md)，数学审查记录见 [MATH_REVIEW.md](MATH_REVIEW.md)。
习题提示见 [HINTS.md](HINTS.md)，练习答案见 [SOLUTIONS.md](SOLUTIONS.md)。跨章节综合题见 [COMPREHENSIVE_EXERCISES.md](COMPREHENSIVE_EXERCISES.md)，答案见 [COMPREHENSIVE_SOLUTIONS.md](COMPREHENSIVE_SOLUTIONS.md)。
外部输入依赖图见 [THEOREM_DEPENDENCIES.md](THEOREM_DEPENDENCIES.md)，逐章来源注释见 [CHAPTER_SOURCE_NOTES.md](CHAPTER_SOURCE_NOTES.md)。

## 总目录

### 第一部分：数学结构与基本公设

0. [序章：范围、单位制和严格性标准](00_preface_scope_and_units.md)
1. [第一章：Hilbert 空间、态与射线](01_hilbert_spaces_states_and_rays.md)
2. [第二章：有界算子、投影与谱分解](02_bounded_operators_projections_and_spectral_decomposition.md)
3. [第三章：无界可观测量与谱定理](03_unbounded_observables_and_spectral_theorem.md)
4. [第四章：量子公设、Born 规则与态更新](04_postulates_born_rule_and_state_update.md)
5. [第五章：时间演化、Stone 定理与 Schrodinger 方程](05_time_evolution_stone_theorem_and_schrodinger_equation.md)

### 第二部分：基本模型

6. [第六章：一维系统、势垒与束缚态](06_one_dimensional_systems_barriers_and_bound_states.md)
7. [第七章：谐振子、升降算符与 Hermite 基](07_harmonic_oscillator_ladder_operators_and_hermite_basis.md)
8. [第八章：张量积、复合系统与纠缠](08_tensor_products_composite_systems_and_entanglement.md)
9. [第九章：角动量、旋转与自旋](09_angular_momentum_rotations_and_spin.md)
10. [第十章：对称性、守恒律与 Wigner 定理](10_symmetries_conservation_laws_and_wigner_theorem.md)
11. [第十一章：正则对易关系与表示](11_canonical_commutation_relations_and_representations.md)

### 第三部分：近似、散射与多体

12. [第十二章：定态与含时扰动理论](12_stationary_and_time_dependent_perturbation_theory.md)
13. [第十三章：变分法、WKB 与半经典近似](13_variational_wkb_and_semiclassical_methods.md)
14. [第十四章：散射理论、Born 近似与截面](14_scattering_theory_born_approximation_and_cross_sections.md)
15. [第十五章：相同粒子、对称化与 Fock 空间](15_identical_particles_symmetrization_and_fock_space.md)
16. [第十六章：绝热定理、Berry 相位与有效动力学](16_adiabatic_theorem_berry_phase_and_effective_dynamics.md)

### 第四部分：现代测量、信息与边界

17. [第十七章：密度算子、偏迹与开放系统](17_density_operators_partial_trace_and_open_systems.md)
18. [第十八章：POVM、量子仪器与 Kraus 表示](18_povms_quantum_instruments_and_kraus_representations.md)
19. [第十九章：量子信息、熵与信道](19_quantum_information_entropy_and_channels.md)
20. [第二十章：路径积分与传播子](20_path_integrals_and_propagators.md)
21. [第二十一章：相对论一粒子方程与适用边界](21_relativistic_one_particle_equations_and_limits.md)

### 第五部分：核心模型补遗与规范结构

22. [第二十二章：中心势、氢原子与球谐函数](22_central_potentials_hydrogen_and_spherical_harmonics.md)
23. [第二十三章：电磁耦合、规范变换与磁场](23_electromagnetic_coupling_gauge_and_magnetic_fields.md)

### 第六部分：基础定理与精细结构

24. [第二十四章：不确定性、Ehrenfest 定理与概率流](24_uncertainty_ehrenfest_and_probability_current.md)
25. [第二十五章：相互作用图像、Dyson 展开与跃迁率](25_interaction_picture_dyson_series_and_transition_rates.md)
26. [第二十六章：角动量耦合、Clebsch-Gordan 系数与选择定则](26_angular_momentum_addition_clebsch_gordan_and_selection_rules.md)
27. [第二十七章：标准精算例题：Gaussian 波包、自旋进动与 Rabi 振荡](27_worked_examples_gaussian_spin_and_rabi_oscillations.md)

### 附录

- [附录 A：泛函分析背景](A_functional_analysis_background.md)
- [附录 B：Fourier 变换、分布与常用积分](B_fourier_transform_distributions_and_integrals.md)
- [附录 C：矩阵、Lie 代数与群表示速查](C_matrices_lie_algebras_and_representation_reference.md)
- [附录 D：外部输入定理索引](D_external_theorem_index.md)
- [附录 E：内部完整性矩阵](E_internal_completeness_matrix.md)
- [附录 F：范围边界与外部输入政策](F_scope_boundary_and_external_input_policy.md)
- [附录 G：正式教材化审查标准](G_formal_textbookization_audit.md)
- [附录 H：教材内容收口审查](H_content_closure_audit.md)

## 当前范围

当前版本按连续教材组织：每章由具体物理问题或可计算模型进入，把所需
前置概念融入叙述，在定义、命题、证明、近似边界和例子之间建立过渡，
并以内容特定的段落收束到练习。提示手册与答案手册覆盖全部章末练习和
综合题；统一索引、压缩术语表和外部输入索引已接入。

本书不把完整泛函分析、偏微分方程、表示论、散射完备性或量子场论放入内部闭包。这些主题在正文中以“外部输入定理”或“边界说明”出现，并在索引文件中记录用途。

## 审稿辅助文件

- [TERM_INDEX.md](TERM_INDEX.md)：压缩核心术语索引。
- [BOOK_INDEX.md](BOOK_INDEX.md)：统一阅读索引、练习答案入口和外部输入入口。
- [HINTS.md](HINTS.md)：章末练习、附录练习和综合题提示手册。
- [CHAPTER_SOURCE_NOTES.md](CHAPTER_SOURCE_NOTES.md)：逐章资料源、内部闭包和外部输入边界。
- [THEOREM_DEPENDENCIES.md](THEOREM_DEPENDENCIES.md)：外部输入定理依赖图。
- [E_internal_completeness_matrix.md](E_internal_completeness_matrix.md)：内部证明闭包与外部输入闭包矩阵。
- [MATH_REVIEW.md](MATH_REVIEW.md)：数学审查清单和当前风险。
- [G_formal_textbookization_audit.md](G_formal_textbookization_audit.md)：正式教材化审查标准。
- [H_content_closure_audit.md](H_content_closure_audit.md)：教材内容收口审查。
- [validate.py](validate.py)：本目录的内部链接、章节结构、占位标记和习题答案覆盖检查。
