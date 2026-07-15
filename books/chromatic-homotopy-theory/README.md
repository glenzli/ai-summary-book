# Chromatic Homotopy Theory：高度、局部化与谱的色层

作者：Dr. Stochastic Parrot
状态：教材内容基本收口，出版级 theorem/page locator 与版式校对未完成
核查日期：2026-07-08
主资料源：Adams, Ravenel, Hovey-Strickland, Hopkins-Smith, Devinatz-Hopkins-Smith, Goerss-Hopkins-Miller, Devinatz-Hopkins, Rognes, Barthel-Beaudry, Burklund-Hahn-Levy-Schlank, Burklund-Schlank-Yuan, Hahn-Wilson

本书目标是写成一部严格的中文 Chromatic Homotopy Theory 教材，而不是主题导览。正文从稳定谱、Bousfield 局部化、复定向上同调和形式群开始，逐步进入 Brown-Peterson theory、Morava K-theory、Morava E-theory、有限谱的 type、nilpotence/periodicity/thick subcategory theorem、chromatic tower、$K(n)$-局部范畴、Morava stabilizer group descent、telescope 反例、redshift 和 2026 年前沿边界。

## 写作约束

本书写作约束见 [SKILL.md](SKILL.md)。后续扩写必须遵守：

- 定义先于直觉，非平凡命题带证明或明确的外部输入标记。
- 全书固定素数 $p$ 和稳定 infinity-范畴口径。
- $K(n)$、$E(n)$、$E_n$、$T(n)$、$L_n$、$L_n^f$ 和 $M_n$ 的符号必须与 [NOTATION.md](NOTATION.md) 一致。
- 同伦范畴中的三角、稳定 infinity-范畴中的 fiber/cofiber、谱序列和连续群上同调必须说明模型和收敛口径。
- 前沿结果必须能在 [FRONTIER_SOURCE_AUDIT_2026_07_08.md](FRONTIER_SOURCE_AUDIT_2026_07_08.md) 追溯，除非完成定位，不进入基础证明链。

## 总目录

### 第一部分：稳定谱与色层入口

1. [序章：范围、严格性标准和资料源](00_preface_and_scope.md)
2. [第一章：稳定谱、Bousfield 类与局部化](01_stable_spectra_localization_and_bousfield_classes.md)
3. [第二章：复定向、形式群律与 Brown-Peterson theory](02_complex_orientations_formal_groups_and_BP.md)
4. [第三章：Morava K/E theories 与高度](03_morava_K_E_and_height.md)

### 第二部分：有限谱、周期性与 chromatic tower

5. [第四章：有限谱的 type、nilpotence 与 periodicity](04_finite_spectra_type_and_periodicity.md)
6. [第五章：Chromatic localization、fracture 与 convergence](05_chromatic_localization_fracture_and_convergence.md)
7. [第六章：$K(n)$-局部范畴、Morava stabilizer group 与 descent](06_Kn_local_category_stabilizer_and_descent.md)

### 第三部分：高度二、前沿和局部几何

8. [第七章：Telescope、redshift 与 2026 研究边界](07_redshift_telescope_and_frontier_2026.md)
9. [第八章：Elliptic cohomology、tmf 与高度二几何](08_elliptic_tmf_and_height_two.md)
10. [第九章：Higher semiadditivity 与 transchromatic character](09_higher_semiadditivity_and_transchromatic_character.md)
11. [第十章：Chromatic splitting、Gross-Hopkins duality 与 Picard groups](10_chromatic_splitting_duality_and_picard.md)
12. [第十一章：Equivariant 和 motivic chromatic homotopy](11_equivariant_and_motivic_chromatic_homotopy.md)
13. [第十二章：计算工具、Adams--Novikov 与谱序列核验](12_computational_tools_adams_novikov_and_machine_checks.md)

### 附录

- [附录 A：形式群律和高度的逐项验算](A_formal_group_law_checks.md)
- [附录 B：谱序列、filtration 和收敛约定](B_spectral_sequence_conventions.md)
- [附录 C：Hopf algebroid、comodules 与 change of rings](C_hopf_algebroids_comodules_and_change_of_rings.md)
- [附录 D：资料源定理索引和 theorem locator](D_source_theorem_index.md)
- [附录 E：Bousfield lattice 与局部化失败模式](E_bousfield_lattice_and_localization_failure_modes.md)
- [附录 F：低高度计算样例](F_low_height_worked_examples.md)
- [附录 G：前沿预印本进入正文的验证协议](G_frontier_preprint_validation_protocol.md)
- [附录 H：稳定局部化与 $K(n)$-module 细节](H_stable_localization_and_module_field_details.md)
- [附录 I：$v_n$-periodicity 与 telescope 约定](I_vn_periodicity_and_telescope_conventions.md)
- [附录 J：Morava modules、stabilizer group 与 descent 细节](J_morava_modules_stabilizer_and_descent_details.md)
- [附录 K：Elliptic cohomology、tmf、level structure 与 power operation 约定](K_elliptic_tmf_level_and_power_operation_conventions.md)
- [附录 L：Gross-Hopkins duality 与 Picard group 约定表](L_gross_hopkins_picard_convention_table.md)
- [附录 M：Adams-Novikov 低阶样例与 hidden extension](M_adams_novikov_low_stem_examples.md)
- [附录 N：低高度与 fracture worked examples](N_low_height_and_fracture_worked_examples.md)
- [附录 O：综合习题与解题提示](O_problem_sets_and_solution_sketches.md)
- [附录 Q：低阶 stable stems 与 Adams-Novikov 校验表](Q_low_stem_and_anss_reference_tables.md)

## 当前内容索引

- [SKILL.md](SKILL.md)：本教材的写作约束。
- [NOTATION.md](NOTATION.md)：全书符号和小性约定。
- [SOURCES.md](SOURCES.md)：主要资料源清单和近期研究入口。
- [THEOREM_LEDGER.md](THEOREM_LEDGER.md)：内部证明、外部输入和研究边界账本。
- [DEPENDENCY_GRAPH.md](DEPENDENCY_GRAPH.md)：定义、证明和外部输入依赖图。
- [MATH_REVIEW.md](MATH_REVIEW.md)：数学审查记录和当前风险。
- [FRONTIER_SOURCE_AUDIT_2026_07_08.md](FRONTIER_SOURCE_AUDIT_2026_07_08.md)：近期前沿文献版本核查记录。
- [D_source_theorem_index.md](D_source_theorem_index.md)：外部输入定理索引和 locator 待办。
- [SOLUTIONS.md](SOLUTIONS.md)：综合习题和解题提示的标准入口，正文维护位置为附录 O。
- [PUBLICATION_CLOSURE_MATRIX.md](PUBLICATION_CLOSURE_MATRIX.md)：正式教材范围、内部完整性和细节完整性闭包矩阵。
- [P0_REFERENCE_LOCATORS_BATCH_1.md](P0_REFERENCE_LOCATORS_BATCH_1.md)：基础 chromatic 定理包；DHS/HS、Bousfield 分解、smash product、fracture 与 convergence 已有 theorem/section locator。
- [P0_REFERENCE_LOCATORS_BATCH_2.md](P0_REFERENCE_LOCATORS_BATCH_2.md)：Morava descent、tmf、Gross-Hopkins、Picard 与计算的第二批 bibliographic locator。
- [P1_REFERENCE_LOCATORS_FRONTIER.md](P1_REFERENCE_LOCATORS_FRONTIER.md)：前沿、半加性、equivariant/motivic 接口定理包的 content-level locator。
- [INTERNAL_CHAPTER_COMPLETENESS_AUDIT.md](INTERNAL_CHAPTER_COMPLETENESS_AUDIT.md)：逐章正文态、接口正文态和剩余缺口审计。

## 当前判定

本目录目前是教材内容基本收口稿，不是 camera-ready 出版版本。当前已完成的是：

1. 固定全书口径、符号和前沿准入规则；
2. 写入稳定谱、复定向、形式群、Morava K/E、有限谱 type、chromatic tower、$K(n)$-local descent、redshift/telescope、semiadditivity、splitting/duality/Picard、equivariant/motivic 和计算工具的第一版正文；
3. 建立 theorem ledger、frontier audit、Hopf algebroid 附录、失败模式附录、$v_n$/telescope 约定、Morava descent 细节、tmf/GH/Picard convention 和前沿准入协议，防止近期结果被误用。

后续主要工作是出版/维护层：theorem/section/page 级 locator、完整 ANSS 大表、低高度 tmf/Picard 案例加厚、版式与交叉引用统一。
