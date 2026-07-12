# 定义、证明和外部输入依赖图

本文档记录《Prismatic / p-adic Hodge Theory》的逻辑依赖。它不新增数学定理。

## 核心定义链

1. 固定素数 $p$、Koszul-model derived completion、$D(A)$；区分 ordinary、derived 与 completed tensor。
2. $\delta$-环 $\Rightarrow$ Frobenius lift $\phi_A(x)=x^p+p\delta_A(x)$。
3. Cartier divisor ideal 与 derived $(p,I)$-completion。
4. Prism $(A,I)$：$\delta$-环 + Cartier divisor + derived completeness + $p\in I+\phi(I)A$。
5. Bounded prism：$A/I$ 的 $p^\infty$-torsion 有界。
6. Prism-map rigidity $J=IB$ 与 relative prismatic site $(X/A)_\Delta$；covers 使用 $(p,IB)$-complete faithful flatness。
7. $R\Gamma_\Delta(X/A)$、Frobenius-semilinear map、ordinary linearization 与 completed twist $C^{(1)}$。
8. Hodge--Tate plain derived reduction、completed Frobenius-twisted de Rham/crystalline base change、finite-level etale fixed points。
9. $A_{\inf}$ and Breuil-Kisin specializations。
10. Prismatic $F$-crystals and crystalline Galois lattices。

## 不得倒用的外部输入

- Perfect prism 等价于 perfectoid ring 不能用于证明 prism 定义本身，只能用于例子和 special case。
- Prismatic comparison theorem 不能用于定义 $R\Gamma_\Delta$，只能用于识别其 specialization。
- BMS integral comparison 不能用于证明 Bhatt-Scholze prismatic comparison；本书只说明后者如何回收前者。
- Classical Fontaine admissibility 不能替代 prismatic $F$-crystal 的定义；二者通过外部输入定理连接。
- Prismatization 和 $F$-gauges 不能用于基础章节证明，只能作为后续重解释或研究边界。

## 正文依赖图

- `00_preface_and_scope.md`
  - 依赖：全局资料源。
  - 输出：本书范围、严格性标准和外部输入政策。
- `01_delta_rings_witt_vectors_and_perfectoid_background.md`
  - 依赖：固定素数、环论、Witt vectors 背景。
  - 输出：$\delta$-环、Frobenius lift、distinguished element、perfectoid interface。
- `02_prisms_and_prismatic_sites.md`
  - 依赖：第一章全部定义及附录 A 的 Koszul completion/complete flatness。
  - 输出：prism、bounded prism、ideal rigidity、relative prismatic site 与 semilinear Frobenius typing。
- `03_prismatic_cohomology_comparisons.md`
  - 依赖：第二章 prismatic site。
  - 输出：$R\Gamma_\Delta$、completed/twisted comparison interface、base-change Tor boundary。
- `04_fontaine_period_rings_and_classical_p_adic_hodge.md`
  - 依赖：classical Galois representations。
  - 输出：$B$-admissibility and comparison interface。
- `05_a_inf_breuil_kisin_and_bms_integral_theory.md`
  - 依赖：perfectoid and prism examples。
  - 输出：$A_{\inf}$ and Breuil-Kisin specialization。
- `06_prismatic_f_crystals_and_galois_representations.md`
  - 依赖：prismatic site, Frobenius, crystals。
  - 输出：prismatic $F$-crystal theorem interface。
- `07_nygaard_syntomic_and_tate_twists.md`
  - 依赖：Frobenius and prismatic cohomology。
  - 输出：Nygaard/syntomic/Tate-twist framework。
- `08_prismatization_f_gauges_and_frontier.md`
  - 依赖：all previous chapters。
  - 输出：frontier map; no foundation-level theorem promoted.
- `09_hodge_tate_de_rham_and_conjugate_filtration.md`
  - 依赖：第三章 Hodge-Tate/de Rham comparison。
  - 输出：conjugate filtration、Hodge filtration 和不可混用判别。
- `10_crystalline_de_rham_witt_and_q_de_rham.md`
  - 依赖：第二、三、五章和附录 B。
  - 输出：crystalline、de Rham-Witt、$q$-de Rham specialization 分层。
- `11_etale_comparison_frobenius_fixed_and_syntomic_tower.md`
  - 依赖：第三、七章和附录 F。
  - 输出：derived Frobenius fixed points、syntomic tower、cup product 要求。
- `12_breuil_kisin_bkf_modules_and_lattices.md`
  - 依赖：第四至六章及命题 5.16 的 ordinary Tor boundary。
  - 输出：BK/BKF exact finiteness classes、torsion thresholds 与 evaluation/descent lattice boundary。
- `13_coefficients_hodge_tate_crystals_and_nonabelian_boundary.md`
  - 依赖：第六、七、九章。
  - 输出：coefficients and non-abelian Hodge boundary。
- `14_artin_stacks_shimura_and_arithmetic_applications.md`
  - 依赖：第六至八、十二章。
  - 输出：Artin stacks、Shimura、Brauer、finite flat groups 的应用边界。
- `15_closure_failure_modes_and_open_problems.md`
  - 依赖：全书。
  - 输出：定义闭合判定、错误模式和开放问题目录。

## 元文档依赖

- `C_comparison_hypotheses_and_structure_tables.md` 依赖第三至十二章，不新增定理。
- `D_theorem_locator_index.md` 依赖 `SOURCES.md` 和前沿核查记录，不新增定理。
- `E_label_ledger.md` 依赖所有章节编号，不新增定理。
- `F_nygaard_tate_twist_crosswalk.md` 依赖第七、九、十一章，不新增定理。
- `G_formal_schemes_sites_and_derived_global_sections.md` 为第二、三章提供 formal scheme/site/derived global section 基础。
- `H_delta_prism_detailed_proofs.md` 为第一、二章提供 $\delta$-环和 prism 条件的逐项证明。
- `I_crystals_descent_and_vector_bundles.md` 为第六、十三章提供 crystals/descent/vector bundles 基础。
- `J_linear_algebra_of_periods_and_lattices.md` 为第四、十二章提供 semilinear Frobenius、filtered vector spaces 和 lattices 基础。
- `K_worked_examples_and_local_models.md` 为第九至十二章提供局部计算模型。
- `SOLUTIONS.md`、`TERM_INDEX.md` 和 `INTERNAL_COMPLETENESS_AUDIT.md` 分别记录教学解答、术语定位和内部完整性判定。
- `PUBLICATION_CLOSURE_MATRIX.md` 和 `FORMAL_TEXTBOOK_EXPANSION_AUDIT.md` 只记录状态。
