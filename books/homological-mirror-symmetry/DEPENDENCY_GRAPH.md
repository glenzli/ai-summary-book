# 依赖图

本文件记录当前章节之间的定义和证明依赖，防止后续扩写引用尚未建立的结构。

## 基础层

- `NOTATION.md` 固定宇宙、基域、cohomological grading、shift、dg/$A_\infty$ 记号。
- `00_preface_and_scope.md` 固定 HMS 的增强范畴形态和外部输入策略。

## 增强范畴层

- `01_dg_and_a_infinity_categories.md`
  - 依赖：`NOTATION.md`。
  - 提供：dg category、$H^0$、$A_\infty$ category、quasi-equivalence、modules、Yoneda、Morita equivalence、twisted complexes、pretriangulated envelope。
  - 被依赖：B-side enhancement、Fukaya category、HMS 断言和生成元法。

## B-side 层

- `02_derived_categories_and_b_side_enhancements.md`
  - 依赖：`01_dg_and_a_infinity_categories.md`。
  - 提供：derived category、perfect complex、h-injective
    $\operatorname{Perf}_{\mathrm{dg}}(X)$ model、
    $\mathrm D^b\operatorname{Coh}(X)$ 的三角影子、enhanced
    Fourier--Mukai transform、affine matrix-factorization dg model。
  - 被依赖：HMS 断言模板、后续标准例子、Landau-Ginzburg 章节。

## A-side 层

- `03_symplectic_lagrangian_and_floer_foundations.md`
  - 依赖：`01_dg_and_a_infinity_categories.md` 的复形语言。
  - 提供：symplectic manifold、Lagrangian、exactness、brane data、Floer cochains、Floer differential、continuation invariance 边界。
  - 被依赖：`04_holomorphic_polygons_and_fukaya_categories.md`。

- `04_holomorphic_polygons_and_fukaya_categories.md`
  - 依赖：`01_dg_and_a_infinity_categories.md`、
    `03_symplectic_lagrangian_and_floer_foundations.md`、附录 B 与 E。
  - 提供：compact exact Hamiltonian-chord complexes、coherent polygon
    data、orientation/local-system operations、精确 $A_\infty$ 方程、
    cohomological units/strictification 边界、split-closed derived Fukaya
    category、wrapped 边界。
  - 被依赖：后续 wrapped/stopped/Fukaya-Seidel 章节与 HMS 例子。

- `05_obstruction_bounding_cochains_and_novikov_coefficients.md`
  - 依赖：`01_dg_and_a_infinity_categories.md`、`04_holomorphic_polygons_and_fukaya_categories.md`。
  - 提供：Novikov field、curved $A_\infty$ structures、bounding cochains、Maurer-Cartan deformation、potential values。
  - 被依赖：toric/LG、quantum cohomology 分块、matrix factorization 章节。

- `06_liouville_sectors_and_wrapped_fukaya_categories.md`
  - 依赖：`03_symplectic_lagrangian_and_floer_foundations.md`、`04_holomorphic_polygons_and_fukaya_categories.md`。
  - 提供：Liouville manifolds/domains/sectors、exact conical branes、
    cofinal Hamiltonian chords、telescope wrapped complexes、wrapped
    analytic/categorical package、cocore generation。
  - 被依赖：stops、sectorial descent、microlocal sheaves、wrapped HMS。

- `07_stops_partially_wrapped_categories_and_localization.md`
  - 依赖：`06_liouville_sectors_and_wrapped_fukaya_categories.md`。
  - 提供：stops、partially wrapped categories、linking disks、stop removal、Viterbo/Orlov functoriality。
  - 被依赖：functorial HMS、microlocal sheaf models。

## HMS 断言层

- `08_hms_statement_enhancements_and_invariants.md`
  - 依赖：`01_dg_and_a_infinity_categories.md`、`02_derived_categories_and_b_side_enhancements.md`、`03_symplectic_lagrangian_and_floer_foundations.md`。
  - 提供：HMS 数据包、raw/triangulated/quasi-equivalence/Morita 层级、
    生成元比较原则、不变量检查和陈述模板。
  - 被依赖：所有具体 HMS 例子章节。

## 标准例子层

- `09_elliptic_curves_complex_tori_and_syz_first_model.md`
  - 依赖：`08_hms_statement_enhancements_and_invariants.md`。
  - 提供：椭圆曲线 HMS 数据包、斜率圆/稳定丛字典、theta 乘法外部输入。

- `10_toric_fano_landau_ginzburg_potentials_and_jacobian_rings.md`
  - 依赖：`05_obstruction_bounding_cochains_and_novikov_coefficients.md`、`08_hms_statement_enhancements_and_invariants.md`。
  - 提供：toric Fano/LG potential、disk potential、Jacobian ring、Abouzaid toric HMS 入口。

- `11_fukaya_seidel_categories_and_picard_lefschetz_theory.md`
  - 依赖：`04_holomorphic_polygons_and_fukaya_categories.md`、`06_liouville_sectors_and_wrapped_fukaya_categories.md`。
  - 提供：Lefschetz fibration、vanishing cycles、Fukaya-Seidel categories、directed HMS strategy。

- `12_k3_quartics_and_calabi_yau_hypersurfaces.md`
  - 依赖：`08_hms_statement_enhancements_and_invariants.md`、`11_fukaya_seidel_categories_and_picard_lefschetz_theory.md`.
  - 提供：Calabi-Yau hypersurface HMS 数据包、Seidel/Sheridan/Batyrev 外部输入边界。

- `13_pairs_of_pants_tropical_degeneration_and_hypersurfaces.md`
  - 依赖：`06_liouville_sectors_and_wrapped_fukaya_categories.md`、`07_stops_partially_wrapped_categories_and_localization.md`、`12_k3_quartics_and_calabi_yau_hypersurfaces.md`。
  - 提供：pair-of-pants、tropical degeneration、hypersurfaces in tori、local-to-global HMS 模板。

## 生成、descent 与 sheaf 层

- `14_split_generation_open_closed_and_abouzaid_criterion.md`
  - 依赖：`01_dg_and_a_infinity_categories.md`、
    `06_liouville_sectors_and_wrapped_fukaya_categories.md`、
    `08_hms_statement_enhancements_and_invariants.md`。
  - 提供：degree-shifted wrapped open-closed、closed-open、Abouzaid
    Theorem 1.1 的 global-unit criterion、Morita HMS 推论、closed-open
    idempotent projector 分块与 curvature-value 边界。

- `15_sectorial_descent_for_wrapped_fukaya_categories.md`
  - 依赖：`06_liouville_sectors_and_wrapped_fukaya_categories.md`、`14_split_generation_open_closed_and_abouzaid_criterion.md`。
  - 提供：sectorial descent、wrapped Kunneth、HMS descent diagram。

- `16_nadler_zaslow_microlocal_sheaves_and_cotangent_bundles.md`
  - 依赖：`02_derived_categories_and_b_side_enhancements.md`、`06_liouville_sectors_and_wrapped_fukaya_categories.md`。
  - 提供：constructible sheaves、microsupport、Nadler-Zaslow、microlocal sheaf model。

- `17_stop_removal_viterbo_functors_and_functorial_hms.md`
  - 依赖：`07_stops_partially_wrapped_categories_and_localization.md`、`15_sectorial_descent_for_wrapped_fukaya_categories.md`。
  - 提供：functorial HMS、Orlov/Viterbo functors、strict functorial square。

- `18_hochschild_closed_open_and_categorical_enumerative_checks.md`
  - 依赖：`14_split_generation_open_closed_and_abouzaid_criterion.md`、`02_derived_categories_and_b_side_enhancements.md`。
  - 提供：Hochschild invariants、HKR、categorical enumerative checks。

## 研究边界层

- `19_rabinowitz_fukaya_singularities_and_matrix_factorizations.md`
  - 依赖：`02_derived_categories_and_b_side_enhancements.md`、`06_liouville_sectors_and_wrapped_fukaya_categories.md`。
  - 提供：Rabinowitz Fukaya categories、Milnor fibers、matrix factorizations。

- `20_functorial_wall_crossing_bps_and_2026_research_boundary.md`
  - 依赖：`17_stop_removal_viterbo_functors_and_functorial_hms.md`、`18_hochschild_closed_open_and_categorical_enumerative_checks.md`。
  - 提供：functorial HMS、wall-crossing、BPS categories 和研究边界规则。

## 计算与解答层

- `I_low_arity_a_infinity_and_curvature_calculations.md`
  - 依赖：`01_dg_and_a_infinity_categories.md`、`05_obstruction_bounding_cochains_and_novikov_coefficients.md`。
  - 提供：低阶 $A_\infty$ 方程、curvature、Maurer-Cartan 变形的展开计算。

- `J_elliptic_toric_and_fukaya_seidel_worked_models.md`
  - 依赖：`09_elliptic_curves_complex_tori_and_syz_first_model.md`、`10_toric_fano_landau_ginzburg_potentials_and_jacobian_rings.md`、`11_fukaya_seidel_categories_and_picard_lefschetz_theory.md`。
  - 提供：斜率交点、Jacobian ring、critical values、Kronecker quiver、pair-of-pants 模型计算。

- `K_generation_descent_and_localization_templates.md`
  - 依赖：`14_split_generation_open_closed_and_abouzaid_criterion.md`、`15_sectorial_descent_for_wrapped_fukaya_categories.md`、`07_stops_partially_wrapped_categories_and_localization.md`。
  - 提供：生成元比较、open-closed 生成、sectorial descent、stop removal 的证明模板。

- `SOLUTIONS.md`
  - 依赖：主体章节练习。
  - 提供：第零章至第二十章练习解答与提示。

## 在线版范围外的出版工作

- Theorem locator 已达在线闭合；印刷版还需执行逐条版本、页码与精确编号校勘。
- 标准例子的核心计算已存在；印刷版可继续扩充逐项高阶
  $A_\infty$ 运算表。
- 当前交叉引用由文件名与正文编号承担；独立的印刷版 label ledger 属于
  排版阶段工作。
