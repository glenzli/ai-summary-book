# 数学审查记录

核查日期：2026-06-30。

## 总体风险

- Operad theory 横跨集合、拓扑空间、链复形、模型范畴和 infinity-范畴；不同语境的“operad”不能互相替换。
- 基础定义中是否允许 nullary operations 会影响 unital algebra、Com-algebra 和自由对象；本书默认允许 arity $0$。
- 对称群作用存在左右作用约定差异；本书基础定义优先使用有限集和双射群胚，arity 公式只作为派生写法。
- Infinity-operad 至少有 dendroidal set、simplicial operad nerve、Lurie operadic fibration 等模型；模型比较应作为外部输入定理处理。
- 2026 年预印本需要后续版本核查，不应直接吸收入核心定理链。

## 当前文件审查

### `00_preface_and_scope.md`

- 已声明本书范围、严格性标准和研究边界口径。
- 已建立附录 D 的外部输入索引；后续需要把具体来源精确到定理编号、页码或 arXiv 版本。

### `SECOND_PASS_STRICTIFICATION_PLAN.md`

- 已建立第二轮严格化路线图，明确从第一轮草稿进入可审校教材形态的通过标准。
- 已把后续工作分为基础定义链、线性符号系统、模型范畴假设、例子计算和前沿版本核查五类。
- 该文件不新增定理；它约束后续审校必须补齐的证明、符号、来源和模型假设。

### `FRONTIER_SOURCE_AUDIT_2026_06_29.md`

- 已把近期 arXiv 条目的使用状态固定为研究边界。
- 已记录每个条目对应的本书逻辑位置和进入正文前必须补齐的信息。
- 该文件不证明任何前沿定理；若未来引用其中结果，仍需精确版本、定理编号和模型约定。

### `FRONTIER_SOURCE_AUDIT_2026_06_30.md`

- 已按 2026-06-30 的官方 arXiv 页面重新核查近期条目的版本状态。
- 已把 infinity-operadic Koszul duality、operadic categories、relative dendroidal Rezk nerve 和 Fukaya categories 高阶 operadic structure 分别列为需要专章处理的前沿方向。
- 已明确这些条目目前只能作为研究边界或已登记的边界 locator，不得进入基础章节的证明链。

### `01_symmetric_sequences_and_operads.md`

- 已使用有限集上的对称序列定义，避免早期左右作用歧义。
- 已把代入乘积的幺半相干性从证明草图加强为可复合有限集映射 $S\to U\to T$ 的共同群胚表示。
- 已给出代入乘积、单位对称序列、operad、endomorphism operad、operad 代数、Ass 和 Com 的基础定义。
- 当前证明主要覆盖集合值 operad；后续推广到 $\mathbf{Mod}_R$ 和 $\mathbf{Ch}_R$ 时必须重新检查 coinvariants、直和、张量分配和符号规则。

### `02_operad_algebras_free_algebras_and_monads.md`

- 已给出 $\mathcal O$-代数同态、限制标量、自由代数、自由-遗忘伴随和 monad 识别。
- 已补充自由代数的有限集 coend 公式
  $$
  T_{\mathcal O}(A)\cong\int^{S\in\mathbf B_{\mathcal U}}\mathcal O(S)\times A^S
  $$
  并把自由代数动作改写为无坐标纤维复合；空代表集 $T_s=\varnothing$ 仍保留外层槽，不再被误写成“空块”。
- Finitary monad 证明已改为有限集因子化和 coend/filtered colimit 相容论证；最终版仍可继续压缩商关系细节。

### `03_nonsymmetric_operads_partial_compositions_and_trees.md`

- 已给出非对称 operad、偏复合公理和树收缩顺序无关性。
- 第十六章已区分 Moerdijk-Weiss 树范畴 $\Omega$；附录 H 已补充 planar tree、leaf-labelled tree 与非平面 rooted tree 的系统对照表。

### `04_free_operads_generators_and_relations.md`

- 已给出自由非对称/对称 operad 的装饰树公式、泛性质和 operadic congruence。
- 当前树定义服务于基础自由构造；第十六章已另行给出 $\Omega$ 中的 rooted tree 定义。

### `05_colored_operads_multicategories_and_typed_algebras.md`

- 已用有限集输入颜色函数定义 colored symmetric sequence、colored substitution product 和 colored operad。
- 已证明 colored endomorphism operad 和 symmetric multicategory 等价口径。
- 已接入附录 K；enriched colored operad 需要把集合值 Hom 换成指定对称幺半范畴中的对象，并另行检查 admissibility。

### `06_linear_operads_schur_functors_and_classical_examples.md`

- 已给出 $R$-模值代入乘积、coinvariants arity 公式、Schur functor 和 Ass/Com/Lie/Pois 例子。
- Lie operad 在一般底环上的反对称/alternating 约定已标出风险；进入特征 $2$ 或一般环时不能省略。
- 第九章已补充链复形符号和 Koszul sign rule；附录 E 已集中记录悬挂、张量、Hom differential 和 Hochschild 基础符号。

### `07_props_properads_and_wheeled_variants.md`

- 已定义 PROP、$\mathbb S$-双模、endomorphism PROP、双代数 PROP、properad 和 wheeled contraction。
- Properad 到自由 PROP 的构造已标为外部输入定理；完整图群胚商和相干性证明尚未展开。
- Wheeled endomorphism 例子已声明有限生成投射/trace 条件，避免在无限维对象上误用 trace。

### `08_quadratic_operads_and_koszul_duality.md`

- 已明确本章采用 reduced 非含单位口径，避免 arity $0$ 单位混入二次齐次理论。
- 已把自由 operad 完整代入的权重改为 $r+\sum_t s_t$；外层 arity 为 $n$ 且内层同权 $s$ 时为 $r+ns$，并用二槽例子区分完整代入与一次偏复合。
- 已定义二次数据、二次对偶和 Koszul 性；Ass/Com/Lie 的 Koszul 性只作为外部输入定理。
- 第九章已补充 dg cooperad、twisting morphism 和基本符号规则；附录 E 已给出 operadic suspension 和 suspended brace 的基础算法，完整 $L_\infty$ 展开符号仍需最终版逐项核对。

### `09_bar_cobar_constructions_and_twisting_morphisms.md`

- 已固定同调分次和 Koszul sign rule，并定义 dg-operad、dg-cooperad、cofree conilpotent cooperad、twisting morphism、bar/cobar 构造。
- 已把 $M\circ_{(1)}N$ 定义为 $M\circ(I_k\oplus N)$ 对 $N$ 的一次齐次分量，并由 $\mathcal C\cong I_k\oplus\overline{\mathcal C}$ 构造 $\Delta_{(1)}$；convolution pre-Lie 与 Maurer--Cartan 方程现统一在 $\operatorname{Hom}_{\mathbb S}(\overline{\mathcal C},\overline{\mathcal P})$ 中作类型检查。
- 已证明 bar-cobar 泛性质；Koszul twisting morphism 的 quasi-isomorphism 判别标为外部输入定理。
- 已接入附录 I 的 convolution Lie algebra、twisted composite product、Koszul complex 和权重滤过约定；进入具体 $A_\infty/L_\infty$ 章节时仍需与定义 E.18--定义 E.23 和定义 J.1--外部输入定理 J.19 的 signs 对齐。

### `10_a_infinity_l_infinity_and_e_n_operads.md`

- 已以 bar-cobar 定义为主定义给出 $A_\infty$、$L_\infty$ 和 $C_\infty$，手写恒等式只作为展开说明。
- 已区分 $E_n$-operad 与 Lurie-style infinity-operad。
- $H_\*(\mathcal C_n)\cong\operatorname{Pois}_n$、May recognition principle 和形式性均标为外部输入定理。
- 已接入附录 L 的 suspended coderivation 口径和 $E_n$ 模型层级边界；具体 unsuspended signs 仍需逐项校验。

### `11_gerstenhaber_bv_and_deligne_conjecture.md`

- 已切换到 Hochschild 传统上同调分次，并明确 bracket 次数为 $-1$。
- Deligne 猜想、framed $E_2$ 同调为 BV operad 均标为外部输入定理。
- BV bracket 的符号采用常见约定；后续若与链级 operadic suspension 连接，需要重新核对符号。

### `12_brace_operad_and_hochschild_cochains.md`

- 已给出未分次结合代数上的 Hochschild differential、cup product、insertion、brace operations 和 brace operad 路线。
- 已新增 dg 情形的 suspended Hochschild brace 约定，并与附录 E 的符号算法对齐。
- Brace operad 与 $E_2$ 链模型弱等价标为外部输入定理。
- Graded Hochschild cochains 的基础 insertion/cup signs 与 suspended brace signs 已有统一算法；最终版仍需与 Berger-Fresse/McClure-Smith 具体链模型逐项核对。

### `13_homotopy_transfer_and_minimal_models.md`

- 已定义 contraction、homological perturbation lemma、同伦转移定理、$A_\infty/L_\infty$ 树公式、minimal model 和 formality。
- Homological perturbation lemma、完整转移定理、minimal model 存在唯一性均标为外部输入定理；其中 Markl strongly homotopy transfer existence 已定位到 MHT-1--MHT-8。
- 已接入附录 J 的 normalized contraction、$A_\infty$ 平面二叉树递归、$L_\infty$ shuffle 反对称化和低阶恒等式检查。
- $m_3$ 和 Massey products 的关系只作说明；完整选择依赖仍需后续例子章节展开。

### `14_operads_in_model_categories.md`

- 已定义对称幺半模型范畴、monoid axiom、对称序列 projective 模型结构、operad transferred 模型结构、admissible operad、rectification、Boardman-Vogt resolution 和 derived mapping space。
- Berger--Moerdijk 转移定理、代数范畴 admissibility 和 rectification criterion 均按各自假设标为外部输入；完整 $W$-construction 因缺统一 theorem locator 保留为外部边界 14.31。
- 已明确 positive characteristic / general ring 中 $E_\infty$ 与严格 commutative dg algebra 不能无条件 rectification。
- 已接入附录 G 的模型结构假设检查表；具体 transferred model structure 的假设仍需在最终版逐例核对到文献定理编号。

### `15_simplicial_and_topological_operads.md`

- 已定义 $\mathbf{sSet}$、$\mathbf{Top}$、simplicial operad、topological operad、well-pointed 与 $\Sigma$-free 条件、little cubes operad、chains on spaces 和 unary colored simplicial operads。
- Kan--Quillen 模型结构、$\mathbf{sSet}$ 与 $\mathbf{Top}$ 的底范畴 Quillen equivalence、Eilenberg--Zilber 相干性和 May recognition principle 保持外部输入；simplicial/topological operad 模型结构已分别对齐 BM-1，operad-level realization--Sing Quillen equivalence 因缺 change-of-base locator 降为外部边界 15.18。
- 已说明 $C_\*(\mathcal C_d;k)$ 依赖 lax monoidal chains，形式性结论不能由取 chains 自动推出。
- 第十六章已把 simplicial categories 作为线性树入口，并定义 Moerdijk-Weiss 树范畴 $\Omega$ 和 dendroidal nerve。

### `16_dendroidal_sets_and_tree_category.md`

- 已从本章开头固定 $\mathcal U$-小 rooted-tree 代表集，并把 $\Omega$ 定义为该固定骨架上的 $\mathcal U$-小树范畴；dendroidal presheaf、representable 和 category-of-elements colimit 的 universe 层级已逐项写明。
- 已定义 rooted tree、inner/outer edges、单位树、corolla、线性树、自由 colored operad $\Omega(T)$、树范畴 $\Omega$、dendroidal set、representable、dendroidal nerve、Segal core、faces、degeneracies、boundaries 和 inner horns。
- 已证明 dendroidal nerve 在 $\eta$ 和 corollas 上读取颜色与运算，并证明 strict Segal 性。
- Dendroidal nerve fully faithfulness 和 $\Delta\hookrightarrow\Omega$ fully faithfulness 已由 MW-1--MW-2 定位；树范畴 generalized Reedy 分解仍标为外部输入或说明性背景。
- 后续第十七章需要把 inner horn fillers 与 homotopy operads/model structure 精确连接。

### `17_dendroidal_inner_kan_and_homotopy_operads.md`

- 已定义 inner horn fillers、inner Kan dendroidal set/dendroidal infinity-operad、homotopy coherent operations、normal dendroidal set、normal monomorphism、inner anodyne map 和 operadic weak equivalence。
- 已证明 inner Kan dendroidal set 的线性限制是 quasi-category，strict operad nerve 有唯一 inner fillers，inner anodynes 对 inner Kan objects 有左提升性质。
- Cisinski-Moerdijk operadic model structure、operadic weak equivalence 的完整刻画和 nerve/model comparison 均标为外部输入。
- 后续第十八章需要把 Lurie-style infinity-operad 与 dendroidal model 的比较定理分离出来，避免混用定义。

### `18_lurie_infinity_operads_and_operadic_fibrations.md`

- 已定义 $\mathbf{Fin}_*$、active/inert morphisms、inert projections、coCartesian edges、Lurie-style infinity-operad、symmetric monoidal infinity-category、algebras over infinity-operads 和 category of operators。
- 已说明 dendroidal nerve 与 category of operators nerve 处于不同模型；比较定理作为外部输入。
- Active-inert 分解方向在本章中显式警告；后续若使用 Lurie 原文定理，需要逐条核对 convention。
- CoCartesian fibration 技术、mapping space 条件和 dendroidal-Lurie model comparison 均标为外部输入；附录 M 已给出跨模型使用路径。

### `19_model_comparison_straightening_and_operadic_localization.md`

- 已定义 relative category、relative functor、Dwyer-Kan localization、DK-equivalence、underlying infinity-category、straightening/unstraightening、operadic straightening、monoidal localization 和 operadic localization。
- 已删除“大模型范畴任取小代表全子范畴”的过强说法：默认在更大 universe 中取 localization；只有本质小性或一个已验证的 small DK presentation 才允许小化，且 cofibrant-fibrant replacement 本身不提供小性。
- 已证明 relative functor 诱导 localization 后的函子、derived tensor product 的 cofibrant replacement 计算、rectification 后 localization 等价。
- Dwyer-Kan localization 存在性、simplicial model category 的 coherent nerve 比较、Quillen equivalence 到 infinity-equivalence 和 operadic straightening 均标为外部输入；ordinary straightening 已由 HTT-1 定位，monoidal Bousfield localization preserves operad/colored-operad algebras 的模型范畴版本已由 WHT-1--WHT-4 与 WY-1--WY-3 定位，完整 infinity-categorical algebra localization comparison 仍为外部输入。
- 已明确“先取代数再 localization”与“先 localization 再取代数”不自动交换。
- 已接入附录 M 的 strict/dendroidal/Lurie/model-category 依赖图。

### `20_factorization_algebras_fukaya_categories_and_geometry.md`

- 已定义 $\mathbf{Disk}_n$、$\mathbf{Disk}_{n/M}$、prefactorization algebra、factorization algebra、locally constant factorization algebra、factorization homology、$A_\infty$-category 和 Fukaya category 的结构性描述。
- Locally constant factorization algebras 与 $E_n$-algebras 的等价、factorization homology excision、Dunn additivity、Fukaya category 构造、operadic Fukaya structures 和 gluing 定理均标为外部输入。
- 已明确 Fukaya category 的完整构造依赖 brane data、transversality、compactness、orientation 和 obstruction theory，不能只由 operad 公理推出。
- 已接入附录 N 的 factorization homology 计算边界和附录 O 的失败模式清单。
- 后续研究边界章节若引用 2026 Fukaya 高阶 operad 结果，需要补充具体模型和版本。

### `DEPENDENCY_GRAPH.md`

- 已建立全书定义、证明和外部输入依赖图。
- 已把基础层、线性与同伦代数层、模型范畴/infinity-operad 层、几何层分开，避免高级比较定理倒用。
- 该文件不新增数学定理；它只约束阅读顺序和审校路径。

### `THEOREM_LEDGER.md`

- 已按基础 operad、线性/Koszul/同伦代数、模型范畴/infinity-operad、几何/前沿和元文档分层，区分内部证明、外部输入和边界说明。
- 已明确当前草稿已经是 operad theory 数学收口版，但不是 camera-ready 出版版本。
- 该文件与附录 D 分工：附录 D 索引外部输入来源，本文件记录每章结论的可使用状态。

### `A_set_theory_universes_finite_sets_and_symmetric_groups.md`

- 已固定 Grothendieck universes、$\mathbf{Fin}_{\mathcal U}$、$\mathbf B_{\mathcal U}$、骨架 $[n]$、$\Sigma_n$、左右作用互译、coinvariants/invariants 和 coends。
- 已证明 $\mathbf B_{\mathcal U}\simeq\coprod B\Sigma_n$、右作用转换公式、特征 $0$ 下 invariants/coinvariants 同构和 $BG$ 上 coend 等于 coinvariants。
- 已警告一般底环上 coinvariants 不 exact，这是 rectification 和 commutative dg algebra 风险来源。

### `B_trees_partitions_substitution_and_coinvariants.md`

- 已定义有限集分块、refinement、分块拉平、全映射对称序列代入乘积、单位对称序列、arity coinvariants 公式和平面树代入。
- 已证明全映射代入乘积结合律、含 arity $0$ 的单位律、arity 公式和树代入结合律；分块拉平只保留为满射特例。
- 已明确 arity 公式依赖附录 A 的右作用转换，避免左右作用混用。

### `H_tree_conventions_and_free_operad_quotients.md`

- 已区分平面有根树、$S$-叶标号非平面树和 Moerdijk-Weiss 树范畴中的 rooted trees。
- 已给出自由对称 operad 的 $\mathcal U$-小树群胚骨架 colimit 公式
  $$
  \mathbb F(E)(S)
  =
  \operatorname*{colim}_{T\in\mathbf{Tree}^{\mathrm{sk}}_S}
  \prod_{v\in V(T)}E(\operatorname{In}(v))
  $$
  及其自由性证明；原先只有协变因子的伪 coend 已移除。
- 已说明 $\Omega(T)$ 是由单棵树生成的 colored operad，不是自由单色 operad $\mathbb F(E)$ 的 arity 值。

### `I_koszul_bar_cobar_strict_conventions.md`

- 已固定 reduced/augmented/coaugmented/conilpotent 约定。
- 已定义 free operad 上的 derivation、cofree cooperad 上的 coderivation、quasi-free/quasi-cofree 对象、convolution dg Lie algebra、twisted composite products 和 Koszul complexes。
- 已同步定义 9.11 的 coaugmentation splitting，并用在单位因子上取零的延拓写出 typed convolution pre-Lie 公式。
- 已证明 derivation/coderivation 的泛性质、Maurer-Cartan 方程与 twisted differential square-zero 的关系、以及 bar/cobar 权重行为。
- Koszul 判别等价、$\Omega\mathcal P^¡\to\mathcal P$ quasi-isomorphism criterion 仍标为外部输入。

### `J_homotopy_transfer_tree_formulas.md`

- 已固定 normalized contraction 和 side conditions。
- 已给出 $A_\infty$ 转移的平面二叉树递归、$I_\infty$ 分量、低阶 $A_\infty$ 恒等式检查。
- 已给出 $L_\infty$ 转移的有根树加 shuffle 反对称化框架。
- 完整 $A_\infty/L_\infty$ 高阶 signs、转移定理和 minimal model 唯一性仍标为外部输入。

### `K_colored_operads_modules_and_enrichment.md`

- 已给出 $C$-轮廓群胚骨架、colored substitution coend 口径、自由 colored operad 的 colored tree 公式。
- 已证明自由 colored operad 的泛性质，并展开代数同态、左模、双模的 colored operad 编码。
- 已定义 enriched colored operad；模型结构 admissibility 仍标为外部输入。

### `L_infinity_algebras_and_en_operad_conventions.md`

- 已固定 $\mathcal P_\infty=\Omega\mathcal P^¡$ 与任意 cofibrant replacement 的区别。
- 已用 $T^c(sA)$ 和 $S^c(sV)$ 上的 square-zero coderivations 给出 $A_\infty/L_\infty$ 的安全定义。
- 已记录 $E_n$、$E_\infty$、Poisson 同调、additivity 和 rectification 的边界。

### `M_dendroidal_lurie_and_model_comparison_map.md`

- 已区分 strict operads、dendroidal sets、Lurie-style infinity-operads 和模型范畴中代数对象四类模型。
- 已给出 strict operad 的 dendroidal nerve 与 category of operators nerve 的不同目标范畴。
- 已列出跨模型允许路径和禁止捷径；White/White--Yau 的模型范畴 localization preservation 已定位，dendroidal-Lurie comparison、category of operators nerve 和 infinity-categorical algebra localization comparison 仍作为外部输入。

### `N_factorization_homology_examples_and_geometry.md`

- 已固定 framed/tangential disk category、factorization homology colimit、不交并公式、disk 归一化和 excision 使用条件。
- 已把圆周计算写成 $HH_\*(A)$ 或 $A\otimes^{\mathbf L}_{A\otimes A^{op}}A$ 的外部输入，并记录 derivedness、边界版本和 cyclic bar 模型边界。
- 已分离交换系数退化、球面 excision 表达式、locally constant factorization algebra 重建和 Fukaya 型 gluing 模式。

### `O_failure_modes_counterexamples_and_boundary_cases.md`

- 已收集 arity $0$、左右群作用、coinvariants/invariants、$E_\infty$ rectification、$\mathcal P_\infty$ 记号、chains、$E_n$ 形式性、dendroidal/Lurie 比较、localization、factorization homology、Fukaya category 和预印本使用中的错误命题。
- 每条均给出失败原因和正确边界，作为正文跨模型断言的检查清单。
- 该附录不新增证明链；它把已有风险集中为可审校约束。

### `P_low_arity_checks_and_worked_computations.md`

- 已给出代入乘积 arity $0,1,2$、自由 operad 二槽权重 $r+\sum_t s_t$、endomorphism operad 结合律、$\operatorname{Ass}$ 字典序复合、$\operatorname{Com}$ arity $0$ 单位、Lie 生成元关系、suspended $A_\infty$ 低阶关系、Hochschild bracket 低阶计算、dendroidal inner horn 最小例子和 cyclic bar levels。
- 该附录用于检查符号、单位、左右作用、唯一填充/存在填充和 derived relative tensor product 的低阶表现。
- 涉及 factorization homology 与 dendroidal inner Kan 的模型比较仍按附录 D 标为外部输入。

### `Q_koszul_complexes_and_bar_cobar_examples.md`

- 已补充二元二次 operad 的权重-arity 关系、非对称 Ass 的关系、二次对偶低阶形状、Koszul twisting morphism 的权重行为、bar/cobar differential 低权重公式和 bar-cobar counit 的低权重检查；counit 现明确杀掉 bar 权重 $>1$ 的单个 cobar 生成元，并由权重 $2$ 的 bar 收缩项与 cobar 分解项相消验证链映射性。
- 附录 Q 已内部证明 $\operatorname{Ass}_{ns}$ 定向结合律终止且唯一临界对合流；由 LV-3 承担“合流二次 rewriting 推出 Koszul”这一步。Koszul 四项判别仍由 LV-1--LV-2、bar-cobar model-context/cofibrancy 由 Fresse 包分别控制；谱序列收敛只在正文写明的逐 arity 有界条件下使用。
- 该附录修复了第八、九章只有抽象定义、缺少可手算局部模型的问题。

### `R_model_category_case_studies.md`

- 已把附录 G 的检查表应用到 $\mathbf{sSet}$、compactly generated spaces、$\mathbf{Ch}_k$ 特征 $0$、一般 $\mathbf{Ch}_R$、非负链复形、spectra、colored operads 和 enriched categories。
- 已明确 rectification 正例必须写出底范畴假设、operad weak equivalence、admissibility、cofibrancy/flatness 和引用定理。
- 已明确正特征和一般底环中 $E_\infty\to\operatorname{Com}$ 逐 arity quasi-isomorphism 不足以推出代数范畴 Quillen equivalence。

### `S_homotopy_transfer_worked_examples.md`

- 已补充 dg associative algebra 的转移乘法 $m_2^H$、三元运算 $m_3^H$、结合律同伦边界、Massey product 关系、formality 低阶判据和 dg Lie algebra 的 $\ell_2/\ell_3$ 转移形状。
- 已明确 $m_3^H$ 依赖 contraction 选择，Massey product 有不定性，二者不能无条件等同。
- 已区分 $A_\infty$-formality 与 strict dg formality，后者需要 rectification 或额外严格化输入。

### `T_dendroidal_horns_segal_and_normality_examples.md`

- 已给出两顶点树 inner horn、三顶点线性树、Segal core、boundary 与 horn 差异、corolla automorphism 和 degeneracy 的低阶样例。
- Unary degeneracy 已按树箭头 $L_1\to\eta$、预层箭头 $X_\eta\to X_{L_1}$ 修正，并与 simplicial $s_0:X_0\to X_1$ 对齐。
- 已明确 strict nerve 中 filler 唯一，一般 inner Kan dendroidal set 只要求 filler 存在。
- Normal monomorphism 和 fully faithfulness 仍作为外部输入，不在附录中伪证。

### `U_props_properads_graphical_calculus_examples.md`

- 已补充 PROP interchange law、双代数兼容关系、Frobenius 关系、properad 连通图复合、PROP/properad 不连通差异、wheeled trace 公式和图替换结合律。
- 已明确 properad 不含任意水平张量，wheeled contraction 需要 dualizability/trace。
- 自由 properad/PROP 的完整图群胚构造仍作为外部输入。

### `V_stratified_and_boundary_factorization_examples.md`

- 已补充半空间、区间、圆周 trace、分层区间、hypersurface defect、corners 和 Fukaya skeleta 的使用边界。
- 已明确无边界 disk 归一化不能直接用于带边界或分层空间。
- Stratified factorization homology、sectorial descent 和 Fukaya skeletal descent 均保留为外部输入。

### `W_sign_convention_crosswalk.md`

- 已把同调分次、operadic suspension、suspended Hochschild cochains、brace signs、$A_\infty/L_\infty$ suspended 主定义和同伦转移符号检查放入同一转换表。
- 已明确 unsuspended 高阶公式不是本书主定义；最终版若展示全公式，必须从 suspended convention 推出。
- 该附录修复了定理账本中“符号已有低阶样例但缺少总表”的缺口。

### `X_concrete_algebraic_examples_and_counterexamples.md`

- 已加入 arity $0$ 与自由交换代数、tensor/symmetric algebra 差异、正特征 coinvariants 不 exact、特征 $2$ Lie 边界、$E_\infty$ rectification 风险、$HH_\*(k)$、$HH_0(M_n(k))$、带边界区间 module 条件例子，以及正特征中 $\operatorname{Sym}^p$ 不保持 acyclic complex 的显式计算。
- 已把正特征 rectification、Morita invariance 和带边界 factorization 的深层结论保持为外部输入或边界说明。
- 该附录用于防止把特征 $0$、交换、无边界或 strict 模型的直觉错误推广。

### `C_model_categories_and_quillen_adjunctions.md`

- 已定义 lifting property、weak factorization system、模型范畴、cofibrant/fibrant replacement、homotopy category、Quillen adjunction、derived functor、Quillen equivalence、monoidal model category 和 monoid axiom。
- 已证明 weak factorization system 的 retract 封闭、replacement 存在、Quillen adjunction 左右条件等价、导出伴随。
- Homotopy category 的 mapping 计算和 Quillen equivalence 导出范畴等价作为标准外部基础事实处理。

### `D_source_theorem_index.md`

- 已按章节索引所有主要外部输入定理，并标出主要来源和后续需补的严格化信息。
- 已新增外部输入状态标签、引用包定义和不倒用原则，区分外部可用、外部候选、研究边界和禁用为证明步骤的材料。
- 已为 Koszul/bar-cobar、同伦转移、operad 模型结构、dendroidal 模型、Lurie-style 比较、localization、factorization homology、Fukaya category 和 2026 前沿建立最小可出版引用包。
- Koszul/bar-cobar 引用包的 GK classical core 已由 GK-1--GK-7 定位，Loday--Vallette Theorems 6.6.2/7.4.6/8.1.1 已由 LV-1--LV-3 定位，Fresse modern cobar/cofibrant replacement 由 FRE-1--FRE-6 定位，Hinich dg-operad model context 由 HIN-1--HIN-2 定位。

### `PUBLICATION_CLOSURE_MATRIX.md`

- 已定义核心可读教材态、基本完本严格草稿态和最终出版态。
- 已判定当前书稿达到基本完本严格草稿态和 operad theory 数学收口态，但尚未达到 camera-ready 出版态。
- 已把基本完本封口项推进到 B13；production 剩余项压缩为 bibliography、page/tag 核验、局部公式指称校对、符号逐模型核查、模型假设翻译和正式参考文献。

### `FINAL_OPERAD_THEORY_CLOSURE.md`

- 已把剩余项目最终分类为内部证明、外部 locator、边界关闭或出版社级 production work。
- 已判定 Koszul/bar-cobar 的 theorem locator 不再空缺；剩余仅为逐符号 crosswalk。HPT/同伦转移符号与 Fukaya/分层几何仍分别属于 convention package 或外部几何边界。
- 已给出最终规则：后续不得把边界关闭项升级为内部证明，不得重新把已定位项目标为未定位主题。

### `PUBLICATION_PROOFING_LEDGER.md`

- 已建立最终出版校对账本，明确出版校对不再横向增加主题。
- 已记录第一轮出版校对动作：修正命题 X.9 自指证明、AF-2 圆周定位、附录 N/E 的符号入口和第十四章 rectification 定位边界。
- 已把主要 theorem locator 收口到批次文件：Ginzburg--Kapranov classical Koszul core、Loday--Vallette LV-1--LV-3、Fresse modern cobar/cofibrant replacement、Hinich dg-operad model context、Markl homotopy transfer existence、Moerdijk--Weiss dendroidal nerve core、White/White--Yau model-category localization preservation、Pavlov--Scholbach modern admissibility/rectification、Hinich Dwyer--Kan localization、Heuts--Hinich--Moerdijk dendroidal-Lurie comparison、Lurie category-of-operators/algebra/monoidal-model comparison、Pratali operadic straightening preprint locator、Deligne locator 和 Dunn additivity locator 均已登记。HPT/transfer signs 和几何/Fukaya 假设包保持为 convention package 或外部边界。

### `INTERNAL_OPERAD_CLOSURE_AUDIT.md`

- 已把审校重心切换到 operad theory 主体的内部闭合：有限集口径、代入乘积、operad 幺半对象定义、endomorphism operad、自由 operad、colored operad、Schur functor 和低阶例子。
- 已确认第一至第七章及附录 A/B/H/K/P/U/X 的 operad theory 主体达到内部闭合严格草稿态。
- 已修正第六章 Schur functor 中的左/右 $\Sigma_n$ 作用转换缺口，并在 `NOTATION.md` 中同步。
- 后续内部任务应优先处理稳定编号、交叉引用、证明压缩和符号表，而不是继续横向扩张外部命题定位。

### `INTERNAL_NUMBERING_AND_CROSSREF_AUDIT.md`

- 已完成第一至第七章编号第一轮审计。
- 已确认 2.8.1、5.16.1、6.2.1 属于可保留的插入编号。
- 已把第七章 7.15 统一为“说明 7.15”。
- 已把自由对称 operad 树群胚公式登记为“第四章加附录 H 内部闭合，最终出版只需文献对照”。

### `LABEL_LEDGER_CH01_07.md`

- 已为第一至第七章生成稳定 label 表。
- 已把 `展开`、`解释`、`说明`、`注`、`警告` 纳入正式可引用 statement type。
- 已确认 `2.8.1`、`2.8.2`、`5.16.1`、`6.2.1` 是可保留插入编号；`注 6.19` 已纳入稳定 label 表。
- 后续正文编辑应优先使用该 label 表替换“上面”“前面”等散文引用。

### `LABEL_LEDGER_CORE_APPENDICES.md`

- 已为附录 A/B/H/K/P/U/X 生成稳定 label 表。
- 已把 `计算 P.7` 和 `边界 X.8` 纳入正式可引用 statement type。
- 已核对核心附录共有 107 个正式编号项，含新增反例 B.10.1/P.0 与命题 X.14.1，全部进入 label 表。
- 该表与第一至第七章 label 表共同覆盖 operad theory 主体内部闭合所依赖的核心引用目标。

### `LABEL_LEDGER_CH08_21.md`

- 已为第八至第二十一章生成稳定 label 规则和编号项清单。
- 已使用 statement type 白名单抽取，避免把证明行中的反向引用误登记为 label。
- 已核对第八至第二十一章共有 416 个正式编号项，其中包括新增 `9.7.1`、`17.18.1`、`19.29.1` 及既有插入编号。
- 结合第一至第七章 label 表，正文二十一章的可引用目标已经闭合。

### `LABEL_LEDGER_REMAINING_APPENDICES.md`

- 已为附录 C/D/E/F/G/I/J/L/M/N/O/Q/R/S/T/V/W/Y/Z 生成稳定 label 规则和编号项清单。
- 已把 `错误命题`、`正确边界`、`事实`、`案例`、`模板`、`检查`、`表`、`低阶形状`、`使用规则` 和 `失败模式` 纳入正式可引用 statement type。
- 已核对剩余附录共有 398 个正式编号项，含新增 G.17、I.1.1 与 I.22.1，全部进入 label 体系。
- 结合核心附录 label 表，附录 A--Z 的可引用目标已经闭合。

### `CROSSREF_REWRITE_AUDIT.md`

- 已完成第一至第七章和核心附录 A/B/H/K/P/U/X 的第一轮散文交叉引用替换。
- 已完成第八至第二十章、主要剩余附录和相关元文档的第二轮散文交叉引用替换。
- 已把可直接定位的“上述规则”“由附录 A/B/H/K”、符号附录泛称和“第八、九章”等替换为具体编号引用。
- 已登记剩余未替换项的类型：局部公式指称、结构标题、主题级概览和书目粒度统一。
- 当前剩余项不阻断 operad theory 主体内部闭合。

### `REFERENCE_LOCATOR_LEDGER.md`

- 已把最终出版前的外部输入定位分为 P0、P1、P2 和 R 四类。
- 已列出 P0 证明链必需包、P1 结构解释必需包、P2 背景源和 R 研究边界源。
- 已明确 P0/P1 theorem locator 属于最终出版引用审校；已定位批次覆盖 Berger--Moerdijk、Cisinski--Moerdijk、Lurie HTT、Ayala--Francis、Ginzburg--Kapranov classical Koszul core、Fresse modern cobar/cofibrant replacement、Hinich dg-operad model context、Markl homotopy transfer existence、Moerdijk--Weiss dendroidal nerve core、White/White--Yau localization preservation、Pavlov--Scholbach admissibility/rectification、Hinich Dwyer--Kan localization、HHM dendroidal-Lurie comparison、Lurie algebra/category-of-operators comparison、Pratali operadic straightening preprint locator、Deligne locator 和 Dunn additivity locator。

### `P0_REFERENCE_LOCATORS_BATCH_1.md`

- 已定位 Berger-Moerdijk `arXiv:math/0206094v3` 中 operad transferred model structure、固定 operad algebra transferred structure 和早期 rectification 相关结论。
- 已定位 Cisinski-Moerdijk `arXiv:0902.1954v2` 中 normal monomorphisms、inner anodyne/horn calculus、operadic model structure 和 fibrant weak equivalence criterion。
- 已标记 Cisinski-Moerdijk erratum 影响需要最终出版前复核。

### `P0_REFERENCE_LOCATORS_BATCH_2.md`

- 已定位 Lurie *Higher Topos Theory* Theorem 3.2.0.1 作为 ordinary straightening/unstraightening 的 P0 来源。
- 已明确该定位不覆盖 operadic straightening、monoidal localization 或 algebra localization comparison。

### `P0_REFERENCE_LOCATORS_BATCH_3.md`

- 已定位 Ayala-Francis `arXiv:1206.5522v6` 中 factorization homology 的总览定理、excision、圆周 Hochschild 计算、Eilenberg--Steenrod 型刻画、带边界版本和交换系数公式。
- 精确定位为 Theorem 1.2、Lemma 3.18、Theorem 3.19、Theorem 3.24、Theorem 3.26 和 Proposition 5.1（AF-5）。
- 已明确该定位不覆盖 Costello-Gwilliam/Lurie 的 locally constant factorization algebra 完整等价、stratified factorization homology、Fukaya descent 或 Hochschild sign model 的逐项核对；Dunn/Lurie additivity 已由 `P1_REFERENCE_LOCATORS_FINAL_SWEEP.md` 中 DUNN-1 定位。

### `P0_REFERENCE_LOCATORS_BATCH_4.md`

- 已定位 Ginzburg-Kapranov `arXiv:0709.1228` 中 quadratic operad Koszul duality 的 classical core。
- 精确定位为 Definition 4.1.3、Proposition 4.1.4、Theorem 4.1.13、Theorem 4.2.5、Corollary 4.2.7、Theorem 3.2.16 和 Section 4.2.12。
- 已定位 Loday--Vallette author-hosted draft v0.99 的 Theorem 6.6.2（LV-1）、Theorem 7.4.6（LV-2）与 Theorem 8.1.1 + following $\operatorname{As}$ example（LV-3），分别覆盖 connected weight-graded twisting-morphism 四项等价、$\mathcal P^¡=\mathcal C(sE,s^2R)$ 的 quadratic Koszul criterion 与 nonsymmetric rewriting criterion。
- 已明确 LV-1--LV-2 给出 quasi-isomorphism criterion，不自动给出所用 dg-operad 模型结构中的 cofibrancy；后者由 Fresse/Hinich 包分开控制。

### `P0_REFERENCE_LOCATORS_BATCH_5.md`

- 已定位 Fresse `arXiv:0902.0177` 中 operadic cobar construction、twisted composite acyclicity、bar-cobar resolution entry、quasi-free/cofibrant replacement 和 homotopy morphism 入口。
- 精确定位为 Section 3.7、Theorem 3.9、Theorem 3.10、Section 3.14、Theorem 4.2.4、Proposition 4.2.7 和 Proposition 4.2.8。
- 已定位 Hinich `arXiv:q-alg/9702015` 中 dg-operad model structure 和 $\Sigma$-split operad algebra homotopy-category comparison，精确定位为 Theorem 6.1.1 和 Theorem 4.7.4。
- 已明确 FRE/HIN 包与 LV-1--LV-2 的分工：前者控制来源中的 $C$-cofibrancy、operad cofibrancy 和 model context，后者控制 connected weight-graded quasi-isomorphism criterion。

### `P0_REFERENCE_LOCATORS_BATCH_6.md`

- 已定位 Markl `arXiv:math/9907138v3` 中 strongly homotopy structures transfer over chain homotopy equivalences 的 operadic existence theorem。
- 精确定位为 Definition 17、Theorem 19、Lemma 20、Theorem 27、Proposition 31、Theorem 33、Proposition 34、Proposition 35 和 Proposition 36。
- 已明确该定位不覆盖 basic perturbation lemma 显式级数、Kadeishvili/Merkulov/Loday--Vallette tree signs、minimal model uniqueness 或 full formality obstruction theory。

### `P0_REFERENCE_LOCATORS_BATCH_7.md`

- 已定位 Moerdijk-Weiss `arXiv:math/0701293v2` 中 dendroidal nerve fully faithfulness、$\Delta\subset\Omega$、strict nerve unique fillers、homotopy coherent dendroidal nerve inner Kan 和 internal Hom inner Kan 条件。
- 精确定位为 Section 3、Example 4.2、Section 4 after Example 4.1、Example 7.1、Proposition 7.2 和 Theorem 7.5。
- 已明确该定位不覆盖 Cisinski-Moerdijk erratum 影响、树范畴 generalized Reedy 分解或 Heuts-Hinich-Moerdijk dendroidal-Lurie comparison。

### `P0_REFERENCE_LOCATORS_BATCH_8.md`

- 已定位 White `arXiv:1404.5197` 中 Bousfield localization preserves operad algebras 的模型范畴版本。
- 精确定位为 Definition 3.1、Theorem 3.2、Corollary 3.4 和 Section 4/Theorems 4.5--4.6 的 monoidal localization criteria。
- 已定位 White--Yau `arXiv:1503.06720` 中 colored operad 版本，精确定位为 Definition 7.2.1、Theorem 7.2.3 和 Theorems 7.4.1--7.4.3。
- 已明确该定位不覆盖 operadic straightening、dendroidal-Lurie comparison、Pavlov--Scholbach 全部 symmetric flatness/rectification 或 Lurie/Hinich 型 infinity-categorical algebra localization comparison。

### `P0_REFERENCE_LOCATORS_BATCH_9.md`

- 已定位 Pavlov--Scholbach `arXiv:1410.5675v4` 中 colored symmetric operad admissibility、strong admissibility、rectification、strict-to-infinity algebra comparison 和 weak symmetric monoidal Quillen adjunction transport。
- 精确定位为 Definition 2.1、Theorem 5.11、Theorems 6.3/6.7、Theorem 7.5、Theorem 7.11 和 Theorem 8.10。
- 已定位 Pavlov--Scholbach `arXiv:1510.04969v3` 中 symmetric h-monoidality、symmetroidality 和 symmetric flatness 的 transfer/localization 稳定性，精确定位为 Theorems 5.6/5.7 和 Theorems 6.4/6.5。
- 已定位 Lurie *Higher Algebra* Theorems 4.1.8.4 与 4.5.4.7 作为 associative/commutative strict algebra 到 infinity-categorical algebra objects 的比较入口。
- 已定位 Lurie *Higher Algebra* Proposition 4.1.7.4 + Example 4.1.7.6（HA-MON-1）及 Corollary 4.1.7.16（HA-MON-2），分别给出模型范畴的 underlying symmetric monoidal infinity-category 与 simplicial 情形的显式 operadic-nerve 模型。

### `P0_REFERENCE_LOCATORS_BATCH_10.md`

- 已定位 Hinich `arXiv:1311.4128` 中 Dwyer--Kan localization revisited 的 infinity localization、hammock comparison、underlying infinity-category、mapping spaces、fibrant-cofibrant subcategory 和 Quillen-pair passage。
- 精确定位为 Section 1.1.2/1.1.3、Proposition 1.2.1、Definition 1.3.1、Theorem 1.3.3、Propositions 1.3.4/1.3.5 和 Proposition 1.5.1。
- 已定位 Heuts--Hinich--Moerdijk dendroidal-Lurie comparison，精确定位为 Theorems 2.4.1、2.5.1、2.5.3、Corollary 2.5.4 和 Theorem 5.3.14。
- 已定位 Lurie *Higher Algebra* Example 2.1.1.21、Definition 2.1.1.23 和 Proposition 2.1.1.27 作为 category-of-operators entry；Pratali `arXiv:2501.05263` 的 Theorem 2.10、Proposition 3.8、Proposition 4.6、Theorem 5.1 和 Corollary 5.2 作为 operadic straightening 的最新/P1 preprint locator。

### `P1_REFERENCE_LOCATORS_FINAL_SWEEP.md`

- 已定位 Lurie *Higher Algebra* Theorem 5.1.2.2 作为 Dunn additivity locator。
- 已定位 McClure--Smith `arXiv:math/9910126v2` 和 Berger--Fresse `arXiv:math/0109158v2` 的 Deligne conjecture/brace/surjection operad locator，精确到 MS-1--MS-3 与 BF-1--BF-4。
- 已把 May recognition、Poisson homology、$E_n$ formality、framed $E_2$/BV、stratified factorization 和 Fukaya geometry 记录为 P1 或几何边界 locator；这些不是 operad theory 内部证明。

### `E_signs_suspensions_and_graded_conventions.md`

- 已定义 Koszul sign rule、tensor differential、suspension/desuspension、Hom differential、operadic suspension、graded Hochschild insertion/cup product、suspended Hochschild braces 和低阶 $A_\infty$ 符号边界。
- 已证明 braiding involutivity、tensor differential square-zero、suspension square-zero 和 Hom differential square-zero。
- Suspended brace signs 已给出统一算法；完整 $L_\infty$ 反对称符号仍需最终版逐项核对。

### `F_classical_operads_and_checked_examples.md`

- 已展开 Ass、Com、Endomorphism、Lie 和 Poisson operad 的逐项验算。
- 已证明集合值线性化保持 operad 结构、$R[\operatorname{Ass}]$ 的 Schur functor 为张量代数、$R[\operatorname{Com}]$ 的 Schur functor 为对称代数，并验证 arity $0$ 元素给出单位。
- Lie 的组合模型、PBW/Shirshov-Witt 型自由 Lie 定理和 little cubes 同调识别均已标为外部输入。

### `G_model_structure_hypotheses_and_rectification.md`

- 已分离 operad 模型结构、固定 operad 代数模型结构和 rectification 三个不同问题。
- 已给出 T0-T3 假设包、自由代数对称幂风险、常见底范畴状态表、admissibility 检查表和 rectification 检查表。
- Operad transferred model structure、colored admissibility、rectification schema 和正特征反例边界均标为外部输入。

### `21_research_frontier_2026.md`

- 已把 2026 年核查过的近期 arXiv 条目标为研究边界，而非正文定理。
- 已加入版本表、模型差异表和进入正文的验证流程。
- 后续若使用其中结果，必须补充版本号、定理编号、模型约定、符号转换和依赖链。

### `Y_infinity_operadic_homology_and_koszul_frontier.md`

- 已把 infinity-operadic homology 与 algebras over infinity-operads 的 Koszul 对偶方向转化为 strict operad 可验证接口。
- 已证明 strict operad 的树指标线性化、Segal-linear strict 特化、ordinary algebra 与 dendroidal natural transformation 的等价，以及 Koszul extension 的 strict specialization test。
- Hoffbeck-Moerdijk 的新结果仍保持研究边界；本附录不把 linear infinity-operad 的 Koszul duality 并入第八、九章定理链。

### `Z_operadic_categories_relative_rezk_and_fukaya_frontier.md`

- 已给出 operadic category 数据包、Boardman-Vogt interchange 低阶形式、relative dendroidal object、dendroidal Rezk nerve 接口和 Fukaya operadic interface。
- 已证明 finite-set fiber flattening，并把非空分块识别为无空纤维特例；另已证明 relative strict nerve 的基本例子、线性限制退化为 ordinary relative object，以及 Fukaya $A_\infty$ 关系的条件性边界证明。
- Operadic nerve、relative dendroidal Rezk nerve 和 Fukaya 高阶结构仍保持外部输入或研究边界。

## 2026-07-11 OET 严格性复核

- **代入乘积.** 序章、第一至第六章、第十五章与附录 A/B/F/K/P 已统一为对所有有限集映射取代入；空纤维编码 nullary slots。自由代数、自由树 operad、operadic congruence、线性化、simplicial/topological operad 和 chains 构造均已同步。结合律由可组合映射 $S\to U\to T$ 的群胚识别证明，左右单位包含 $S=\varnothing$；反例 B.10.1/P.0 说明非空分块公式会破坏左单位。
- **模型结构.** 第十四章与附录 G 已分开 operad 模型结构、固定 operad 的 algebra transfer、strong admissibility 和 rectification；BM-1、PSAR-2、PSAR-4/PSAR-5 的假设与结论逐项登记。X.15--X.16 给出正特征对称幂不保持平凡 cofibration 的内部计算。
- **Koszul/bar-cobar.** 第八、九章与附录 I/Q 已区分 $\mathcal P^!$ 和 $\mathcal P^¡$，补出 conilpotence、直和/完成化、bar 递增与 cobar 递减滤过及逐 arity 收敛条件。I.22.1 给出 unary 情形下直和与完成化不同的反例。
- **Dendroidal 比较.** 第十七、十九章与附录 M/T 已把完整 inner horn 论证交还 MW-4/CM-3，并明确 HHM zig-zag 只覆盖 open/no-constants operads；本书默认 arity $0$ 不能无检查套用。
- **Factorization homology.** 第二十章与附录 N/V 已固定 framed/tangential 语境，disk normalization 改用 slice final object，球面切割改用 open collar pieces；N.30 降为研究边界，交换系数公式定位到 AF-5。

## 后续审查清单

- 每章新增术语是否已进入 `NOTATION.md`。
- 每个“定理”是否有证明或外部输入来源。
- 每个例子是否验证单位、结合律和等变性。
- 每处“algebra over an operad”是否说明底层范畴。
- 每处“homotopy operad”是否说明模型结构。
- 每处“infinity-operad”是否说明模型。
- 每个近期论文结论是否保留了发布日期、版本和来源链接。
