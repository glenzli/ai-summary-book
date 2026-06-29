# 数学审查记录

核查日期：2026-06-29。

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

### `01_symmetric_sequences_and_operads.md`

- 已使用有限集上的对称序列定义，避免早期左右作用歧义。
- 已把代入乘积的幺半相干性从证明草图加强为多层分块拉平的自然同构论证。
- 已给出代入乘积、单位对称序列、operad、endomorphism operad、operad 代数、Ass 和 Com 的基础定义。
- 当前证明主要覆盖集合值 operad；后续推广到 $\mathbf{Mod}_R$ 和 $\mathbf{Ch}_R$ 时必须重新检查 coinvariants、直和、张量分配和符号规则。

### `02_operad_algebras_free_algebras_and_monads.md`

- 已给出 $\mathcal O$-代数同态、限制标量、自由代数、自由-遗忘伴随和 monad 识别。
- 已补充自由代数的有限集 coend 公式
  $$
  T_{\mathcal O}(A)\cong\int^{S\in\mathbf B_{\mathcal U}}\mathcal O(S)\times A^S
  $$
  并把自由代数动作改写为无坐标分块复合。
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
- 后续若进入 enriched colored operad，需要把集合值 Hom 换成指定对称幺半范畴中的对象。

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
- 已定义二次数据、二次对偶和 Koszul 性；Ass/Com/Lie 的 Koszul 性只作为外部输入定理。
- 第九章已补充 dg cooperad、twisting morphism 和基本符号规则；附录 E 已给出 operadic suspension 和 suspended brace 的基础算法，完整 $L_\infty$ 展开符号仍需最终版逐项核对。

### `09_bar_cobar_constructions_and_twisting_morphisms.md`

- 已固定同调分次和 Koszul sign rule，并定义 dg-operad、dg-cooperad、cofree conilpotent cooperad、twisting morphism、bar/cobar 构造。
- 已证明 bar-cobar 泛性质；Koszul twisting morphism 的 quasi-isomorphism 判别标为外部输入定理。
- 已接入附录 I 的 convolution Lie algebra、twisted composite product、Koszul complex 和权重滤过约定；进入具体 $A_\infty/L_\infty$ 章节时仍需与附录 E/J 的 signs 对齐。

### `10_a_infinity_l_infinity_and_e_n_operads.md`

- 已以 bar-cobar 定义为主定义给出 $A_\infty$、$L_\infty$ 和 $C_\infty$，手写恒等式只作为展开说明。
- 已区分 $E_n$-operad 与 Lurie-style infinity-operad。
- $H_\*(\mathcal C_n)\cong\operatorname{Pois}_n$、May recognition principle 和形式性均标为外部输入定理。
- 具体 signs 仍需逐项校验；附录 E 已固定 Koszul signs、suspension 和 Hochschild 约定的基础版本。

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
- Homological perturbation lemma、完整转移定理、minimal model 存在唯一性均标为外部输入定理。
- 已接入附录 J 的 normalized contraction、$A_\infty$ 平面二叉树递归、$L_\infty$ shuffle 反对称化和低阶恒等式检查。
- $m_3$ 和 Massey products 的关系只作说明；完整选择依赖仍需后续例子章节展开。

### `14_operads_in_model_categories.md`

- 已定义对称幺半模型范畴、monoid axiom、对称序列 projective 模型结构、operad transferred 模型结构、admissible operad、rectification、Boardman-Vogt resolution 和 derived mapping space。
- Berger-Moerdijk 转移定理、代数范畴 admissibility、rectification criterion 和 $W$-construction 均标为外部输入定理。
- 已明确 positive characteristic / general ring 中 $E_\infty$ 与严格 commutative dg algebra 不能无条件 rectification。
- 已接入附录 G 的模型结构假设检查表；具体 transferred model structure 的假设仍需在最终版逐例核对到文献定理编号。

### `15_simplicial_and_topological_operads.md`

- 已定义 $\mathbf{sSet}$、$\mathbf{Top}$、simplicial operad、topological operad、well-pointed 与 $\Sigma$-free 条件、little cubes operad、chains on spaces 和 unary colored simplicial operads。
- Kan-Quillen 模型结构、$\mathbf{sSet}$ 与 $\mathbf{Top}$ 的 Quillen equivalence、operad 模型结构提升、Eilenberg-Zilber 相干性和 May recognition principle 均标为外部输入定理。
- 已说明 $C_\*(\mathcal C_d;k)$ 依赖 lax monoidal chains，形式性结论不能由取 chains 自动推出。
- 第十六章已把 simplicial categories 作为线性树入口，并定义 Moerdijk-Weiss 树范畴 $\Omega$ 和 dendroidal nerve。

### `16_dendroidal_sets_and_tree_category.md`

- 已定义 rooted tree、inner/outer edges、单位树、corolla、线性树、自由 colored operad $\Omega(T)$、树范畴 $\Omega$、dendroidal set、representable、dendroidal nerve、Segal core、faces、degeneracies、boundaries 和 inner horns。
- 已证明 dendroidal nerve 在 $\eta$ 和 corollas 上读取颜色与运算，并证明 strict Segal 性。
- Dendroidal nerve fully faithfulness、$\Delta\hookrightarrow\Omega$ fully faithfulness 和树范畴 generalized Reedy 分解均标为外部输入或说明性背景。
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
- CoCartesian fibration 技术、mapping space 条件和 dendroidal-Lurie model comparison 均标为外部输入。

### `19_model_comparison_straightening_and_operadic_localization.md`

- 已定义 relative category、relative functor、Dwyer-Kan localization、DK-equivalence、underlying infinity-category、straightening/unstraightening、operadic straightening、monoidal localization 和 operadic localization。
- 已证明 relative functor 诱导 localization 后的函子、derived tensor product 的 cofibrant replacement 计算、rectification 后 localization 等价。
- Dwyer-Kan localization 存在性、simplicial model category 的 coherent nerve 比较、Quillen equivalence 到 infinity-equivalence、straightening/unstraightening、monoidal localization 和 algebra localization comparison 均标为外部输入。
- 已明确“先取代数再 localization”与“先 localization 再取代数”不自动交换。

### `20_factorization_algebras_fukaya_categories_and_geometry.md`

- 已定义 $\mathbf{Disk}_n$、$\mathbf{Disk}_{n/M}$、prefactorization algebra、factorization algebra、locally constant factorization algebra、factorization homology、$A_\infty$-category 和 Fukaya category 的结构性描述。
- Locally constant factorization algebras 与 $E_n$-algebras 的等价、factorization homology excision、Dunn additivity、Fukaya category 构造、operadic Fukaya structures 和 gluing 定理均标为外部输入。
- 已明确 Fukaya category 的完整构造依赖 brane data、transversality、compactness、orientation 和 obstruction theory，不能只由 operad 公理推出。
- 后续研究边界章节若引用 2026 Fukaya 高阶 operad 结果，需要补充具体模型和版本。

### `A_set_theory_universes_finite_sets_and_symmetric_groups.md`

- 已固定 Grothendieck universes、$\mathbf{Fin}_{\mathcal U}$、$\mathbf B_{\mathcal U}$、骨架 $[n]$、$\Sigma_n$、左右作用互译、coinvariants/invariants 和 coends。
- 已证明 $\mathbf B_{\mathcal U}\simeq\coprod B\Sigma_n$、右作用转换公式、特征 $0$ 下 invariants/coinvariants 同构和 $BG$ 上 coend 等于 coinvariants。
- 已警告一般底环上 coinvariants 不 exact，这是 rectification 和 commutative dg algebra 风险来源。

### `B_trees_partitions_substitution_and_coinvariants.md`

- 已定义有限集分块、refinement、分块拉平、对称序列代入乘积、单位对称序列、arity coinvariants 公式和平面树代入。
- 已证明分块拉平结合律、代入乘积结合律、单位律、arity 公式和树代入结合律。
- 已明确 arity 公式依赖附录 A 的右作用转换，避免左右作用混用。

### `H_tree_conventions_and_free_operad_quotients.md`

- 已区分平面有根树、$S$-叶标号非平面树和 Moerdijk-Weiss 树范畴中的 rooted trees。
- 已给出自由对称 operad 的树群胚 coend 公式
  $$
  \mathbb F(E)(S)=\int^{T\in\mathbf{Tree}_S}\prod_{v\in V(T)}E(\operatorname{In}(v))
  $$
  及其自由性证明。
- 已说明 $\Omega(T)$ 是由单棵树生成的 colored operad，不是自由单色 operad $\mathbb F(E)$ 的 arity 值。

### `I_koszul_bar_cobar_strict_conventions.md`

- 已固定 reduced/augmented/coaugmented/conilpotent 约定。
- 已定义 free operad 上的 derivation、cofree cooperad 上的 coderivation、quasi-free/quasi-cofree 对象、convolution dg Lie algebra、twisted composite products 和 Koszul complexes。
- 已证明 derivation/coderivation 的泛性质、Maurer-Cartan 方程与 twisted differential square-zero 的关系、以及 bar/cobar 权重行为。
- Koszul 判别等价、$\Omega\mathcal P^¡\to\mathcal P$ quasi-isomorphism criterion 仍标为外部输入。

### `J_homotopy_transfer_tree_formulas.md`

- 已固定 normalized contraction 和 side conditions。
- 已给出 $A_\infty$ 转移的平面二叉树递归、$I_\infty$ 分量、低阶 $A_\infty$ 恒等式检查。
- 已给出 $L_\infty$ 转移的有根树加 shuffle 反对称化框架。
- 完整 $A_\infty/L_\infty$ 高阶 signs、转移定理和 minimal model 唯一性仍标为外部输入。

### `C_model_categories_and_quillen_adjunctions.md`

- 已定义 lifting property、weak factorization system、模型范畴、cofibrant/fibrant replacement、homotopy category、Quillen adjunction、derived functor、Quillen equivalence、monoidal model category 和 monoid axiom。
- 已证明 weak factorization system 的 retract 封闭、replacement 存在、Quillen adjunction 左右条件等价、导出伴随。
- Homotopy category 的 mapping 计算和 Quillen equivalence 导出范畴等价作为标准外部基础事实处理。

### `D_source_theorem_index.md`

- 已按章节索引所有主要外部输入定理，并标出主要来源和后续需补的严格化信息。
- 当前索引仍是初稿；最终版需要把“主要来源”改成精确到定理编号、页码或 arXiv 版本。

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
- 后续若使用其中结果，必须补充版本号、定理编号、模型约定和依赖链。

## 后续审查清单

- 每章新增术语是否已进入 `NOTATION.md`。
- 每个“定理”是否有证明或外部输入来源。
- 每个例子是否验证单位、结合律和等变性。
- 每处“algebra over an operad”是否说明底层范畴。
- 每处“homotopy operad”是否说明模型结构。
- 每处“infinity-operad”是否说明模型。
- 每个近期论文结论是否保留了发布日期、版本和来源链接。
