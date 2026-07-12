# 第一至第七章稳定 label 表

本文件是出版化交叉引用的第一批稳定 label 表。它不新增数学内容；它把第一至第七章中已经编号的定义、展开、解释、约定、注、警告、命题、定理、推论、例、说明、外部输入定理和练习登记为可迁移的引用目标。

## 0. Label 规则

**规则 0.1（label 形态）.** 正文 label 采用
`chNN-kind-number-slug`
形态，其中 `NN` 是两位章号，`kind` 取 `def`、`exp`、`conv`、`warn`、`prop`、`thm`、`lem`、`cor`、`ex`、`note`、`extthm` 或 `exercise`。

**规则 0.2（编号优先）.** label 绑定到数学声明编号，而不是绑定到当前 Markdown 行号。后续重排段落时不得改变本表中的 label；若最终自动编号改变显示编号，应另建迁移表。

**规则 0.3（statement type）.** `展开`、`解释`、`说明`、`注`、`警告` 均可作为引用目标。出版时可以统一样式，但不得在未同步本表前删除其编号。

**规则 0.4（插入编号）.** `2.8.1`、`2.8.2`、`5.16.1`、`6.2.1` 是已登记的插入编号。它们不表示编号错误；后续引用应直接使用这些编号。

## 1. 第一章

| 编号 | label | 主题 |
| --- | --- | --- |
| 定义 1.1 | `ch01-def-01-01-finite-set-groupoid` | 有限集群胚 $\mathbf B_{\mathcal U}$ |
| 例 1.2 | `ch01-ex-01-02-sigma-action` | 骨架 $[n]$ 与 $\Sigma_n$ 作用 |
| 定义 1.3 | `ch01-def-01-03-symmetric-sequences` | 对称序列范畴 |
| 定义 1.4 | `ch01-def-01-04-partitions` | 有限集分块 |
| 定义 1.4.1 | `ch01-def-01-04-01-fiber-decomposition-groupoid` | 允许空纤维的有限映射群胚 $\operatorname{Fib}(S)$ |
| 定义 1.5 | `ch01-def-01-05-substitution-product` | 代入乘积 $X\circ Y$ |
| 说明 1.5.1 | `ch01-note-01-05-01-partition-formula-boundary` | 非空分块公式只适用于内层 arity $0$ 为空 |
| 命题 1.6 | `ch01-prop-01-06-substitution-functor` | 代入乘积的函子性 |
| 定义 1.7 | `ch01-def-01-07-unit-symmetric-sequence` | 单位对称序列 |
| 命题 1.8 | `ch01-prop-01-08-monoidal-symmetric-sequences` | 对称序列的幺半范畴结构 |
| 定义 1.9 | `ch01-def-01-09-operad-monoid-object` | operad 作为幺半对象 |
| 展开 1.10 | `ch01-exp-01-10-operad-substitution` | operad 代入的有限集展开 |
| 定义 1.11 | `ch01-def-01-11-operad-morphism` | operad morphism |
| 定义 1.12 | `ch01-def-01-12-endomorphism-operad` | endomorphism operad |
| 命题 1.13 | `ch01-prop-01-13-endomorphism-operad-structure` | $\operatorname{End}_X$ 的 operad 结构 |
| 定义 1.14 | `ch01-def-01-14-operad-algebra` | 集合值 operad 代数 |
| 命题 1.15 | `ch01-prop-01-15-algebra-action-equivalence` | operad morphism 口径与动作口径等价 |
| 定义 1.16 | `ch01-def-01-16-commutative-operad` | $\operatorname{Com}$ |
| 命题 1.17 | `ch01-prop-01-17-com-algebras` | $\operatorname{Com}$-代数与交换幺半群 |
| 定义 1.18 | `ch01-def-01-18-associative-operad` | $\operatorname{Ass}$ |
| 命题 1.19 | `ch01-prop-01-19-ass-algebras` | $\operatorname{Ass}$-代数与幺半群 |
| 命题 1.20 | `ch01-prop-01-20-arity-equivalence` | 有限集口径与 arity 口径等价 |
| 练习 1.1 | `ch01-exercise-01-01-unit-substitution` | $I\circ X\cong X$ |
| 练习 1.2 | `ch01-exercise-01-02-endomorphism-low-arity` | endomorphism operad 低阶项 |
| 练习 1.3 | `ch01-exercise-01-03-com-nullary-unit` | $\operatorname{Com}$ 的零元运算 |
| 练习 1.4 | `ch01-exercise-01-04-associative-order-substitution` | $\operatorname{Ass}$ 的全序代入 |
| 练习 1.5 | `ch01-exercise-01-05-right-action-arity-form` | 右作用 arity 写法 |

## 2. 第二章

| 编号 | label | 主题 |
| --- | --- | --- |
| 定义 2.1 | `ch02-def-02-01-algebra-morphism` | $\mathcal O$-代数同态 |
| 命题 2.2 | `ch02-prop-02-02-algebra-category` | $\mathcal O$-代数范畴 |
| 定义 2.3 | `ch02-def-02-03-forgetful-functor` | 遗忘函子 |
| 定义 2.4 | `ch02-def-02-04-restriction-of-scalars` | 限制标量函子 |
| 命题 2.5 | `ch02-prop-02-05-restriction-functoriality` | 限制标量的函子性 |
| 定义 2.6 | `ch02-def-02-06-free-algebra-underlying-set` | 自由代数底集合 |
| 解释 2.7 | `ch02-exp-02-07-formal-expressions` | 自由代数元素的形式表达式解释 |
| 定义 2.8 | `ch02-def-02-08-unit-inclusion` | 单位插入 $\iota_A$ |
| 命题 2.8.1 | `ch02-prop-02-08-1-finite-set-coend` | 有限集 coend 公式 |
| 说明 2.8.2 | `ch02-note-02-08-2-safe-coend-formula` | coend 安全公式说明 |
| 定义 2.9 | `ch02-def-02-09-free-algebra-action` | 自由代数动作 |
| 引理 2.10 | `ch02-lem-02-10-representative-independence` | 动作与代表元选择无关 |
| 命题 2.11 | `ch02-prop-02-11-free-algebra-is-algebra` | 自由代数结构 |
| 定理 2.12 | `ch02-thm-02-12-free-algebra-universal-property` | 自由代数泛性质 |
| 定义 2.13 | `ch02-def-02-13-operad-monad` | operad 单子 |
| 命题 2.14 | `ch02-prop-02-14-eilenberg-moore-equivalence` | operad 代数与 Eilenberg-Moore 代数 |
| 命题 2.15 | `ch02-prop-02-15-filtered-colimits` | operad monad 保持滤过余极限 |
| 命题 2.16 | `ch02-prop-02-16-free-com-multisets` | 自由交换代数 |
| 命题 2.17 | `ch02-prop-02-17-free-ass-lists` | 自由结合代数 |
| 练习 2.1 | `ch02-exercise-02-01-orbit-relation` | 自由代数商关系 |
| 练习 2.2 | `ch02-exercise-02-02-extension-homomorphism` | 泛性质中的延拓同态 |
| 练习 2.3 | `ch02-exercise-02-03-free-com-low-degree` | 低次数自由交换代数 |
| 练习 2.4 | `ch02-exercise-02-04-free-ass-order` | 自由结合代数中的顺序 |
| 练习 2.5 | `ch02-exercise-02-05-ass-to-com-restriction` | $\operatorname{Ass}\to\operatorname{Com}$ 的限制标量 |

## 3. 第三章

| 编号 | label | 主题 |
| --- | --- | --- |
| 定义 3.1 | `ch03-def-03-01-nonsymmetric-sequence` | 非对称序列 |
| 定义 3.2 | `ch03-def-03-02-nonsymmetric-substitution` | 非对称代入乘积 |
| 命题 3.3 | `ch03-prop-03-03-monoidal-nonsymmetric-sequences` | 非对称序列的幺半范畴结构 |
| 定义 3.4 | `ch03-def-03-04-nonsymmetric-operad` | 非对称 operad |
| 定义 3.5 | `ch03-def-03-05-partial-composition` | 偏复合 |
| 命题 3.6 | `ch03-prop-03-06-partial-composition-identities` | 偏复合恒等式 |
| 定义 3.7 | `ch03-def-03-07-partial-composition-operad` | 偏复合型非对称 operad |
| 定理 3.8 | `ch03-thm-03-08-equivalence-of-nonsymmetric-definitions` | 两种非对称 operad 定义等价 |
| 定义 3.9 | `ch03-def-03-09-underlying-nonsymmetric-operad` | 对称 operad 的底层非对称 operad |
| 定义 3.10 | `ch03-def-03-10-symmetric-partial-composition` | 对称 operad 的偏复合 |
| 命题 3.11 | `ch03-prop-03-11-symmetric-partial-identities` | 对称偏复合满足非对称恒等式 |
| 定义 3.12 | `ch03-def-03-12-planar-rooted-tree` | 平面有根树 |
| 定义 3.13 | `ch03-def-03-13-decorated-planar-tree` | $\mathcal P$-装饰平面树 |
| 定义 3.14 | `ch03-def-03-14-edge-contraction` | 内部边收缩 |
| 命题 3.15 | `ch03-prop-03-15-contraction-independence` | 收缩顺序无关 |
| 推论 3.16 | `ch03-cor-03-16-treewise-composition` | 树形复合口径 |
| 练习 3.1 | `ch03-exercise-03-01-low-arity-nonsymmetric-substitution` | 非对称代入低阶项 |
| 练习 3.2 | `ch03-exercise-03-02-partial-associativity` | 偏复合公理计算 |
| 练习 3.3 | `ch03-exercise-03-03-chain-tree-identity` | 链形树与嵌套代入 |
| 练习 3.4 | `ch03-exercise-03-04-forgetting-symmetry` | 忘掉对称作用 |
| 练习 3.5 | `ch03-exercise-03-05-ass-partial-compositions` | $\operatorname{Ass}$ 的偏复合 |

## 4. 第四章

| 编号 | label | 主题 |
| --- | --- | --- |
| 定义 4.1 | `ch04-def-04-01-free-nonsymmetric-sequence` | 自由非对称 operad 的底序列 |
| 定义 4.2 | `ch04-def-04-02-grafting` | 装饰平面树 grafting |
| 命题 4.3 | `ch04-prop-04-03-free-nonsymmetric-operad` | 自由非对称 operad 结构 |
| 定理 4.4 | `ch04-thm-04-04-free-nonsymmetric-universal-property` | 自由非对称 operad 泛性质 |
| 定义 4.5 | `ch04-def-04-05-s-labelled-rooted-tree` | $S$-标号有根树 |
| 定义 4.6 | `ch04-def-04-06-e-decorated-labelled-tree` | $E$-装饰 $S$-标号树 |
| 定义 4.7 | `ch04-def-04-07-free-symmetric-sequence` | 对称自由 operad 的底对称序列 |
| 定义 4.8 | `ch04-def-04-08-tree-substitution` | 对称树代入 |
| 命题 4.9 | `ch04-prop-04-09-free-symmetric-operad` | 对称自由 operad 结构 |
| 定理 4.10 | `ch04-thm-04-10-free-symmetric-universal-property` | 对称自由 operad 泛性质 |
| 定义 4.11 | `ch04-def-04-11-operadic-congruence` | operadic congruence |
| 命题 4.12 | `ch04-prop-04-12-quotient-operad` | operad 商 |
| 定义 4.13 | `ch04-def-04-13-presentation-by-generators-relations` | 生成元与关系表示 |
| 命题 4.14 | `ch04-prop-04-14-presentation-universal-property` | 表示的泛性质 |
| 例 4.15 | `ch04-ex-04-15-ass-presentation` | $\operatorname{Ass}$ 的生成元关系 |
| 命题 4.16 | `ch04-prop-04-16-ass-presentation-algebras` | $\operatorname{Ass}$ 表示的代数 |
| 例 4.17 | `ch04-ex-04-17-com-presentation` | $\operatorname{Com}$ 的生成元关系 |
| 命题 4.18 | `ch04-prop-04-18-com-presentation-algebras` | $\operatorname{Com}$ 表示的代数 |
| 练习 4.1 | `ch04-exercise-04-01-unit-tree` | 自由非对称 operad 的单位树 |
| 练习 4.2 | `ch04-exercise-04-02-ternary-generator-arity-five` | 三元生成元的 arity 5 元素 |
| 练习 4.3 | `ch04-exercise-04-03-labelled-leaves` | 对称自由 operad 的叶标号 |
| 练习 4.4 | `ch04-exercise-04-04-intersections-of-congruences` | congruence 的交 |
| 练习 4.5 | `ch04-exercise-04-05-nonunital-ass` | 无单位结合 operad |

## 5. 第五章

| 编号 | label | 主题 |
| --- | --- | --- |
| 定义 5.1 | `ch05-def-05-01-colors-and-profiles` | 颜色与轮廓 |
| 定义 5.2 | `ch05-def-05-02-colored-symmetric-sequence` | colored symmetric sequence |
| 例 5.3 | `ch05-ex-05-03-one-color-reduction` | 单色退化 |
| 定义 5.4 | `ch05-def-05-04-colored-substitution-product` | colored substitution product |
| 定义 5.5 | `ch05-def-05-05-colored-unit` | 单位 colored symmetric sequence |
| 命题 5.6 | `ch05-prop-05-06-monoidal-colored-symseq` | colored symmetric sequence 的幺半范畴结构 |
| 定义 5.7 | `ch05-def-05-07-colored-operad` | colored operad |
| 展开 5.8 | `ch05-exp-05-08-colored-substitution-expansion` | colored operad 代入展开 |
| 定义 5.9 | `ch05-def-05-09-colored-operad-morphism` | colored operad morphism |
| 定义 5.10 | `ch05-def-05-10-colored-endomorphism-operad` | colored endomorphism operad |
| 命题 5.11 | `ch05-prop-05-11-colored-endomorphism-operad-structure` | $\operatorname{End}_A$ 的 colored operad 结构 |
| 定义 5.12 | `ch05-def-05-12-colored-operad-algebra` | colored operad 代数 |
| 展开 5.13 | `ch05-exp-05-13-colored-algebra-operations` | colored 代数动作展开 |
| 定义 5.14 | `ch05-def-05-14-colored-algebra-morphism` | colored 代数同态 |
| 定义 5.15 | `ch05-def-05-15-symmetric-multicategory` | symmetric multicategory |
| 命题 5.16 | `ch05-prop-05-16-colored-operads-multicategories` | colored operad 与 multicategory 等价 |
| 说明 5.16.1 | `ch05-note-05-16-1-appendix-k-interface` | 附录 K 接口 |
| 例 5.17 | `ch05-ex-05-17-category-as-colored-operad` | 小范畴给出 colored operad |
| 命题 5.18 | `ch05-prop-05-18-category-algebras-functors` | 范畴例子的代数 |
| 例 5.19 | `ch05-ex-05-19-bimodule-action-colored-operad` | 双边作用的 colored operad |
| 例 5.20 | `ch05-ex-05-20-homomorphism-colored-operad` | 指定同态的 colored operad |
| 练习 5.1 | `ch05-exercise-05-01-one-color-substitution` | 单色退化为普通代入 |
| 练习 5.2 | `ch05-exercise-05-02-colored-endomorphism-associativity` | colored endomorphism 结合律 |
| 练习 5.3 | `ch05-exercise-05-03-category-example-unary` | 小范畴例子的非一元运算 |
| 练习 5.4 | `ch05-exercise-05-04-right-action-colored-presentation` | 右作用的 colored 表示 |
| 练习 5.5 | `ch05-exercise-05-05-homomorphism-needs-colors` | 普通 operad 不能直接编码同态 |

## 6. 第六章

| 编号 | label | 主题 |
| --- | --- | --- |
| 约定 6.1 | `ch06-conv-06-01-base-ring` | 底交换环 $R$ |
| 定义 6.2 | `ch06-def-06-02-r-linear-symmetric-sequence` | $R$-模值对称序列 |
| 约定 6.2.1 | `ch06-conv-06-02-1-schur-right-action` | Schur functor 中的右作用 |
| 定义 6.3 | `ch06-def-06-03-linear-substitution-product` | 线性代入乘积 |
| 定义 6.4 | `ch06-def-06-04-linear-unit` | 线性单位对称序列 |
| 命题 6.5 | `ch06-prop-06-05-monoidal-linear-symseq` | 线性对称序列的幺半范畴结构 |
| 定义 6.6 | `ch06-def-06-06-linear-operad` | $R$-线性 operad |
| 命题 6.7 | `ch06-prop-06-07-linear-arity-formula` | 线性代入的 arity 公式 |
| 警告 6.8 | `ch06-warn-06-08-coinvariants-not-invariants` | coinvariants 与 invariants 的区别 |
| 定义 6.9 | `ch06-def-06-09-schur-functor` | Schur functor |
| 命题 6.10 | `ch06-prop-06-10-schur-composition` | Schur functor 与代入乘积 |
| 推论 6.11 | `ch06-cor-06-11-operad-schur-monad` | operad 的 Schur monad |
| 定义 6.12 | `ch06-def-06-12-linear-operad-algebra` | 线性 operad 代数 |
| 定义 6.13 | `ch06-def-06-13-linearization` | 集合值 operad 的线性化 |
| 命题 6.14 | `ch06-prop-06-14-linearized-operad` | 线性化给出线性 operad |
| 命题 6.15 | `ch06-prop-06-15-linear-ass-algebras` | $R[\operatorname{Ass}]$-代数 |
| 命题 6.16 | `ch06-prop-06-16-linear-com-algebras` | $R[\operatorname{Com}]$-代数 |
| 定义 6.17 | `ch06-def-06-17-lie-operad` | Lie operad |
| 命题 6.18 | `ch06-prop-06-18-lie-algebras` | Lie operad 的代数 |
| 注 6.19 | `ch06-note-06-19-lie-alternating-convention` | Lie 代数 alternating 约定 |
| 定义 6.20 | `ch06-def-06-20-poisson-operad` | Poisson operad |
| 命题 6.21 | `ch06-prop-06-21-poisson-algebras` | Poisson operad 的代数 |
| 练习 6.1 | `ch06-exercise-06-01-low-arity-linear-substitution` | 线性代入低阶项 |
| 练习 6.2 | `ch06-exercise-06-02-schur-unit` | $S_{I_R}(V)\cong V$ |
| 练习 6.3 | `ch06-exercise-06-03-free-commutative-algebra` | 自由交换代数与对称代数 |
| 练习 6.4 | `ch06-exercise-06-04-jacobi-in-endomorphism-operad` | Jacobi 的树形复合 |
| 练习 6.5 | `ch06-exercise-06-05-characteristic-two-lie-warning` | 特征 $2$ 下的反对称关系 |

## 7. 第七章

| 编号 | label | 主题 |
| --- | --- | --- |
| 定义 7.1 | `ch07-def-07-01-prop` | PROP |
| 展开 7.2 | `ch07-exp-07-02-prop-data` | PROP 的展开数据 |
| 命题 7.3 | `ch07-prop-07-03-prop-definitions-equivalent` | 两种 PROP 定义等价 |
| 定义 7.4 | `ch07-def-07-04-s-bimodule` | $\mathbb S$-双模 |
| 定义 7.5 | `ch07-def-07-05-endomorphism-prop` | endomorphism PROP |
| 命题 7.6 | `ch07-prop-07-06-endomorphism-prop-structure` | $\operatorname{End}_X$ 是 PROP |
| 定义 7.7 | `ch07-def-07-07-prop-algebra` | PROP 代数 |
| 定义 7.8 | `ch07-def-07-08-prop-generated-by-operad` | operad 生成的 PROP |
| 命题 7.9 | `ch07-prop-07-09-generated-prop-existence` | $\operatorname{Prop}(\mathcal O)$ 的存在 |
| 例 7.10 | `ch07-ex-07-10-prop-ass` | $\operatorname{Prop}(\operatorname{Ass})$ |
| 定义 7.11 | `ch07-def-07-11-bialgebra-prop` | 双代数 PROP |
| 命题 7.12 | `ch07-prop-07-12-bialgebra-prop-algebras` | $\mathsf{Bialg}$-代数 |
| 定义 7.13 | `ch07-def-07-13-directed-graphs` | directed $(m,n)$-graphs |
| 定义 7.14 | `ch07-def-07-14-properad` | properad |
| 说明 7.15 | `ch07-note-07-15-properad-vs-prop` | properad 与 PROP 的区别 |
| 命题 7.16 | `ch07-prop-07-16-prop-underlying-properad` | PROP 给出 properad |
| 外部输入定理 7.17 | `ch07-extthm-07-17-free-prop-on-properad` | properad 生成的自由 PROP |
| 定义 7.18 | `ch07-def-07-18-wheeled-properad` | wheeled properad |
| 例 7.19 | `ch07-ex-07-19-endomorphism-wheeled-example` | 有限生成投射模上的 wheeled 结构 |
| 警告 7.20 | `ch07-warn-07-20-trace-not-automatic` | trace 不无条件存在 |
| 练习 7.1 | `ch07-exercise-07-01-interchange-law` | endomorphism PROP 的 interchange law |
| 练习 7.2 | `ch07-exercise-07-02-one-output-operad` | PROP 的一输出 operad |
| 练习 7.3 | `ch07-exercise-07-03-bialgebra-compatibility` | 双代数兼容关系 |
| 练习 7.4 | `ch07-exercise-07-04-properad-connected-graph` | properad 连通图复合 |
| 练习 7.5 | `ch07-exercise-07-05-wheeled-infinite-dimensional-warning` | 无限维 contraction 的限制 |

## 8. 本轮判定

第一至第七章所有已编号声明均已进入稳定 label 表。当前不需要为 operad theory 主体重排编号；最终出版化的下一步是把正文中的散文引用逐步替换为本表 label。核心附录 A/B/H/K/P/U/X 的同格式表见 [LABEL_LEDGER_CORE_APPENDICES.md](LABEL_LEDGER_CORE_APPENDICES.md)。
