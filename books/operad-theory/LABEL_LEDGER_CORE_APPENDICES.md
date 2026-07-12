# 核心附录稳定 label 表

本文件登记附录 A/B/H/K/P/U/X 的稳定交叉引用 label。它不新增数学内容；用途是把 operad theory 主体依赖的集合论、分块、树、colored、低阶计算、PROP/properad 图计算和反例边界固定为可引用目标。

## 0. Label 规则

**规则 0.1（label 形态）.** 附录 label 采用
`appX-kind-X-NN-slug`
形态，其中 `X` 是附录字母，`kind` 取 `def`、`conv`、`prop`、`thm`、`cor`、`ex`、`counterex`、`note`、`warn`、`extthm`、`calc`、`exp` 或 `boundary`。

**规则 0.2（statement type）.** `说明`、`警告`、`解释`、`计算`、`边界` 均是正式可引用目标。它们在最终排版中可以统一样式，但不得无迁移表删除编号。

**规则 0.3（范围）.** 本轮只覆盖 operad theory 主体闭合直接依赖的附录 A/B/H/K/P/U/X。模型范畴、Koszul、同伦转移、dendroidal、factorization 和前沿附录另行编号。

## 1. 附录 A

| 编号 | label | 主题 |
| --- | --- | --- |
| 约定 A.1 | `appA-conv-A-01-universes` | Grothendieck universes |
| 定义 A.2 | `appA-def-A-02-smallness` | $\mathbf{Set}_{\mathcal U}$ 与小性 |
| 约定 A.3 | `appA-conv-A-03-universe-levels` | 基础对象与范畴层级 |
| 命题 A.4 | `appA-prop-A-04-presheaf-size` | presheaf category 的 universe 控制 |
| 定义 A.5 | `appA-def-A-05-finite-sets-groupoid` | $\mathbf{Fin}_{\mathcal U}$ 与 $\mathbf B_{\mathcal U}$ |
| 定义 A.6 | `appA-def-A-06-skeleton-sigma` | $[n]$ 与 $\Sigma_n$ |
| 命题 A.7 | `appA-prop-A-07-finite-set-groupoid-skeleton` | $\mathbf B_{\mathcal U}\simeq\coprod B\Sigma_n$ |
| 推论 A.8 | `appA-cor-A-08-symmetric-sequence-arity-data` | 函子口径与 arity 数据 |
| 命题 A.9 | `appA-prop-A-09-left-to-right-action` | 左作用到右作用的转换 |
| 约定 A.10 | `appA-conv-A-10-finite-set-vs-arity` | 有限集口径与 arity 右作用口径 |
| 定义 A.11 | `appA-def-A-11-coinvariants` | coinvariants |
| 定义 A.12 | `appA-def-A-12-invariants` | invariants |
| 命题 A.13 | `appA-prop-A-13-norm-isomorphism-char-zero` | 特征 $0$ 下 norm 同构 |
| 警告 A.14 | `appA-warn-A-14-positive-characteristic` | 一般底环与正特征风险 |
| 定义 A.15 | `appA-def-A-15-coend` | coend |
| 命题 A.16 | `appA-prop-A-16-coend-over-bg` | $BG$ 上 coend 与 coinvariants |
| 说明 A.17 | `appA-note-A-17-substitution-coend` | 代入乘积中的 coend 公式 |

## 2. 附录 B

| 编号 | label | 主题 |
| --- | --- | --- |
| 定义 B.1 | `appB-def-B-01-partitions` | 有限集分块 |
| 定义 B.2 | `appB-def-B-02-refinement` | 分块 refinement |
| 命题 B.3 | `appB-prop-B-03-flattening` | 分块拉平 |
| 命题 B.4 | `appB-prop-B-04-flattening-associativity` | 分块拉平结合律 |
| 定义 B.5 | `appB-def-B-05-substitution-product` | 代入乘积 |
| 命题 B.6 | `appB-prop-B-06-bijection-action` | 双射诱导的代入映射 |
| 定理 B.7 | `appB-thm-B-07-substitution-associativity` | 代入乘积结合律 |
| 定义 B.8 | `appB-def-B-08-unit-symmetric-sequence` | 单位对称序列 |
| 命题 B.9 | `appB-prop-B-09-unit-isomorphisms` | 左右单位自然同构 |
| 命题 B.10 | `appB-prop-B-10-arity-coinvariants-formula` | arity coinvariants 公式 |
| 反例 B.10.1 | `appB-counterex-B-10-01-partitions-lose-nullary` | 非空分块公式破坏 arity $0$ 左单位 |
| 警告 B.11 | `appB-warn-B-11-action-conventions` | 左右作用方向风险 |
| 定义 B.12 | `appB-def-B-12-tree-substitution` | 平面树替换 |
| 命题 B.13 | `appB-prop-B-13-tree-substitution-associativity` | 树代入结合律 |
| 说明 B.14 | `appB-note-B-14-free-operad-tree-boundary` | 非对称/对称自由 operad 与 dendroidal 树口径 |

## 3. 附录 H

| 编号 | label | 主题 |
| --- | --- | --- |
| 定义 H.1 | `appH-def-H-01-planar-rooted-tree` | 平面有根树 |
| 说明 H.2 | `appH-note-H-02-planar-vs-labels` | 平面结构与叶标号 |
| 命题 H.3 | `appH-prop-H-03-planar-substitution-associativity` | 平面树代入结合律 |
| 定义 H.4 | `appH-def-H-04-s-labelled-tree` | $S$-叶标号有根树 |
| 定义 H.5 | `appH-def-H-05-e-decoration` | $E$-装饰 |
| 定义 H.6 | `appH-def-H-06-tree-groupoid-formula` | 自由对称 operad 的树群胚公式 |
| 命题 H.7 | `appH-prop-H-07-comparison-with-chapter-four` | 与第四章装饰树公式一致 |
| 定义 H.8 | `appH-def-H-08-labelled-tree-grafting` | 叶标号树 grafting |
| 命题 H.9 | `appH-prop-H-09-operad-composition` | 树 grafting 诱导 operad 复合 |
| 命题 H.10 | `appH-prop-H-10-operad-laws` | operad 结合律和单位律 |
| 定理 H.11 | `appH-thm-H-11-free-operad-universal-property` | 自由 operad 泛性质 |
| 命题 H.12 | `appH-prop-H-12-planar-quotient-form` | 非平面公式到平面树商 |
| 警告 H.13 | `appH-warn-H-13-symmetric-tree-quotient` | 对称自由 operad 的重标号商 |
| 定义 H.14 | `appH-def-H-14-moerdijk-weiss-omega-tree` | $\Omega(T)$ |
| 说明 H.15 | `appH-note-H-15-omega-not-free-operad-value` | $\Omega(T)$ 与 $\mathbb F(E)(S)$ 的区别 |
| 命题 H.16 | `appH-prop-H-16-dendroidal-nerve-boundary` | dendroidal nerve 接口 |

## 4. 附录 K

| 编号 | label | 主题 |
| --- | --- | --- |
| 定义 K.1 | `appK-def-K-01-colored-profiles` | colored profiles |
| 命题 K.2 | `appK-prop-K-02-colored-profile-skeleton` | colored 轮廓群胚骨架 |
| 说明 K.3 | `appK-note-K-03-colored-stabilizers` | colored arity 稳定子 |
| 定义 K.4 | `appK-def-K-04-colored-substitution` | colored substitution |
| 命题 K.5 | `appK-prop-K-05-colored-substitution-functor` | colored substitution 的自然性与函子性 |
| 命题 K.6 | `appK-prop-K-06-colored-flattening-associativity` | colored finite-map 拉平与结合相干性 |
| 定义 K.7 | `appK-def-K-07-colored-labelled-tree` | colored 叶标号树 |
| 定义 K.8 | `appK-def-K-08-free-colored-operad` | 自由 colored operad |
| 命题 K.9 | `appK-prop-K-09-free-colored-operad` | 自由 colored operad 泛性质 |
| 定义 K.10 | `appK-def-K-10-morass` | $\operatorname{MorAss}$ |
| 命题 K.11 | `appK-prop-K-11-morass-algebras` | 结合代数同态 |
| 定义 K.12 | `appK-def-K-12-left-module-operad` | $\operatorname{LMod}$ |
| 命题 K.13 | `appK-prop-K-13-left-module-algebras` | 左模结构 |
| 定义 K.14 | `appK-def-K-14-bimodule-operad` | $\operatorname{Bimod}$ |
| 命题 K.15 | `appK-prop-K-15-bimodule-algebras` | 双模结构 |
| 定义 K.16 | `appK-def-K-16-enriched-colored-symseq` | enriched colored symmetric sequence |
| 说明 K.17 | `appK-note-K-17-enriched-examples` | 线性、dg、simplicial、topological colored operad |
| 命题 K.18 | `appK-prop-K-18-enriched-colored-substitution` | enriched colored substitution 结合律 |
| 警告 K.19 | `appK-warn-K-19-enriched-admissibility` | enriched 代数模型结构风险 |
| 外部输入定理 K.20 | `appK-extthm-K-20-colored-admissibility` | colored admissibility |

## 5. 附录 P

| 编号 | label | 主题 |
| --- | --- | --- |
| 反例 P.0 | `appP-counterex-P-00-partition-left-unit-failure` | 非空分块公式的左单位失败 |
| 命题 P.1 | `appP-prop-P-01-endomorphism-associativity` | endomorphism operad 结合律 |
| 命题 P.2 | `appP-prop-P-02-ass-operad-structure` | $\operatorname{Ass}$ 的 operad 结构 |
| 推论 P.3 | `appP-cor-P-03-linear-ass-algebra` | 线性结合代数低阶识别 |
| 命题 P.4 | `appP-prop-P-04-com-algebras` | $\operatorname{Com}$-代数 |
| 说明 P.5 | `appP-note-P-05-lie-characteristic-two` | Lie 反对称与 alternating 边界 |
| 命题 P.6 | `appP-prop-P-06-lie-algebra-operad-algebra` | Lie algebra 给出 Lie operad 代数 |
| 计算 P.7 | `appP-calc-P-07-hochschild-brace` | Hochschild brace 低阶计算 |
| 解释 P.8 | `appP-exp-P-08-strict-operad-map` | strict operad 的 infinity 接口 |
| 说明 P.9 | `appP-note-P-09-cyclic-bar-hochschild` | cyclic bar 与 Hochschild chains |

## 6. 附录 U

| 编号 | label | 主题 |
| --- | --- | --- |
| 命题 U.1 | `appU-prop-U-01-endomorphism-prop-interchange` | endomorphism PROP interchange |
| 命题 U.2 | `appU-prop-U-02-bialgebra-sweedler` | 双代数兼容的 Sweedler 公式 |
| 说明 U.3 | `appU-note-U-03-frobenius-vs-bialgebra` | Frobenius 与双代数 PROP 的差异 |
| 命题 U.4 | `appU-prop-U-04-frobenius-trace` | Frobenius algebra 的 trace 型结构 |
| 命题 U.5 | `appU-prop-U-05-properad-wiring` | properad 复合记录接线图 |
| 说明 U.6 | `appU-note-U-06-operad-tree-special-case` | operad 树是 properad 图的特例 |
| 命题 U.7 | `appU-prop-U-07-forgetting-horizontal-tensor` | 从 PROP 到 properad 的遗忘损失 |
| 命题 U.8 | `appU-prop-U-08-basis-independence` | contraction 基无关性 |
| 警告 U.9 | `appU-warn-U-09-dualizability-needed` | wheeled trace 需要 dualizability |
| 命题 U.10 | `appU-prop-U-10-graph-substitution-associativity` | properad 图替换公理 |
| 说明 U.11 | `appU-note-U-11-operad-prop-comparison` | operad 树代入与 PROP interchange 对照 |

## 7. 附录 X

| 编号 | label | 主题 |
| --- | --- | --- |
| 命题 X.1 | `appX-prop-X-01-free-object-universal-properties` | 自由对象泛性质差异 |
| 例 X.2 | `appX-ex-X-02-tensor-vs-symmetric` | 张量代数与对称代数最小差异 |
| 说明 X.3 | `appX-note-X-03-ass-vs-com` | $\operatorname{Ass}$ 与 $\operatorname{Com}$ 的自由代数差异 |
| 命题 X.4 | `appX-prop-X-04-coinvariants-not-exact` | coinvariants 不 exact |
| 推论 X.5 | `appX-cor-X-05-symmetric-powers-positive-characteristic` | 正特征对称幂风险 |
| 例 X.6 | `appX-ex-X-06-characteristic-two-lie` | 特征 $2$ 下 Lie 约定例子 |
| 说明 X.7 | `appX-note-X-07-lie-convention-boundary` | Lie operad 的 alternating/antisymmetry 边界 |
| 边界 X.8 | `appX-boundary-X-08-operad-map-not-rectification` | operad map 不自动给出 rectification |
| 命题 X.9 | `appX-prop-X-09-circle-factorization-unit` | $A=k$ 的圆周计算 |
| 说明 X.10 | `appX-note-X-10-circle-not-ordinary-homology` | 圆周积分不是普通同调 |
| 命题 X.11 | `appX-prop-X-11-matrix-algebra-hochschild` | 矩阵代数的 Hochschild 计算 |
| 说明 X.12 | `appX-note-X-12-noncommutative-factorization` | factorization homology 的非交换性 |
| 例 X.13 | `appX-ex-X-13-interval-module-regular` | 区间 module 边界条件 |
| 说明 X.14 | `appX-note-X-14-interval-boundary-conditions` | 区间值依赖边界条件 |
| 命题 X.14.1 | `appX-prop-X-14-01-dual-numbers-boundary-modules` | dual numbers 的端点条件改变区间同调 |
| 命题 X.15 | `appX-prop-X-15-symmetric-power-nonzero-class` | $\operatorname{Sym}^p$ 的非零同调类 |
| 推论 X.16 | `appX-cor-X-16-free-cdga-not-left-quillen` | 正特征自由 cdga 不保持准同构 |
| 说明 X.17 | `appX-note-X-17-minimal-algebraic-risk` | 对称 coinvariants 的最小代数风险 |

## 8. 本轮判定

附录 A/B/H/K/P/U/X 的 107 个正式编号项均已进入稳定 label 表。结合 [LABEL_LEDGER_CH01_07.md](LABEL_LEDGER_CH01_07.md)，operad theory 主体及其核心附录的可引用目标已经闭合；下一步是把散文引用替换为这些 label，并继续为模型范畴、Koszul、同伦转移和 infinity-operad 附录做同样处理。
