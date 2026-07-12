# P0 引用定位批次 3：Factorization homology

本文件记录第三批已精确定位的 P0 外部输入：Ayala--Francis 的 topological manifolds 上 factorization homology。它只覆盖本书第二十章和附录 N/V 中最基础的 homology theory、excision、圆周 Hochschild 计算和边界版本，不覆盖 locally constant factorization algebras 与 $E_n$-algebras 的全部等价、不覆盖 Dunn/Lurie additivity、不覆盖 stratified factorization homology，也不覆盖 Fukaya descent。

## 1. Ayala--Francis topological manifolds

**主来源.** David Ayala and John Francis, “Factorization homology of topological manifolds,” arXiv:1206.5522v6.

**本书对应位置.** 第二十章、附录 N、附录 V、附录 D.7、REFERENCE_LOCATOR_LEDGER 中 P0 “Factorization homology excision”。

### 1.1 总览定理

**定位 AF-0.** arXiv:1206.5522v6, Theorem 1.2.

**本书使用.** 用作第二十章和附录 N 的总览来源：factorization homology 是 manifolds 上的 homology theory；圆周情形给 Hochschild homology；Eilenberg--Steenrod 型公理刻画 factorization homology；并包含 nonabelian Poincare duality 方向。

**允许用法.** AF-0 只作为总览入口。正文证明链引用时应优先使用下列更精确的 AF-1--AF-5，而不是只写 Theorem 1.2。

### 1.2 Excision

**定位 AF-1.** arXiv:1206.5522v6, Lemma 3.18.

**本书使用.** 第二十章和附录 N 中的 collar gluing / excision 公式：
$$
\int_M A \simeq \int_{M_-}A\otimes_{\int_{N\times\mathbb R}A}^{\mathbb L}\int_{M_+}A
$$
或其 infinity-categorical relative tensor product 版本。

**需要同时记录的假设.**

1. $M$ 是来源中允许的 topological manifold，并带有相应 tangential 或 framed structure。
2. 分解必须是 collar-gluing 型，而不是任意闭覆盖。
3. 系数 $A$ 是相应 disk algebra；tensor product 是来源语境中的 derived 或 infinity-categorical relative tensor product。
4. 若出现 boundary、stratified 或 Fukaya 版本，必须按 P1/boundary locator 或 final closure boundary 处理；AF-1 不自动覆盖它们。

**允许用法.** 可支撑附录 N.15 的 factorization homology excision 和第二十章中“factorization homology 满足非交换 Mayer--Vietoris 型切割公式”的外部输入。

### 1.3 圆周与 Hochschild homology

**定位 AF-2.** arXiv:1206.5522v6, Theorem 3.19.

**本书使用.** 第二十章和附录 N 中
$$
\int_{S^1}A\simeq HH_\*(A)
$$
的外部输入，其中 $A$ 是 $E_1$ 或 associative algebra 型系数。

**需要同时记录的假设.**

1. $S^1$ 使用一维 framed manifold 结构。
2. $A$ 作为 $E_1$-algebra 或 associative algebra object 进入 factorization homology。
3. 右侧 Hochschild object 的具体模型需与本书第十一、十二章和附录 E/W 的 Hochschild 符号约定分开核对。

**允许用法.** 可用于圆周 factorization homology 与 cyclic bar / Hochschild homology 的识别。不得把该定理扩张为“任意闭一维空间的普通同调系数公式”。

### 1.4 Eilenberg--Steenrod 型刻画

**定位 AF-3.** arXiv:1206.5522v6, Theorem 3.24.

**本书使用.** 附录 N 中“factorization homology 是由 disk algebra 决定的 homology theory for manifolds”这一结构性外部输入。

**原文结论在本书中的转写.** 在来源假设下，$n$-manifolds 上的 homology theories 与 $\operatorname{Disk}_n$-algebras 之间存在等价。因此 factorization homology 不是任意定义的积分符号，而是由 disk-local 数据和 excision 条件刻画的 functor。

**允许用法.** 可支撑第二十章中 factorization homology 的公理化说明。不得把 AF-3 直接替换为 Costello--Gwilliam 风格的 locally constant factorization algebras on $\mathbb R^n$ 与 $E_n$-algebras 的完整等价；后者按 [FINAL_OPERAD_THEORY_CLOSURE.md](FINAL_OPERAD_THEORY_CLOSURE.md) 作为外部几何/field-theoretic boundary 管理。

### 1.5 带边界版本

**定位 AF-4.** arXiv:1206.5522v6, Theorem 3.26.

**本书使用.** 附录 V 中带边界 factorization homology、区间计算和 module 边界条件的来源之一。

**需要同时记录的假设.**

1. Manifolds with boundary 的 disk category 与无边界情形不同。
2. 边界条件通常由 module 或带颜色的 disk algebra 数据给出。
3. 半空间、角、分层空间和 Fukaya skeletal descent 不由 AF-4 自动推出。

**允许用法.** 可支撑“带边界版本需要额外 boundary/module 数据”的外部输入，并解释为什么无边界 disk normalization 不能直接用于区间、半空间或带角空间。

### 1.6 交换系数计算

**定位 AF-5.** Ayala--Francis, arXiv:1206.5522v6, Proposition 5.1.

**本书使用.** 附录 N 外部输入定理 N.28 和命题 N.20 中，若 $\mathcal V$ 是 tensor-presentable symmetric monoidal infinity-category，$A\in\operatorname{CAlg}(\mathcal V)$，则
$$
\int_M A\simeq M\otimes A
$$
自然成立；右侧使用 $M$ 的 underlying space 对 commutative algebra $A$ 的 tensor。

**需要同时记录的假设.**

1. 系数是严格意义上的 commutative algebra object，而不只是未指定的“足够交换” $E_n$-algebra；
2. $\operatorname{CAlg}(\mathcal V)$ 必须可由 spaces tensor，且底层 symmetric monoidal infinity-category 满足来源的 tensor-presentability；
3. $M$ 的 tangential structure 通过遗忘 commutative algebra 到相应 Disk algebra 使用，公式只依赖 underlying space；
4. 结论位于 $\mathcal V$ 或 $\operatorname{CAlg}(\mathcal V)$ 的等价层，不是先取同调后的群同构。

**允许用法.** 可支撑 $HH(B)\simeq S^1\otimes B$ 的交换系数特化。不得把 $M\otimes A$ 写成普通 singular homology group，除非另行选择链模型并取同调。

## 2. 与本书现有文件的替换规则

| 旧表述 | 替换为 |
| --- | --- |
| Factorization homology 总览 | AF-0，证明链优先用 AF-1--AF-5 |
| Factorization homology excision | AF-1 |
| $\int_{S^1}A\simeq HH_\*(A)$ | AF-2 |
| Factorization homology 的 Eilenberg--Steenrod 型刻画 | AF-3 |
| Manifolds with boundary 的 factorization homology | AF-4 |
| Commutative coefficients $\int_MA\simeq M\otimes A$ | AF-5 |

## 3. 本批次未解决

本批次自身不解决下列项目；后续文件已经分层处理：

1. Costello--Gwilliam 或 Lurie 语境中 locally constant factorization algebras 与 $E_n$-algebras 的完整等价，已由 [FINAL_OPERAD_THEORY_CLOSURE.md](FINAL_OPERAD_THEORY_CLOSURE.md) 关闭为外部几何/field-theoretic boundary；
2. Dunn/Lurie additivity，已由 [P1_REFERENCE_LOCATORS_FINAL_SWEEP.md](P1_REFERENCE_LOCATORS_FINAL_SWEEP.md) 中 DUNN-1 定位；
3. stratified factorization homology 与 defect gluing，已作为几何边界 locator 登记；
4. factorization homology of categories、higher Morita category 或 TFT 型应用，仍属于后续专题；
5. Fukaya category 构造、sectorial descent、skeletal descent 或 wrapped Fukaya gluing，已关闭为外部几何边界；
6. Hochschild chain model 与本书 suspended Hochschild sign convention 的逐项符号核对，属于 production-level sign convention work。

因此 AF-0--AF-5 只用于 topological-manifold factorization homology 及其交换系数计算，不得被读成上述几何/field-theoretic 定理的替代证明。
