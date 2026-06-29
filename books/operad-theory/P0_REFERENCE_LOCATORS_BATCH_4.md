# P0 引用定位批次 4：Koszul duality 与 classical bar-cobar core

本文件记录第四批已精确定位的 P0 外部输入：Ginzburg--Kapranov 的 quadratic operad Koszul duality。它覆盖 classical quadratic operad 的 Koszul 定义、对偶 Koszul 性、生成函数反函数关系、Koszul complex 判别、$\operatorname{As}$、$\operatorname{Com}$、$\operatorname{Lie}$ 的 Koszul 性，以及 Ginzburg--Kapranov dg-duality 下的 homotopy $\mathcal P$-algebra 入口。

本批次不完全覆盖 Loday--Vallette/Fresse/Hinich 现代记号中的 conilpotent cooperad bar-cobar counit/unit、$\Omega B\mathcal P\to\mathcal P$、$\Omega\mathcal P^¡\to\mathcal P$ 的模型范畴 cofibrant resolution 版本。Fresse/Hinich 的 modern cobar/cofibrant replacement 后续由 [P0_REFERENCE_LOCATORS_BATCH_5.md](P0_REFERENCE_LOCATORS_BATCH_5.md) 部分覆盖；Loday--Vallette/Fresse 书本口径已由 [FINAL_OPERAD_THEORY_CLOSURE.md](FINAL_OPERAD_THEORY_CLOSURE.md) 关闭为 convention/bibliography production work。

## 1. Ginzburg--Kapranov classical Koszul duality

**主来源.** Victor Ginzburg and Mikhail Kapranov, “Koszul duality for operads,” *Duke Mathematical Journal* 76 (1994), no. 1, 203--272；arXiv:0709.1228.

**本书对应位置.** 第八、九、十章，附录 I/Q/L，附录 D.2、D.12.1，REFERENCE_LOCATOR_LEDGER 中 P0 “Koszul 判别”和 P0 “Bar-cobar resolution”的 classical core。

### 1.1 Koszul operad 的 classical 定义

**定位 GK-1.** Ginzburg--Kapranov, arXiv:0709.1228, Definition 4.1.3.

**本书使用.** 第八章和附录 I 中“quadratic operad 的 Koszul 性不是二次对偶的存在性，而是 canonical morphism 的 quasi-isomorphism 条件”的 classical 来源。

**需要同时记录的假设.**

1. 来源使用 admissible dg-operads、semisimple unary algebra 和 finite-dimensional duality 的语境。
2. 来源中的 $D(\mathcal P)$、$\mathcal P^!$、determinant twist 与本书的 $\mathcal P^¡$、operadic suspension、homological grading 需要逐项转换。
3. 本定位只给 classical quadratic definition；若正文使用 conilpotent cooperad 的现代 $\Omega\mathcal P^¡$ 记号，必须说明 convention translation。

### 1.2 对偶 Koszul 性与生成函数关系

**定位 GK-2.** Ginzburg--Kapranov, arXiv:0709.1228, Proposition 4.1.4.

**本书使用.** 第八章外部输入定理 8.19 中“若 $\mathcal P$ Koszul，则 $\mathcal P^!$ Koszul，并且生成函数满足 Ginzburg--Kapranov 反函数关系”的来源。

**需要同时记录的假设.**

1. 生成函数使用来源的 Euler characteristic 和 admissible operad 约定。
2. 单色情形的公式是多色 formal map identity 的特例。
3. 本书若写 $g_{\mathcal P}(-g_{\mathcal P^!}(-t))=t$ 或等价公式，必须注明符号、悬挂和变量 convention。

### 1.3 Koszul complex 判别

**定位 GK-3.** Ginzburg--Kapranov, arXiv:0709.1228, Theorem 4.1.13.

**本书使用.** 第八章定义 8.16、附录 I 定义 I.17--定义 I.18、附录 Q 的 Koszul complex 计算中，“Koszul 性可由 Koszul complexes 的 exactness 判别”的 classical 来源。

**需要同时记录的假设.**

1. 来源中的 Koszul complex 写作 $\mathcal P((\mathcal P^!)^\vee)$ 型复形；本书写作 twisted composite product 时必须给出转换。
2. Exactness 判别位于 quadratic operad 语境，不自动推广到非二次、curved、inhomogeneous 或 unital 变体。
3. 左/右 Koszul complex 和 augmentation 到单位的表述，需要按本书附录 I 的 convention 检查。

### 1.4 Free algebra homology 判别与三个经典 operad

**定位 GK-4.** Ginzburg--Kapranov, arXiv:0709.1228, Theorem 4.2.5.

**定位 GK-5.** Ginzburg--Kapranov, arXiv:0709.1228, Corollary 4.2.7.

**本书使用.** 第八章外部输入定理 8.18 中 $\operatorname{Ass}$、$\operatorname{Com}$、$\operatorname{Lie}$ 的 Koszul 性；附录 Q 中 free algebra homology 与 Koszul complex 的边界说明。

**需要同时记录的假设.**

1. GK-4 通过 free $\mathcal P$-algebra 的 $\mathcal P$-homology 消失给出判别，不能替代所有 Gröbner/PBW/distributive law 判别。
2. GK-5 支撑的是 $\operatorname{As}$、$\operatorname{Com}$、$\operatorname{Lie}$ 的 classical Koszul 性；含单位、正特征中额外 divided power 或非齐次版本仍需另行说明。
3. 本书在第十章把 $A_\infty$、$L_\infty$、$C_\infty$ 作为 homotopy operad 时，还需要 GK-7 或现代 bar-cobar 定位连接到 resolution。

### 1.5 Ginzburg--Kapranov dg-duality 与 homotopy $\mathcal P$-algebra

**定位 GK-6.** Ginzburg--Kapranov, arXiv:0709.1228, Theorem 3.2.16.

**定位 GK-7.** Ginzburg--Kapranov, arXiv:0709.1228, Section 4.2.12.

**本书使用.** 第九、十章和附录 I/L 中“classical Koszul operad 的 homotopy $\mathcal P$-algebra 可由对偶 dg-operad 控制”的来源之一。

**需要同时记录的假设.**

1. GK-6 是来源中 dual dg-operad 构造的双对偶 quasi-isomorphism，不是现代模型范畴中任意 cofibrant replacement 的一般定理。
2. GK-7 给出 Koszul quadratic $\mathcal P$ 情形中 $D(\mathcal P^!)\to\mathcal P$ 的 quasi-isomorphism，并据此定义 homotopy $\mathcal P$-algebra。
3. 把 GK-7 转写为本书 $\mathcal P_\infty=\Omega\mathcal P^¡$ 需要 determinant twist、operadic suspension、homological/cohomological grading、finite type duality 的 convention crosswalk。

## 2. 与本书现有文件的替换规则

| 旧表述 | 替换为 |
| --- | --- |
| Ginzburg--Kapranov Koszul operad definition | GK-1 |
| Koszul dual of a Koszul operad is Koszul | GK-2 |
| Ginzburg--Kapranov generating series identity | GK-2，需说明生成函数和符号 convention |
| Koszul complex exactness criterion | GK-3 |
| Free $\mathcal P$-algebra homology criterion | GK-4 |
| $\operatorname{As}$、$\operatorname{Com}$、$\operatorname{Lie}$ are Koszul | GK-5 |
| Classical dg-dual/bar-cobar core behind homotopy $\mathcal P$-algebras | GK-6 + GK-7 |

## 3. 本批次未解决

本批次不解决：

1. Loday--Vallette 现代写法中 $\Omega\mathcal P^¡\to\mathcal P$ 的精确定理 locator；
2. 一般 augmented operad 的 $\Omega B\mathcal P\to\mathcal P$ bar-cobar counit quasi-isomorphism；
3. Conilpotent cooperad $\mathcal C$ 与 augmented operad $\mathcal P$ 之间 twisting morphism 的现代 model-category acyclicity theorem；
4. Hinich/Fresse 语境中 cofibrant resolution、admissibility 和 algebra homotopy theory 的 theorem locator；
5. 与本书附录 E/W/J/L 的 full sign convention crosswalk；
6. Curved、unital、inhomogeneous 或 positive-characteristic divided-power operadic Koszul duality。

这些不由 GK-1--GK-7 单独推出；后续使用时应引用对应 modern locator 或按 [FINAL_OPERAD_THEORY_CLOSURE.md](FINAL_OPERAD_THEORY_CLOSURE.md) 的 convention/boundary 规则处理。
