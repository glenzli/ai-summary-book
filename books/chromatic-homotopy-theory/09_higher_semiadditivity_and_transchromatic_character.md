# 第九章：Higher semiadditivity 与 transchromatic character

## 本章目标

本章把 higher semiadditivity 作为 chromatic theory 的现代结构工具引入。我们只证明稳定范畴中有限和余有限的基础形式事实；$K(n)$-local 和 $T(n)$-local 范畴的 higher semiadditivity 作为外部输入。transchromatic character 作为高度之间传递信息的接口处理。

## 依赖前置知识

需要第五章的 chromatic localization、第六章的 $K(n)$-local category 和第七章的 redshift/telescope 分层。需要熟悉 finite group actions、homotopy fixed points、homotopy orbits 和 norm map 的基本语言。

## 9.1 Semiadditivity 的最低阶形式

**定义 9.1.** 一个 infinity-范畴 $\mathcal C$ 称为 0-semiadditive，若它有有限乘积和有限余积，并且对任意有限集合 $I$ 和对象族 $\{X_i\}_{i\in I}$，自然映射
$$
\coprod_{i\in I}X_i\longrightarrow \prod_{i\in I}X_i
$$
是等价。

**命题 9.2.** 任意稳定 infinity-范畴若有有限直和，则它是 0-semiadditive。

**证明.** 稳定 infinity-范畴中有限乘积和有限余积相同，记为 biproduct。对两个对象 $X,Y$，cofiber/fiber 公理给出对象 $X\oplus Y$，同时满足 product 和 coproduct 的泛性质。有限集合情形由归纳得到，空集合对应零对象。证毕。

**定义 9.3.** 对有限群 $G$ 作用在稳定 infinity-范畴对象 $X$ 上，homotopy orbits 和 homotopy fixed points 分别记作
$$
X_{hG},\qquad X^{hG}.
$$
norm map 是自然映射
$$
N:X_{hG}\to X^{hG}.
$$

**定义 9.4.** 稳定 infinity-范畴 $\mathcal C$ 称为 1-semiadditive，若对所有有限群 $G$ 和所有 $G$-对象 $X$，norm map $X_{hG}\to X^{hG}$ 是等价。

**警告 9.5.** 1-semiadditivity 不是任意稳定范畴的形式性质。例如普通谱范畴中 norm map 的 cofiber 是 Tate construction $X^{tG}$，一般不为零。

## 9.2 Higher semiadditivity

**定义 9.6.** 设 $m\ge0$。一个空间 $A$ 称为 $m$-finite，若它有有限多个连通分支，且对每个基点 $a$，同伦群 $\pi_i(A,a)$ 是有限群，并且当 $i>m$ 时为零。

**定义 9.7.** 一个 presentable infinity-范畴 $\mathcal C$ 称为 $m$-semiadditive，若对任意 $m$-finite 空间 $A$ 和任意函子 $F:A\to\mathcal C$，自然 norm map
$$
\operatorname*{colim}_{a\in A}F(a)\longrightarrow \operatorname*{lim}_{a\in A}F(a)
$$
是等价。若对所有 $m$ 成立，则称为 $\infty$-semiadditive。

**例 9.8.** 当 $m=0$ 时，$m$-finite 空间等价于有限集合，定义 9.7 退化为定义 9.1。若 $A=BG$ 且 $G$ 为有限群，则 $1$-semiadditivity 包含定义 9.4。

**外部输入定理 9.9 (Hopkins-Lurie, Carmeli-Schlank-Yanovski).** $K(n)$-local spectra 和 $T(n)$-local spectra 的 infinity-范畴在适当意义下是 $\infty$-semiadditive。不同版本的定理对 $K(n)$、$T(n)$、height 和 presentability 的表述不同，调用前必须定位具体版本。

**使用限制 9.10.** 定理 9.9 不允许推出普通谱范畴 $\mathbf{Sp}$ 是 $\infty$-semiadditive，也不允许把所有 Tate constructions 判为零。它只在指定 chromatic local category 中使用。

## 9.3 Semiadditive height

**定义 9.11.** 在 $\infty$-semiadditive chromatic 范畴中，可定义 $\pi$-finite spaces 的 semiadditive cardinality。对 $m$-finite 空间 $A$，记
$$
|A|_{\mathcal C}\in \pi_0\operatorname{End}_{\mathcal C}(\mathbbm 1)
$$
为常值图的 colimit-limit comparison 给出的标量。

**边界 9.12.** Semiadditive height 是通过这些 cardinalities 和其消失/可逆行为测量 chromatic height 的现代工具。完整定义依赖 CSY 和后续文献，本书当前只登记接口。

**命题 9.13.** 若 $\mathcal C$ 是 0-semiadditive，则有限集合 $I$ 的 cardinality 作用在单位对象上等于 $|I|$ 次单位自同态之和。

**证明.** 常值图 $I\to\mathcal C$ 取值 $\mathbbm 1$。其 colimit 和 limit 都是 $|I|$ 个 $\mathbbm 1$ 的 biproduct。comparison map 在矩阵表示下是所有分量为 identity 的对角求和，诱导的单位端标量是 $|I|\cdot \operatorname{id}_{\mathbbm 1}$。证毕。

## 9.4 Transchromatic character

**定义 9.14.** Transchromatic character theory 指把高度 $n$ 的 Morava E-theory 信息通过 character map 转移到较低高度或混合高度对象上的理论。典型输入是有限群或 $\pi$-finite 空间的 equivariant/cohomological 信息。

**外部输入 9.15.** Hopkins-Kuhn-Ravenel generalized character theory 给出 Morava E-theory 在有限群 classifying spaces 上的 character map。后续 transchromatic character theory 将其推广到高度变化和更高范畴版本。

**边界 9.16.** Ben-Moshe 2024 的 transchromatic higher semiadditivity 结果说明 categorified transchromatic character 与 semiadditive integration 相容。当前版本把它作为前沿输入，进入证明链前需 locator。

## 9.5 与 redshift 的关系

**解释 9.17.** Redshift 与 higher semiadditivity 的交汇点是：代数 $K$-theory 将高度 $n$ 的对象推向高度 $n+1$，而 semiadditivity 提供了在高度层之间传递 finite homotopy type 积分的机制。Ben-Moshe 2025 的高度归纳证明正是利用这类联系。

**警告 9.18.** 这不是说 redshift 定理自动推出全部 higher semiadditivity。每个证明路线都有独立输入，包括 algebraic K-theory、descent、Tate vanishing 和具体高度归纳步骤。

## 9.6 Norm、Tate 与半加性

**定义 9.19.** 对有限群 $G$ 作用的对象 $X$，Tate object 定义为
$$
X^{tG}=\operatorname{cofib}(X_{hG}\xrightarrow{N}X^{hG}).
$$

**命题 9.20.** 对有限群 $G$，norm map $N:X_{hG}\to X^{hG}$ 是等价，当且仅当 $X^{tG}\simeq0$。

**证明.** $X^{tG}$ 按定义是 $N$ 的 cofiber。稳定范畴中映射为等价当且仅当其 cofiber 为零。证毕。

**推论 9.21.** 1-semiadditivity 等价于所有有限群作用对象的 Tate object 消失。

**证明.** 1-semiadditivity 的定义要求所有有限群 $G$ 和所有 $G$-对象 $X$ 的 norm map 为等价。由命题 9.20，这等价于所有对应 Tate objects 为零。证毕。

**警告 9.22.** Chromatic Tate vanishing theorem 的具体形式依赖局部化。不能把 $K(n)$-local Tate vanishing、$T(n)$-local Tate vanishing 和 ordinary Tate construction 混用。

## 9.7 Semiadditive cardinality 的低阶检查

**例 9.23.** 在 0-semiadditive 范畴中，有限集合 $I$ 的 cardinality 是整数 $|I|$ 作用在单位对象上。

**证明.** 见命题 9.13。若 $I=\{1,\ldots,r\}$，常值图的 colimit-limit comparison 是 $r$ 个单位对象 biproduct 上的求和-对角复合，得到 $r\cdot\operatorname{id}_{\mathbbm 1}$。证毕。

**边界 9.24.** 对 $A=BG$，semiadditive cardinality 与 $|G|^{-1}$ 型现象相关，但在 chromatic local categories 中不能按普通有理数解释。需要 higher semiadditive cardinality 的正式理论。

## 本章小结

Higher semiadditivity 把有限集合的 biproduct 现象推广到 $\pi$-finite spaces 的 indexed limit/colimit 比较。普通稳定范畴只自动具有 0-semiadditivity；chromatic local categories 的高阶版本是深层外部输入。Transchromatic character 将高度之间的信息传递与 semiadditive integration 联系起来，是现代 chromatic theory 的核心前沿之一。

## 练习

**练习 9.1.** 在稳定 infinity-范畴中证明三对象有限直和同时满足 product 和 coproduct 泛性质。

**练习 9.2.** 对有限群 $G$ 的平凡作用，写出 $X_{hG}$ 和 $X^{hG}$ 的定义，并说明 norm map 与 $BG$ 的 colimit/limit comparison 的关系。

**练习 9.3.** 解释为什么 Tate construction 的非零性会阻碍 1-semiadditivity。

**练习 9.4.** 查阅 Hopkins-Kuhn-Ravenel character theory 的一个低高度例子，说明它与普通复表示 character 的相似点和差异。
