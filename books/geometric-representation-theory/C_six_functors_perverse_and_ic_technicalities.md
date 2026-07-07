# 附录 C：六函子、perverse sheaves 与 IC 技术细节

## 本章目标

本附录登记第三章和后续章节使用的六函子、perverse t-structure、IC sheaf 和 decomposition theorem 的技术条件。当前版本是第一版约束表，不替代 BBD 或 Kashiwara-Schapira 的完整证明。

## C.1 Sheaf theory 模型表

**约定 C.1.** 本书允许三种主要 sheaf theory 模型：

| 模型 | 空间 | 系数 | 主要用途 |
| --- | --- | --- | --- |
| Betti constructible sheaves | complex analytic spaces | characteristic zero field | D-module 和 Riemann-Hilbert |
| l-adic sheaves | finite type schemes over suitable fields | $\mathbb Q_\ell$ or extensions | weights、purity、finite field trace |
| mixed/Hodge modules | complex algebraic varieties | rational or complex coefficients | decomposition theorem、purity、Hodge-theoretic refinements |

同一证明中若要跨模型，必须给出 comparison functor 或说明只是在类比层面。

## C.2 Perverse normalization

**约定 C.2.** 若 $X$ 光滑纯维数 $d$，局部系统 $\mathcal L$ 的 perverse normalization 为
$$
\mathcal L[d].
$$
Schubert cell $X_w\simeq\mathbb A^{\ell(w)}$ 上的常值局部系统对应 perverse object
$$
E_{X_w}[\ell(w)].
$$

**定义 C.3.** 对 locally closed embedding $j:U\hookrightarrow X$，middle extension
$$
j_{!*}:\operatorname{Perv}(U)\to\operatorname{Perv}(X)
$$
定义为 $j_!$ 到 $j_\ast$ 的自然 morphism 的 image：
$$
j_{!*}\mathcal F=\operatorname{im}\big({}^pH^0(j_!\mathcal F)\to{}^pH^0(j_\ast\mathcal F)\big).
$$

**外部输入定理 C.4.** $j_{!*}$ 的 image 定义给出无 subobject 或 quotient 支撑在边界 $X\setminus U$ 上的唯一延拓。  
来源：BBD。

## C.3 Decomposition theorem 使用规则

**规则 C.5.** 使用 decomposition theorem 时必须记录：

1. morphism $f:X\to Y$ 是否 proper；
2. 输入对象是否 semisimple perverse sheaf 或光滑源上的 shifted local system；
3. 使用 Betti、l-adic、mixed Hodge module 还是其他版本；
4. 是否需要 relative hard Lefschetz；
5. 是否需要 purity 或 weight argument。

**外部输入定理 C.6.** 对 proper map $f:X\to Y$ 和合适 semisimple perverse input，$Rf_\ast$ 在 derived category 中分解为 shifted semisimple perverse sheaves 的直和。  
本书中 Springer sheaf、KL-IC theorem 和 geometric Satake 都会调用此类结果。

## 本章小结

本附录固定 perverse normalization 和 decomposition theorem 的调用格式。后续每个使用 IC 或 decomposition theorem 的章节都必须指回本附录或附录 D 的 locator。

