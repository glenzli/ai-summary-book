# 第十二章：Breuil-Kisin、Breuil-Kisin-Fargues modules 与 lattices

## 本章目标

本章把积分 $p$-adic Hodge theory 中的 module-theoretic 对象与 prismatic cohomology 联系起来。重点是区分 Breuil-Kisin modules、Breuil-Kisin-Fargues modules、filtered $\varphi$-modules、Galois lattices 和 prismatic $F$-crystals。

## 依赖前置知识

依赖第四章 Galois representations、第五章 BMS、第二章 Breuil-Kisin prism 和第六章 prismatic $F$-crystals。

## 12.1 Breuil-Kisin prism

**约定 12.1.** 令 $K/\mathbf Q_p$ 为 complete discretely valued field，剩余域 $k$ 完美，uniformizer 为 $\pi$。令
$$
\mathfrak S=W(k)[[u]],\qquad \phi(u)=u^p,
$$
并令 $E(u)$ 为 $\pi$ 的 Eisenstein polynomial。

**定义 12.2.** Breuil-Kisin prism 是
$$
(\mathfrak S,(E(u))).
$$

**说明 12.3.** 该 prism 依赖 uniformizer $\pi$。改变 $\pi$ 会改变坐标 presentation，但相应理论有比较。正式使用时必须声明所选 $\pi$。

## 12.2 Breuil-Kisin modules

**定义 12.4（基本模型）.** 一个 Breuil-Kisin module 的基本模型是有限生成 $\mathfrak S$-module $M$，配有 Frobenius-semilinear map
$$
\varphi_M:\phi^\ast M\to M
$$
使得 $\varphi_M$ 在 invert $E(u)$ 后为同构。

**警告 12.5.** 不同文献对 Breuil-Kisin module 还会加入 height condition、projectivity、torsion-free、filtered data 或 Galois descent data。本书的定义 12.4 是最小模型，不替代具体定理中的精确定义。

**命题 12.6.** 若 $M$ 为 finite projective $\mathfrak S$-module 且 $\varphi_M[1/E(u)]$ 为同构，则 $(M,\varphi_M)$ 在 generic locus $\operatorname{Spec}\mathfrak S[1/E(u)]$ 上是 Frobenius module。

**证明.** Localizing at $E(u)$ gives
$$
\phi^\ast M[1/E(u)]\to M[1/E(u)].
$$
按假设该映射为同构。Finite projectivity 在局部化下保持，因此得到带 Frobenius isomorphism 的 finite projective module。证毕。

## 12.3 Breuil-Kisin-Fargues modules

**定义 12.7（工作定义）.** 令 $A_{\inf}$ 为 perfectoid base 的 Fontaine ring。Breuil-Kisin-Fargues module 的工作模型是有限呈示 $A_{\inf}$-module $M$，配有 Frobenius-semilinear map
$$
\varphi_M:\phi^\ast M[1/\xi]\xrightarrow{\sim}M[1/\xi],
$$
并满足适当 projectivity/torsion 条件。

**警告 12.8.** 定义 12.7 是教材工作定义。完整定义需要 Fargues 的 admissibility and vector bundle on Fargues-Fontaine curve 口径；本书只在 BMS integral cohomology 的输出对象层面使用。

## 12.4 Cohomology 输出的 module 结构

**外部输入定理 12.9.** 对 proper smooth formal scheme $\mathfrak X/\mathcal O_C$，BMS complex 的 cohomology 在适当 finiteness 假设下给出 Breuil-Kisin-Fargues module 型对象。

**外部输入定理 12.10.** 对 proper smooth formal scheme $\mathfrak X/\mathcal O_K$，Breuil-Kisin prism 上的 prismatic cohomology 与 Breuil-Kisin cohomology 比较，其 cohomology groups 携带 Breuil-Kisin module 型结构。

**说明 12.11.** 这些 module structures 是 integral comparison 的关键输出。它们比 rational filtered $\varphi$-module 包含更多 torsion 和 lattice 信息。

## 12.5 与 Galois lattices 的接口

**外部输入定理 12.12.** Prismatic $F$-crystals over $\mathcal O_K$ 与 crystalline $G_K$-representations 中的 $\mathbf Z_p$-lattices 范畴等价。

**命题 12.13.** 定理 12.12 与 Breuil-Kisin prism 的局部 presentation 相容时，Breuil-Kisin module 给出 crystalline lattice 的坐标化描述。

**证明草图.** Breuil-Kisin prism 给出 $\mathcal O_K$ 的 prismatic thickening presentation。Prismatic $F$-crystal 在该 prism 上的取值给出带 Frobenius 的 $\mathfrak S$-module。外部输入的 classification theorem 把该 crystal 对应到 crystalline Galois lattice。故 Breuil-Kisin module 是该 lattice 的 prism 坐标表达。完整证明需要 comparison functor 的 fully faithfulness 和 essential surjectivity。证毕。

## 12.6 Height 条件的代数读法

**定义 12.14（工作定义）.** 对 Breuil-Kisin module $(M,\varphi_M)$，若 $M$ finite projective，且 cokernel of
$$
\varphi_M:\phi^\ast M\to M
$$
被 $E(u)^h$ 杀死，则称其 height at most $h$。

**命题 12.15.** 若 $M=\mathfrak S e$ 且 $\varphi(1\otimes e)=E(u)^h e$，则 $M$ 的 height at most $h$。

**证明.** 映射 $\mathfrak S e\to\mathfrak S e$ 为乘以 $E(u)^h$。其 cokernel 为 $\mathfrak S/(E(u)^h)$，被 $E(u)^h$ 杀死。证毕。

**警告 12.16.** Height convention 在文献中可能与 Hodge-Tate weights normalization 相连。正式比较 Galois representations 时必须说明 normalization。

## 本章小结

Breuil-Kisin 和 Breuil-Kisin-Fargues modules 是 integral $p$-adic Hodge theory 中承载 lattice 信息的线性代数对象。Prismatic theory 通过 prism presentation 和 $F$-crystals 解释这些对象，但不能把它们与 Galois representations、filtered $\varphi$-modules 或 prismatic crystals 混为一个定义。

## 练习

**练习 12.1.** 说明定义 12.4 中 invert $E(u)$ 的作用。

**练习 12.2.** 比较 Breuil-Kisin module 与 Breuil-Kisin-Fargues module 的底环和 invert 元素。

**练习 12.3.** 解释为什么 lattice 信息不能只由 rational filtered $\varphi$-module 完全表达。
