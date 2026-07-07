# 第十一章：Soergel bimodules、Hodge theory 与 Hecke categorification

## 本章目标

本章介绍 Soergel bimodules 及其对 Hecke algebra 和 Kazhdan-Lusztig theory 的范畴化。它是从 Schubert 几何到纯代数 Hodge theory 的桥梁，也是现代几何表示论中“几何对象的代数替代物”的典型案例。

## 依赖前置知识

需要第四章的 Hecke algebra 和 KL basis，以及基本 graded commutative algebra。

## 11.1 Reflection faithful representation 和多项式环

**定义 11.1.** 令 $(W,S)$ 为 Coxeter system。一个有限维实表示 $\mathfrak h$ 称为 reflection faithful，若每个 reflection 的 fixed space 为 hyperplane，且只有 reflections 具有 hyperplane fixed space。

**定义 11.2.** 令
$$
R=\operatorname{Sym}(\mathfrak h^\ast)
$$
并赋予 grading $\deg(\mathfrak h^\ast)=2$。对 $s\in S$，记 $R^s$ 为 $s$-invariant subring。

**定义 11.3.** simple Soergel bimodule 定义为
$$
B_s=R\otimes_{R^s}R(1),
$$
其中 $(1)$ 是 grading shift。Bott-Samelson bimodule 是形如
$$
B_{s_1}\otimes_R\cdots\otimes_R B_{s_m}
$$
的 graded $R$-bimodule。

**定义 11.4.** Soergel bimodule category $\mathsf{SBim}$ 是 graded $R$-bimodules 的 full additive monoidal Karoubian subcategory，由所有 $B_s$ 和 grading shifts 在有限直和、直和项和 tensor product 下生成。

**命题 11.5.** $\mathsf{SBim}$ 是 monoidal category，单位为 $R$。

**证明.** tensor product over $R$ 给出 $R$-bimodules 的 monoidal structure，单位为 $R$。生成对象 $B_s$ 的 tensor products 仍在由定义封闭的子范畴中；有限直和、直和项和 grading shifts 与 tensor product 相容。因此 $\mathsf{SBim}$ 是 monoidal subcategory。$\square$

## 11.2 Grothendieck group 和 Hecke algebra

**外部输入定理 11.6.** Soergel categorification theorem：split Grothendieck group $[\mathsf{SBim}]$ 与 Hecke algebra $\mathcal H_W$ 同构，且 $[B_s]$ 对应适当归一化的 Kazhdan-Lusztig generator。

**定义 11.7.** 对 $w\in W$，indecomposable Soergel bimodule $B_w$ 是任一 reduced expression 的 Bott-Samelson bimodule 中唯一新的 indecomposable summand，按 shift 归一化。存在性和唯一性属于 Soergel theorem。

**外部输入定理 11.8.** Soergel conjecture：对所有 $w\in W$，
$$
\operatorname{ch}(B_w)=\underline H_w,
$$
其中 $\underline H_w$ 是 KL basis element。

## 11.3 Elias-Williamson Hodge theory

**外部输入定理 11.9.** Elias-Williamson Hodge theory of Soergel bimodules 证明 Soergel conjecture，并推出任意 Coxeter system 的 Kazhdan-Lusztig positivity。其核心包括 Soergel bimodules 的 hard Lefschetz theorem 和 Hodge-Riemann bilinear relations。  
资料入口：Elias-Williamson, arXiv:1212.0791。

**边界说明 11.10.** 该定理不是从 Schubert varieties 的 decomposition theorem 直接推出，因为任意 Coxeter system 不一定有对应 flag variety。Elias-Williamson 的贡献在于构造纯代数的 Hodge 型结构。

## 11.4 与几何 Hecke category 的关系

**外部输入定理 11.11.** 对 Weyl group 情形，Soergel bimodules 可通过 equivariant cohomology of parity sheaves 或 IC sheaves on flag varieties 与几何 Hecke category 联系起来。在特征 $0$ 和合适 parity 条件下，indecomposable Soergel bimodules 对应 Schubert IC sheaves。

**警告 11.12.** 正特征下 IC sheaves、parity sheaves 和 Soergel bimodules 的关系会出现 torsion 和 modular phenomena。不能把特征 $0$ 的 KL positivity 直接转写成 modular character formula。

## 11.5 Character map 的具体形式

**定义 11.13.** 对 $B\in\mathsf{SBim}$ 和 $w\in W$，令 $B_Q=B\otimes_R Q$，其中 $Q$ 是 $R$ 的分式域。Soergel 理论中 $B_Q$ 分解为由 $w$ 标号的图像支撑部分
$$
B_Q\simeq\bigoplus_{w\in W}B_Q^w,
$$
其中右 $R$-作用在 $B_Q^w$ 上经 $w:R\to R$ 扭曲。定义 character
$$
\operatorname{ch}(B)=\sum_{w\in W}\operatorname{grk}_R(B_Q^w)H_w
$$
取值于 Hecke algebra 的适当归一化。

**警告 11.14.** 这个定义依赖 realization 和 grading shift convention。正式使用时必须说明 $H_s$、$\underline H_s$ 与第四章 $T_s,C_s$ 的换元关系。

**命题 11.15.** 对 simple reflection $s$，$B_s=R\otimes_{R^s}R(1)$ 在分式域上分解为两个 rank-one 部分，对应 $e$ 和 $s$。

**证明.** 在 $Q$ 上，$Q$ 是 $Q^s$ 的二次 Galois 扩张，且
$$
Q\otimes_{Q^s}Q\simeq Q\oplus Q_s
$$
作为 $Q$-双模，其中第一项的右作用为通常右乘，第二项的右作用经 $s$ 扭曲。具体同构可由两个嵌入 $Q\otimes_{Q^s}Q\to Q$ 给出：
$$
a\otimes b\mapsto ab,\qquad a\otimes b\mapsto a\,s(b).
$$
这两个映射在分式域上分离二次扩张的两个共轭分支。加入 grading shift 后得到 $B_s$ 的两个分式域支撑部分。$\square$

## 11.6 $A_1$ 计算

**例 11.16.** 令 $W=\{e,s\}$，$R=E[\alpha]$，$\deg\alpha=2$，$s(\alpha)=-\alpha$。则
$$
R^s=E[\alpha^2],\qquad B_s=R\otimes_{R^s}R(1).
$$
作为左 $R$-module，
$$
B_s\simeq R(1)\oplus R(-1).
$$
这来自 $R=R^s\oplus R^s\alpha$。因此 $B_s$ 的 graded rank 为 $v+v^{-1}$，与 $A_1$ Hecke algebra 中 $\underline H_s$ 的 character 相容。

**命题 11.17.** 在 $A_1$ 情形中，Bott-Samelson bimodule $B_s\otimes_R B_s$ 分解为两个 shift 的 $B_s$。

**证明.** 作为 $R^s$-module，$R\simeq R^s\oplus R^s\alpha$。于是
$$
B_s\otimes_R B_s
=R\otimes_{R^s}R\otimes_{R^s}R(2)
\simeq R\otimes_{R^s}(R^s\oplus R^s\alpha)\otimes_{R^s}R(2).
$$
两个直和项分别同构于 $B_s(1)$ 和 $B_s(-1)$，具体 shift 由 $\deg\alpha=2$ 给出。因此
$$
B_s\otimes_R B_s\simeq B_s(1)\oplus B_s(-1).
$$
这与 Hecke 关系 $\underline H_s^2=(v+v^{-1})\underline H_s$ 一致。$\square$

## 11.7 与 Schubert 几何的正式接口

**定义 11.18.** 在 Weyl group 情形，令 $X=G/B$。对 $B$-equivariant complex $\mathcal F$，其 equivariant hypercohomology
$$
\mathbb H_B^\ast(X,\mathcal F)
$$
是 $H_B^\ast(pt)$-双模，其中左右作用来自 $X$ 的两个等价描述 $B\backslash G/B$ 中的左右 $B$-结构。

**外部输入定理 11.19.** 在特征 $0$ 和合适 parity/semisimplicity 假设下，Schubert IC sheaf 的 equivariant hypercohomology 给出 indecomposable Soergel bimodule。

**边界说明 11.20.** 第 11.19 条不是定义，而是几何与代数模型比较定理。若底域、系数或 parity 条件变化，IC sheaf 可能不再对应同一个 indecomposable Soergel bimodule。

## 本章小结

本章定义了 Soergel bimodules 和 Bott-Samelson bimodules，证明其 monoidal 类型闭合，给出 character map 的分式域分解口径和 $A_1$ 计算，并把 categorification theorem、Soergel conjecture 和 Elias-Williamson Hodge theory列为外部输入。该章为后续 modular representation theory 和 parity sheaves 留出接口。

## 练习

**练习 11.1.** 对 $W=\mathbb Z/2=\{e,s\}$，写出 $R^s$ 和 $B_s$。

**练习 11.2.** 证明 $R$ 是 $\mathsf{SBim}$ 的 tensor unit。

**练习 11.3.** 说明为什么任意 Coxeter system 的 Soergel theory 不能依赖 Schubert variety 的存在。

**练习 11.4.** 在 $A_1$ 情形中直接写出 $B_s\otimes_R B_s\simeq B_s(1)\oplus B_s(-1)$ 的两个投影映射。

**练习 11.5.** 比较第四章的 $C_s$ 和本章的 $\underline H_s$，写出一种可能的 $v$-归一化换元。
