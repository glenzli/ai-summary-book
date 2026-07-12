# 第六章：Prismatic $F$-crystals 与 crystalline Galois representations

## 本章目标

本章定义 absolute prismatic site 上的 vector bundles 与 localized
Frobenius structure，并陈述 Bhatt-Scholze 的 crystalline-lattice 分类。
核心边界是：一般 $F$-crystal 只在 prism divisor 外有 Frobenius 同构；
integral Frobenius map 是额外的 effective 条件。

## 依赖前置知识

依赖第二章的 prismatic site、第四章的 crystalline representations、第五章
的 Breuil-Kisin 与 $A_{\inf}$ 背景，以及附录 I 的 crystal/descent 语言。

## 6.1 Crystals on the prismatic site

**定义 6.1.** 令 $X$ 为 $p$-adic formal scheme，并固定 absolute
prismatic site $X_\Delta$；若使用 relative site，则另行写出 base prism。
一个 prismatic crystal $\mathcal E$ 是 $\mathcal O_\Delta$-module sheaf，
使得对任意对应于环方向 $B\to B'$ 的 prism-probe 态射，canonical map
$$
\mathcal E(B,J)\otimes_BB'\longrightarrow\mathcal E(B',J')
$$
为同构。

**定义 6.2.** 若每个 $\mathcal E(B,J)$ 是 finite projective $B$-module，
并且定义 6.1 的 transition isomorphisms 保持这些结构，则称 $\mathcal E$
为 vector bundle crystal 或 finite locally free prismatic crystal。

**警告 6.3.** Sheaf condition 控制 covers 上的 descent；crystal condition
控制所有 probe morphisms 上的 scalar-pullback rigidity。二者逻辑不同。

## 6.2 $F$-crystals

**定义 6.4（prismatic $F$-crystal）.** 令
$\mathcal I_\Delta\subset\mathcal O_\Delta$ 为 prism ideal sheaf，其在
probe $(B,J)$ 上的值为 $J$。一个 vector-bundle-valued prismatic
$F$-crystal 是 finite locally free prismatic crystal $\mathcal E$，配有
localized linearized Frobenius 同构
$$
\varphi_{\mathcal E}^{\mathrm{lin}}:
\phi^*\mathcal E[1/\mathcal I_\Delta]
\xrightarrow{\sim}
\mathcal E[1/\mathcal I_\Delta].
$$
这里 $\phi^*\mathcal E$ 是沿结构层 Frobenius 的 scalar pullback；该
linearized map 等价于 localized $\phi$-semilinear Frobenius。若此同构把
$\phi^*\mathcal E$ 映入 $\mathcal E$，即由 integral map
$$
\phi^*\mathcal E\longrightarrow\mathcal E
$$
诱导，则称该 $F$-crystal effective。

**说明 6.5.** 若某 probe 上 $J=(d)$，则
$[1/\mathcal I_\Delta]$ 在该 probe 上写作 $[1/d]$。换生成元只差单位，
故 localization 不变。Absolute site 上 $J$ 随 probe 变化，不能统一误写为
某个固定 relative base ideal $I$。

**例 6.6.** Tensor unit 是 $\mathcal O_\Delta$ 配 canonical localized
identification
$$
\phi^*\mathcal O_\Delta[1/\mathcal I_\Delta]
\cong\mathcal O_\Delta[1/\mathcal I_\Delta].
$$
这与 Bhatt-Scholze, Definition 4.1（locator `BS-FCRYS`）一致。这里是
module pullback identification，不是说任意 prismatic cohomology complex 的
integral linearized Frobenius 已可逆。

## 6.3 Crystalline representations

**定义 6.7.** 令 $V$ 为 finite-dimensional $\mathbf Q_p$-representation
of $G_K$。若 $V$ 为 $B_{\mathrm{cris}}$-admissible，则称 $V$ crystalline。
若 $T\subset V$ 为 $G_K$-stable finite free $\mathbf Z_p$-lattice，则称
$T$ 为 crystalline lattice。

**警告 6.8.** “$T$ crystalline”是说 $T[1/p]$ crystalline；它不表示
$T$ 由 rational representation 唯一决定。不同 integral lattices 可以有同一
rationalization。

## 6.4 Bhatt-Scholze 分类定理

**外部输入定理 6.9（prismatic $F$-crystal classification）.** 令 $K$ 为
complete discretely valued field of mixed characteristic $(0,p)$，residue
field perfect，$X=\operatorname{Spf}(\mathcal O_K)$。Etale realization 给出
范畴等价
$$
\mathrm{Vect}^{\varphi}(X_\Delta,\mathcal O_\Delta)
\xrightarrow{\sim}
\mathrm{Rep}^{\mathrm{crys}}_{\mathbf Z_p}(G_K),
$$
其中右侧对象为 finite free $\mathbf Z_p$-modules $T$ 配 continuous
$G_K$-action，且 $T[1/p]$ crystalline。来源为 Bhatt-Scholze, Theorem
5.6（locator `BS-FCRYS`）；本书不重证 full faithfulness 或 essential
surjectivity。

**说明 6.10.** 左侧只含 vector-bundle-valued $F$-crystals，不是所有
quasi-coherent sheaves，也不是 arbitrary perfect complexes。右侧是 integral
lattices，不是只有 $\mathbf Q_p$-representations 的 rational category。

**形式推论 6.11.** 若 $\mathcal E$ 为定理 6.9 左侧的 prismatic
$F$-crystal，则其 etale realization rationalized 后是 crystalline
$G_K$-representation。

**证明.** 定理 6.9 的输出是 $T\in
\mathrm{Rep}^{\mathrm{crys}}_{\mathbf Z_p}(G_K)$。按该范畴的定义，
$T[1/p]$ crystalline。证毕。

## 6.5 与 Breuil-Kisin modules 的关系

**外部输入说明 6.12.** 选择 uniformizer 后，在 Breuil-Kisin prism
$(\mathfrak S,(E(u)))$ 上 evaluation，得到 finite projective
Breuil-Kisin module 及其在 prismatic Cech nerve 上的 descent datum。第十二章
定理 12.13 精确记录该接口。只保留 degree-zero evaluation 而丢掉 descent
datum，不足以重建原 prismatic $F$-crystal。

**警告 6.13.** Breuil-Kisin modules、BKF modules、filtered
$\varphi$-modules 与 prismatic $F$-crystals 位于不同底环和范畴。比较
functor 或 classification theorem 不能被当作定义相等。

## 6.6 系数与相对变体

**研究边界 6.14.** 带系数、relative 与 analytic crystalline variants
不进入本章分类链。使用这些变体时必须重新声明 coefficients、site、
Frobenius divisor 和 realization functor。

## 6.7 Tensor operations and duals

**定义 6.15.** 对 finite locally free prismatic crystals
$\mathcal E,\mathcal F$，其 sheaf tensor product 仍可在 probes 上逐项计算：
$$
(\mathcal E\otimes\mathcal F)(B,J)
=\mathcal E(B,J)\otimes_B\mathcal F(B,J).
$$
定义 dual 为
$$
\mathcal E^\vee(B,J)=\operatorname{Hom}_B(\mathcal E(B,J),B).
$$

**命题 6.16.** Tensor product 与 dual 保持 finite locally free prismatic
crystals。若输入带 prismatic $F$-structure，则 tensor product 与 dual 也
自然带 prismatic $F$-structure。

**证明.** 对 $B\to B'$，crystal condition 给出
$$
\mathcal E(B')\cong\mathcal E(B)\otimes_BB',\qquad
\mathcal F(B')\cong\mathcal F(B)\otimes_BB'.
$$
Sheaf tensor product 与 sheaf internal Hom 可在 finite locally free objects
上局部逐项计算；在局部平凡化后上述 formulas 显然满足 descent，再由
crystal transitions 黏合。Tensor associativity 给出
$$
(\mathcal E\otimes\mathcal F)(B')
\cong(\mathcal E(B)\otimes_B\mathcal F(B))\otimes_BB'.
$$
Finite projectivity 给出 dual base change
$$
\operatorname{Hom}_B(M,B)\otimes_BB'
\cong\operatorname{Hom}_{B'}(M\otimes_BB',B').
$$
在 invert $\mathcal I_\Delta$ 后定义
$$
\varphi_{\mathcal E\otimes\mathcal F}^{\mathrm{lin}}
=\varphi_{\mathcal E}^{\mathrm{lin}}
\otimes\varphi_{\mathcal F}^{\mathrm{lin}};
$$
它是同构。Dual 的 Frobenius 是
$(\varphi_{\mathcal E}^{\mathrm{lin}})^{-1}$ 的 dual；finite local freeness
保证 dual 与 $\phi^*$ 相容。各构造与 probe transitions 相容，故给出
crystals。证毕。

**说明 6.17.** 上述 dual statement 针对 localized $F$-structure。若限制到
effective objects，还需检查 dual 的 integral lattice 是否仍落在所选
effective cone；这不是 localized isomorphism 的形式后果。

## 本章小结

Prismatic $F$-crystal 的基础 datum 是 vector bundle 加 prism divisor 外的
Frobenius isomorphism；effective 是额外积分条件。Bhatt-Scholze 的外部输入
把这些对象与 crystalline $\mathbf Z_p$-lattices 等价。Breuil-Kisin
evaluation 只有连同 descent datum 才保留坐标无关信息。

## 练习

**练习 6.1.** 写出 prismatic crystal 的 transition isomorphism，并说明它
与 sheaf restriction map 的差异。

**练习 6.2.** 在 probe ideal $J=(d)$ 时，把定义 6.4 的 localization 写成
$[1/d]$，并说明换生成元不改变它。

**练习 6.3.** 解释定理 6.9 为什么是范畴等价，而不是 cohomology
comparison theorem。
