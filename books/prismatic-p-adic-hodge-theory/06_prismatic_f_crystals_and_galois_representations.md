# 第六章：Prismatic $F$-crystals 与 crystalline Galois representations

## 本章目标

本章把 prismatic site 上的线性对象与 classical crystalline Galois representations 联系起来。核心定理是 Bhatt-Scholze 的范畴等价：$\mathcal O_K$ 上的 prismatic $F$-crystals 对应于 crystalline $G_K$-representations 中的 lattices。该定理是外部输入；本章负责固定定义和接口。

## 依赖前置知识

依赖第二章的 prismatic site，第四章的 crystalline representations，第五章的 Breuil-Kisin 和 $A_{\inf}$ 背景。需要熟悉 crystals、finite locally free modules 和 Frobenius-semilinear maps。

## 6.1 Crystals on the prismatic site

**定义 6.1.** 令 $X$ 为 $p$-adic formal scheme。一个 prismatic crystal $\mathcal E$ 是在 absolute 或 relative prismatic site 上的 $\mathcal O_\Delta$-module，使得对任意 prism 态射
$$
(B,J)\to(B',J')
$$
诱导的 pullback map
$$
\mathcal E(B,J)\otimes_B B'\longrightarrow \mathcal E(B',J')
$$
为同构。

**定义 6.2.** 若对每个对象 $(B,J)$，$\mathcal E(B,J)$ 是 finite projective $B$-module，并且 crystal transition maps 与 finite projective structure 相容，则称 $\mathcal E$ 为 vector bundle crystal 或 finite locally free prismatic crystal。

**警告 6.3.** Crystal 条件不是 sheaf 条件的重复。Sheaf 条件控制覆盖下降；crystal 条件控制 nilpotent/prismatic thickening 方向上的刚性。

## 6.2 $F$-crystals

**定义 6.4.** 令 $(A,I)$ 为 prism。一个 prismatic $F$-crystal 是 prismatic crystal $\mathcal E$，配有 Frobenius-semilinear map
$$
\varphi_{\mathcal E}:\phi^\ast\mathcal E\longrightarrow \mathcal E
$$
使得在 invert $I$ 后成为同构：
$$
\varphi_{\mathcal E}[1/I]:\phi^\ast\mathcal E[1/I]\xrightarrow{\sim}\mathcal E[1/I].
$$

**说明 6.5.** 若 $I$ 由 $d$ 生成，则“invert $I$”可写作 invert $d$。未选定生成元时应写成对 Cartier divisor complement 的局部化。

**例 6.6.** 单位 crystal $\mathcal O_\Delta$ 是 prismatic $F$-crystal 的基本测试对象：只有在所采用的 prismatic $F$-crystal convention 中，结构 Frobenius 的 linearization 在 invert $I$ 后为同构时，$\mathcal O_\Delta$ 才配成单位 $F$-crystal。最终版应把这一点与 Bhatt-Scholze 的定义逐项对齐。

## 6.3 Crystalline representations

**定义 6.7.** 令 $V$ 为 $G_K$ 的 $p$-adic representation。若 $V$ 为 $B_{\mathrm{cris}}$-admissible，则称 $V$ 为 crystalline representation。若 $T\subset V$ 为 $G_K$-stable $\mathbf Z_p$-lattice，则称 $T$ 为 crystalline lattice。

**警告 6.8.** “$T$ crystalline”严格说是指 $T\otimes_{\mathbf Z_p}\mathbf Q_p$ crystalline，且 $T$ 是其中一个 $G_K$-stable lattice。积分 lattice 层面的信息比 rational representation 更细。

## 6.4 Bhatt-Scholze 分类定理

**外部输入定理 6.9（prismatic $F$-crystal classification）.** 令 $K$ 为 complete discretely valued field of mixed characteristic $(0,p)$，剩余域完美。则 $\mathcal O_K$ 上有限局部自由的 prismatic $F$-crystals（即 vector-bundle-valued crystals with Frobenius）范畴等价于 crystalline $G_K$-representations 中的 $\mathbf Z_p$-lattices 范畴。

**说明 6.10.** 定理 6.9 的左侧是 prismatic site 上的有限局部自由几何线性对象，右侧是 Galois representation 的积分 lattice。它不是单个 cohomology group 的比较，也不是任意 quasi-coherent $F$-crystal 的分类，而是有限秩向量丛型对象的范畴等价。

**形式推论 6.11.** 若 $\mathcal E$ 为 prismatic $F$-crystal，则其 generic fibre 对应的 $p$-adic representation 是 crystalline。

**证明.** 定理 6.9 给出范畴等价，右侧对象按定义是 crystalline representation 的 lattice。将 lattice 张量 $\mathbf Q_p$ 得到 crystalline representation。证毕。

## 6.5 与 Breuil-Kisin modules 的关系

**外部输入说明 6.12.** 在选择 uniformizer 并使用 Breuil-Kisin prism $(\mathfrak S,(E(u)))$ 后，prismatic $F$-crystals 可与 Breuil-Kisin module 型对象比较。该比较依赖选择 $\pi$ 和 Eisenstein polynomial，因此不是完全坐标无关的表述。

**警告 6.13.** Breuil-Kisin modules、Breuil-Kisin-Fargues modules、filtered $\varphi$-modules 和 prismatic $F$-crystals 是不同范畴中的对象。它们之间有比较和分类定理，但不能把定义互相替代。

## 6.6 系数与相对变体

**研究边界 6.14.** 2023 以后出现了带系数、relative 和 analytic crystalline 方向的 prismatic $F$-crystal 研究，例如 analytic crystalline representations、$E$-crystalline variants 和 coefficients in $A_{\inf}$-cohomology。本书只把这些方向列为研究边界，不纳入基础定理链。

## 6.7 Tensor operations and duals

**定义 6.15.** 若 $\mathcal E$、$\mathcal F$ 是 prismatic crystals，定义
$$
(\mathcal E\otimes\mathcal F)(B,J)=\mathcal E(B,J)\otimes_B\mathcal F(B,J).
$$
若 $\mathcal E$ finite locally free，定义 dual crystal
$$
\mathcal E^\vee(B,J)=\operatorname{Hom}_B(\mathcal E(B,J),B).
$$

**命题 6.16.** 若 $\mathcal E$ 和 $\mathcal F$ 是 prismatic crystals，则 $\mathcal E\otimes\mathcal F$ 仍为 prismatic crystal。若 $\mathcal E$ finite locally free，则 $\mathcal E^\vee$ 也是 prismatic crystal。

**证明.** 对 prism morphism $B\to B'$，crystal 条件给出
$$
\mathcal E(B')\cong\mathcal E(B)\otimes_BB',\qquad
\mathcal F(B')\cong\mathcal F(B)\otimes_BB'.
$$
张量后得到
$$
(\mathcal E\otimes\mathcal F)(B')\cong(\mathcal E(B)\otimes_B\mathcal F(B))\otimes_BB'.
$$
Dual 情形使用 finite locally free module 的 base change 相容性：
$$
\operatorname{Hom}_B(M,B)\otimes_BB'\cong\operatorname{Hom}_{B'}(M\otimes_BB',B').
$$
证毕。

**说明 6.17.** $F$-crystal 上的 tensor product 还需检查 Frobenius maps 的 tensor product 在 invert $I$ 后仍为同构。这是第六章从线性对象走向 Tannakian 结构的入口。

## 本章小结

Prismatic $F$-crystals 是 prismatic site 上带 Frobenius 的 finite locally free crystal。Bhatt-Scholze 的核心外部输入定理把它们与 crystalline Galois representations 的 lattices 等价起来，从而把 prismatic geometry 与 classical $p$-adic Hodge theory 的表示论对象连接。

## 练习

**练习 6.1.** 写出 prismatic crystal 的 transition isomorphism，并说明它与 sheaf restriction map 的差异。

**练习 6.2.** 若 $I=(d)$，把定义 6.4 中的 $\mathcal E[1/I]$ 改写为 $\mathcal E[1/d]$，并说明为什么换生成元不改变局部化。

**练习 6.3.** 解释定理 6.9 为什么是范畴等价，而不是 cohomology comparison theorem。
