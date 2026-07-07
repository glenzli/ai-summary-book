# 第十三章：带系数 prismatic cohomology 与非阿贝尔边界

## 本章目标

本章记录带系数 prismatic cohomology、Hodge-Tate prismatic crystals、$q$-Higgs modules 和 non-abelian $p$-adic Hodge theory 的边界。该部分目前仍在快速发展，因此本章以定义接口、风险边界和已核查预印本位置为主。

## 依赖前置知识

依赖第六章 prismatic crystals、第七章 Nygaard/syntomic、第九章 Hodge-Tate specialization。需要 Higgs modules、connections 和 non-abelian Hodge theory 的基本背景。

## 13.1 带系数 prismatic cohomology

**定义 13.1.** 令 $\mathcal E$ 为 $(X/A)_\Delta$ 上的 prismatic crystal。其带系数 prismatic cohomology 定义为
$$
R\Gamma_\Delta(X/A,\mathcal E)
=R\Gamma((X/A)_\Delta,\mathcal E).
$$

**命题 13.2.** 若 $\mathcal E=\mathcal O_\Delta$，则定义 13.1 回到普通 prismatic cohomology。

**证明.** 将 $\mathcal E$ 取为结构层 $\mathcal O_\Delta$，定义 13.1 右侧即
$$
R\Gamma((X/A)_\Delta,\mathcal O_\Delta)=R\Gamma_\Delta(X/A).
$$
证毕。

**警告 13.3.** 带系数理论需要额外控制 crystal 的 finiteness、flatness、Frobenius 和 descent 条件。不能把所有 sheaves 当作可接受系数。

## 13.2 $A_{\inf}$-cohomology with coefficients

**研究边界 13.4.** Tsuji 2025 年工作研究 smooth $p$-adic formal scheme 上 locally finite free prismatic crystal 的 prismatic cohomology，并与对应 relative Breuil-Kisin-Fargues module 的 $A_{\inf}$-cohomology 比较。其方法使用 $q$-Higgs modules 和 cohomological descent。

**说明 13.5.** 本书把该结果作为带系数理论的重要入口，但不将其写入基础定理链。后续若要升级，必须补齐 crystal category、$q$-Higgs category、descent hypotheses 和 tensor compatibility 的 locator。

## 13.3 Hodge-Tate prismatic crystals

**定义框架 13.6.** Hodge-Tate prismatic crystal 是在 Hodge-Tate specialization 或 $\overline{\mathcal O}_\Delta$ 层面具有 crystal-like rigidity 的线性对象。其精确定义依赖所选 base prism、rationalization 和 quasi-l.c.i hypotheses。

**研究边界 13.7.** Qu-Yu 研究 bounded prism $(A,I)$ 和 bounded quasi-l.c.i algebra $R$ 上 rational Hodge-Tate prismatic crystals，并与 Hodge-Tate cohomology ring 上 topologically nilpotent integrable connections 建立范畴等价。arXiv 当前核查版本为 v3，2026-01-13。

**警告 13.8.** 定理 13.7 的对象不是第六章的 prismatic $F$-crystal。前者位于 Hodge-Tate/rational/non-abelian 边界，后者是 crystalline Galois representations 的积分线性对象。

## 13.4 Non-abelian boundary

**说明 13.9.** Classical non-abelian Hodge theory 把 local systems、Higgs bundles 和 connections 联系起来。$p$-adic non-abelian Hodge theory 的 prismatic 版本试图通过 Hodge-Tate prismatic crystals、$v$-vector bundles 和 Higgs-type structures 给出积分或 $p$-adic 替代。

**命题 13.10（边界判别）.** 若一个 prismatic non-abelian statement 同时涉及 crystals、connections 和 $v$-vector bundles，则它至少需要说明：

1. 使用 rational 还是 integral objects；
2. base prism 是否 bounded；
3. algebra 是否 smooth、quasi-l.c.i 或 semistable；
4. connections 是否 topologically nilpotent；
5. restriction functor 的目标 topology 是 pro-etale、v 还是另一个 site。

**证明.** 五项分别决定对象范畴、完备性、cotangent complex 行为、connection convergence 和 descent topology。缺少任一项，范畴等价的 source 或 target 都不确定。证毕。

## 13.5 系数对象的最低结构包

**定义 13.11.** 一个可用于比较定理的 prismatic coefficient package 至少包含：

1. 一个 prismatic crystal $\mathcal E$；
2. finite locally free 或 perfectness 条件；
3. 可选 Frobenius structure；
4. 与 pullback/base change 相容的 transition isomorphisms；
5. 若涉及 tensor operations，则需给出 tensor compatibility。

**命题 13.12.** 若 $\mathcal E$ 不满足 crystal transition isomorphism，则 $R\Gamma_\Delta(X/A,\mathcal E)$ 不能称为带 crystal 系数的 prismatic cohomology。

**证明.** “带 crystal 系数”要求系数对象在 prismatic thickenings 方向 rigid。若 transition map 不是同构，则对象只是一个 sheaf 或 presheaf 系数，缺少 crystal rigidity。故名称不成立。证毕。

**例 13.13.** 若 $\mathcal E=\mathcal O_\Delta^{\oplus r}$，则它是 finite free prismatic crystal；其 cohomology 为
$$
R\Gamma_\Delta(X/A)^{\oplus r}.
$$

**证明.** 结构层 $\mathcal O_\Delta$ 满足 crystal transition；有限直和保持 transition isomorphism。Derived global sections 与有限直和相容。证毕。

## 本章小结

带系数和非阿贝尔方向是 prismatic theory 的前沿延伸。本章将其组织为研究边界：普通带系数 cohomology 可以定义，但深层 comparison、Higgs correspondence 和 $v$-bundle restriction 需要严格 hypotheses 和 locator 支撑。

## 练习

**练习 13.1.** 解释为什么 $\mathcal E=\mathcal O_\Delta$ 时带系数 cohomology 回到普通 cohomology。

**练习 13.2.** 列出 Hodge-Tate prismatic crystal 与 prismatic $F$-crystal 的三个差异。

**练习 13.3.** 选择命题 13.10 中任一条件，说明省略它会造成什么定义歧义。
