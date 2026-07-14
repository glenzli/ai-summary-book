# 第八章：Prismatization、$F$-gauges 与 2026 研究边界

Prismatic site 擅长定义上同调与 crystal，却不直接把这些对象呈现为一个可做几何的模空间。Prismatization 和 Cartier--Witt stack 试图把同一算术增厚数据堆化，使 crystal 变成 stack 上的 quasicoherent sheaf，$F$-gauge 则把 Frobenius 与滤过放进更适合模问题的对象。本章从前七章的 site、crystal、Nygaard 与 syntomic 结构出发，说明这种几何化保留了什么信息，以及 operations、Shimura varieties、finite flat group schemes 和非阿贝尔方向还缺哪些比较定理。代数栈与高阶范畴语言作为背景；2025--2026 的结果逐项标为外部输入或研究边界，不进入基础证明链。

## 8.1 Prismatization 的动机

**说明 8.1.** Prismatic site 把 $X$ 的 arithmetic thickenings 组织成 site；prismatization 试图把同一信息几何化为某种 stack，使 prismatic crystals 可以解释为 stack 上的 quasicoherent sheaves。这样做的目的不是改变基础 cohomology，而是给 crystals、filtrations 和 moduli 问题更几何的承载空间。

**外部输入定义 8.2.** Bhatt-Lurie 对 $p$-adic formal scheme $X$ 构造 Cartier-Witt stack
$$
\mathrm{WCart}_X
$$
及其变体。相对版本在 bounded prism $(A,I)$ 和 $X/\overline A$ 上记为
$$
\mathrm{WCart}_{X/A}.
$$
它把 Cartier-Witt divisor 与到 $X$ 的点组合成 stack-valued functor，并带有到 $\operatorname{Spf}(A)$ 的结构映射。Hodge-Tate locus 由 base change 到 $\overline A=A/I$ 得到。

**外部输入定理 8.3.** 在适当有利情形，$X$ 上 prismatic crystals 及其 cohomology 可由 $\mathrm{WCart}_X$ 上的 quasicoherent sheaf theory 重新解释。
更具体地，若 $X/\overline A$ 满足 $p$-completely lci 或 $p$-quasisyntomic 类型假设，则有以下两类比较：
$$
R\Gamma(\mathrm{WCart}_{X/A},\mathcal O)
\simeq
R\Gamma_{\Prism}(X/A),
$$
以及
$$
\mathcal D_{qc}(\mathrm{WCart}_{X/A})
\simeq
\widehat{\mathcal D}_{\mathrm{crys}}((X/A)_\Prism,\mathcal O_\Prism).
$$
绝对版本在 $p$-quasisyntomic 假设下给出 $\mathcal D_{qc}(\mathrm{WCart}_X)$ 与 absolute prismatic crystals 的等价。

**警告 8.4.** Bhatt--Lurie 论文在 arXiv 页面标注为 preliminary version，因此这里不由 prismatization 反推第二、三章的基础比较定理。可使用的接口限于已明确定位的 cohomology、crystal 与 pushforward 对应，并始终保留其完备性和几何假设。

## 8.2 Prismatic $F$-gauges

**定义框架 8.5.** Prismatic $F$-gauge 可以粗略理解为把 filtered object、Frobenius、Nygaard-type divisibility 和 prismatic realization 组织在一起的结构。不同文献中的 precise definition 可能位于 stack、filtered derived category、display theory 或 spectral enhancement 中。

**警告 8.6.** “$F$-crystal”和“$F$-gauge”不是同义词。$F$-crystal 是 prismatic site 上的 crystal 加 Frobenius；$F$-gauge 通常编码额外 filtration/gauge data，并常出现在 prismatization 或 display-theoretic 表述中。

**研究边界 8.7.** Ito 的 prismatic $G$-displays 工作把 prismatic deformation theory 与 local Shimura varieties 联系起来，并说明该理论可用 prismatic $F$-gauges 解释。

**研究边界 8.8.** Imai-Kato-Youcis 构造 Shimura varieties of abelian type 的 prismatic realization functor，并发展 integral analogue of $D_{\mathrm{crys}}$。该方向把 prismatic theory 带入 Shimura varieties 的 integral model 研究。

## 8.3 Operations and duality

**研究边界 8.9.** Carmeli-Feng 构造 syntomic Steenrod algebra，并将其用于 Brauer groups 的 arithmetic duality。他们的 spectral syntomic cohomology 和 spectral prismatic $F$-gauges 表明 prismatic methods 可以承载 cohomology operations，而不仅是 comparison theorem。

**研究边界 8.10.** Ambrosi-Newton-Pagano 使用 prismatic cohomology 研究 wild Brauer classes 和 weak approximation 的障碍。该方向展示 prismatic methods 在 arithmetic geometry 中的应用层扩展。

## 8.4 Coefficients and non-abelian directions

**研究边界 8.11.** Tsuji 的 2025 年工作研究带系数 prismatic cohomology 与 $A_{\inf}$-cohomology 的比较，并使用 $q$-Higgs modules 与 cohomological descent。

**研究边界 8.12.** Qu-Yu 的 2025 年工作研究 rational Hodge-Tate prismatic crystals of quasi-l.c.i algebras，并连接 non-abelian $p$-adic Hodge theory。该方向说明 prismatic crystals 正在从线性表示论向非阿贝尔结构扩展。

## 8.5 Finite flat group schemes

**研究边界 8.13.** Mondal-Olsson 2026 年工作描述 smooth positive characteristic variety 上 height one finite flat group scheme 所对应的 prismatic $F$-gauge，并由此恢复 crystalline Dieudonne module 和 flat cohomology 的结果。

**说明 8.14.** 该方向把 prismatic $F$-gauges 与 classical Dieudonné theory 重新连接。要同第六章 prismatic $F$-crystal classification 比较，还需构造从 height-one group scheme 的 gauge 数据到相应 crystal 的函子，并证明它保持 Frobenius、Verschiebung 与 descent；没有这些步骤时两项结果不能直接合并。

## 8.6 前沿接口所缺的比较数据

| 方向 | 核心对象 | 可比较的输出 | 尚需固定的数据 |
| --- | --- | --- | --- |
| Prismatization | $\mathrm{WCart}_X$, QCoh | prismatic cohomology 与 crystals | 完备性、lci/quasisyntomic 假设及 pushforward |
| $F$-gauges | filtered/Frobenius gauge objects | filtration 与 Frobenius 的统一对象 | gauge 范畴及其到 $F$-crystals 的函子 |
| Syntomic operations | Steenrod operations, spectral syntomic | cup products 与上同调运算 | spectral enhancement 和次数约定 |
| Coefficients | prismatic crystals with coefficients | 带系数 comparison | tensor/descent 与 torsion 假设 |
| Non-abelian Hodge | rational Hodge--Tate crystals | 模空间层的对应 | 非阿贝尔对象范畴与高阶 descent |
| Shimura varieties | prismatic realization | integral realization functor | group datum、level 与 integral model |
| finite flat groups | height-one group schemes | Dieudonné 型 gauge | Frobenius/Verschiebung 与 height 条件 |

## 8.7 从 site 到 stack 的信息保真

**命题 8.15（重解释的最低要求）.** 若某个 stack-theoretic construction 声称重解释 prismatic crystals，则它至少必须给出：

1. 从 prismatic probes 到 stack-valued points 的 functor；
2. structure sheaf 或 quasicoherent sheaves 的对应；
3. crystals 的 pullback rigidity 与 quasicoherent sheaf descent 的对应；
4. Frobenius 或 Cartier-Witt structure 的对应；
5. cohomology comparison。

**证明.** Prismatic crystal 不只是 sheaf，它还依赖 probe category、structure sheaf、crystal transition isomorphism 和 Frobenius structure。若 stack 重解释缺少任一项，则不能恢复原始 crystal theory 或其 cohomology。证毕。

**警告 8.16.** Prismatization 是重新组织 prismatic 信息的几何语言，不是把 prismatic site 删除后留下一个普通 stack 的替代品。特别是 $\mathrm{WCart}$ 侧的 quasicoherent sheaf statement 要带完备性、quasisyntomic/lci 假设和 derived stack caveat；省略这些条件会把前沿接口误写成无条件基础定理。

## 8.8 从 site 到几何化模空间

Prismatic theory 的前沿已经从 comparison theorem 扩展到 stack-theoretic reinterpretation、$F$-gauges、syntomic operations、Shimura varieties、Brauer groups 和 non-abelian Hodge theory。它们共享 $\delta$-环、prism、site 与 Frobenius 语言，却分别要求新的完备性、descent 或模空间结构。Prismatization 的最低可比较数据是 $\mathrm{WCart}_{X/A}$、cohomology comparison、crystals-as-QCoh 与 pushforward compatibility；少掉其中任何一项，都只能得到部分重解释而非原 crystal theory 的等价模型。

## 练习

**练习 8.1.** 解释为什么 prismatization 是 prismatic site 的几何重解释，而不是 prismatic cohomology 的替代定义。

**练习 8.2.** 列出 $F$-crystal 与 $F$-gauge 至少两个结构差异。

**练习 8.3.** 从本章表格中选择一个方向，写出从核心对象到目标输出仍缺少的比较函子或假设。
