# 附录 E：高阶范畴的技术模型

## 本章目标

本附录补充第十七至第十九章中被压缩处理的技术模型：join、slice quasi-category、Joyal 模型结构、marked simplicial sets，以及 Cartesian fibration 的 horn lifting 口径。正文仍以 quasi-category 为默认模型。

## E.1 join 与 slice

**定义 E.1.** 单纯集 $X,Y$ 的 join $X\star Y$ 是由如下公式定义的单纯集：

$$
(X\star Y)_n=
X_n\sqcup Y_n\sqcup
\coprod_{i+j+1=n}X_i\times Y_j,
$$

面和退化映射由 $X$、$Y$ 的面退化映射以及跨越连接处的规则给出。几何上，$X\star Y$ 把 $X$ 的每个顶点放在 $Y$ 的每个顶点之前。

**例子 E.2.** $\Delta^m\star\Delta^n\cong\Delta^{m+n+1}$。特别地，$\Delta^0\star K$ 是 $K$ 上的右锥，$K\star\Delta^0$ 是左锥。

**定义 E.3.** 设 $C$ 是 quasi-category，$p:K\to C$ 是图形。slice quasi-category $C_{/p}$ 由以下泛性质刻画：对任意单纯集 $T$，

$$
\mathbf{sSet}(T,C_{/p})
\cong
\{\,T\star K\to C\mid (T\star K)|_K=p\,\}.
$$

对偶地，$C_{p/}$ 用 $K\star T$ 定义。

**命题 E.4.** 若 $C=N(\mathcal C)$ 且 $p:K\to N(\mathcal C)$ 来自普通图形，则 $C_{/p}$ 的 $0$-单纯形是普通锥，$1$-单纯形是锥之间的态射。

**证明.** $0$-单纯形是映射 $\Delta^0\to C_{/p}$，等价于 $\Delta^0\star K\to N(\mathcal C)$，其在 $K$ 上等于 $p$。这正是给图形 $p$ 加一个锥顶和从锥顶到图形各对象的相容态射。$1$-单纯形对应 $\Delta^1\star K\to N(\mathcal C)$，即两个锥顶及其间态射，并与所有锥边相容。由于 nerve 中单纯形等价于函子，这正是普通锥之间的态射。$\square$

## E.2 inner anodyne 与 Joyal 等价

**定义 E.5.** inner horn inclusion 是嵌入

$$
\Lambda_i^n\hookrightarrow\Delta^n,\qquad 0<i<n.
$$

由这些嵌入经 pushout、transfinite composition 和 retract 生成的态射类称为 inner anodyne morphisms。

**定义 E.6.** 单纯集映射 $f:X\to Y$ 称为 categorical equivalence 或 Joyal equivalence，若对任意 quasi-category $C$，诱导映射

$$
\operatorname{Map}(Y,C)\to\operatorname{Map}(X,C)
$$

在适当的同伦函数复形意义下为弱同伦等价。

**外部输入定理 E.7（Joyal 模型结构）.** $\mathbf{sSet}$ 上存在模型结构，其 cofibration 为单射，fibrant objects 为 quasi-categories，weak equivalences 为 categorical equivalences。该模型结构称为 Joyal 模型结构。

来源见 Joyal、Lurie HTT、Cisinski 和 Kerodon。

## E.3 marked simplicial sets

**定义 E.8.** marked simplicial set 是二元组 $(X,E)$，其中 $X$ 是单纯集，$E$ 是 $X_1$ 的子集，包含所有退化边。$E$ 中的边称为 marked edges。

**例子 E.9.** 若 $C$ 是 quasi-category，可令 $C^\natural$ 为把所有等价边标记的 marked simplicial set。也可令 $C^\sharp$ 标记所有边，$C^\flat$ 只标记退化边。

**定义 E.10.** marked anodyne morphisms 是 marked simplicial sets 中一类由内角填充、标记等价边和若干饱和条件生成的态射。它们用于给 Cartesian fibrations 建立模型结构。

**外部输入定理 E.11.** 对固定 quasi-category $S$，存在 marked simplicial sets over $S$ 的 Cartesian model structure，其 fibrant objects 是 Cartesian fibrations $X\to S$ 连同 Cartesian edges 的标记。

来源见 Lurie HTT, §3.1，以及 Kerodon 的 marked anodyne 章节。

## E.4 Cartesian fibration 的 horn 口径

**定义 E.12.** 内纤维 $p:X\to S$ 是 Cartesian fibration，当且仅当：

1. $p$ 是 inner fibration；
2. 对每条边 $\alpha:s\to t$ 和每个 $y\in X_t$，存在边 $\tilde\alpha:x\to y$ 覆盖 $\alpha$；
3. 该边 $\tilde\alpha$ 满足 Cartesian lifting 条件，即对所有 $n\ge2$ 的适当 horn extension 问题有唯一到可缩空间的填充。

第三条可用映射空间同伦拉回条件等价表达，正是第十九章定义 19.1 使用的版本。

**外部输入命题 E.13.** 对普通 Grothendieck fibration $p:E\to B$，若把普通 Cartesian arrows 标记，则 nerve

$$
N(E)^\natural\to N(B)
$$

给出 marked simplicial set 语境中的 Cartesian fibration。

该命题是普通 Cartesian lift 泛性质与 horn lifting 条件的比较。证明见 Lurie HTT 与 Kerodon 中关于 ordinary fibrations and nerves 的相关结果；本书把它作为 straightening/unstraightening 的一阶影子使用。

## E.5 Joyal 模型与 Kan-Quillen 模型的关系

**命题 E.14.** Kan complex 是 quasi-category；在 Kan complex 中每条边都是等价边。

**证明.** Kan complex 对所有 horn 有填充，特别对所有 inner horn 有填充，因此是 quasi-category。外 horn 填充给出每条边的左右同伦逆，故其在同伦范畴中为同构。$\square$

**注 E.15.** Kan-Quillen 模型结构建模 spaces 或 $\infty$-groupoids；Joyal 模型结构建模 $\infty$-categories。前者要求所有 horn 填充，后者只要求 inner horn 填充。

## E.6 scaled simplicial sets

**定义 E.16.** scaled simplicial set 是二元组 $(X,T)$，其中 $X$ 是单纯集，$T\subseteq X_2$ 是一族 $2$-单纯形，包含所有退化 $2$-单纯形。$T$ 中的 $2$-单纯形称为 thin $2$-simplices。

**例子 E.17.** 若 $\mathcal B$ 是严格 $2$-范畴，则其 scaled nerve $N^{sc}(\mathcal B)$ 的低维部分如下：

1. $0$-单纯形为 $\mathcal B$ 的对象；
2. $1$-单纯形为 $\mathcal B$ 的 $1$-态射；
3. $2$-单纯形记录可复合 $1$-态射和一个比较 $2$-态射；
4. thin $2$-单纯形记录指定为可逆或相干等式的比较。

完整定义使用 normal lax functors $[n]\to\mathcal B$。本书只需要它说明 $2$-态射和相干关系如何被单纯模型编码。

**外部输入定理 E.18.** scaled simplicial sets 上存在建模 $(\infty,2)$-范畴的模型结构；严格 $2$-范畴的 scaled nerve 与该模型结构相容。该理论用于把 walking adjunction、双范畴和高阶 Morita 结构嵌入 $\infty$-范畴技术框架。

## E.7 本章小结

Join 和 slice 给出 $\infty$-范畴中锥、极限和逗号对象的精确定义。Joyal 模型结构把 quasi-category 放入模型范畴框架。Marked simplicial sets 是 Cartesian fibrations 和 straightening/unstraightening 的技术载体；scaled simplicial sets 则记录 $2$-态射和 $(\infty,2)$-范畴相干性。

## 练习

**练习 E.1.** 验证 $\Delta^0\star\Delta^0\cong\Delta^1$。

**练习 E.2.** 用定义 E.3 描述 $C_{/x}$ 的对象，其中 $x:\Delta^0\to C$ 是一个对象。

**练习 E.3.** 说明为什么 Kan complex 自动是 quasi-category。

**练习 E.4.** 对 quasi-category $C$，比较 $C^\natural$、$C^\sharp$ 和 $C^\flat$ 的标记边。

**练习 E.5.** 解释 Cartesian fibration 的映射空间定义为什么应等价于 horn lifting 定义。

**练习 E.6.** 比较 marked simplicial set 与 scaled simplicial set 分别标记什么维度的单纯形。

**练习 E.7.** 在严格 $2$-范畴的 scaled nerve 中，$2$-单纯形应记录哪些低维数据？

**练习 E.8.** 说明为什么 walking adjunction 比普通有向图多需要 $2$-态射数据。
