# 应用章：费马大定理作为 `GL(2)/\mathbb Q` 模性的实例

## 本章目标

本章给出费马大定理的严格逻辑证明，前提是接受三个外部输入：Frey 曲线的基本性质、Ribet 降层定理和半稳定椭圆曲线的模性定理。本章的目的不是重证 Wiles-Taylor-Wiles 机器，而是说明费马大定理如何作为 `GL(2)/\mathbb Q` Langlands 思想的一个具体应用出现。

## 依赖前置知识

需要椭圆曲线、模形式、Galois 表示和导子的基础定义。相关对象将在本书第二部分系统展开；本章先给出足够支撑逻辑链的定义和外部输入。附录 T 解释模形式 Galois 表示和 residual representations 的来源，附录 U 解释模性提升背后的 p-adic Hodge/patching 接口，附录 AD 解释 Frey 曲线的判别式、半稳定性、Tate algorithm 局部导子和 residual conductor 降到级 $2$ 的接口；本章不重证这些外部输入。

收口归一化回指：本章只使用费马应用所需的逻辑链；Frey 曲线 Frobenius trace、残余表示、导子和模形式 L 函数比较按 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 5、6、7、8 节。

## 90.1 费马大定理和指数归约

**定理 90.1（费马大定理）.** 对任意整数 $n>2$，方程
$$
x^n+y^n=z^n
$$
没有满足 $xyz\ne 0$ 的整数解。

**证明安排.** 本定理的证明在定理 90.10 完成：先用引理 90.2 归约
指数，再调用外部输入 90.2A、90.5、90.7、90.8，并以引理 90.9 给出
最终矛盾。这里先陈述主定理，以便固定本章目标。

**引理 90.2（指数归约）.** 若费马大定理对每个奇素数指数 $p$ 以及指数 $4$ 成立，则它对所有整数 $n>2$ 成立。

**证明.** 设 $n>2$。若 $n$ 有奇素因子 $p$，写 $n=pm$。若存在
$$
x^n+y^n=z^n,
$$
则
$$
(x^m)^p+(y^m)^p=(z^m)^p
$$
给出指数 $p$ 的反例。若 $n$ 没有奇素因子，则 $n=2^r$ 且 $r\ge 2$，于是 $4\mid n$。写 $n=4m$，则
$$
(x^m)^4+(y^m)^4=(z^m)^4
$$
给出指数 $4$ 的反例。因此只需处理指数 $4$ 和奇素数指数。$\square$

**外部输入 90.2A（低指数 Fermat 情形）.** 指数 $3$ 与指数 $4$ 的
Fermat 方程没有非零整数解。两者有经典下降证明；本章不重建该初等
数论支线，只把它们用于引理 90.2 后剩余的低指数情形。本章的
Langlands 型论证处理素数 $p\ge 5$。

## 90.2 Frey 曲线

**约定 90.3.** 设 $p\ge 5$ 为素数，并假设存在 primitive 反例
$$
a^p+b^p=c^p,\qquad abc\ne 0,\qquad \gcd(a,b,c)=1.
$$
通过交换 $a,b$ 和改变符号，可取满足标准奇偶条件的代表；这些初等整理不影响存在性。

**定义 90.4.** 与该反例相关的 Frey 曲线定义为
$$
E_{a,b,p}:\quad y^2=x(x-a^p)(x+b^p).
$$

这是 $\mathbb Q$ 上的椭圆曲线，因为三根 $0,a^p,-b^p$ 两两不同。其判别式为
$$
\Delta(E_{a,b,p})=16a^{2p}b^{2p}c^{2p}
$$
在上述方程下成立，具体最小判别式和导子需要按素数 $2$ 及除 $abc$ 的奇素数分别分析。

**外部输入定理 90.5（Frey 曲线性质）.** 若存在 primitive 反例 $a^p+b^p=c^p$，$p\ge 5$，则 Frey 曲线 $E=E_{a,b,p}$ 满足：

1. $E/\mathbb Q$ 是半稳定椭圆曲线。
2. 模 $p$ Galois 表示
   $$
   \overline\rho_{E,p}:G_{\mathbb Q}\to\operatorname{GL}_2(\mathbb F_p)
   $$
   满足 Ribet 降层所需的不可约性和局部条件。
3. 这些局部条件的导子计算正是 Ribet 降层定理在本情形下降到级 $2$ 所需的输入。

## 90.3 模性和降层

**定义 90.6.** 椭圆曲线 $E/\mathbb Q$ 称为模的（modular），若存在权 $2$ newform
$$
f(q)=\sum_{n\ge 1}a_nq^n
$$
其级等于 $E$ 的导子 $N_E$，并且对几乎所有素数 $\ell$ 有
$$
a_\ell=\ell+1-\#E(\mathbb F_\ell).
$$
等价地，Hasse-Weil L 函数 $L(E,s)$ 与 $f$ 的 L 函数 $L(f,s)$ 在坏素数 Euler 因子按第八章导子和局部约化类型约定补齐后相同。

**外部输入定理 90.7（半稳定模性定理，Wiles-Taylor-Wiles）.** 每条半稳定椭圆曲线 $E/\mathbb Q$ 都是模的。

该定理是 Taniyama-Shimura-Weil 猜想在半稳定情形的证明，是 Wiles 和 Taylor-Wiles 工作的核心输出。

**外部输入定理 90.8（Ribet 降层定理的当前形式）.** 设 $p\ge 5$，$E/\mathbb Q$ 为满足 Frey 曲线局部条件的半稳定椭圆曲线。若 $E$ 是模的，则模 $p$ 表示 $\overline\rho_{E,p}$ 来自权 $2$、级 $2$ 的 newform。第十章把该输入拆解为局部-整体相容、残余导子计算和 Ribet 降层三个组成部分。

这里“来自”指存在 newform $f\in S_2(\Gamma_0(2))$ 和其系数域中位于 $p$ 上方的素理想 $\mathfrak p$，使得对几乎所有素数 $\ell$，
$$
\operatorname{tr}\overline\rho_{E,p}(\operatorname{Frob}_\ell)
\equiv
a_\ell(f)\pmod{\mathfrak p}
$$
并且行列式也相容。

**注 90.8.1.** 附录 W 把“权 $2$、级 $2$ newform”所在的 classical new subspace、old/new 分解和 $S_2(\Gamma_0(2))=0$ 的模曲线来源集中说明。费马应用只使用这些结果的逻辑后果。

**注 90.8.2.** 附录 AD 把 Frey 曲线局部性质拆为可检查的局部椭圆曲线语句：判别式为 $16a^{2p}b^{2p}c^{2p}$，奇坏素数处为乘法约化，半稳定导子在这些素数处指数为 $1$，而模 $p$ residual conductor 删除这些奇素数。第十章用这些输入执行 Ribet 降层。

**收口精修 90.A（费马应用输入表）.** 本章证明只需要以下有限条输入，不需要完整数域 `GL(n)` Langlands：

| 输入 | 位置 | 作用 |
|---|---|---|
| Frey 曲线及其局部性质 | 本章 90.5、附录 AD | 从 Fermat 反例构造半稳定椭圆曲线和 residual conductor |
| 半稳定模性定理 | 本章 90.7、第九章 | 把 Frey 曲线的 Galois 表示放到模形式侧 |
| Ribet 降层 | 本章 90.8、第十章 | 把模形式级降到权 $2$、级 $2$ |
| $S_2(\Gamma_0(2))=0$ | 本章 90.9、附录 W | 排除目标 newform |
| 反证法 | 本章 90.10 | 得出 Fermat 反例不存在 |

## 90.4 级 `2` 处没有权 `2` cusp form

**引理 90.9.**
$$
S_2(\Gamma_0(2))=0.
$$

**证明.** 对任意 $N\ge 1$，权 $2$ cusp forms 空间 $S_2(\Gamma_0(N))$ 与模曲线 $X_0(N)$ 上的全纯微分空间同构，因此
$$
\dim S_2(\Gamma_0(N))=g(X_0(N)).
$$
对 $N=2$，使用 genus 公式
$$
g(X_0(N))
=
1+\frac{\mu}{12}-\frac{e_2}{4}-\frac{e_3}{3}-\frac{c}{2},
$$
其中 $\mu=[\operatorname{SL}_2(\mathbb Z):\Gamma_0(N)]$，$e_2,e_3$ 分别为椭圆点数，$c$ 为 cusp 数。对 $N=2$，
$$
\mu=3,\qquad e_2=1,\qquad e_3=0,\qquad c=2.
$$
代入得
$$
g(X_0(2))
=
1+\frac{3}{12}-\frac{1}{4}-0-\frac{2}{2}
=0.
$$
故 $\dim S_2(\Gamma_0(2))=0$。$\square$

## 90.5 费马大定理的条件证明

**定理 90.10.** 接受外部输入定理 90.5、90.7 和 90.8，则费马大定理成立。

**证明.** 由引理 90.2，只需排除指数 $4$ 和奇素数指数。指数 $4$ 和 $3$ 的情形由经典初等证明处理。设 $p\ge 5$，并假设存在 primitive 反例
$$
a^p+b^p=c^p.
$$
构造 Frey 曲线
$$
E:\quad y^2=x(x-a^p)(x+b^p).
$$
由外部输入定理 90.5，$E$ 是半稳定椭圆曲线，并满足 Ribet 降层所需的局部条件。由半稳定模性定理 90.7，$E$ 是模的。于是由 Ribet 降层定理 90.8，模 $p$ 表示 $\overline\rho_{E,p}$ 来自某个权 $2$、级 $2$ 的 newform。

这意味着
$$
S_2(\Gamma_0(2))\ne 0.
$$
但引理 90.9 已证明 $S_2(\Gamma_0(2))=0$。矛盾。因此不存在指数 $p\ge 5$ 的 primitive 反例。结合指数归约和指数 $3,4$ 的经典情形，费马大定理成立。$\square$

## 90.6 这为什么属于 Langlands 主线

上述证明不是完整 Langlands 纲领的直接推论，而是 `GL(2)/\mathbb Q` 情形中模性思想的应用。

椭圆曲线 $E/\mathbb Q$ 给出二维 Galois 表示
$$
\rho_{E,\ell}:G_{\mathbb Q}\to\operatorname{GL}_2(\mathbb Q_\ell).
$$
模性定理断言，这个 Galois 表示来自某个权 $2$ 自守对象，即 modular newform 或等价的 `GL(2,\mathbb A_\mathbb Q)` 自守表示。用 Langlands 语言说，这是把 Galois 侧的二维表示与自守侧的 `GL(2)` 表示相联系。

Ribet 降层则说明，在模 $p$ 同余意义下，某些局部 ramification 条件会强迫自守侧的级数下降。Frey 曲线把费马方程的假想反例转化为一个具有异常局部性质的 Galois 表示；模性和降层把这种异常性质转移到模形式空间中，最终落到不存在的空间 $S_2(\Gamma_0(2))$。

因此费马大定理展示的是：

$$
\text{丢番图方程}
\longrightarrow
\text{椭圆曲线}
\longrightarrow
\text{Galois 表示}
\longleftrightarrow
\text{自守形式}
\longrightarrow
\text{有限维空间的矛盾}.
$$

这条链正是 Langlands 纲领把数论问题转化为表示论和自守形式问题的典型机制。

## 90.7 本章小结

本章证明了：在接受半稳定模性定理、Ribet 降层和 Frey 曲线局部性质的前提下，费马大定理严格推出。该证明的 Langlands 含义在于，椭圆曲线产生的二维 Galois 表示通过模性对应到 `GL(2)` 自守表示，再由降层和模形式空间维数得到矛盾。

## 练习

**练习 90.1.** 证明引理 90.2 中的指数归约。

**练习 90.2.** 设 $E/\mathbb Q$ 是椭圆曲线。说明 $\#E(\mathbb F_\ell)$ 与 $\rho_{E,\ell}$ 的 Frobenius trace 之间的关系。

**练习 90.3.** 查阅模曲线 genus 公式，验证 $X_0(2)$ 的 genus 为 $0$。

**练习 90.4.** 用本章语言解释为什么“费马大定理由 Langlands 纲领证明”这句话不够精确。
