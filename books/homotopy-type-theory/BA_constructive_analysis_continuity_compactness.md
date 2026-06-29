# 附录 BA：构造性分析中的连续性、紧致性与典型定理

附录 AK、AR、AW 建立实数对象、序和 Cauchy-Dedekind 比较。本附录继续给出构造性分析的基础定理接口：连续函数、一致连续、紧区间和中值/极值型定理的构造性版本。度量空间、级数、一致收敛、Banach 不动点和 Riemann 积分的工作层见附录 BO。

## BA.1 连续性

**定义 BA.1（点态连续）.** 函数 $f:\mathbb R_C\to\mathbb R_C$ 在 $x$ 处连续，若
$$
\prod_{\varepsilon>0}
\left\|
\sum_{\delta>0}
\prod_y |x-y|<\delta\to |f(x)-f(y)|<\varepsilon
\right\|.
$$

**定义 BA.2（一致连续）.** $f$ 在子类型 $D\subseteq\mathbb R_C$ 上一致连续，若
$$
\prod_{\varepsilon>0}
\left\|
\sum_{\delta>0}
\prod_{x,y:D}|x-y|<\delta\to |f(x)-f(y)|<\varepsilon
\right\|.
$$

**命题 BA.3（多项式一致连续于有界区间）.** 任意有理系数多项式函数在闭有界区间 $[a,b]$ 上一致连续。

**证明核.** 用 Horner 表达式和 AR.8 的乘法 Cauchy 估计。闭有界区间给出统一界 $M$；每个加法和乘法步骤给出 Lipschitz 多项式界。对多项式次数归纳，合成误差预算得到统一 $\delta$。$\square$

## BA.2 Totally bounded 与 Cauchy compact

**定义 BA.4（totally bounded）.** 类型 $X$ 带度量，若对每个 $\varepsilon>0$，存在有限列表 $x_1,\ldots,x_n:X$，使每个 $x:X$ 距某个 $x_i$ 小于 $\varepsilon$。存在性用命题截断封装。

**定义 BA.5（Cauchy complete metric type）.** 度量类型 $X$ Cauchy complete，若每个 Cauchy 近似有极限。

**定义 BA.6（constructively compact）.** $X$ 构造性紧，若它 totally bounded 且 Cauchy complete。

**命题 BA.7（闭区间 totally bounded）.** 对 $a<b$，闭区间 $[a,b]\subseteq\mathbb R_C$ totally bounded。

**证明核.** 给定 $\varepsilon>0$，由有理数阿基米德性质取 $N$ 使 $(b-a)/N<\varepsilon$。用有限网格
$$
a,\ a+(b-a)/N,\ldots,b
$$
覆盖区间。每个 $x\in[a,b]$ 的所在网格段由 located order 或 Dedekind locatedness 给出；若没有可判定段选择，则覆盖存在性保留在命题截断中。$\square$

**命题 BA.8（闭区间 Cauchy complete）.** $[a,b]$ 是 Cauchy complete。

**证明核.** Cauchy 近似在 $\mathbb R_C$ 中有极限 AK.9。闭区间条件 $a\le x_n\le b$ 对极限闭合：若极限 $x<a$，由严格序开放性和 Cauchy 收敛，足够后项也 $<a$，矛盾；$x>b$ 同理。因此极限仍在 $[a,b]$。$\square$

## BA.3 一致连续定理

**定理 BA.9（紧类型上的连续函数一致连续，构造性版本）.** 若 $X$ totally bounded 且 Cauchy complete，$Y$ 为度量类型，函数 $f:X\to Y$ 点态连续并且连续模可局部选择，则 $f$ 一致连续。

**证明核.** 对 $\varepsilon$，每个 $x:X$ 给出局部半径 $\delta_x$。用 totally bounded 取足够细有限网格；对网格点选择有限多个局部模，取最小正半径。任意 $x,y$ 足够近时，选同一网格点 $x_i$，由三角不等式将 $f(x)$、$f(y)$ 都控制在 $f(x_i)$ 附近。构造性关键是局部连续模的有限选择；若只有命题截断存在性，需要额外 compact choice 或把连续性定义加强为携带模。$\square$

## BA.4 中值与极值

**定理 BA.10（中值定理，located 版本）.** 设 $f:[a,b]\to\mathbb R_C$ 连续并携带一致连续模，且
$$
f(a)<0<f(b).
$$
若 $\mathbb R_C$ 的序 located，则存在 $x:[a,b]$ 使
$$
f(x)=0.
$$

**证明核.** 用二分法构造嵌套区间 $[a_n,b_n]$，保持符号变化且长度趋零。locatedness 用于在中点 $m$ 判断 $f(m)<0$ 或 $0<f(m)$ 或足够接近零。端点序列是 Cauchy，完备性给极限 $x$。一致连续性推出 $f(x)=0$。$\square$

**定理 BA.11（极值定理，近似版本）.** 若 $f:[a,b]\to\mathbb R_C$ 一致连续，则对任意 $\varepsilon>0$，存在 $x:[a,b]$ 使
$$
\prod_y f(y)\le f(x)+\varepsilon.
$$

**证明核.** 用 BA.7 的有限 $\delta$-网格。取网格点中函数值最大的点 $x_i$；任意 $y$ 距某个网格点 $x_j$ 小于 $\delta$，一致连续给 $f(y)$ 与 $f(x_j)$ 相差 $<\varepsilon$，而 $f(x_j)\le f(x_i)$。有限最大值使用有理/实数有限比较；构造性地可得到近似最大。精确最大需要更强 compactness 或 located order。$\square$

## BA.5 本附录关闭的缺口

构造性分析不再停留在“有实数对象”：现在有连续性、紧性、一致连续、中值和极值近似定理的证明核；附录 BO 进一步给出度量空间、级数和积分接口。剩余工作是为每个定理固定选择原则口径，并逐项展开误差预算。
