# 附录 AV：Serre、Atiyah-Hirzebruch 与 Adams 谱序列接口

附录 AQ 给出 exact couple 的代数核。本附录把三类经典谱序列写成 HoTT 可引用的严格输入格式：Serre cohomology spectral sequence、Atiyah-Hirzebruch spectral sequence 和 Adams spectral sequence。这里的重点是清楚列出输入、页、微分、收敛目标和证明架构。

## AV.1 过滤对象到 exact couple

**定义 AV.1（filtered cohomology object）.** 给定类型或谱对象 $X$ 的递增过滤
$$
\varnothing=X_{-1}\to X_0\to X_1\to\cdots\to X,
$$
以及上同调理论 $E^\ast(-)$。令
$$
D_1^{p,q}\coloneqq E^{p+q}(X_p),\qquad
E_1^{p,q}\coloneqq E^{p+q}(X_p,X_{p-1}).
$$

**输入 AV.2（cofiber 长正合列）.** 每个 cofiber sequence
$$
X_{p-1}\to X_p\to X_p/X_{p-1}
$$
诱导上同调长正合列。

**命题 AV.3（过滤给出 exact couple）.** 在 AV.1-AV.2 下，连接映射、限制映射和商映射组成 exact couple，从而生成谱序列
$$
E_r^{p,q}.
$$

**证明（证明核）.** 对每个 $p$ 取 pair/cofiber 长正合列。令 $i$ 来自 $X_{p-1}\to X_p$ 的限制，$j$ 来自 $X_p\to(X_p,X_{p-1})$，$k$ 为连接同态。exactness 正是长正合列在三项处的 exactness。把双次数平移记录进 $i,j,k$ 的 degree，即得 AQ.2 的 exact couple。$\square$

## AV.2 Serre cohomology spectral sequence

**输入 AV.4（Serre fibration 的 HoTT 版本）.** 给定 fiber sequence
$$
F\to E\to B
$$
并假设 $B$ 具有有限骨架过滤或等价的 cellular/HIT 过滤，且上同调理论 $H^\ast(-;G)$ 满足 Eilenberg-Steenrod 型性质。

**定理 AV.5（Serre cohomology spectral sequence，外部输入 / 证明架构）.** 若 $\pi_1(B)$ 对 $H^q(F;G)$ 的作用平凡，存在一阶象限谱序列
$$
E_2^{p,q}\cong H^p(B;H^q(F;G))
\Rightarrow H^{p+q}(E;G).
$$
微分次数为
$$
d_r:E_r^{p,q}\to E_r^{p+r,q-r+1}.
$$

**证明架构.**

1.  用 $B$ 的骨架过滤 $B_p$ 拉回得到 $E_p\to B_p$；
2.  由 $E_{p-1}\to E_p\to E_p/E_{p-1}$ 的 cofiber 长正合列构造 exact couple；
3.  识别 $E_1$ 页为 cellular cochains
    $$
    C^p(B;H^q(F;G));
    $$
4.  $d_1$ 识别为 cellular coboundary；
5.  因此
    $$
    E_2^{p,q}\cong H^p(B;H^q(F;G));
    $$
6.  若过滤有限、exhaustive 且 separated，则谱序列收敛到 $H^\ast(E;G)$ 的 associated graded。

**HoTT 边界.** 需要 cellular filtration 的 HIT 版本、局部系数、fiber transport 对上同调的作用，以及 cohomology long exact sequence。当前本书只把 AV.5 作为高级外部输入，不把它作为已完成书内定理。

## AV.3 Atiyah-Hirzebruch spectral sequence

**输入 AV.6（广义上同调理论）.** 设 $E$ 为 spectrum，定义广义上同调
$$
E^n(X)\coloneqq\|X\to_\ast E_n\|_0
$$
或等价的 spectrum mapping group，并假设它满足 cofiber 长正合列与悬挂公理。

**定理 AV.7（AHSS，外部输入 / 证明架构）.** 对具有 cellular filtration 的 pointed 类型 $X$，有谱序列
$$
E_2^{p,q}\cong H^p(X;E^q(\ast))
\Rightarrow E^{p+q}(X).
$$
在同调版本中：
$$
E^2_{p,q}\cong H_p(X;E_q(\ast))
\Rightarrow E_{p+q}(X).
$$

**证明架构.**

1.  用 $X$ 的骨架过滤 $X_p$ 构造 exact couple；
2.  识别相对层
    $$
    X_p/X_{p-1}
    $$
    为 $p$-球的 wedge；
3.  由悬挂公理和 wedge 加性得到
    $$
    E^{p+q}(X_p,X_{p-1})\cong C^p(X;E^q(\ast));
    $$
4.  $d_1$ 是 cellular coboundary；
5.  $E_2$ 页为 ordinary cohomology with coefficients in $E^q(\ast)$；
6.  收敛依赖 filtration 的有限性或完备性条件。

**证明核 AV.8（相对层识别）.** 若
$$
X_p/X_{p-1}\simeq\bigvee_{\alpha\in I_p}\mathbb S^p,
$$
则
$$
E^{p+q}(X_p,X_{p-1})
\cong
\prod_{\alpha\in I_p}E^q(\ast).
$$
证明使用 wedge 的映射泛性质、spectrum 悬挂同构和有限/集合索引积。$\square$

## AV.4 Adams spectral sequence

**输入 AV.9（Adams resolution）.** 固定 generalized homology/cohomology theory $E$，例如 mod-$p$ cohomology。对 spectrum 或 pointed type $X$，给出 Adams resolution：
$$
X\to X_0\to X_1\to X_2\to\cdots
$$
使得每层 cofiber 是 $E$-injective 或 $E$-module-like 对象。

**定理 AV.10（Adams spectral sequence，外部输入）.** 在经典 mod-$p$ 情形，对合适有限型 spectrum $X,Y$，存在谱序列
$$
E_2^{s,t}\cong
\operatorname{Ext}_{\mathcal A}^{s,t}(H^\ast(Y;\mathbb F_p),H^\ast(X;\mathbb F_p))
\Rightarrow [X,Y]_{t-s}^{\wedge}_p,
$$
其中 $\mathcal A$ 是 Steenrod algebra，右侧是 $p$-完备稳定同伦映射群。

**HoTT 边界.** 要在 HoTT 中内部化 AV.10，需要：

1.  spectrum category 或足够强的 stable homotopy interface；
2.  Steenrod algebra 的合成构造；
3.  Ext 群的代数开发；
4.  Adams resolution 的构造；
5.  完备化和收敛定理。

本书当前不把 AV.10 作为已证明结果，只把它作为稳定同伦论扩展的精确目标格式。

## AV.5 本附录关闭的缺口

附录 AQ 说明“谱序列是什么”；本附录说明三类核心谱序列需要哪些 HoTT 输入。剩余缺口已经具体化为：cellular filtration、局部系数、cofiber 长正合列、spectrum 范畴、Steenrod algebra、Ext 代数和收敛证明。
