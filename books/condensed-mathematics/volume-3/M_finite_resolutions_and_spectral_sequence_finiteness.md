# 附录 M：有限分解、谱序列与有限性边界

## M.0 目标

第三卷讨论相干上同调有限性。容易误解的一点是：

> 有限 Stein 覆盖与 Cartan B 给出计算方法，但不自动给出有限维性。

原因是 Stein 开集上的相干层全局截面可能是无限维向量空间；例如单位圆盘上的全纯函数空间。本附录把有限性证明中真正的形式部分写清楚：

1. 有限维向量空间组成的有界复形有有限维同调。
2. 有界谱序列若某一页有限维，则极限对象有限维。
3. 有限分解只传播已经存在的有限性；它不能凭空产生有限性。
4. 复几何中的有限维性需要 Grauert、Fredholm-Hodge 或 Clausen-Scholze 的有限性输入。

## M.1 线性代数有限性

**引理 M.1（有限过滤）.** 设 $V$ 是复向量空间，并有有限递增过滤

$$
0=F^{-1}V\subset F^0V\subset\cdots\subset F^mV=V.
$$

若每个分级片

$$
\operatorname{gr}^p_FV=F^pV/F^{p-1}V
$$

有限维，则 $V$ 有限维，且

$$
\dim V=\sum_{p=0}^m\dim\operatorname{gr}^p_FV.
$$

**证明.** 对短正合列

$$
0\to F^{p-1}V\to F^pV\to\operatorname{gr}^p_FV\to0
$$

逐步使用维数可加性。由归纳得 $F^pV$ 有限维并满足维数公式。取 $p=m$ 得结论。证毕。

**引理 M.2（有界有限维复形）.** 设

$$
C^\bullet=(\cdots\to C^{n-1}\xrightarrow{d^{n-1}}C^n\xrightarrow{d^n}C^{n+1}\to\cdots)
$$

是复向量空间复形。若 $C^\bullet$ 有界，且每个 $C^n$ 有限维，则每个

$$
H^n(C^\bullet)
$$

有限维。

**证明.** 有包含

$$
H^n(C^\bullet)=\ker d^n/\operatorname{im}d^{n-1}.
$$

$\ker d^n$ 是有限维空间 $C^n$ 的子空间，因此有限维；其商仍有限维。证毕。

**命题 M.3（Euler characteristic）.** 在引理 M.2 的假设下，

$$
\sum_n(-1)^n\dim H^n(C^\bullet)
=
\sum_n(-1)^n\dim C^n.
$$

**证明.** 记 $Z^n=\ker d^n$，$B^n=\operatorname{im}d^{n-1}$。短正合列

$$
0\to Z^n\to C^n\to B^{n+1}\to0
$$

和

$$
0\to B^n\to Z^n\to H^n(C^\bullet)\to0
$$

给出

$$
\dim C^n=\dim Z^n+\dim B^{n+1},
\qquad
\dim Z^n=\dim B^n+\dim H^n.
$$

代入并对有限多个 $n$ 取交错和，$\dim B^n$ 项两两相消。证毕。

## M.2 谱序列的有限性传播

**定义 M.4.** 第一象限谱序列 $E_r^{p,q}$ 称为总次数 $n$ 上有限，如果存在有限集合 $S_n\subset\mathbb Z_{\ge0}^2$，使得当 $p+q=n$ 且 $(p,q)\notin S_n$ 时，$E_r^{p,q}=0$。

第一象限谱序列自动在每个固定总次数上有限。

**命题 M.5（谱序列有限性传播）.** 设 $E_r^{p,q}$ 是收敛到带有限过滤对象 $H^{p+q}$ 的第一象限谱序列。若某一页 $E_{r_0}^{p,q}$ 全部有限维，则每个 $H^n$ 有限维。

**证明.** 对任意 $r\ge r_0$，有

$$
E_{r+1}^{p,q}
=
\ker(d_r:E_r^{p,q}\to E_r^{p+r,q-r+1})
/
\operatorname{im}(d_r:E_r^{p-r,q+r-1}\to E_r^{p,q}).
$$

有限维向量空间的子商仍有限维，因此所有后续页有限维。固定总次数 $n$ 时，第一象限条件使可能的 $(p,q)$ 只有有限多个。收敛性给出有限过滤

$$
0=F^{m+1}H^n\subset F^mH^n\subset\cdots\subset F^0H^n=H^n
$$

并有

$$
\operatorname{gr}^p_FH^n\cong E_\infty^{p,n-p}.
$$

每个分级片有限维，由引理 M.1 得 $H^n$ 有限维。证毕。

**推论 M.6（有限非零区间）.** 若 $E_2^{p,q}$ 只在有限矩形

$$
0\le p\le a,\qquad 0\le q\le b
$$

内非零，且其中每项有限维，则所有 abutment $H^n$ 有限维，并且 $H^n=0$ 当 $n>a+b$。

**证明.** 应用命题 M.5。总次数超过 $a+b$ 时没有非零分级片，故 $H^n=0$。证毕。

## M.3 有限分解传播有限性

设 $X$ 是空间，$\mathcal A$ 是 $X$ 上阿贝尔 sheaf 范畴。

**命题 M.7（有限 acyclic 分解）.** 设

$$
0\to\mathcal F\to K^0\to K^1\to\cdots\to K^m\to0
$$

是 $\mathcal A$ 中正合列。若每个 $K^j$ 对全局截面函子 acyclic，即

$$
H^i(X,K^j)=0,\qquad i>0,
$$

并且每个 $\Gamma(X,K^j)$ 有限维，则所有

$$
H^n(X,\mathcal F)
$$

有限维，且 $H^n(X,\mathcal F)=0$ 当 $n>m$。

**证明.** 正合列给出复形 $K^\bullet$，它是 $\mathcal F$ 的 acyclic resolution。因此

$$
R\Gamma(X,\mathcal F)\simeq\Gamma(X,K^\bullet).
$$

右侧是长度 $m$ 的有限维向量空间复形。由引理 M.2，其同调有限维；由复形长度得 $n>m$ 时同调为零。证毕。

**命题 M.8（有限分解与 two-out-of-three）.** 设性质 $P(\mathcal F)$ 为“所有 $H^n(X,\mathcal F)$ 有限维”。若短正合列

$$
0\to\mathcal F'\to\mathcal F\to\mathcal F''\to0
$$

诱导长正合上同调列，且其中两项满足 $P$，则第三项满足 $P$。

**证明.** 对每个 $n$，长正合列给出精确片段

$$
H^n(\mathcal F')\to H^n(\mathcal F)\to H^n(\mathcal F'')
\to H^{n+1}(\mathcal F').
$$

若 $\mathcal F'$ 和 $\mathcal F''$ 满足 $P$，则

$$
\ker\bigl(H^n(\mathcal F)\to H^n(\mathcal F'')\bigr)
$$

是 $H^n(\mathcal F')$ 的商，有限维；而

$$
\operatorname{im}\bigl(H^n(\mathcal F)\to H^n(\mathcal F'')\bigr)
$$

是 $H^n(\mathcal F'')$ 的子空间，有限维。于是 $H^n(\mathcal F)$ 是这两个有限维空间的扩张，故有限维。

若 $\mathcal F$ 和 $\mathcal F''$ 满足 $P$，取精确片段

$$
H^{n-1}(\mathcal F'')\to H^n(\mathcal F')
\to H^n(\mathcal F)\to H^n(\mathcal F'').
$$

$H^n(\mathcal F')$ 的核是 $H^{n-1}(\mathcal F'')$ 的商，像是 $H^n(\mathcal F)$ 的子空间，因此 $H^n(\mathcal F')$ 有限维。

若 $\mathcal F'$ 和 $\mathcal F$ 满足 $P$，取精确片段

$$
H^n(\mathcal F')\to H^n(\mathcal F)
\to H^n(\mathcal F'')\to H^{n+1}(\mathcal F').
$$

$H^n(\mathcal F'')$ 的核是 $H^n(\mathcal F)$ 的商，像是 $H^{n+1}(\mathcal F')$ 的子空间，因此 $H^n(\mathcal F'')$ 有限维。证毕。

## M.4 双复形与超上同调

**命题 M.9（有限双复形的总同调）.** 设 $C^{p,q}$ 是第一象限双复形。假设：

1. $C^{p,q}=0$ 除了有限多个 $(p,q)$。
2. 每个 $C^{p,q}$ 有限维。

则总复形

$$
\operatorname{Tot}^nC=\bigoplus_{p+q=n}C^{p,q}
$$

的同调有限维。

**证明.** 每个 $\operatorname{Tot}^nC$ 是有限多个有限维空间的直和，因此有限维。总复形有界，由引理 M.2 得同调有限维。证毕。

**命题 M.10（超上同调有限性判别）.** 设 $K^\bullet$ 是有界下方且有界上方的 sheaf 复形。若存在谱序列

$$
E_2^{p,q}=H^p(X,\mathcal H^q(K^\bullet))
\Rightarrow
\mathbb H^{p+q}(X,K^\bullet),
$$

且 $E_2^{p,q}$ 在有限矩形内非零并全部有限维，则超上同调

$$
\mathbb H^n(X,K^\bullet)
$$

有限维。

**证明.** 这是推论 M.6 的直接应用。证毕。

## M.5 为什么 Stein-Cech 不自动给有限维

设 $X$ 是紧复流形，$\mathfrak U=\{U_i\}$ 是有限 Stein 覆盖，$\mathcal F$ 是相干解析层。Cartan B 给出有限交上的高上同调消失，因此 Cech 复形

$$
C^p(\mathfrak U,\mathcal F)
=
\prod_{i_0<\cdots<i_p}
\mathcal F(U_{i_0\cdots i_p})
$$

计算 $H^\bullet(X,\mathcal F)$。

但一般来说，

$$
\mathcal F(U_{i_0\cdots i_p})
$$

不是有限维向量空间。例如单位圆盘 $\Delta\subset\mathbb C$ 上

$$
\mathcal O(\Delta)
$$

包含所有收敛幂级数，作为复向量空间无限维。因此命题 M.9 不能直接应用于 Stein-Cech 复形。

**结论 M.11.** Stein-Cech 方法给出计算模型；有限维性需要额外输入。可用输入包括：

1. Grauert 有限性定理。
2. Dolbeault elliptic complex 的 Fredholm-Hodge 理论。
3. Clausen-Scholze 在 condensed/analytic 框架中的有限型结论。

## M.6 与第三卷正文的连接

第三卷第四章的有限性证明路线应理解为：

1. Dolbeault 或 Cech 模型给出 $R\Gamma(X,\mathcal F)$ 的计算复形。
2. Fredholm-Hodge 或 Grauert 输入给出该导出对象的有限性。
3. 本附录的线性代数和谱序列命题负责把有限性沿复形、过滤、分解和 exact sequence 传播。

因此，本书内部证明的是有限性的形式传播机制；有限性本身的深层来源仍是复几何输入定理。

## M.7 练习

**练习 M.1.** 给出命题 M.8 中另外两种 two-out-of-three 情况的详细证明。

**练习 M.2.** 构造一个有界复形，其每项无限维但同调为零。说明“每项有限维”不是判断同调有限维的必要条件。

**练习 M.3.** 证明单位圆盘上的 $\mathcal O(\Delta)$ 作为复向量空间无限维。

**练习 M.4.** 设 $E_2^{p,q}$ 只在 $0\le p\le2$、$0\le q\le1$ 中非零。写出 $H^2$ 的可能过滤分级片。
