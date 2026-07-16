# 信源、信道与编码定理

编码问题有两种方向。信源编码把高概率数据块压缩到较小索引集；信道编码把消息嵌入输入序列，使经过噪声后仍可恢复。两者都使用“码率”一词，但随机对象、错误概率和 converse 论证不同。本章限定于有限字母表、离散无记忆信源和离散无记忆信道，所有对数以 $2$ 为底。

对正整数 $M$，记 $[M]=\{1,\ldots,M\}$。对 $x^n=(x_1,\ldots,x_n)$ 使用同样的上标记块，而不是幂。

## 9.1 固定长度信源编码

**定义 9.1（离散无记忆信源）。** 设 $\mathcal X$ 为非空有限集合，$P$ 为其上的概率分布。离散无记忆信源（DMS）是独立同分布过程 $X_1,X_2,\ldots$，每个 $X_i\sim P$。其 $n$ 块分布为

$$
P^n(x^n)=\prod_{i=1}^nP(x_i),
\qquad x^n\in\mathcal X^n.
$$

**定义 9.2（固定长度信源块码、错误与码率）。** 一个 $(n,M)$ 固定长度信源码由函数

$$
f_n:\mathcal X^n\to[M],
\qquad
g_n:[M]\to\mathcal X^n
$$

组成。对 DMS $P$，块错误概率和码率分别为

$$
P_{e,n}^{\mathrm{src}}
=P^n\{x^n:g_n(f_n(x^n))\ne x^n\},
\qquad
R_n^{\mathrm{src}}=\frac1n\log M.
$$

这是对源分布取平均的错误概率。对确定性无失真源码，若把错误对所有 $x^n$ 取最大值，则只会得到 $0$ 或 $1$，不能表达“忽略低概率序列”的渐近压缩问题，因此本书不把它作为信源编码准则。

对 $P^n(x^n)>0$，定义块自信息

$$
\imath_P(x^n)=-\log P^n(x^n).
$$

零概率序列不由信源产生，其自信息可置为 $+\infty$。

**定理 9.1（信源编码计数 converse）。** 设 $(f_n,g_n)$ 是任意 $(n,M)$ 信源码。

1. 对每个 $\gamma>0$，
   $$
   1-P_{e,n}^{\mathrm{src}}
   \le
   \mathbb P\bigl(\imath_P(X^n)<\log M+\gamma\bigr)
   +2^{-\gamma}.
   $$
2. 若 $P$ 为有限字母表 DMS 且
   $$
   \limsup_{n\to\infty}R_n^{\mathrm{src}}<H(P),
   $$
   则 $P_{e,n}^{\mathrm{src}}\to1$。
3. 因而，任何满足 $P_{e,n}^{\mathrm{src}}\to0$ 的码序列都必须满足
   $$
   \liminf_{n\to\infty}R_n^{\mathrm{src}}\ge H(P).
   $$

**证明.** 定义正确解码集合

$$
\mathcal C_n=\{x^n:g_n(f_n(x^n))=x^n\}.
$$

$f_n$ 在 $\mathcal C_n$ 上必为单射：若 $x^n,x'^n\in\mathcal C_n$ 且 $f_n(x^n)=f_n(x'^n)$，对该索引应用 $g_n$ 得 $x^n=x'^n$。故 $|\mathcal C_n|\le M$。

令

$$
\mathcal A_\gamma
=\{x^n:\imath_P(x^n)\ge\log M+\gamma\}.
$$

对 $x^n\in\mathcal A_\gamma$ 有 $P^n(x^n)\le2^{-\log M-\gamma}=M^{-1}2^{-\gamma}$，于是

$$
P^n(\mathcal C_n\cap\mathcal A_\gamma)
\le |\mathcal C_n|M^{-1}2^{-\gamma}
\le2^{-\gamma}.
$$

又 $1-P_{e,n}^{\mathrm{src}}=P^n(\mathcal C_n)$，将 $\mathcal C_n$ 按 $\mathcal A_\gamma$ 分解即得第一条。

为证第二条，令 $\mathcal S=\{x:P(x)>0\}$。随机变量

$$
Z_i=-\log P(X_i)
$$

在有限集合 $\mathcal S$ 上有界、独立同分布，且 $\mathbb E Z_i=H(P)$。第 6 章弱大数律给出

$$
\frac1n\imath_P(X^n)=\frac1n\sum_{i=1}^nZ_i
\to H(P)
$$

依概率成立。由码率假设，可取 $\delta>0$，使充分大 $n$ 时

$$
\log M\le n(H(P)-2\delta).
$$

在第一条中取 $\gamma=n\delta$，得到

$$
1-P_{e,n}^{\mathrm{src}}
\le
\mathbb P\left(\frac1n\imath_P(X^n)<H(P)-\delta\right)
+2^{-n\delta}\to0.
$$

故错误概率趋于 $1$。若第三条失败，则存在 $\delta>0$ 和子序列使该子序列码率不超过 $H(P)-2\delta$；同一论证迫使该子序列错误概率趋于 $1$，与整体错误概率趋于 $0$ 矛盾。证毕。

**外部输入定理 9.2（DMS 固定长度信源编码 direct 部分，EI-11）。** 对有限字母表 DMS $P$，存在固定长度信源码序列满足

$$
P_{e,n}^{\mathrm{src}}\to0,
\qquad
\limsup_{n\to\infty}R_n^{\mathrm{src}}\le H(P).
$$

等价地，对每个 $R>H(P)$，充分大 $n$ 时存在码率不超过 $R$ 且错误概率趋于 $0$ 的码。该结论的 achievability 作为 EI-11 登记；本书采用下列可审查证明路线，不把路线标作书内证明：

1. 给定 $\delta>0$，定义典型集
   $$
   \mathcal T_{n,\delta}
   =\left\{x^n:
   \left|-\frac1n\log P^n(x^n)-H(P)\right|\le\delta
   \right\}.
   $$
2. 对 $Z_i=-\log P(X_i)$ 使用弱大数律，得到 $P^n(\mathcal T_{n,\delta})\to1$。
3. 每个典型序列的概率至少为 $2^{-n(H(P)+\delta)}$，故
   $$
   |\mathcal T_{n,\delta}|\le2^{n(H(P)+\delta)}.
   $$
4. 把典型序列单射编码，把所有非典型序列送到一个失败索引；错误概率至多 $P^n(\mathcal T_{n,\delta}^c)$。取 $\delta<R-H(P)$ 并处理整数取整即可得到结论。

来源定位及“未重证 direct 部分”的边界见 [SOURCES.md](SOURCES.md)。定理 9.1 与 EI-11 合在一起给出有限字母表 DMS 的最优渐近固定长度压缩率 $H(P)$。

## 9.2 前缀码与单符号平均码长

**定义 9.3（二元前缀码）。** 对非空有限字母表 $\mathcal X$，二元前缀码是单射

$$
c:\mathcal X\to\{0,1\}^*,
$$

使任意不同 $x,x'$ 的码字 $c(x)$ 都不是 $c(x')$ 的前缀。记 $\ell(x)=|c(x)|\in\{0,1,2,\ldots\}$。空码字只可能在 $|\mathcal X|=1$ 时出现。

**定理 9.3（Kraft 不等式的必要方向）。** 每个有限二元前缀码都满足

$$
\sum_{x\in\mathcal X}2^{-\ell(x)}\le1.
$$

**证明.** 令 $L=\max_x\ell(x)$。每个码字 $c(x)$ 有恰好 $2^{L-\ell(x)}$ 个长度为 $L$ 的二元延拓。前缀条件保证不同码字的延拓集合不交，否则同一个长度 $L$ 字符串会同时以两个码字开头，而其中较短者必为较长者的前缀。因此

$$
\sum_x2^{L-\ell(x)}\le2^L.
$$

除以 $2^L$ 即得结论。证毕。

**定理 9.4（前缀码平均长度下界）。** 若 $X\sim p$，$c$ 为二元前缀码，则

$$
\mathbb E[\ell(X)]\ge H(X).
$$

**证明.** 令 $K=\sum_x2^{-\ell(x)}$。字母表非空且码长有限，故 $K>0$；定理 9.3 给出 $K\le1$。定义严格正分布

$$
q(x)=\frac{2^{-\ell(x)}}K.
$$

逐项展开可得

$$
\begin{aligned}
\mathbb E[\ell(X)]-H(X)
&=\sum_xp(x)\log\frac{p(x)}{2^{-\ell(x)}}\\
&=D(p\|q)-\log K\ge0,
\end{aligned}
$$

其中使用 Gibbs 不等式和 $K\le1$。证毕。

这个定理讨论单符号可变长度、零错误前缀码；定理 9.1--9.2 讨论允许小块错误的固定长度块码。二者的编码对象不同，不能只凭相同的熵下界互相替代。

## 9.3 离散无记忆信道与块码

**定义 9.4（离散无记忆信道）。** 设 $\mathcal X,\mathcal Y$ 为非空有限集合。离散无记忆信道（DMC）是条件概率矩阵

$$
W(y\mid x)\ge0,
\qquad
\sum_{y\in\mathcal Y}W(y\mid x)=1
$$

对每个 $x\in\mathcal X$ 成立。它的 $n$ 次无记忆乘积信道为

$$
W^n(y^n\mid x^n)=\prod_{t=1}^nW(y_t\mid x_t).
$$

**定义 9.5（DMC 块码、平均错误、最大错误与码率）。** 一个确定性 $(n,M)$ 信道码由编码函数和解码函数

$$
e_n:[M]\to\mathcal X^n,
\qquad
d_n:\mathcal Y^n\to[M]
$$

组成。消息 $J$ 在 $[M]$ 上均匀，$X^n=e_n(J)$，条件于 $X^n=x^n$ 时 $Y^n\sim W^n(\cdot\mid x^n)$，估计为 $\widehat J=d_n(Y^n)$。第 $j$ 个消息的条件错误、平均错误、最大错误和码率分别为

$$
\lambda_j
=W^n\{y^n:d_n(y^n)\ne j\mid e_n(j)\},
$$

$$
P_{e,n}^{\mathrm{av}}=\frac1M\sum_{j=1}^M\lambda_j,
\qquad
P_{e,n}^{\max}=\max_{1\le j\le M}\lambda_j,
\qquad
R_n^{\mathrm{ch}}=\frac1n\log M.
$$

总有 $P_{e,n}^{\mathrm{av}}\le P_{e,n}^{\max}$。

**定义 9.6（可达率与操作容量）。** 若存在码序列使

$$
\liminf_{n\to\infty}R_n^{\mathrm{ch}}\ge R,
\qquad
P_{e,n}^{\max}\to0,
$$

则称 $R$ 在最大错误准则下可达。把最大错误换成平均错误，得到平均错误可达率。两类可达率的上确界分别记为 $C_{\max}^{\mathrm{op}}(W)$ 与 $C_{\mathrm{av}}^{\mathrm{op}}(W)$。

对输入分布 $p\in\mathcal P(\mathcal X)$，令联合分布为 $p(x)W(y\mid x)$，并把相应互信息记为 $I_p(X;Y)$。定义单字母容量

$$
C(W)=\max_{p\in\mathcal P(\mathcal X)}I_p(X;Y).
$$

最大值确实达到：有限维概率单纯形紧，而 $I_p(X;Y)=H_p(Y)-H_p(Y\mid X)$ 是 $p$ 的连续函数，端点处使用 $0\log0=0$ 的连续延拓。操作容量是可达率的上确界；下面的 direct 定理只需对严格小于 $C(W)$ 的率构造码，并不预设端点率由某个固定有限块码达到。

## 9.4 Fano 不等式与信道 converse

**定理 9.5（Fano 不等式的有限形式）。** 设 $J$ 取值于 $[M]$，$M\ge2$，$\widehat J$ 也是 $[M]$ 值估计，且

$$
P_e=\mathbb P(\widehat J\ne J).
$$

则

$$
H(J\mid\widehat J)
\le h_2(P_e)+P_e\log(M-1).
$$

这里不要求 $J$ 均匀；均匀性只在信道 converse 中用于 $H(J)=\log M$。

**证明.** 令 $E=\mathbf 1_{\{J\ne\widehat J\}}$。因为 $E$ 由 $(J,\widehat J)$ 决定，链式法则给出

$$
H(E,J\mid\widehat J)
=H(J\mid\widehat J)+H(E\mid J,\widehat J)
=H(J\mid\widehat J).
$$

按另一顺序展开，

$$
H(E,J\mid\widehat J)
=H(E\mid\widehat J)+H(J\mid E,\widehat J).
$$

第 8 章的“条件化不增加熵”给出 $H(E\mid\widehat J)\le H(E)=h_2(P_e)$。在 $E=0$ 时 $J=\widehat J$，故相应条件熵为 $0$；在 $E=1$ 且 $\widehat J=\hat j$ 时，$J$ 至多取 $[M]\setminus\{\hat j\}$ 中的 $M-1$ 个值，推论 8.5 给出条件熵至多 $\log(M-1)$。对条件事件加权，

$$
H(J\mid E,\widehat J)\le P_e\log(M-1).
$$

合并即得结论。证毕。

**引理 9.6（DMC 的 $n$ 块互信息上界）。** 若 $X^n$ 具有任意分布，且条件于 $X^n=x^n$ 时 $Y^n\sim W^n(\cdot\mid x^n)$，则

$$
I(X^n;Y^n)\le\sum_{t=1}^nI(X_t;Y_t)\le nC(W).
$$

**证明.** 熵链式法则与“条件化不增加熵”给出

$$
H(Y^n)=\sum_{t=1}^nH(Y_t\mid Y^{t-1})
\le\sum_{t=1}^nH(Y_t).
$$

另一方面，乘积信道条件下，给定 $X^n=x^n$ 后各 $Y_t$ 条件独立，且 $Y_t$ 的条件分布只依赖 $x_t$。因此

$$
\begin{aligned}
H(Y^n\mid X^n)
&=\sum_{t=1}^nH(Y_t\mid Y^{t-1},X^n)\\
&=\sum_{t=1}^nH(Y_t\mid X_t).
\end{aligned}
$$

相减得到第一不等式。每个 $X_t$ 的边缘分布都是 $\mathcal X$ 上某个输入分布，故 $I(X_t;Y_t)\le C(W)$，再求和得到第二不等式。证毕。

**定理 9.7（DMC 信道编码的弱 converse）。** 对任意 DMC $(n,M)$ 码，若 $M\ge2$，则

$$
(1-P_{e,n}^{\mathrm{av}})\log M
\le nC(W)+h_2(P_{e,n}^{\mathrm{av}}).
$$

因此，若一列码满足 $P_{e,n}^{\mathrm{av}}\to0$，则

$$
\limsup_{n\to\infty}R_n^{\mathrm{ch}}\le C(W).
$$

同一结论当然适用于最大错误概率趋零的码序列。

**证明.** 令 $J$ 均匀，按定义产生 $X^n,Y^n,\widehat J$。存在 Markov 链

$$
J\longrightarrow X^n\longrightarrow Y^n\longrightarrow\widehat J,
$$

其中首尾映射确定，中间转移为 $W^n$。数据处理不等式和引理 9.6 给出

$$
I(J;\widehat J)\le I(X^n;Y^n)\le nC(W).
$$

又 $H(J)=\log M$。由 Fano 不等式及 $\log(M-1)\le\log M$，

$$
\begin{aligned}
\log M
&=I(J;\widehat J)+H(J\mid\widehat J)\\
&\le nC(W)+h_2(P_{e,n}^{\mathrm{av}})
+P_{e,n}^{\mathrm{av}}\log M.
\end{aligned}
$$

移项得到有限块不等式。除以 $n(1-P_{e,n}^{\mathrm{av}})$ 并令 $n\to\infty$，使用 $0\le h_2\le1$，得到渐近结论。证毕。

**外部输入定理 9.8（DMC 信道编码 direct 部分，EI-12）。** 对有限输入、输出字母表的 DMC $W$ 和每个

$$
0\le R<C(W),
$$

存在确定性码序列使

$$
\liminf_{n\to\infty}R_n^{\mathrm{ch}}\ge R,
\qquad
P_{e,n}^{\max}\to0.
$$

该 achievability 作为 EI-12，不在本书写成短证明。其可审查随机编码路线如下：

1. 选输入分布 $p$ 使 $I_p(X;Y)>R$，再取 $\delta>0$ 使 $R<I_p(X;Y)-2\delta$。
2. 独立抽取 $M_n$ 个码字，每个码字服从 $p^n$。对输出 $y^n$，用信息密度阈值
   $$
   \imath(x^n;y^n)
   =\log\frac{W^n(y^n\mid x^n)}{(pW)^n(y^n)}
   >n(I_p(X;Y)-\delta)
   $$
   选择唯一候选码字；没有或不唯一时判错。
3. 真码字与输出在联合分布 $(pW)^n$ 下独立同分布。单字母信息密度在其正概率支持上有有限期望 $I_p(X;Y)$；弱大数律使真码字未过阈值的概率趋于 $0$。
4. 对独立竞争码字 $\widetilde X^n\sim p^n$ 与输出 $Y^n\sim(pW)^n$，似然比估计给出
   $$
   \mathbb P\bigl(\imath(\widetilde X^n;Y^n)
   >n(I_p-\delta)\bigr)
   \le2^{-n(I_p-\delta)}.
   $$
   并合界后，所有竞争码字造成的平均错误至多
   $(M_n-1)2^{-n(I_p-\delta)}$，在 $R<I_p-2\delta$ 时趋于 $0$。
5. 由随机码平均值小于趋零上界，选择一个确定性码。删除条件错误最大的至多一半消息，保留至少 $M_n/2$ 个码字，并使最大错误至多原平均错误的两倍；码率损失至多 $1/n$。

定理 9.7 与 EI-12 共同给出

$$
C_{\mathrm{av}}^{\mathrm{op}}(W)
=C_{\max}^{\mathrm{op}}(W)=C(W).
$$

本书证明的是识别容量所需的弱 converse；“固定 $R>C$ 时错误概率趋于 $1$”的强 converse 是更强结果，不进入本书主线。

## 9.5 BSC 与 BEC 的完整容量计算

**例 9.1（非均匀输入下的 BSC 互信息）。** 取 BSC$(1/4)$ 和输入 $X\sim\operatorname{Bernoulli}(1/4)$。输出为 $1$ 的概率是

$$
\frac14\left(1-\frac14\right)
+\frac34\cdot\frac14=\frac38.
$$

因此

$$
I(X;Y)=h_2(3/8)-h_2(1/4).
$$

这个值严格小于均匀输入取得的 $1-h_2(1/4)$；下面的命题证明后者是全体输入分布中的最大值。

**命题 9.9（二元对称信道容量）。** 对 $0\le\varepsilon\le1$，BSC$(\varepsilon)$ 的输入、输出字母表均为 $\{0,1\}$，且

$$
W(y\mid x)=
\begin{cases}
1-\varepsilon,&y=x,\\
\varepsilon,&y\ne x.
\end{cases}
$$

其容量为

$$
C_{\mathrm{BSC}}=1-h_2(\varepsilon).
$$

**证明.** 令 $q=\mathbb P(X=1)$，并写 $Y=X\oplus N$，其中 $N\sim\operatorname{Bernoulli}(\varepsilon)$ 与 $X$ 独立。则

$$
\mathbb P(Y=1)=q(1-\varepsilon)+(1-q)\varepsilon
=\varepsilon+(1-2\varepsilon)q.
$$

给定任一 $X=x$，输出行分布的熵都是 $h_2(\varepsilon)$，所以

$$
I_q(X;Y)
=h_2\bigl(\varepsilon+(1-2\varepsilon)q\bigr)
-h_2(\varepsilon)
\le1-h_2(\varepsilon).
$$

取均匀输入 $q=1/2$ 时输出也均匀，上界取到。故最大值为所述容量。$\varepsilon=0$ 或 $1$ 时信道分别原样或翻转地无噪传输，容量均为 $1$；$\varepsilon=1/2$ 时输出独立于输入，容量为 $0$。证毕。

**命题 9.10（二元擦除信道容量）。** 对 $0\le\epsilon\le1$，BEC$(\epsilon)$ 的输入字母表为 $\{0,1\}$、输出字母表为 $\{0,1,?\}$，且

$$
W(?\mid x)=\epsilon,
\qquad
W(x\mid x)=1-\epsilon,
\qquad
W(1-x\mid x)=0.
$$

其容量为

$$
C_{\mathrm{BEC}}=1-\epsilon.
$$

**证明.** 可令擦除指示 $E\sim\operatorname{Bernoulli}(\epsilon)$ 与 $X$ 独立；当 $E=0$ 时 $Y=X$，当 $E=1$ 时 $Y=?$。若 $X\sim\operatorname{Bernoulli}(q)$，未擦除时 $Y$ 完全确定 $X$，擦除时 $X$ 的后验仍为先验。因此

$$
H(X\mid Y)=(1-\epsilon)\cdot0+\epsilon H(X)
=\epsilon h_2(q),
$$

从而

$$
I_q(X;Y)=H(X)-H(X\mid Y)
=(1-\epsilon)h_2(q)\le1-\epsilon.
$$

均匀输入 $q=1/2$ 取到上界。$\epsilon=0$ 时容量为 $1$，$\epsilon=1$ 时输出恒为擦除符号，容量为 $0$。证毕。

## 练习

**练习 9.1.** 对字母概率 $(1/2,1/4,1/4)$，给出达到平均长度 $3/2$ 的二元前缀码，并验证 Kraft 和式。

**练习 9.2.** 从 Fano 不等式推出：若 $J$ 在 $[M]$ 上均匀，则

$$
I(J;\widehat J)
\ge\log M-h_2(P_e)-P_e\log(M-1).
$$

**练习 9.3.** 对 BEC$(\epsilon)$ 和 Bernoulli$(q)$ 输入，直接从联合概率表计算 $H(Y)$、$H(Y\mid X)$ 与 $I(X;Y)$，并与命题 9.10 的计算核对。

**练习 9.4.** 设信源码的正确解码集合为 $\mathcal C_n$。完整证明 $f_n|_{\mathcal C_n}$ 单射，并由此说明任意零错误 $(n,M)$ 信源码必须满足 $M\ge|\operatorname{supp}P|^n$。

**练习 9.5.** 证明任意信道码满足 $P_{e,n}^{\mathrm{av}}\le P_{e,n}^{\max}$。给出两个消息的例子，使平均错误严格小于最大错误。

**练习 9.6.** 对 BSC$(\varepsilon)$，从联合概率表验证

$$
I_q(X;Y)=h_2(\varepsilon+(1-2\varepsilon)q)-h_2(\varepsilon).
$$

**练习 9.7.** 若一个 DMC 的所有行 $W(\cdot\mid x)$ 相同，证明 $C(W)=0$。反之，证明若存在两行不同，则 $C(W)>0$。

**练习 9.8.** 从定理 9.7 推出有限块界

$$
R_n^{\mathrm{ch}}
\le\frac{C(W)+1/n}{1-P_{e,n}^{\mathrm{av}}}
$$

并说明当错误概率不趋于零时，为什么该式本身不能推出码率不超过容量。
