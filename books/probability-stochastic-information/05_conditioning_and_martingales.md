# 条件期望、滤过与鞅

有限概率表中，给定事件 $B$ 且 $\mathbb P(B)>0$ 时可以用除法定义条件概率。若条件信息由一整个 $\sigma$-代数给出，或者单点条件事件的概率为零，这种逐点除法不再可靠。条件期望改用一族积分恒等式刻画“在现有信息下的平均值”，因此只要求几乎处处唯一。

## 5.1 条件期望的存在与唯一性

**定义 5.1（条件期望）。** 设 $(\Omega,\mathcal F,\mathbb P)$ 为概率空间，$\mathcal G\subseteq\mathcal F$ 为子 $\sigma$-代数，$X\in L^1(\Omega,\mathcal F,\mathbb P)$。若随机变量 $Y$ 满足：

1. $Y$ 是 $\mathcal G$-可测的；
2. $Y\in L^1(\Omega,\mathcal G,\mathbb P|_{\mathcal G})$；
3. 对每个 $A\in\mathcal G$，
   $$
   \int_A Y\,d\mathbb P=\int_A X\,d\mathbb P,
   $$

则称 $Y$ 为 $X$ 关于 $\mathcal G$ 的一个条件期望版本，记作 $Y=\mathbb E[X\mid\mathcal G]$。这个记号表示一个几乎处处等价类，而不是逐点唯一函数。

**定理 5.1（条件期望的存在与几乎处处唯一性）。** 对每个 $X\in L^1$ 和每个子 $\sigma$-代数 $\mathcal G\subseteq\mathcal F$，条件期望 $\mathbb E[X\mid\mathcal G]$ 存在，且任意两个版本几乎处处相等。存在性唯一使用 Radon--Nikodym 外部输入 EI-3。

**证明.** 在可测空间 $(\Omega,\mathcal G)$ 上定义两个有限测度

$$
\nu^+(A)=\int_A X^+\,d\mathbb P,
\qquad
\nu^-(A)=\int_A X^-\,d\mathbb P,
\qquad A\in\mathcal G.
$$

它们有限，因为 $X\in L^1$，并且都对 $\mathbb P|_{\mathcal G}$ 绝对连续。由 EI-3，存在非负 $\mathcal G$-可测函数 $Y^+,Y^-$，使

$$
\nu^\pm(A)=\int_A Y^\pm\,d\mathbb P
$$

对所有 $A\in\mathcal G$ 成立。取 $A=\Omega$ 可知 $Y^\pm\in L^1$。令 $Y=Y^+-Y^-$，则 $Y$ 为 $\mathcal G$-可测、可积，并且

$$
\int_A Y\,d\mathbb P
=\nu^+(A)-\nu^-(A)
=\int_A X\,d\mathbb P.
$$

故 $Y$ 是一个版本。

再设 $Y,Z$ 都是版本，令 $W=Y-Z$。对每个 $A\in\mathcal G$ 有 $\int_AW\,d\mathbb P=0$。集合 $A_+=\{W>0\}$ 属于 $\mathcal G$。若 $\mathbb P(A_+)>0$，则某个 $m\ge1$ 满足 $\mathbb P(W\ge1/m)>0$，从而

$$
\int_{A_+}W\,d\mathbb P\ge \frac1m\mathbb P(W\ge1/m)>0,
$$

矛盾。因此 $W\le0$ 几乎处处。同理对 $A_-=\{W<0\}$ 应用该论证得到 $W\ge0$ 几乎处处，故 $Y=Z$ 几乎处处。证毕。

## 5.2 条件期望演算

后续鞅计算所需的性质都由定义和唯一性推出。

**定理 5.2（线性、序、提出因子与塔性质）。** 设 $X,Y\in L^1$，$a,b\in\mathbb R$，$\mathcal H\subseteq\mathcal G\subseteq\mathcal F$。下列等式均按几乎处处理解。

1. **线性：**
   $$
   \mathbb E[aX+bY\mid\mathcal G]
   =a\mathbb E[X\mid\mathcal G]+b\mathbb E[Y\mid\mathcal G].
   $$
2. **保序与收缩：** 若 $X\le Y$ 几乎处处，则
   $$
   \mathbb E[X\mid\mathcal G]\le\mathbb E[Y\mid\mathcal G],
   $$
   并且
   $$
   |\mathbb E[X\mid\mathcal G]|
   \le\mathbb E[|X|\mid\mathcal G],
   \qquad
   \mathbb E|\mathbb E[X\mid\mathcal G]|\le\mathbb E|X|.
   $$
3. **已知量不变：** 若 $X$ 本身 $\mathcal G$-可测，则 $\mathbb E[X\mid\mathcal G]=X$。
4. **全期望：** $\mathbb E[\mathbb E[X\mid\mathcal G]]=\mathbb E[X]$。
5. **提出有界可测因子：** 若 $Z$ 为有界 $\mathcal G$-可测实随机变量，则 $ZX\in L^1$，且
   $$
   \mathbb E[ZX\mid\mathcal G]=Z\mathbb E[X\mid\mathcal G].
   $$
6. **塔性质：**
   $$
   \mathbb E[\mathbb E[X\mid\mathcal G]\mid\mathcal H]
   =\mathbb E[X\mid\mathcal H].
   $$

**证明.** 对第一条，右端是 $\mathcal G$-可测且可积的。对任意 $A\in\mathcal G$，期望的线性给出

$$
\int_A\bigl(a\mathbb E[X\mid\mathcal G]
+b\mathbb E[Y\mid\mathcal G]\bigr)\,d\mathbb P
=\int_A(aX+bY)\,d\mathbb P.
$$

唯一性给出结论。

为证保序，只需证明 $U\ge0$ 时 $V=\mathbb E[U\mid\mathcal G]\ge0$。若 $B=\{V<0\}$ 有正概率，则某个 $m$ 使 $B_m=\{V\le-1/m\}$ 有正概率，而

$$
0\le\int_{B_m}U\,d\mathbb P
=\int_{B_m}V\,d\mathbb P
\le-\frac1m\mathbb P(B_m)<0,
$$

矛盾。因此条件期望保序。由 $-|X|\le X\le|X|$、线性和保序得到绝对值不等式；对它取期望并使用第四条即可得到 $L^1$ 收缩。这里第四条可先独立由定义中取 $A=\Omega$ 得到，因而没有循环。

第三条中，$X$ 自身满足定义的三个条件，故由唯一性得到结论。第四条已经由 $A=\Omega$ 得证。

对第五条，先设 $Z=\mathbf 1_B$，其中 $B\in\mathcal G$。令 $V=\mathbb E[X\mid\mathcal G]$。对任意 $A\in\mathcal G$，

$$
\int_A\mathbf 1_BV\,d\mathbb P
=\int_{A\cap B}V\,d\mathbb P
=\int_{A\cap B}X\,d\mathbb P
=\int_A\mathbf 1_BX\,d\mathbb P.
$$

因此结论对示性函数成立，由线性推广到有界简单 $\mathcal G$-可测函数。对一般有界 $Z$，存在有界简单 $\mathcal G$-可测 $Z_n$，使 $\|Z_n-Z\|_\infty\to0$。于是

$$
\mathbb E|Z_nX-ZX|
\le\|Z_n-Z\|_\infty\mathbb E|X|\to0,
$$

并且对 $V$ 有同样估计。对每个 $A\in\mathcal G$ 令 $n\to\infty$，得到 $ZV$ 满足 $ZX$ 的条件期望定义。

最后令 $V=\mathbb E[X\mid\mathcal G]$。它可积且 $\mathcal H$-条件期望存在。对任意 $A\in\mathcal H\subseteq\mathcal G$，

$$
\int_A V\,d\mathbb P=\int_A X\,d\mathbb P.
$$

故 $\mathbb E[V\mid\mathcal H]$ 也是 $\mathbb E[X\mid\mathcal H]$ 的版本，唯一性给出塔性质。证毕。

**例 5.1（有限分割上的条件期望）。** 设 $\mathcal G=\sigma(A_1,\ldots,A_r)$，其中 $A_i$ 两两不交、并为 $\Omega$。对每个满足 $\mathbb P(A_i)>0$ 的原子令

$$
m_i=\frac{\mathbb E[X\mathbf 1_{A_i}]}{\mathbb P(A_i)}.
$$

在零概率原子上任取有限常数 $m_i$。则

$$
Y=\sum_{i=1}^r m_i\mathbf 1_{A_i}
$$

是 $\mathbb E[X\mid\mathcal G]$ 的一个版本。事实上，任意 $B\in\mathcal G$ 是若干原子的并；逐原子求和即验证积分恒等式。零概率原子上的取值说明了版本为何不能逐点唯一。

## 5.3 滤过、鞅与有界停时

**定义 5.2（滤过与适应过程）。** 离散时间滤过是递增子 $\sigma$-代数列

$$
\mathcal F_0\subseteq\mathcal F_1\subseteq\cdots\subseteq\mathcal F.
$$

过程 $(X_n)_{n\ge0}$ 称为适应于 $(\mathcal F_n)$，若 $X_n$ 对 $\mathcal F_n$ 可测。

**定义 5.3（鞅）。** 适应于 $(\mathcal F_n)$ 的实值过程 $(M_n)_{n\ge0}$ 称为鞅，若每个 $M_n\in L^1$，且对所有 $n\ge0$，

$$
\mathbb E[M_{n+1}\mid\mathcal F_n]=M_n
$$

几乎处处。

**定义 5.4（停时）。** 取值于 $\{0,1,2,\ldots\}\cup\{\infty\}$ 的随机变量 $\tau$ 称为关于 $(\mathcal F_n)$ 的停时，若 $\{\tau\le n\}\in\mathcal F_n$ 对每个 $n$ 成立。若存在确定的 $N$ 使 $\tau\le N$ 几乎处处，则称 $\tau$ 有界。

**定理 5.3（有限时域的可选停止等式）。** 设 $(M_n)_{0\le n\le N}$ 是鞅，$\tau$ 是取值于 $\{0,\ldots,N\}$ 的停时。则 $M_\tau\in L^1$，且

$$
\mathbb E[M_\tau]=\mathbb E[M_0].
$$

**证明.** 因为

$$
M_\tau=\sum_{j=0}^N M_j\mathbf 1_{\{\tau=j\}},
$$

有限和说明 $M_\tau$ 可积。另有逐点恒等式

$$
M_\tau=M_0+\sum_{k=1}^N(M_k-M_{k-1})\mathbf 1_{\{\tau\ge k\}}.
$$

停时性质给出

$$
\{\tau\ge k\}=\{\tau\le k-1\}^c\in\mathcal F_{k-1}.
$$

该示性函数有界且 $\mathcal F_{k-1}$-可测。由定理 5.2 的提出因子与全期望，

$$
\begin{aligned}
\mathbb E[(M_k-M_{k-1})\mathbf 1_{\{\tau\ge k\}}]
&=\mathbb E\!\left[
\mathbf 1_{\{\tau\ge k\}}
\mathbb E[M_k-M_{k-1}\mid\mathcal F_{k-1}]
\right]\\
&=0.
\end{aligned}
$$

对有限和取期望即得结论。证明没有使用增量有界性；关键假设是停时被确定常数 $N$ 截断。证毕。

## 5.4 条件分布的外部输入边界

条件期望总是存在，但把它同时表示成随条件值变化的概率核需要状态空间正则性。

**外部输入定理 5.4（标准 Borel 空间上的正则条件分布，EI-6）。** 设 $S,T$ 为标准 Borel 空间，$X:\Omega\to S$、$Y:\Omega\to T$ 可测。则存在从 $T$ 到 $S$ 的 Markov 核 $K$，使对每个 $A\in\mathcal B(S)$、$B\in\mathcal B(T)$，

$$
\mathbb P(X\in A,Y\in B)
=\int_B K(y,A)\,\mathcal L(Y)(dy).
$$

等价地，对每个固定 $A$，$K(Y,A)$ 是 $\mathbb E[\mathbf 1_{\{X\in A\}}\mid\sigma(Y)]$ 的一个版本。两个这样的核对 $\mathcal L(Y)$-几乎每个 $y$ 给出同一概率测度。该定理在非标准 Borel 空间上可能失败；本书不使用无空间假设的逐点条件分布。来源和未重证边界见 [SOURCES.md](SOURCES.md) 的 EI-6。

## 练习

**练习 5.1.** 设 $Y$ 取有限集合 $\mathcal Y$ 值。写出 $\mathbb E[X\mid\sigma(Y)]$ 的一个版本，并处理 $\mathbb P(Y=y)=0$ 的情形。

**练习 5.2.** 设 $X_1,X_2,\ldots$ 相互独立、可积且 $\mathbb E X_k=0$。证明 $M_n=\sum_{k=1}^nX_k$ 关于自然滤过 $\mathcal F_n=\sigma(X_1,\ldots,X_n)$ 是鞅。

**练习 5.3.** 对简单对称随机游走 $S_n=\sum_{k=1}^n\xi_k$，其中 $\xi_k$ 独立且等概率取 $\pm1$，令

$$
\tau=\min\bigl(\{n\le N:S_n=1\}\cup\{N\}\bigr).
$$

验证 $\tau$ 是有界停时，并计算 $\mathbb E S_\tau$。

**练习 5.4.** 若 $X\in L^1$ 与子 $\sigma$-代数 $\mathcal G$ 独立，证明 $\mathbb E[X\mid\mathcal G]=\mathbb E X$ 几乎处处。

**练习 5.5.** 设 $X\in L^1$，$\mathcal H\subseteq\mathcal G$。证明

$$
\mathbb E\left[\mathbb E[X\mid\mathcal G]\right]=\mathbb E X,
\qquad
\mathbb E\left[\mathbb E[X\mid\mathcal G]\mid\mathcal H\right]
=\mathbb E[X\mid\mathcal H],
$$

并分别指出这两个等式使用定义中的哪个事件族。
