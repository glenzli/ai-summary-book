# 附录 D 概率与决策强化结果

卷三第 2--8 章承担概率空间、随机变量、基本不等式、独立性、收敛、熵、评分、随机算法和因果语言的课程主线。本附录只保留比正文版本更一般、需要较长构造，或被跨卷证明直接调用的结果。

本附录包括一般 Jensen 与两两独立弱大数律、有限原子信息下的条件期望、有限 KL 数据处理、有限随机核的单随机源实现，以及有限随机分配下的 ATE 识别。Markov/Chebyshev、有限 KL 非负、Brier 分解、Bayes 行动、自回归归一化、温度熵、收敛关系和机制核归一化直接在卷三相应章节证明，不在这里重复。

## D.2 一般 Jensen 不等式

**定理 D.7（Jensen 不等式）.** 设 $I\subset\mathbb R$ 是开区间，$\varphi:I\to\mathbb R$ 为凸函数，随机变量 $X$ 取值于 $I$，且 $X$ 与 $\varphi(X)$ 可积。若 $\mu=\mathbb E[X]\in I$，则

$$
\varphi(\mathbb E[X])\le \mathbb E[\varphi(X)].
$$

**证明.** 凸函数在开区间内的每一点都有支撑直线：存在 $g\in\mathbb R$ 使对所有 $x\in I$，

$$
\varphi(x)\ge \varphi(\mu)+g(x-\mu).
$$

把 $x$ 换成 $X(\omega)$，得到逐点不等式

$$
\varphi(X)\ge \varphi(\mu)+g(X-\mu).
$$

两边取期望：

$$
\mathbb E[\varphi(X)]
\ge
\varphi(\mu)+g(\mathbb E[X]-\mu)
=
\varphi(\mu).
$$

证毕。

**反例 D.8（平均不推出单次保证）.** 若一个系统在 $100$ 个同分布任务上的平均损失为 $0.01$，Markov 不等式只能给出诸如

$$
\mathbb P(\text{损失}\ge0.1)\le0.1
$$

这类依赖抽样模型的概率界。它不能推出下一次具体运行损失小于 $0.1$，也不能推出高风险样本没有集中在某个子群体。

## D.3 独立性与弱大数律

**定义 D.9（独立事件与随机变量）.** 事件 $A,B\in\mathcal F$ 独立，若

$$
\mathbb P(A\cap B)=\mathbb P(A)\mathbb P(B).
$$

事件族 $(A_i)_{i\in I}$ 相互独立，若任意有限子族 $i_1,\ldots,i_k$ 满足

$$
\mathbb P(A_{i_1}\cap\cdots\cap A_{i_k})
=
\prod_{r=1}^k\mathbb P(A_{i_r}).
$$

随机变量族 $(X_i)$ 独立，若它们生成的 $\sigma$-代数相互独立；等价地，在有限取值情形下，任意取值组合的联合概率分解为边缘概率乘积。

**反例 D.10（两两独立不等于相互独立）.** 令 $U,V$ 为独立公平比特，取值于 $\{0,1\}$，并令

$$
W=U\oplus V
$$

为异或。则 $U,V,W$ 中任意两个都是独立公平比特。但三者不相互独立，因为

$$
U\oplus V\oplus W=0
$$

恒成立，联合取值只落在四个三元组上，而不是八个三元组上。

**定理 D.11（弱大数律，有限方差版）.** 设 $X_1,X_2,\ldots$ 两两独立，且同分布，满足

$$
\mathbb E[X_i]=\mu,\qquad \operatorname{Var}(X_i)=\sigma^2<\infty.
$$

令样本均值

$$
\bar X_n=\frac1n\sum_{i=1}^n X_i.
$$

则对任意 $\epsilon>0$，

$$
\mathbb P(|\bar X_n-\mu|\ge\epsilon)\to0.
$$

**证明.** 由线性性，

$$
\mathbb E[\bar X_n]=\mu.
$$

由于两两独立，协方差项为零：

$$
\operatorname{Var}(\bar X_n)
=
\operatorname{Var}\left(\frac1n\sum_{i=1}^n X_i\right)
=
\frac1{n^2}\sum_{i=1}^n\operatorname{Var}(X_i)
=
\frac{\sigma^2}{n}.
$$

对 $\bar X_n$ 应用卷三定理 P2.13 的 Chebyshev 不等式：

$$
\mathbb P(|\bar X_n-\mu|\ge\epsilon)
\le
\frac{\operatorname{Var}(\bar X_n)}{\epsilon^2}
=
\frac{\sigma^2}{n\epsilon^2}.
$$

右侧随 $n\to\infty$ 收敛到 $0$。证毕。

**边界 D.12.** 独立性是联合分布的分解性质，不是因果无关。弱大数律说明样本均值在概率意义下接近期望；它不说明单个样本正确，不说明几乎处处收敛，不处理重尾无限方差情形，也不消除评测集与部署分布不一致的问题。

## D.12 有限信息下的条件期望与 Bayes 公式

先处理有限 $\sigma$-代数。这样可以把条件期望直接写成有限个原子上的平均，而不暗中调用一般的 Radon--Nikodym 定理。

**引理 D.36（有限 $\sigma$-代数的原子分割）.** 若 $\mathcal G$ 是 $\Omega$ 上的有限 $\sigma$-代数，则存在有限个两两不交的非空集合 $G_1,\ldots,G_m\in\mathcal G$，其并为 $\Omega$，且 $\mathcal G$ 中每个集合都是某些 $G_j$ 的并。这样的 $G_j$ 称为 $\mathcal G$ 的原子。

**证明.** 对 $\omega,\omega'\in\Omega$，定义

$$
\omega\sim\omega'
\quad\Longleftrightarrow\quad
\text{对所有 }A\in\mathcal G,
\ \omega\in A\text{ 当且仅当 }\omega'\in A.
$$

这是等价关系。因为 $\mathcal G$ 只有有限多个集合，每个等价类都可写成有限个 $\mathcal G$ 中集合或其补集的交，因此属于 $\mathcal G$。等价类至多有 $2^{|\mathcal G|}$ 个，故只有有限个非空类；它们两两不交且覆盖 $\Omega$。按等价关系的定义，任意 $A\in\mathcal G$ 要么包含某个等价类的全部点，要么与该类不交，所以 $A$ 是若干等价类的并。证毕。

**定理 D.37（有限 $\sigma$-代数上的条件期望存在唯一）.** 设 $X$ 是可积实随机变量，$\mathcal G\subseteq\mathcal F$ 是有限 $\sigma$-代数，其原子为 $G_1,\ldots,G_m$。定义

$$
Y
=
\sum_{j:\,\mathbb P(G_j)>0}
\frac{\mathbb E[X\mathbf 1_{G_j}]}{\mathbb P(G_j)}
\mathbf 1_{G_j},
$$

并在零概率原子上令 $Y=0$。则 $Y$ 可积且 $\mathcal G$-可测，并对每个 $A\in\mathcal G$ 满足

$$
\int_A Y\,d\mathbb P=\int_A X\,d\mathbb P.
$$

满足这两个条件的随机变量在几乎处处意义下唯一，记为 $\mathbb E[X\mid\mathcal G]$。

**证明.** $Y$ 在每个原子上为常数，所以由引理 D.36 可知它 $\mathcal G$-可测。并且

$$
\mathbb E[|Y|]
=\sum_{j:\,\mathbb P(G_j)>0}
\left|\mathbb E[X\mathbf 1_{G_j}]\right|
\le\sum_j\mathbb E[|X|\mathbf 1_{G_j}]
=\mathbb E[|X|]<\infty.
$$

任意 $A\in\mathcal G$ 是若干原子的并。对正概率原子，定义直接给出

$$
\int_{G_j}Y\,d\mathbb P
=\mathbb E[X\mathbf 1_{G_j}]
=\int_{G_j}X\,d\mathbb P;
$$

对零概率原子，两边都为 $0$。对组成 $A$ 的原子求和，即得积分恒等式。

为证唯一，设 $Y_1,Y_2$ 都满足条件，令 $W=Y_1-Y_2$。则 $W$ 可积且 $\mathcal G$-可测，并对每个 $A\in\mathcal G$ 有 $\int_AW\,d\mathbb P=0$。集合 $A_+=\{W>0\}$ 属于 $\mathcal G$。若 $\mathbb P(A_+)>0$，则因

$$
A_+=\bigcup_{k\ge1}\{W\ge1/k\},
$$

至少存在 $k$ 使 $\mathbb P(W\ge1/k)>0$，从而

$$
\int_{A_+}W\,d\mathbb P
\ge\frac1k\mathbb P(W\ge1/k)>0,
$$

与积分恒等式矛盾。因此 $\mathbb P(W>0)=0$。对 $-W$ 同理，$\mathbb P(W<0)=0$，故 $Y_1=Y_2$ 几乎处处。证毕。

**定理 D.38（有限条件期望的线性、保序与塔式性质）.** 设 $X,Y$ 可积，$a,b\in\mathbb R$，且 $\mathcal H\subseteq\mathcal G\subseteq\mathcal F$ 都是有限 $\sigma$-代数。则：

1. 线性：

   $$
   \mathbb E[aX+bY\mid\mathcal G]
   =a\mathbb E[X\mid\mathcal G]
   +b\mathbb E[Y\mid\mathcal G]
   \quad\text{a.s.};
   $$

2. 保序：若 $X\le Y$ a.s.，则

   $$
   \mathbb E[X\mid\mathcal G]
   \le\mathbb E[Y\mid\mathcal G]
   \quad\text{a.s.};
   $$

3. 塔式性质：

   $$
   \mathbb E[\mathbb E[X\mid\mathcal G]\mid\mathcal H]
   =\mathbb E[X\mid\mathcal H]
   \quad\text{a.s.}
   $$

**证明.** 令 $X_{\mathcal G}=\mathbb E[X\mid\mathcal G]$、$Y_{\mathcal G}=\mathbb E[Y\mid\mathcal G]$。随机变量 $aX_{\mathcal G}+bY_{\mathcal G}$ 可积且 $\mathcal G$-可测。对任意 $A\in\mathcal G$，

$$
\begin{aligned}
\int_A(aX_{\mathcal G}+bY_{\mathcal G})\,d\mathbb P
&=a\int_AX\,d\mathbb P+b\int_AY\,d\mathbb P\\
&=\int_A(aX+bY)\,d\mathbb P.
\end{aligned}
$$

由定理 D.37 的唯一性得到线性。

若 $X\le Y$ a.s.，考察 $\mathcal G$ 的任一正概率原子 $G_j$。由定理 D.37 的显式公式，

$$
\mathbb E[X\mid\mathcal G]\big|_{G_j}
=\frac{\mathbb E[X\mathbf 1_{G_j}]}{\mathbb P(G_j)}
\le
\frac{\mathbb E[Y\mathbf 1_{G_j}]}{\mathbb P(G_j)}
=\mathbb E[Y\mid\mathcal G]\big|_{G_j}.
$$

所有零概率原子的并仍为零测集，因此保序性几乎处处成立。

最后，$\mathbb E[X\mid\mathcal G]$ 可积。对任意 $A\in\mathcal H$，因为 $\mathcal H\subseteq\mathcal G$，有

$$
\int_A\mathbb E[X\mid\mathcal G]\,d\mathbb P
=\int_AX\,d\mathbb P.
$$

所以 $\mathbb E[\mathbb E[X\mid\mathcal G]\mid\mathcal H]$ 满足 $\mathbb E[X\mid\mathcal H]$ 的定义；由唯一性，两者几乎处处相等。证毕。

**定理 D.39（有限 Bayes 公式）.** 设状态变量 $\Theta$ 与观测变量 $Y$ 都只有有限值域。记

$$
\pi(\theta)=\mathbb P(\Theta=\theta),
\qquad
L(y\mid\theta)=\mathbb P(Y=y\mid\Theta=\theta)
$$

其中当 $\pi(\theta)>0$ 时，$L(y\mid\theta)$ 是通常的条件概率；当 $\pi(\theta)=0$ 时约定 $L(y\mid\theta)=0$。若 $\mathbb P(Y=y)>0$，则

$$
\mathbb P(\Theta=\theta\mid Y=y)
=
\frac{L(y\mid\theta)\pi(\theta)}
{\sum_{\vartheta}L(y\mid\vartheta)\pi(\vartheta)}.
$$

**证明.** 由有限条件概率定义，

$$
\mathbb P(\Theta=\theta\mid Y=y)
=\frac{\mathbb P(\Theta=\theta,Y=y)}{\mathbb P(Y=y)}.
$$

若 $\pi(\theta)>0$，乘法公式给出

$$
\mathbb P(\Theta=\theta,Y=y)
=L(y\mid\theta)\pi(\theta);
$$

若 $\pi(\theta)=0$，联合概率和右侧都为 $0$。状态事件 $\{\Theta=\vartheta\}$ 构成有限分割，所以

$$
\mathbb P(Y=y)
=\sum_\vartheta\mathbb P(Y=y,\Theta=\vartheta)
=\sum_\vartheta L(y\mid\vartheta)\pi(\vartheta).
$$

分母因假设为正。代入即得结论。证毕。

**外部输入（一般条件期望存在性）.** 在任意概率空间上，若 $X$ 可积且 $\mathcal G\subseteq\mathcal F$ 是任意子 $\sigma$-代数，则存在可积、$\mathcal G$-可测的 $Y$，使

$$
\int_A Y\,d\mathbb P=\int_A X\,d\mathbb P
\quad\text{对所有 }A\in\mathcal G,
$$

并且 $Y$ 在 a.s. 意义下唯一。一般证明依赖 Radon--Nikodym 定理，本附录把它作为外部输入。定理 D.37--D.38 已经在有限 $\sigma$-代数范围内给出了不依赖该输入的完整证明。

## D.15 有限分布上的 log-sum 与数据处理

本节所有字母表均为有限集合，所有 KL 都采用 D.4 的扩展实数约定。

**定理 D.43（log-sum 不等式）.** 对有限个非负数 $a_i,b_i$，令 $A=\sum_i a_i$、$B=\sum_i b_i$。采用 $0\log(0/b)=0$ 以及 $a\log(a/0)=+\infty$（$a>0$）的约定，则

$$
\sum_i a_i\log\frac{a_i}{b_i}
\ge
A\log\frac AB
$$

在两边有定义的扩展实数意义下成立。特别地，当 $A,B>0$ 且 $a_i>0$ 蕴含 $b_i>0$ 时，两边都有限。

**证明.** 若 $A=0$，则所有 $a_i=0$，两边都按约定为 $0$。若 $A>0$ 而 $B=0$，则至少一个 $a_i>0$ 且所有 $b_i=0$，两边都为 $+\infty$。现设 $A,B>0$。若某个 $a_i>0,b_i=0$，左侧为 $+\infty$，不等式成立。否则定义概率分布

$$
p_i=\frac{a_i}{A},
\qquad
q_i=\frac{b_i}{B}.
$$

直接整理得

$$
\begin{aligned}
\sum_i a_i\log\frac{a_i}{b_i}
-A\log\frac AB
&=A\sum_i p_i\log\frac{p_i}{q_i}\\
&=A D_{\mathrm{KL}}(p\Vert q)\\
&\ge0,
\end{aligned}
$$

最后一步由卷三定理 P6.5 的 Gibbs 不等式。证毕。

**定义 D.44（有限 Markov 核）.** 从有限集合 $\mathcal X$ 到有限集合 $\mathcal Y$ 的 Markov 核是数族

$$
K(y\mid x)\ge0,
\qquad
\sum_{y\in\mathcal Y}K(y\mid x)=1
$$

对每个 $x\in\mathcal X$ 成立。分布 $p$ 经过 $K$ 后的输出分布为

$$
(pK)(y)=\sum_xp(x)K(y\mid x).
$$

**定理 D.45（有限 KL 的数据处理不等式）.** 对有限集合上的分布 $p,q$ 和 Markov 核 $K$，

$$
D_{\mathrm{KL}}(pK\Vert qK)
\le D_{\mathrm{KL}}(p\Vert q).
$$

**证明.** 若 $D_{\mathrm{KL}}(p\Vert q)=+\infty$，结论平凡。以下设其有限，因此 $p(x)>0$ 时必有 $q(x)>0$。对每个固定 $y$，把 log-sum 不等式用于

$$
a_x=p(x)K(y\mid x),
\qquad
b_x=q(x)K(y\mid x).
$$

得到

$$
(pK)(y)\log\frac{(pK)(y)}{(qK)(y)}
\le
\sum_xp(x)K(y\mid x)
\log\frac{p(x)K(y\mid x)}{q(x)K(y\mid x)}.
$$

当 $p(x)K(y\mid x)=0$ 时相应项按约定为 $0$；当它为正时，有限输入 KL 保证 $q(x)>0$，且 $K(y\mid x)>0$，故核因子可以约去。对 $y$ 求和并交换有限求和次序，

$$
\begin{aligned}
D_{\mathrm{KL}}(pK\Vert qK)
&\le
\sum_{x,y}p(x)K(y\mid x)\log\frac{p(x)}{q(x)}\\
&=\sum_xp(x)\log\frac{p(x)}{q(x)}
\sum_yK(y\mid x)\\
&=D_{\mathrm{KL}}(p\Vert q).
\end{aligned}
$$

证毕。

**定理 D.46（有限双射下 KL 不变）.** 若 $\phi:\mathcal X\to\mathcal Y$ 是有限集合之间的双射，并定义

$$
p^\phi(y)=p(\phi^{-1}(y)),
\qquad
q^\phi(y)=q(\phi^{-1}(y)),
$$

则

$$
D_{\mathrm{KL}}(p^\phi\Vert q^\phi)
=D_{\mathrm{KL}}(p\Vert q).
$$

**证明.** 双射只重命名求和指标，并保持零概率支撑关系。因此

$$
\begin{aligned}
D_{\mathrm{KL}}(p^\phi\Vert q^\phi)
&=\sum_{y\in\mathcal Y}
p(\phi^{-1}(y))
\log\frac{p(\phi^{-1}(y))}{q(\phi^{-1}(y))}\\
&=\sum_{x\in\mathcal X}p(x)\log\frac{p(x)}{q(x)}\\
&=D_{\mathrm{KL}}(p\Vert q).
\end{aligned}
$$

若任一侧因支撑不匹配为 $+\infty$，双射使另一侧出现完全对应的不匹配，等式仍成立。证毕。

数据处理不等式允许随机映射丢失区分 $p$ 与 $q$ 的信息；双射有确定性逆映射，所以没有信息丢失，KL 恰好保持。

## D.16 有限 Markov 核的单随机源实现

**定理 D.47（单一均匀随机变量的逆变换实现）.** 设 $K$ 是从有限集合 $\mathcal X$ 到有限集合

$$
\mathcal Y=\{y_1,\ldots,y_m\}
$$

的 Markov 核。存在确定性函数

$$
F:\mathcal X\times[0,1)\to\mathcal Y
$$

使得：若 $U\sim\operatorname{Unif}[0,1)$，则对每个固定 $x$，

$$
\mathbb P(F(x,U)=y_j)=K(y_j\mid x).
$$

更一般地，若随机变量 $X$ 与同一个 $U$ 独立，则

$$
\mathbb P(F(X,U)=y_j\mid X=x)=K(y_j\mid x)
$$

对每个满足 $\mathbb P(X=x)>0$ 的 $x$ 成立。

**证明.** 对每个 $x$ 定义累计和

$$
c_0(x)=0,
\qquad
c_j(x)=\sum_{r=1}^jK(y_r\mid x),
\quad j=1,\ldots,m.
$$

则 $0=c_0(x)\le\cdots\le c_m(x)=1$。令

$$
F(x,u)=y_j
\quad\Longleftrightarrow\quad
c_{j-1}(x)\le u<c_j(x).
$$

零长度区间对应零概率输出，不影响定义；所有半开区间两两不交并覆盖 $[0,1)$。均匀分布按区间长度赋概率，所以

$$
\mathbb P(F(x,U)=y_j)
=c_j(x)-c_{j-1}(x)
=K(y_j\mid x).
$$

若 $X$ 与 $U$ 独立，则给定 $X=x$ 不改变 $U$ 的均匀分布，故同一计算给出条件分布公式。因为 $\mathcal X,\mathcal Y$ 有限，而每个逆像是有限个形如 $\{x\}\times[c_{j-1}(x),c_j(x))$ 的并，$F$ 对离散 $\sigma$-代数与 Borel $\sigma$-代数可测。证毕。

一个 $U$ 已足以实现一次有限核采样；定理并未声称重复调用时可以反复复用同一 $U$ 而仍得到条件独立样本。若需要独立重复采样，必须提供相应的独立随机源或一个已证明等价的随机流构造。

## D.17 有限随机分配下的 ATE 识别

设协变量 $X$ 取有限值域 $\mathcal X$，处理 $Z\in\{0,1\}$，潜在结果 $Y(1),Y(0)$ 是有限值域实随机变量，观察结果为 $Y$。定义平均处理效应

$$
\operatorname{ATE}=\mathbb E[Y(1)-Y(0)].
$$

以下三个条件是识别结论的一部分，而不是从观测数据自动推出的事实。

1. **条件独立性（随机分配）**：

   $$
   (Y(1),Y(0))\perp Z\mid X.
   $$

   在有限值域中，这表示对每个正概率协变量层 $x$，给定 $X=x$ 后，处理分配与两个潜在结果的联合分布分解。

2. **正性**：对每个 $\mathbb P(X=x)>0$ 的 $x$，

   $$
   0<e(x):=\mathbb P(Z=1\mid X=x)<1.
   $$

3. **一致性**：

   $$
   Y=Y(Z)=ZY(1)+(1-Z)Y(0)
   \quad\text{a.s.}
   $$

**定理 D.48（有限分层随机分配下的 ATE 识别）.** 在上述条件下，

$$
\operatorname{ATE}
=
\sum_{x\in\mathcal X}\mathbb P(X=x)
\left[
\mathbb E(Y\mid Z=1,X=x)
-
\mathbb E(Y\mid Z=0,X=x)
\right],
$$

其中只对 $\mathbb P(X=x)>0$ 的层求和。等价地，

$$
\operatorname{ATE}
=
\mathbb E\left[
\frac{ZY}{e(X)}
-
\frac{(1-Z)Y}{1-e(X)}
\right].
$$

**证明.** 固定一个正概率层 $x$。正性保证事件 $\{Z=z,X=x\}$ 对 $z=0,1$ 都有正概率，所以相应条件均值有定义。设 $S_z$ 是 $Y(z)$ 的有限值域。由条件独立性，对每个 $y\in S_z$，

$$
\mathbb P(Y(z)=y\mid Z=z,X=x)
=\mathbb P(Y(z)=y\mid X=x).
$$

乘以 $y$ 并对有限集合 $S_z$ 求和，得到

$$
\mathbb E[Y(z)\mid Z=z,X=x]
=\mathbb E[Y(z)\mid X=x].
$$

由一致性，在事件 $\{Z=z\}$ 上有 $Y=Y(z)$，因此

$$
\mathbb E[Y\mid Z=z,X=x]
=\mathbb E[Y(z)\mid Z=z,X=x]
=\mathbb E[Y(z)\mid X=x].
$$

对 $z=1$ 与 $z=0$ 相减，再按 $X$ 的有限分布求和，得到

$$
\begin{aligned}
&\sum_x\mathbb P(X=x)
\left[
\mathbb E(Y\mid Z=1,X=x)
-\mathbb E(Y\mid Z=0,X=x)
\right]\\
&\quad=
\sum_x\mathbb P(X=x)
\left[
\mathbb E(Y(1)\mid X=x)
-\mathbb E(Y(0)\mid X=x)
\right]\\
&\quad=\mathbb E[Y(1)-Y(0)].
\end{aligned}
$$

这证明了分层公式。

再证明逆概率加权公式。固定正概率层 $x$，利用一致性、条件独立性和 $e(x)>0$，

$$
\begin{aligned}
\mathbb E\left[\frac{ZY}{e(X)}\,
\middle|\,X=x\right]
&=\frac1{e(x)}
\mathbb E[ZY(1)\mid X=x]\\
&=\frac1{e(x)}
\mathbb P(Z=1\mid X=x)
\mathbb E[Y(1)\mid Z=1,X=x]\\
&=\mathbb E[Y(1)\mid X=x].
\end{aligned}
$$

同理，由 $1-e(x)>0$，

$$
\mathbb E\left[\frac{(1-Z)Y}{1-e(X)}\,
\middle|\,X=x\right]
=\mathbb E[Y(0)\mid X=x].
$$

按有限个 $x$ 求全期望并相减，即得第二个公式。证毕。

识别公式只说明：在三项假设成立时，观测分布唯一决定所写的 ATE 泛函。若存在未记录的共同原因、某层只接受一种处理、实际处理偏离记录处理，或不同处理版本不能由一个二元 $Z$ 表示，则证明中的等号会在对应步骤失效。

## 练习

**练习 D.1.** 给出一个非负随机变量，使 Markov 不等式在某个 $a$ 上取等号；再给出一个例子说明均值很小仍可能存在低概率大损失。

**练习 D.2.** 用 Chebyshev 不等式证明：若 $X_i$ 两两独立但不要求同分布，且 $\operatorname{Var}(X_i)\le C$、$\mathbb E[X_i]=\mu_i$，则 $\frac1n\sum_i(X_i-\mu_i)$ 依概率收敛到 $0$。

**练习 D.3.** 对三分类分布 $p=(1/2,1/3,1/6)$，写出期望对数损失与 KL 项的分解，并说明为什么报告 $q=p$ 最优。

**练习 D.4.** 固定 logits $(2,0,-1)$，写出 $q_T$、$H(T)$ 和 $\frac{dH}{dT}$ 的表达式，并判断 $T\to0$ 与 $T\to\infty$ 的熵极限。

**练习 D.5.** 设 $\mathcal G$ 由有限分割 $G_1,\ldots,G_m$ 生成。直接使用原子公式计算 $\mathbb E[X\mid\mathcal G]$，并证明若 $X$ 本身 $\mathcal G$-可测，则 $\mathbb E[X\mid\mathcal G]=X$ a.s.。

**练习 D.6.** 构造一个依概率收敛但不在 $L^1$ 中收敛的随机变量序列；显式计算其尾概率与一阶绝对矩，并指出为什么卷三第 5 章从 $L^1$ 收敛推出依概率收敛时使用的 Markov 不等式不能反向使用。

**练习 D.7.** 对给定的二元输入分布 $p,q$ 和二元 Markov 核 $K$，显式计算 $D_{\mathrm{KL}}(p\Vert q)$ 与 $D_{\mathrm{KL}}(pK\Vert qK)$，并找出数据处理不等式严格取小于号和取等号的各一个例子。

**练习 D.8.** 在三个协变量层的随机分配模型中，给定各层概率、处理倾向与两个处理组的条件均值，分别用分层公式和逆概率加权公式表示 ATE；再说明若某一正概率层满足 $e(x)=0$，证明的哪一步失效。
