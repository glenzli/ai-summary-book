# 第三章：独立性、乘积与随机序列

取两个独立公平比特 $U,V$，再令 $W=U\oplus V$。四种等可能结果为

| $U$ | $V$ | $W$ | 概率 |
|---:|---:|---:|---:|
| 0 | 0 | 0 | $1/4$ |
| 0 | 1 | 1 | $1/4$ |
| 1 | 0 | 1 | $1/4$ |
| 1 | 1 | 0 | $1/4$ |

任取其中两个比特，四种二元取值都以概率 $1/4$ 出现；但三个比特只落在满足
$u\oplus v=w$ 的四个点上。这个计算迫使我们区分“两两独立”和“相互独立”，也
表明独立性描述的是联合分布怎样分解，而不是“互不影响”的日常判断。

## 3.1 独立事件与独立信息

**定义 3.1（有限独立事件）.** 事件 $A_1,\ldots,A_n\in\mathcal F$ 称为相互独立，若对每个非空 $I\subseteq\{1,\ldots,n\}$，

$$
\mathbb P\left(\bigcap_{i\in I}A_i\right)=\prod_{i\in I}\mathbb P(A_i).
$$

只检查两两交集得到的是两两独立，不足以推出相互独立。章首的比特表中，
$\mathbb P(U=0,V=0,W=0)=1/4$，而三个边缘概率之积为 $1/8$，所以三者不相互独立。

**定义 3.2（独立 $\sigma$-代数）.** 子 $\sigma$-代数 $\mathcal G_1,\ldots,\mathcal G_n\subseteq\mathcal F$ 称为相互独立，若对任意 $G_i\in\mathcal G_i$，

$$
\mathbb P(G_1\cap\cdots\cap G_n)=\prod_{i=1}^n\mathbb P(G_i).
$$

一族 $\{\mathcal G_i\}_{i\in I}$ 独立，是指其每个有限子族独立。

**定义 3.3（独立随机元素）.** 随机元素 $X_i:(\Omega,\mathcal F)\to(S_i,\mathcal S_i)$ 相互独立，是指 $\sigma(X_i)\coloneqq\{X_i^{-1}(B):B\in\mathcal S_i\}$ 相互独立。

等价地，对每个有限子族和可测集合 $B_i\in\mathcal S_i$，联合事件的概率分解为边缘概率之积。

**命题 3.4（可测函数保持独立）.** 若 $X_1,\ldots,X_n$ 相互独立，且 $f_i:(S_i,\mathcal S_i)\to(T_i,\mathcal T_i)$ 可测，则 $f_1(X_1),\ldots,f_n(X_n)$ 相互独立。

**证明.** 由复合可测性，$f_i(X_i)$ 是随机元素。对每个 $C_i\in\mathcal T_i$，

$$
\{f_i(X_i)\in C_i\}=
\{X_i\in f_i^{-1}(C_i)\},
$$

且 $f_i^{-1}(C_i)\in\mathcal S_i$。对这些事件应用 $X_i$ 的独立性，得到乘积分解。证毕。

## 3.2 乘积测度

**外部输入 3.5（有限乘积测度、Fubini--Tonelli 与有限测度唯一性）.** 设 $(S_i,\mathcal S_i,\mu_i)$ 为 $n$ 个 $\sigma$-有限测度空间。则在

$$
\left(\prod_{i=1}^nS_i,\bigotimes_{i=1}^n\mathcal S_i\right)
$$

上存在唯一乘积测度 $\bigotimes_i\mu_i$，使所有可测矩形满足

$$
\left(\bigotimes_i\mu_i\right)(B_1\times\cdots\times B_n)
=\prod_i\mu_i(B_i).
$$

对非负可测函数可按任意次序迭代积分；对可积函数亦然。本章还使用如下有限测度唯一性原则：若 $\mathcal C$ 是 $S$ 上包含 $S$ 的 $\pi$-系统，$\sigma(\mathcal C)=\mathcal S$，而有限测度 $\mu,\nu$ 在 $\mathcal C$ 上取值相同，则 $\mu=\nu$ 于 $\mathcal S$。后一句是 Dynkin $\pi$-$\lambda$ 定理的直接推论。上述结果的来源见 [SOURCES.md](SOURCES.md)；本书用它们构造独立副本、计算乘积期望并从矩形上的等式推出联合测度相等。

**命题 3.6（独立性与联合分布分解）.** 设 $X_i$ 取值于可测空间 $(S_i,\mathcal S_i)$，并给乘积配备 $\bigotimes_i\mathcal S_i$。则 $X_1,\ldots,X_n$ 独立，当且仅当

$$
\mathcal L(X_1,\ldots,X_n)=\bigotimes_{i=1}^n\mathcal L(X_i).
$$

**证明.** 映射 $X=(X_1,\ldots,X_n)$ 可测：令 $\mathcal D$ 为乘积空间中逆像属于 $\mathcal F$ 的集合族，则 $\mathcal D$ 是包含全部可测矩形的 $\sigma$-代数，故包含 $\bigotimes_i\mathcal S_i$。若联合分布为乘积测度，则对可测矩形直接得到独立性。反之，独立性说明联合分布与乘积测度在所有可测矩形上一致。可测矩形构成包含全空间并生成乘积 $\sigma$-代数的 $\pi$-系统；由外部输入 3.5 的有限测度唯一性原则，两概率测度在整个乘积 $\sigma$-代数上一致。证毕。

定义 3.3 只在矩形事件上给出乘积分解，命题却把它升级为整个乘积 $\sigma$-代数
上的测度等式。承担这一步的是有限测度在生成 $\pi$-系统上的唯一性。有了联合测度
的等式，乘积期望便可由 Fubini--Tonelli 直接计算。

**推论 3.7（独立变量乘积的期望）.** 若 $X,Y$ 独立且 $X,Y\in L^1$，则 $XY\in L^1$ 且

$$
\mathbb E[XY]=\mathbb E[X]\mathbb E[Y].
$$

**证明.** 联合分布是边缘分布的乘积。由 Tonelli 定理，

$$
\mathbb E|XY|=
\int|x||y|\,d(\mathbb P_X\otimes\mathbb P_Y)
=\mathbb E|X|\mathbb E|Y|<\infty.
$$

再由 Fubini 定理分离 $xy$ 的积分。证毕。

**推论 3.8（独立和的方差）.** 若 $X_1,\ldots,X_n$ 独立且平方可积，则

$$
\operatorname{Var}\left(\sum_{i=1}^nX_i\right)
=\sum_{i=1}^n\operatorname{Var}(X_i).
$$

**证明.** 展开中心化和的平方。交叉项为

$$
\mathbb E[(X_i-\mathbb EX_i)(X_j-\mathbb EX_j)]=0
$$

因为中心化变量仍独立且各自期望为零。余下对角项正是方差之和。证毕。

## 3.3 独立同分布序列

**定义 3.9（i.i.d.）.** 序列 $(X_n)_{n\ge1}$ 独立同分布，是指所有有限子族相互独立，且存在概率测度 $\mu$ 使每个 $\mathcal L(X_n)=\mu$。

“独立重复采样”包含两个条件。只说每次使用同一解码分布不能保证独立，因为后一次采样的分布可能依赖前一次输出、共享缓存或外部状态。

**外部输入 3.10（可数乘积概率空间）.** 对可数族概率空间 $(S_n,\mathcal S_n,\mu_n)$，在 $(\prod_{n\ge1}S_n,\bigotimes_{n\ge1}\mathcal S_n)$ 上存在唯一概率测度 $\bigotimes_{n\ge1}\mu_n$，使每个有限维边缘是对应有限乘积。坐标投影因而构成相互独立、边缘分布为 $\mu_n$ 的随机元素。这里的唯一性只针对乘积 $\sigma$-代数；若另行完备化，事件族会扩大。该结果见 [SOURCES.md](SOURCES.md) 的乘积测度资料；也可由外部输入 3.13 对不依赖历史的核推出。

## 3.4 条件生成与随机核

i.i.d. 序列的下一步分布不依赖过去。更一般的生成过程会根据已经出现的历史改变
下一步规律；此时一个固定概率测度不再足以描述单步接口，需要让概率测度随输入变化。

**定义 3.11（Markov 核）.** 从 $(S,\mathcal S)$ 到 $(T,\mathcal T)$ 的 Markov 核是函数

$$
K:S\times\mathcal T\to[0,1]
$$

满足：对每个 $x\in S$，$A\mapsto K(x,A)$ 是 $T$ 上的概率测度；对每个 $A\in\mathcal T$，$x\mapsto K(x,A)$ 是 $\mathcal S$-可测函数。

**外部输入 3.12（测度与核的乘积）.** 若 $\mu$ 是 $(S,\mathcal S)$ 上的概率测度，$K$ 是从 $S$ 到 $T$ 的 Markov 核，则在 $(S\times T,\mathcal S\otimes\mathcal T)$ 上存在唯一概率测度 $\mu\otimes K$，满足

$$
(\mu\otimes K)(B\times A)=\int_BK(x,A)\,d\mu(x),
\qquad B\in\mathcal S,\ A\in\mathcal T.
$$

此外，对每个非负 $\mathcal S\otimes\mathcal T$-可测函数 $f$，

$$
\int f\,d(\mu\otimes K)
=\int_S\left(\int_T f(x,y)K(x,dy)\right)\mu(dx).
$$

这是核积分构造的标准定理；有限状态时公式退化为有限和。本书用它把条件机制与输入分布组合为联合分布，来源见 [SOURCES.md](SOURCES.md) 的核积条目。

**外部输入 3.13（Ionescu--Tulcea 定理）.** 设 $(S_n,\mathcal S_n)_{n\ge0}$ 是可测空间，$\mu_0$ 是 $S_0$ 上的概率测度；对每个 $n\ge1$，设 $K_n$ 是从 $(\prod_{j<n}S_j,\bigotimes_{j<n}\mathcal S_j)$ 到 $(S_n,\mathcal S_n)$ 的 Markov 核。则路径空间 $(\prod_{n\ge0}S_n,\bigotimes_{n\ge0}\mathcal S_n)$ 上存在唯一概率测度，其前 $n$ 步联合分布由迭代核积

$$
\mu_0\otimes K_1\otimes\cdots\otimes K_n
$$

给出。本书第十章直接证明有限词表、有限长度版本；无限序列的测度构造不在书内重证。精确来源为 [SOURCES.md](SOURCES.md) 所列 Kallenberg 第三版定理 8.24。

## 3.5 独立不是因果无关

独立性是某个联合分布的因子分解。它不说明干预后的行为，也不排除共同机制。

例如令公平比特 $U$，并设 $X=U$、$Y=U\oplus1$。则 $X$ 与 $Y$ 完全依赖。若再引入独立公平比特 $V$ 并定义 $Z=U\oplus V$，则 $U$ 与 $Z$ 独立，尽管 $Z$ 的结构方程显式使用了 $U$。改变 $U$ 的机制会改变逐样本的 $Z$，但在公平 $V$ 下边缘分布不变。第九章将形式化这一区别。

乘积测度给出独立联合分布，随机核则允许下一步规律依赖已经观察到的状态。要把这种
依赖解释为“知道某些信息以后怎样平均”，还需要条件期望。第四章从有限划分开始建立
这一对象。

## 练习

**练习 3.1.** 验证公平比特 $U,V,U\oplus V$ 两两独立但不相互独立。

**练习 3.2.** 证明若 $X,Y$ 独立且 $f,g$ 有界可测，则 $\mathbb E[f(X)g(Y)]=\mathbb E[f(X)]\mathbb E[g(Y)]$。

**练习 3.3.** 给出两个不独立但协方差为零的随机变量，并验证计算。

**练习 3.4.** 在有限状态空间上，验证 $\gamma(x,y)=\mu(x)K(x,\{y\})$ 是概率质量函数。

**练习 3.5.** 说明共享同一个随机种子的两次“采样”为什么通常不独立。
