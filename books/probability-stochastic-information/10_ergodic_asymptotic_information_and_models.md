# 平稳性、遍历性与渐近信息

单符号熵不能描述相关过程的长期信息量。块熵 $H(X_1^n)$ 记录前 $n$ 个坐标的联合不确定性；熵率把它按长度归一化。平稳性保证这种平均量存在，遍历性则控制一条典型样本路径上的时间平均。Shannon--McMillan--Breiman 定理进一步处理块概率的对数；它不是 Birkhoff 定理的一句改写。

本章只讨论取值于非空有限字母表 $\mathcal X$ 的单边离散时间过程 $(X_n)_{n\ge1}$。记 $X_i^j=(X_i,\ldots,X_j)$。

## 10.1 平稳性与遍历性

**定义 10.1（平稳过程）。** 过程 $(X_n)_{n\ge1}$ 称为严格平稳，若对所有 $m\ge1$、$k\ge0$，

$$
(X_1,\ldots,X_m)\overset d=
(X_{1+k},\ldots,X_{m+k}).
$$

令 $\mu=\mathcal L((X_n)_{n\ge1})$ 为路径空间 $\mathcal X^{\mathbb N}$ 上的分布，并令左移

$$
T(x_1,x_2,x_3,\ldots)=(x_2,x_3,x_4,\ldots).
$$

平稳性等价于 $\mu(T^{-1}A)=\mu(A)$ 对每个路径可测集 $A$ 成立。

**定义 10.2（遍历过程）。** 平稳过程称为遍历，若每个模零不变事件 $A$，即

$$
\mu(T^{-1}A\mathbin{\triangle}A)=0,
$$

都满足 $\mu(A)\in\{0,1\}$。换言之，左移的不变 $\sigma$-代数在模零意义下平凡。

平稳并不推出遍历。令隐藏变量 $\Theta$ 等概率取 $0$ 与 $1$，并令 $X_n=\Theta$ 对所有 $n$。这个过程平稳，但事件“整条路径恒为 $1$”在左移下不变且概率为 $1/2$。沿单条路径的符号 $1$ 频率收敛到随机变量 $\Theta$，而不是总体均值 $1/2$。

## 10.2 熵率存在性

**定义 10.3（块熵与熵率）。** 定义

$$
H_n=H(X_1^n).
$$

若极限存在，则

$$
h(X)=\lim_{n\to\infty}\frac{H_n}{n}
$$

称为过程的熵率。有限字母表保证 $0\le H_n\le n\log|\mathcal X|$，但对非平稳过程，这个商仍可能没有极限。

**例 10.1（非平稳过程的熵率可不存在）。** 取相互独立的 $X_n$，每个 $X_n$ 或恒为 $0$，或均匀取 $\{0,1\}$。把时间分成交替的“均匀块”和“确定块”，递归选择第 $k$ 块长度 $L_k$，使

$$
L_k\ge k\sum_{j<k}L_j.
$$

奇数块使用均匀变量，偶数块使用确定变量。独立性给出 $H(X_1^n)$ 等于前 $n$ 个位置中均匀变量的个数。在每个奇数块末端，该比例至少为 $k/(k+1)$；在每个偶数块末端，它至多为 $1/(k+1)$。因此 $H_n/n$ 有趋近 $1$ 和趋近 $0$ 的两个子序列，极限不存在。

平稳性排除了这种熵密度的人为振荡。

**定理 10.1（平稳有限字母表过程的熵率存在）。** 若 $(X_n)_{n\ge1}$ 平稳且取值于有限字母表，则熵率存在。令

$$
c_1=H(X_1),
\qquad
c_n=H(X_n\mid X_1^{n-1})\quad(n\ge2).
$$

则 $(c_n)$ 单调不增，且

$$
h(X)=\lim_{n\to\infty}c_n
=\inf_{n\ge1}c_n
=\lim_{n\to\infty}\frac1n\sum_{k=1}^nc_k.
$$

此外，块熵次可加：$H_{m+n}\le H_m+H_n$。

**证明.** 由“增加条件不增加条件熵”和平稳性，

$$
\begin{aligned}
c_{n+1}
&=H(X_{n+1}\mid X_1^n)\\
&\le H(X_{n+1}\mid X_2^n)\\
&=H(X_n\mid X_1^{n-1})=c_n.
\end{aligned}
$$

因此 $(c_n)$ 单调下降且下界为 $0$，故收敛到 $c_*=\inf_nc_n$。熵链式法则给出

$$
H_n=\sum_{k=1}^nc_k.
$$

收敛序列的 Cesaro 平均收敛到同一极限，故 $H_n/n\to c_*$，这证明熵率存在并给出公式。

最后，

$$
\begin{aligned}
H_{m+n}
&=H(X_1^m)+H(X_{m+1}^{m+n}\mid X_1^m)\\
&\le H_m+H(X_{m+1}^{m+n})
=H_m+H_n,
\end{aligned}
$$

其中最后一步使用平稳性。证毕。

这个定理只需平稳，不需遍历。遍历性将在样本路径结论中出现。

## 10.3 平稳 Markov 链的熵率

**定理 10.2（平稳有限 Markov 链的熵率公式）。** 设 $(X_n)_{n\ge1}$ 是有限状态平稳 Markov 链，不变分布为 $\pi$，转移矩阵为 $P=(P(i,j))$。则

$$
h(X)=H(X_2\mid X_1)
=\sum_i\pi_iH(P(i,\cdot))
=-\sum_{i,j}\pi_iP(i,j)\log P(i,j).
$$

公式不要求链不可约或非周期；平稳初始分布已经足够。

**证明.** 对 $n\ge2$，Markov 性给出

$$
H(X_n\mid X_1^{n-1})=H(X_n\mid X_{n-1}).
$$

平稳性使右端不依赖 $n$。按 $X_{n-1}=i$ 条件化，

$$
H(X_n\mid X_{n-1})
=\sum_i\pi_iH(P(i,\cdot)).
$$

定理 10.1 说明熵率等于这些条件熵的极限，因此得到前两个等式。展开每一行熵得到最后一个等式；约定 $0\log0=0$。证毕。

对第 7 章二状态链

$$
P=\begin{pmatrix}1-b&b\\1-a&a\end{pmatrix},
\qquad
\pi_0=\frac{1-a}{1-a+b},\quad
\pi_1=\frac b{1-a+b},
$$

若 $0<a,b<1$，则

$$
h(X)=\pi_0h_2(b)+\pi_1h_2(a).
$$

## 10.4 Birkhoff 定理：时间平均

**外部输入定理 10.3（Birkhoff 点态遍历定理，EI-9）。** 设 $(\Omega,\mathcal F,\mathbb P)$ 为概率空间，$T:\Omega\to\Omega$ 可测且保测，即 $\mathbb P(T^{-1}A)=\mathbb P(A)$ 对所有 $A\in\mathcal F$ 成立。令

$$
\mathcal I
=\{A\in\mathcal F:\mathbb P(T^{-1}A\mathbin{\triangle}A)=0\}
$$

为模零不变 $\sigma$-代数。对每个 $f\in L^1(\mathbb P)$，

$$
\frac1n\sum_{k=0}^{n-1}f\circ T^k
\longrightarrow\mathbb E[f\mid\mathcal I]
\qquad\mathbb P\text{-几乎处处}.
$$

若 $T$ 遍历，则 $\mathcal I$ 模零平凡，极限为常数 $\mathbb E f$。本书用 EI-9 把平稳遍历过程的时间频率连接到单时刻概率，不重证最大遍历不等式与点态收敛论证。

对平稳过程的路径空间取 $f(x)=\mathbf 1_{\{x_1=a\}}$。若过程遍历，EI-9 给出每个 $a\in\mathcal X$ 的经验频率结论

$$
\frac1n\sum_{k=1}^n\mathbf 1_{\{X_k=a\}}
\longrightarrow\mathbb P(X_1=a)
\qquad\text{几乎处处}.
$$

这仍不是块概率的 AEP；函数 $-\log P(X_1^n)$ 随 $n$ 改变，不能直接把它当作一个固定 $L^1$ 函数套入 Birkhoff 定理。

## 10.5 Shannon--McMillan--Breiman 与 AEP

**外部输入定理 10.4（Shannon--McMillan--Breiman，EI-10）。** 设 $(X_n)_{n\ge1}$ 是有限字母表上的平稳遍历过程，$h(X)$ 为定理 10.1 保证存在的熵率。对 $x^n\in\mathcal X^n$ 记

$$
p_n(x^n)=\mathbb P(X_1^n=x^n).
$$

则在 $p_n(X_1^n)>0$ 的概率一事件上，

$$
-\frac1n\log p_n(X_1^n)
\longrightarrow h(X)
\qquad\text{几乎处处}.
$$

该定理独立登记为 EI-10。它比 Birkhoff 的固定函数时间平均更专门，也需要针对条件信息函数的收敛论证；本书不把两者合并为一个“遍历/AEP 定理”。来源与版本定位见 [SOURCES.md](SOURCES.md)。

**推论 10.5（有限字母表 AEP 与典型集大小）。** 在 EI-10 的假设下，对 $\delta>0$ 定义

$$
\mathcal T_{n,\delta}
=\left\{x^n:
2^{-n(h+\delta)}
\le p_n(x^n)
\le2^{-n(h-\delta)}
\right\}.
$$

则

$$
\mathbb P(X_1^n\in\mathcal T_{n,\delta})\to1.
$$

并且对每个 $n$，

$$
\mathbb P(X_1^n\in\mathcal T_{n,\delta})
\,2^{n(h-\delta)}
\le |\mathcal T_{n,\delta}|
\le2^{n(h+\delta)}.
$$

**证明.** 典型集事件正是

$$
\left\{
\left|-\frac1n\log p_n(X_1^n)-h\right|\le\delta
\right\},
$$

故其概率由 EI-10 趨于 $1$。对上界，典型集内每个序列概率至少为 $2^{-n(h+\delta)}$，总概率不超过 $1$，所以序列数不超过 $2^{n(h+\delta)}$。对下界，典型集内每个序列概率至多为 $2^{-n(h-\delta)}$，故

$$
\mathbb P(X_1^n\in\mathcal T_{n,\delta})
\le|\mathcal T_{n,\delta}|2^{-n(h-\delta)}.
$$

整理即得。证毕。

AEP 说明高概率质量集中在指数规模约为 $2^{nh}$ 的集合上。把这个集合变成具体编码器还需要指定编码对象和错误准则；第 9 章已对 i.i.d. 信源单独陈述固定长度编码定理。

## 10.6 有限模型接口：温度缩放

给定有限实数 logits $z_1,\ldots,z_m$ 与温度 $T>0$，定义

$$
p_i(T)=\frac{\exp(z_i/T)}{\sum_{j=1}^m\exp(z_j/T)}.
$$

分母严格为正且有限，所以这是 $[m]$ 上严格正的概率分布。

**定理 10.6（温度缩放的熵单调性）。** 若 $z_i$ 不全相等，令

$$
H_T=-\sum_{i=1}^mp_i(T)\ln p_i(T)
$$

使用自然对数。则 $H_T$ 在 $T\in(0,\infty)$ 上严格递增。若所有 $z_i$ 相等，则 $p(T)$ 对所有 $T$ 均匀，熵恒为 $\ln m$。

**证明.** 令 $\beta=1/T$，$Z(\beta)=\sum_ie^{\beta z_i}$，并以 $\mathbb E_\beta$ 表示分布 $p_i=e^{\beta z_i}/Z(\beta)$ 下的期望。直接展开得

$$
H(\beta)=\ln Z(\beta)-\beta\mathbb E_\beta[z].
$$

有限和允许逐项求导，且

$$
\frac d{d\beta}\ln Z=\mathbb E_\beta[z],
\qquad
\frac d{d\beta}\mathbb E_\beta[z]
=\operatorname{Var}_\beta(z).
$$

因此

$$
\frac{dH}{d\beta}
=-\beta\operatorname{Var}_\beta(z).
$$

当 logits 不全相等时，严格正分布 $p(\beta)$ 下的方差严格为正；又 $\beta>0$ 且 $d\beta/dT=-1/T^2<0$，故 $dH_T/dT>0$。全相等情形直接代入定义。证毕。

这个结论只描述有限输出分布的 Shannon 熵。它不把熵等同于文本质量、语义多样性或创造力；这些概念需要额外的可观测指标和模型。

## 练习

**练习 10.1.** 对独立同分布有限字母表过程证明 $H(X_1^n)=nH(X_1)$，从而 $h(X)=H(X_1)$。

**练习 10.2.** 对二状态平稳 Markov 链写出熵率公式，并计算 $a=b=p$ 时的结果。

**练习 10.3.** 对定义 10.2 前的隐藏常量过程，计算 $H_n$ 与熵率，并解释为什么 Birkhoff 极限不等于总体均值。

**练习 10.4.** 补全例 10.1 的比例估计，严格证明两个子序列分别趋近 $1$ 与 $0$。

**练习 10.5.** 从推论 10.5 推出：对每个 $\eta\in(0,1)$ 和 $\delta>0$，充分大 $n$ 时存在集合 $A_n\subseteq\mathcal X^n$，满足 $\mathbb P(X_1^n\in A_n)\ge1-\eta$ 且 $|A_n|\le2^{n(h+\delta)}$。

**练习 10.6.** 说明 Birkhoff 定理应用于 $f(x)=\mathbf 1_{\{x_1=a\}}$ 时，每个对象所在的概率空间、变换和不变 $\sigma$-代数。

**练习 10.7.** 解释为什么定理 10.6 不能推出“温度越高，生成文本越有创造力”，并指出要把这句话变成可检验命题至少还需定义哪些对象。
