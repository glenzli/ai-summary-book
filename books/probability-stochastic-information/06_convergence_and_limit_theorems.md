# 收敛方式与极限定理

“误差趋于零”至少有四种常用含义。样本点逐条收敛、偏差概率消失、平均 $p$ 次误差消失以及分布测试值收敛所保留的信息不同。大数律关心样本均值，中心极限定理关心归一化波动；二者不能仅凭一句“趋于稳定”合并。本章先证明能够无条件成立的蕴含，再用具体反例切断错误的反向箭头。这样，后文每一次极限调用都能回答三个问题：随机变量是否位于同一概率空间，收敛发生在哪种拓扑或概率意义下，以及结论是否允许再取期望。

## 6.1 四种收敛与蕴含关系

**定义 6.1（几乎处处、依概率与 $L^p$ 收敛）。** 设 $X_n,X$ 是同一概率空间 $(\Omega,\mathcal F,\mathbb P)$ 上的实值随机变量，$1\le p<\infty$。

1. 写作 $X_n\to X$ a.s.，若
   $$
   \mathbb P\{\omega:X_n(\omega)\to X(\omega)\}=1.
   $$
2. 写作 $X_n\to X$ in $\mathbb P$，若对每个 $\varepsilon>0$，
   $$
   \mathbb P(|X_n-X|>\varepsilon)\to0.
   $$
3. 写作 $X_n\to X$ in $L^p$，若 $X_n,X\in L^p$ 且
   $$
   \mathbb E|X_n-X|^p\to0.
   $$

**定义 6.2（分布收敛）。** 随机变量 $X_n$ 与 $X$ 可以定义在不同概率空间上。若对每个有界连续函数 $f:\mathbb R\to\mathbb R$，

$$
\mathbb E[f(X_n)]\to\mathbb E[f(X)],
$$

则称 $X_n$ 依分布收敛到 $X$，记作 $X_n\Rightarrow X$。该定义只涉及分布，不涉及把 $X_n$ 与 $X$ 放在同一概率空间后的逐点距离。

**定理 6.1（基本收敛蕴含）。** 对同一概率空间上的实值随机变量：

$$
X_n\to X\text{ a.s.}\quad\Longrightarrow\quad
X_n\to X\text{ in }\mathbb P
\quad\Longrightarrow\quad X_n\Rightarrow X,
$$

并且对任意 $1\le p<\infty$，

$$
X_n\to X\text{ in }L^p
\quad\Longrightarrow\quad X_n\to X\text{ in }\mathbb P.
$$

此外，若 $X_n\to X$ a.s. 且存在 $Y\in L^p$ 使 $|X_n|\le Y$ 几乎处处对所有 $n$ 成立，则 $X_n\to X$ in $L^p$。

**证明.** 若 $X_n\to X$ a.s.，固定 $\varepsilon>0$，则

$$
\mathbf 1_{\{|X_n-X|>\varepsilon\}}\to0
$$

几乎处处，且该示性函数被可积常数 $1$ 控制。由控制收敛 EI-2c，

$$
\mathbb P(|X_n-X|>\varepsilon)
=\mathbb E\mathbf 1_{\{|X_n-X|>\varepsilon\}}\to0.
$$

若 $X_n\to X$ in $L^p$，对 $|X_n-X|^p$ 使用 Markov 不等式，得到

$$
\mathbb P(|X_n-X|>\varepsilon)
\le\frac{\mathbb E|X_n-X|^p}{\varepsilon^p}\to0.
$$

现在设 $X_n\to X$ in $\mathbb P$，并取 $f\in C_b(\mathbb R)$，令 $M=\sup_x|f(x)|$。给定 $\eta>0$，先取 $K>0$ 使 $\mathbb P(|X|>K)<\eta$。函数 $f$ 在紧区间 $[-K-1,K+1]$ 上一致连续，故存在 $0<\delta\le1$，使该区间内 $|x-y|\le\delta$ 时 $|f(x)-f(y)|<\eta$。将期望按事件

$$
A_n=\{|X|\le K,\ |X_n-X|\le\delta\}
$$

分解，有

$$
\begin{aligned}
|\mathbb E f(X_n)-\mathbb E f(X)|
&\le\mathbb E|f(X_n)-f(X)|\\
&\le\eta+2M\bigl[\mathbb P(|X|>K)
+\mathbb P(|X_n-X|>\delta)\bigr].
\end{aligned}
$$

令 $n\to\infty$ 后再令 $\eta\downarrow0$，得到分布收敛。

最后，若 $|X_n|\le Y\in L^p$ 且 $X_n\to X$ a.s.，则 $|X|\le Y$ a.s.，且

$$
|X_n-X|^p\le(2Y)^p\in L^1.
$$

对 $|X_n-X|^p$ 使用 EI-2c，得到 $L^p$ 收敛。证毕。

这些箭头通常不能反向。

**例 6.1（反向蕴含失败）。**

1. 在带 Lebesgue 概率的 $(0,1]$ 上，把每一行半开等长分割依次枚举：
   $$
   (0,1],\quad(0,1/2],(1/2,1],\quad
   (0,1/3],(1/3,2/3],(2/3,1],\ldots
   $$
   令 $X_n$ 为第 $n$ 个区间的示性函数。区间长度趋于 $0$，故 $X_n\to0$ in $\mathbb P$；但每个 $\omega$ 在每一行都落入一个区间，所以 $X_n(\omega)=1$ 无穷多次，不逐点收敛到 $0$。
2. 若 $X$ 等概率取 $\pm1$，令 $X_n=-X$。则每个 $X_n$ 与 $X$ 同分布，所以 $X_n\Rightarrow X$；但 $|X_n-X|=2$ 几乎处处，故不依概率收敛。
3. 在 $(0,1)$ 上令 $X_n=n\mathbf 1_{(0,1/n)}$。则 $X_n\to0$ a.s.，但 $\mathbb E|X_n|=1$，故不在 $L^1$ 中收敛。

## 6.2 Borel--Cantelli 引理

**定理 6.2（第一 Borel--Cantelli 引理）。** 若事件列 $(A_n)_{n\ge1}$ 满足

$$
\sum_{n=1}^{\infty}\mathbb P(A_n)<\infty,
$$

则

$$
\mathbb P(A_n\ \text{无限次发生})
=\mathbb P\left(\limsup_{n\to\infty}A_n\right)=0,
$$

其中

$$
\limsup_{n\to\infty}A_n
=\bigcap_{m=1}^{\infty}\bigcup_{n\ge m}A_n.
$$

此结论不要求事件独立。

**证明.** 对每个 $m$，可列次可加性给出

$$
\mathbb P\left(\bigcup_{n\ge m}A_n\right)
\le\sum_{n\ge m}\mathbb P(A_n).
$$

因为 $\limsup_nA_n$ 包含于每个尾并集，

$$
0\le\mathbb P(\limsup_nA_n)
\le\sum_{n\ge m}\mathbb P(A_n).
$$

令 $m\to\infty$，收敛级数的尾和趋于 $0$，故结论成立。证毕。

## 6.3 弱大数律

弱大数律的方差计算需要一个独立乘积接口。

**引理 6.3（独立 $L^2$ 随机变量协方差为零）。** 若实随机变量 $U,V\in L^2$ 独立，则

$$
\mathbb E[UV]=\mathbb E[U]\mathbb E[V],
\qquad
\operatorname{Cov}(U,V)=0.
$$

**证明.** 先设 $U,V\ge0$。分别取关于 $\sigma(U)$ 与 $\sigma(V)$ 可测的非负简单函数 $U_n\uparrow U$、$V_n\uparrow V$。把简单函数写成示性函数的有限线性组合；$\sigma(U)$ 与 $\sigma(V)$ 的独立性逐项给出

$$
\mathbb E[U_nV_n]=\mathbb E[U_n]\mathbb E[V_n].
$$

因为 $U_nV_n\uparrow UV$，对三处使用单调收敛，得到非负情形的乘积公式，允许两边为 $+\infty$。

一般 $U,V\in L^2$ 时，Cauchy--Schwarz 给出 $UV\in L^1$。把 $U,V$ 分解为正负部，对四个非负乘积使用上式；每项都被 $|UV|$ 控制而有限。线性组合后得到 $\mathbb E[UV]=\mathbb E U\mathbb E V$。对 $U-\mathbb EU$ 与 $V-\mathbb EV$ 应用该式，即得协方差为零。证毕。

**定理 6.4（方差型弱大数律）。** 设 $X_1,X_2,\ldots$ 两两独立且 $X_k\in L^2$。记 $m_k=\mathbb E X_k$、$\sigma_k^2=\operatorname{Var}(X_k)$。若

$$
\frac1{n^2}\sum_{k=1}^n\sigma_k^2\longrightarrow0,
$$

则

$$
\frac1n\sum_{k=1}^n(X_k-m_k)\to0
$$

同时在 $L^2$ 中和依概率成立。

特别地，若 $X_k$ 两两独立、同分布，$\mathbb E X_1=\mu$ 且 $\operatorname{Var}(X_1)=\sigma^2<\infty$，则

$$
\overline X_n=\frac1n\sum_{k=1}^nX_k\to\mu
$$

在 $L^2$ 中并依概率成立。

**证明.** 令 $S_n=\sum_{k=1}^n(X_k-m_k)$。引理 6.3 给出不同项之间协方差为零，故

$$
\mathbb E\left|\frac{S_n}{n}\right|^2
=\operatorname{Var}\left(\frac{S_n}{n}\right)
=\frac1{n^2}\sum_{k=1}^n\sigma_k^2\to0.
$$

这正是 $L^2$ 收敛；定理 6.1 再给出依概率收敛。同分布情形中右端等于 $\sigma^2/n$。证毕。

**例 6.2（Bernoulli 频率的有限样本界）。** 若 $X_k$ 独立且服从 Bernoulli$(p)$，则 $\overline X_n$ 是前 $n$ 次试验中的成功频率，且

$$
\mathbb E\overline X_n=p,
\qquad
\operatorname{Var}(\overline X_n)=\frac{p(1-p)}n.
$$

因此对每个 $\varepsilon>0$，

$$
\mathbb P(|\overline X_n-p|\ge\varepsilon)
\le\frac{p(1-p)}{n\varepsilon^2}.
$$

极限陈述来自右端趋零；不等式同时保留了一个虽不尖锐但完全显式的有限样本保证。

## 6.4 强大数律与中心极限定理：两个不同输入

**外部输入定理 6.5（独立同分布强大数律，EI-7）。** 设 $X_1,X_2,\ldots$ 是同一概率空间上的独立同分布实随机变量，且

$$
\mathbb E|X_1|<\infty.
$$

令 $\mu=\mathbb E X_1$。则

$$
\frac1n\sum_{k=1}^nX_k\longrightarrow\mu
\qquad\text{几乎处处}.
$$

本书用它把独立重复观测的样本频率升级为逐样本路径结论。证明需要截断、独立和式控制及 Borel--Cantelli 等完整论证，登记为 EI-7，不在正文重证。注意它只要求一阶绝对矩有限，而定理 6.4 的简短方差证明要求二阶矩有限。

**外部输入定理 6.6（Lindeberg--Levy 中心极限定理，EI-8）。** 设 $X_1,X_2,\ldots$ 为独立同分布实随机变量，

$$
\mathbb E X_1=\mu,
\qquad
0<\operatorname{Var}(X_1)=\sigma^2<\infty.
$$

则

$$
\frac{\sum_{k=1}^nX_k-n\mu}{\sigma\sqrt n}
\Rightarrow Z,
\qquad Z\sim N(0,1).
$$

这里结论是分布收敛，不是样本均值几乎处处收敛。若 $\sigma^2=0$，则 $X_1=\mu$ 几乎处处，标准化表达式无定义；该退化情形由大数律直接处理。EI-8 的证明依赖特征函数或 Lindeberg 方法，来源定位见 [SOURCES.md](SOURCES.md)。

## 练习

**练习 6.1.** 不直接引用定理 6.1，使用 Markov 不等式证明 $L^p$ 收敛推出依概率收敛。

**练习 6.2.** 若 $\sum_n\mathbb E|X_n|<\infty$，证明 $X_n\to0$ 几乎处处。

**练习 6.3.** 对独立 Bernoulli$(p)$ 样本，写出弱大数律和强大数律各自给出的频率陈述，并说明量词差异。

**练习 6.4.** 验证例 6.1 的三组反例分别否定哪一个反向蕴含。

**练习 6.5.** 设 $X_k$ 两两独立且 $\operatorname{Var}(X_k)\le Ck^\alpha$。求使定理 6.4 的方差条件成立的所有 $\alpha\in\mathbb R$。

**练习 6.6.** 若 $X_k$ 独立同分布，均值 $\mu$、方差 $\sigma^2\in(0,\infty)$，把中心极限定理改写为样本均值 $\overline X_n$ 的标准化形式。
