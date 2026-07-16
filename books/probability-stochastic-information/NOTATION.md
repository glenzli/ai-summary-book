# 符号表

本表记录全书统一符号。正文若改变对象类型，会在使用点重新声明。

## 测度、随机变量与积分

| 符号 | 类型与含义 |
|---|---|
| $(\Omega,\mathcal F,\mathbb P)$ | 概率空间 |
| $(E,\mathcal E)$ | 一般可测空间 |
| $\mathcal B(E)$ | 拓扑空间 $E$ 的 Borel $\sigma$-代数 |
| $X:\Omega\to E$ | $E$ 值随机变量 |
| $\mathcal L(X)=X_\#\mathbb P$ | $X$ 的分布或推前测度 |
| $\mathbf 1_A$ | 事件 $A$ 的示性函数 |
| $\mathbb E X$ | 可积或非负随机变量的期望；后者可为 $+\infty$ |
| $L^p$、$\|X\|_p$ | 模几乎处处相等的 $p$ 次可积随机变量空间及其范数，$1\le p<\infty$ |
| $\operatorname{Var}(X)$ | $X\in L^2$ 的方差 |
| $\operatorname{Cov}(X,Y)$ | $X,Y\in L^2$ 的协方差 |
| $\mathbb E[X\mid\mathcal G]$ | 关于子 $\sigma$-代数 $\mathcal G$ 的条件期望等价类 |

## 核、过程与收敛

| 符号 | 类型与含义 |
|---|---|
| $K(x,B)$ | 从 $(E,\mathcal E)$ 到 $(F,\mathcal H)$ 的 Markov 核 |
| $\mu K$ | 输入分布 $\mu$ 经核 $K$ 得到的输出分布 |
| $P^n$ | Markov 核或转移矩阵的 $n$ 步复合，$P^0$ 为恒等核 |
| $(X_t)_{t\in T}$ | 同一概率空间上的随机变量族，即随机过程 |
| $(\mathcal F_n)_{n\ge0}$ | 离散时间滤过 |
| $\tau$ | 取值于非负整数及可能的 $+\infty$ 的停时 |
| $X_n\to X$ a.s. | 几乎处处收敛 |
| $X_n\to X$ in $\mathbb P$ | 依概率收敛 |
| $X_n\to X$ in $L^p$ | $p$ 次平均收敛 |
| $X_n\Rightarrow X$ | 依分布收敛 |

## 有限字母表信息量

| 符号 | 类型与含义 |
|---|---|
| $\mathcal X,\mathcal Y$ | 非空有限字母表 |
| $p_X,p(x)$ | 有限随机变量的概率质量函数 |
| $H(X)$ | Shannon 熵 |
| $H(X\mid Y)$ | 有限随机变量的条件熵 |
| $D(P\|Q)$ | KL 散度，可取 $+\infty$ |
| $I(X;Y)$ | 互信息 |
| $I(X;Y\mid Z)$ | 条件互信息 |
| $h_2(p)$ | 二元熵函数 |
| $X_i^j$ | 块 $(X_i,\ldots,X_j)$；$X^n=X_1^n$ |
| $H_n=H(X_1^n)$ | 长度 $n$ 的块熵 |
| $h(X)$ 或 $h$ | 存在时的熵率 $\lim_nH_n/n$ |
| $p_n(x^n)$ | 过程块概率 $\mathbb P(X_1^n=x^n)$ |
| $T$ | 路径空间上的左移 |
| $\mathcal I$ | 左移的模零不变 $\sigma$-代数 |
| $\mathcal T_{n,\delta}$ | 指定阈值下的信息典型集 |

## 信源与信道编码

| 符号 | 类型与含义 |
|---|---|
| $[M]$ | 消息或索引集合 $\{1,\ldots,M\}$ |
| $P^n(x^n)$ | DMS 的 $n$ 重乘积分布 |
| $\imath_P(x^n)$ | DMS 块自信息 $-\log P^n(x^n)$ |
| $f_n:\mathcal X^n\to[M]$ | 固定长度信源编码器 |
| $g_n:[M]\to\mathcal X^n$ | 固定长度信源解码器 |
| $P_{e,n}^{\mathrm{src}}$ | 信源块平均错误概率 |
| $R_n^{\mathrm{src}}$ | 信源码率 $n^{-1}\log M$ |
| $c:\mathcal X\to\{0,1\}^*$ | 二元前缀码 |
| $\ell(x)$ | 码字 $c(x)$ 的长度 |
| $W(y\mid x)$ | 有限 DMC 的单次转移概率 |
| $W^n(y^n\mid x^n)$ | DMC 的 $n$ 次无记忆乘积信道 |
| $e_n:[M]\to\mathcal X^n$ | 信道编码器 |
| $d_n:\mathcal Y^n\to[M]$ | 信道解码器 |
| $\lambda_j$ | 消息 $j$ 的条件错误概率 |
| $P_{e,n}^{\mathrm{av}}$ | 均匀消息下的平均错误概率 |
| $P_{e,n}^{\max}$ | 对消息取最大值的错误概率 |
| $R_n^{\mathrm{ch}}$ | 信道码率 $n^{-1}\log M$ |
| $C(W)$ | 单字母容量 $\max_{p\in\mathcal P(\mathcal X)}I_p(X;Y)$ |
| $C_{\mathrm{av}}^{\mathrm{op}},C_{\max}^{\mathrm{op}}$ | 平均/最大错误准则下的操作容量 |

## 对数与边界约定

除非公式显式写 $\ln$，第 8--10 章的 $\log$ 以 $2$ 为底，熵与码率单位为 bit。约定

$$
0\log0=0,\qquad
0\log\frac0q=0,\qquad
p\log\frac p0=+\infty\quad(p>0).
$$

这些是加权项的连续延拓约定，不把 $0/0$ 本身定义成实数。条件概率 $p(x\mid y)$ 只在 $p_Y(y)>0$ 时由比值定义；零概率 $y$ 上可任选版本，所有加权公式不受影响。

所有随机变量默认可测。路径空间写作 $(E^T,\mathcal E^{\otimes T})$；连续或右连续样本路径从不由该记号自动保证。
