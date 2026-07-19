# 附录 概率基础的最小工具箱

本附录只提供卷三正文实际使用的有限或离散概率工具。更一般的测度论、随机过程与统计推断应参考专门教材；它们不需要在模型概率卷中重新展开。

## A.1 样本空间与事件

有限样本空间记为 $\Omega$。概率函数 $P$ 满足：

1. $P(\omega)\ge0$；
2. $\sum_{\omega\in\Omega}P(\omega)=1$。

事件 $A\subseteq\Omega$ 的概率为

$$
P(A)=\sum_{\omega\in A}P(\omega).
$$

因此 $P(\varnothing)=0$，$P(A^c)=1-P(A)$；若 $A\cap B=\varnothing$，则 $P(A\cup B)=P(A)+P(B)$。

语言模型的下一 token 样本空间通常是有限词表 $V$。完整可变长文本的空间是可数集合，有限情形的求和直觉仍适用，但停止规则必须包含在事件定义中。

## A.2 条件概率与乘法公式

若 $P(B)>0$，定义

$$
P(A\mid B)=\frac{P(A\cap B)}{P(B)}.
$$

于是

$$
P(A\cap B)=P(A\mid B)P(B).
$$

对随机变量 $X_1,\ldots,X_n$，反复应用乘法公式得到链式分解

$$
P(x_{1:n})
=P(x_1)\prod_{t=2}^n
P(x_t\mid x_{<t}).
$$

这就是自回归序列概率的基础。它是任意联合分布的恒等分解，并不单独规定条件概率由哪种神经网络实现。

## A.3 全概率与 Bayes 公式

若 $(B_j)_j$ 是互斥完备划分，且 $P(B_j)>0$，则

$$
P(A)=\sum_jP(A\mid B_j)P(B_j).
$$

这解释了隐藏条件混合。Bayes 公式为

$$
P(B_j\mid A)
=\frac{P(A\mid B_j)P(B_j)}
{\sum_kP(A\mid B_k)P(B_k)}.
$$

神经语言模型通常直接近似预测分布 $P(Y\mid X)$；它的 softmax 不能仅因形式上归一化，就被视为对所有未知参数的 Bayesian posterior。

## A.4 独立与条件独立

$A,B$ 独立指

$$
P(A\cap B)=P(A)P(B).
$$

给定 $C$ 条件独立则指

$$
P(A\cap B\mid C)
=P(A\mid C)P(B\mid C)
$$

在相应条件概率有定义处成立。无条件独立不蕴含给定任意 $C$ 后独立，反之亦然。

重复 API 请求只有在模型版本、会话状态和服务环境固定且随机源独立时，才可近似视为 iid。共享缓存、滚动升级或复用 seed 都可能破坏这一假设。

## A.5 随机变量、期望与方差

随机变量 $X:\Omega\to\mathbb R$ 的期望为

$$
\mathbb E[X]=\sum_{\omega}X(\omega)P(\omega),
$$

方差为

$$
\operatorname{Var}(X)
=\mathbb E[(X-\mathbb E X)^2]
=\mathbb E[X^2]-(\mathbb EX)^2.
$$

期望线性：

$$
\mathbb E[aX+bY]
=a\mathbb EX+b\mathbb EY,
$$

不要求 $X,Y$ 独立。方差相加通常需要零协方差；独立是充分条件。

## A.6 全期望与全方差

对离散条件变量 $Z$，

$$
\mathbb E[X]
=\mathbb E_Z[\mathbb E(X\mid Z)].
$$

全方差公式为

$$
\operatorname{Var}(X)
=\mathbb E_Z[\operatorname{Var}(X\mid Z)]
+\operatorname{Var}_Z(\mathbb E[X\mid Z]).
$$

第一项是组内变异，第二项是组间均值变异。卷三用它区分固定 checkpoint 内的采样变异与 checkpoint 之间的训练变异。

## A.7 Monte Carlo 与标准误差

若 $X_1,\ldots,X_n$ iid，均值为 $\mu$、方差为 $\sigma^2<\infty$，样本均值

$$
\bar X_n=\frac1n\sum_iX_i
$$

满足

$$
\mathbb E[\bar X_n]=\mu,
\qquad
\operatorname{Var}(\bar X_n)=\frac{\sigma^2}{n}.
$$

所以估计误差的标准差按 $1/\sqrt n$ 缩小，而不是 $1/n$。把 Monte Carlo 标准误差减半，通常需要约四倍独立样本。

Chebyshev 不等式给出

$$
P(|\bar X_n-\mu|\ge\varepsilon)
\le\frac{\sigma^2}{n\varepsilon^2},
$$

右侧随 $n\to\infty$ 趋于 0，这给出有限方差 iid 情形的弱大数律。

## A.8 Bernoulli 事件

对事件指示量 $X_i=\mathbf1\{Y_i\in A\}$，$X_i\sim\operatorname{Bernoulli}(p)$，于是

$$
\mathbb E[X_i]=p,
\qquad
\operatorname{Var}(X_i)=p(1-p).
$$

频率 $\hat p=n^{-1}\sum_iX_i$ 的标准误差为

$$
\sqrt{p(1-p)/n},
$$

实践中以 $\hat p$ 代替未知 $p$。小样本或极端频率应使用 Wilson、Clopper–Pearson 等二项区间，而不是只给点估计。

## A.9 熵、交叉熵与 KL

对有限分布 $p,q$，定义

$$
H(p)=-\sum_xp(x)\log p(x),
$$

$$
H(p,q)=-\sum_xp(x)\log q(x),
$$

$$
D_{KL}(p\Vert q)
=\sum_xp(x)\log\frac{p(x)}{q(x)}.
$$

约定 $0\log0=0$；若某个 $p(x)>0$ 而 $q(x)=0$，则交叉熵和 KL 为 $+\infty$。直接展开可得

$$
H(p,q)=H(p)+D_{KL}(p\Vert q).
$$

若存在 $p(x)>0$ 而 $q(x)=0$，KL 已按约定为 $+\infty$，非负性立即成立。以下只需考虑 $p(x)>0$ 时都有 $q(x)>0$ 的情形。对 $p$ 的支持应用 $\log u\le u-1$，

$$
-D_{KL}(p\Vert q)
=\sum_xp(x)\log\frac{q(x)}{p(x)}
\le\sum_xp(x)
\left(\frac{q(x)}{p(x)}-1\right)=0,
$$

故 $D_{KL}(p\Vert q)\ge0$；等号当且仅当 $p=q$。这证明交叉熵在可表示真实分布时由 $q=p$ 最小化。

KL 不对称，也不是距离：一般

$$
D_{KL}(p\Vert q)\ne D_{KL}(q\Vert p).
$$

## A.10 相关、预测与因果

条件概率 $P(Y\mid X)$ 是观察分布。干预 $P(Y\mid do(X=x))$ 需要额外因果模型或实验设计。仅凭联合分布通常不能唯一识别因果方向。

这个边界对模型分析同样重要：prompt 与输出的相关变化可以证明模型对输入敏感；只有明确修改内部变量并控制替代路径，才开始支持计算图内的干预结论；它仍不自动成为现实世界因果结论。

## A.11 使用范围

本附录足以支持本卷的条件分布、序列概率、交叉熵、熵、KL、采样频率和方差分解。遇到连续高维分布、严格条件期望、随机过程、Bayesian 推断或因果识别时，应转向专门教材，并明确新增假设。
