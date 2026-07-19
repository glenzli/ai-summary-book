# 附录 A.1 优化基础：梯度下降与随机逼近

本附录为卷一第二章的训练算法提供最小数学底座。重点是区分三件事：负梯度为何是局部下降方向，有限步长何时真的下降，以及随机梯度的收敛结论究竟依赖哪些假设。这里不把凸优化定理外推为深度网络的全局保证。

## A.1.1 记号与基本定义

参数写作 $\theta\in\mathbb R^d$，目标函数为 $F:\mathbb R^d\to\mathbb R$。本附录使用欧氏内积 $\langle u,v\rangle=u^\mathsf Tv$ 与范数 $\|u\|=\sqrt{u^\mathsf Tu}$，标量函数对向量的梯度统一写成列向量。

若 $F$ 可微，则在方向 $v$ 上的方向导数为

$$
D_vF(\theta)
=\lim_{t\to0}\frac{F(\theta+tv)-F(\theta)}t
=\langle\nabla F(\theta),v\rangle.
$$

由 Cauchy--Schwarz 不等式，对所有 $\|v\|=1$，

$$
D_vF(\theta)\ge-\|\nabla F(\theta)\|,
$$

且梯度非零时等号在 $v=-\nabla F(\theta)/\|\nabla F(\theta)\|$ 取得。因此负梯度只是**欧氏几何下的一阶最陡下降方向**；改变范数、坐标或预条件器会改变“最陡”的含义。

后文使用两个标准条件。

**定义 A.1.1（光滑性）** 若存在 $L>0$ 使

$$
\|\nabla F(x)-\nabla F(y)\|\le L\|x-y\|,
$$

则称 $F$ 的梯度为 $L$-Lipschitz，或简称 $F$ 为 $L$-smooth。

**定义 A.1.2（强凸性）** 若存在 $\mu>0$，对所有 $x,y$ 都有

$$
F(y)\ge F(x)+\langle\nabla F(x),y-x\rangle
+\frac\mu2\|y-x\|^2,
$$

则称 $F$ 为 $\mu$-强凸。可微强凸函数至多有一个全局极小点；若极小点 $\theta^*$ 存在于无约束域内部，则 $\nabla F(\theta^*)=0$。

## A.1.2 下降引理与确定性梯度下降

**引理 A.1.3（下降引理）** 若 $F$ 为 $L$-smooth，则

$$
F(y)\le F(x)+\langle\nabla F(x),y-x\rangle
+\frac L2\|y-x\|^2.
\tag{A.1.1}
$$

**证明** 令 $d=y-x$，由微积分基本定理，

$$
F(y)-F(x)=\int_0^1\langle\nabla F(x+td),d\rangle\,dt.
$$

减去 $\langle\nabla F(x),d\rangle$，再用 Cauchy--Schwarz 与 Lipschitz 条件：

$$
\begin{aligned}
F(y)-F(x)-\langle\nabla F(x),d\rangle
&=\int_0^1
\langle\nabla F(x+td)-\nabla F(x),d\rangle\,dt\\
&\le\int_0^1 Lt\|d\|^2\,dt
=\frac L2\|d\|^2.
\end{aligned}
$$

这就是 (A.1.1)。$\square$

取 $y=x-\eta\nabla F(x)$，得到

$$
F(x-\eta\nabla F(x))
\le F(x)-\eta\left(1-\frac{L\eta}{2}\right)
\|\nabla F(x)\|^2.
\tag{A.1.2}
$$

所以当 $0<\eta<2/L$ 且梯度非零时，这一步严格下降。该结论不覆盖非光滑目标，也不表示每个随机小批量更新都会降低完整训练损失。

<img src="images/gradient_descent_1d.png" width="60%" />

## A.1.3 经验风险、SGD 与条件无偏性

对有限训练集，经验风险常写成

$$
F(\theta)=\frac1N\sum_{i=1}^N\ell_i(\theta).
$$

完整批量梯度是 $\nabla F(\theta)$。若在给定当前参数时均匀抽取索引 $I_t$，则

$$
g_t=\nabla\ell_{I_t}(\theta_t),
\qquad
\mathbb E[g_t\mid\theta_t]=\nabla F(\theta_t).
$$

更严谨地，令 $\mathcal F_t$ 包含第 $t$ 步采样前的全部历史，则 SGD 假设写为

$$
\theta_{t+1}=\theta_t-\eta_tg_t,
\qquad
\mathbb E[g_t\mid\mathcal F_t]=\nabla F(\theta_t).
\tag{A.1.3}
$$

无偏性不保证 $g_t$ 与 $\nabla F(\theta_t)$ 每次同向。非均匀采样、样本加权、截断、梯度裁剪和有状态数据管线也可能改变条件期望；此时必须把相应权重或偏差写进分析。

大小为 $B$ 的独立小批量用梯度均值更新。在同分布且条件独立时，其噪声协方差相对单样本按 $1/B$ 缩放；真实训练中的重复样本、序列 packing 与分布式相关性会破坏这一理想比例。

## A.1.4 一个条件明确的 SGD 收敛定理

下述定理是随机逼近的一个窄而完整版本，不是对任意深度网络训练的描述。

**定理 A.1.4（强凸情形的几乎必然收敛）** 设：

1. $F$ 可微、$\mu$-强凸，并存在极小点 $\theta^*$；
2. $\theta_0$ 为平方可积的 $\mathcal F_0$-可测随机变量，$\theta_t$ 按 (A.1.3) 更新，且 $g_t$ 对 $\mathcal F_t$ 条件无偏；
3. 沿实际迭代存在常数 $G<\infty$，使 $\mathbb E[\|g_t\|^2\mid\mathcal F_t]\le G^2$；
4. $\eta_t$ 是确定性步长，$0<\eta_t\le1/(2\mu)$，并满足 Robbins--Monro 条件

   $$
   \sum_{t=0}^\infty\eta_t=\infty,
   \qquad
   \sum_{t=0}^\infty\eta_t^2<\infty.
   $$

则 $\theta_t\to\theta^*$ 几乎必然。

**证明** 令 $V_t=\|\theta_t-\theta^*\|^2$。条件于 $\mathcal F_t$ 展开：

$$
\begin{aligned}
\mathbb E[V_{t+1}\mid\mathcal F_t]
&=V_t-2\eta_t
\langle\nabla F(\theta_t),\theta_t-\theta^*\rangle
+\eta_t^2\mathbb E[\|g_t\|^2\mid\mathcal F_t].
\end{aligned}
$$

可微 $\mu$-强凸函数的梯度是 $\mu$-强单调的；结合 $\nabla F(\theta^*)=0$，

$$
\langle\nabla F(\theta_t),\theta_t-\theta^*\rangle
\ge\mu V_t.
$$

故

$$
\mathbb E[V_{t+1}\mid\mathcal F_t]
\le V_t-2\mu\eta_tV_t+G^2\eta_t^2.
\tag{A.1.4}
$$

Robbins--Siegmund 超鞅收敛引理应用于 (A.1.4)，给出 $V_t$ 几乎必然收敛且
$\sum_t\eta_tV_t<\infty$。若其极限在某个正概率事件上为 $c>0$，则该事件上最终有 $V_t\ge c/2$，与 $\sum_t\eta_t=\infty$ 矛盾。因此 $V_t\to0$，即 $\theta_t\to\theta^*$。$\square$

二阶矩界只要求沿迭代成立；若想由模型本身推出它，常需紧参数域、投影、梯度界或其他稳定性条件。强凸函数在整个 $\mathbb R^d$ 上通常不可能同时具有全局有界梯度，因此不能把这些假设不加检查地拼在一起。

满足两项步长条件的幂律例子是 $\eta_t=(t+1)^{-\alpha}$，其中 $1/2<\alpha\le1$。$1/\sqrt{t+1}$ 的平方和发散，不满足本定理。

## A.1.5 结论边界与反例

**固定学习率不会自动收敛到单点。** 取一维目标 $F(\theta)=\theta^2/2$，随机梯度 $g_t=\theta_t+\xi_t$，其中 $\xi_t$ 独立同分布、均值为零、方差为 $\sigma^2>0$。对 $0<\eta<2$，

$$
\theta_{t+1}=(1-\eta)\theta_t-\eta\xi_t.
$$

若 $\theta_0$ 确定且 $\xi_t$ 独立于过去，则二阶矩满足

$$
\mathbb E[\theta_{t+1}^2]
=(1-\eta)^2\mathbb E[\theta_t^2]+\eta^2\sigma^2,
$$

因而

$$
\lim_{t\to\infty}\mathbb E[\theta_t^2]
=\frac{\eta\sigma^2}{2-\eta}>0.
$$

而且 $\operatorname{Var}(\xi_1)>0$ 意味着存在 $a>0$ 使 $p:=\Pr(|\xi_1|>a)>0$。由同分布性，$\sum_t\Pr(|\xi_t|>a)=\sum_t p=\infty$；再由独立性与第二 Borel--Cantelli 引理，$|\xi_t|>a$ 几乎必然发生无穷多次，所以 $\xi_t$ 不会几乎必然趋于零。若 $\theta_t\to0$，递推式反而会推出 $\xi_t=((1-\eta)\theta_t-\theta_{t+1})/\eta\to0$，矛盾。这说明常数学习率即使在强凸二次目标上也通常只到达噪声邻域，而非几乎必然收敛到最优点。这里的同分布性或等价的统一尾概率条件是必要假设；仅有独立、零均值和统一方差，不能推出上述几乎必然反例。实际深度学习使用 warmup、分段衰减、余弦调度或常数尾段时，应按其实际目标讨论有限时间优化，不应直接引用定理 A.1.4。

Adam、RMSProp 与 AdamW 的更新和边界见[附录 A.8](a.8_advanced_optimization.md)。

## A.1.6 来源

- Robbins & Monro, [*A Stochastic Approximation Method*](https://doi.org/10.1214/aoms/1177729586), 1951。
- Robbins & Siegmund, [*A Convergence Theorem for Non Negative Almost Supermartingales and Some Applications*](https://doi.org/10.1016/B978-0-12-604550-5.50015-8), 1971。
- Bottou, Curtis & Nocedal, [*Optimization Methods for Large-Scale Machine Learning*](https://doi.org/10.1137/16M1080173), 2018。
