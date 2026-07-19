# 附录 A.3 统计学习：偏差、容量与一致收敛

本附录补足卷一第二章使用的两个经典结果：平方损失回归的偏差--方差分解，以及二分类假设类的 VC 一致收敛界。两者使用不同损失和概率空间；VC 结论不能直接替代语言模型交叉熵或生成分布的专门分析。

## A.3.1 平方损失的偏差--方差分解

固定测试输入 $x$。设测试标签

$$
Y^*=f(x)+\varepsilon^*,
\qquad
\mathbb E[\varepsilon^*\mid x]=0,
\quad
\operatorname{Var}(\varepsilon^*\mid x)=\sigma^2(x).
$$

训练集 $\mathcal D$ 独立于测试噪声，并由学习算法产生预测 $\widehat f_{\mathcal D}(x)$。记

$$
\overline f(x)=\mathbb E_{\mathcal D}[\widehat f_{\mathcal D}(x)].
$$

**命题 A.3.1** 若上述二阶矩有限，则

$$
\begin{aligned}
\mathbb E_{\mathcal D,\varepsilon^*}
[(Y^*-\widehat f_{\mathcal D}(x))^2]
={}&(f(x)-\overline f(x))^2\\
&+\mathbb E_{\mathcal D}
[(\widehat f_{\mathcal D}(x)-\overline f(x))^2]
+\sigma^2(x).
\end{aligned}
\tag{A.3.1}
$$

**证明** 写成

$$
Y^*-\widehat f_{\mathcal D}
=(f-\overline f)
+(\overline f-\widehat f_{\mathcal D})
+\varepsilon^*.
$$

平方展开后，$\mathbb E_{\mathcal D}[\overline f-\widehat f_{\mathcal D}]=0$；测试噪声与训练集独立且条件均值为零，所以另外两个交叉项期望也为零。三个平方项分别给出 (A.3.1)。$\square$

第一项是点态偏差平方，第二项是训练集随机性引起的点态方差，第三项是该输入处的测试噪声。若再对 $X$ 积分，需要明确测试输入分布。该恒等式不证明“模型越大，方差必然越大”；双下降等现象也不与恒等式矛盾。

<img src="../../vol-01/chapter_02/images/bias_variance_tradeoff.png" width="80%" />

## A.3.2 从固定假设到数据依赖选择

令 $Z_1,\ldots,Z_N$ 独立同分布，二分类损失 $\ell_h(Z)\in\{0,1\}$，并定义

$$
R(h)=\mathbb E[\ell_h(Z)],
\qquad
\widehat R_N(h)=\frac1N\sum_{i=1}^N\ell_h(Z_i).
$$

对训练前固定的 $h$，Hoeffding 不等式给出

$$
\Pr\bigl(|R(h)-\widehat R_N(h)|>\epsilon\bigr)
\le2e^{-2N\epsilon^2}.
\tag{A.3.2}
$$

学习算法却会用同一批数据选择 $h$，所以不能在训练后把随机输出 $\widehat h(Z_{1:N})$ 当作预先固定的 $h$ 套用 (A.3.2)。极端例子是连续输入上的随机标签：记忆全部训练点、在其他点恒输出 $0$ 的规则有零训练误差，但独立测试误差仍为 $1/2$。控制数据依赖选择需要假设类上的统一界或算法稳定性等其他工具。

## A.3.3 增长函数与 VC 维

令 $\mathcal H\subseteq\{-1,+1\}^{\mathcal X}$。对有限点集 $S=\{x_1,\ldots,x_N\}$，定义限制集

$$
\mathcal H|_S
=\{(h(x_1),\ldots,h(x_N)):h\in\mathcal H\},
$$

以及增长函数

$$
m_{\mathcal H}(N)
=\max_{|S|=N}\lvert\mathcal H|_S\rvert.
$$

若 $\lvert\mathcal H|_S\rvert=2^N$，称 $\mathcal H$ 打散 $S$。VC 维定义为

$$
\operatorname{VCdim}(\mathcal H)
=\sup\{|S|:S\text{ 可被 }\mathcal H\text{ 打散}\},
$$

并允许取 $\infty$。

### 仿射半空间的 VC 维

令 $\mathcal H_d$ 为 $\mathbb R^d$ 上的仿射半空间分类器，约定分数非负时输出 $+1$。

**命题 A.3.2** $\operatorname{VCdim}(\mathcal H_d)=d+1$。

**证明（下界）** 取 $d+1$ 个仿射独立点 $x_i$。其增广向量 $\widetilde x_i=(x_i,1)$ 构成 $\mathbb R^{d+1}$ 的一组基。对任意标签 $y_i\in\{-1,+1\}$，线性方程

$$
w^\mathsf Tx_i+b=y_i,
\qquad i=1,\ldots,d+1,
$$

有唯一解，故这些点被打散。

**证明（上界）** 任取 $d+2$ 个点。其增广向量线性相关，故存在不全为零的 $\lambda_i$ 使

$$
\sum_i\lambda_ix_i=0,
\qquad
\sum_i\lambda_i=0.
$$

令 $I=\{i:\lambda_i>0\}$、$J=\{i:\lambda_i<0\}$，并以正负系数和归一化，得到两个凸包的公共点 $z$。把 $I$ 标为 $+1$、$J$ 标为 $-1$，对 $\lambda_i=0$ 的点任意赋标。若某仿射分类器实现该标记，则从 $I$ 一侧的凸组合得到 $w^\mathsf Tz+b\ge0$，从 $J$ 一侧得到 $w^\mathsf Tz+b<0$，矛盾。故任意 $d+2$ 点都不能被打散。$\square$

这个 Radon 分割证明覆盖凸位置、内点、共线和重合等情形。只画正方形的一种 XOR 标记不能单独证明上界，因为上界含有“任意点集”的量词。

<img src="images/vc_dimension.png" width="90%" />

## A.3.4 Sauer--Shelah 引理

**引理 A.3.3** 若 $d=\operatorname{VCdim}(\mathcal H)<\infty$，则

$$
m_{\mathcal H}(N)
\le\sum_{j=0}^{\min(d,N)}\binom Nj.
\tag{A.3.3}
$$

特别地，当 $N\ge d\ge1$ 时，

$$
m_{\mathcal H}(N)\le\left(\frac{eN}{d}\right)^d.
\tag{A.3.4}
$$

**证明** 对 $N$ 归纳。固定 $N$ 个点，把最后一个坐标删除。令 $A$ 为至少能以一种末位标签出现的前缀集合，$B$ 为两种末位标签都能出现的前缀集合。每个前缀贡献一种末位标记，若属于 $B$ 再多贡献一种，所以

$$
\lvert\mathcal H|_S\rvert=|A|+|B|.
$$

$A$ 的 VC 维至多为 $d$。若 $B$ 能打散 $d$ 个前缀点，则加上末位点可打散 $d+1$ 个点，故 $B$ 的 VC 维至多为 $d-1$。由归纳假设与 Pascal 恒等式，

$$
|A|+|B|
\le\sum_{j=0}^d\binom{N-1}{j}
+\sum_{j=0}^{d-1}\binom{N-1}{j}
=\sum_{j=0}^d\binom Nj.
$$

基例 $N=0$ 或 $d=0$ 直接成立，得到 (A.3.3)；(A.3.4) 是标准二项式和估计。$\square$

## A.3.5 一致收敛界与 ERM 推论

令损失类 $\mathcal G=\{(x,y)\mapsto\mathbf 1\{h(x)\ne y\}:h\in\mathcal H\}$。下文假设它是点态可测的：存在可数子类 $\mathcal G_0$，使每个 $g\in\mathcal G$ 都是 $\mathcal G_0$ 中某序列的逐点极限。这样所有上确界可约化到可数类；有限或可数假设类是直接特例。若不作此类可测性假设，定理需改用外概率表述。

**定理 A.3.4（一个保守的 VC 界）** 对任意 $\epsilon>0$，

$$
\Pr\left(
\sup_{h\in\mathcal H}|R(h)-\widehat R_N(h)|>\epsilon
\right)
\le
4m_{\mathcal H}(2N)
\exp\left(-\frac{N\epsilon^2}{8}\right).
\tag{A.3.5}
$$

**证明** 取独立同分布的幽灵样本 $Z'_1,\ldots,Z'_N$，相应经验风险记作 $\widehat R'_N$。若 $N\epsilon^2<2$，(A.3.5) 的右侧大于 $1$，结论平凡。以下设 $N\epsilon^2\ge2$。

点态可测性与损失有界性使风险及经验风险都可由同一可数子类 $\mathcal G_0$ 逐点逼近，因此上确界事件可写成可数并。固定枚举 $\mathcal G_0=(g_j)_{j\ge1}$；在上确界严格超过 $\epsilon$ 的事件上，令 $j^*$ 为第一个满足相应偏差严格超过 $\epsilon$ 的索引。这样得到的见证是原样本的可测函数，并不要求上确界实际取到。条件于原样本后，$\widehat R'_N(g_{j^*})$ 的方差至多为 $1/(4N)$。Chebyshev 不等式给出

$$
\Pr\left(
|\widehat R'_N(g_{j^*})-R(g_{j^*})|\le\epsilon/2
\mid Z_{1:N}
\right)\ge\frac12.
$$

因此标准对称化步骤得到

$$
\Pr\left(\sup_h|R(h)-\widehat R_N(h)|>\epsilon\right)
\le2\Pr\left(
\sup_h|\widehat R_N(h)-\widehat R'_N(h)|>\epsilon/2
\right).
\tag{A.3.5a}
$$

条件于合并后的 $2N$ 个观测，并对每对 $(Z_i,Z'_i)$ 独立随机交换。令 $\xi_i$ 为相应 Rademacher 符号；对固定 $h$，经验风险差与

$$
\frac1N\sum_{i=1}^N
\xi_i\bigl(\ell_h(Z_i)-\ell_h(Z'_i)\bigr)
$$

同分布。每个括号属于 $\{-1,0,1\}$，故 Hoeffding 不等式给出

$$
\Pr_\xi\left(
\left|\frac1N\sum_i\xi_i
(\ell_h(Z_i)-\ell_h(Z'_i))\right|>\epsilon/2
\right)
\le2e^{-N\epsilon^2/8}.
$$

$\mathcal G$ 在这 $2N$ 个带标签观测上产生的不同损失向量不超过 $m_{\mathcal H}(2N)$ 个。对这些向量取联合界，再代入 (A.3.5a)，即得 (A.3.5)。$\square$

若 $d=\operatorname{VCdim}(\mathcal H)$ 且 $N\ge d\ge1$，则对任意 $\delta\in(0,1)$，以至少 $1-\delta$ 的概率，

$$
\sup_{h\in\mathcal H}|R(h)-\widehat R_N(h)|
\le
\min\left\{1,
\sqrt{\frac{8}{N}
\left(d\log\frac{2eN}{d}+\log\frac4\delta\right)}
\right\}.
\tag{A.3.6}
$$

令右侧为 $\varepsilon_N$，并令经验风险最小化器
$\widehat h\in\arg\min_{h\in\mathcal H}\widehat R_N(h)$ 存在。对任意 $h^*\in\arg\min_{h\in\mathcal H}R(h)$，在同一事件上

$$
R(\widehat h)
\le\widehat R_N(\widehat h)+\varepsilon_N
\le\widehat R_N(h^*)+\varepsilon_N
\le R(h^*)+2\varepsilon_N.
\tag{A.3.7}
$$

这才把统一偏差控制转化为学习器的风险界。它仍不保证优化器找到 ERM，也不控制假设类外的近似误差。在标准分布无关二分类 PAC 框架及适当可测性条件下，有限 VC 维刻画可学习性；该结论不是任意回归、结构化预测或生成任务的充要条件。

## A.3.6 来源

- Hoeffding, [*Probability Inequalities for Sums of Bounded Random Variables*](https://doi.org/10.1080/01621459.1963.10500830), 1963。
- Vapnik & Chervonenkis, [*On the Uniform Convergence of Relative Frequencies of Events to Their Probabilities*](https://doi.org/10.1137/1116025), 1971。
- Sauer, [*On the Density of Families of Sets*](https://doi.org/10.1016/0097-3165(72)90019-2), 1972。
