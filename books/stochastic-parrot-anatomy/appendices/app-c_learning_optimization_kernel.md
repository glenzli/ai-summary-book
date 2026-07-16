# 附录 C 学习与优化证明内核

本附录给出本书讨论模型工件、训练目标、后训练和序列计算时所需的最小数学核。适用范围是有限维实向量空间中的监督学习、可微计算图、常见随机优化更新、缩放点积自注意力和成对偏好优化。它不声称覆盖统计学习理论、凸优化、深度网络泛化理论或强化学习的完整课程。

先修包括：向量内积与范数、矩阵乘法、偏导数、链式法则、基本概率记号和有限集合上的对数似然。除非明确说明，本附录中的目标函数只是在声明样本、分布或偏好数据下定义的数学对象；优化目标的极小化不自动推出事实正确、校准、泛化、安全或责任归属。

## C.1 线性分类与可分间隔

设样本为

$$
(x_i,y_i)_{i=1}^n,\qquad x_i\in\mathbb R^d,\quad y_i\in\{-1,+1\}.
$$

线性分类器由 $w\in\mathbb R^d$ 和 $b\in\mathbb R$ 给出，其打分函数为

$$
f_{w,b}(x)=\langle w,x\rangle+b.
$$

预测规则通常取 $\operatorname{sign} f_{w,b}(x)$。为了把偏置写入同一向量，令

$$
\tilde x_i=(x_i,1)\in\mathbb R^{d+1},\qquad
\tilde w=(w,b)\in\mathbb R^{d+1}.
$$

于是 $f_{w,b}(x_i)=\langle \tilde w,\tilde x_i\rangle$。

**定义 C.1（线性可分与规范化间隔）.** 训练集称为以间隔 $\gamma>0$ 被单位向量 $u\in\mathbb R^{d+1}$ 分开，若

$$
\|u\|=1,\qquad y_i\langle u,\tilde x_i\rangle\ge \gamma
\quad (1\le i\le n).
$$

若还存在 $R>0$ 使 $\|\tilde x_i\|\le R$，则称该样本集具有半径界 $R$。有限样本一旦存在严格分离超平面，把分离向量归一化后，最小正间隔就是某个 $\gamma>0$；若只知道训练误差为零但不控制 $\gamma$ 和 $R$，则不能得到统一迭代步数界。

## C.2 感知机算法与有限步收敛

感知机算法在增广空间中运行。令 $a_i=y_i\tilde x_i$。初始化 $\tilde w_0=0$。按任意固定顺序反复扫描样本；当遇到

$$
\langle \tilde w,a_i\rangle\le 0
$$

时，执行一次更新

$$
\tilde w\leftarrow \tilde w+a_i.
$$

若一次完整扫描中没有更新，则当前 $\tilde w$ 正确分类全部训练样本。

**定理 C.2（感知机收敛定理）.** 若存在单位向量 $u$ 与常数 $\gamma,R>0$ 满足

$$
\langle u,a_i\rangle=y_i\langle u,\tilde x_i\rangle\ge\gamma,
\qquad
\|a_i\|=\|\tilde x_i\|\le R
$$

对所有 $i$ 成立，则感知机算法的更新次数至多为

$$
\left(\frac R\gamma\right)^2.
$$

因此算法在有限次更新后停止，并返回一个在训练集上无误分类的线性分类器。

**证明.** 只在发生错误或边界命中时计数。设第 $m$ 次更新使用的向量为 $a_{j_m}$，更新后权重为

$$
\tilde w_m=\tilde w_{m-1}+a_{j_m},
\qquad \tilde w_0=0.
$$

第一步给出沿真实分离方向的线性增长：

$$
\langle u,\tilde w_m\rangle
=\sum_{r=1}^m\langle u,a_{j_r}\rangle
\ge m\gamma.
$$

由于 $\|u\|=1$，Cauchy-Schwarz 不等式给出

$$
\|\tilde w_m\|\ge \langle u,\tilde w_m\rangle\ge m\gamma.
$$

第二步给出范数平方的至多线性增长。第 $m$ 次更新发生时有

$$
\langle \tilde w_{m-1},a_{j_m}\rangle\le0,
$$

故

$$
\begin{aligned}
\|\tilde w_m\|^2
&=\|\tilde w_{m-1}+a_{j_m}\|^2\\
&=\|\tilde w_{m-1}\|^2
  +2\langle \tilde w_{m-1},a_{j_m}\rangle
  +\|a_{j_m}\|^2\\
&\le \|\tilde w_{m-1}\|^2+R^2.
\end{aligned}
$$

归纳得

$$
\|\tilde w_m\|^2\le mR^2.
$$

把两个估计放在一起：

$$
m^2\gamma^2\le \|\tilde w_m\|^2\le mR^2.
$$

若 $m>0$，两边除以 $m\gamma^2$ 得

$$
m\le \frac{R^2}{\gamma^2}.
$$

所以更新次数不可能超过 $(R/\gamma)^2$。若算法永不停止，就会产生任意大的更新次数，与上界矛盾。因此有限次更新后存在一次完整扫描没有错误，算法停止。证毕。

**反例 C.3（不可分时不收敛）.** 在一维中取两个样本 $x_1=1,y_1=+1$ 与 $x_2=1,y_2=-1$，并暂不使用偏置。若 $w=0$，第一个样本使 $w$ 增加到 $1$；第二个样本又使 $w$ 回到 $0$。重复扫描会重复同一路径。问题不在步长调得不够好，而在同一点被赋予相反标签，线性分类器不可能同时满足两个约束。

**边界 C.4.** 感知机定理是训练集上的有限步错误修正定理。它不说明测试误差，不说明概率校准，不说明最大间隔最优性，也不说明深度网络训练。样本顺序会影响最终得到的分离超平面；定理只保证更新次数有上界。

## C.3 计算图、链式法则与反向传播

一个前馈计算图可看作有向无环图。每个节点 $v_k$ 存储一个向量，且

$$
v_k=\phi_k(v_{p_1(k)},\ldots,v_{p_r(k)};\theta_k),
$$

其中 $p_j(k)<k$。最终损失为标量

$$
L=\ell(v_m).
$$

**定义 C.5（伴随量）.** 对每个节点 $v_k$，若 $L$ 对 $v_k$ 可微，记

$$
\bar v_k=\frac{\partial L}{\partial v_k}
$$

为反向传播中的伴随量。这里采用列向量约定时，微分写作

$$
dL=\bar v_k^\top dv_k+\cdots .
$$

**定理 C.6（反向累积公式）.** 对于可微计算图，若已知后继节点 $v_j$ 的伴随量，则父节点 $v_i$ 从该后继获得的贡献为

$$
\left(\frac{\partial v_j}{\partial v_i}\right)^\top \bar v_j.
$$

父节点的总伴随量为所有直接后继贡献之和：

$$
\bar v_i
=
\sum_{j:\, i\in p(j)}
\left(\frac{\partial v_j}{\partial v_i}\right)^\top \bar v_j.
$$

参数梯度同理为

$$
\frac{\partial L}{\partial \theta_j}
=
\left(\frac{\partial v_j}{\partial \theta_j}\right)^\top \bar v_j,
$$

若同一参数在多个节点共享，则对所有出现位置求和。

**证明.** 对节点 $v_j=\phi_j(\ldots,v_i,\ldots;\theta_j)$，其一阶微分中关于 $v_i$ 的部分为

$$
dv_j=
\frac{\partial v_j}{\partial v_i}\,dv_i+\cdots .
$$

若 $L$ 对 $v_j$ 的微分贡献是

$$
dL=\bar v_j^\top dv_j+\cdots ,
$$

代入上式得

$$
\bar v_j^\top dv_j
=
\bar v_j^\top
\frac{\partial v_j}{\partial v_i}dv_i+\cdots
=
\left(
\left(\frac{\partial v_j}{\partial v_i}\right)^\top\bar v_j
\right)^\top dv_i+\cdots .
$$

因此该后继节点对 $\bar v_i$ 的贡献正是

$$
\left(\frac{\partial v_j}{\partial v_i}\right)^\top\bar v_j.
$$

若 $v_i$ 有多个后继，微分 $dL$ 中关于 $dv_i$ 的线性项相加，得到总和公式。参数 $\theta_j$ 的公式完全相同，只需把 $dv_i$ 换成 $d\theta_j$。证毕。

**边界 C.7.** 反向传播是链式法则在计算图上的高效组织。它在给定实现的可微路径上给出精确梯度；遇到 ReLU 的零点、截断、采样、排序、检索或外部工具调用时，必须说明使用的是次梯度、直通估计、连续松弛、策略梯度、停止梯度，还是不对该部分求导。反向传播本身不证明优化会找到全局最小值。

## C.4 Softmax 与交叉熵梯度

给定 logits $z\in\mathbb R^K$，softmax 分布为

$$
p_k(z)=\frac{\exp z_k}{\sum_{j=1}^K\exp z_j}.
$$

设目标标签为概率向量 $y\in\Delta^{K-1}$，即 $y_k\ge0$ 且 $\sum_k y_k=1$。交叉熵损失为

$$
\ell(z,y)=-\sum_{k=1}^K y_k\log p_k(z).
$$

**定理 C.8（softmax-交叉熵梯度）.** 对每个 $j$，

$$
\frac{\partial \ell}{\partial z_j}=p_j(z)-y_j.
$$

**证明.** 记

$$
Z=\sum_{r=1}^K\exp z_r.
$$

则

$$
\log p_k(z)=z_k-\log Z.
$$

代入损失：

$$
\begin{aligned}
\ell(z,y)
&=-\sum_k y_k z_k+\sum_k y_k\log Z\\
&=-\sum_k y_k z_k+\log Z,
\end{aligned}
$$

其中最后一步使用 $\sum_k y_k=1$。对 $z_j$ 求导：

$$
\frac{\partial \ell}{\partial z_j}
=-y_j+\frac{1}{Z}\frac{\partial Z}{\partial z_j}
=-y_j+\frac{\exp z_j}{Z}
=p_j(z)-y_j.
$$

证毕。

**边界 C.9.** 公式 $p-y$ 是损失对 logits 的梯度，不是对概率向量 $p$ 的梯度。若 $y$ 不是总质量为 $1$ 的概率向量，而是总质量 $\alpha$ 的非负权重，则梯度变为 $\alpha p-y$。Softmax 对平移不变：

$$
p(z+c\mathbf 1)=p(z),
$$

因此实现中常减去 $\max_j z_j$ 来提高数值稳定性；这不改变数学分布。

## C.5 正则化与 MAP 的条件边界

监督学习中常见目标为

$$
J(\theta)=\sum_{i=1}^n \ell_\theta(x_i,y_i)+\lambda\Omega(\theta).
$$

当 $\ell_\theta(x_i,y_i)=-\log p_\theta(y_i\mid x_i)$ 且样本在给定 $\theta$ 后条件独立时，前一项是负对数似然。

**定义 C.10（MAP 估计）.** 给定先验密度 $\pi(\theta)$ 与似然 $p_\theta(D)$，后验密度正比于

$$
\pi(\theta)p_\theta(D).
$$

最大后验估计为

$$
\hat\theta_{\mathrm{MAP}}\in
\operatorname*{arg\,max}_\theta \pi(\theta)p_\theta(D),
$$

等价地最小化

$$
-\log p_\theta(D)-\log\pi(\theta),
$$

只要这些量在同一参数化和同一基准测度下定义。

**命题 C.11（正则化目标成为 MAP 的条件）.** 若

$$
p_\theta(D)=\prod_{i=1}^n p_\theta(y_i\mid x_i),
$$

且存在可归一化先验密度

$$
\pi(\theta)=\frac{1}{Z_\pi}\exp(-\lambda\Omega(\theta)),
$$

则最小化

$$
\sum_i-\log p_\theta(y_i\mid x_i)+\lambda\Omega(\theta)
$$

与最大化后验密度具有相同的最优解集合。

**证明.** 在上述条件下，

$$
\begin{aligned}
-\log\bigl(\pi(\theta)p_\theta(D)\bigr)
&=
-\log\pi(\theta)-\sum_i\log p_\theta(y_i\mid x_i)\\
&=
\lambda\Omega(\theta)+\log Z_\pi
+\sum_i-\log p_\theta(y_i\mid x_i).
\end{aligned}
$$

常数 $\log Z_\pi$ 与 $\theta$ 无关，不改变最优解集合。证毕。

**反例 C.12（参数化边界）.** 在一维参数 $\theta$ 上使用二次惩罚 $\lambda\theta^2$，可对应某个高斯型先验密度。改用光滑双射 $\alpha=\sinh\theta$ 后，同一先验在 $\alpha$ 坐标中的密度为

$$
\pi_\alpha(\alpha)
=\pi_\theta(\operatorname{arsinh}\alpha)
\left|\frac{d}{d\alpha}\operatorname{arsinh}\alpha\right|
=\frac{\pi_\theta(\operatorname{arsinh}\alpha)}
{\sqrt{1+\alpha^2}}.
$$

因此其负对数不仅含 $\lambda(\operatorname{arsinh}\alpha)^2$，还含 Jacobian 项。直接写 $\lambda\alpha^2$ 对应另一个先验，而不是原先验的同一描述。MAP 不是无条件参数化不变的对象。

**边界 C.13.** 正则化可有数值稳定、容量控制、稀疏性、先验偏好或工程约束等多种解释。只有在损失确为负对数似然、惩罚确为可归一化先验的负对数密度、参数化与基准测度固定、优化足够接近所需极值时，MAP 解释才成立。MAP 给出一个点估计，不给出后验不确定性；权重衰减在自适应优化器中也不自动等同于某个贝叶斯后验计算。

## C.6 SGD、动量与 AdamW

设总体目标为

$$
F(\theta)=\mathbb E_{\xi}[f(\theta;\xi)].
$$

小批量梯度 $g_t$ 通常写作

$$
g_t=\frac1{|B_t|}\sum_{\xi\in B_t}\nabla_\theta f(\theta_t;\xi).
$$

若在给定 $\theta_t$ 后有

$$
\mathbb E[g_t\mid\theta_t]=\nabla F(\theta_t),
$$

则称它是无偏随机梯度；实际系统中的数据过滤、重采样、截断、混合权重和分布漂移可能使这个等式不成立。

**定义 C.14（SGD）.** 随机梯度下降更新为

$$
\theta_{t+1}=\theta_t-\eta_t g_t,
$$

其中 $\eta_t>0$ 是学习率。

**定义 C.15（重球动量）.** 一种常见动量形式为

$$
v_{t+1}=\beta v_t+g_t,\qquad
\theta_{t+1}=\theta_t-\eta_t v_{t+1},
$$

其中 $0\le\beta<1$。也有把 $(1-\beta)$ 乘入梯度项的等价尺度约定；比较实现时必须写清楚。

**定义 C.16（AdamW）.** AdamW 的典型逐坐标更新为

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
$$

$$
s_t=\beta_2s_{t-1}+(1-\beta_2)(g_t\odot g_t),
$$

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat s_t=\frac{s_t}{1-\beta_2^t},
$$

$$
\theta_{t+1}
=
\theta_t
-\eta_t\frac{\hat m_t}{\sqrt{\hat s_t}+\epsilon}
-\eta_t\lambda\theta_t.
$$

这里 $\odot$、平方根和除法均逐坐标进行。最后一项是 decoupled weight decay；它不是把 $\lambda\|\theta\|^2$ 的梯度简单加进 $g_t$ 的 Adam 版本。

**反例 C.17（步长过大导致发散）.** 令

$$
F(\theta)=\frac12\theta^2.
$$

精确梯度下降为

$$
\theta_{t+1}=\theta_t-\eta\theta_t=(1-\eta)\theta_t.
$$

若 $\eta>2$，则 $|1-\eta|>1$，只要 $\theta_0\ne0$，序列 $|\theta_t|=|1-\eta|^t|\theta_0|$ 发散。即使目标是光滑凸函数，错误步长也能破坏收敛。

**边界 C.18.** SGD、动量和 AdamW 是更新规则，不是成功证明。它们不保证非凸目标的全局最优，不保证训练样本之外的泛化，不保证概率校准，不保证对分布漂移稳健，不保证偏好目标与真实任务一致，也不保证不同硬件、不同内核或不同数据顺序给出同一模型工件。随机优化中的随机性也不是后验采样；除非另有定理，不能把多次训练的散布解释为贝叶斯不确定性。

## C.7 缩放点积自注意力

给定序列表示

$$
X\in\mathbb R^{L\times d_{\mathrm{model}}},
$$

单头 self-attention 通常定义为

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V,
$$

其中

$$
Q,K\in\mathbb R^{L\times d_k},\qquad
V\in\mathbb R^{L\times d_v}.
$$

缩放点积注意力为

$$
\operatorname{Attn}(X)
=
\operatorname{softmax}_{\mathrm{row}}
\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V,
$$

其中 $M$ 是掩码或偏置矩阵。

**命题 C.19（缩放项的方差理由）.** 若 $q,k\in\mathbb R^{d_k}$ 的各坐标独立，均值为 $0$，方差为 $1$，且 $q$ 与 $k$ 独立，则

$$
\operatorname{Var}\langle q,k\rangle=d_k,
\qquad
\operatorname{Var}\left(\frac{\langle q,k\rangle}{\sqrt{d_k}}\right)=1.
$$

**证明.** 点积为

$$
\langle q,k\rangle=\sum_{r=1}^{d_k}q_rk_r.
$$

每项满足

$$
\mathbb E[q_rk_r]=\mathbb E[q_r]\mathbb E[k_r]=0,
$$

且

$$
\operatorname{Var}(q_rk_r)
=\mathbb E[q_r^2k_r^2]
=\mathbb E[q_r^2]\mathbb E[k_r^2]
=1.
$$

不同坐标项独立，所以方差相加：

$$
\operatorname{Var}\langle q,k\rangle=d_k.
$$

再除以 $\sqrt{d_k}$，方差除以 $d_k$，得到 $1$。证毕。

**复杂度 C.20.** 计算 $QK^\top$ 需要 $O(L^2d_k)$ 次乘加，注意力矩阵乘以 $V$ 需要 $O(L^2d_v)$ 次乘加，显式注意力权重需要 $O(L^2)$ 存储。若有 $h$ 个头且 $hd_k$、$hd_v$ 与 $d_{\mathrm{model}}$ 同阶，则注意力主项为 $O(L^2d_{\mathrm{model}})$，线性投影另有 $O(Ld_{\mathrm{model}}^2)$ 主项。自回归推理使用 KV cache 时，单个新 token 对既有长度 $L$ 的注意力打分是线性的 $O(Ld_k)$；生成一段长度为 $n$ 的新文本时，注意力总量仍随前缀长度累积增长。

**边界 C.21.** 除以 $\sqrt{d_k}$ 是尺度控制，不是语义正确性定理。注意力权重高说明该层该头在该次前向中分配了较大混合系数；它不单独证明因果机制、来源支持或人类可读解释。稀疏注意力、低秩近似、块状注意力和特定内核可改变常数、存储路径或近似对象；若声称与全注意力等价，必须说明等价的数学对象和数值误差。

## C.8 偏好优化目标的范围

设提示为 $x$，候选回答为 $y$。成对偏好数据记录

$$
(x,y^+,y^-),
$$

表示在该记录的偏好标准下 $y^+$ 优于 $y^-$。这不是事实真值标签，除非偏好协议本身定义为事实核验且核验器可靠。

**定义 C.22（Bradley-Terry 奖励模型）.** 若奖励函数为 $r_\phi(x,y)$，成对偏好概率可写为

$$
P_\phi(y^+\succ y^-\mid x)
=
\sigma(r_\phi(x,y^+)-r_\phi(x,y^-)),
$$

其中 $\sigma(t)=1/(1+e^{-t})$。对应负对数损失为

$$
-\log\sigma(r_\phi(x,y^+)-r_\phi(x,y^-)).
$$

**定义 C.23（KL 正则化策略目标）.** 在有限回答集合 $\mathcal Y$ 上，给定参考策略 $\pi_0(y\mid x)>0$ 与奖励 $r(x,y)$，常见目标为

$$
\max_{\pi(\cdot\mid x)}
\left[
\sum_{y\in\mathcal Y}\pi(y\mid x)r(x,y)
-\beta
\sum_{y\in\mathcal Y}\pi(y\mid x)
\log\frac{\pi(y\mid x)}{\pi_0(y\mid x)}
\right],
$$

其中 $\beta>0$。

**命题 C.24（有限 KL 正则化最优策略）.** 对固定 $x$，上述目标的唯一最优策略为

$$
\pi^*(y\mid x)
=
\frac{\pi_0(y\mid x)\exp(r(x,y)/\beta)}
{\sum_{u\in\mathcal Y}\pi_0(u\mid x)\exp(r(x,u)/\beta)}.
$$

**证明.** 省略固定的 $x$。令

$$
Z=\sum_{u\in\mathcal Y}\pi_0(u)\exp(r(u)/\beta),
$$

并定义 $q(y)=\pi_0(y)\exp(r(y)/\beta)/Z$。对任意策略 $\pi$，

$$
\begin{aligned}
&\sum_y\pi(y)r(y)
-\beta\sum_y\pi(y)\log\frac{\pi(y)}{\pi_0(y)}\\
&=
\beta\sum_y\pi(y)\log\frac{\pi_0(y)\exp(r(y)/\beta)}{\pi(y)}\\
&=
\beta\sum_y\pi(y)\log\frac{Zq(y)}{\pi(y)}\\
&=
\beta\log Z
-\beta\sum_y\pi(y)\log\frac{\pi(y)}{q(y)}\\
&=
\beta\log Z-\beta D_{\mathrm{KL}}(\pi\Vert q),
\end{aligned}
$$

其中

$$
D_{\mathrm{KL}}(\pi\Vert q)
=
\sum_y\pi(y)\log\frac{\pi(y)}{q(y)}.
$$

卷三定理 P6.5 的 Gibbs 不等式给出 $D_{\mathrm{KL}}(\pi\Vert q)\ge0$，且等号当且仅当 $\pi=q$。因此最大值在且仅在 $\pi=q$ 时取得。证毕。

**定义 C.25（DPO 型成对目标）.** 给定策略 $\pi_\theta$ 与参考策略 $\pi_0$，DPO 型目标常使用

$$
-\log\sigma\left(
\beta\left[
\log\frac{\pi_\theta(y^+\mid x)}{\pi_0(y^+\mid x)}
-
\log\frac{\pi_\theta(y^-\mid x)}{\pi_0(y^-\mid x)}
\right]\right).
$$

它直接改变成对偏好下的相对概率比，不需要显式训练一个在线环境中的动作价值函数。

**反例 C.26（偏好不等于真值）.** 若标注协议系统性偏好“语气肯定但事实错误”的回答，而不偏好“承认未知并请求核验”的回答，则偏好目标会推动策略提高前者的相对概率。此时优化目标按数据定义成功，事实可靠性却可能下降。问题不在公式代数错误，而在偏好数据和任务目标不一致。

**边界 C.27.** 偏好优化目标的范围由提示分布、候选生成机制、标注说明、参考策略支持集、奖励模型容量和 KL 系数共同限定。它不保证所有用户都会偏好，不保证所有事实主张为真，不保证安全约束在分布外成立，不保证可见理由忠实，也不保证工具行动合规。审计偏好优化时，应把训练信号、部署策略、当前输出证据和风险控制分别记录。

## C.9 平方损失的偏差—方差分解

设 $S$ 是训练样本及训练随机性，$\widehat f_S$ 是由 $S$ 得到的实值预测器，$(X,Y)$ 是独立于训练过程的新测试样本。固定一个使下列条件矩存在的输入 $x$，并记

$$
m(x)=\mathbb E[Y\mid X=x],\qquad
\sigma^2(x)=\operatorname{Var}(Y\mid X=x),
$$

$$
\overline f(x)=\mathbb E_S[\widehat f_S(x)].
$$

这里的条件期望可理解为离散输入上的通常条件期望，或连续输入上选定的正则条件分布版本。关键条件是训练随机对象 $S$ 与新测试响应 $Y$ 在给定 $X=x$ 后独立；没有这个条件，训练过程与测试噪声之间可能出现额外交叉项。

**定理 C.28（条件偏差—方差分解）.** 在上述独立性和二阶矩条件下，

$$
\begin{aligned}
\mathbb E_{S,Y}\!\left[
(Y-\widehat f_S(x))^2\mid X=x
\right]
&=\sigma^2(x)
+\bigl(m(x)-\overline f(x)\bigr)^2\\
&\quad+
\mathbb E_S\!\left[
(\widehat f_S(x)-\overline f(x))^2
\right].
\end{aligned}
$$

右侧三项依次称为条件噪声、平方偏差和预测器方差。

**证明.** 为缩短记号，固定 $x$ 后写 $m=m(x)$、$H=\widehat f_S(x)$。先对 $S$ 条件化。由给定 $X=x$ 时 $S$ 与 $Y$ 的条件独立性，

$$
\mathbb E[Y\mid S,X=x]=m,
\qquad
\operatorname{Var}(Y\mid S,X=x)=\sigma^2(x).
$$

因此条件平方损失满足

$$
\begin{aligned}
\mathbb E[(Y-H)^2\mid S,X=x]
&=\mathbb E[((Y-m)+(m-H))^2\mid S,X=x]\\
&=\mathbb E[(Y-m)^2\mid S,X=x]
+(m-H)^2\\
&\quad+2(m-H)\mathbb E[Y-m\mid S,X=x]\\
&=\sigma^2(x)+(m-H)^2.
\end{aligned}
$$

对 $S$ 再取期望，得到

$$
\mathbb E_{S,Y}[(Y-H)^2\mid X=x]
=\sigma^2(x)+\mathbb E_S[(m-H)^2].
$$

令 $\overline f=\mathbb E_S[H]$。第二项继续展开为

$$
\begin{aligned}
\mathbb E_S[(m-H)^2]
&=\mathbb E_S[((m-\overline f)+(\overline f-H))^2]\\
&=(m-\overline f)^2
+\mathbb E_S[(H-\overline f)^2]\\
&\quad+2(m-\overline f)\mathbb E_S[\overline f-H].
\end{aligned}
$$

最后一项为零，因为 $\mathbb E_S[\overline f-H]=0$。代回即得所述分解。证毕。

**边界 C.29.** 该等式针对平方损失和指定的训练随机机制。若损失不是平方损失，通常不存在同样的三项恒等式；若 $S$ 与测试样本共享个体、时间段或泄漏特征，条件独立性可能失败；若再对 $X$ 积分，得到的是相对于所声明测试输入分布的总体分解，而不是对任意部署分布都成立的结论。较小训练误差也不能单独识别右侧三项。

## C.10 仿射半空间的 VC 维

令

$$
\mathcal H_d=
\left\{
x\longmapsto \mathbf 1\{\langle w,x\rangle+b\ge0\}:
w\in\mathbb R^d,\ b\in\mathbb R
\right\}.
$$

**定义 C.30（打散与 VC 维）.** 若有限点集 $A=\{x_1,\ldots,x_n\}$ 的每个二元标记 $(\ell_1,\ldots,\ell_n)\in\{0,1\}^n$ 都能由某个 $h\in\mathcal H_d$ 实现，即 $h(x_i)=\ell_i$，则称 $\mathcal H_d$ 打散 $A$。VC 维是可被打散点集大小的上确界；若任意大有限点集都可被打散，则记为无穷。

上界需要一个有限维凸几何事实。这里先在书内证明它。

**引理 C.31（有限维 Radon 定理）.** 任取 $d+2$ 个点 $x_1,\ldots,x_{d+2}\in\mathbb R^d$，指标集可分成两个不交的非空集合 $A,B$，使

$$
\operatorname{conv}\{x_i:i\in A\}
\cap
\operatorname{conv}\{x_i:i\in B\}
\ne\varnothing.
$$

**证明.** 考察 $d+2$ 个增广向量

$$
(x_i,1)\in\mathbb R^{d+1}.
$$

它们在 $d+1$ 维空间中线性相关，所以存在不全为零的实数 $\alpha_1,\ldots,\alpha_{d+2}$，满足

$$
\sum_i\alpha_i x_i=0,
\qquad
\sum_i\alpha_i=0.
$$

令 $P=\{i:\alpha_i>0\}$、$N=\{i:\alpha_i<0\}$。因为系数不全为零且总和为零，$P,N$ 都非空，并且

$$
c:=\sum_{i\in P}\alpha_i
=-\sum_{i\in N}\alpha_i>0.
$$

于是

$$
z=
\sum_{i\in P}\frac{\alpha_i}{c}x_i
=
\sum_{i\in N}\frac{-\alpha_i}{c}x_i.
$$

两边的系数均非负且各自和为 $1$，故 $z$ 同时属于 $P$ 中点的凸包和 $N$ 中点的凸包。把满足 $\alpha_i=0$ 的其余指标任意放入 $P$ 一侧，得到覆盖全部指标的分拆 $A,B$；扩大一侧点集不会移除原来的凸包交点。因此所需分拆存在。证毕。

**定理 C.32（仿射半空间的 VC 维）.** 对 $d\ge1$，

$$
\operatorname{VCdim}(\mathcal H_d)=d+1.
$$

**证明.** 先证下界。取 $d+1$ 个点

$$
x_0=0,\qquad x_i=e_i\quad(1\le i\le d),
$$

其中 $e_i$ 是第 $i$ 个标准基向量。给定任意标签 $\ell_i\in\{0,1\}$，令 $s_i=2\ell_i-1\in\{-1,+1\}$，并取

$$
b=s_0,\qquad w_i=s_i-s_0.
$$

则

$$
\langle w,x_0\rangle+b=s_0,
\qquad
\langle w,x_i\rangle+b=w_i+b=s_i.
$$

所有打分都严格等于 $+1$ 或 $-1$，所以阈值规则恰好实现给定标签。故这 $d+1$ 个点被打散，VC 维至少为 $d+1$。

再证上界。任取 $d+2$ 个点。由引理 C.31，存在非空分拆 $A,B$ 和一点

$$
z\in\operatorname{conv}\{x_i:i\in A\}
\cap\operatorname{conv}\{x_i:i\in B\}.
$$

把 $A$ 中点标为 $1$，把 $B$ 中点标为 $0$。若某个仿射函数 $f(x)=\langle w,x\rangle+b$ 实现该标记，则 $i\in A$ 时 $f(x_i)\ge0$，所以由仿射性，$z$ 作为这些点的凸组合满足 $f(z)\ge0$。另一方面，$i\in B$ 时必须有 $f(x_i)<0$；$B$ 非空，而 $z$ 是 $B$ 中点的凸组合，故

$$
f(z)=\sum_{i\in B}\lambda_i f(x_i)<0,
$$

其中 $\lambda_i\ge0$、$\sum_{i\in B}\lambda_i=1$。这与 $f(z)\ge0$ 矛盾。因此任意 $d+2$ 个点都有一种不能实现的标记，VC 维至多为 $d+1$。结合上下界即得结论。证毕。

**边界 C.33.** 该结论针对 $\mathbb R^d$ 上带偏置的全部仿射半空间。若强制 $b=0$，齐次半空间的 VC 维改变；若限制权重、间隔或输入域，还需重新分析。VC 维控制的是一个假设类打散有限点集的组合容量，不直接给出某个具体训练算法的测试误差，也不表示 $d+1$ 个任意点都能被打散；下界使用了特定仿射无关点集。

## C.11 一维分段线性函数的有限 ReLU 精确表示

记 $\rho(u)=\max\{u,0\}$。设 $f:\mathbb R\to\mathbb R$ 连续，并存在有限断点

$$
t_1<t_2<\cdots<t_m,
$$

使 $f$ 在每个区间 $(-\infty,t_1),(t_1,t_2),\ldots,(t_m,\infty)$ 上都是仿射函数。记这些区间上的斜率依次为 $s_0,s_1,\ldots,s_m$。

**定理 C.34（有限折线的 ReLU 精确表示）.** 存在常数 $a$，使对所有 $x\in\mathbb R$，

$$
f(x)=a+s_0x+
\sum_{j=1}^m(s_j-s_{j-1})\rho(x-t_j).
$$

因此，允许仿射直连项时，$m$ 个断点由 $m$ 个 ReLU 单元精确表示；若网络形式不允许输入到输出的线性直连，则利用

$$
x=\rho(x)-\rho(-x)
$$

可由至多 $m+2$ 个隐藏 ReLU 单元实现同一函数。

**证明.** 在最左区间上写 $f(x)=a+s_0x$；由于该区间非空，常数 $a$ 唯一确定。令

$$
g(x)=a+s_0x+
\sum_{j=1}^m(s_j-s_{j-1})\rho(x-t_j).
$$

当 $x<t_1$ 时所有 ReLU 项为零，故 $g(x)=f(x)$。当 $t_k<x<t_{k+1}$ 时，恰有前 $k$ 个 ReLU 项处于线性区间，因此 $g$ 的斜率为

$$
s_0+\sum_{j=1}^k(s_j-s_{j-1})=s_k,
$$

这与 $f$ 在该区间的斜率相同；在 $x>t_m$ 时同样得到斜率 $s_m$。于是 $f-g$ 在每个开区间上为常数。两者都连续，并且在最左区间上相等；跨过每个断点时，连续性迫使相邻区间上的常数差仍为零。归纳越过全部断点可知 $f=g$ 在整个实线上成立。最后用 $x=\rho(x)-\rho(-x)$ 替换仿射直连项，即得标准单隐藏层表示。证毕。

**边界 C.35.** 这是对一维、连续、只有有限个断点的分段线性函数的精确表示定理，不是一般的通用逼近定理。它不声称用有限 ReLU 精确表示任意连续函数，也不处理无限断点、跳跃不连续或一般高维分片几何。通用逼近结论讨论的通常是在指定紧集和误差范数下逼近函数；其量词、误差和宽度依赖必须另行声明。

## C.12 近端稀疏化、inverted dropout 与 EMA

**定义 C.36（$L^1$ 近端映射）.** 对 $v\in\mathbb R^d$ 和 $\lambda\ge0$，定义

$$
\operatorname{prox}_{\lambda\|\cdot\|_1}(v)
=
\operatorname*{arg\,min}_{u\in\mathbb R^d}
\left\{
\frac12\|u-v\|_2^2+\lambda\|u\|_1
\right\}.
$$

二次项严格凸，所以极小点唯一。

**命题 C.37（soft-threshold 公式）.** 上述近端映射逐坐标满足

$$
[\operatorname{prox}_{\lambda\|\cdot\|_1}(v)]_i
=\operatorname{sign}(v_i)(|v_i|-\lambda)_+,
$$

其中 $(r)_+=\max\{r,0\}$，并约定 $\operatorname{sign}(0)=0$。

**证明.** 目标按坐标可分，只需最小化标量函数

$$
q(u)=\frac12(u-v)^2+\lambda|u|.
$$

最优性条件是

$$
0\in u-v+\lambda\,\partial|u|.
$$

若 $u>0$，则 $\partial|u|=\{1\}$，故 $u=v-\lambda$；该解满足 $u>0$ 当且仅当 $v>\lambda$。若 $u<0$，则 $\partial|u|=\{-1\}$，故 $u=v+\lambda$；该解满足 $u<0$ 当且仅当 $v<-\lambda$。若 $u=0$，则 $\partial|u|=[-1,1]$，条件变为 $v\in[-\lambda,\lambda]$。三个情形恰好给出

$$
u=
\begin{cases}
v-\lambda,&v>\lambda,\\
0,&|v|\le\lambda,\\
v+\lambda,&v<-\lambda,
\end{cases}
$$

即 soft-threshold 公式。逐坐标组合后得到向量结论。证毕。

**命题 C.38（inverted dropout 的条件矩）.** 固定向量 $h\in\mathbb R^d$，令保留率 $q\in(0,1]$，并令 $M_i$ 相互独立且满足 $M_i\sim\operatorname{Bernoulli}(q)$。定义

$$
\widetilde h_i=\frac{M_i}{q}h_i.
$$

则

$$
\mathbb E[\widetilde h_i\mid h]=h_i,
\qquad
\operatorname{Var}(\widetilde h_i\mid h)
=\frac{1-q}{q}h_i^2,
$$

且 $i\ne j$ 时 $\operatorname{Cov}(\widetilde h_i,\widetilde h_j\mid h)=0$。

**证明.** 因为 $\mathbb E[M_i]=q$，有

$$
\mathbb E[\widetilde h_i\mid h]
=\frac{h_i}{q}\mathbb E[M_i]=h_i.
$$

又因 $M_i^2=M_i$，

$$
\begin{aligned}
\operatorname{Var}(\widetilde h_i\mid h)
&=\frac{h_i^2}{q^2}\operatorname{Var}(M_i)\\
&=\frac{h_i^2}{q^2}q(1-q)
=\frac{1-q}{q}h_i^2.
\end{aligned}
$$

不同坐标的掩码独立，故条件协方差为零。证毕。

**命题 C.39（指数移动平均展开）.** 给定 $0\le\beta<1$，递推

$$
m_t=\beta m_{t-1}+(1-\beta)x_t
$$

满足

$$
m_t=\beta^t m_0
+(1-\beta)\sum_{k=1}^t\beta^{t-k}x_k.
$$

特别地，若 $m_0=0$，历史样本权重之和为 $1-\beta^t$。

**证明.** $t=1$ 时公式就是递推定义。若公式对 $t-1$ 成立，则

$$
\begin{aligned}
m_t
&=\beta\left[
\beta^{t-1}m_0
+(1-\beta)\sum_{k=1}^{t-1}\beta^{t-1-k}x_k
\right]+(1-\beta)x_t\\
&=\beta^t m_0
+(1-\beta)\sum_{k=1}^{t}\beta^{t-k}x_k.
\end{aligned}
$$

故结论由归纳法成立。若 $m_0=0$，权重和是等比级数

$$
(1-\beta)\sum_{k=1}^t\beta^{t-k}=1-\beta^t.
$$

证毕。

**边界 C.40.** Soft-threshold 是给定欧氏近端子问题的精确解，不表示一般神经网络加 $L^1$ 惩罚后可得到全局最优。Inverted dropout 只在线性激活坐标上保持条件期望；对非线性 $\phi$，通常 $\mathbb E[\phi(\widetilde h)]\ne\phi(h)$，方差也随 $q\downarrow0$ 增大。EMA 在 $m_0=0$ 时除以 $1-\beta^t$ 可消除恒定均值序列的初始化缩放，但对分布漂移序列只得到加权历史平均，不能无条件称为当前值的无偏估计。

## C.13 有限一维离散卷积的反向梯度

为固定索引约定，设输入 $x=(x_0,\ldots,x_{N-1})$、核 $k=(k_0,\ldots,k_{K-1})$，其中 $1\le K\le N$。采用深度学习实现中常见的互相关记号，定义 valid 输出

$$
y_t=\sum_{r=0}^{K-1}k_r x_{t+r},
\qquad 0\le t<T:=N-K+1.
$$

设标量损失 $L=L(y_0,\ldots,y_{T-1})$ 可微，并记 $\delta_t=\partial L/\partial y_t$。

**定理 C.41（单通道卷积层的反向公式）.** 对 $0\le r<K$ 和 $0\le i<N$，

$$
\frac{\partial L}{\partial k_r}
=\sum_{t=0}^{T-1}\delta_t x_{t+r},
$$

$$
\frac{\partial L}{\partial x_i}
=
\sum_{\substack{0\le r<K\\0\le i-r<T}}
\delta_{i-r}k_r.
$$

**证明.** 输出微分为

$$
dy_t=\sum_{r=0}^{K-1}(x_{t+r}\,dk_r+k_r\,dx_{t+r}).
$$

代入 $dL=\sum_{t=0}^{T-1}\delta_t\,dy_t$，得到

$$
\begin{aligned}
dL
&=\sum_{t=0}^{T-1}\sum_{r=0}^{K-1}
\delta_t x_{t+r}\,dk_r\\
&\quad+
\sum_{t=0}^{T-1}\sum_{r=0}^{K-1}
\delta_t k_r\,dx_{t+r}.
\end{aligned}
$$

第一项按 $dk_r$ 收集系数，直接给出核梯度。第二项中令 $i=t+r$；固定 $i$ 时，合法项必须同时满足 $0\le r<K$ 和 $0\le i-r<T$，收集 $dx_i$ 的系数即得输入梯度。证毕。

多通道、多输出通道的推广只增加有限求和指标。若

$$
y_{o,t}=b_o+
\sum_{c=1}^{C_{\mathrm{in}}}
\sum_{r=0}^{K-1}k_{o,c,r}x_{c,t+r},
$$

并记 $\delta_{o,t}=\partial L/\partial y_{o,t}$，则同一微分计算给出

$$
\frac{\partial L}{\partial k_{o,c,r}}
=\sum_{t=0}^{T-1}\delta_{o,t}x_{c,t+r},
\qquad
\frac{\partial L}{\partial b_o}
=\sum_{t=0}^{T-1}\delta_{o,t},
$$

$$
\frac{\partial L}{\partial x_{c,i}}
=\sum_o
\sum_{\substack{0\le r<K\\0\le i-r<T}}
\delta_{o,i-r}k_{o,c,r}.
$$

**边界 C.42.** 上式固定了 valid、步长 $1$、膨胀率 $1$ 和互相关索引约定。若把“卷积”定义为先翻转核，或加入 padding、stride、dilation、groups、循环边界，合法索引集合会相应改变；反向公式仍由同一个微分收集过程推出，不能在未对齐约定时只凭图形直觉翻转核。公式证明的是给定算子的局部梯度，不证明卷积架构适合任何具体数据。

## C.14 RNN Jacobian 链与 LSTM 门控边界

考虑隐藏维数为 $m$ 的基本循环网络

$$
a_t=Wh_{t-1}+Ux_t+b,
\qquad
h_t=\phi(a_t),
$$

其中 $\phi$ 逐坐标作用。沿一条可微前向路径，令

$$
D_t=\operatorname{diag}(\phi'(a_t)),
\qquad
J_t=\frac{\partial h_t}{\partial h_{t-1}}=D_tW.
$$

**定理 C.43（RNN 反向 Jacobian 乘积及条件界）.** 若损失只通过 $h_T$ 依赖时刻 $t\le T$ 的隐藏状态，记 $g_T=\nabla_{h_T}L$，则

$$
\nabla_{h_t}L
=J_{t+1}^{\top}J_{t+2}^{\top}\cdots J_T^{\top}g_T.
$$

采用 Euclidean 算子范数，若对 $k=t+1,\ldots,T$ 有

$$
\|J_k\|_2\le\rho_+,
$$

则

$$
\|\nabla_{h_t}L\|_2
\le \rho_+^{T-t}\|g_T\|_2.
$$

若还对这些方阵有

$$
\sigma_{\min}(J_k)\ge\rho_->0,
$$

则

$$
\|\nabla_{h_t}L\|_2
\ge \rho_-^{T-t}\|g_T\|_2.
$$

因此 $\rho_+<1$ 是该路径梯度按距离指数衰减的充分条件，而 $\rho_->1$ 是非零终端梯度按距离指数增长的充分条件。

**证明.** 链式法则给出

$$
\nabla_{h_{T-1}}L=J_T^\top g_T.
$$

反复应用同一公式，归纳得到所述 Jacobian 乘积。由算子范数的次乘性和 $\|A^\top\|_2=\|A\|_2$，

$$
\begin{aligned}
\|\nabla_{h_t}L\|_2
&\le
\prod_{k=t+1}^T\|J_k^\top\|_2\,\|g_T\|_2\\
&\le \rho_+^{T-t}\|g_T\|_2.
\end{aligned}
$$

另一方面，对任意方阵 $A$ 和向量 $v$，奇异值定义给出

$$
\|A^\top v\|_2\ge\sigma_{\min}(A)\|v\|_2.
$$

从乘积最右侧开始逐次应用该不等式，得到

$$
\|J_{t+1}^{\top}\cdots J_T^{\top}g_T\|_2
\ge
\prod_{k=t+1}^T\sigma_{\min}(J_k)\|g_T\|_2
\ge\rho_-^{T-t}\|g_T\|_2.
$$

证毕。

一种不含 peephole 连接的 LSTM 门控接口可写为

$$
\begin{aligned}
i_t&=\sigma(W_i x_t+U_i h_{t-1}+b_i),\\
f_t&=\sigma(W_f x_t+U_f h_{t-1}+b_f),\\
o_t&=\sigma(W_o x_t+U_o h_{t-1}+b_o),\\
\widetilde c_t&=\tanh(W_c x_t+U_c h_{t-1}+b_c),\\
c_t&=f_t\odot c_{t-1}+i_t\odot\widetilde c_t,\\
h_t&=o_t\odot\tanh(c_t).
\end{aligned}
$$

沿显式的 cell-state 直连边、把门值视为该次前向已经固定的量时，

$$
\left.\frac{\partial c_t}{\partial c_{t-1}}\right|_{\mathrm{direct}}
=\operatorname{diag}(f_t),
$$

所以从 $c_t$ 到 $c_T$ 的直接路径因子为

$$
\operatorname{diag}
\left(
\prod_{k=t+1}^T f_k
\right),
$$

其中乘积逐坐标进行。这个接口允许网络在 $f_k$ 接近 $1$ 时减弱直接路径的衰减。

**边界 C.44.** $\|J_k\|_2>1$ 的上界本身不证明梯度必然爆炸，因为矩阵方向、奇异向量和后续乘积可能抵消；同样，某个平均谱量小于 $1$ 也不能替代逐路径条件。若损失在多个时刻直接注入，梯度是多条 Jacobian 乘积之和。LSTM 的门值依赖 $h_{t-1}$，总 Jacobian 还包含经过各门的间接项；上面的对角乘积只描述 cell-state 直连路径。有限精度、门饱和、训练目标和参数状态都可能破坏信息保留，因而不能从门控接口无条件推出长期记忆。

## C.15 有限动作策略梯度与 PPO 代理目标

考虑有限时域 $T$、有限状态和动作集合。初始分布 $\mu$ 与转移核 $P$ 不依赖参数 $\theta$，策略 $\pi_\theta(a\mid s)$ 可微，并在所考虑轨迹上为正。轨迹

$$
\tau=(s_0,a_0,s_1,a_1,\ldots,s_T)
$$

的概率为

$$
p_\theta(\tau)
=\mu(s_0)
\prod_{t=0}^{T-1}
\pi_\theta(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t).
$$

设有界回报 $R(\tau)$ 本身不显含 $\theta$，并定义 $J(\theta)=\mathbb E_{\tau\sim p_\theta}[R(\tau)]$。

**定理 C.45（有限轨迹策略梯度恒等式）.** 在上述条件下，

$$
\nabla_\theta J(\theta)
=
\mathbb E_{\tau\sim p_\theta}
\left[
R(\tau)
\sum_{t=0}^{T-1}
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right].
$$

**证明.** 轨迹集合有限，故可逐项求导：

$$
\nabla_\theta J(\theta)
=\sum_\tau R(\tau)\nabla_\theta p_\theta(\tau).
$$

在 $p_\theta(\tau)>0$ 的轨迹上，

$$
\nabla_\theta p_\theta(\tau)
=p_\theta(\tau)\nabla_\theta\log p_\theta(\tau).
$$

初始分布和转移核与 $\theta$ 无关，所以

$$
\nabla_\theta\log p_\theta(\tau)
=\sum_{t=0}^{T-1}
\nabla_\theta\log\pi_\theta(a_t\mid s_t).
$$

代回有限和并识别为对 $p_\theta$ 的期望，即得结论。证毕。

**命题 C.46（单步 importance ratio 恒等式）.** 固定状态 $s$。若旧策略满足

$$
\pi_{\mathrm{old}}(a\mid s)>0
\quad\text{whenever}\quad
\pi_\theta(a\mid s)>0,
$$

定义

$$
r_\theta(s,a)
=\frac{\pi_\theta(a\mid s)}
{\pi_{\mathrm{old}}(a\mid s)}.
$$

则对有限动作集合上的任意函数 $q(s,a)$，

$$
\mathbb E_{a\sim\pi_{\mathrm{old}}(\cdot\mid s)}
[r_\theta(s,a)q(s,a)]
=
\mathbb E_{a\sim\pi_\theta(\cdot\mid s)}[q(s,a)].
$$

**证明.** 直接展开有限和：

$$
\begin{aligned}
\sum_a\pi_{\mathrm{old}}(a\mid s)
r_\theta(s,a)q(s,a)
&=\sum_a\pi_\theta(a\mid s)q(s,a).
\end{aligned}
$$

支持集条件保证所有需要的比值都有定义。证毕。

**定义 C.47（PPO clipped surrogate）.** 给定旧策略采集的数据和优势估计 $\widehat A_t$，PPO clipped surrogate 定义为最大化

$$
L^{\mathrm{clip}}(\theta)
=\mathbb E_{\mathrm{old}}
\left[
\min\left\{
r_t(\theta)\widehat A_t,
\operatorname{clip}(r_t(\theta),1-\varepsilon,1+\varepsilon)
\widehat A_t
\right\}
\right],
$$

其中

$$
r_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\mathrm{old}}(a_t\mid s_t)},
\qquad \varepsilon>0.
$$

当 $\widehat A_t>0$ 时，样本项截住比值向上增长带来的代理收益；当 $\widehat A_t<0$ 时，它截住比值向下缩小带来的代理收益。

**反例 C.48（clipping 不是硬比值约束）.** 对一个满足 $\widehat A_t>0$ 的样本，只要 $r_t(\theta)\ge1+\varepsilon$，该样本的 clipped 项恒为 $(1+\varepsilon)\widehat A_t$。因此把比值从 $1+\varepsilon$ 继续增大不会再改变这个样本项，却也不会被该项禁止。若旧动作概率很小，新旧概率比可以远大于 $1+\varepsilon$。所以目标中的 clip 不能解释为约束每个样本都满足 $r_t\in[1-\varepsilon,1+\varepsilon]$。

**边界 C.49.** 单步 importance ratio 只修正固定状态下的动作分布；若要精确改写整条轨迹分布，通常需要各时刻比值的乘积，并满足轨迹支持集条件。策略梯度恒等式不降低估计方差，也不保证有限样本估计准确；若奖励、环境或动作掩码显含 $\theta$，还会出现额外导数项。PPO 目标依赖旧策略数据、优势估计、批次和重复更新，它不是一般的单调性能改进定理，不保证每状态 KL 上界、全局最优、事实正确、安全约束或分布外行为。

## C.16 InfoNCE 的行交叉熵结构

**定义 C.50（批内 InfoNCE 行损失）.** 给定批大小 $B\ge2$，设 $s_{ij}$ 是第 $i$ 个锚点与第 $j$ 个候选的相似度，温度 $\tau>0$，并假定第 $i$ 行的指定正例是 $j=i$。定义 logits 和行概率

$$
z_{ij}=\frac{s_{ij}}{\tau},
\qquad
p_{ij}=\frac{e^{z_{ij}}}{\sum_{k=1}^B e^{z_{ik}}}.
$$

InfoNCE 批损失就是逐行交叉熵的平均：

$$
L_{\mathrm{NCE}}
=-\frac1B\sum_{i=1}^B\log p_{ii}
=-\frac1B\sum_{i=1}^B
\log\frac{\exp(s_{ii}/\tau)}
{\sum_{j=1}^B\exp(s_{ij}/\tau)}.
$$

**定理 C.51（InfoNCE 对相似度 logits 的梯度）.** 对任意 $i,j$，

$$
\frac{\partial L_{\mathrm{NCE}}}{\partial z_{ij}}
=\frac1B(p_{ij}-\mathbf 1\{i=j\}),
$$

因而

$$
\frac{\partial L_{\mathrm{NCE}}}{\partial s_{ij}}
=\frac1{B\tau}(p_{ij}-\mathbf 1\{i=j\}).
$$

**证明.** 第 $i$ 行损失为

$$
\ell_i=-z_{ii}+\log\sum_{k=1}^B e^{z_{ik}}.
$$

对 $z_{ij}$ 求导得到

$$
\frac{\partial\ell_i}{\partial z_{ij}}
=-\mathbf 1\{i=j\}
+\frac{e^{z_{ij}}}{\sum_k e^{z_{ik}}}
=p_{ij}-\mathbf 1\{i=j\}.
$$

总损失对各行取 $1/B$ 平均，得到第一式。再由 $z_{ij}=s_{ij}/\tau$ 应用链式法则，得到第二式。证毕。

**外部输入 C.52（InfoNCE 互信息下界）.** 可调用的标准版本需要明确如下概率实验：先相互独立地采样 $X\sim P_X$ 和均匀正例位置 $I\in\{1,\ldots,B\}$；条件于 $(X,I)$，令 $Y_I\sim P_{Y\mid X}(\cdot\mid X)$，并令 $(Y_j)_{j\ne I}$ 相互独立地服从边缘分布 $P_Y$，且与 $Y_I$ 条件独立。对严格为正、可测且使下列期望有限的评分函数 $f(X,Y)$，相应的总体分类损失

$$
\mathcal L_B
=\mathbb E\left[
-\log\frac{f(X,Y_I)}{\sum_{j=1}^B f(X,Y_j)}
\right]
$$

满足外部定理

$$
I(X;Y)\ge\log B-\mathcal L_B.
$$

本附录不重证该不等式；任何使用都必须提供可核查的定理版本，并逐项核对联合分布、负例独立同分布、正例位置、评分函数可积性和总体期望。实际批次中的相关样本、无放回抽样、hard-negative mining、多个真阳性或假阴性不会自动满足这些假设。

**边界 C.53.** 行交叉熵定义及其梯度是给定有限 logits 后的代数事实，不依赖互信息解释。温度改变 logits 尺度和梯度尺度；相似度还通过编码器参数产生，完整参数梯度需继续应用链式法则。只有满足外部输入 C.52 的采样模型时，才能把特定总体目标登记为互信息下界；经验小批量损失较低本身不证明表征保留了任务所需信息、因果信息或可解释语义。

## 练习

**练习 C.1.** 设样本在线性可分条件下满足 $\|\tilde x_i\|\le R$，按感知机规则只在 $y_i\langle\tilde w,\tilde x_i\rangle<0$ 时更新。证明更新次数仍有不超过 $(R/\gamma)^2$ 的上界，或指出需要调整的边界条件。

**练习 C.2.** 对两层网络 $h=\operatorname{ReLU}(Wx+b)$、$o=Uh+c$ 和平方损失 $\frac12\|o-y\|^2$，写出一次反向传播中 $U,c,W,b$ 的梯度表达式，并说明 ReLU 在零点处的约定。

**练习 C.3.** 证明 softmax 对 logits 的共同平移不变，并解释为什么交叉熵对 logits 的梯度各坐标之和为零。

**练习 C.4.** 给出一个偏好数据协议，使 DPO 型目标可能提高格式满意度但降低事实准确率；写清楚提示分布、候选回答和偏好规则。

**练习 C.5.** 在定理 C.28 的设定中去掉 $S$ 与 $Y$ 在给定 $X=x$ 后的条件独立性。完整展开平方损失，写出新增的协方差项，并构造一个该项非零的有限概率例子。

**练习 C.6.** 证明实线上的闭区间指示函数不能由有限个 ReLU 的连续线性组合精确表示；再构造一个连续三段折线函数，用定理 C.34 写出其最少断点表示，并验证每个区间的斜率。

**练习 C.7.** 对长度为 $5$ 的输入、长度为 $3$ 的核和 valid 互相关，逐项写出全部输出及输入梯度；随后把同一计算改成左、右各补一个零的 padding 约定，比较合法索引集合。

**练习 C.8.** 在两动作单状态模型中取旧策略概率 $(0.99,0.01)$，为第二个动作指定正优势。构造一个新策略，使该动作的 importance ratio 超过 $1+\varepsilon$ 而对应 PPO clipped 样本项保持不变；再写出同一对动作作为 $B=2$ 的一行 InfoNCE logits 时的两个梯度分量。
