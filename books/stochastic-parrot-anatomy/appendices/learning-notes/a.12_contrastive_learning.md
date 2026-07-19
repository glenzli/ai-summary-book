# 附录 A.12 对比学习、InfoNCE 与互信息边界

本附录为卷一第七章的双编码器目标补足三个细节：batch 内分类形式、表示梯度，以及互信息下界成立所需的负样本采样假设。InfoNCE 是一个目标族；并非任意“拉近正样本、推远负样本”的损失都满足同一个信息论结论。

## A.12.1 配对 batch 与相似度

令 $(x_i,y_i)_{i=1}^B$ 为独立同分布配对样本。两个编码器输出非零向量，再归一化为

$$
u_i=\frac{f_\theta(x_i)}{\|f_\theta(x_i)\|},
\qquad
v_i=\frac{g_\phi(y_i)}{\|g_\phi(y_i)\|}.
\tag{A.12.1}
$$

取温度 $\tau>0$，定义

$$
s_{ij}=\frac{u_i^\mathsf Tv_j}{\tau}.
\tag{A.12.2}
$$

归一化后 $u_i^\mathsf Tv_j\in[-1,1]$；若不做归一化，表示范数也会改变 logit 尺度，此时 $\tau$ 不能单独解释尖锐程度。

## A.12.2 单向与双向 InfoNCE

从 $x$ 检索 $y$ 的行 softmax 为

$$
p_{ij}^{x\to y}
=\frac{e^{s_{ij}}}{\sum_{k=1}^Be^{s_{ik}}}.
$$

单向损失是

$$
L_{x\to y}
=-\frac1B\sum_{i=1}^B
\log p_{ii}^{x\to y}.
\tag{A.12.3}
$$

反方向使用列 softmax：

$$
L_{y\to x}
=-\frac1B\sum_{j=1}^B
\log
\frac{e^{s_{jj}}}{\sum_{i=1}^Be^{s_{ij}}}.
\tag{A.12.4}
$$

CLIP 使用两者平均的一种对称形式：

$$
L_{\mathrm{sym}}
=\frac12(L_{x\to y}+L_{y\to x}).
\tag{A.12.5}
$$

每个分母都应使用 log-sum-exp 稳定计算。分布式训练若把其他设备样本 `all-gather` 为 negatives，分母、有效 batch 大小和梯度是否穿过 gather 都属于目标实现的一部分。

## A.12.3 Logit 与表示梯度

对 (A.12.3)，

$$
\boxed{
\frac{\partial L_{x\to y}}{\partial s_{ij}}
=\frac1B(p_{ij}^{x\to y}-\mathbf 1\{i=j\}).}
\tag{A.12.6}
$$

由 $s_{ij}=u_i^\mathsf Tv_j/\tau$，

$$
\boxed{
\nabla_{u_i}L_{x\to y}
=\frac1{B\tau}
\left(\sum_jp_{ij}^{x\to y}v_j-v_i\right).}
\tag{A.12.7}
$$

所以梯度提高正配对相对于当前 softmax 加权候选均值的相似度；它不是给每个非对角元素施加相同大小的排斥力。$v_j$ 的梯度还会累加所有把它当候选的行。对称损失再加入列方向的对应贡献。

若未归一化表示为 $h\ne0$、$u=h/\|h\|$，则

$$
\frac{\partial u}{\partial h}
=\frac1{\|h\|}(I-uu^\mathsf T),
\tag{A.12.8}
$$

所以回到 $h$ 的梯度位于球面的切空间，与 $u$ 正交。实现为避免零范数而加入 $\epsilon$ 时，精确 Jacobian 会随所用归一化公式变化。

较小 $\tau$ 放大 logits，也在 (A.12.7) 中显式放大局部梯度尺度；它同时改变 softmax 权重，不能只把温度理解为一个独立学习率。

## A.12.4 Negatives 的统计含义

对固定 $x_i$，其他独立配对中的 $y_j$ 在理想 iid batch 下来自边缘分布 $p_Y$，并与 $x_i$ 独立，因此可作为 noise candidates。这个结论在以下情形会改变：

- batch 按类别、用户或时间分层而非 iid 抽样；
- 多个 $y_j$ 都是 $x_i$ 的有效描述，形成 false negatives；
- 同一实体或近重复样本同时出现；
- negatives 来自队列或旧编码器，产生时滞分布；
- 数据增强使同一源样本的多个 view 相关。

损失仍可计算，但“分母近似边缘分布”的理论解释必须按实际采样方案重写。false negative 也不表示代码出错，而是训练配对定义没有表达多重正确匹配。

## A.12.5 InfoNCE 互信息下界

下面陈述的是**总体期望损失**的结论，不是单个有限 batch 的无偏互信息估计。

**定理 A.12.1（InfoNCE 下界）** 假设 $I(X;Y)<\infty$，并取整数 $N\ge2$。令 $I$ 在 $\{1,\ldots,N\}$ 上均匀分布。先采样 $X\sim p_X$；条件于 $(X,I)$，令

$$
Y_I\sim p_{Y\mid X}(\cdot\mid X),
$$

其余 $Y_j$ 独立采自 $p_Y$。对任意正 critic $f(x,y)>0$，定义

$$
q_f(i\mid x,y_{1:N})
=\frac{f(x,y_i)}{\sum_{j=1}^Nf(x,y_j)},
$$

$$
L_N(f)
=\mathbb E[-\log q_f(I\mid X,Y_{1:N})].
\tag{A.12.9}
$$

若 $L_N(f)<\infty$，则

$$
\boxed{
I(X;Y)\ge\log N-L_N(f).}
\tag{A.12.10}
$$

**证明** 记 $Z=(X,Y_{1:N})$，上述生成分布为 $P_{I,Z}$。令 $Q_{I,Z}$ 是如下完全独立的参照分布：$I$ 仍均匀，$X\sim p_X$，且所有 $Y_j$ 独立采自 $p_Y$，并与 $(I,X)$ 独立。固定 $I=i$ 时，$P_{Z\mid I=i}$ 与 $Q_Z$ 的唯一区别是 $(X,Y_i)$ 在前者服从联合分布而在后者服从边缘乘积，故

$$
D_{\mathrm{KL}}(P_{Z\mid I=i}\|Q_Z)=I(X;Y).
$$

$P_I=Q_I$，所以 KL 的条件分解给出

$$
D_{\mathrm{KL}}(P_{I,Z}\|Q_{I,Z})=I(X;Y).
\tag{A.12.11}
$$

另一方面，先对 $Z$ 分解同一个 KL。由于 $Q_{I\mid Z}=Q_I$ 是均匀分布，

$$
\begin{aligned}
D_{\mathrm{KL}}(P_{I,Z}\|Q_{I,Z})
&=D_{\mathrm{KL}}(P_Z\|Q_Z)
+\mathbb E_{P_Z}
D_{\mathrm{KL}}(P_{I\mid Z}\|Q_I)\\
&=D_{\mathrm{KL}}(P_Z\|Q_Z)+I_P(I;Z)\\
&\ge I_P(I;Z).
\end{aligned}
\tag{A.12.12}
$$

因此 $I_P(I;Z)\le I(X;Y)$。对索引分类问题，交叉熵还可分解为

$$
L_N(f)
=H_P(I\mid Z)
+\mathbb E_{P_Z}
D_{\mathrm{KL}}(P_{I\mid Z}\|q_f(\cdot\mid Z))
\ge H_P(I\mid Z).
\tag{A.12.13}
$$

由于 $I$ 均匀，$H(I)=\log N$，于是

$$
\log N-L_N(f)
\le\log N-H_P(I\mid Z)
=I_P(I;Z).
$$

再结合 (A.12.12) 即得 (A.12.10)。在联合分布对边缘乘积绝对连续时，Bayes 最优索引分类器的 score 与 Radon--Nikodym 密度比 $dP_{Y\mid X=x}/dP_Y(y)$ 成比例；一般神经 critic 只给出相应变分下界。Poole 等人给出了这一精确多样本下界的系统推导。CPC 原附录的早期论证含有大 $N$ 近似，因此这里使用上面的精确 KL 分解。$\square$

该界还有三个直接边界：

1. 因为 $L_N(f)\ge0$，证书不超过 $\log N$；高互信息时有限 negatives 会使界饱和。
2. 实际 batch loss 是 (A.12.9) 的 Monte Carlo 量，单批的 $\log N-L$ 不必低于真实互信息。
3. 若 negatives 不来自边缘分布或与 $X$ 不独立，(A.12.10) 不能原样引用；需要针对新采样分布重新推导。

因此 InfoNCE 可作为有理论背景的表示学习目标，但低损失不等于“模型已经理解语义”，也不等于相似度可直接解释为事实概率。

## A.12.6 来源

- van den Oord, Li & Vinyals, [*Representation Learning with Contrastive Predictive Coding*](https://arxiv.org/abs/1807.03748), 2018。
- Poole et al., [*On Variational Bounds of Mutual Information*](https://proceedings.mlr.press/v97/poole19a.html), 2019。
- Radford et al., [*Learning Transferable Visual Models From Natural Language Supervision*](https://arxiv.org/abs/2103.00020), 2021。
