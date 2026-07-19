# 附录 A.7 Softmax、交叉熵与数值稳定性

本附录推导从 logits 到多类负对数似然的完整接口，并说明分类 batch、语言模型 token mask 与 attention mask 中容易混淆的边界。概率校准与世界不确定性属于卷三；softmax 归一化本身不保证校准。

## A.7.1 定义、平移不变性与 log-sum-exp

给定整数 $K\ge2$ 和 $K$ 类 logits $z\in\mathbb R^K$，定义

$$
\operatorname{LSE}(z)=\log\sum_{j=1}^Ke^{z_j},
$$

$$
p_i=\operatorname{softmax}(z)_i
=\exp(z_i-\operatorname{LSE}(z)).
\tag{A.7.1}
$$

于是 $p_i>0$ 且 $\sum_ip_i=1$。对任意常数 $c$，

$$
\operatorname{softmax}(z+c\mathbf 1)
=\operatorname{softmax}(z).
\tag{A.7.2}
$$

所以单个 logit 的绝对零点不可识别，只有差值决定概率比：

$$
\log\frac{p_i}{p_j}=z_i-z_j.
\tag{A.7.3}
$$

若目标 $y\in\mathbb R^K$ 满足 $y_i\ge0$、$\sum_i y_i=1$，交叉熵为

$$
\ell(z,y)=-\sum_i y_i\log p_i
=\operatorname{LSE}(z)-y^\mathsf Tz.
\tag{A.7.4}
$$

one-hot 目标类别为 $c$ 时，$\ell=\operatorname{LSE}(z)-z_c=-\log p_c$，即分类模型的负对数似然。软标签和 label smoothing 仍使用 (A.7.4)，但不能再简写为单个 $-\log p_c$。

## A.7.2 梯度、Jacobian 与 Hessian

由 (A.7.1)，

$$
\frac{\partial p_i}{\partial z_j}
=p_i(\mathbf 1\{i=j\}-p_j).
\tag{A.7.5}
$$

矩阵形式为

$$
J_{\mathrm{softmax}}(z)
=\operatorname{diag}(p)-pp^\mathsf T.
\tag{A.7.6}
$$

又因为 $\nabla_z\operatorname{LSE}(z)=p$，由 (A.7.4) 直接得到

$$
\boxed{\nabla_z\ell=p-y.}
\tag{A.7.7}
$$

不需要先形成 $\partial\ell/\partial p_i=-y_i/p_i$ 再与 Jacobian 相乘；融合公式更简洁，也避免对已经下溢为零的概率取对数。

Hessian 为

$$
\nabla_z^2\ell
=\operatorname{diag}(p)-pp^\mathsf T.
\tag{A.7.8}
$$

对任意 $v$，

$$
v^\mathsf T\nabla_z^2\ell\,v
=\sum_i p_iv_i^2-\left(\sum_i p_iv_i\right)^2
=\operatorname{Var}_{i\sim p}(v_i)\ge0.
$$

因此交叉熵对 logits 是凸的。由于 (A.7.2)，$\mathbf 1$ 是 Hessian 的零方向；当所有 $p_i>0$ 时，其零空间恰为 $\operatorname{span}\{\mathbf 1\}$。这不表示损失对产生 logits 的深层网络参数也是凸的。

## A.7.3 Batch、token mask 与归一化

语言模型通常有

$$
Z\in\mathbb R^{B\times T\times K},
\qquad
Y\in\mathbb R^{B\times T\times K},
$$

以及 loss mask $M\in\{0,1\}^{B\times T}$。设

$$
N_{\mathrm{eff}}=\sum_{b,t}M_{bt}>0.
$$

按有效 token 平均的损失是

$$
L=-\frac1{N_{\mathrm{eff}}}
\sum_{b,t}M_{bt}
\sum_{k=1}^KY_{btk}\log P_{btk}.
\tag{A.7.9}
$$

若每个有效位置的目标权重和为 $1$，则

$$
\frac{\partial L}{\partial Z_{btk}}
=\frac{M_{bt}}{N_{\mathrm{eff}}}
(P_{btk}-Y_{btk}).
\tag{A.7.10}
$$

按序列平均、按 token 平均和先对每条序列平均再对 batch 平均是不同目标；长度不同时，它们给样本的权重不同。padding、prompt token 和答案 token 是否进入 $M$ 必须由训练目标明确规定。

## A.7.4 数值稳定实现

直接计算 $e^{z_i}$ 可能上溢。取

$$
c=\max_i z_i,
$$

则恒等式

$$
\operatorname{LSE}(z)
=c+\log\sum_i e^{z_i-c}
\tag{A.7.11}
$$

使最大的指数输入为 $0$，其余不大于 $0$。稳定的 one-hot 损失直接计算

$$
\ell(z,c_{\mathrm{target}})
=c+\log\sum_i e^{z_i-c}-z_{c_{\mathrm{target}}}.
\tag{A.7.12}
$$

极小的 $e^{z_i-c}$ 仍可能下溢为零，但只要至少一个有限最大项存在，分母不会因此变成零；与先算 softmax、再对目标概率取 `log` 相比，(A.7.12) 避免了目标概率下溢后产生无穷损失的常见路径。低精度实现还需决定累加精度和 kernel 融合，数学等价式在浮点中不必逐位相同。

## A.7.5 Masked softmax 的定义域

attention mask 与 (A.7.9) 的 loss mask 作用位置不同。对一行分数 $s$ 和非空可见集合 $\mathcal V$，定义

$$
m=\max_{j\in\mathcal V}s_j,
$$

$$
a_j=
\begin{cases}
\dfrac{e^{s_j-m}}
{\sum_{r\in\mathcal V}e^{s_r-m}},&j\in\mathcal V,\\[8pt]
0,&j\notin\mathcal V.
\end{cases}
\tag{A.7.13}
$$

若 $\mathcal V=\varnothing$，最大值和归一化分母都无定义。把整行填成 $-\infty$ 后直接 softmax 常得到 `NaN`；实现必须保证每个有效 query 至少能看见一个 key，或为全遮蔽行定义额外语义。

在 softmax 后把某些概率乘零但不重新归一化，不等价于 (A.7.13)。使用有限大负数代替 $-\infty$ 也只在给定浮点格式下近似产生零概率；精确的可见性语义应由显式 mask 定义。

attention 的 masked softmax 反向公式见[附录 A.10](a.10_transformer_math.md)。

## A.7.6 来源

- Blanchard, Higham & Higham, [*Accurately Computing the Log-Sum-Exp and Softmax Functions*](https://doi.org/10.1093/imanum/draa038), 2021。
- Bridle, [*Probabilistic Interpretation of Feedforward Classification Network Outputs, with Relationships to Statistical Pattern Recognition*](https://doi.org/10.1007/978-3-642-76153-9_28), 1990。
