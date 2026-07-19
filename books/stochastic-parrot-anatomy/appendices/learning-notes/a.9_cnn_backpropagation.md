# 附录 A.9 CNN：互相关算子及其反向传播

卷一第二章给出 Conv2D 的前向索引。本附录把该索引视为线性算子，逐项推导对 bias、kernel 和 input 的梯度。深度学习库通常把前向算子称为 convolution，但实际计算的是不翻转 kernel 的互相关。

## A.9.1 前向定义与输出形状

先取 `groups=1`。令

$$
X\in\mathbb R^{B\times C_{\mathrm{in}}\times H\times W},
$$

$$
K\in\mathbb R^{C_{\mathrm{out}}\times C_{\mathrm{in}}\times K_h\times K_w},
\qquad
b\in\mathbb R^{C_{\mathrm{out}}}.
$$

步幅、padding 和 dilation 分别为
$(s_h,s_w)$、$(p_h,p_w)$、$(d_h,d_w)$，其中步幅与 dilation 是正整数，padding 是非负整数，并假定下式给出的输出尺寸为正。越界输入按零处理。前向互相关是

$$
Y_{n,o,i,j}
=b_o+
\sum_{c=0}^{C_{\mathrm{in}}-1}
\sum_{u=0}^{K_h-1}
\sum_{v=0}^{K_w-1}
K_{o,c,u,v}
X_{n,c,\,is_h-p_h+ud_h,\,js_w-p_w+vd_w}.
\tag{A.9.1}
$$

输出空间尺寸为

$$
H_{\mathrm{out}}
=\left\lfloor
\frac{H+2p_h-d_h(K_h-1)-1}{s_h}
\right\rfloor+1,
$$

$$
W_{\mathrm{out}}
=\left\lfloor
\frac{W+2p_w-d_w(K_w-1)-1}{s_w}
\right\rfloor+1.
\tag{A.9.2}
$$

若 padding 左右不对称，应在 (A.9.2) 中分别使用两侧 padding 总和；不能只套对称公式。

## A.9.2 三类梯度

设标量损失为 $L$，上游梯度

$$
\Delta_{n,o,i,j}
=\frac{\partial L}{\partial Y_{n,o,i,j}}
$$

与 $Y$ 同形。若卷积后接逐元素激活 $A=\phi(Y)$，这里的 $\Delta$ 应先由
$\Delta_Y=\Delta_A\odot\phi'(Y)$ 得到。

### Bias 梯度

同一个 $b_o$ 被全部样本和空间位置共享，因此

$$
\boxed{
\frac{\partial L}{\partial b_o}
=\sum_{n=0}^{B-1}
\sum_{i=0}^{H_{\mathrm{out}}-1}
\sum_{j=0}^{W_{\mathrm{out}}-1}
\Delta_{n,o,i,j}.}
\tag{A.9.3}
$$

### Kernel 梯度

由 (A.9.1)，

$$
\frac{\partial Y_{n,o,i,j}}
{\partial K_{o,c,u,v}}
=X_{n,c,\,is_h-p_h+ud_h,\,js_w-p_w+vd_w}.
$$

对所有使用同一 kernel 参数的位置求和：

$$
\boxed{
\frac{\partial L}{\partial K_{o,c,u,v}}
=\sum_{n,i,j}
\Delta_{n,o,i,j}
X_{n,c,\,is_h-p_h+ud_h,\,js_w-p_w+vd_w}.}
\tag{A.9.4}
$$

这说明 kernel 梯度是输入 patch 与输出误差之间的相关求和；batch 求均值还是求和由 $\Delta$ 中是否已含归一化决定。

### Input 梯度

固定输入位置 $(n,c,r,s)$。它只影响满足

$$
r=is_h-p_h+ud_h,
\qquad
s=js_w-p_w+vd_w
$$

的输出项。因此

$$
\boxed{
\frac{\partial L}{\partial X_{n,c,r,s}}
=\sum_{o,i,j,u,v}
\Delta_{n,o,i,j}K_{o,c,u,v}
\mathbf 1\!\left[
\begin{array}{l}
r=is_h-p_h+ud_h,\\
s=js_w-p_w+vd_w
\end{array}
\right].}
\tag{A.9.5}
$$

(A.9.5) 对 stride、dilation 和 padding 都有效，也明确给出了边界裁剪所需的索引条件。

## A.9.3 单位步幅下的翻转关系

为看清常见的 `rot180` 说法，取单样本、单输入/输出通道、$s_h=s_w=d_h=d_w=1$，并用以零为中心的有限 offset 表示 kernel：

$$
Y_{i,j}=\sum_{u,v}X_{i+u,j+v}K_{u,v}.
$$

由 (A.9.5)，

$$
\frac{\partial L}{\partial X_{a,b}}
=\sum_{u,v}\Delta_{a-u,b-v}K_{u,v}.
\tag{A.9.6}
$$

若数学卷积定义为

$$
(A*B)_{a,b}=\sum_{u,v}A_{a-u,b-v}B_{u,v},
$$

则 (A.9.6) 是

$$
\nabla_XL=\Delta*K.
$$

若仍用前向互相关算子 $\star$ 表示，则

$$
\nabla_XL
=\Delta\star\operatorname{rot180}(K),
\tag{A.9.7}
$$

并按前向 padding 选择 full 结果中的原输入区域。数组下标下的
$\operatorname{rot180}(K)_{u,v}=K_{K_h-1-u,K_w-1-v}$；只有在先固定边界和 offset 约定后，(A.9.7) 才不会产生一格偏移。

## A.9.4 Transposed convolution 是伴随，不是逆

固定 kernel 后，(A.9.1) 对 $X$ 是线性映射 $\mathcal C_K:X\mapsto Y-b$。按 Frobenius 内积，输入梯度满足

$$
\langle\Delta,\mathcal C_K(X)\rangle_F
=\langle\mathcal C_K^*(\Delta),X\rangle_F,
$$

所以

$$
\nabla_XL=\mathcal C_K^*(\Delta).
\tag{A.9.8}
$$

框架的 transposed convolution 实现的正是这种伴随索引：stride 大于一时把误差放回相应格点，dilation 决定 kernel offset，再处理 padding、`output_padding` 和裁剪。伴随一般不是逆；stride、边界与通道混合会丢失信息，`output_padding` 只解决输出形状歧义，不恢复被丢弃的输入。

## A.9.5 多通道、groups 与 depthwise 情形

(A.9.3)--(A.9.5) 已包含普通多输入/输出通道。若 `groups=G`，要求 $G$ 同时整除 $C_{\mathrm{in}}$ 和 $C_{\mathrm{out}}$，每个输出通道只连接对应组内的 $C_{\mathrm{in}}/G$ 个输入通道；上述对 $c$ 或 $o$ 的求和必须限制在连接集合内，kernel 形状变为

$$
C_{\mathrm{out}}\times(C_{\mathrm{in}}/G)\times K_h\times K_w.
$$

当 $G=C_{\mathrm{in}}=C_{\mathrm{out}}$ 时得到 depthwise convolution，每个输出通道只使用一个输入通道。若有 depth multiplier，则每个输入通道可对应多个输出通道，仍按连接集合求和。

## A.9.6 来源

- LeCun et al., [*Gradient-Based Learning Applied to Document Recognition*](https://doi.org/10.1109/5.726791), 1998。
- Dumoulin & Visin, [*A Guide to Convolution Arithmetic for Deep Learning*](https://arxiv.org/abs/1603.07285), 2016。
