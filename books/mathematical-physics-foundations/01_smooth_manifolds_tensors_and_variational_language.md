# 第一章：流形、张量与变分语言

物理量通常不是坐标表中的数字，而是换坐标后仍有确定含义的对象。速度是切向量，动量是余切向量，电磁场强是二形式，作用量是一类泛函。若不先区分这些对象的坐标表达与内在定义，后续的 Hamilton 形式、规范场和量子场论都会被符号遮蔽。本章建立有限维流形、张量、微分形式和一阶变分的最小闭合语言。

## 1.1 流形和张量

**定义 1.1.** 一个 $n$ 维光滑流形 $M$ 是带有光滑坐标图册的 Hausdorff、第二可数拓扑空间，使得任意坐标变换为 $C^\infty$ 映射。

**定义 1.2.** 点 $p\in M$ 的切空间 $T_pM$ 是所有在 $p$ 处作用于芽的导子集合。余切空间 $T_p^*M$ 是其对偶空间。

**定义 1.3.** 一个 $(r,s)$ 型张量场是光滑截面
$$
T\in \Gamma\big((TM)^{\otimes r}\otimes (T^*M)^{\otimes s}\big).
$$

**命题 1.1 (`P`).** 若 $T=T^{i_1\cdots i_r}{}_{j_1\cdots j_s}\partial_{i_1}\otimes\cdots\otimes dx^{j_s}$ 是 $(r,s)$ 型张量，则换坐标 $x\mapsto y$ 下分量满足
$$
T^{a_1\cdots a_r}{}_{b_1\cdots b_s}
=
\frac{\partial y^{a_1}}{\partial x^{i_1}}\cdots
\frac{\partial y^{a_r}}{\partial x^{i_r}}
\frac{\partial x^{j_1}}{\partial y^{b_1}}\cdots
\frac{\partial x^{j_s}}{\partial y^{b_s}}
T^{i_1\cdots i_r}{}_{j_1\cdots j_s}.
$$

**证明.** 在两张坐标图的交集上，链式法则给出切基与余切基的变换
$$
\frac{\partial}{\partial x^i}=\frac{\partial y^a}{\partial x^i}\frac{\partial}{\partial y^a},
\qquad
dx^j=\frac{\partial x^j}{\partial y^b}dy^b.
$$
把它们逐个代入
$$
T=T^{i_1\cdots i_r}{}_{j_1\cdots j_s}
\partial_{x^{i_1}}\otimes\cdots\otimes\partial_{x^{i_r}}
\otimes dx^{j_1}\otimes\cdots\otimes dx^{j_s}.
$$
新坐标张量基线性无关，比较其系数，正好得到命题中的公式。该计算也说明公式在三重坐标交上满足复合律，因而与坐标图选择相容。$\square$

## 1.2 微分形式

**定义 1.4.** $k$-形式是 $\Omega^k(M)=\Gamma(\Lambda^kT^*M)$ 中的截面。外积 $\wedge$ 是余切张量代数的交错乘法。

**定义 1.5.** 外微分 $d:\Omega^k(M)\to\Omega^{k+1}(M)$ 在局部写作
$$
d\left(\sum_I a_I dx^{i_1}\wedge\cdots\wedge dx^{i_k}\right)
=\sum_I da_I\wedge dx^{i_1}\wedge\cdots\wedge dx^{i_k}.
$$

**命题 1.2 (`P`).** 外微分满足 $d^2=0$。

**证明.** 在一张坐标图内，任意 $k$-形式可写成 $\alpha=\sum_I a_I dx^I$。由于 $d(dx^i)=0$，分次 Leibniz 规则给出
$$
d^2\alpha=\sum_I d^2a_I\wedge dx^I.
$$
对每个系数函数 $a_I$，
$$
d^2a_I=\sum_{i,j}\partial_i\partial_ja_I\,dx^i\wedge dx^j
=\frac12\sum_{i,j}(\partial_i\partial_j-
\partial_j\partial_i)a_I\,dx^i\wedge dx^j=0,
$$
其中使用了光滑函数混合偏导可交换。故 $d^2\alpha$ 在每张坐标图上为零，从而全局为零。$\square$

**定义 1.6.** 给定向量场 $X$，内乘 $\iota_X:\Omega^k(M)\to\Omega^{k-1}(M)$ 定义为
$$
(\iota_X\alpha)(X_1,\ldots,X_{k-1})=\alpha(X,X_1,\ldots,X_{k-1}).
$$

**命题 1.3 (`P`, Cartan 公式).** 对任意向量场 $X$，
$$
\mathcal L_X=d\iota_X+\iota_Xd
$$
作为微分形式上的分次导子成立。

**证明.** 记 $D=d\iota_X+\iota_Xd$。由 $d$ 的分次 Leibniz 规则和 $\iota_X$ 的反导子规则直接展开可知，$D$ 是次数零导子。Lie 导数 $\mathcal L_X$ 也是次数零导子，所以只需在生成微分形式代数的函数与一形式上比较二者。

对 $f\in C^\infty(M)$，$Df=\iota_Xdf=df(X)=Xf=\mathcal L_Xf$。对一形式 $\alpha$ 和任意向量场 $Y$，外微分公式给出
$$
\begin{aligned}
(D\alpha)(Y)
&=(d\,\iota_X\alpha)(Y)+(\iota_Xd\alpha)(Y)\\
&=Y(\alpha(X))+d\alpha(X,Y)\\
&=Y(\alpha(X))+X(\alpha(Y))-Y(\alpha(X))-\alpha([X,Y])\\
&=X(\alpha(Y))-\alpha([X,Y])
=(\mathcal L_X\alpha)(Y).
\end{aligned}
$$
故两导子在生成元上一致，因而在全部微分形式上一致。$\square$

## 1.3 一阶变分

**定义 1.7.** 给定区间 $[a,b]$、构型空间 $Q$ 和 Lagrangian $L:TQ\to\mathbb R$，曲线 $q(t)$ 的作用量为
$$
S[q]=\int_a^b L(q(t),\dot q(t))\,dt.
$$

**命题 1.4 (`P`, Euler-Lagrange 方程).** 设 $L\in C^2(TQ)$，$q\in C^2([a,b],Q)$，且 $q([a,b])$ 位于一张坐标图中。若 $q$ 对所有 $C^1$ 固定端点变分都是临界曲线，则
$$
\frac{d}{dt}\frac{\partial L}{\partial \dot q^i}-\frac{\partial L}{\partial q^i}=0.
$$

**证明.** 在所给坐标图中取任意 $\xi\in C_c^1((a,b),\mathbb R^n)$。当 $|\epsilon|$ 足够小时，$q_\epsilon=q+\epsilon\xi$ 仍落在坐标图内。由 $L\in C^2$ 和区间紧性，可在积分号内对 $\epsilon$ 求导，得到
$$
\delta S=\int_a^b\left(\frac{\partial L}{\partial q^i}\xi^i+
\frac{\partial L}{\partial\dot q^i}\dot\xi^i\right)dt.
$$
分部积分并使用 $\xi(a)=\xi(b)=0$，得到
$$
0=\delta S[q](\xi)=\int_a^b F_i(t)\xi^i(t)\,dt,
\qquad
F_i=\frac{\partial L}{\partial q^i}-
\frac d{dt}\frac{\partial L}{\partial\dot q^i}.
$$
每个 $F_i$ 连续。若某点 $t_0$ 有 $F_i(t_0)>0$，连续性给出一个开区间 $I\ni t_0$，使 $F_i>0$ 于 $I$；取非负且不恒为零的光滑 bump 函数 $\xi^i$ 支撑于 $I$，其余分量为零，则积分严格为正，矛盾。$F_i(t_0)<0$ 同理不可能。因此所有 $F_i$ 恒为零，即得所述方程。若曲线需要多张坐标图，以上论证对支撑在每张图中的变分分别适用，结论因变分的内禀定义而相容。$\square$

**例 1.5（谐振子的变分方程）.** 取 $Q=\mathbb R$ 和
$$
L(q,\dot q)=\frac m2\dot q^2-\frac k2q^2,
\qquad m,k>0.
$$
此时 $\partial L/\partial\dot q=m\dot q$、$\partial L/\partial q=-kq$，命题 1.4 给出
$$
m\ddot q+kq=0.
$$
令 $\omega_0=\sqrt{k/m}$，全部实解为
$q(t)=A\cos(\omega_0t)+B\sin(\omega_0t)$。由初值 $q(0)=q_0$、$\dot q(0)=v_0$ 得 $A=q_0$、$B=v_0/\omega_0$。端点条件只用于消去变分边界项，并不是运动方程本身附带的初值。

## 练习

**练习 1.1.** 证明一形式 $\alpha=\alpha_i dx^i$ 的外微分为 $d\alpha=\partial_i\alpha_j dx^i\wedge dx^j$，并写成反对称分量形式。

**练习 1.2.** 对 $L=\frac12m g_{ij}(q)\dot q^i\dot q^j-V(q)$ 推出 Euler-Lagrange 方程，并解释 Christoffel 符号出现的位置。
