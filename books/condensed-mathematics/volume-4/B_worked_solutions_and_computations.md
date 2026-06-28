# 附录 B：练习解答与计算样板

## B.1 三元覆盖的 Čech 复形

设 $\{U_1,U_2,U_3\}$ 覆盖 $U$，$F$ 是 sheaf of abelian groups。记

$$
U_{ij}=U_i\times_UU_j,\qquad
U_{ijk}=U_i\times_UU_j\times_UU_k.
$$

则

$$
C^0=\prod_iF(U_i),
\qquad
C^1=\prod_{i,j}F(U_{ij}),
\qquad
C^2=\prod_{i,j,k}F(U_{ijk}).
$$

微分为

$$
(d^0s)_{ij}=s_j|_{U_{ij}}-s_i|_{U_{ij}},
$$

$$
(d^1t)_{ijk}
=t_{jk}|_{U_{ijk}}
-t_{ik}|_{U_{ijk}}
+t_{ij}|_{U_{ijk}}.
$$

计算得 $d^1d^0=0$，因为每个三重交上的三项相消：

$$
(s_k-s_j)-(s_k-s_i)+(s_j-s_i)=0.
$$

## B.2 可表 sheaf 中紧性和 Hausdorff 性的用途

命题 2.3.1 的证明中，紧性和 Hausdorff 性分别用在：

1. $\coprod_iS_i$ 紧：有限个紧空间的余并仍紧。
2. $S$ Hausdorff：紧空间到 Hausdorff 空间的连续映射把闭集送到闭集。
3. 连续满闭映射是商映射：这一步让 $f\circ q$ 连续推出 $f$ 连续。

若没有 Hausdorff 性，紧集未必闭，$q$ 未必是闭映射；若没有紧性，也不能保证连续像紧，从而不能按此证明商映射性质。

## B.3 有限集合自由对象的 Ext

设 $S$ 有限。则

$$
\mathbb Z[\underline S]\cong
\bigoplus_{s\in S}\mathbb Z[\underline{\ast}].
$$

点 $\ast$ 是极不连通 compact Hausdorff 空间，因此 $\mathbb Z[\underline{\ast}]$ 投射。有限直和的投射对象仍投射：若 $P,Q$ 投射，则

$$
\operatorname{Hom}(P\oplus Q,-)
\cong
\operatorname{Hom}(P,-)\times\operatorname{Hom}(Q,-)
$$

保持满射。于是 $\mathbb Z[\underline S]$ 投射。由命题 3.2.1，

$$
\operatorname{Ext}^i(\mathbb Z[\underline S],A)=0
\quad(i>0).
$$

## B.4 两项分解中的 Ext 元素

设

$$
0\to P_1\xrightarrow dP_0\to M\to0
$$

是投射分解。命题 3.3.1 给出

$$
\operatorname{Ext}^1(M,A)
\cong
\operatorname{Hom}(P_1,A)/d^\vee\operatorname{Hom}(P_0,A).
$$

给定 $\phi:P_1\to A$，可构造推出图

$$
\begin{array}{ccc}
P_1 & \longrightarrow & P_0\\
\downarrow \phi & & \downarrow\\
A & \longrightarrow & E
\end{array}
$$

得到短正合列

$$
0\to A\to E\to M\to0.
$$

若 $\phi$ 改变一个边界 $\psi\circ d$，则对应推出图给出的扩张同构。因此余核中的类对应扩张类。这是 $\operatorname{Ext}^1$ 分类扩张的具体版本。

## B.5 Tor 的两项计算

设

$$
0\to P_1\xrightarrow dP_0\to M\to0
$$

是 $R$-模投射分解。对 $N$ 张量后：

$$
P_1\otimes_RN\xrightarrow{d\otimes1}
P_0\otimes_RN\to0.
$$

于是

$$
\operatorname{Tor}_1^R(M,N)=
\ker(d\otimes1),
$$

$$
M\otimes_RN=
\operatorname{coker}(d\otimes1).
$$

若 $d\otimes1$ 仍单射，则 $\operatorname{Tor}_1^R(M,N)=0$。这给出“张量后是否保持单射”的可计算判别。

## B.6 普通张量积不保持无限乘积

考虑自然映射

$$
\left(\prod_{n\ge1}\mathbb Z\right)\otimes\mathbb Q
\to
\prod_{n\ge1}\mathbb Q.
$$

左侧任一元素可写成 $(a_n)_n/m$，其中 $m\ne0$ 是统一分母。因此它在右侧的像满足：所有坐标的分母都整除某个固定整数 $m$。

元素

$$
x=\left(1,\frac12,\frac13,\frac14,\ldots\right)
$$

不满足这个性质。若 $x=(a_n)/m$，则 $1/n=a_n/m$，即 $m=na_n$，所以每个 $n$ 都整除 $m$，不可能。故自然映射不是满射。

这就是第四章中强调 solid 张量积不是普通张量积的最小反例。

## B.7 Fréchet 逆极限与紧参数

设 $V=\varprojlim_nV_n$，每个 $V_n$ 是 Banach 空间。对紧 Hausdorff $S$，证明

$$
\operatorname{Cont}(S,V)
\cong
\varprojlim_n\operatorname{Cont}(S,V_n).
$$

证明如下。若 $f:S\to V$ 连续，则坐标复合 $f_n:S\to V_n$ 连续且相容。反过来，相容连续族 $(f_n)$ 给出唯一集合映射 $f:S\to V$。由于 $V$ 带逆极限拓扑，$f$ 连续当且仅当所有坐标 $f_n$ 连续。因此得到双射。线性结构逐点给出同构。

## B.8 pro-etale 与 condensed 的一句话判别

若一个陈述的对象形如 $U\to X$ 且 $U$ 是 scheme 或 adic space 的 pro-etale 对象，它属于 pro-etale 语境。

若一个陈述的对象形如 $S\in\mathbf{CHaus}$，并用

$$
F(S)
$$

测试某个对象的连续族，它属于 condensed 语境。

二者可以共享“投射测试对象”的证明模式，但不能共享同一个对象集合。
