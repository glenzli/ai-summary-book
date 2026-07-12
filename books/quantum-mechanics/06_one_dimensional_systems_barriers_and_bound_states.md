# 第六章：一维系统、势垒与束缚态

## 本章目标

本章研究一维 Schrodinger 算子的基本模型：自由粒子、无限深方势阱、势垒散射和束缚态。

## 依赖前置知识

需要常微分方程、Fourier 变换直觉和第五章的 Hamiltonian 演化。

## 6.1 一维 Hamiltonian

**定义 6.1.** 取 $m>0$。在 $\mathcal H=L^2(\mathbb R)$ 上，自由
Hamiltonian 为
$$
H_0=-\frac{1}{2m}\frac{d^2}{dx^2},
\qquad \mathcal D(H_0)=H^2(\mathbb R).
$$
对几乎处处有限的可测实函数 $V:\mathbb R\to\mathbb R$，记同名乘法
算子
$$
(V\psi)(x)=V(x)\psi(x),
\qquad
\mathcal D(V)=\{\psi\in L^2(\mathbb R):V\psi\in L^2(\mathbb R)\}.
$$
一维粒子的形式 Hamiltonian 为
$$
H=-\frac{1}{2m}\frac{d^2}{dx^2}+V(x).
$$
实值条件保证该最大乘法算子自伴；它在任意稠密线性子定义域上的限制
都是对称算子。复值势一般不满足这一点。下面的外部输入给出一类严格的
自伴实现。

**外部输入定理 6.2（Kato-Rellich，QM-EXT-3）.** 设 $A$ 自伴，
$B$ 对称，并且 $\mathcal D(A)\subseteq\mathcal D(B)$。若存在
$0\le a<1$ 与 $b\ge0$，使每个 $\psi\in\mathcal D(A)$ 都满足
$$
\|B\psi\|\le a\|A\psi\|+b\|\psi\|,
$$
则 $A+B$ 在 $\mathcal D(A)$ 上自伴；若 $\mathscr C$ 是 $A$ 的算子
核心，则限制 $(A+B)|_{\mathscr C}$ 本质自伴。

应用到定义 6.1 时，还须假设
$H^2(\mathbb R)\subseteq\mathcal D(V)$，并对某个 $a<1,b\ge0$ 有
$$
\|V\psi\|_2\le a\|H_0\psi\|_2+b\|\psi\|_2,
\qquad \psi\in H^2(\mathbb R).
$$
由于 $V$ 实值，$B=V$ 是对称乘法算子，故
$H=H_0+V$ 在 $\mathcal D(H)=H^2(\mathbb R)$ 上自伴。特别地，
有界实势满足 $a=0$；本章的有限实值分段常数势属于这一情形。

## 6.2 无限深方势阱

**例子 6.3.** 区间 $(0,L)$ 上无限深方势阱的 Hamiltonian 为
$$
H=-\frac{1}{2m}\frac{d^2}{dx^2}
$$
并满足边界条件 $\psi(0)=\psi(L)=0$。本征方程 $H\psi=E\psi$ 给出
$$
\psi_n(x)=\sqrt{\frac2L}\sin\frac{n\pi x}{L},\qquad
E_n=\frac{n^2\pi^2}{2mL^2},\qquad n\ge1.
$$

**命题 6.4.** 函数 $\psi_n$ 构成 $L^2(0,L)$ 的正交归一族。

**证明.** 对 $n,k\ge1$，
$$
\int_0^L\sin\frac{n\pi x}{L}\sin\frac{k\pi x}{L}\,dx
=\begin{cases}
0,&n\ne k,\\
L/2,&n=k.
\end{cases}
$$
乘以归一化因子 $2/L$ 得 $\langle\psi_n,\psi_k\rangle=\delta_{nk}$。完备性是 Fourier sine 展开的标准定理，作为外部输入定理 QM-EXT-14 使用。$\square$

## 6.3 势垒与反射

**定义 6.5.** 对分段常数势 $V(x)$，定态方程
$$
-\frac1{2m}\psi''+V\psi=E\psi
$$
在每个常势区间化为二阶常系数方程。波函数和一阶导数在有限跳跃势处连续。

**命题 6.6.** 对有限势阶跃，波函数和导数连续。

**证明.** 在 $x_0$ 附近积分定态方程：
$$
-\frac1{2m}\int_{x_0-\epsilon}^{x_0+\epsilon}\psi''\,dx
+\int_{x_0-\epsilon}^{x_0+\epsilon}(V-E)\psi\,dx=0.
$$
若 $V$ 有界且 $\psi$ 局部有界，第二项随 $\epsilon\to0$ 趋于 $0$，故
$$
\psi'(x_0+)-\psi'(x_0-)=0.
$$
$\psi$ 若有跳跃，则二阶导数含 delta 型奇性，与方程右侧局部有界性不相容。故 $\psi$ 连续。$\square$

## 6.4 反射率与透射率

**定义 6.7.** 对定态波
$$
\psi(x)=Ae^{ikx}+Be^{-ikx}
$$
在常势区域中的概率流为
$$
j=\frac{k}{m}(|A|^2-|B|^2).
$$
若左侧入射振幅为 $1$、反射振幅为 $r$，右侧透射振幅为 $t$ 且波数为 $q>0$，定义
$$
R=|r|^2,\qquad T=\frac{q}{k}|t|^2.
$$

**命题 6.8.** 对实值分段常数势的一维定态散射，若左右两端都是传播波，则
$$
R+T=1.
$$

**证明.** 定态方程的 Wronskian
$$
W=\overline\psi\,\psi'-\overline{\psi'}\,\psi
$$
在实势区域满足 $W'=0$，因为 $\psi''=2m(V-E)\psi$ 且 $V-E$ 实值。概率流
$$
j=\frac1{2mi}(\overline\psi\,\psi'-\overline{\psi'}\,\psi)
$$
因此在全线常数。左端流为 $(k/m)(1-|r|^2)$，右端流为 $(q/m)|t|^2$。二者相等即 $1-R=T$。$\square$

**说明 6.9.** 若 $E<V$ 的区域出现指数衰减解，该区域本身不携带传播流；隧穿概率由势垒另一侧的传播波流量定义，而不是由势垒内部的指数函数模平方直接定义。

## 本章小结

一维量子力学把谱问题化为二阶微分方程和边界条件。无限深方势阱展示离散谱；势垒问题展示反射、透射和边界匹配。自伴性与谱完备性需要泛函分析定理支撑。

## 练习

**练习 6.1.** 验证无限深方势阱的 $\psi_n$ 满足边界条件和本征方程。

**练习 6.2.** 设波函数在势阶跃处满足连续匹配。写出入射、反射、透射振幅的线性方程组。
