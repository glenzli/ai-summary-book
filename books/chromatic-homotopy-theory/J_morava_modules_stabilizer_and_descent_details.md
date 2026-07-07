# 附录 J：Morava modules、stabilizer group 与 descent 细节

## J.1 Morava module

**定义 J.1.** 对谱 $X$，其第 $n$ 高度 Morava module 是
$$
(E_n)_*X=\pi_*(E_n\otimes X),
$$
连同来自 $\mathbb G_n$ 作用的连续 semilinear action。这里 semilinear 表示 $\mathbb G_n$ 同时作用在系数环 $(E_n)_*$ 和模上。

**警告 J.2.** 忘记 semilinear action 会丢失 descent 所需数据。$(E_n)_*X$ 作为普通 graded module 不能决定 $K(n)$-local homotopy type。

**定义 J.3.** 若 $M$ 是 $(E_n)_*$-module 且 $\mathbb G_n$ 作用满足
$$
g(a m)=g(a)g(m),
$$
则称 $M$ 是 Morava module。

## J.2 拓扑与连续性

**定义 J.4.** $(E_n)_0=W(k)[[u_1,\ldots,u_{n-1}]]$ 带 $\mathfrak m=(p,u_1,\ldots,u_{n-1})$-adic topology。$(E_n)_*$ 带由 $(E_n)_0$ 给出的拓扑和 $u$-periodic grading。

**定义 J.5.** $\mathbb G_n$ 是 profinite group。Morava module $M$ 的 action 称为连续，若作用映射
$$
\mathbb G_n\times M\to M
$$
在指定拓扑下连续。离散、profinite 和 completed module 情形需要分开指定。

**警告 J.6.** 连续群上同调 $H_c^s(\mathbb G_n;M)$ 的 $M$ 必须带拓扑。不同拓扑给出不同 cochain complex。

## J.3 Devinatz-Hopkins 公式

**外部输入 J.7.** 对有限谱 $X$，有等价
$$
L_{K(n)}X\simeq (E_n\otimes X)^{h\mathbb G_n}.
$$
对一般谱的版本需要更精细的连续 action 和 completed smash product 模型。

**命题 J.8.** 若 J.7 对 $X$ 成立，则 Morava descent spectral sequence 的目标为 $\pi_*L_{K(n)}X$。

**证明.** 对连续 $\mathbb G_n$-spectrum $E_n\otimes X$ 的 homotopy fixed point spectral sequence，形式为
$$
H_c^s(\mathbb G_n;\pi_t(E_n\otimes X))\Rightarrow \pi_{t-s}(E_n\otimes X)^{h\mathbb G_n}.
$$
由 J.7，右侧 abutment 等于 $\pi_{t-s}L_{K(n)}X$。证毕。

## J.4 Change of rings 与 descent 的关系

**外部输入 J.9.** Morava change-of-rings 将 height $n$ 局部的 $BP_*BP$-comodule Ext 与 $\mathbb G_n$ 的连续群上同调联系起来。这给出 Adams-Novikov $E_2$ 页的 height $n$ 层与 Morava descent 谱序列之间的桥梁。

**警告 J.10.** J.9 不是说 Adams-Novikov 谱序列和 Morava descent 谱序列逐项相同。二者的对象、completion、filtration 和收敛条件需要比较。

## J.5 Finite subgroup descent

**定义 J.11.** 若 $H\subset\mathbb G_n$ 是有限子群，定义 higher real K-theory 型对象
$$
E_n^{hH}.
$$

**例 J.12.** 高度 $1$ 中，有限子群和 Adams operations 给出 real/complex K-theory 型例子。高度 $2$ 中，supersingular elliptic curve 的 automorphism groups 给出 tmf 的局部 building blocks。

**警告 J.13.** 有限子群 $H$ 的 homotopy fixed points 比全 $\mathbb G_n$ 的 homotopy fixed points容易，但它不是 $K(n)$-local sphere，除非 $H=\mathbb G_n$ 在合适连续意义下。

## 本附录小结

Morava descent 的对象不是裸系数环，而是带连续 semilinear $\mathbb G_n$ 作用的 Morava module。第六章的公式、Adams-Novikov 的 height 层和 Picard descent 都依赖这一连续 descent 结构。
