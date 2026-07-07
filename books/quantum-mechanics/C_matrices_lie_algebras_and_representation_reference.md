# 附录 C：矩阵、Lie 代数与群表示速查

## 本章目标

本附录汇总 Pauli 矩阵、交换子、指数映射和有限维表示的基本公式。

## 依赖前置知识

需要矩阵乘法、特征值和复指数。

## C.1 Pauli 矩阵

**定义 C.1.** Pauli 矩阵为
$$
\sigma_x=\begin{pmatrix}0&1\\1&0\end{pmatrix},\quad
\sigma_y=\begin{pmatrix}0&-i\\i&0\end{pmatrix},\quad
\sigma_z=\begin{pmatrix}1&0\\0&-1\end{pmatrix}.
$$

**命题 C.2.** 对任意 $a\in\mathbb R^3$，
$$
(a\cdot\sigma)^2=|a|^2I.
$$

**证明.** 使用恒等式
$$
\sigma_i\sigma_j=\delta_{ij}I+i\sum_k\epsilon_{ijk}\sigma_k.
$$
于是
$$
(a\cdot\sigma)^2=\sum_{i,j}a_ia_j\sigma_i\sigma_j
=|a|^2I+i\sum_{i,j,k}a_ia_j\epsilon_{ijk}\sigma_k.
$$
最后一项因 $a_ia_j$ 对称而 $\epsilon_{ijk}$ 反对称为零。$\square$

## C.2 指数公式

**命题 C.3.** 若 $n\in\mathbb R^3$ 且 $|n|=1$，则
$$
e^{-i\theta n\cdot\sigma/2}
=\cos\frac\theta2\,I-i\sin\frac\theta2\,n\cdot\sigma.
$$

**证明.** 由 $(n\cdot\sigma)^2=I$，指数级数的偶次项给出余弦，奇次项给出正弦。$\square$

## C.3 表示

**定义 C.4.** 群 $G$ 在 Hilbert 空间 $\mathcal H$ 上的酉表示是映射 $U:G\to\mathcal U(\mathcal H)$，满足 $U(gh)=U(g)U(h)$ 与 $U(e)=I$。

## 本章小结

Pauli 矩阵实现 $\mathfrak{su}(2)$ 的基本二维表示。指数公式解释自旋 $1/2$ 在旋转下的半角结构。

## 练习

**练习 C.1.** 证明 $\sigma_x\sigma_y=i\sigma_z$。

**练习 C.2.** 计算 $e^{-i\theta\sigma_z/2}$。

