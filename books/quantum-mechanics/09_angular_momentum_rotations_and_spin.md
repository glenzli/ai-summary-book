# 第九章：角动量、旋转与自旋

## 本章目标

本章建立角动量代数、升降算符、自旋 $1/2$ 表示和旋转对称性。

## 依赖前置知识

需要矩阵交换子、Hilbert 空间张量积和基本 Lie 代数语言。

## 9.1 角动量代数

**定义 9.1.** 三个自伴算子 $J_1,J_2,J_3$ 称为角动量算子，若在共同不变定义域上满足
$$
[J_i,J_j]=i\sum_k\epsilon_{ijk}J_k.
$$
定义
$$
J^2=J_1^2+J_2^2+J_3^2,\qquad J_\pm=J_1\pm iJ_2.
$$

**命题 9.2.** 有
$$
[J_3,J_\pm]=\pm J_\pm,\qquad [J^2,J_i]=0.
$$

**证明.** 第一式由
$$
[J_3,J_1]=iJ_2,\qquad [J_3,J_2]=-iJ_1
$$
直接相加得到。第二式以 $i=3$ 为例：
$$
[J_1^2,J_3]=J_1[J_1,J_3]+[J_1,J_3]J_1=-i(J_1J_2+J_2J_1),
$$
而
$$
[J_2^2,J_3]=J_2[J_2,J_3]+[J_2,J_3]J_2=i(J_2J_1+J_1J_2).
$$
两项相消，$[J_3^2,J_3]=0$。其余指标同理按循环置换得到。$\square$

## 9.2 自旋 $1/2$

**定义 9.3.** Pauli 矩阵为
$$
\sigma_x=\begin{pmatrix}0&1\\1&0\end{pmatrix},\quad
\sigma_y=\begin{pmatrix}0&-i\\i&0\end{pmatrix},\quad
\sigma_z=\begin{pmatrix}1&0\\0&-1\end{pmatrix}.
$$
自旋 $1/2$ 算子为 $S_i=\frac12\sigma_i$。

**命题 9.4.** $S_i$ 满足角动量对易关系。

**证明.** 直接计算 Pauli 矩阵满足
$$
[\sigma_i,\sigma_j]=2i\sum_k\epsilon_{ijk}\sigma_k.
$$
因此
$$
[S_i,S_j]=\frac14[\sigma_i,\sigma_j]
=i\sum_k\epsilon_{ijk}S_k.
$$
$\square$

## 9.3 总角动量

**定义 9.5.** 两个系统角动量分别为 $J^{(1)}$ 与 $J^{(2)}$ 时，总角动量定义为
$$
J_i=J_i^{(1)}\otimes I+I\otimes J_i^{(2)}.
$$

**命题 9.6.** 总角动量仍满足角动量对易关系。

**证明.** 因不同张量因子上的算子交换，
$$
[J_i,J_j]=[J_i^{(1)},J_j^{(1)}]\otimes I+I\otimes[J_i^{(2)},J_j^{(2)}]
=i\sum_k\epsilon_{ijk}J_k.
$$
$\square$

## 9.4 升降系数

**命题 9.7.** 若
$$
J^2|j,m\rangle=j(j+1)|j,m\rangle,\qquad
J_z|j,m\rangle=m|j,m\rangle,
$$
并且 $|j,m\rangle$ 已归一化，则
$$
J_\pm|j,m\rangle
=\sqrt{j(j+1)-m(m\pm1)}\,|j,m\pm1\rangle
$$
至多差一个相位。按标准相位约定取上式正根。

**证明.** 由 $[J_z,J_\pm]=\pm J_\pm$，$J_\pm|j,m\rangle$ 若非零，则是 $J_z$ 本征值 $m\pm1$ 的向量。又
$$
J_\mp J_\pm=J^2-J_z^2\mp J_z.
$$
取范数：
$$
\|J_\pm|j,m\rangle\|^2
=\langle j,m|J_\mp J_\pm|j,m\rangle
=j(j+1)-m(m\pm1).
$$
归一化即得系数。$\square$

**说明 9.8.** 条件 $j(j+1)-m(m\pm1)\ge0$ 强迫 $m$ 在有限链中终止，这正是 $m=-j,-j+1,\dots,j$ 的代数来源。

## 本章小结

角动量由 $\mathfrak{su}(2)$ 对易关系控制。自旋 $1/2$ 是二维不可约表示，总角动量来自张量积表示。升降算符组织 $J_3$ 本征值并导出量子数结构。

## 练习

**练习 9.1.** 计算 $\sigma_x\sigma_y$ 与 $\sigma_y\sigma_x$。

**练习 9.2.** 证明 $S^2=\frac34I$ 对自旋 $1/2$ 成立。
