# 第二章：有界算子、投影与谱分解

## 本章目标

本章研究有限维和有界算子情形下的可观测量模型：伴随、正规算子、正交投影、谱分解和期望值。

## 依赖前置知识

需要矩阵、特征值、正交投影和有限维内积空间。

## 2.1 有界算子与伴随

**定义 2.1.** Hilbert 空间 $\mathcal H$ 上的线性算子 $A:\mathcal H\to\mathcal H$ 称为有界，若存在 $C\ge0$ 使
$$
\|A\psi\|\le C\|\psi\|
$$
对所有 $\psi\in\mathcal H$ 成立。所有有界算子组成代数 $\mathcal B(\mathcal H)$。

**定义 2.2.** 有界算子 $A$ 的伴随 $A^*$ 是唯一满足
$$
\langle A^*\psi,\phi\rangle=\langle\psi,A\phi\rangle
$$
的有界算子。若 $A=A^*$，称 $A$ 为自伴有界算子。

**命题 2.3.** 若 $A,B\in\mathcal B(\mathcal H)$，则 $(AB)^*=B^*A^*$。

**证明.** 对任意 $\psi,\phi$，
$$
\langle (AB)^*\psi,\phi\rangle=\langle\psi,AB\phi\rangle
=\langle A^*\psi,B\phi\rangle
=\langle B^*A^*\psi,\phi\rangle.
$$
由内积非退化性得 $(AB)^*=B^*A^*$。$\square$

## 2.2 投影

**定义 2.4.** 算子 $P\in\mathcal B(\mathcal H)$ 称为正交投影，若
$$
P^2=P,\qquad P^*=P.
$$

**命题 2.5.** 若 $P$ 是正交投影，则 $\operatorname{Ran}P$ 与 $\ker P$ 正交，且
$$
\mathcal H=\operatorname{Ran}P\oplus\ker P.
$$

**证明.** 对 $x=Pu\in\operatorname{Ran}P$ 和 $y\in\ker P$，
$$
\langle x,y\rangle=\langle Pu,y\rangle=\langle u,Py\rangle=0.
$$
任意 $\psi$ 可写为 $\psi=P\psi+(I-P)\psi$，第一项在 $\operatorname{Ran}P$ 中，第二项由 $P(I-P)=0$ 在 $\ker P$ 中。若交中有 $z$，则 $z=Pz$ 且 $Pz=0$，故 $z=0$。$\square$

## 2.3 有限维谱分解

**定理 2.6（有限维谱定理）.** 设 $A$ 是有限维 Hilbert 空间上的自伴算子。则存在互异实数 $\lambda_1,\dots,\lambda_m$ 和两两正交的正交投影 $P_1,\dots,P_m$，使
$$
A=\sum_{r=1}^m\lambda_rP_r,\qquad \sum_{r=1}^mP_r=I.
$$

**证明.** 复矩阵的 Schur 分解给出正交归一基，使 $A$ 为上三角矩阵。因 $A=A^*$，该上三角矩阵同时等于其共轭转置，故非对角元为零且对角元为实数。按相同特征值分组，令 $P_r$ 为对应特征子空间的正交投影，即得分解。$\square$

**定义 2.7.** 在纯态 $\psi$ 中，自伴有界算子 $A$ 的期望值定义为
$$
\langle A\rangle_\psi=\langle\psi,A\psi\rangle.
$$

**命题 2.8.** 若 $A=\sum_r\lambda_rP_r$，则在态 $\psi$ 中测得 $\lambda_r$ 的概率为
$$
\|P_r\psi\|^2,
$$
且期望值为 $\sum_r\lambda_r\|P_r\psi\|^2$。

**证明.** 由投影正交性，
$$
\langle\psi,A\psi\rangle=\sum_r\lambda_r\langle\psi,P_r\psi\rangle
=\sum_r\lambda_r\|P_r\psi\|^2.
$$
因为 $\sum_rP_r=I$，概率和为 $\sum_r\|P_r\psi\|^2=\|\psi\|^2=1$。$\square$

## 2.4 正算子与 effect

**定义 2.9.** 有界自伴算子 $A$ 称为正算子，记作 $A\ge0$，若
$$
\langle\psi,A\psi\rangle\ge0
$$
对所有 $\psi\in\mathcal H$ 成立。若 $0\le E\le I$，称 $E$ 为 effect。

**命题 2.10.** 正交投影 $P$ 是 effect。

**证明.** 对任意 $\psi$，
$$
\langle\psi,P\psi\rangle=\langle P\psi,P\psi\rangle=\|P\psi\|^2\ge0,
$$
故 $P\ge0$。同理
$$
\langle\psi,(I-P)\psi\rangle=\|(I-P)\psi\|^2\ge0,
$$
故 $P\le I$。$\square$

**说明 2.11.** 投影测量只使用特殊 effect，即满足 $P^2=P$ 的 effect。第十八章的 POVM 允许一般 effect，因此能描述非理想测量和带辅助系统的测量。

## 本章小结

有限维自伴算子可分解为实特征值乘以正交投影的和。这个分解同时给出可观测量的可能测量值、概率和期望值。有界算子情形保留同样的代数结构，但无限维一般谱需要第三章的谱测度。

## 练习

**练习 2.1.** 证明若 $P$ 是正交投影，则 $I-P$ 也是正交投影。

**练习 2.2.** 对 Pauli 矩阵 $\sigma_z=\begin{pmatrix}1&0\\0&-1\end{pmatrix}$ 写出谱投影。
