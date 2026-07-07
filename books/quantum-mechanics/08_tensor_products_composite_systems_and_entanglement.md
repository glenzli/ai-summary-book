# 第八章：张量积、复合系统与纠缠

## 本章目标

本章建立复合量子系统的 Hilbert 空间张量积，定义可分态、纠缠态、Schmidt 分解和约化态。

## 依赖前置知识

需要有限维 Hilbert 空间、矩阵、谱分解和纯态射线。

## 8.1 张量积系统

**公设 8.1（复合系统）.** 若两个系统的 Hilbert 空间分别为 $\mathcal H_A$ 与 $\mathcal H_B$，则复合系统的 Hilbert 空间为
$$
\mathcal H_A\otimes\mathcal H_B.
$$
若子系统态为 $\psi\in\mathcal H_A$ 与 $\phi\in\mathcal H_B$，复合纯态为 $\psi\otimes\phi$。

**定义 8.2.** 纯态 $\Psi\in\mathcal H_A\otimes\mathcal H_B$ 称为可分态，若存在 $\psi,\phi$ 使 $\Psi=\psi\otimes\phi$。否则称为纠缠态。

**例子 8.3.** Bell 态
$$
\Phi^+=\frac{1}{\sqrt2}(|00\rangle+|11\rangle)
$$
不是可分态。

**证明.** 若 $\Phi^+=(a|0\rangle+b|1\rangle)\otimes(c|0\rangle+d|1\rangle)$，则系数满足
$$
ac=1/\sqrt2,\quad bd=1/\sqrt2,\quad ad=0,\quad bc=0.
$$
由 $ac\ne0$ 得 $a,c\ne0$，由 $bd\ne0$ 得 $b,d\ne0$，这与 $ad=0$ 矛盾。$\square$

## 8.2 Schmidt 分解

**定理 8.4（有限维 Schmidt 分解）.** 对有限维双系统任意单位向量 $\Psi\in\mathcal H_A\otimes\mathcal H_B$，存在正交归一族 $(e_r)$、$(f_r)$ 和正数 $s_r$，使
$$
\Psi=\sum_{r=1}^k s_r e_r\otimes f_r,\qquad \sum_rs_r^2=1.
$$

**证明.** 固定正交基，把 $\Psi$ 的系数看作矩阵 $C=(c_{ij})$。奇异值分解给出
$$
C=U\Sigma V^*.
$$
令 $e_r$ 为 $U$ 的列向量，$f_r$ 为 $\overline V$ 的相应列向量，奇异值为 $s_r$，得到分解。范数公式来自 Hilbert-Schmidt 范数 $\|C\|_{HS}^2=\sum_rs_r^2$。$\square$

**推论 8.5.** 纯态可分当且仅当 Schmidt 秩为 $1$。

## 8.3 偏迹

**定义 8.6.** 对有限维复合系统上的算子 $T$，偏迹 $\operatorname{tr}_B T$ 是 $\mathcal H_A$ 上唯一满足
$$
\operatorname{tr}_{\mathcal H_A}(X\,\operatorname{tr}_B T)
=\operatorname{tr}_{\mathcal H_A\otimes\mathcal H_B}((X\otimes I)T)
$$
对所有 $X\in\mathcal B(\mathcal H_A)$ 成立的算子。

**命题 8.7.** 若 $\Psi=\sum_rs_r e_r\otimes f_r$ 是 Schmidt 分解，则
$$
\operatorname{tr}_B|\Psi\rangle\langle\Psi|
=\sum_rs_r^2 |e_r\rangle\langle e_r|.
$$

**证明.** 对任意 $X$，
$$
\operatorname{tr}((X\otimes I)|\Psi\rangle\langle\Psi|)
=\langle\Psi,(X\otimes I)\Psi\rangle
=\sum_{r,t}s_rs_t\langle e_r,Xe_t\rangle\langle f_r,f_t\rangle.
$$
正交性使 $r=t$，故等于
$$
\sum_rs_r^2\langle e_r,Xe_r\rangle
=\operatorname{tr}\left(X\sum_rs_r^2|e_r\rangle\langle e_r|\right).
$$
由偏迹定义得结论。$\square$

## 8.4 约化态纯度与纠缠判别

**命题 8.8.** 对有限维双系统纯态 $\Psi$，下列条件等价：

1. $\Psi$ 是可分态。
2. 约化态 $\rho_A=\operatorname{tr}_B|\Psi\rangle\langle\Psi|$ 是纯态。
3. $\operatorname{tr}(\rho_A^2)=1$。

**证明.** 取 Schmidt 分解
$$
\Psi=\sum_rs_re_r\otimes f_r.
$$
则
$$
\rho_A=\sum_rs_r^2|e_r\rangle\langle e_r|.
$$
由第十七章纯态判别，$\rho_A$ 纯当且仅当仅有一个非零本征值。因 $\sum_rs_r^2=1$，这等价于 Schmidt 秩为 $1$，也等价于 $\Psi$ 可分。第三条与第二条由 $\operatorname{tr}\rho_A^2=1$ 的判别等价。$\square$

**例子 8.9.** Bell 态的约化态为 $I/2$，故
$$
\operatorname{tr}(I/2)^2=1/2<1,
$$
因此它是纠缠态。

## 本章小结

复合系统由张量积描述。纠缠是复合纯态不能分解为单纯张量的性质。Schmidt 分解给出有限维双体纯态的标准形，偏迹给出子系统约化态。

## 练习

**练习 8.1.** 计算 Bell 态 $\Phi^+$ 的约化密度矩阵。

**练习 8.2.** 证明两个纯态的张量积仍为纯态投影的张量积。
