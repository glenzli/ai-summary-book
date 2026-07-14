# 第八章：张量积、复合系统与纠缠

若两个二能级系统分别有两个基态，复合系统应有四个基向量
$|00\rangle,|01\rangle,|10\rangle,|11\rangle$，这已经迫使态空间采用张量积而不是直和。更深的变化是，并非每个四维向量都能写成两个局部向量的乘积。Bell 态 $(|00\rangle+|11\rangle)/\sqrt2$ 的整体态是纯的，但任一子系统的全部局部统计都像均匀混合；这种不能归因于某个局部纯态的相关性就是纠缠。

本章在有限维范围内把这一现象完全算清。先以复合系统公设定义可分态，再把双体态的系数矩阵作奇异值分解，得到 Schmidt 标准形。偏迹随后由“保留所有局部可观测量期望”这一性质定义，并直接算出约化算子的谱。最后用 Schmidt 系数在本章内部证明可分性、秩一约化态与纯度 $1$ 等价；证明只依赖前面已经建立的张量积、矩阵谱分解和迹，不借用后续密度算子章节的结论。

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

逐项比较系数可以排除某个具体乘积分解，Schmidt 分解则一次性给出任意双体纯态离乘积态有多少个独立方向。

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

**证明.** 若 Schmidt 秩为 $1$，则
$\Psi=s_1e_1\otimes f_1$，归一化给 $s_1=1$，故为乘积态。反之，若
$\Psi=e\otimes f$，其系数矩阵是两个列向量的外积，矩阵秩为 $1$；
Schmidt 秩等于该系数矩阵的秩，所以 Schmidt 秩为 $1$。$\square$

Schmidt 系数描述整体态，但局部实验只能作用在一个张量因子上。偏迹的定义正是要求一个单系统算子复现所有这类局部期望。

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

命题 8.7 同时给出 $\rho_A\ge0$ 与
$\operatorname{tr}\rho_A=\sum_rs_r^2=1$。在本章中称这样的约化算子为
纯的，专指它是秩一正交投影；这个条件可以直接从其本征值判断。

## 8.4 约化态纯度与纠缠判别

**命题 8.8.** 对有限维双系统纯态 $\Psi$，下列条件等价：

1. $\Psi$ 是可分态。
2. 约化态 $\rho_A=\operatorname{tr}_B|\Psi\rangle\langle\Psi|$ 是秩一正交投影。
3. $\operatorname{tr}(\rho_A^2)=1$。

**证明.** 取 Schmidt 分解
$$
\Psi=\sum_rs_re_r\otimes f_r.
$$
则
$$
\rho_A=\sum_rs_r^2|e_r\rangle\langle e_r|.
$$
令 $p_r=s_r^2$，则 $p_r>0$ 且 $\sum_rp_r=1$。由推论 8.5，
$\Psi$ 可分当且仅当只有一个 Schmidt 系数，也就是 $\rho_A$ 只有一个
非零本征值；归一化又迫使该本征值为 $1$，所以这等价于 $\rho_A$ 是
秩一正交投影。另一方面，
$$
1-\operatorname{tr}(\rho_A^2)
=\left(\sum_rp_r\right)^2-\sum_rp_r^2
=2\sum_{r<t}p_rp_t.
$$
右侧为非负数之和，等于零当且仅当至多一个 $p_r$ 非零。因此第三个
条件也与前两个条件等价。$\square$

**例子 8.9.** Bell 态的约化态为 $I/2$，故
$$
\operatorname{tr}(I/2)^2=1/2<1,
$$
因此它是纠缠态。

张量积不仅增加维数，还允许 Schmidt 秩大于一的纯态。对这类态，偏迹
得到的约化算子有多个非零本征值，局部纯度严格小于 $1$；Bell 态给出
极值 $1/2$。这些结论都来自本章的矩阵分解与偏迹计算。下一章研究旋转
自由度时，张量积还会承担另一项任务：把两个角动量组合成总角动量。

## 练习

**练习 8.1.** 计算 Bell 态 $\Phi^+$ 的约化密度矩阵。

**练习 8.2.** 证明两个纯态的张量积仍为纯态投影的张量积。
