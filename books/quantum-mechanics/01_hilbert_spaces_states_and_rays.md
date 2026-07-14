# 第一章：Hilbert 空间、态与射线

在二能级系统中，列向量 $(1,0)^T$ 与 $e^{i\theta}(1,0)^T$ 的坐标不同，却给出完全相同的测量统计；向量 $(1,1)^T/\sqrt2$ 则与它们只有一半的转移概率。因而量子态既不能只是一个坐标表，也不能把向量空间的线性结构全部丢掉：叠加需要向量，物理纯态却要把整体相位识别为同一点。要同时容纳这两件事，最小的语言是复 Hilbert 空间中的射线。

本章从内积和完备性出发，先说明内积怎样控制归一化与重叠，再把单位向量按整体相位取商，并用秩一投影给出不依赖代表元的表示。所需计算只涉及复向量空间、正交分解和有限维矩阵；无限维例子 $L^2(\mathbb R^d)$ 会提醒我们，完备性与“函数按几乎处处相等取商”同样属于态空间的定义，而不是技术附注。

## 1.1 Hilbert 空间

**定义 1.1.** 一个复 Hilbert 空间（Hilbert space）是带内积 $\langle-,-\rangle$ 的复向量空间 $\mathcal H$，满足由范数
$$
\|\psi\|=\sqrt{\langle\psi,\psi\rangle}
$$
诱导的度量是完备的。本书约定内积对第二变量线性。

**例子 1.2.** $\mathbb C^n$ 带
$$
\langle z,w\rangle=\sum_{j=1}^n\overline{z_j}w_j
$$
是 Hilbert 空间。$L^2(\mathbb R^d)$ 中向量是平方可积函数的等价类，内积为
$$
\langle f,g\rangle=\int_{\mathbb R^d}\overline{f(x)}g(x)\,dx.
$$

**命题 1.3（Cauchy-Schwarz 不等式）.** 对任意 $\psi,\phi\in\mathcal H$，
$$
|\langle\psi,\phi\rangle|\le \|\psi\|\,\|\phi\|.
$$

**证明.** 若 $\psi=0$ 则成立。设 $\psi\ne0$，令
$$
\lambda=\frac{\langle\psi,\phi\rangle}{\|\psi\|^2}.
$$
由正性，
$$
0\le \|\phi-\lambda\psi\|^2
=\|\phi\|^2-\overline{\lambda}\langle\psi,\phi\rangle-\lambda\langle\phi,\psi\rangle+|\lambda|^2\|\psi\|^2.
$$
代入 $\lambda$ 得
$$
0\le \|\phi\|^2-\frac{|\langle\psi,\phi\rangle|^2}{\|\psi\|^2}.
$$
整理即得结论。$\square$

Cauchy--Schwarz 不等式保证两个归一化向量的内积模不超过 $1$，于是它正好可以成为概率幅的几何尺度。但向量的整体相位仍会改变内积本身，必须先检查取绝对值平方后是否真正只依赖物理态。

## 1.2 态矢量与射线

**定义 1.4.** 一个归一化态矢量是满足 $\|\psi\|=1$ 的向量。两个单位向量 $\psi,\phi$ 表示同一纯态，若存在 $\theta\in\mathbb R$ 使
$$
\phi=e^{i\theta}\psi.
$$
等价类 $[\psi]$ 称为射线（ray）。

**定义 1.5.** 两个纯态 $[\psi]$ 与 $[\phi]$ 的转移概率定义为
$$
p([\psi],[\phi])=|\langle\psi,\phi\rangle|^2.
$$

**命题 1.6.** 转移概率与代表元选择无关。

**证明.** 若 $\psi'=e^{i\alpha}\psi$、$\phi'=e^{i\beta}\phi$，则
$$
\langle\psi',\phi'\rangle=e^{i(\beta-\alpha)}\langle\psi,\phi\rangle.
$$
取绝对值平方后相位因子消失。$\square$

**例子 1.6A（二能级态的重叠）.** 在 $\mathbb C^2$ 中取
$$
\psi=\begin{pmatrix}1\\0\end{pmatrix},\qquad
\phi=\frac1{\sqrt2}\begin{pmatrix}1\\1\end{pmatrix},\qquad
\psi_\theta=e^{i\theta}\psi.
$$
则 $[\psi_\theta]=[\psi]$，并且
$$
p([\psi],[\phi])
=|\langle\psi,\phi\rangle|^2
=\left|\frac1{\sqrt2}\right|^2=\frac12.
$$
相位 $\theta$ 不进入结果，而改变 $\phi$ 的第二个分量会改变它与其他方向的重叠；这就是整体相位与相对相位的区别。

单个重叠只能比较两条射线。要同时读取一个态相对于一组互斥方向的全部振幅，需要把 Hilbert 空间分解到正交归一基上。

## 1.3 正交分解

**定义 1.7.** 向量族 $(e_j)_{j\in J}$ 称为正交归一族，若 $\langle e_j,e_k\rangle=\delta_{jk}$。若其闭线性张成为整个 $\mathcal H$，称为正交归一基。

**命题 1.8.** 若 $(e_j)_{j=1}^n$ 是有限维 Hilbert 空间的正交归一基，则
$$
\psi=\sum_{j=1}^n e_j\langle e_j,\psi\rangle,\qquad
\|\psi\|^2=\sum_{j=1}^n|\langle e_j,\psi\rangle|^2.
$$

**证明.** 令 $\eta=\psi-\sum_j e_j\langle e_j,\psi\rangle$。对每个 $k$，
$$
\langle e_k,\eta\rangle=\langle e_k,\psi\rangle-\sum_j\langle e_k,e_j\rangle\langle e_j,\psi\rangle=0.
$$
因基张成全空间，$\eta=0$。范数公式由正交性展开内积得到。$\square$

正交展开依赖代表向量，但由射线得到的全部概率不应依赖整体相位。把射线改写为秩一投影，可以在保留线性算子运算的同时自动消去这项冗余。

## 1.4 射线的投影表示

**定义 1.9.** 对单位向量 $\psi$，定义秩一投影
$$
P_\psi=|\psi\rangle\langle\psi|.
$$
它作用在 $\phi\in\mathcal H$ 上为
$$
P_\psi\phi=\psi\langle\psi,\phi\rangle.
$$

**命题 1.10.** 两个单位向量 $\psi,\phi$ 表示同一射线，当且仅当
$$
P_\psi=P_\phi.
$$

**证明.** 若 $\phi=e^{i\theta}\psi$，则
$$
|\phi\rangle\langle\phi|
=e^{i\theta}|\psi\rangle e^{-i\theta}\langle\psi|
=P_\psi.
$$
反过来若 $P_\psi=P_\phi$，则
$$
\psi=P_\psi\psi=P_\phi\psi=\phi\langle\phi,\psi\rangle.
$$
因 $\|\psi\|=\|\phi\|=1$，系数 $\langle\phi,\psi\rangle$ 的模为 $1$，故二者只差相位。$\square$

**说明 1.11.** 用秩一投影表示纯态可自动消去整体相位，并为第十七章的密度算子口径做准备。

至此，纯态的两种表示各自承担了清楚的任务：单位向量便于展开和演化，秩一投影则直接表示射线并消去整体相位。下一章把这些投影放入更一般的有界算子代数中；届时，可观测量的特征子空间会把这里的几何重叠变成一组可归一化的测量概率。

## 练习

**练习 1.1.** 证明若两个单位向量满足 $|\langle\psi,\phi\rangle|=1$，则它们表示同一射线。

**练习 1.2.** 在 $\mathbb C^2$ 中令 $\psi=(1,0)$，$\phi=(1,1)/\sqrt2$。计算转移概率。
