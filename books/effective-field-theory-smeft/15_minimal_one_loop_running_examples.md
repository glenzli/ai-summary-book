# 第十五章：一圈运行的最小例子

## 本章目标

本章不重算 SMEFT 的完整反常维数矩阵，而是用最小例子说明 Wilson 系数如何运行、算符如何混合，以及为什么匹配尺度和实验尺度不能混为一谈。

## 依赖前置知识

需要第三章的 RGE 和第十三章的算符表。

## 15.1 单系数运行

**定义 15.1（一圈 leading-log 解）.** 若
$$
\mu\frac{dC}{d\mu}=\frac{\gamma}{16\pi^2}C
$$
且 $\gamma$ 在积分区间内近似常数，则
$$
C(\mu)=C(\Lambda)
\left[
1+\frac{\gamma}{16\pi^2}\ln\frac{\mu}{\Lambda}
+O\!\left(\frac{1}{(16\pi^2)^2}\right)
\right].
$$

**证明（书内推导）.** 将方程写为 $dC/C=(\gamma/16\pi^2)d\ln\mu$，积分并展开指数函数。$\square$

## 15.2 两算符混合

**命题 15.2（上三角混合）.** 若
$$
\mu\frac{d}{d\mu}
\begin{pmatrix}C_1\\ C_2\end{pmatrix}
=
\frac{1}{16\pi^2}
\begin{pmatrix}
\gamma_{11}&\gamma_{12}\\
0&\gamma_{22}
\end{pmatrix}
\begin{pmatrix}C_1\\ C_2\end{pmatrix},
$$
且 $C_1(\Lambda)=0$、$C_2(\Lambda)\ne0$，则在 leading-log 阶
$$
C_1(\mu)
=
\frac{\gamma_{12}}{16\pi^2}C_2(\Lambda)\ln\frac{\mu}{\Lambda}
+O((16\pi^2)^{-2}).
$$

**证明（书内推导）.** 在 $C_1$ 方程右侧用零阶 $C_2(\mu)=C_2(\Lambda)$ 代入，积分即可。$\square$

**物理含义 15.3.** 即使 UV 匹配只产生一个 Wilson 系数，低能也会因 RG 产生其他算符。单系数拟合不是 RG 稳定概念。

## 15.3 两算符系统的精确常系数解

令
$$
L=\log{\mu\over\Lambda},\qquad k=16\pi^2.
$$
若 $\gamma_{11}\ne\gamma_{22}$，常系数上三角系统的矩阵指数解为
$$
C_2(\mu)=
\exp\left({\gamma_{22}L\over k}\right)C_2(\Lambda),
$$
$$
C_1(\mu)=
\exp\left({\gamma_{11}L\over k}\right)C_1(\Lambda)
 +{\gamma_{12}\over\gamma_{22}-\gamma_{11}}
\left[
\exp\left({\gamma_{22}L\over k}\right)
-\exp\left({\gamma_{11}L\over k}\right)
\right]C_2(\Lambda).
$$
若 $\gamma_{11}=\gamma_{22}=\gamma$，则
$$
C_2(\mu)=e^{\gamma L/k}C_2(\Lambda),
\qquad
C_1(\mu)=e^{\gamma L/k}
\left[
C_1(\Lambda)+{\gamma_{12}L\over k}C_2(\Lambda)
\right].
$$
展开到 $1/k$ 的一阶即回到命题 15.2。等价地，完整解可写为
$$
\binom{C_1(\mu)}{C_2(\mu)}
=
\exp\left[
{1\over16\pi^2}
\begin{pmatrix}
\gamma_{11}&\gamma_{12}\\
0&\gamma_{22}
\end{pmatrix}
\log{\mu\over\Lambda}
\right]
\binom{C_1(\Lambda)}{C_2(\Lambda)}.
$$
矩阵指数定义了高阶 logs 的重求和。对本书一圈教材例子，只需保留第一阶矩阵指数展开。

## 15.4 SMEFT 外部输入

**外部输入 15.4（完整 SMEFT 反常维数矩阵）.** 维数六 SMEFT 一圈 RGE 的完整矩阵由 Jenkins-Manohar-Trott、Alonso 等文献给出。本书只使用其结构性结论：规范、Yukawa 和 Higgs quartic 相互作用都会导致算符混合。

**例 15.5（dipole mixing 口径）.** Lepton dipole 算符
$$
\mathcal O_{eB},\quad \mathcal O_{eW}
$$
在电弱破缺后组合成电磁偶极矩结构，相关 Wilson 系数运行和混合进入 $\ell_i\to\ell_j\gamma$、$(g-2)_\ell$ 和 EDM 约束。精确矩阵元属于外部输入。

## 15.5 阈值与再匹配

当运行穿过电弱尺度时，SMEFT 不再是低能自由度的正确 EFT。流程为
$$
C_i^{\rm SMEFT}(\Lambda)
\xrightarrow{\rm RGE}
C_i^{\rm SMEFT}(m_W)
\xrightarrow{\rm matching}
C_a^{\rm LEFT}(m_W)
\xrightarrow{\rm RGE}
C_a^{\rm LEFT}(\mu_{\rm low}).
$$
这就是 flavor、EDM 和低能精密过程不能只报告高尺度 SMEFT Wilson 系数的原因。

## 本章小结

RGE 让 Wilson 系数成为尺度依赖对象。SMEFT 分析必须说明系数定义尺度，并在必要时执行匹配-运行-再匹配。

## 练习

**练习 15.1.** 解常数 $2\times2$ 对角反常维数矩阵的 RGE。

**练习 15.2.** 若 $C_2(1\,\mathrm{TeV})=1$、$\gamma_{12}=4$，估计运行到 $m_Z$ 时诱导的 $C_1$ leading-log 大小。

**练习 15.3.** 说明为什么 $b$ 物理中的 Wilson 系数通常需要 SMEFT-to-LEFT 或 WET 匹配。
