# 第十五章：一圈运行的最小例子

若 UV 匹配在 $1\,\mathrm{TeV}$ 只生成 $C_2$，低能处通常不会仍然只有这一列非零。一圈反项可以把 $\mathcal O_2$ 混合到量子数相容的 $\mathcal O_1$，而累积的 $\log(\mu/\mu_{\rm match})$ 决定诱导项是否需要重求和。一个常系数的上三角二算符系统足以把这一机制完整算出：先由一次 Picard 迭代得到 leading log，再用矩阵指数求精确常系数解，并在简并本征值时取正确极限。这个模型同时说明反常维数矩阵的非对角元如何破坏“单系数”假设。把运行跨过电弱阈值时，还必须把 SMEFT 系数匹配到只含轻自由度的 LEFT，再继续演化到 flavor、EDM 或低能精密实验的尺度。

## 15.1 单系数运行

**定义 15.1（一圈 leading-log 解）.** 固定初始重整化尺度 $\mu_0>0$。若
$$
\mu\frac{dC}{d\mu}=\frac{\gamma}{16\pi^2}C
$$
且 $\gamma$ 在积分区间内近似常数，则
$$
C(\mu)=C(\mu_0)
\left[
1+\frac{\gamma}{16\pi^2}\ln\frac{\mu}{\mu_0}
+O\!\left(
{\gamma^2\over(16\pi^2)^2}
\ln^2{\mu\over\mu_0}
\right)
\right].
$$

**证明（书内推导）.** 将方程写为 $dC/C=(\gamma/16\pi^2)d\ln\mu$，积分得 $C(\mu)=C(\mu_0)e^{\gamma L/(16\pi^2)}$，其中 $L=\ln(\mu/\mu_0)$；在固定 $L$ 下展开指数函数即得。该截断受控还要求 $|\gamma L|/(16\pi^2)\ll1$。若该组合不小，应使用指数解重求和，不能把显示的余项视为小量。$\square$

## 15.2 两算符混合

**命题 15.2（上三角混合）.** 若积分区间内的反常维数矩阵近似为常数，且
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
且 $C_1(\mu_0)=0$、$C_2(\mu_0)\ne0$，则在 leading-log 阶
$$
C_1(\mu)
=
\frac{\gamma_{12}}{16\pi^2}C_2(\mu_0)\ln\frac{\mu}{\mu_0}
+C_2(\mu_0)
O\!\left(
{\|\Gamma\|^2\over(16\pi^2)^2}
\ln^2{\mu\over\mu_0}
\right),
$$
其中 $\Gamma$ 是上式的 $2\times2$ 反常维数矩阵，$\|\cdot\|$ 可取任一 submultiplicative matrix norm。

**证明（书内推导）.** 记 $L=\ln(\mu/\mu_0)$。对积分方程作一次 Picard 迭代，在 $C_1$ 方程右侧代入零阶向量 $(0,C_2(\mu_0))^T$，得到显示的 leading-log 项；第二次迭代由 $\|\Gamma\|^2L^2/[2(16\pi^2)^2]$ 控制，更高次依次增加 $\|\Gamma\||L|/(16\pi^2)$。因此该式是固定 $L$ 的展开，并要求 $\|\Gamma\||L|/(16\pi^2)\ll1$；大对数情形应使用第 15.3 节的矩阵指数。$\square$

**物理含义 15.3.** 即使 UV 匹配只产生一个 Wilson 系数，低能也会因 RG 产生其他算符。单系数拟合不是 RG 稳定概念。

## 15.3 两算符系统的精确常系数解

令
$$
L=\log{\mu\over\mu_0},\qquad k=16\pi^2.
$$
若 $\gamma_{11}\ne\gamma_{22}$，常系数上三角系统的矩阵指数解为
$$
C_2(\mu)=
\exp\left({\gamma_{22}L\over k}\right)C_2(\mu_0),
$$
$$
C_1(\mu)=
\exp\left({\gamma_{11}L\over k}\right)C_1(\mu_0)
 +{\gamma_{12}\over\gamma_{22}-\gamma_{11}}
\left[
\exp\left({\gamma_{22}L\over k}\right)
-\exp\left({\gamma_{11}L\over k}\right)
\right]C_2(\mu_0).
$$
若 $\gamma_{11}=\gamma_{22}=\gamma$，则
$$
C_2(\mu)=e^{\gamma L/k}C_2(\mu_0),
\qquad
C_1(\mu)=e^{\gamma L/k}
\left[
C_1(\mu_0)+{\gamma_{12}L\over k}C_2(\mu_0)
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
\log{\mu\over\mu_0}
\right]
\binom{C_1(\mu_0)}{C_2(\mu_0)}.
$$
矩阵指数重求和了由这个常系数一圈矩阵生成的 $[(16\pi^2)^{-1}L]^n$ 项。它不包含两圈反常维数或耦合常数运行；若要求相应的 next-to-leading logarithmic 精度，还须把这些输入一并加入。

## 15.4 SMEFT 外部输入

**外部输入 15.4（完整 SMEFT 反常维数矩阵）.** 维数六 SMEFT 一圈 RGE 的完整矩阵由 Jenkins-Manohar-Trott、Alonso 等文献给出。本书只使用其结构性结论：规范、Yukawa 和 Higgs quartic 相互作用都会导致算符混合。

**例 15.5（dipole mixing 口径）.** Lepton dipole 算符
$$
\mathcal O_{eB},\quad \mathcal O_{eW}
$$
在电弱破缺后组合成电磁偶极矩结构，相关 Wilson 系数运行和混合进入 $\ell_i\to\ell_j\gamma$、$(g-2)_\ell$ 和 EDM 约束。精确矩阵元属于外部输入。

## 15.5 阈值与再匹配

当运行穿过电弱尺度时，SMEFT 不再是低能自由度的正确 EFT。尺度链为
$$
C_i^{\rm SMEFT}(\mu_0)
\xrightarrow{\rm RGE}
C_i^{\rm SMEFT}(m_W)
\xrightarrow{\rm matching}
C_a^{\rm LEFT}(m_W)
\xrightarrow{\rm RGE}
C_a^{\rm LEFT}(\mu_{\rm low}).
$$
这就是 flavor、EDM 和低能精密过程不能只报告高尺度 SMEFT Wilson 系数的原因。

## 15.6 跨阈值演化留下什么

$\mu_0$ 只是 Wilson 系数初值的重整化尺度，不是 $\Lambda_{\rm ref}$ 或 $M_{\rm gap}$。在固定算符基与方案中，矩阵指数把这个初值演化到阈值；越过电弱尺度后，匹配矩阵把 SMEFT 坐标改写为 LEFT 坐标，再由低能 RGE 继续演化。观测量依赖整条乘积而不是其中某个单独系数，因此任何单系数初值都必须允许在低尺度生成混合方向。

## 练习

**练习 15.1.** 解常数 $2\times2$ 对角反常维数矩阵的 RGE。

**练习 15.2.** 若 $C_2(1\,\mathrm{TeV})=1$、$\gamma_{12}=4$，估计运行到 $m_Z$ 时诱导的 $C_1$ leading-log 大小。

**练习 15.3.** 说明为什么 $b$ 物理中的 Wilson 系数通常需要 SMEFT-to-LEFT 或 WET 匹配。
