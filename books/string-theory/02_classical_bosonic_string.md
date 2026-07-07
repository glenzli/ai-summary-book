# 第二章：经典玻色弦、Nambu-Goto 与 Polyakov 作用量

## 本章目标

本章从弦世界面面积泛函出发，定义 Nambu-Goto 和 Polyakov 作用量，推导运动方程、边界条件与 Virasoro constraints，并说明 conformal gauge 下的开闭弦经典解。

## 依赖前置知识

需要第一章的作用量变分和 stress tensor。世界面与 target-space 归一化见 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md)。本章默认平坦 target metric $\eta_{\mu\nu}$，除非特别声明。

## 2.1 世界面和诱导度量

**定义 2.1（string worldsheet）.** 一条 classical string 在 target spacetime $M$ 中的历史是映射
$$
X:\Sigma\to M.
$$
若 $\sigma^a=(\tau,\sigma)$ 是世界面坐标，则诱导度量为
$$
\gamma_{ab}=\partial_aX^\mu\partial_bX^\nu g_{\mu\nu}(X).
$$

**定义 2.2（Nambu-Goto action）.** Nambu-Goto 作用量为
$$
S_{NG}=-T\int_\Sigma d^2\sigma\sqrt{-\det\gamma_{ab}},
\qquad
T=\frac1{2\pi\alpha'}.
$$

**命题 2.3（面积泛函与重参数化不变性）.** $S_{NG}$ 等于弦世界面 Lorentzian 面积乘以 $-T$，并在世界面重参数化下不变。

**证明.** $\sqrt{-\det\gamma}\,d^2\sigma$ 是诱导 Lorentzian 面积元。坐标变换 $\sigma\mapsto\sigma'$ 下，$\det\gamma$ 与 Jacobian 的平方反向变换，故面积元不变。$\square$

**注 2.4.** $S_{NG}$ 几何直观清楚，但 square root 使量子化不便。Polyakov 作用量引入独立世界面 metric，使 gauge symmetry 和 CFT 结构显式化。

## 2.2 Polyakov 作用量和对称性

**定义 2.5A（Polyakov action）.** Lorentzian Polyakov 作用量为
$$
S_P=-\frac{1}{4\pi\alpha'}\int d^2\sigma\sqrt{-h}\,
h^{ab}\partial_aX^\mu\partial_bX_\mu.
$$
这里 $h_{ab}$ 是独立 worldsheet metric。

**命题 2.5（Polyakov 与 Nambu-Goto 的经典等价）.** 在 classical level，消去 $h_{ab}$ 后 Polyakov 作用量等价于 Nambu-Goto 作用量。

**证明草图.** 对 $h^{ab}$ 变分得到
$$
\partial_aX\cdot\partial_bX
-\frac12h_{ab}h^{cd}\partial_cX\cdot\partial_dX=0.
$$
记 $\gamma_{ab}=\partial_aX\cdot\partial_bX$。该方程在二维中说明 $h_{ab}$ 与 $\gamma_{ab}$ Weyl 等价：
$$
h_{ab}=e^{2\omega}\gamma_{ab}.
$$
代回 $S_P$，Weyl factor 消去，得到 $S_{NG}$。$\square$

**命题 2.6（方程和约束）.** 在平坦 target 中，Polyakov 作用量的 $X^\mu$ 变分给出
$$
\frac1{\sqrt{-h}}\partial_a(\sqrt{-h}h^{ab}\partial_bX^\mu)=0.
$$
对 $h^{ab}$ 的变分给出约束
$$
T_{ab}=0.
$$

**证明.** $X^\mu$ 变分为
$$
\delta_XS_P
=\frac1{2\pi\alpha'}\int d^2\sigma\,
\partial_a(\sqrt{-h}h^{ab}\partial_bX_\mu)\delta X^\mu
+\delta S_{\partial\Sigma}.
$$
bulk 项给出 Laplace-Beltrami 方程。$h^{ab}$ 变分由第一章命题 1.4 给出 stress tensor 约束。$\square$

**命题 2.7（Polyakov action 的局部对称性）.** $S_P$ 具有：

1. worldsheet diffeomorphism invariance；
2. Weyl invariance $h_{ab}\mapsto e^{2\omega}h_{ab}$；
3. 平坦 target 中的 Poincare invariance。

**证明草图.** 前两项来自二维积分和 $\sqrt{-h}h^{ab}$ 在 Weyl 变换下不变；第三项来自 $\partial_aX^\mu\partial_bX_\mu$ 的 Lorentz invariance 与平移不变性。$\square$

## 2.3 Conformal gauge 和 Virasoro constraints

**定义 2.8（conformal gauge）.** Conformal gauge 是局部取
$$
h_{ab}=e^{2\omega}\eta_{ab}.
$$
在该 gauge 下运动方程变为
$$
(\partial_\tau^2-\partial_\sigma^2)X^\mu=0.
$$

**命题 2.9（Virasoro constraints）.** Conformal gauge 下的约束为
$$
(\partial_\tau X\pm\partial_\sigma X)^2=0.
$$

**证明.** 取 light-cone coordinates
$$
\sigma^\pm=\tau\pm\sigma,\qquad
\partial_\pm=\frac12(\partial_\tau\pm\partial_\sigma).
$$
在 conformal gauge 中，stress tensor 的独立分量为
$$
T_{++}=\partial_+X\cdot\partial_+X,\qquad
T_{--}=\partial_-X\cdot\partial_-X.
$$
约束 $T_{ab}=0$ 等价于 $T_{++}=T_{--}=0$，即
$$
(\partial_\tau X+\partial_\sigma X)^2=0,\qquad
(\partial_\tau X-\partial_\sigma X)^2=0.
$$
$\square$

**注 2.10（残余共形变换）.** Conformal gauge 固定后仍有残余变换
$$
\sigma^+\mapsto f(\sigma^+),\qquad
\sigma^-\mapsto g(\sigma^-).
$$
量子化后这些残余对称性由 Virasoro algebra 表示。

## 2.4 闭弦经典解

闭弦取
$$
\sigma\sim\sigma+2\pi.
$$
波动方程的一般解为左右移动之和：
$$
X^\mu(\tau,\sigma)=X_L^\mu(\tau+\sigma)+X_R^\mu(\tau-\sigma).
$$

**命题 2.11（闭弦模展开）.** 非紧平坦 target 中，闭弦解可写为
$$
X^\mu(\tau,\sigma)=x^\mu+\alpha'p^\mu\tau
+i\sqrt{\frac{\alpha'}2}\sum_{n\ne0}\frac1n
\left(
\alpha_n^\mu e^{-in(\tau-\sigma)}
+\tilde\alpha_n^\mu e^{-in(\tau+\sigma)}
\right).
$$

**证明草图.** 对周期变量 $\sigma$ 作 Fourier 展开，并把零模分解为 center-of-mass position 与 momentum。归一化选择是为了第四章 oscillator commutator 取标准形式。$\square$

## 2.5 开弦边界条件

开弦取
$$
\sigma\in[0,\pi].
$$
Polyakov 作用量在 conformal gauge 的边界变分为
$$
\delta S_{\partial\Sigma}
=-\frac1{2\pi\alpha'}\int d\tau\,
\delta X_\mu\,\partial_\sigma X^\mu\bigg|_{\sigma=0}^{\sigma=\pi}.
$$

**定义 2.12（Neumann 与 Dirichlet 条件）.** 开弦端点可满足 Neumann boundary condition
$$
\partial_\sigma X^\mu|_{\partial\Sigma}=0,
$$
或 Dirichlet boundary condition
$$
\delta X^\mu|_{\partial\Sigma}=0.
$$

**命题 2.13（开弦 Neumann 模展开）.** 若所有方向均取 Neumann 条件，则开弦解为
$$
X^\mu(\tau,\sigma)=x^\mu+2\alpha'p^\mu\tau
+i\sqrt{2\alpha'}\sum_{n\ne0}\frac{\alpha_n^\mu}{n}
e^{-in\tau}\cos n\sigma.
$$

**证明草图.** 波动方程给出左右移动解。Neumann 条件要求 $\partial_\sigma X^\mu=0$ 于 $\sigma=0,\pi$，因此空间依赖为 $\cos n\sigma$。零模归一化由总动量
$$
p^\mu=\int_0^\pi d\sigma\,P^\mu
$$
固定。$\square$

**注 2.14（D-brane 预告）.** Dirichlet 条件固定弦端点落在 target 中某个子流形上。量子理论中该子流形成为 D-brane 的几何模型。

## 本章小结

Nambu-Goto 作用量直接描述面积，Polyakov 作用量引入独立世界面 metric，使 gauge symmetry、Virasoro constraints 和量子化结构显式。Conformal gauge 把经典动力学化为二维波动方程，但必须同时保留 $T_{++}=T_{--}=0$ 约束。

## 练习

**练习 2.1.** 证明 Nambu-Goto 作用量在世界面重参数化下不变。

**练习 2.2.** 在 conformal gauge 下写出闭弦的一般经典解。

**练习 2.3.** 从开弦边界变分推出 Neumann 与 Dirichlet 条件。

