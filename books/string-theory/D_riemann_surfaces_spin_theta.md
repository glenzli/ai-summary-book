# 附录 D：Riemann surfaces、spin structures 和 theta functions

## 目标

本附录提供高 genus 扰动论、GSO projection 和 modular invariance 所需的 Riemann surface 语言。

## D.1 Riemann surfaces

**定义 D.1.** Riemann surface 是一维复流形。Genus $g$ compact Riemann surface 的 holomorphic one-forms 空间维数为 $g$。

取 canonical homology basis
$$
a_i,b_i\in H_1(\Sigma_g,\mathbb Z),\qquad
a_i\cdot b_j=\delta_{ij},
$$
并取 normalized holomorphic one-forms $\omega_i$：
$$
\oint_{a_j}\omega_i=\delta_{ij}.
$$
Period matrix 定义为
$$
\Omega_{ij}=\oint_{b_j}\omega_i.
$$
它满足 $\Omega^T=\Omega$ 且 $\operatorname{Im}\Omega>0$。

**定义 D.1A（moduli dimension）.** 对 $2g-2+n>0$，带 $n$ 个 marked points 的 genus $g$ Riemann surfaces moduli space 有
$$
\dim_\mathbb C\mathcal M_{g,n}=3g-3+n.
$$
这是高 genus 振幅中 Beltrami differentials 与 $b$-ghost 插入数的来源。

## D.2 Spin structures

**定义 D.2.** Spin structure 是 square root $K^{1/2}$ of canonical bundle $K$。其 parity 由相应 Dirac operator 的零模数模 $2$ 给出。

**注 D.3.** Superstring genus expansion 必须对 spin structures 求和；GSO projection 与该求和的 modular invariance 紧密相关。

在 genus $g$ 上，spin structures 可由 characteristics
$$
\begin{bmatrix}\alpha\\ \beta\end{bmatrix},
\qquad \alpha,\beta\in \frac12\mathbb Z^g/\mathbb Z^g
$$
标记。Parity 为
$$
(-1)^{4\alpha\cdot\beta}.
$$
Even spin structures 数为 $2^{g-1}(2^g+1)$，odd spin structures 数为 $2^{g-1}(2^g-1)$。

## D.3 Theta functions

Theta function with characteristic 定义为
$$
\theta\begin{bmatrix}\alpha\\ \beta\end{bmatrix}(z|\Omega)
=
\sum_{n\in\mathbb Z^g}
\exp\left[
\pi i(n+\alpha)^T\Omega(n+\alpha)
+2\pi i(n+\alpha)^T(z+\beta)
\right].
$$

Genus one 时，四个 spin structures 对应四个 Jacobi theta functions。Superstring one-loop amplitude 中的 spin-structure sum 通过 theta identities 实现 spacetime supersymmetry 所需的 cancellation。

## D.4 Degeneration

当 Riemann surface 退化时，局部 sewing parameter $q$ 满足 $q\to0$，振幅边界出现形式
$$
\int \frac{d^2q}{|q|^2}q^{L_0-a}\bar q^{\tilde L_0-a}.
$$
该表达式说明 moduli space 边界与 target-space propagator pole 的关系。因子化一致性要求边界贡献分解为低阶振幅与中间弦态求和。

Superstring 中还必须处理 supermoduli 与 picture-changing insertions；本书把完整构造列为外部输入，只使用其因子化和 modular invariance 后果。
