# 附录 C：公式、约定与常用表

本附录固定公式常数和约定，避免正文中同一对象因不同物理文献规范而变形。

## 指标和度量

- Minkowski 度量：$\eta_{\mu\nu}=\operatorname{diag}(-,+,\ldots,+)$。
- $\partial^\mu=\eta^{\mu\nu}\partial_\nu$。
- d'Alembert 算子：
  $$
  \Box=\eta^{\mu\nu}\partial_\mu\partial_\nu=-\partial_t^2+\Delta.
  $$
- Levi-Civita 符号取 $\epsilon_{0123}=+1$；升降指标会引入度量符号。

## Fourier 变换

本书使用
$$
\widehat f(k)=\int_{\mathbb R^n}e^{-ikx}f(x)\,dx,\qquad
f(x)=\frac1{(2\pi)^n}\int_{\mathbb R^n}e^{ikx}\widehat f(k)\,dk.
$$

**命题 C.1 (`P`).** 若 $f,g\in L^1(\mathbb R^n)$，则 $f*g$ 几乎处处有定义、属于 $L^1$，且该约定下
$$
\widehat{f*g}(k)=\widehat f(k)\widehat g(k).
$$

**证明.** Tonelli 定理给出
$$
\int_{\mathbb R^n}\int_{\mathbb R^n}
|f(x-y)g(y)|\,dy\,dx
=\|f\|_1\|g\|_1<\infty.
$$
因此 $f*g$ 几乎处处存在且属于 $L^1$，并可用 Fubini 交换下式的积分：
$$
\widehat{f*g}(k)=\int e^{-ikx}\int f(x-y)g(y)\,dy\,dx.
$$
令 $u=x-y$；平移换元的 Jacobi 行列式为 $1$，于是上式等于
$$
\int e^{-ik\cdot u}f(u)\,du
\int e^{-ik\cdot y}g(y)\,dy
=\widehat f(k)\widehat g(k).
$$
$\square$

## Lie 代数公式

- $SU(2)$ 生成元：
  $$
  [J_i,J_j]=i\epsilon_{ijk}J_k.
  $$
- 升降算符：
  $$
  J_\pm=J_1\pm iJ_2,\qquad [J_3,J_\pm]=\pm J_\pm.
  $$
- Casimir：
  $$
  J^2=J_1^2+J_2^2+J_3^2,\qquad J^2|j,m\rangle=j(j+1)|j,m\rangle.
  $$

## 微分几何公式

- Cartan 公式：
  $$
  \mathcal L_X=d\iota_X+\iota_Xd.
  $$
- 曲率：
  $$
  F_A=dA+\frac12[A\wedge A].
  $$
- Bianchi 恒等式：
  $$
  D_AF_A=0.
  $$
- Yang-Mills 方程：
  $$
  D_A*F_A=0.
  $$

## 量子场论公式

- 自由标量场 Lagrange 密度：
  $$
  \mathcal L=-\frac12\partial_\mu\phi\partial^\mu\phi-\frac12m^2\phi^2.
  $$
- Klein-Gordon 方程：
  $$
  (\Box-m^2)\phi=0.
  $$
- 等时对易关系：
  $$
  [\phi(t,\mathbf x),\pi(t,\mathbf y)]=i\delta(\mathbf x-\mathbf y).
  $$
- Euclidean Gaussian 源公式：
  $$
  Z[J]=Z[0]\exp\left(\frac12\langle J,K^{-1}J\rangle\right)
  $$
  其中 $K$ 为正的二次型算符，行列式和逆在场论中需要正规化。
