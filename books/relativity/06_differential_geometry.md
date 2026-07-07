# 第六章 流形、张量、联络与曲率

## 6.1 时空作为 Lorentz 流形

广义相对论把 Minkowski 时空推广为四维光滑流形 $M$，并在每一点的切空间 $T_pM$ 上给出 Lorentz 型内积。

**定义 6.1 (时空).** 一个相对论时空是二元组 $(M,g)$，其中 $M$ 是四维光滑流形，$g$ 是号差 $(-,+,+,+)$ 的光滑 Lorentz 度规。

局部坐标 $x^\mu$ 下，

$$
g=g_{\mu\nu}(x)\,dx^\mu\otimes dx^\nu.
$$

弯曲时空的间隔为

$$
ds^2=g_{\mu\nu}dx^\mu dx^\nu.
$$

这里 $g_{\mu\nu}$ 可以随位置变化，因果分类仍由 $ds^2$ 的符号决定。

## 6.2 张量和坐标变换

坐标变换 $x^\mu\mapsto x'^\alpha$ 下，反变矢量和协变矢量分量满足

$$
V'^\alpha=\frac{\partial x'^\alpha}{\partial x^\mu}V^\mu,
\qquad
\omega'_\alpha=\frac{\partial x^\mu}{\partial x'^\alpha}\omega_\mu.
$$

$(r,s)$ 型张量按相应 Jacobian 变换。张量方程的意义是：如果它在一个坐标系成立，就在所有坐标系成立。

度规给出升降指标：

$$
V_\mu=g_{\mu\nu}V^\nu,\qquad
V^\mu=g^{\mu\nu}V_\nu.
$$

其中 $g^{\mu\nu}$ 是 $g_{\mu\nu}$ 的逆矩阵。

## 6.3 协变导数

普通偏导数 $\partial_\mu V^\nu$ 不是张量，因为坐标变换会产生二阶导数项。需要引入联络。

**定义 6.2 (联络).** 协变导数 $\nabla$ 在坐标中由 Christoffel 符号 $\Gamma^\rho{}_{\mu\nu}$ 给出：

$$
\nabla_\mu V^\rho=\partial_\mu V^\rho+\Gamma^\rho{}_{\mu\nu}V^\nu,
$$

对协变矢量

$$
\nabla_\mu \omega_\nu=\partial_\mu\omega_\nu-\Gamma^\rho{}_{\mu\nu}\omega_\rho.
$$

对一般张量按 Leibniz 法则和每个指标加减 Christoffel 项。

## 6.4 Levi-Civita 联络

广义相对论使用由度规唯一确定的无挠、度规相容联络。

**定理 6.1 (Levi-Civita 联络).** 给定度规 $g$，存在唯一联络满足

$$
\nabla_\rho g_{\mu\nu}=0,\qquad
\Gamma^\rho{}_{\mu\nu}=\Gamma^\rho{}_{\nu\mu}.
$$

其坐标表达为

$$
\Gamma^\rho{}_{\mu\nu}
=\frac12 g^{\rho\sigma}
(\partial_\mu g_{\nu\sigma}
+\partial_\nu g_{\mu\sigma}
-\partial_\sigma g_{\mu\nu}).
$$

**证明.** 将 $\nabla_\rho g_{\mu\nu}=0$ 展开：

$$
\partial_\rho g_{\mu\nu}
=\Gamma^\sigma{}_{\rho\mu}g_{\sigma\nu}
+\Gamma^\sigma{}_{\rho\nu}g_{\mu\sigma}.
$$

对指标 $(\rho,\mu,\nu)$ 循环写三式，取前两式相加减第三式，并用 $\Gamma^\rho{}_{\mu\nu}$ 的下指标对称性，得到

$$
2\Gamma^\sigma{}_{\mu\nu}g_{\sigma\rho}
=\partial_\mu g_{\nu\rho}
+\partial_\nu g_{\mu\rho}
-\partial_\rho g_{\mu\nu}.
$$

再乘以 $g^{\rho\lambda}$ 得公式。唯一性随公式立即成立，存在性由直接代入验证。证毕。

## 6.5 平行移动和测地线

沿曲线 $x^\mu(\lambda)$ 的向量 $V^\mu(\lambda)$ 若满足

$$
\frac{dV^\mu}{d\lambda}
+\Gamma^\mu{}_{\rho\sigma}
\frac{dx^\rho}{d\lambda}V^\sigma=0,
$$

则称为沿曲线平行移动。

曲线切向量若沿自身平行移动，则得到测地线方程：

$$
\frac{d^2x^\mu}{d\lambda^2}
+\Gamma^\mu{}_{\rho\sigma}
\frac{dx^\rho}{d\lambda}
\frac{dx^\sigma}{d\lambda}=0.
$$

这将在下一章从作用量重新推导。

## 6.6 曲率

曲率测量平行移动绕小闭合回路后的失败。定义

$$
(\nabla_\mu\nabla_\nu-\nabla_\nu\nabla_\mu)V^\rho
=R^\rho{}_{\sigma\mu\nu}V^\sigma.
$$

坐标表达为

$$
R^\rho{}_{\sigma\mu\nu}
=
\partial_\mu\Gamma^\rho{}_{\nu\sigma}
-\partial_\nu\Gamma^\rho{}_{\mu\sigma}
+\Gamma^\rho{}_{\mu\lambda}\Gamma^\lambda{}_{\nu\sigma}
-\Gamma^\rho{}_{\nu\lambda}\Gamma^\lambda{}_{\mu\sigma}.
$$

Ricci 张量和标量曲率为

$$
R_{\mu\nu}=R^\rho{}_{\mu\rho\nu},
\qquad
R=g^{\mu\nu}R_{\mu\nu}.
$$

Riemann 张量满足基本对称性：

$$
R_{\rho\sigma\mu\nu}=-R_{\sigma\rho\mu\nu}
=-R_{\rho\sigma\nu\mu},
\qquad
R_{\rho\sigma\mu\nu}=R_{\mu\nu\rho\sigma}.
$$

还有第一 Bianchi 恒等式

$$
R_{\rho[\sigma\mu\nu]}=0.
$$

## 6.7 正规坐标

在任意点 $p\in M$，可以选择局部坐标使

$$
g_{\mu\nu}(p)=\eta_{\mu\nu},
\qquad
\partial_\rho g_{\mu\nu}(p)=0.
$$

于是

$$
\Gamma^\rho{}_{\mu\nu}(p)=0.
$$

但一般不能让二阶导数同时消失；曲率正是无法在一点邻域内完全消去的量。

## 习题

1. 对二维球面度规 $ds^2=d\theta^2+\sin^2\theta\,d\phi^2$ 计算非零 Christoffel 符号。
2. 验证 Levi-Civita 公式满足 $\nabla_\rho g_{\mu\nu}=0$。
3. 证明标量场 $\phi$ 的二阶协变导数满足 $\nabla_\mu\nabla_\nu\phi=\nabla_\nu\nabla_\mu\phi$。
4. 解释为什么 $\Gamma^\rho{}_{\mu\nu}$ 不是张量。
5. 在平直 Minkowski 坐标中计算 Riemann 张量。
