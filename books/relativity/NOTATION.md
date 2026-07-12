# 符号、单位与指标约定

## 单位

默认采用自然单位：

$$
c=1.
$$

在需要恢复光速时，会显式写出 $c$。广义相对论中默认保留 $G$，Einstein 方程写为

$$
G_{\mu\nu}+\Lambda g_{\mu\nu}=8\pi G\,T_{\mu\nu}.
$$

第九、十章保留 $G$，其中 $M$ 是物理质量，几何长度为 $GM$；恢复
光速时 $GM$ 替换为 $GM/c^2$。第十五章单独取 $G=c=1$，其中
$M=GM_{\mathrm{phys}}/c^2$ 是几何化质量，
$a=J_{\mathrm{phys}}/(M_{\mathrm{phys}}c)$。

## 符号

- 希腊指标 $\mu,\nu,\rho,\sigma,\ldots$ 取 $0,1,2,3$。
- 拉丁空间指标 $i,j,k,\ldots$ 取 $1,2,3$。
- 重复一上一下指标默认求和。
- 坐标记为 $x^\mu=(t,x^1,x^2,x^3)$。
- 四维时空度规号差采用

$$
(-,+,+,+).
$$

平直 Minkowski 度规为

$$
\eta_{\mu\nu}=\operatorname{diag}(-1,1,1,1).
$$

## 导数

- 普通偏导数：$\partial_\mu=\partial/\partial x^\mu$。
- 协变导数：$\nabla_\mu$。
- 沿曲线参数 $\lambda$ 的导数：$\dot{x}^\mu=dx^\mu/d\lambda$。
- 平直背景 d'Alembert 算子：
  $\Box=\partial_\mu\partial^\mu=-\partial_t^2+\nabla^2$。
- 固有时 $\tau$ 下的四速度：

$$
u^\mu=\frac{dx^\mu}{d\tau}.
$$

## 曲率约定

Riemann 曲率张量定义为

$$
(\nabla_\mu\nabla_\nu-\nabla_\nu\nabla_\mu)V^\rho
=R^\rho{}_{\sigma\mu\nu}V^\sigma.
$$

Ricci 张量和标量曲率为

$$
R_{\mu\nu}=R^\rho{}_{\mu\rho\nu},\qquad R=g^{\mu\nu}R_{\mu\nu}.
$$

Einstein 张量为

$$
G_{\mu\nu}=R_{\mu\nu}-\frac12 Rg_{\mu\nu}.
$$

## 物理量

- 四动量：$p^\mu=m u^\mu$。
- 能动张量：$T^{\mu\nu}$。
- 电磁势：$A_\mu$。
- 电磁场强：

$$
F_{\mu\nu}=\partial_\mu A_\nu-\partial_\nu A_\mu.
$$

第四章另固定 $\epsilon^{0123}=+1$、$\epsilon^{123}=+1$，并采用

$$
F^{0i}=E^i,
\qquad
F^{ij}=\epsilon^{ijk}B_k,
\qquad
\partial_\mu F^{\nu\mu}=j^\nu.
$$

因此 Lorenz 规范下的势方程是 $\Box A^\nu=-j^\nu$。第二个电磁缩并
$\tilde F_{\mu\nu}F^{\mu\nu}$ 在 proper Lorentz 群下不变，但在反转
四维取向的变换下变号。

## 3+1 约定

未来单位法向量满足 $n^\mu n_\mu=-1$，并固定

$$
K_{ij}=-\frac12\mathcal L_n\gamma_{ij},
\qquad
j_i=-T_{\mu\nu}n^\mu\gamma^\nu{}_i.
$$

在这一配对下，无 $\Lambda$ 的动量约束为
$D_j(K^{ij}-\gamma^{ij}K)=8\pi G j^i$。

## 常见缩写

- SR: Special Relativity，狭义相对论。
- GR: General Relativity，广义相对论。
- EEP: Einstein Equivalence Principle，Einstein 等效原理。
- FLRW: Friedmann-Lemaitre-Robertson-Walker 度规。
- TT: transverse-traceless gauge，横向无迹规范。
