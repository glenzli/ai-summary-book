# 第二十二章：中心势、氢原子与球谐函数

## 本章目标

本章补齐非相对论量子力学的核心模型：三维中心势、角动量分离变量、球谐函数、径向方程和氢原子能级。严格谱完备性仍作为外部输入，正文给出可检查的代数与微分方程推导。

## 依赖前置知识

需要角动量、Schrodinger 方程、Fourier/分布基础和常微分方程。

## 22.1 中心势与角动量守恒

**定义 22.1.** 三维中心势 Hamiltonian 是
$$
H=-\frac1{2m}\Delta+V(r),\qquad r=|x|,
$$
其中 $V$ 只依赖径向变量。轨道角动量为
$$
L=x\times P,\qquad P=-i\nabla.
$$

**命题 22.2.** 在光滑紧支撑函数的共同定义域上，中心势 Hamiltonian 与 $L^2$、$L_z$ 对易。

**证明.** Laplacian 在旋转下不变，故与旋转生成元 $L_i$ 对易。可直接按分量验证：
$$
L_i=-i\sum_{j,k}\epsilon_{ijk}x_j\partial_k.
$$
因 $\Delta$ 与所有旋转生成元交换，$[-\Delta,L_i]=0$。又 $V(r)$ 为径向乘法算子。沿旋转生成元求导径向函数为零：
$$
L_i(V(r)f)=V(r)L_if-i\sum_{j,k}\epsilon_{ijk}x_j(\partial_kV(r))f.
$$
这里 $\partial_kV(r)=V'(r)x_k/r$，而 $\sum_{j,k}\epsilon_{ijk}x_jx_k=0$，故 $[V(r),L_i]=0$。于是 $[H,L_i]=0$，从而 $[H,L^2]=0$ 且 $[H,L_z]=0$。$\square$

## 22.2 球谐函数

**定义 22.3.** 球谐函数 $Y_\ell^m$ 是单位球面 $S^2$ 上同时满足
$$
L^2Y_\ell^m=\ell(\ell+1)Y_\ell^m,\qquad
L_zY_\ell^m=mY_\ell^m
$$
的归一化函数，其中 $\ell=0,1,2,\dots$，$m=-\ell,\dots,\ell$。

**外部输入定理 22.4（球谐完备性，QM-EXT-10）.** 族 $\{Y_\ell^m\}$ 构成 $L^2(S^2)$ 的正交归一基。

**命题 22.5.** 若 $\psi(r,\Omega)=R(r)Y_\ell^m(\Omega)$，则
$$
\Delta\psi=
\left(\frac1{r^2}\frac d{dr}r^2\frac{dR}{dr}
-\frac{\ell(\ell+1)}{r^2}R\right)Y_\ell^m.
$$

**证明.** 球坐标中 Laplacian 分解为
$$
\Delta=\frac1{r^2}\frac\partial{\partial r}r^2\frac\partial{\partial r}
-\frac{L^2}{r^2}.
$$
作用在 $R(r)Y_\ell^m(\Omega)$ 上时，径向部分只作用于 $R$，角向部分用 $L^2Y_\ell^m=\ell(\ell+1)Y_\ell^m$，得到公式。$\square$

## 22.3 径向方程

**定义 22.6.** 对中心势定态方程 $H\psi=E\psi$，令
$$
\psi(r,\Omega)=R(r)Y_\ell^m(\Omega),\qquad u(r)=rR(r).
$$
则径向方程为
$$
-\frac1{2m}u''(r)+\left(V(r)+\frac{\ell(\ell+1)}{2mr^2}\right)u(r)=Eu(r).
$$

**命题 22.7.** 上述径向方程由三维定态方程推出。

**证明.** 由命题 22.5，
$$
-\frac1{2m}\left(R''+\frac2rR'-\frac{\ell(\ell+1)}{r^2}R\right)+V R=ER.
$$
代入 $R=u/r$。直接计算
$$
R''+\frac2rR'=\frac{u''}{r}.
$$
乘以 $r$ 后得到
$$
-\frac1{2m}u''+\frac{\ell(\ell+1)}{2mr^2}u+Vu=Eu.
$$
$\square$

## 22.4 氢原子能级

**定义 22.8.** 氢型原子的 Coulomb Hamiltonian 为
$$
H=-\frac1{2\mu}\Delta-\frac{Z e^2}{r},
$$
其中 $\mu$ 为约化质量，且沿用序章的 $\hbar=1$ 约定。这里的 $e^2$ 表示库仑耦合常数；若使用 SI 单位，应把它替换为 $e^2/(4\pi\varepsilon_0)$，并恢复相应的 $\hbar$ 因子。

**公式 22.9（氢型束缚态能级）.** 束缚态能级为
$$
E_n=-\frac{\mu Z^2e^4}{2n^2},\qquad n=1,2,\dots.
$$
每个 $n$ 对应 $\ell=0,\dots,n-1$ 与 $m=-\ell,\dots,\ell$。

**说明 22.10.** 公式 22.9 的严格推导需要 Coulomb Hamiltonian 的自伴性、径向方程的边界条件和 Laguerre 多项式解的完备性。本书把自伴性与完备性列为外部输入；径向方程和量子数关系在本章内部给出。

## 本章小结

中心势 Hamiltonian 与角动量对易，因此可同时对角化 $H,L^2,L_z$。球谐函数承担角向自由度，径向方程包含离心势。氢原子能级的 $1/n^2$ 结构来自 Coulomb 势的特殊可解性。

## 练习

**练习 22.1.** 证明径向替换 $R=u/r$ 时有 $R''+2R'/r=u''/r$。

**练习 22.2.** 对 $\ell=0$ 写出中心势径向方程。

**练习 22.3.** 计算固定主量子数 $n$ 时忽略自旋的氢型束缚态简并度。
