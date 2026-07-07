# 第十三章：Calabi-Yau 紧化、模空间和四维有效理论

## 本章目标

本章建立 Calabi-Yau compactification 的主线：

1. Ricci-flat Kahler geometry 怎样满足最低阶 string background equations；
2. $SU(3)$ holonomy 怎样保留四维 supersymmetry；
3. Hodge numbers 怎样计数 complex structure 与 Kahler moduli；
4. type II 与 heterotic 紧化怎样产生四维有效理论。

本章只展开 string compactification 所需的几何接口；Yau theorem、Hodge theory 和 vector bundle stability 的完整证明属于外部数学理论。

## 依赖前置知识

需要第十一章低能有效作用、附录 A 的复几何语言和附录 E 的 spinor convention。

## 13.1 Calabi-Yau manifolds

**定义 13.1（Calabi-Yau manifold）.** Calabi-Yau $n$-fold 是紧 Kahler 流形 $X$，满足
$$
c_1(X)=0.
$$
在物理紧化语境中，常进一步要求 holonomy 包含于 $\operatorname{SU}(n)$。等价地，在适当条件下，$X$ 存在 nowhere-vanishing holomorphic $n$-form
$$
\Omega\in H^0(X,K_X).
$$

**外部输入定理 13.2（Yau theorem）.** 若 $X$ 是紧 Kahler 流形且 $c_1(X)=0$，则每个 Kahler class 中存在唯一 Ricci-flat Kahler metric。

**命题 13.3（Ricci-flat metric 给出最低阶背景）.** 设 ten-dimensional background 为
$$
M^{3,1}\times X
$$
的直积，$X$ 为 Ricci-flat Calabi-Yau threefold，且
$$
H=0,\qquad \Phi=\text{constant}.
$$
则最低阶 NS-NS beta function 方程被满足。

**证明.** 第十一章给出的最低阶 metric beta function 为
$$
R_{\mu\nu}+2\nabla_\mu\nabla_\nu\Phi-\frac14H_{\mu\rho\sigma}H_\nu^{\ \rho\sigma}=0.
$$
在直积 metric、常 dilaton 和 $H=0$ 下，该方程化为 $R_{\mu\nu}=0$。Minkowski 因子 Ricci-flat，$X$ 的 Ricci-flatness 由 Yau theorem 提供，故方程成立。$\square$

## 13.2 Holonomy 与 supersymmetry

**命题 13.4（$SU(3)$ holonomy 与 covariantly constant spinor）.** Calabi-Yau threefold 的 $SU(3)$ holonomy 保留至少一个 covariantly constant internal spinor。十维 supersymmetry parameter 分解后，四维保留的 supersymmetry 由该 internal spinor 的数目决定。

**证明草图.** Riemannian spinor 在平行移动下按 holonomy group 变换。若 holonomy 包含于 $SU(3)\subset SO(6)$，则六维 spin representation 中存在 $SU(3)$ singlet。该 singlet 对应 covariantly constant spinor，满足内部 gravitino variation 为零。$\square$

**推论 13.5（type II on Calabi-Yau threefold）.** Type II string theory 在 Calabi-Yau threefold 上紧化，在无 flux 且保持四维 Poincare invariance 的最低阶背景中给出四维 $\mathcal N=2$ supersymmetry。

**证明草图.** Type II 十维理论有两份 supersymmetry。Calabi-Yau threefold 的 internal singlet spinor 对每份十维 supersymmetry 各保留四维一份 supercharge，合计为四维 $\mathcal N=2$。$\square$

## 13.3 Hodge numbers 与 moduli

**定义 13.6（Kahler 与 complex structure moduli）.** 对 Calabi-Yau threefold $X$：

1. Kahler moduli 的 infinitesimal deformations 由
   $$
   H^{1,1}(X)
   $$
   控制，维数为 $h^{1,1}(X)$。
2. Complex structure moduli 的 infinitesimal deformations 由
   $$
   H^{2,1}(X)
   $$
   控制，维数为 $h^{2,1}(X)$。

**命题 13.7（moduli counting）.** 在无 obstruction 的 Calabi-Yau threefold 情形，metric moduli 的实维数为
$$
2h^{2,1}(X)+h^{1,1}(X),
$$
其中 complex structure moduli 为 complex scalars，Kahler class moduli 为 real scalars；与 $B$-field 结合后，Kahler moduli 也复化。

**证明草图.** Complex structure deformation 由 Beltrami differentials 表示，并经 contraction with $\Omega$ 与 $H^{2,1}(X)$ 同构。Kahler deformations 由 harmonic $(1,1)$ forms 展开。NS-NS $B$-field 的 harmonic $(1,1)$ 分量与 Kahler class 组合成 complexified Kahler moduli。$\square$

**定义 13.8（complexified Kahler form）.** 记 Kahler form 为 $J$，则 complexified Kahler parameter 由
$$
B+iJ
$$
在 $H^2(X,\mathbb C)$ 中的展开给出。

## 13.4 四维有效理论

**命题 13.9（四维 Planck scale）.** 若十维 string-frame background 近似为直积且 dilaton 常数，则四维有效引力耦合满足
$$
\frac1{\kappa_4^2}
\sim
\frac{\operatorname{Vol}(X)}{\kappa_{10}^2}
$$
在相应 frame convention 下成立。

**证明草图.** 将十维 Einstein-Hilbert 项在紧空间 $X$ 上积分。零模 truncation 中四维 Ricci scalar 与内部坐标无关，因此内部积分只给出体积因子。$\square$

**命题 13.10（type II moduli multiplets）.** Type IIA on Calabi-Yau threefold 中，Kahler moduli 位于 vector multiplets，complex structure moduli 位于 hypermultiplets；type IIB 中二者角色交换。

**证明草图.** 该分配由 R-R potentials 在 harmonic forms 上的展开和 mirror symmetry 兼容性决定。完整 multiplet 构造需要四维 $\mathcal N=2$ supergravity 表示论。$\square$

## 13.5 Heterotic compactification 的接口

**定义 13.11（Hermitian Yang-Mills 条件）.** Heterotic compactification 还需在 $X$ 上选择 gauge bundle $V$。保持 supersymmetry 的最低阶条件包括
$$
F^{0,2}=F^{2,0}=0,\qquad
J^{mn}F_{mn}=0.
$$

**外部输入定理 13.12（Donaldson-Uhlenbeck-Yau）.** Holomorphic vector bundle admits a Hermitian Yang-Mills connection if and only if it is poly-stable with slope zero.

**注 13.13.** 该定理属于复微分几何。本书只用它说明 heterotic compactification 中 gauge bundle data 与 supersymmetry 条件之间的关系。

## 13.6 Flux 与 moduli stabilization 的边界

**注 13.14.** 无 flux 的 Calabi-Yau compactification 通常产生 massless moduli。Flux、brane instantons、gaugino condensation 和 orientifold planes 可产生 superpotential 或 D-terms，从而提升部分 moduli。系统讨论放在第十九章；本章只建立无 flux 几何紧化基线。

## 13.7 标准例子：quintic threefold

**定义 13.15（quintic）.** Quintic Calabi-Yau threefold 是
$$
X=\{[z_0:\cdots:z_4]\in\mathbb P^4\mid
P_5(z)=0\},
$$
其中 $P_5$ 是横截的五次齐次多项式。

**命题 13.16（quintic 的 Hodge numbers）.** 光滑 quintic threefold 满足
$$
h^{1,1}(X)=1,\qquad h^{2,1}(X)=101.
$$

**证明草图.** $h^{1,1}=1$ 来自 ambient projective hyperplane class 的限制。Complex structure deformations 由五次齐次多项式的系数计数，再商去整体缩放和 $\operatorname{PGL}(5)$ 坐标变换：
$$
\binom{5+4}{4}-1-24=126-1-24=101.
$$
严格证明需处理 Jacobian ring 与 deformation theory。$\square$

**例 13.17（Euler characteristic）.** Calabi-Yau threefold 的 Euler characteristic 为
$$
\chi(X)=2(h^{1,1}-h^{2,1}).
$$
Quintic 因此有
$$
\chi(X)=2(1-101)=-200.
$$

## 13.8 四维场的零模展开

**命题 13.18（harmonic forms 与 massless fields）.** 紧化中，高维 $p$-form gauge field 沿 $X$ 的 harmonic $q$-forms 展开会产生低维 $(p-q)$-form fields。非 harmonic modes 通常获得 Kaluza-Klein 质量。

**证明草图.** 在直积背景上，内部 Laplacian 的 eigenmodes 给出四维质量平方。Harmonic modes 是零 eigenvalue，因此对应 massless fields；非零 eigenvalue modes 的质量由内部尺度控制。$\square$

## 本章小结

Calabi-Yau compactification 的核心是 Ricci-flat Kahler geometry、$SU(3)$ holonomy 和 Hodge-theoretic moduli counting。它把十维 string backgrounds 降维为四维 supersymmetric effective theories，但不自动解决 vacuum selection 或 moduli stabilization。

## 练习

**练习 13.1.** 说明 $c_1(X)=0$ 与 canonical bundle 平凡之间的关系。

**练习 13.2.** 对 Calabi-Yau threefold，解释为什么 complex structure moduli 由 $H^{2,1}(X)$ 计数。

**练习 13.3.** 用五次齐次多项式计数推导 quintic 的 $h^{2,1}=101$。

