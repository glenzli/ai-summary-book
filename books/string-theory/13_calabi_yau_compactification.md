# 第十三章：Calabi-Yau 紧化、模空间和四维有效理论

十维理论要产生四维低能物理，内部六维空间既不能任意弯曲，也不能破坏所有
supersymmetry。最低阶 Einstein 方程要求 Ricci-flat metric，平行 spinor 又把
holonomy 收紧到 $SU(3)$；在 compact Kahler threefold 上，这些条件把问题引向
Calabi--Yau geometry。几何随后不只是背景：harmonic forms 决定质量为零的
Kaluza--Klein modes，$H^{1,1}$ 与 $H^{2,1}$ 分别参数化 Kahler 和 complex-structure
deformations。以下使用第十一章有效作用、附录 A 的复几何与附录 E 的 spinors，
把这些对象转译为 type II/heterotic 四维场；Yau、Hodge 与 bundle stability 的大型
结果会以精确外部输入出现，而非压缩成形式证明。

## 13.1 Calabi-Yau manifolds

**定义 13.1（本书的 Calabi--Yau 口径）.** 本章的 Calabi--Yau $n$-fold 是
compact connected Kahler manifold $X$，其 canonical bundle
$K_X=\Lambda^{n,0}T^*X$ holomorphically trivial，并选定一个 nowhere-vanishing
holomorphic volume form
$$
\Omega\in H^0(X,K_X).
$$
因此 $c_1(X)=0$ 作为 integral class。部分文献只要求
$c_1(X)=0$ in $H^2(X,\mathbb R)$；若 canonical bundle 有 torsion，这个较弱条件
不自动给出全局 $\Omega$，故本书不把两种定义无条件视为等价。Ricci-flat metric 的
holonomy 由定义和 Yau theorem 推得包含于 $SU(n)$；“恰等于 $SU(n)$”还需排除
可约 holonomy、非平凡平坦因子等退化情形。

**外部输入定理 13.2（Calabi--Yau theorem）.** 若 $X$ 是 compact Kahler
manifold 且 $c_1(X)=0$ in $H^2(X,\mathbb R)$，则每个 Kahler class 中存在唯一
Ricci-flat Kahler metric。唯一性是在固定 complex structure 与 Kahler class 的
口径下成立，不表示不同 classes 给出同一个 metric。

**命题 13.3（Ricci-flat metric 给出最低阶背景）.** 设 ten-dimensional background 为
$$
M^{3,1}\times X
$$
的直积，$X$ 为 Ricci-flat Calabi-Yau threefold，且
$$
H=0,\qquad \Phi=\text{constant}.
$$
则显示到 $O(\alpha')$ 的最低阶 NS--NS beta function 方程被满足。

**证明.** 第十一章给出的最低阶 metric beta function 为
$$
R_{\mu\nu}+2\nabla_\mu\nabla_\nu\Phi-\frac14H_{\mu\rho\sigma}H_\nu^{\ \rho\sigma}=0.
$$
在直积 metric、常 dilaton 和 $H=0$ 下，该方程化为 $R_{\mu\nu}=0$。Minkowski
因子 Ricci-flat，$X$ 的 Ricci-flatness 由 Yau theorem 提供，故方程成立。这只证明
最低阶背景方程；它不把未经计算的 higher-$\alpha'$、worldsheet-instanton 或
string-loop corrections 宣称为零。$\square$

## 13.2 Holonomy 与 supersymmetry

**外部输入定理 13.4（holonomy principle）.** 对 spin Riemannian manifold，
parallel spinors 与 spin representation 中被 holonomy group 固定的向量一一对应。
特别地，六维 irreducible holonomy 恰为 $SU(3)$ 时，每个适当 chirality 有一个
complex invariant spinor；holonomy 是 $SU(3)$ 的真子群时可出现更多 parallel
spinors。

**使用边界.** Holonomy 与 parallel sections 的对应是 spin geometry 外部输入。
把该 spinor 代入十维 gravitino variation、从而计数未破缺 supercharges，是标准
低能 supergravity 推导；它只适用于给定 spin structure、无 flux 且忽略
$\alpha'$ corrections 的背景。

**推论 13.5（type II on irreducible Calabi--Yau threefold）.** Type II string
theory 在 compact threefold $X$ 上紧化；若 Ricci-flat metric 的 holonomy 恰为
$SU(3)$，且无 flux、orientifold、离散 quotient 或其他 supersymmetry-breaking
source，并保持四维 Poincare invariance，则最低阶背景给出四维 $\mathcal N=2$
supersymmetry。

**推导说明（标准物理口径）.** Type II 十维理论有两份 supersymmetry。
$SU(3)$-invariant internal spinor 对每份十维 supersymmetry 各保留一份四维 Weyl
supercharge，合计八个 real supercharges，即四维 $\mathcal N=2$。若 holonomy
严格小于 $SU(3)$，结论一般增强而非仍恰为 $\mathcal N=2$。$\square$

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

**外部输入定理 13.7（Calabi--Yau infinitesimal moduli）.** 对 compact
Calabi--Yau threefold，Bogomolov--Tian--Todorov unobstructedness 与 contraction by
$\Omega$ 给出局部 complex-structure moduli tangent space
$$
H^1(X,T_X)\cong H^{2,1}(X).
$$
固定 complex structure 后，Kahler classes 构成 $H^{1,1}(X,\mathbb R)$ 中的开锥。
因此在无额外 quotient 且位于 smooth moduli point 时，Ricci-flat metric moduli 的
局部实维数为
$$
2h^{2,1}(X)+h^{1,1}(X),
$$
其中 complex structure moduli 为 complex scalars，Kahler class moduli 为 real scalars；与 $B$-field 结合后，Kahler moduli 也复化。

**使用边界.** Unobstructedness、Hodge decomposition 与 Ricci-flat metric 对
Kahler class 的光滑依赖是外部几何输入。有限维计数只是
$2\dim_\mathbb C H^{2,1}+\dim_\mathbb R H^{1,1}$。NS--NS $B$-field 的 harmonic
$(1,1)$ 分量在局部场坐标中与 Kahler class 组合；large gauge transformations、
worldsheet instantons 和 moduli-space singularities 会给出全局 identifications，
所以 $B+iJ$ 只是一张局部坐标图。

**定义 13.8（complexified Kahler form）.** 记 Kahler form 为 $J$，则 complexified Kahler parameter 由
$$
B+iJ
$$
在 $H^2(X,\mathbb C)$ 中的展开给出。

## 13.4 四维有效理论

**命题 13.9（无 warp 直积的四维 Planck coefficient）.** 若十维 string-frame
background 为无 warp 直积，dilaton 为常数 $\Phi_0$，并只保留四维 metric zero
mode，则
$$
\frac1{\kappa_{4,S}^2}
=\frac{e^{-2\Phi_0}\operatorname{Vol}_S(X)}{\kappa_0^2}.
$$
等价地，若从十维 Einstein-frame action
$(2\kappa_{10}^2)^{-1}\int\sqrt{-g_E}R_E$ 出发，则
$$
\frac1{\kappa_{4,E}^2}
=\frac{\operatorname{Vol}_E(X)}{\kappa_{10}^2}.
$$

**证明.** 在 string frame 把
$$
\frac1{2\kappa_0^2}\int_{M_4\times X}
d^{10}x\sqrt{-g_S}\,e^{-2\Phi_0}R_S
$$
代入 product metric。四维 Ricci scalar 与内部坐标无关，volume form 分解，故其
系数恰为 $e^{-2\Phi_0}\operatorname{Vol}_S(X)/(2\kappa_0^2)$。Einstein-frame
公式同理且无 dilaton factor。这里是等号而非“正比”，但只在声明的无 warp、常
dilaton 和 zero-mode truncation 下成立；warping 或 varying dilaton 会把内部体积
改成相应加权积分。$\square$

**注 13.9A（Kaluza--Klein 截断边界）.** 若 $\lambda_1>0$ 是相关内部 Laplacian
的第一非零本征值，零模有效理论至少要求 $E^2\ll\lambda_1$。这与
$E^2\alpha'\ll1$ 不同：大体积可产生远低于 string scale 的 KK modes，小 cycles
则可产生轻 winding states。只检查十维曲率小不足以保证某个有限四维 truncation
闭合。

**命题 13.10（type II moduli multiplets）.** Type IIA on Calabi-Yau threefold 中，Kahler moduli 位于 vector multiplets，complex structure moduli 位于 hypermultiplets；type IIB 中二者角色交换。

**推导说明（标准物理口径）.** 该分配由 R-R potentials 在 harmonic forms 上的展开和 mirror symmetry 兼容性决定。完整 multiplet 构造需要四维 $\mathcal N=2$ supergravity 表示论。$\square$

## 13.5 Heterotic compactification 的接口

**定义 13.11（Hermitian Yang-Mills 条件）.** Heterotic compactification 还需在 $X$ 上选择 gauge bundle $V$。保持 supersymmetry 的最低阶条件包括
$$
F^{0,2}=F^{2,0}=0,\qquad
J^{mn}F_{mn}=0.
$$

**外部输入定理 13.12（Donaldson--Uhlenbeck--Yau）.** 设 $X$ 为 compact Kahler
manifold，$V\to X$ 为 holomorphic vector bundle。$V$ admits a Hermitian--Einstein
connection if and only if $V$ is polystable；connection 的中心常数由 slope
$\mu(V)$ 决定。特别地，当 $\mu(V)=0$ 时，该方程化为定义 13.11 的 primitive/Hermitian
Yang--Mills 条件 $J^{mn}F_{mn}=0$。

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

**外部输入定理 13.16（quintic 的 Hodge numbers）.** 光滑 quintic threefold 满足
$$
h^{1,1}(X)=1,\qquad h^{2,1}(X)=101.
$$

**验证计算（使用外部几何输入）.** Lefschetz hyperplane theorem 给出
$h^{1,1}=1$。Kodaira--Spencer/Jacobian-ring deformation theory 保证 smooth
quintic 的 infinitesimal complex deformations 由五次齐次多项式的系数计数，再商去
整体缩放和 $\operatorname{PGL}(5)$ 坐标变换：
$$
\binom{5+4}{4}-1-24=126-1-24=101.
$$
最后一行是完整有限维算术；把它识别为 $h^{2,1}$ 的证明责任由已明确引用的
deformation theory 承担，本书不伪造其短证明。

**例 13.17（Euler characteristic）.** Calabi-Yau threefold 的 Euler characteristic 为
$$
\chi(X)=2(h^{1,1}-h^{2,1}).
$$
Quintic 因此有
$$
\chi(X)=2(1-101)=-200.
$$

## 13.8 四维场的零模展开

**外部输入定理 13.18（线性化 Kaluza--Klein/Hodge 分解）.** 设 $X$ compact、
smooth、无边界，background 是无 warp 直积且无 flux。对自由线性化高维 $p$-form，
内部 Hodge Laplacian 有离散非负谱
$$
\Delta_X\omega_I=\lambda_I\omega_I,
\qquad \lambda_I\ge0.
$$
沿 harmonic $q$-forms（$\lambda_I=0$）的展开产生低维 massless
$(p-q)$-forms；$\lambda_I>0$ 的 mode 具有 $m_I^2=\lambda_I$。

**证明路线（外部输入）.** Compact Hodge theorem 给出 self-adjoint Laplacian 的
离散谱和 $\ker\Delta_X$ 与 de Rham cohomology 的同构。把正交本征展开代入 quadratic
高维 kinetic term，内部导数项逐模给出 $\lambda_I$，故 $m_I^2=\lambda_I$。Flux、
warping、gauging、Stueckelberg coupling 与 quantum potential 可提升原 harmonic
modes；因此定理只陈述 free product-background 的线性谱。

零模展开把几何计数落实为四维质量谱：内部 Laplacian 的 harmonic representatives
给出线性化 massless fields，正本征值给出 Kaluza--Klein masses；quintic 的
$h^{1,1}=1,h^{2,1}=101$ 则提供了可计算实例。Ricci-flat Kahler metric 与
$SU(3)$ holonomy 保留 supersymmetry，却也留下连续 moduli。Flux、warping、gauging
或量子效应可提升这些零模，所以 Calabi--Yau 紧化建立的是无 flux 基线，而不是
vacuum selection 或 stabilization 的答案。

## 练习

**练习 13.1.** 说明 $c_1(X)=0$ 与 canonical bundle 平凡之间的关系。

**练习 13.2.** 对 Calabi-Yau threefold，解释为什么 complex structure moduli 由 $H^{2,1}(X)$ 计数。

**练习 13.3.** 用五次齐次多项式计数推导 quintic 的 $h^{2,1}=101$。
