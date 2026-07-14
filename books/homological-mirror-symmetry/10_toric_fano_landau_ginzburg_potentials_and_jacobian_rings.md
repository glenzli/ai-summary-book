# 第十章：toric Fano、Landau-Ginzburg potential 与 Jacobian ring

在 toric Fano 几何中，Landau--Ginzburg 势函数并不是事后猜出的 Laurent 多项式：moment polytope 的每个 facet 产生一族 Maslov 指数二圆盘，而它们的面积和边界 holonomy 正好组成 $W$ 的各项。于是第五章的 weak bounding cochain potential 变得可计算；它的临界方程控制 torus fiber 的变形 Floer 上同调，Jacobian 商则给出 closed-string 侧的代数影子。本章从 fan/polytope 数据走到 disk counts，再对 $\mathbb P^1$ 与 $\mathbb P^n$ 做显式临界点计算，最后区分 Fano/LG 镜像的两个方向。

## 10.1 Toric 数据

**定义 10.1.** 设 $X_\Sigma$ 是由 fan $\Sigma$ 定义的光滑 projective toric variety。若 anticanonical divisor $-K_X$ ample，则称 $X$ 为 toric Fano。

**定义 10.2.** toric Fano 的 Hori-Vafa 型镜像 Landau-Ginzburg 模型是
$$
((\mathbb C^\ast)^n,W)
$$
其中 $W$ 是由 polytope facets 或 fan rays 给出的 Laurent polynomial。典型形式为
$$
W(z)=\sum_{\rho\in\Sigma(1)} c_\rho z^{m_\rho}.
$$

**例 10.3.** 对 $X=\mathbb P^1$，镜像 potential 为
$$
W(z)=z+qz^{-1}.
$$
对 $X=\mathbb P^n$，标准 potential 为
$$
W(z_1,\ldots,z_n)=z_1+\cdots+z_n+q(z_1\cdots z_n)^{-1}.
$$

## 10.2 Disk potential

**定义 10.4.** 设 $L_u$ 是 moment polytope 内点 $u$ 对应的 Lagrangian torus fiber。其 disk potential 定义为
$$
W_L(y)=\sum_{\beta:\mu(\beta)=2} n_\beta T^{\omega(\beta)}y^{\partial\beta},
$$
其中 $\beta$ 遍历 Maslov index $2$ disk classes，$n_\beta$ 是开 Gromov-Witten 型计数，$y^{\partial\beta}$ 记录局部系统 holonomy。

**外部输入定理 10.5（toric disk potential）.** 对光滑 compact toric Fano，基本 Maslov index $2$ disks 与 moment polytope facets 对应，disk potential 与 Hori-Vafa Laurent polynomial 匹配。
来源：Cho--Oh 与 FOOO 的 toric Floer 理论；本章只调用 facet--disk 对应和
势函数公式，一般虚拟计数仍由这些来源承担。

## 10.3 Jacobian ring

**定义 10.6.** Laurent polynomial $W\in k[z_1^{\pm1},\ldots,z_n^{\pm1}]$ 的 Jacobian ring 定义为
$$
\operatorname{Jac}(W)=
\frac{k[z_1^{\pm1},\ldots,z_n^{\pm1}]}
{\left(z_1\frac{\partial W}{\partial z_1},\ldots,
z_n\frac{\partial W}{\partial z_n}\right)}.
$$

**命题 10.7.** 对 $W(z)=z+qz^{-1}$，若 $\operatorname{char}k\ne2$ 且 $q\ne0$，则
$$
\operatorname{Jac}(W)\cong k[z^{\pm1}]/(z-qz^{-1})
\cong k[z]/(z^2-q).
$$

**证明.** 有
$$
z\frac{dW}{dz}=z-qz^{-1}.
$$
在 Laurent polynomial ring 中商去该关系等价于 $z=qz^{-1}$，两边乘以 $z$ 得 $z^2=q$。反向关系 $z^2-q=0$ 且 $z$ 可逆，因为 $q\ne0$，所以两个商同构。证毕。

$\mathbb P^1$ 的计算在任意维数都有同一个消元机制。

**命题 10.7A（$\mathbb P^n$ 势函数）.** 设 $q\in k^\times$，并令
$$
W=z_1+\cdots+z_n+q(z_1\cdots z_n)^{-1}
\in k[z_1^{\pm1},\ldots,z_n^{\pm1}].
$$
则
$$
\operatorname{Jac}(W)\cong k[t]/(t^{n+1}-q).
\tag{10.1}
$$
若 $k$ 代数闭、$\operatorname{char}k\nmid n+1$，则 $W$ 有 $n+1$ 个非退化
临界点
$$
(z_1,\ldots,z_n)=(t,\ldots,t),\qquad t^{n+1}=q,
$$
相应临界值为 $(n+1)t$。

**证明.** 写 $p=q(z_1\cdots z_n)^{-1}$。对数导数为
$$
z_i\frac{\partial W}{\partial z_i}=z_i-p,
$$
所以 Jacobian 商中 $z_1=\cdots=z_n=p=:t$。代回 $p$ 的定义得到
$t=q t^{-n}$，即 $t^{n+1}=q$。反之，该关系使 $t$ 可逆，并给出从 Laurent
商到 $k[t]/(t^{n+1}-q)$ 的逆映射，故有 (10.1)。在所列特征假设下，
$t^{n+1}-q$ 与其导数 $(n+1)t^n$ 无公共根，因此各局部 Jacobian 代数均为
一维，临界点非退化。最后将 $z_i=t$ 与
$q(z_1\cdots z_n)^{-1}=t$ 代入 $W$，得临界值 $(n+1)t$。证毕。

**外部输入定理 10.8（closed-open/Jacobian 对应）.** 在 toric Fano 的适当假设下，quantum cohomology、disk potential 的 Jacobian ring 和 Fukaya category 的 closed-open 结构之间存在同构或分块对应。
来源：FOOO、Auroux、Abouzaid 等；需根据具体模型定位。

## 10.4 HMS 形态

toric Fano HMS 有多个互补版本。

**版本 10.9（Fano/LG）。** B-side 是 toric Fano $X$，A-side 是其 Landau-Ginzburg mirror 的 Fukaya-Seidel 或 wrapped category：
$$
\mathcal F\mathcal S((\mathbb C^\ast)^n,W)\simeq \operatorname{Perf}(X).
$$

**版本 10.10（A-side toric fiber）。** A-side 取 toric variety $X$ 的 Fukaya category，B-side 取 mirror Landau-Ginzburg model 的 matrix factorization categories：
$$
\mathcal F(X)\simeq \bigoplus_{\lambda}\operatorname{MF}((\mathbb C^\ast)^n,W-\lambda)
$$
在合适 semisimplicity 和 monotone/Novikov 假设下理解。

**警告 10.11.** 版本 10.9 和版本 10.10 交换了 A/B 侧角色，取决于把 $X$ 放在 A-model 还是 B-model。写 HMS 命题时必须声明方向。

## 10.5 Toric HMS 的生成元方法

**外部输入定理 10.12（Abouzaid 的 toric 生成子范畴比较）.** 对 smooth
projective toric variety $X$，Abouzaid 构造与其 Landau--Ginzburg mirror
相关、边界落在势函数一个 level set 上的 Lagrangian $A_\infty$ category，
并证明它与 $X$ 上 line bundles 组成的 dg category quasi-equivalent。这给出
toric HMS 的生成子范畴部分；升级为全范畴 Morita 等价仍需相应生成性。
来源：Abouzaid, *Morse Homology, Tropical Geometry, and Homological Mirror Symmetry for Toric Varieties*。

**解释 10.13.** 在定义 8.15 的语言中，A-side Lagrangians 与 B-side line
bundles 给出候选生成对象，tropical/Morse 模型比较它们的完整
$A_\infty$/dg endomorphism 数据；closed-string Jacobian 计算只是这条增强
比较的必要旁证。

从 facet 圆盘到 Laurent 势函数，再从对数导数到 Jacobian ring，toric Fano 例子把 obstruction 理论转化成了显式代数。不过 Jacobian/quantum 对应只检测 closed-string 层，不能替代 Fukaya--Seidel 范畴与 $\operatorname{Perf}(X)$ 的增强比较。要看到该比较中的生成对象及其有向态射，下一章把 $W$ 本身视为 Lefschetz fibration。

## 练习

**练习 10.1.** 计算 $\mathbb P^2$ potential 的 critical point equations。

**练习 10.2.** 对 $W=z+qz^{-1}$，计算 critical values。

**练习 10.3.** 说明为什么 $\operatorname{Jac}(W)$ 的定义使用 $z_i\partial W/\partial z_i$ 而不是普通偏导更适合 Laurent polynomial。

**练习 10.4.** 按定义 8.15 写出 $\mathbb P^1$ 的 toric HMS 比较数据。
