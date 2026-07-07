# 第十章：toric Fano、Landau-Ginzburg potential 与 Jacobian ring

## 本章目标

本章说明 toric Fano mirror symmetry 中的基本机制：B-side toric variety 的镜像通常是 Landau-Ginzburg 模型，A-side disk counts 给出 potential，critical locus 与 Floer cohomology、Jacobian ring 与 quantum cohomology 相联系。

## 依赖前置知识

需要第五章的 Novikov/obstruction 理论和第八章的 HMS 数据包。需要知道 toric variety 的 moment polytope 和 Laurent polynomial 的基本概念。

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
来源：Cho-Oh、FOOO 及 toric Floer theory 文献；本书后续 theorem locator 需补精确定位。

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

**外部输入定理 10.12（Abouzaid toric HMS 入口）.** 对 smooth projective toric varieties，Abouzaid 构造了与 Landau-Ginzburg mirror 相关的 Lagrangian $A_\infty$ category，并证明其与由 line bundles 生成的 dg category quasi-equivalent，从而建立 toric HMS 的重要部分。  
来源：Abouzaid, *Morse Homology, Tropical Geometry, and Homological Mirror Symmetry for Toric Varieties*。

**解释 10.13.** 证明策略符合模板 8.15：选取 A-side Lagrangians，选取 B-side line bundles，计算两边 endomorphism algebras，并用 tropical/Morse 模型比较。

## 本章小结

Toric Fano mirror symmetry 中，Landau-Ginzburg potential 是核心对象。A-side disk counts 产生 $W$，critical locus 与 nonzero Floer cohomology 相关，Jacobian ring 与 closed-string invariants 匹配。HMS 的范畴证明通常通过生成对象和 endomorphism algebra 比较完成。

## 练习

**练习 10.1.** 计算 $\mathbb P^2$ potential 的 critical point equations。

**练习 10.2.** 对 $W=z+qz^{-1}$，计算 critical values。

**练习 10.3.** 说明为什么 $\operatorname{Jac}(W)$ 的定义使用 $z_i\partial W/\partial z_i$ 而不是普通偏导更适合 Laurent polynomial。

**练习 10.4.** 按模板 8.15 写出 $\mathbb P^1$ 的 toric HMS 数据包。
