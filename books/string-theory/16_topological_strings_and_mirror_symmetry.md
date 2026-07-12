# 第十六章：topological strings、A/B model 和 mirror symmetry

## 本章目标

本章建立 topological string 的主线：

1. 从二维 $N=(2,2)$ SCFT 进行 topological twist；
2. A-model 只依赖 symplectic/Kahler 数据，并计算 Gromov-Witten 型不变量；
3. B-model 只依赖 complex structure 数据，并由 periods 与 variation of Hodge structure 控制；
4. Mirror symmetry 交换 A/B models，并把 enumerative geometry 问题转化为 period 计算。

## 依赖前置知识

需要第十三章 Calabi-Yau 紧化、第三章 CFT、附录 A 的复几何和附录 D 的 Riemann surface 语言。

## 16.1 Topological twist

**定义 16.1（topological twist）.** 给定二维 $N=(2,2)$ SCFT，topological twist 是把 worldsheet Lorentz generator 与 $U(1)$ R-symmetry current 组合，改变 fields 的 spin，使某个 supercharge 成为 scalar operator $Q$。物理算子定义为
$$
Q\text{-cohomology}.
$$

**命题 16.2（topological invariance）.** 若 stress tensor 是 $Q$-exact：
$$
T_{ab}=\{Q,G_{ab}\},
$$
则 $Q$-closed observables 的 correlators 对 worldsheet metric deformation 不变。

**证明.** 对 metric 变分有
$$
\delta\langle\mathcal O\rangle
\sim
\left\langle
\mathcal O\int \delta h^{ab}T_{ab}
\right\rangle
=
\left\langle
\mathcal O\int \delta h^{ab}\{Q,G_{ab}\}
\right\rangle.
$$
若 $\mathcal O$ 为 $Q$-closed 且路径积分测度无 anomaly，则右侧为 $Q$-exact 期望值，等于零。$\square$

## 16.2 A-model

**定义 16.3（A-model）.** A-model 是对 Calabi-Yau target 的 $N=(2,2)$ sigma model 作 A-twist 得到的 topological field theory。其 observables 与 target 的 de Rham cohomology 相关。

**定理 16.4（A-model localization）.** A-model path integral 局域化到 holomorphic maps
$$
f:\Sigma_g\to X.
$$
其 correlators 依赖 complexified Kahler class $B+iJ$，不依赖 complex structure deformation。

**推导说明（标准物理口径）.** A-twist 后 action 可写为 $Q$-exact 项加上 topological term
$$
\int_\Sigma f^*(B+iJ).
$$
改变非 Kahler 的 metric data 只改变 $Q$-exact 项；stationary locus 由相应 fermionic variation 给出 Cauchy-Riemann equation，即 holomorphic maps。$\square$

**定义 16.5（Gromov-Witten generating function）.** A-model genus $g$ free energy 形式上写为
$$
F_g^A(t)=
\sum_{\beta\in H_2(X,\mathbb Z)}
N_{g,\beta}\,e^{2\pi i\langle t,\beta\rangle},
$$
其中 $N_{g,\beta}$ 是 Gromov-Witten invariants 或其适当虚计数版本。

## 16.3 B-model

**定义 16.6（B-model）.** B-model 是对 Calabi-Yau target 的 $N=(2,2)$ sigma model 作 B-twist 得到的 topological field theory。其 observables 与 polyvector-valued forms 相关，并由 complex structure 控制。

**定理 16.7（B-model dependence）.** B-model correlators 依赖 complex structure moduli，不依赖 Kahler moduli。

**推导说明（标准物理口径）.** B-twist 后，Kahler class 的变化进入 $Q$-exact deformation，而 complex structure 改变 BRST operator 与 chiral ring 结构。故 $Q$-cohomology correlators 只感知 complex structure。$\square$

**定义 16.8（periods）.** 对 Calabi-Yau threefold 的 holomorphic three-form $\Omega$，periods 定义为
$$
\Pi_\Gamma=\int_\Gamma\Omega,
\qquad
\Gamma\in H_3(X,\mathbb Z).
$$
B-model genus-zero special geometry 由这些 periods 及其 Picard-Fuchs equations 控制。

## 16.4 Mirror symmetry

**物理猜想 16.9（mirror symmetry）.** Mirror symmetry 断言存在 Calabi-Yau pairs $(X,Y)$，使
$$
A\text{-model on }X
\simeq
B\text{-model on }Y,
$$
并交换 Hodge numbers
$$
h^{1,1}(X)=h^{2,1}(Y),\qquad
h^{2,1}(X)=h^{1,1}(Y).
$$

**命题 16.10（mirror map 的计算意义）.** 若 mirror pair $(X,Y)$ 已知，则 $X$ 上的 genus-zero Gromov-Witten invariants 可由 $Y$ 上的 period integrals 和 mirror map 计算。

**推导说明（标准物理口径）.** Mirror symmetry 把 A-model Kahler moduli 映到 B-model complex structure moduli。A-model 的 genus-zero prepotential 包含 curve counting data；B-model prepotential 由 periods 的 special geometry 给出。通过 mirror map 对齐 flat coordinates 后，展开 B-model prepotential 即读出 A-model invariants。$\square$

**注 16.11（数学状态）.** Mirror symmetry 在许多具体族和适当形式化下已有严格数学定理，但一般物理陈述仍作为对偶性原则使用。本书把其全局形式标为物理猜想，把已数学化的局部计算作为外部输入或例子。

## 16.5 Topological string amplitudes

**定义 16.12（topological string free energy）.** Topological string partition function 形式写为
$$
Z_{\mathrm{top}}
=\exp\left(\sum_{g\ge0}g_{\mathrm{top}}^{2g-2}F_g\right).
$$
其中 $F_g$ 是 genus $g$ topological amplitude。

**注 16.13（物理应用）.** Topological string amplitudes 控制 type II compactification 中某些受保护的 F-terms，并与 black hole entropy、Donaldson-Thomas theory 和 enumerative geometry 相连。这些接口在第十七章和第二十章继续使用。

## 16.6 Quintic mirror 的 Picard-Fuchs 接口

**定义 16.14（quintic mirror family 的参数）.** Quintic mirror family 的 complex structure moduli 常用参数 $z$ 描述。其 periods 满足四阶 Picard-Fuchs equation
$$
\left[
\theta^4
-5z(5\theta+1)(5\theta+2)(5\theta+3)(5\theta+4)
\right]\Pi(z)=0,
\qquad
\theta=z\frac{d}{dz}.
$$

**外部输入定理 16.15（mirror theorem for quintic 的接口）.** Quintic 的 genus-zero Gromov-Witten invariants 可由 mirror family 的 periods、mirror map 和 B-model Yukawa coupling 计算。

**使用边界.** 本书不证明 Givental/Lian-Liu-Yau mirror theorem；只使用其作为 A-model curve counting 与 B-model period calculation 等价的标准例子。

**命题 16.16（mirror map 的局部形式）.** 在 large complex structure point 附近，mirror map 具有形式
$$
t(z)=\frac{\Pi_1(z)}{\Pi_0(z)},
\qquad
q=e^{2\pi i t},
$$
其中 $\Pi_0$ 是 holomorphic period，$\Pi_1$ 是含 logarithm 的 period。

**推导说明（标准物理口径）.** Picard-Fuchs 方程在 maximally unipotent monodromy point 附近有 Frobenius basis：一个 holomorphic solution 和一个 logarithmic solution。其比值给出 flat coordinate，即 A-model complexified Kahler parameter。$\square$

## 本章小结

Topological string 是 string theory 与现代几何之间最精确的接口之一。A-model 把 holomorphic curve counting 编码为物理 correlators；B-model 把 complex structure variation 和 periods 组织为 special geometry；mirror symmetry 把二者等同。

## 练习

**练习 16.1.** 说明 A-model 为什么只依赖 Kahler moduli 而不依赖 complex structure moduli。

**练习 16.2.** 解释 mirror symmetry 如何把 curve counting 转化为 period 计算。

**练习 16.3.** 写出 quintic mirror Picard-Fuchs operator，并说明 holomorphic period 与 logarithmic period 在 mirror map 中的作用。

