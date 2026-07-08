# 第十五章：Riemann surfaces、moduli of curves 和高 genus 扰动论

## 本章目标

本章解释 string perturbation theory 如何由 Riemann surface topology 组织。相对于点粒子 Feynman graphs，弦的相互作用由世界面的拓扑和 puncture moduli 控制。

## 依赖前置知识

需要第五章 path integral、 第六章顶点算子和附录 D 的 Riemann surface 语言。

## 15.1 Genus expansion

**定义 15.1（闭弦 genus expansion）.** 闭弦 genus $g$、$n$ 点振幅具有形式
$$
\mathcal A_{g,n}
\sim
g_s^{2g-2+n}
\int_{\mathcal M_{g,n}}\Omega_{g,n},
$$
其中 $\mathcal M_{g,n}$ 是带 $n$ 个 marked points 的 genus $g$ Riemann surface moduli space。

**命题 15.2（moduli dimension）.** 若 $2g-2+n>0$，则
$$
\dim_{\mathbb C}\mathcal M_{g,n}=3g-3+n.
$$

**证明草图.** 该结论来自 Riemann surface deformation theory：complex structure infinitesimal deformations 由 Beltrami differentials modulo trivial deformations 给出，其 dual space 是 quadratic differentials with allowed simple poles at punctures。Riemann-Roch 给出维数 $3g-3+n$。$\square$

## 15.2 Gauge fixing measure

**定义 15.3（Beltrami differentials 与 $b$-ghost insertions）.** 设 $\mu_a$ 是 moduli space 切向量对应的 Beltrami differentials。Bosonic string 中固定 complex structure 后的 measure 含有
$$
\prod_{a=1}^{3g-3+n}
\left(\int_\Sigma \mu_a b\right)
\left(\int_\Sigma \bar\mu_a \bar b\right)
$$
插入。

**命题 15.4（高 genus 振幅结构）.** Bosonic closed string 的高 genus 振幅可写成
$$
\mathcal A_{g,n}
=
g_s^{2g-2+n}
\int_{\mathcal M_{g,n}}
\left\langle
\prod_a (b,\mu_a)(\bar b,\bar\mu_a)
\prod_{i=1}^n V_i
\right\rangle_{\Sigma_g}.
$$

**证明草图.** 从 Polyakov path integral 出发，对 metrics 商去 diffeomorphism 与 Weyl transformations。局部切片由 complex structure moduli 参数化；Faddeev-Popov determinant 由 $bc$ ghosts 表示；moduli 方向的 ghost zero modes 通过 $b$-ghost 与 Beltrami differentials 配对吸收。$\square$

## 15.3 Torus partition function 与 modular invariance

**定义 15.5（torus parameter）.** Genus-one Riemann surface 可写为
$$
E_\tau=\mathbb C/(\mathbb Z+\tau\mathbb Z),
\qquad \operatorname{Im}\tau>0.
$$
同构曲面由 modular group
$$
SL(2,\mathbb Z)
$$
作用识别：
$$
\tau\mapsto\frac{a\tau+b}{c\tau+d}.
$$

**定义 15.6（torus partition function）.** 闭弦一圈 partition function 具有 trace 形式
$$
Z(\tau,\bar\tau)
=
\operatorname{Tr}_{\mathcal H}
\left(
q^{L_0-c/24}\bar q^{\tilde L_0-\tilde c/24}
\right),
\qquad q=e^{2\pi i\tau}.
$$

**外部输入定理 15.7（modular invariance）.** 一致 closed string theory 的 torus partition function 必须在 $SL(2,\mathbb Z)$ 下不变，并在 fundamental domain 上积分。

**使用边界.** Modular invariance 的证明依赖 Riemann surface mapping class group 与 path integral gauge fixing。正文使用其对 spectrum、GSO projection 和 heterotic lattice 的约束。

## 15.4 Degeneration 与 factorization

**命题 15.8（边界退化与中间态）.** 当 Riemann surface 退化为由长细管连接的两个曲面时，string amplitude 在对应 channel 上因子化为一组中间 string states 的传播。

**证明草图.** Plumbing fixture parameter $q$ 描述细管。CFT 在细管上的传播给出
$$
\sum_\alpha q^{L_0^\alpha-a}\bar q^{\tilde L_0^\alpha-a}
|\alpha\rangle\langle\alpha|.
$$
对 $q$ 积分在 on-shell 条件处产生 pole，其 residue 为两侧低阶振幅的乘积。$\square$

**注 15.9（Deligne-Mumford compactification）.** Moduli space 的边界由 nodal curves 描述。String perturbation 的 unitarity 与这些边界退化的正确因子化密切相关。

## 15.5 Superstring supermoduli 的接口

**注 15.10.** Superstring 高 genus 振幅需要 super Riemann surfaces 或 picture-changing operator 形式处理 odd moduli。本书不展开 supermoduli 几何，只保留以下规则：RNS amplitude 必须满足 ghost number、picture number、spin structure sum 和 modular invariance 条件。

## 15.6 Torus partition function 示例

**定义 15.11（fundamental domain）.** $SL(2,\mathbb Z)$ 在 upper half-plane 的常用 fundamental domain 为
$$
\mathcal F=\{\tau\in\mathbb H\mid |\tau|\ge1,\ -\tfrac12\le\operatorname{Re}\tau\le\tfrac12\}.
$$

**命题 15.12（单个 compact boson 的 lattice sum）.** 半径 $R$ 的 compact boson 在 torus 上的零模贡献为
$$
Z_{R}^{\mathrm{zero}}(\tau,\bar\tau)
=
\sum_{n,w\in\mathbb Z}
q^{\frac{\alpha'}4p_L^2}
\bar q^{\frac{\alpha'}4p_R^2},
$$
其中
$$
p_L=\frac nR+\frac{wR}{\alpha'},
\qquad
p_R=\frac nR-\frac{wR}{\alpha'}.
$$

**证明草图.** Torus trace 中左、右 Virasoro 零模分别含有 $\alpha'p_L^2/4$ 和 $\alpha'p_R^2/4$。对所有 momentum/winding sectors 求和即得。Poisson resummation 显示该 lattice sum 在 T-duality 和 modular transformation 下具有正确协变性。$\square$

**命题 15.13（level matching from modular $T$）.** Torus partition function 在 $T:\tau\mapsto\tau+1$ 下不变要求物理态满足
$$
L_0-\tilde L_0\in\mathbb Z.
$$
闭弦传播的 on-shell 子空间进一步给出 level matching。

**证明.** Trace 中每个态贡献相位
$$
e^{2\pi i(L_0-\tilde L_0)}
$$
于 $T$ 变换下。要使 partition function 单值，需 $L_0-\tilde L_0$ 为整数；物理闭弦态再由约束选出左右匹配部分。$\square$

## 本章小结

高 genus 弦扰动论是 Riemann surface moduli space 上的积分。Ghost insertions 提供 measure，modular invariance 避免重复计数并约束 spectrum，边界退化保证振幅因子化。

## 练习

**练习 15.1.** 说明 torus modular parameter $\tau$ 的基本区域为什么避免重复计数。

**练习 15.2.** 用 degeneration 图像解释为什么高 genus 振幅边界应出现低阶振幅的因子化。

**练习 15.3.** 说明 compact boson torus partition function 中 momentum/winding lattice sum 如何体现 T-duality。
