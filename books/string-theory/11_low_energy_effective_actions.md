# 第十一章：低能有效作用、supergravity 和 alpha-prime 修正

## 本章目标

本章说明弦理论如何在低能极限中产生 gravity、antisymmetric tensor、dilaton、gauge fields 和 supergravity。核心区分是：

1. $\alpha'$ expansion：世界面 sigma model 的短距离或高导数修正；
2. $g_s$ expansion：按 worldsheet genus 组织的 string loop 修正；
3. string frame 与 Einstein frame：同一低能物理的不同场变量。

## 依赖前置知识

需要第四章闭弦低能谱、第五章 Weyl anomaly、第八至十章的超弦谱与异常结构。

## 11.1 背景场 sigma model

**定义 11.1（NS-NS 背景 sigma model）.** 对一般 target fields $g_{\mu\nu},B_{\mu\nu},\Phi$，Euclidean worldsheet action 为
$$
S=\frac1{4\pi\alpha'}\int_\Sigma
\left(
g_{\mu\nu}(X)dX^\mu\wedge *dX^\nu
+B_{\mu\nu}(X)dX^\mu\wedge dX^\nu
+\alpha'\Phi(X)R^{(2)}*1
\right).
$$
记
$$
H=dB.
$$

**外部输入定理 11.2（Weyl invariance 与 beta functions）.** 背景场给出一致 string background 的微扰条件是 worldsheet beta functions 消失。最低非平凡阶为
$$
\beta^g_{\mu\nu}
=\alpha'\left(
R_{\mu\nu}+2\nabla_\mu\nabla_\nu\Phi
-\frac14H_{\mu\rho\sigma}H_\nu^{\ \rho\sigma}
\right)+O(\alpha'^2),
$$
$$
\beta^B_{\mu\nu}
=\alpha'\left(
-\frac12\nabla^\rho H_{\rho\mu\nu}
+\nabla^\rho\Phi\,H_{\rho\mu\nu}
\right)+O(\alpha'^2),
$$
并有相应 dilaton beta function。

**使用边界.** Beta function 的完整多圈计算属于二维 sigma model renormalization。本书只使用最低阶方程与有效作用的等价。

## 11.2 String-frame NS-NS action

**定义 11.3（string-frame action）.** NS-NS sector 的 tree-level string-frame effective action 为
$$
S_{\mathrm{NS}}
=\frac1{2\kappa_0^2}
\int d^Dx\sqrt{-g}\,e^{-2\Phi}
\left(
R+4\nabla_\mu\Phi\nabla^\mu\Phi
-\frac1{12}H_{\mu\nu\rho}H^{\mu\nu\rho}
+O(\alpha')
\right).
$$

**命题 11.4（有效作用与 beta function 方程一致）.** 对定义 11.3 的 action 变分，所得 $g_{\mu\nu}$ 与 $B_{\mu\nu}$ 方程在最低阶等价于定理 11.2 中 $\beta^g=\beta^B=0$。

**证明草图.** 对 $B$ 变分得到
$$
d(e^{-2\Phi}*H)=0,
$$
等价于
$$
\nabla^\rho H_{\rho\mu\nu}-2\nabla^\rho\Phi H_{\rho\mu\nu}=0.
$$
对 metric 变分并整理 dilaton 项得到
$$
R_{\mu\nu}+2\nabla_\mu\nabla_\nu\Phi
-\frac14H_{\mu\rho\sigma}H_\nu^{\ \rho\sigma}=0.
$$
这正是 beta function 消失条件。$\square$

**命题 11.4A（dilaton equation）.** 对 string-frame NS-NS action 的 dilaton 变分给出
$$
4\nabla^2\Phi-4|\nabla\Phi|^2
+R-\frac1{12}H^2+O(\alpha')=0
$$
在临界维数且最低阶近似下成立。

**证明草图.** 对 $e^{-2\Phi}$ 变分给出 $-2$ 倍括号内 Lagrangian；对 $4|\nabla\Phi|^2$ 变分并分部积分给出 $8\nabla^2\Phi-8|\nabla\Phi|^2$ 的组合。整理后得到所列方程。$\square$

## 11.3 String frame 与 Einstein frame

**定义 11.5（Einstein frame）.** Einstein-frame metric 定义为
$$
g^E_{\mu\nu}=e^{-\frac{4\Phi}{D-2}}g^S_{\mu\nu}.
$$
在该 frame 中，Einstein-Hilbert 项不再带整体 $e^{-2\Phi}$。

**命题 11.6（frame transformation 的作用）.** String frame 中 fundamental string 直接耦合于 $g^S$；Einstein frame 中 gravitational kinetic term 规范化为标准 Einstein-Hilbert 形式。两者由局部场重定义相连，因此 on-shell S-matrix 不依赖 frame choice。

**证明草图.** 将 Weyl rescaling 代入 Ricci scalar 的变换公式，选择指数 $-4/(D-2)$ 正好消去 $e^{-2\Phi}$ 与 volume form 的组合。局部可逆场重定义不改变 on-shell scattering amplitudes。$\square$

**定义 11.6A（canonical dilaton）.** 在 Einstein frame 中，dilaton kinetic term 可通过线性重标定写成 canonical scalar kinetic term。具体归一化依赖 $D$，但所有 frame-invariant statements 必须用物理 S-matrix、charges 或无量纲耦合表达。

## 11.4 Type II low-energy actions

**定义 11.7（type II bosonic action 的结构）.** Type II supergravity 的 bosonic action 在 string frame 中具有结构
$$
S_{\mathrm{II}}
=\frac1{2\kappa_0^2}
\int d^{10}x\sqrt{-g}
\left[
e^{-2\Phi}\left(
R+4|\nabla\Phi|^2-\frac12|H_3|^2
\right)
-\frac12\sum_p |F_p|^2
\right]
+S_{\mathrm{CS}}
+O(\alpha').
$$
IIA 中 $p$ 为偶数 field strengths，IIB 中 $p$ 为奇数 field strengths，并需额外施加 $F_5=*F_5$。

**注 11.8（democratic formulation）.** 民主形式同时写入某个 R-R field strength 及其 Hodge dual，再通过 duality constraint 去掉重复自由度。这是 action-level covariance 与自由度计数之间的技术折中。

**命题 11.8A（R-R kinetic term 与 dilaton）.** 在 string frame 中，R-R kinetic terms 不带整体 $e^{-2\Phi}$，而 NS-NS kinetic terms 带该因子。

**证明草图.** 该结构可由 sphere scattering amplitude 的 dilaton dependence 与 RNS sector 区分读出：NS-NS fields 属于 worldsheet sigma model 背景耦合，tree-level action 带 $e^{-2\Phi}$；R-R fields 的规范化在 type II supergravity 中按 democratic field strengths 写成不带该整体因子的 kinetic terms。$\square$

## 11.5 Heterotic low-energy action

**定义 11.9（heterotic tree-level action）.** Heterotic string 的最低阶 string-frame action 含有
$$
S_{\mathrm{het}}
=\frac1{2\kappa_0^2}
\int d^{10}x\sqrt{-g}\,e^{-2\Phi}
\left[
R+4|\nabla\Phi|^2-\frac1{12}|H|^2
-\frac{\alpha'}4\operatorname{Tr}|F|^2
+\frac{\alpha'}4\operatorname{tr}|R_+|^2
+O(\alpha'^2)
\right],
$$
其中
$$
H=dB-\frac{\alpha'}4(\omega_{\mathrm{YM}}-\omega_{\mathrm L})
$$
与第十章 anomaly cancellation 相容。

**注 11.10.** $R_+$ 表示含 torsion connection 的曲率，具体 choice 与 field redefinition convention 有关。本书只固定其在 anomaly cancellation 和 Bianchi identity 中的接口。

## 11.6 Expansion parameters

**定义 11.11（两个展开）.** String perturbation 有两个互相独立但会耦合出现的展开方向：

1. Worldsheet genus expansion：由 $g_s=e^{\Phi_0}$ 控制，genus $g$ 闭弦振幅带因子 $g_s^{2g-2}$。
2. $\alpha'$ expansion：由背景曲率半径、场强变化尺度相对 string length $\ell_s=\sqrt{\alpha'}$ 的大小控制。

**命题 11.12（supergravity 适用条件）.** Classical supergravity 是 string theory 的有效近似，当且仅当
$$
g_s\ll 1,\qquad
\alpha' \mathcal R\ll 1
$$
并且外部能量远低于 massive string scale。

**证明草图.** $g_s\ll1$ 抑制高 genus string loops；$\alpha'\mathcal R\ll1$ 抑制高导数 worldsheet beta function 修正；低能条件允许积分掉 massive string modes。$\square$

## 11.7 与散射振幅的匹配

**命题 11.13（Einstein-Hilbert 项的振幅匹配）.** 闭弦四点 graviton 振幅在低能极限 $\alpha's,\alpha't,\alpha'u\ll1$ 下，其 massless pole 与接触项展开匹配 Einstein-Hilbert action 的 tree-level graviton scattering，并给出高阶 $R^4$ 型 $\alpha'$ 修正。

**证明草图.** 第六章 Virasoro-Shapiro amplitude 的 Gamma functions 在低能展开中产生 $1/stu$ 型 massless exchange poles，与 supergravity 中 graviton、$B$-field、dilaton exchange 相符。去掉 pole 后的解析项对应 higher-derivative contact interactions，type II 中首个受保护修正为 $\alpha'^3R^4$。完整张量结构匹配需要四 graviton polarization 计算。$\square$

## 本章小结

低能有效作用是 worldsheet conformal invariance、massless spectrum 和 string perturbation 的共同结果。String frame 直接反映世界面耦合；Einstein frame 适合讨论引力动力学。Supergravity 不是完整 string theory，而是 $g_s$ 和 $\alpha'$ 双重展开中的最低阶。

## 练习

**练习 11.1.** 说明 string frame 与 Einstein frame 的区别。

**练习 11.2.** 从 $S_{\mathrm{NS}}$ 对 $B$ 变分，推出 $d(e^{-2\Phi}*H)=0$。

**练习 11.3.** 从 string-frame NS-NS action 对 dilaton 变分，推导最低阶 dilaton equation。

