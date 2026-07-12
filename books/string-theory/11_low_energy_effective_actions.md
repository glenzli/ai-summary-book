# 第十一章：低能有效作用、supergravity 和 alpha-prime 修正

## 本章目标

本章说明弦理论如何在低能极限中产生 gravity、antisymmetric tensor、dilaton、gauge fields 和 supergravity。核心区分是：

1. $\alpha'$ expansion：世界面 sigma model 的短距离或高导数修正；
2. $g_s$ expansion：按 worldsheet genus 组织的 string loop 修正；
3. string frame 与 Einstein frame：同一低能物理的不同场变量。

## 依赖前置知识

需要第四章闭弦低能谱、第五章 Weyl anomaly、第八至十章的超弦谱与异常结构。

## 11.1 背景场 sigma model

**定义 11.1（NS-NS 背景 sigma model）.** 对一般 target fields
$g_{\mu\nu},B_{\mu\nu},\Phi$，令
$B=(1/2)B_{\mu\nu}dx^\mu\wedge dx^\nu$。Euclidean worldsheet action 取
$$
S_E=\frac1{4\pi\alpha'}\int_\Sigma d^2\sigma\sqrt h\,
\left[
h^{ab}g_{\mu\nu}(X)\partial_aX^\mu\partial_bX^\nu
+i\epsilon^{ab}B_{\mu\nu}(X)\partial_aX^\mu\partial_bX^\nu
+\alpha'\Phi(X)R^{(2)}
\right],
$$
其中 $\epsilon^{12}=1/\sqrt h$。$i$ 来自 Lorentzian worldsheet 的实 $B$-coupling
作 Wick rotation；删去它会改变 Euclidean action 的 reality convention。下文 beta
functions 由该 Euclidean perturbative QFT 定义，而 target metric 仍取 mostly plus。
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

这里的 $O(\alpha'^2)$ 指在固定 background-field renormalization scheme 中的高阶
导数展开。两圈及以上系数会在局部 field redefinition 下改变；“beta function
消失”应理解为模去 target diffeomorphism、$B$-field gauge transformation 与允许的
scheme change。显示的最低阶张量结构是本章使用的共同代表。

## 11.2 String-frame NS-NS action

**定义 11.3（string-frame 两导数截断）.** 只保留 massless NS--NS fields、sphere
阶与两个 target derivatives 时，string-frame effective action 的最低阶截断为
$$
S_{\mathrm{NS}}
=\frac1{2\kappa_0^2}
\int d^Dx\sqrt{-g}\,e^{-2\Phi}
\left(
R+4\nabla_\mu\Phi\nabla^\mu\Phi
-\frac1{12}H_{\mu\nu\rho}H^{\mu\nu\rho}
+O(\alpha'\nabla^4)
\right).
$$
符号 $O(\alpha'\nabla^4)$ 表示相对两导数作用的 higher-derivative terms，不是一个
已证明收敛的幂级数。玻色弦、heterotic 与 type II 的首个非零修正阶数不同。

**命题 11.4（有效作用与 beta function 方程一致）.** 假设 fields 光滑，变分在
边界消失，并同时施加 dilaton Euler--Lagrange equation。则对定义 11.3 的 action
变分所得 $g_{\mu\nu}$ 与 $B_{\mu\nu}$ 方程，在显示的两导数阶等价于定理 11.2 中
$\beta^g=\beta^B=0$。

**推导说明（标准物理口径）.** 对 $B$ 变分得到
$$
d(e^{-2\Phi}*H)=0,
$$
等价于
$$
\nabla^\rho H_{\rho\mu\nu}-2\nabla^\rho\Phi H_{\rho\mu\nu}=0.
$$
对 metric 的未作 trace reversal 的 Euler tensor 为
$$
E^g_{\mu\nu}
=R_{\mu\nu}-\frac12g_{\mu\nu}R
+2\nabla_\mu\nabla_\nu\Phi-2g_{\mu\nu}\nabla^2\Phi
+2g_{\mu\nu}|\nabla\Phi|^2
-\frac14H_{\mu\rho\sigma}H_\nu{}^{\rho\sigma}
+\frac1{24}g_{\mu\nu}H^2.
$$
令
$$
E^\Phi=R+4\nabla^2\Phi-4|\nabla\Phi|^2-\frac1{12}H^2.
$$
逐项相减可验证
$$
E^g_{\mu\nu}
=\left(R_{\mu\nu}+2\nabla_\mu\nabla_\nu\Phi
-\frac14H_{\mu\rho\sigma}H_\nu{}^{\rho\sigma}\right)
-\frac12g_{\mu\nu}E^\Phi.
$$
故在 $E^\Phi=0$ 后，metric equation 等价于
$$
R_{\mu\nu}+2\nabla_\mu\nabla_\nu\Phi
-\frac14H_{\mu\rho\sigma}H_\nu^{\ \rho\sigma}=0.
$$
这正是 beta function 消失条件。若不同时使用 dilaton equation，把 metric Euler
tensor 直接等同于 $\beta^g/\alpha'$ 并不正确。$\square$

**命题 11.4A（dilaton equation）.** 对 string-frame NS-NS action 的 dilaton 变分给出
$$
4\nabla^2\Phi-4|\nabla\Phi|^2
+R-\frac1{12}H^2+O(\alpha')=0
$$
在临界维数且最低阶近似下成立。

**推导说明（标准物理口径）.** 对 $e^{-2\Phi}$ 变分给出 $-2$ 倍括号内 Lagrangian；对 $4|\nabla\Phi|^2$ 变分并分部积分给出 $8\nabla^2\Phi-8|\nabla\Phi|^2$ 的组合。整理后得到所列方程。$\square$

## 11.3 String frame 与 Einstein frame

**定义 11.5（Einstein frame）.** Einstein-frame metric 定义为
$$
g^E_{\mu\nu}=e^{-\frac{4\Phi}{D-2}}g^S_{\mu\nu}.
$$
在该 frame 中，Einstein-Hilbert 项不再带整体 $e^{-2\Phi}$。

**命题 11.6（frame transformation 的作用）.** String frame 中 fundamental string 直接耦合于 $g^S$；Einstein frame 中 gravitational kinetic term 规范化为标准 Einstein-Hilbert 形式。两者由局部场重定义相连，因此 on-shell S-matrix 不依赖 frame choice。

**证明.** 写
$$
g^S_{\mu\nu}=e^{2\omega}g^E_{\mu\nu},
\qquad \omega=\frac{2\Phi}{D-2}.
$$
则
$$
\sqrt{-g_S}=e^{D\omega}\sqrt{-g_E},
$$
$$
R_S=e^{-2\omega}\left[
R_E-2(D-1)\nabla_E^2\omega
-(D-1)(D-2)|\nabla_E\omega|^2
\right].
$$
Einstein term 的总指数为
$$
D\omega-2\Phi-2\omega
=(D-2)\frac{2\Phi}{D-2}-2\Phi=0,
$$
所以 $R_E$ 前不再有 dilaton。丢弃 $\nabla_E^2\omega$ 边界项后，Ricci rescaling
贡献 $-4(D-1)/(D-2)|\nabla\Phi|^2$，原 kinetic term 贡献
$+4|\nabla\Phi|^2$，合计
$$
-\frac4{D-2}|\nabla_E\Phi|^2.
$$
这证明 frame 变换和 canonical rescaling 所需系数。关于 on-shell S-matrix 的最后
一句另依赖下述 equivalence theorem。$\square$

**定义 11.6A（canonical dilaton）.** 在 Einstein frame 中，dilaton kinetic term 可通过线性重标定写成 canonical scalar kinetic term。具体归一化依赖 $D$，但所有 frame-invariant statements 必须用物理 S-matrix、charges 或无量纲耦合表达。

**外部输入定理 11.6B（微扰 equivalence theorem）.** 对局部、可逆且 Jacobian
可由局部 counterterms 处理的场重定义，使用同一渐近态 normalization 与
renormalization prescription 时，微扰 on-shell S-matrix 不变。该结论不声称两个
off-shell actions、effective potentials 或截断后 Green functions 逐项相同。

## 11.4 Type II low-energy actions

**定义 11.7（type II bosonic pseudo-action 的结构）.** 记
$|F_p|^2=(1/p!)F_{\mu_1\ldots\mu_p}F^{\mu_1\ldots\mu_p}$。Type II
supergravity 的 bosonic democratic pseudo-action 在 string frame 中具有结构
$$
S_{\mathrm{II}}
=\frac1{2\kappa_0^2}
\int d^{10}x\sqrt{-g}
\left[
e^{-2\Phi}\left(
R+4|\nabla\Phi|^2-\frac12|H_3|^2
\right)
-\frac14\sum_p |F_p|^2
\right]
+S_{\mathrm{CS}}
+S_{\mathrm{h.d.}}.
$$
IIA 中 $p$ 为偶数 field strengths，IIB 中 $p$ 为奇数 field strengths。Democratic
写法在变分后还须施加 R-R duality constraints；特别是 IIB 有 $F_5=*F_5$。因此该式
不是把 self-duality 直接由通常 covariant action 变分出来的完整作用量。系数
$-1/4$ 补偿同时写入 $F_p$ 与其 Hodge dual 的重复；只写 independent half 的
non-democratic action 对每项使用 $-1/2|F_p|^2$。$S_{\mathrm{h.d.}}$ 是
$\alpha'$ higher-derivative 与 loop corrections 的占位符，不是已知闭式。

**注 11.8（democratic formulation）.** 民主形式同时写入某个 R-R field strength 及其 Hodge dual，再通过 duality constraint 去掉重复自由度。这是 action-level covariance 与自由度计数之间的技术折中。

**命题 11.8A（R-R kinetic term 与 dilaton）.** 在 string frame 中，R-R kinetic terms 不带整体 $e^{-2\Phi}$，而 NS-NS kinetic terms 带该因子。

**推导说明（标准物理口径）.** 该结构可由 sphere scattering amplitude 的 dilaton dependence 与 RNS sector 区分读出：NS-NS fields 属于 worldsheet sigma model 背景耦合，tree-level action 带 $e^{-2\Phi}$；R-R fields 的规范化在 type II supergravity 中按 democratic field strengths 写成不带该整体因子的 kinetic terms。$\square$

## 11.5 Heterotic low-energy action

**定义 11.9（heterotic tree-level action）.** Heterotic string 的最低阶 string-frame action 含有
$$
S_{\mathrm{het}}
=\frac1{2\kappa_0^2}
\int d^{10}x\sqrt{-g}\,e^{-2\Phi}
\left[
R+4|\nabla\Phi|^2-\frac1{12}H_{\mu\nu\rho}H^{\mu\nu\rho}
-\frac{\alpha'}4\operatorname{Tr}(F_{\mu\nu}F^{\mu\nu})
+\frac{\alpha'}4\operatorname{tr}(R_{+\mu\nu}R_+^{\mu\nu})
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

**命题 11.12（supergravity 的受控参数区）.** 对常 dilaton、无 string-scale
小圈且只激发 massless modes 的平滑背景，classical supergravity 的充分参数条件为
$$
g_s\ll 1,\qquad
|\alpha'\mathcal R|\ll 1,
\qquad |\alpha'F|\ll1,
\qquad E^2\alpha'\ll1,
$$
并要求 $H$、dilaton 与其他 fields 的无量纲梯度不变量同样远小于 $1$。若 dilaton
变化，应逐点使用 $e^{\Phi(x)}\ll1$；若存在小 cycles、强 redshift 或额外轻态，则
还需把 winding/KK/brane modes 纳入有效理论。

**推导说明（标准物理口径）.** $g_s\ll1$ 抑制高 genus string loops；所有
$\alpha'$ 加权曲率、场强和梯度不变量小，才共同抑制 higher-derivative terms；
$E^2\alpha'\ll1$ 允许积分掉 massive oscillator modes。命题给的是常用充分控制区，
不是“当且仅当”：supersymmetry、duality 或 exact CFT 有时可控制超出该区域的量，
而上述小参数若缺少一致 truncation 也未必足够。$\square$

## 11.7 与散射振幅的匹配

**外部输入定理 11.13（type-II sphere 四-graviton 振幅）.** 在十维平坦背景、
on-shell transverse traceless 外态和约定 6.0 的某一完整 normalization 下，type-II
sphere 四-graviton reduced amplitude 是一个 kinematic tensor $\mathcal K$ 乘以
$$
\frac1{stu}\,\mathcal G(s,t,u),
\qquad
\mathcal G(s,t,u)
=\prod_{x\in\{s,t,u\}}
\frac{\Gamma(1-\alpha'x/4)}{\Gamma(1+\alpha'x/4)},
\qquad s+t+u=0.
$$
本书不重做 RNS spin-field/picture 与 polarization tensor 的 sphere correlator；只在
下述计算中展开其精确 Gamma factor。

**计算 11.13A（低能 Gamma 展开与 $R^4$ 阶）.** 令 $z_x=\alpha'x/4$。在
$|z_x|<1$ 时，收敛 Taylor series
$$
\log\frac{\Gamma(1-z)}{\Gamma(1+z)}
=2\gamma z
+2\sum_{m=1}^{\infty}
\frac{\zeta(2m+1)}{2m+1}z^{2m+1}
$$
给出
$$
\log\mathcal G
=\frac{2\zeta(3)}3\sum_xz_x^3+O(z_x^5).
$$
由 $z_s+z_t+z_u=0$ 得
$z_s^3+z_t^3+z_u^3=3z_sz_tz_u$，所以
$$
\mathcal G
=1+2\zeta(3)\left(\frac{\alpha'}4\right)^3stu
+O((\alpha'E^2)^5).
$$
于是 $\mathcal G/(stu)$ 的首项是 massless exchange pole，首个解析 contact term
相对两导数 supergravity 出现在 $\alpha'^3R^4$ 阶。这里使用的是 type-II
四-graviton Gamma factor；第六章带平移后 Gamma arguments 的 bosonic tachyon
Virasoro--Shapiro amplitude 不能替代它。把该 scalar expansion 与完整
$t_8t_8R^4$ tensor 及其绝对系数匹配，仍属于外部输入 11.13 的边界。$\square$

## 本章小结

低能有效作用是 worldsheet conformal invariance、massless spectrum 和 string perturbation 的共同结果。String frame 直接反映世界面耦合；Einstein frame 适合讨论引力动力学。Supergravity 不是完整 string theory，而是 $g_s$ 和 $\alpha'$ 双重展开中的最低阶。

## 练习

**练习 11.1.** 说明 string frame 与 Einstein frame 的区别。

**练习 11.2.** 从 $S_{\mathrm{NS}}$ 对 $B$ 变分，推出 $d(e^{-2\Phi}*H)=0$。

**练习 11.3.** 从 string-frame NS-NS action 对 dilaton 变分，推导最低阶 dilaton equation。
