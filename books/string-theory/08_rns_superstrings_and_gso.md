# 第八章：RNS 超弦和 GSO 投影

## 本章目标

本章建立 RNS formalism 的主线：

1. 在世界面上加入 Majorana fermions $\psi^\mu$；
2. 构造 $N=1$ superconformal matter CFT；
3. 量子化 NS/R sectors，并写出超弦质量公式；
4. 由 anomaly cancellation 得到临界维数 $D=10$；
5. 通过 GSO projection 移除 tachyon，并区分 type IIA 与 type IIB。

本章采用平坦 target background。Curved background 中的 supersymmetric sigma model 与 beta function 条件放在第十一章和第十三章。

## 依赖前置知识

需要第三章 CFT、第四章量子化、第五章 BRST，以及附录 E 的 Clifford algebra 与 spinor convention。

## 8.1 RNS worldsheet action

**定义 8.1（RNS matter action）.** 在 conformal gauge 和 Euclidean complex coordinate 中，平坦背景 RNS matter theory 由 $D$ 个 free bosons $X^\mu$ 与 $D$ 个 worldsheet fermions $\psi^\mu,\tilde\psi^\mu$ 组成。其局部作用量归一化选为使 OPE 满足
$$
X^\mu(z,\bar z)X^\nu(w,\bar w)
\sim
-\frac{\alpha'}2\eta^{\mu\nu}\log|z-w|^2,
$$
以及
$$
\psi^\mu(z)\psi^\nu(w)\sim\frac{\eta^{\mu\nu}}{z-w},
\qquad
\tilde\psi^\mu(\bar z)\tilde\psi^\nu(\bar w)\sim
\frac{\eta^{\mu\nu}}{\bar z-\bar w}.
$$

**定义 8.2（stress tensor 与 supercurrent）.** Holomorphic matter stress tensor 和 supercurrent 取
$$
T_m(z)=
-\frac1{\alpha'}:\partial X^\mu\partial X_\mu:
-\frac12:\psi^\mu\partial\psi_\mu:,
$$
$$
G_m(z)=
i\sqrt{\frac2{\alpha'}}:\psi^\mu\partial X_\mu:.
$$
Antiholomorphic 部分同理。

**命题 8.3（matter central charge）.** RNS matter CFT 的 holomorphic central charge 为
$$
c_m=D+\frac D2=\frac{3D}{2}.
$$

**证明.** 每个 free boson 贡献 $c=1$，每个 Majorana fermion 贡献 $c=1/2$。两者相加即得。$\square$

**命题 8.4（$N=1$ superconformal OPE）.** 定义 8.2 的 $T_m,G_m$ 满足
$$
T_m(z)T_m(w)
\sim
\frac{c_m/2}{(z-w)^4}
+\frac{2T_m(w)}{(z-w)^2}
+\frac{\partial T_m(w)}{z-w},
$$
$$
T_m(z)G_m(w)
\sim
\frac{3G_m(w)/2}{(z-w)^2}
+\frac{\partial G_m(w)}{z-w},
$$
$$
G_m(z)G_m(w)
\sim
\frac{2c_m/3}{(z-w)^3}
+\frac{2T_m(w)}{z-w}.
$$

**证明草图.** 用定义 8.1 的 free-field OPE 和 Wick theorem 逐项收缩。$G_mG_m$ 的三阶极点来自 $\psi^\mu\psi^\nu$ 与 $\partial X_\mu\partial X_\nu$ 的双收缩；一阶极点给出 $2T_m$。$\square$

## 8.2 NS/R sectors 和 super-Virasoro algebra

**定义 8.5（NS 与 R boundary condition）.** 闭弦 holomorphic fermion 在绕空间圈时可取
$$
\psi^\mu(\sigma+2\pi)= -\psi^\mu(\sigma)
$$
或
$$
\psi^\mu(\sigma+2\pi)=+\psi^\mu(\sigma).
$$
前者称 Neveu-Schwarz sector，后者称 Ramond sector。NS sector 的模为半整数 $r\in\mathbb Z+\frac12$，R sector 的模为整数 $r\in\mathbb Z$：
$$
\psi^\mu(z)=\sum_r\frac{\psi_r^\mu}{z^{r+1/2}}.
$$

**命题 8.6（fermion oscillator algebra）.** RNS fermion modes 满足
$$
\{\psi_r^\mu,\psi_s^\nu\}
=\eta^{\mu\nu}\delta_{r+s,0}.
$$

**证明.** 由 OPE $\psi^\mu(z)\psi^\nu(w)\sim\eta^{\mu\nu}/(z-w)$ 和 contour integral mode extraction 得到。$\square$

**定义 8.7（super-Virasoro modes）.** 令
$$
T(z)=\sum_{n\in\mathbb Z}L_nz^{-n-2},
\qquad
G(z)=\sum_rG_rz^{-r-3/2}.
$$
其中 NS sector 取 $r\in\mathbb Z+\frac12$，R sector 取 $r\in\mathbb Z$。

**定理 8.8（super-Virasoro algebra）.** Matter modes 满足
$$
[L_m,L_n]=(m-n)L_{m+n}
+\frac c{12}m(m^2-1)\delta_{m+n,0},
$$
$$
[L_m,G_r]=\left(\frac m2-r\right)G_{m+r},
$$
$$
\{G_r,G_s\}
=2L_{r+s}
+\frac c3\left(r^2-\frac14\right)\delta_{r+s,0}.
$$

**证明草图.** 对命题 8.4 的 OPE 分别取 Laurent coefficients。$\square$

## 8.3 Superconformal ghosts 和临界维数

Gauge fixing RNS worldsheet supergravity 不仅产生 reparametrization ghosts $b,c$，还产生 superconformal ghosts $\beta,\gamma$，其 conformal weights 为
$$
h_\beta=\frac32,\qquad h_\gamma=-\frac12.
$$

**命题 8.9（RNS ghost central charge）.** $bc$ ghosts 的 central charge 为 $-26$，$\beta\gamma$ ghosts 的 central charge 为 $11$，故 ghost sector 总 central charge 为
$$
c_{\mathrm{gh}}=-15.
$$

**证明草图.** $bc$ 系统是权重 $(2,-1)$ 的反交换 ghost，第三章已给出 $c_{bc}=-26$。$\beta\gamma$ 是权重 $(3/2,-1/2)$ 的交换 ghost；代入一阶系统 central charge 公式得到 $c_{\beta\gamma}=11$。$\square$

**定理 8.10（RNS 临界维数）.** 平坦背景 RNS string 的 superconformal anomaly cancellation 要求
$$
c_m+c_{\mathrm{gh}}=0,
$$
即
$$
\frac{3D}{2}-15=0.
$$
因此临界维数为
$$
D=10.
$$

**证明.** 由命题 8.3 和命题 8.9 直接计算。$\square$

## 8.4 RNS 质量公式与零模

**定义 8.11（RNS number operators）.** 开弦 RNS Fock space 中
$$
N=N_\alpha+N_\psi,
$$
其中 bosonic oscillators 与 fermionic oscillators 的贡献分别按其模数加权。NS sector 的 normal ordering constant 为
$$
a_{\mathrm{NS}}=\frac12,
$$
R sector 的 normal ordering constant 为
$$
a_{\mathrm R}=0.
$$

**命题 8.12（开超弦质量公式）.** 开 RNS string 满足
$$
\alpha'M^2=N-a,
$$
其中
$$
a=
\begin{cases}
\frac12,& \mathrm{NS},\\
0,& \mathrm{R}.
\end{cases}
$$

**证明草图.** 与第四章开弦推导相同，但 $L_0$ 中加入 fermion oscillator number 和相应 normal ordering constant。NS 与 R 的 $a$ 值由半整数或整数 fermion 零点能给出。$\square$

**例 8.13（NS sector）.** NS ground state 有
$$
N=0,\qquad M^2=-\frac1{2\alpha'},
$$
是 tachyon。第一激发
$$
\psi^\mu_{-1/2}|0;k\rangle_{\mathrm{NS}}
$$
有 $N=1/2$，因此 massless，并对应 gauge vector。

**例 8.14（R sector）.** R sector 有 fermion zero modes
$$
\{\psi_0^\mu,\psi_0^\nu\}=\eta^{\mu\nu}.
$$
它们生成 target-space Clifford algebra。因此 R ground states 构成 spacetime spinor 表示，且因 $a_R=0$ 为 massless。

## 8.5 GSO projection

**定义 8.15（worldsheet fermion number）.** $(-1)^F$ 表示 worldsheet fermion number parity。GSO projection 是在 NS/R sectors 上选取 $(-1)^F$ 的某一特征子空间：
$$
P_{\mathrm{GSO}}=\frac12(1\pm(-1)^F),
$$
并在闭弦中分别对 left-moving 与 right-moving sectors 选择投影。

**定理 8.16（GSO projection 的作用）.** 适当选择 GSO projection 后：

1. NS tachyon 被移除；
2. NS massless vector 或 NS-NS massless tensor 被保留；
3. R sector ground states 具有确定 chirality；
4. 闭弦谱可组织为 spacetime supersymmetry multiplets；
5. genus-one partition function 的 spin-structure 求和满足 modular invariance。

**证明草图.** 前三点由 $(-1)^F$ 对 NS ground state、NS 第一激发和 R ground state 的作用直接判定。第四点由 NS 与 R sectors 的剩余自由度匹配给出。第五点需要 theta functions 与 spin structures 的 modular transformation，属于附录 D 和第十五章的外部输入。$\square$

**定义 8.17（type IIA 与 type IIB）.** Type II closed superstrings 由左右两个 RNS sectors 组合而成：

- Type IIA：left 与 right Ramond ground states chirality 相反。
- Type IIB：left 与 right Ramond ground states chirality 相同。

**命题 8.18（type II massless sectors）.** Type II 闭弦 massless spectrum 分为
$$
\mathrm{NS\text{-}NS}\oplus \mathrm{NS\text{-}R}\oplus
\mathrm{R\text{-}NS}\oplus \mathrm{R\text{-}R}.
$$
NS-NS 部分包含 graviton、$B$-field 和 dilaton；NS-R 与 R-NS 部分包含 gravitini 与 dilatini；R-R 部分给出 differential form gauge potentials，其 chirality pattern 区分 IIA 与 IIB。

**证明草图.** 左右 sectors 的 massless states 张量相乘。NS-NS 与玻色闭弦第一激发的张量分解相同。R sector ground states 是十维 spinors；spinor bilinear 按 Clifford algebra 分解为 antisymmetric form representations。IIA/IIB 的 chirality 决定允许的 form degree。$\square$

## 8.6 Picture number 的接口

Superconformal ghost 系统有 picture number。实际 amplitude 计算中，NS 顶点常写在 $-1$ picture，R 顶点常写在 $-1/2$ picture，并用 picture-changing operator 调整总 picture anomaly。

**定义 8.19（bosonized superghost）.** Superghost system 可局部 bosonize 为
$$
\beta=e^{-\varphi}\partial\xi,\qquad
\gamma=\eta e^\varphi.
$$
算子 $e^{q\varphi}$ 的 conformal weight 为
$$
h(e^{q\varphi})=-\frac12q(q+2).
$$

**定义 8.20（picture-changing operator）.** Picture-changing operator 定义为
$$
X_{\mathrm{PCO}}(z)=\{Q_B,\xi(z)\}.
$$
它把 vertex operator 的 picture number 增加 $1$，并在 BRST cohomology 中给出等价代表，前提是不存在碰撞奇点或 supermoduli 边界异常。

**例 8.21（open-string NS vector vertices）.** Massless open-string vector 在 $-1$ picture 中可写为
$$
V_A^{(-1)}=c\,e^{-\varphi}\zeta_\mu\psi^\mu e^{ik\cdot X},
$$
在 $0$ picture 中可写为
$$
V_A^{(0)}
=c\,\zeta_\mu
\left(\partial_tX^\mu+i\alpha' k\cdot\psi\,\psi^\mu\right)
e^{ik\cdot X}
$$
至 convention-dependent numerical factors。二者由 picture changing 相关。

## 8.7 Spin fields

**定义 8.22（spin field）.** R sector vertex operators 含有 spin field $S_\alpha$，它实现 worldsheet fermions 的 branch cut：
$$
\psi^\mu(z)S_\alpha(w)
\sim
\frac1{(z-w)^{1/2}}(\Gamma^\mu)_\alpha^{\ \beta}S_\beta(w)+\cdots.
$$

**命题 8.23（十维 spin field dimension）.** 对十个 free Majorana fermions，spin field 的 conformal weight 为
$$
h(S_\alpha)=\frac{10}{16}=\frac58.
$$
因此 massless R vertex 的 matter-spin-superghost 部分
$$
e^{-\varphi/2}S_\alpha e^{ik\cdot X}
$$
在 $k^2=0$ 时具有 weight $1$。

**证明草图.** 两个 real fermions 可 bosonize 为一个 free boson $H_I$，spin field 是
$$
\exp\left(\frac i2\sum_{I=1}^5 s_IH_I\right),
\qquad s_I=\pm1.
$$
每个因子贡献 $1/8$，五个因子合计 $5/8$。此外
$$
h(e^{-\varphi/2})=\frac38,
$$
故总 weight 为 $1$。$\square$

## 本章小结

RNS formalism 把 string worldsheet theory 扩展为 $N=1$ superconformal field theory。Matter central charge 为 $3D/2$，ghost central charge 为 $-15$，故临界维数为 $10$。NS/R sectors 分别给出 spacetime bosonic 与 fermionic states；GSO projection 移除 tachyon，并产生 type II superstring 的 spacetime supersymmetry。

## 练习

**练习 8.1.** 由 fermion OPE 推导 $\{\psi_r^\mu,\psi_s^\nu\}=\eta^{\mu\nu}\delta_{r+s,0}$。

**练习 8.2.** 用 central charge 计数推导 RNS 临界维数。

**练习 8.3.** 说明为什么 R sector ground states 构成 spacetime Clifford algebra 表示。

**练习 8.4.** 用 bosonization 计算十维 RNS spin field 的 conformal weight。

