# 第六章：顶点算子和弦散射振幅

## 本章目标

本章建立三件事：

1. target-space 粒子态怎样表示为 worldsheet vertex operators；
2. on-shell 条件怎样由 conformal weight 或 BRST closedness 给出；
3. tree-level scattering amplitude 怎样化为 CFT correlator 在 puncture moduli 上的积分。

本章只处理平坦背景、tree-level、最低若干态的公式。高 genus measure 和 modular invariance 放在第十五章。

## 依赖前置知识

需要第三章 CFT、第四章谱公式和第五章 BRST 量子化。OPE 归一化固定为
$$
X^\mu(z,\bar z)X^\nu(w,\bar w)
\sim
-\frac{\alpha'}2\eta^{\mu\nu}\log|z-w|^2.
$$

## 6.1 状态、顶点算子和 ghost number

**定义 6.1（matter vertex 与完整顶点）.** 令 $V(z,\bar z)$ 是 matter CFT 中 conformal weight 为 $(h,\bar h)$ 的局部算子。

1. 闭弦未积分顶点算子取
   $$
   U(z,\bar z)=c(z)\bar c(\bar z)V(z,\bar z).
   $$
2. 闭弦积分顶点算子取
   $$
   \mathcal V=\int_\Sigma d^2z\,V(z,\bar z).
   $$
3. 开弦边界上的未积分顶点取
   $$
   U(x)=c(x)V(x),
   $$
   积分顶点取
   $$
   \mathcal V=\int_{\partial\Sigma} dx\,V(x).
   $$

未积分顶点带有 ghost 插入，用来吸收 conformal Killing group 的 ghost zero modes；积分顶点不带 $c$ ghost，并在剩余 puncture 位置上积分。

**命题 6.2（integrated vertex 的权重条件）.** 闭弦积分顶点 $\int d^2z\,V$ 在 conformal gauge 中为坐标无关插入的必要条件是 $V$ 的 matter weights 为 $(1,1)$。开弦边界积分顶点 $\int dx\,V$ 的必要条件是边界 conformal weight 为 $1$。

**证明.** 闭弦局部坐标变换 $z\mapsto f(z)$ 下 $d^2z$ 的权重为 $(-1,-1)$，因此 $V$ 必须有权重 $(1,1)$ 才能使 $d^2z\,V$ 不依赖局部坐标。边界情形中 $dx$ 的权重为 $-1$，所以 $V$ 的边界权重必须为 $1$。$\square$

**命题 6.3（未积分顶点与 BRST closedness）.** 若 $V$ 是 matter primary 且权重为 $(1,1)$，则闭弦算子 $c\bar c V$ 是 BRST closed，忽略 total derivatives 后与积分顶点对应。同理，开弦 $cV$ 对应边界积分顶点。

**证明草图.** BRST current 含有 $cT_{\mathrm{tot}}$ 以及 ghost self-interaction。对 primary $V$ 计算 $Q_B$ 与 $c\bar cV$ 的 OPE，二阶和一阶极点中与 conformal transformation 有关的项在 $h=\bar h=1$ 时相互抵消，剩余项为 total derivative 或 ghost zero mode 约束。完整证明需要第五章的 BRST current 归一化。$\square$

## 6.2 Tachyon、vector 与 graviton 顶点

**例 6.4（闭弦 tachyon）.** 闭弦 tachyon 的 matter vertex 为
$$
V_k(z,\bar z)=:e^{ik\cdot X(z,\bar z)}:.
$$
由第三章的 stress tensor OPE 得
$$
h=\bar h=\frac{\alpha' k^2}{4}.
$$
因此 $(1,1)$ 条件等价于
$$
k^2=\frac4{\alpha'},\qquad M^2=-k^2=-\frac4{\alpha'}.
$$
这与第四章闭弦 $N=\tilde N=0$、$a=1$ 的质量公式一致。

**例 6.5（开弦 tachyon）.** 开弦边界 tachyon vertex 为
$$
V_k(x)=:e^{ik\cdot X(x)}:.
$$
用 doubling trick 得其边界 conformal weight 为 $\alpha'k^2$。因此
$$
\alpha'k^2=1,\qquad M^2=-\frac1{\alpha'}.
$$

**命题 6.6（开弦 massless vector 顶点）.** 开弦 massless vector 的积分顶点可写为
$$
V_A(x)=\zeta_\mu\,\partial_tX^\mu(x)e^{ik\cdot X(x)}.
$$
BRST closedness 给出
$$
k^2=0,\qquad k\cdot \zeta=0,
$$
并且
$$
\zeta_\mu\sim \zeta_\mu+\lambda k_\mu.
$$

**证明草图.** $\partial_tX^\mu$ 的边界 weight 为 $1$，指数部分 weight 为 $\alpha'k^2$，所以边界总 weight 为 $1$ 要求 $k^2=0$。$k\cdot\zeta=0$ 来自 $T$ 与顶点的三阶极点消失，等价于旧协变量子化中的 $L_1|\psi\rangle=0$。极化平移 $\zeta\mapsto\zeta+\lambda k$ 使顶点变为边界全导数或 BRST exact 插入，因此不改变散射振幅。$\square$

**命题 6.7（闭弦 massless 顶点）.** 闭弦第一激发层的积分顶点可写为
$$
V_\epsilon(z,\bar z)
=
\epsilon_{\mu\nu}\,\partial X^\mu(z)\bar\partial X^\nu(\bar z)
e^{ik\cdot X(z,\bar z)}.
$$
BRST closedness 给出
$$
k^2=0,\qquad k^\mu\epsilon_{\mu\nu}=0,\qquad k^\nu\epsilon_{\mu\nu}=0,
$$
并且
$$
\epsilon_{\mu\nu}\sim
\epsilon_{\mu\nu}+k_\mu\xi_\nu+k_\nu\tilde\xi_\mu.
$$
对 $\epsilon_{\mu\nu}$ 分解为对称无迹、反对称和迹部分，分别得到 graviton、Kalb-Ramond field 和 dilaton。

**证明草图.** 该顶点的 matter weights 为
$$
\left(1+\frac{\alpha'k^2}{4},\,1+\frac{\alpha'k^2}{4}\right),
$$
故 $(1,1)$ 条件给出 $k^2=0$。横向条件来自 $L_1,\tilde L_1$ 或等价的 BRST closedness；极化等价来自 BRST exact states。$\square$

## 6.3 Koba-Nielsen 因子

**命题 6.8（闭弦指数算子 correlator）.** 在 genus zero 平面图上，
$$
\left\langle\prod_{i=1}^n:e^{ik_i\cdot X(z_i,\bar z_i)}:\right\rangle
=
(2\pi)^D\delta^{(D)}\!\left(\sum_i k_i\right)
\prod_{i<j}|z_i-z_j|^{\alpha' k_i\cdot k_j}.
$$

**证明.** 零模积分给出动量守恒 delta function。非零模部分由 Wick theorem 给出
$$
\exp\left(
-\sum_{i<j}k_i\cdot k_j
\left\langle X(z_i,\bar z_i)X(z_j,\bar z_j)\right\rangle
\right),
$$
代入 propagator 即得乘积因子。$\square$

**推论 6.9（开弦边界 Koba-Nielsen 因子）.** 对 disk 或 upper half-plane 边界上的开弦指数顶点，
$$
\left\langle\prod_{i=1}^n:e^{ik_i\cdot X(x_i)}:\right\rangle
=
(2\pi)^D\delta^{(D)}\!\left(\sum_i k_i\right)
\prod_{i<j}|x_i-x_j|^{2\alpha' k_i\cdot k_j}.
$$

**证明草图.** 由 doubling trick，把边界场的两点函数变为闭弦 holomorphic contraction 的两倍。$\square$

## 6.4 Tree-level amplitudes

**定义 6.10（sphere 振幅）.** 闭弦 genus-zero $n$ 点振幅写为
$$
\mathcal A^{(0)}_n
=g_s^{\,n-2}
\int_{\mathcal M_{0,n}}
\left\langle
\prod_{a=1}^{3}c\bar cV_a(z_a,\bar z_a)
\prod_{i=4}^{n}\int d^2z_i\,V_i(z_i,\bar z_i)
\right\rangle.
$$
前三个 puncture 的位置由 $PSL_2(\mathbb C)$ 固定。

**定义 6.11（disk ordered open-string 振幅）.** 开弦 disk 上固定 cyclic ordering 后，
$$
\mathcal A^{\mathrm{open}}_n(1,\ldots,n)
=g_o^{\,n-2}
\int_{x_1<\cdots<x_n}
\left\langle
\prod_{a=1}^{3}cV_a(x_a)
\prod_{i=4}^{n}\int dx_i\,V_i(x_i)
\right\rangle,
$$
其中三个边界点由 $PSL_2(\mathbb R)$ 固定。完整开弦振幅还需对非等价 cyclic ordering 求和并附加 Chan-Paton trace。

**命题 6.12（Veneziano amplitude）.** 四个开弦 tachyon 的 ordered tree amplitude 与 beta function 成正比：
$$
\mathcal A_4^{\mathrm{open}}(1,2,3,4)
\propto
g_o^2\,
B(-1-\alpha's,-1-\alpha't),
$$
即
$$
\mathcal A_4^{\mathrm{open}}
\propto
g_o^2
\frac{\Gamma(-1-\alpha's)\Gamma(-1-\alpha't)}
{\Gamma(-2-\alpha'(s+t))}.
$$
这里
$$
s=-(k_1+k_2)^2,\qquad t=-(k_2+k_3)^2,
$$
且外线满足 $k_i^2=1/\alpha'$。

**证明.** 固定 $x_1=0,x_3=1,x_4=\infty$，剩余 $x=x_2\in(0,1)$。由推论 6.9 得
$$
\mathcal A_4^{\mathrm{open}}
\propto
\int_0^1 dx\,
x^{2\alpha'k_1\cdot k_2}
(1-x)^{2\alpha'k_2\cdot k_3}.
$$
利用 $k_i^2=1/\alpha'$ 可化为
$$
2\alpha'k_1\cdot k_2=-\alpha's-2,\qquad
2\alpha'k_2\cdot k_3=-\alpha't-2.
$$
因此积分为 $B(-1-\alpha's,-1-\alpha't)$。$\square$

**命题 6.13（Virasoro-Shapiro amplitude）.** 四个闭弦 tachyon 的 sphere amplitude 与
$$
\mathcal A_4^{\mathrm{closed}}
\propto
g_s^2\,
\frac{
\Gamma(-1-\alpha's/4)
\Gamma(-1-\alpha't/4)
\Gamma(-1-\alpha'u/4)}
{
\Gamma(2+\alpha's/4)
\Gamma(2+\alpha't/4)
\Gamma(2+\alpha'u/4)}
$$
成正比，其中 $s,t,u$ 是 Mandelstam variables，外线满足 $k_i^2=4/\alpha'$。

**证明草图.** 固定三个 punctures 为 $0,1,\infty$，剩余积分为
$$
\int_{\mathbb C}d^2z\,|z|^{\alpha'k_1\cdot k_2}
|1-z|^{\alpha'k_2\cdot k_3}.
$$
该复 beta integral 经解析延拓给出上式。$\square$

## 6.5 Pole、因子化和 string spectrum

**命题 6.14（Veneziano amplitude 的谱极点）.** Ordered open-string 四点 tachyon 振幅在
$$
\alpha's=N-1,\qquad N=0,1,2,\ldots
$$
处有 simple poles。这些 poles 对应开弦第 $N$ 激发层的中间态。

**证明.** Gamma function $\Gamma(-1-\alpha's)$ 在 $-1-\alpha's=-m$、$m\in\mathbb Z_{\ge0}$ 处有 simple pole，即 $\alpha's=m-1$。令 $N=m$，得到极点位置。与第四章开弦质量公式 $M_N^2=(N-1)/\alpha'$ 一致。$\square$

**注 6.15（unitarity 的接口）.** Tree-level amplitude 的 residue 必须分解为两个三点振幅之和，这表达 perturbative unitarity。完整证明需要物理态完备性、ghost decoupling 和 no-ghost theorem；本书把该部分作为第四、五章外部输入的应用。

**命题 6.16（tree factorization）.** 当 punctured sphere 或 disk 退化为两片曲面由长管连接时，tree-level string amplitude 在相应 channel 上因子化为
$$
\mathcal A_n\sim
\sum_{\alpha}
\mathcal A_L(\cdots,\alpha)
\frac1{k^2+M_\alpha^2}
\mathcal A_R(\alpha,\cdots),
$$
其中 $\alpha$ 遍历中间 string spectrum。

**证明草图.** 退化参数 $q$ 描述长管传播。CFT sewing 给出传播因子
$$
q^{L_0-a}\bar q^{\tilde L_0-a}
$$
和一组完备中间态。对 $q$ 的径向积分在 $L_0-a=0$ 即 target-space on-shell 条件处产生 pole，residue 为左右两个低点振幅乘积。$\square$

**定义 6.17（一圈振幅接口）.** 闭弦一圈 $n$ 点振幅具有形式
$$
\mathcal A_{1,n}
=g_s^n\int_{\mathcal M_{1,n}}
\left\langle
\prod_a (b,\mu_a)(\bar b,\bar\mu_a)
\prod_i V_i
\right\rangle_{\Sigma_1}.
$$
这里 $\mu_a$ 是 Beltrami differentials。完整 superstring 一圈振幅还需要 spin structure sum 和 picture number 规则。

**注 6.18（unitarity 与 modular invariance）.** Loop amplitude 的一致性同时依赖 degeneration factorization 和 modular invariance。前者控制 unitarity，后者避免同一 worldsheet complex structure 被重复计数。

## 本章小结

顶点算子把 BRST cohomology 类翻译为 worldsheet 插入。Tree-level 振幅由 CFT correlator、ghost zero mode 吸收和 puncture moduli 积分组成。Veneziano 与 Virasoro-Shapiro 振幅的极点精确再现 string spectrum，这是 string theory 作为散射理论的基本一致性检验。

## 练习

**练习 6.1.** 用第三章 OPE 推导闭弦 tachyon 的 conformal weights，并验证其质量公式。

**练习 6.2.** 推导开弦边界指数算子的 Koba-Nielsen 因子。

**练习 6.3.** 从 Veneziano amplitude 的 gamma function 表达式读出开弦第 $N$ 层的 pole 位置。

**练习 6.4.** 解释 worldsheet degeneration 如何给出 target-space propagator pole。

