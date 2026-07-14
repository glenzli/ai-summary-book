# 第六章：顶点算子和弦散射振幅

量子态若只停留在 Fock space 中，还不能参与可观测的散射过程。径向量子化把一个
渐近态放到 puncture 处，state--operator correspondence 将它变成局部顶点算子；
算子的 conformal weight 重现质量壳条件，BRST closedness 施加横向约束，而 exact
方向实现 gauge equivalence。把若干 punctures 放在 sphere 或 disk 上并除去残余
conformal Killing group，CFT correlator 才成为 target-space 的 tree amplitude。
以下使用第三章 OPE、第四章谱公式和第五章 BRST complex，在平坦背景中计算最低若干
外态；高 genus measure 与 modular invariance 留到第十五章。统一 OPE 归一化为
$$
X^\mu(z,\bar z)X^\nu(w,\bar w)
\sim
-\frac{\alpha'}2\eta^{\mu\nu}\log|z-w|^2.
$$

**约定 6.0（振幅的归一化边界）.** 本章所有外动量均取 incoming。完整连通
$S$-matrix element 与去掉动量 delta function 的 reduced amplitude 分别记为
$$
\mathscr A_n
=i(2\pi)^D\delta^{(D)}\!\left(\sum_i k_i\right)\mathcal M_n,
\qquad \mathcal M_n=\text{reduced amplitude}.
$$
记 $\langle\cdots\rangle'$ 为已除去 $X$ 零模 delta function 的 CFT correlator。
本章明确固定 Koba--Nielsen 指数、moduli measure 的形状、pole 位置和 coupling
幂次，但不固定 sphere/disk vacuum normalization、每个外态的 LSZ normalization、
Chan--Paton trace normalization 以及 $g_o^2/g_s$ 的常数。因此写成 $\propto$ 的
Veneziano 与 Virasoro--Shapiro 公式是 reduced kinematic factor，不是完整数值
$S$-matrix element；这些常数只能通过二点函数、因子化和低能作用的共同约定固定。

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

**推导说明（标准物理口径）.** 先看开弦的 holomorphic 部分。由第五章 BRST current 与权重为 $h$ 的 matter primary 的 OPE，围道留数给出
$$
\{Q_B,cV\}=(1-h)c\partial c\,V,
$$
而对不带 ghost 的插入有
$$
[Q_B,V]=c\partial V+h(\partial c)V.
$$
当 $h=1$ 时，前式为零，后式化为全导数
$$
[Q_B,V]=\partial(cV).
$$
因此 $cV$ 是 BRST closed，且沿边界积分后 $\int dx\,V$ 与它通过 descent relation 对应。闭弦情形对 holomorphic 与 antiholomorphic 两部分分别应用同一计算；$h=\bar h=1$ 时 $c\bar cV$ closed，而 $[Q_B,V]$ 是世界面全导数。该结论假定围道移动不产生模空间边界异常。$\square$

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

**推导说明（标准物理口径）.** $\partial_tX^\mu$ 的边界 weight 为 $1$，指数部分 weight 为 $\alpha'k^2$，所以总 weight 为 $1$ 首先要求 $k^2=0$。在
$$
T(z)\,\zeta\cdot\partial X\,e^{ik\cdot X}(w)
$$
中，把 $T$ 的一个 $\partial X$ 与顶点的 $\partial X$ 收缩、另一个与指数收缩，会产生与 $k\cdot\zeta$ 成正比的三阶极点。Primary 条件要求该极点消失，故 $k\cdot\zeta=0$。若 $\zeta_\mu=\lambda k_\mu$，则
$$
k\cdot\partial_tX\,e^{ik\cdot X}
=\frac1i\partial_t e^{ik\cdot X},
$$
是边界全导数；在未积分表述中同一方向是 BRST exact。因此物理极化是横向向量模去 $\zeta\sim\zeta+\lambda k$。$\square$

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

**推导说明（标准物理口径）.** 该顶点的 matter weights 为
$$
\left(1+\frac{\alpha'k^2}{4},\,1+\frac{\alpha'k^2}{4}\right),
$$
故 $(1,1)$ 条件给出 $k^2=0$。横向条件来自 $L_1,\tilde L_1$ 或等价的 BRST closedness；极化等价来自 BRST exact states。$\square$

## 6.3 Koba-Nielsen 因子

**命题 6.8（闭弦指数算子 correlator）.** 在 genus zero 平面图的 free-boson
Gaussian prescription 下，
$$
\left\langle\prod_{i=1}^n:e^{ik_i\cdot X(z_i,\bar z_i)}:\right\rangle
=
(2\pi)^D\delta^{(D)}\!\left(\sum_i k_i\right)
\prod_{i<j}|z_i-z_j|^{\alpha' k_i\cdot k_j}.
$$

**推导说明（标准物理口径）.** 零模 Lebesgue measure 的标准归一化给出动量守恒
delta function。非零模部分由 point-split Wick theorem 给出
$$
\exp\left(
-\sum_{i<j}k_i\cdot k_j
\left\langle X(z_i,\bar z_i)X(z_j,\bar z_j)\right\rangle
\right),
$$
代入 propagator 即得乘积因子。有限 OPE 计算是 operator statement；零模测度与
Gaussian functional integral normalization 是路径积分输入。$\square$

**推论 6.9（开弦边界 Koba-Nielsen 因子）.** 对 disk 或 upper half-plane 边界上的开弦指数顶点，
$$
\left\langle\prod_{i=1}^n:e^{ik_i\cdot X(x_i)}:\right\rangle
=
(2\pi)^D\delta^{(D)}\!\left(\sum_i k_i\right)
\prod_{i<j}|x_i-x_j|^{2\alpha' k_i\cdot k_j}.
$$

**推导说明（标准物理口径）.** 由 doubling trick，把边界场的两点函数变为闭弦 holomorphic contraction 的两倍。$\square$

## 6.4 Tree-level amplitudes

**定义 6.10（sphere reduced functional）.** 闭弦 genus-zero $n$ 点的
gauge-fixed reduced functional 写为
$$
\widehat{\mathcal M}^{(0)}_n
=
\int_{\mathcal M_{0,n}}
\left\langle
\prod_{a=1}^{3}c\bar cV_a(z_a,\bar z_a)
\prod_{i=4}^{n}\int d^2z_i\,V_i(z_i,\bar z_i)
\right\rangle'.
$$
前三个 puncture 的位置由 $PSL_2(\mathbb C)$ 固定。完整 reduced amplitude 为
$\mathcal M_n=C_{S^2}(\prod_i\mathcal N_i)g_s^{n-2}
\widehat{\mathcal M}^{(0)}_n$，其中 $C_{S^2},\mathcal N_i$ 属于约定 6.0 未固定的
真空与外态 normalization。

**定义 6.11（disk ordered reduced functional）.** 开弦 disk 上固定 cyclic ordering 后，
$$
\widehat{\mathcal M}^{\mathrm{open}}_n(1,\ldots,n)
=
\int_{x_1<\cdots<x_n}
\left\langle
\prod_{a=1}^{3}cV_a(x_a)
\prod_{i=4}^{n}\int dx_i\,V_i(x_i)
\right\rangle',
$$
其中三个边界点由 $PSL_2(\mathbb R)$ 固定。完整 ordered reduced amplitude 另乘
$C_D(\prod_i\mathcal N_i)g_o^{n-2}$；完整开弦振幅还需对非等价 cyclic ordering
求和并附加 Chan--Paton trace。

**命题 6.12（Veneziano amplitude）.** 四个开弦 tachyon 的 ordered tree reduced
amplitude 由下式的亚纯延拓给出：
$$
\mathcal M_4^{\mathrm{open}}(1,2,3,4)
\propto
g_o^2\,
B(-1-\alpha's,-1-\alpha't),
$$
即
$$
\mathcal M_4^{\mathrm{open}}
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

**推导说明（标准物理口径）.** 固定 $x_1=0,x_3=1,x_4=\infty$，其中
$V(\infty)=\lim_{x\to\infty}x^{2h}V(x)$，并把 ghost 三点函数吸收到约定 6.0 的
整体常数。剩余 $x=x_2\in(0,1)$。由推论 6.9 得
$$
\mathcal M_4^{\mathrm{open}}
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
令
$$
a=-1-\alpha's,
\qquad b=-1-\alpha't.
$$
仅在 $\operatorname{Re}a>0$ 且 $\operatorname{Re}b>0$ 时，上式才是绝对收敛的
Euler integral，并等于 $B(a,b)=\Gamma(a)\Gamma(b)/\Gamma(a+b)$。物理运动学
通常不在这一收敛域内；tree amplitude 定义为该结果的唯一亚纯延拓，Lorentzian
pole prescription 再由 target-space $i\varepsilon$ 约定指定。故 beta function
计算是有限积分加解析延拓，不是对物理区域中发散积分的逐点等号。$\square$

**命题 6.13（Virasoro-Shapiro amplitude）.** 四个闭弦 tachyon 的 sphere amplitude 与
$$
\mathcal M_4^{\mathrm{closed}}
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

**推导说明（标准物理口径）.** 固定三个 punctures 为 $0,1,\infty$，剩余积分为
$$
\int_{\mathbb C}d^2z\,|z|^{\alpha'k_1\cdot k_2}
|1-z|^{\alpha'k_2\cdot k_3}.
$$
由 $k_i^2=4/\alpha'$ 与 Mandelstam 变量定义，两个指数分别是
$$
-4-\frac{\alpha's}{2},\qquad
-4-\frac{\alpha't}{2}.
$$
当且仅当
$$
\operatorname{Re}a>0,
\qquad \operatorname{Re}b>0,
\qquad \operatorname{Re}(a+b)<1
$$
时，$z=0,1,\infty$ 三处都局部可积；在这个绝对收敛域内，复 beta 积分满足
（忽略依赖 $d^2z$ convention 的整体常数）
$$
\int_{\mathbb C}d^2z\,|z|^{2a-2}|1-z|^{2b-2}
\propto
\frac{\Gamma(a)\Gamma(b)\Gamma(1-a-b)}
{\Gamma(1-a)\Gamma(1-b)\Gamma(a+b)}.
$$
取
$$
a=-1-\frac{\alpha's}{4},\qquad
b=-1-\frac{\alpha't}{4},
$$
再用 $s+t+u=-16/\alpha'$，恰得到命题中的三个分子与三个分母 Gamma factors。
物理区域的公式由该收敛区域作亚纯解析延拓定义，而非由绝对收敛的 worldsheet
积分直接给出。$\square$

## 6.5 Pole、因子化和 string spectrum

**命题 6.14（Veneziano amplitude 的谱极点）.** 对 generic 固定 $t$，ordered
open-string 四点 tachyon 振幅在
$$
\alpha's=N-1,\qquad N=0,1,2,\ldots
$$
处有 simple poles。这些 poles 对应开弦第 $N$ 激发层的中间态。

**证明.** Gamma function $\Gamma(-1-\alpha's)$ 在
$-1-\alpha's=-N$、$N\in\mathbb Z_{\ge0}$ 处有 simple pole，即
$\alpha's=N-1$。令 $y=-1-\alpha't$。在该 pole 处，另两个 Gamma factors 的
比值为
$$
\frac{\Gamma(y)}{\Gamma(y-N)}
=\prod_{j=1}^{N}(y-j),
$$
其中 $N=0$ 时空积取 $1$。因此 residue 是 $t$ 的次数 $N$ 多项式；只有在这个
多项式的特殊零点上 pole 才会被抵消。Generic $t$ 时 pole 为 simple，并与第四章
$M_N^2=(N-1)/\alpha'$ 一致。$\square$

**注 6.15（unitarity 的接口）.** Tree-level amplitude 的 residue 必须分解为两个三点振幅之和，这表达 perturbative unitarity。完整证明需要物理态完备性、ghost decoupling 和 no-ghost theorem；本书把该部分作为第四、五章外部输入的应用。

**外部输入定理 6.16（tree sewing 与因子化）.** 假设 worldsheet BRST CFT
无 anomaly，plumbing-fixture sewing 有效，BRST exact/null states decouple，且物理
cohomology 上的 BPZ pairing $G_{\alpha\beta}$ 非退化。当 punctured sphere 或 disk
退化为两片曲面由长管连接时，tree-level reduced amplitude 的 pole 部分为
$$
\mathcal M_n=
\sum_{\alpha,\beta}
\mathcal M_L(\cdots,\alpha)
\frac{G^{\alpha\beta}}{k^2+M_\alpha^2-i0}
\mathcal M_R(\beta,\cdots)
+\mathcal M_{\mathrm{reg}},
$$
其中 $\alpha,\beta$ 遍历相应质量层的物理 string cohomology，
$\mathcal M_{\mathrm{reg}}$ 在该 channel 的 pole 处正则。

**证明路线（外部输入）.** 退化参数 $q$ 描述长管传播。CFT sewing 给出传播因子
$$
q^{L_0-a}\bar q^{\tilde L_0-a}
$$
和一组完备中间态。对 $q$ 的径向积分在 $L_0-a=0$ 即 target-space on-shell 条件处
产生 pole，BPZ pairing 收缩左右中间态。完整证明需要 sewing theorem、物理态完备性、
no-ghost theorem 与 BRST decoupling；本书不以这段路线替代这些输入。

**定义 6.17（一圈振幅接口）.** 闭弦一圈 $n$ 点振幅的 gauge-fixed 形式表达式为
$$
\mathcal A_{1,n}
=g_s^n\int_{\mathcal M_{1,n}}
\left\langle
\prod_a (b,\mu_a)(\bar b,\bar\mu_a)
\prod_i V_i
\right\rangle_{\Sigma_1}.
$$
这里 $\mu_a$ 是 Beltrami differentials。完整 superstring 一圈振幅还需要 spin structure sum 和 picture number 规则。

该式尚未指定 degeneration region 的红外 cutoff、Lorentzian $i\varepsilon$、
determinant regulator 或局部 counterterms，因而不是一个自动收敛的 Lebesgue
积分定义。第五章的玻色弦真空例子已显示 modular fundamental domain 与红外有限性
是两件不同的事。

**注 6.18（unitarity 与 modular invariance）.** Loop amplitude 的一致性同时依赖 degeneration factorization 和 modular invariance。前者控制 unitarity，后者避免同一 worldsheet complex structure 被重复计数。

于是，一个 tree amplitude 同时保留了三类信息：顶点算子的 BRST 类指定外态，
ghost zero modes 决定哪些 punctures 可以固定，剩余位置积分产生 Beta/Gamma
函数。Veneziano 与 Virasoro--Shapiro 振幅的 pole 正好落在第四章的弦质量层级上，
其退化极限又把 residue 分解为低点振幅与中间物理态之和。到了 loop order，同一
因子化机制必须与 modular invariance 一起放到 Riemann surface moduli space 上处理。

## 练习

**练习 6.1.** 用第三章 OPE 推导闭弦 tachyon 的 conformal weights，并验证其质量公式。

**练习 6.2.** 推导开弦边界指数算子的 Koba-Nielsen 因子。

**练习 6.3.** 从 Veneziano amplitude 的 gamma function 表达式读出开弦第 $N$ 层的 pole 位置。

**练习 6.4.** 解释 worldsheet degeneration 如何给出 target-space propagator pole。
