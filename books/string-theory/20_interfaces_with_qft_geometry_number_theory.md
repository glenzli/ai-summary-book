# 第二十章：string theory 与量子场论、几何和数论的接口

弦论中的“接口”不是两个领域名词并排出现，而是同一个量能够在两套语言中独立定义、
计算并比较。例如，两张 D-branes 的几何间距会变成规范理论中 $W$ boson 的质量；
内部空间上的 Dirac index 会变成低维手征零模的净数目；worldsheet 的 modular
invariance 又会把 BPS generating function 限制为模形式或 Jacobi form。要使这种
转译有效，必须说明字典作用在哪些对象上、比较的是普通态数还是受保护指数，以及
低能、large-charge 或 topological twist 等近似是否成立。以下建立四条可演算的
接口，并用 K3/D1--D5 系统把量子场论、几何与数论数据放进同一个例子。所需材料来自
第十二章 D-brane 有效理论、第十三与十六章的 Calabi--Yau/拓扑弦几何，以及第十七章
的 BPS index；相关指标定理和模性结果会按外部输入的精确版本调用。

## 20.1 从拉伸开弦到规范理论的 Higgs 机制

先考虑平坦 type II target 中 $N$ 张彼此平行的 D$p$-branes。第 $a$ 张 brane 在
Dirichlet 方向的位置记为 $y_a^I$，其中 $I=p+1,\ldots,9$。一条定向开弦的两个端点
分别落在第 $a$、$b$ 张 brane 上，因而带 Chan--Paton label $(a,b)$。

**定义 20.1（brane position 与 adjoint scalar 的归一化）.** 在低能
$U(N)$ worldvolume theory 中，把 brane position matrix $Y^I$ 与具有质量量纲一的
adjoint scalar $\Phi^I$ 关联为
$$
\Phi^I=\frac{Y^I}{2\pi\alpha'}.
$$
对角背景
$$
\langle Y^I\rangle
=\operatorname{diag}(y_1^I,\ldots,y_N^I)
$$
表示分离的平行 branes；整体平移只改变 center-of-mass $U(1)$ scalar。

**命题 20.2（拉伸开弦质量与规范增强）.** 设
$$
L_{ab}=\left(\sum_I|y_a^I-y_b^I|^2\right)^{1/2}.
$$
在平坦背景、零 worldvolume flux 和弱弦耦合下，$(a,b)$ sector 的 open RNS
string 满足
$$
M_{ab}^2
=\frac{L_{ab}^2}{(2\pi\alpha')^2}
+\frac1{\alpha'}\left(N_{\mathrm{osc}}-\frac12\right).
$$
GSO 投影保留的最低 NS vector 有 $N_{\mathrm{osc}}=1/2$，故
$$
m_{ab}=\frac{L_{ab}}{2\pi\alpha'}
=|\phi_a-\phi_b|,
$$
其中 $\phi_a^I=y_a^I/(2\pi\alpha')$。当所有 $y_a$ 重合时，全部 off-diagonal
vectors 变为无质量，低能 gauge symmetry 增强为 $U(N)$。

**推导说明（标准物理口径）.** 对一条在 $\sigma\in[0,\pi]$ 上连接两张 branes 的
开弦，Dirichlet 方向的经典部分为
$$
X^I_{\mathrm{cl}}(\sigma)
=y_a^I+\frac{\sigma}{\pi}(y_b^I-y_a^I).
$$
将 $\partial_\sigma X_{\mathrm{cl}}^I=(y_b^I-y_a^I)/\pi$ 代入开弦 Hamiltonian，
其对 $L_0$ 的贡献为
$$
\frac{L_{ab}^2}{4\pi^2\alpha'}.
$$
因此物理态条件写成
$$
0=L_0-\frac12
=\alpha'k_\parallel^2
+\frac{L_{ab}^2}{4\pi^2\alpha'}
+N_{\mathrm{osc}}-\frac12.
$$
用 $M^2=-k_\parallel^2$ 即得质量公式。低能 SYM 中
$\operatorname{Tr}(D_a\Phi^I D^a\Phi^I)$ 在对角 vev 周围对 off-diagonal
gauge field 产生 $|\phi_a-\phi_b|^2|A_a{}^b|^2$，与弦谱完全匹配。推导忽略
brane curvature、background fields、string loops 与 massive oscillators。$\square$

**例 20.3（两张 D3-branes）.** 两张 D3-branes 只在 $x^9$ 方向相距 $L$ 时，
低能理论的背景可写为
$$
\langle\Phi^9\rangle
=\operatorname{diag}\left(0,\frac{L}{2\pi\alpha'}\right).
$$
$(1,2)$ 与 $(2,1)$ open strings 给出一对带电 massive vectors，质量
$L/(2\pi\alpha')$；当 $L\to0$ 时，它们与两个对角 photons 一起补成 $U(2)$
adjoint。几何中的 brane coincidence 正是规范理论中的 Coulomb-branch 原点。

这条接口不仅比较最终谱，还固定了参数字典
$Y^I=2\pi\alpha'\Phi^I$。若只说“D-branes 产生 gauge theory”而不写出这个系数，
就无法比较 string tension、Higgs mass 和 worldvolume kinetic term。

## 20.2 从内部几何到低维零模与手征指标

第二类接口从高维场的 Kaluza--Klein 展开开始。设背景在所考察阶可写为
$M_d\times X$，其中 $X$ 是 compact、无边界的 Riemannian manifold；暂不加入
flux、warping 或非平凡 fibration。

**定义 20.4（几何零模）.** 对 $X$ 上的 $r$-form Laplacian
$$
\Delta_X=d_Xd_X^\dagger+d_X^\dagger d_X,
$$
取正交归一本征基 $\omega_I^{(r)}$，
$$
\Delta_X\omega_I^{(r)}=\lambda_I\omega_I^{(r)},\qquad \lambda_I\ge0.
$$
高维 $p$-form potential 可按
$$
C_p(x,y)
=\sum_{r=0}^{\min(p,\dim X)}\sum_I
A_{p-r}^I(x)\wedge\omega_I^{(r)}(y)
$$
展开。满足 $\lambda_I=0$ 的系数场称为该乘积背景的几何零模。

**命题 20.5（Betti number 计数 bosonic 零模）.** 假设高维作用量在
$C_p$ 的 quadratic 部分只含标准 kinetic term
$\int|dC_p|^2$，没有额外 self-duality constraint，且 gauge fixing 已去除 exact
冗余。则 $A_{p-r}^I$ 的
$d$ 维质量满足
$$
m_I^2=\lambda_I.
$$
因此由 $r$-form 内部波函数产生的 massless fields 数目等于
$$
b_r(X)=\dim H^r_{\mathrm{dR}}(X;\mathbb R).
$$

**证明.** 把定义 20.4 的展开代入 quadratic action。外部导数产生
$|d_{M_d}A^I|^2$，内部导数项经分部积分给出
$$
\int_X\omega_I^{(r)}\wedge *\Delta_X\omega_J^{(r)}
=\lambda_I\delta_{IJ},
$$
故该项是 $d$ 维的 $m_I^2|A^I|^2$。Compact Hodge theorem 给出
$\ker\Delta_X\simeq H^r_{\mathrm{dR}}(X;\mathbb R)$，于是零本征空间维数为
$b_r(X)$。Hodge theorem 是这里明确调用的外部数学输入；flux、warping 与
Stueckelberg couplings 会改变质量算符，因而不在命题假设内。$\square$

Bosonic 零模通常由 cohomology 的维数计数；chiral fermions 更自然地由两个核空间
之差计数。这个差不随小的几何扰动变化，正是量子场论 anomaly 与拓扑特征类相遇的
位置。

**外部输入定理 20.6（twisted Dirac index）.** 设 $X$ 是 compact、无边界的偶维
spin manifold，$E\to X$ 是带 Hermitian connection 的复向量丛，
$$
D_E^+:\Gamma(S^+\otimes E)\longrightarrow\Gamma(S^-\otimes E)
$$
是相应 elliptic chiral Dirac operator。则
$$
\operatorname{ind}D_E^+
=\dim\ker D_E^+-\dim\ker D_E^-
=\int_X\widehat A(TX)\operatorname{ch}(E)\big|_{\dim X}.
$$
这个等式计算净 chirality，而不是两个 kernel 各自的维数。若谱穿过零但算符保持
Fredholm，两个维数可以同时改变，index 仍保持不变。

在 compact complex manifold 上，Dolbeault operator 的相应版本为
Hirzebruch--Riemann--Roch：
$$
\chi(X,E)
=\sum_q(-1)^q\dim H^q(X,E)
=\int_X\operatorname{Td}(TX)\operatorname{ch}(E).
$$
实际 string compactification 还要指定十维 fermion representation、bundle embedding、
GSO/orientifold projection 与 anomaly cancellation，不能只凭一个整数宣布完整粒子谱。

**例 20.7（Riemann surface 上的三重净零模）.** 设 $X$ 是 genus $g=1$ 的 compact
Riemann surface，$L\to X$ 是 degree $3$ 的 holomorphic line bundle。Riemann--Roch
外部输入给出
$$
h^0(X,L)-h^1(X,L)=\deg L+1-g=3.
$$
又由 Serre duality，
$h^1(X,L)=h^0(X,K_X\otimes L^{-1})$；在 torus 上 $K_X$ 平凡，而负次数
$L^{-1}$ 没有非零 holomorphic section，故
$$
h^0(X,L)=3,\qquad h^1(X,L)=0.
$$
在把这套 Dolbeault complex 实现为内部 fermion 方程的简化紧化中，它给出三个同
chirality 零模。这个 toy calculation 展示“几何次数 $\to$ QFT 净手征数”的接口，
但不包含真实四维模型所需的全部 gauge representations 与投影。

## 20.3 从 topological string 到曲线计数生成函数

上一节的 cohomology 只计数线性零模。世界面 instantons 会探测 target 中的
holomorphic curves，而 A-model 把这些非线性数据组织成 genus expansion。最容易
直接比较的是 genus-zero prepotential 的三阶导数。

**定义 20.8（genus-zero instanton potential）.** 设 $X$ 是 Calabi--Yau threefold。
采用归一化 flat variables $t^a=\log q_a$，并在 $F_0$ 中吸收与之配套的
$2\pi i$ 幂，使 classical cubic coefficient 仍为 triple intersection
$\kappa_{abc}$。令
$$
q^\beta=\exp\left(\sum_a\beta_a t^a\right),\qquad |q^\beta|<1
$$
对应 large-radius 展开。用 $n_\beta^{(0)}$ 表示有效 curve classes 上的 genus-zero
Gopakumar--Vafa/BPS invariants；$\operatorname{Li}_3$ 将每个 class 的 multiple
covers 组织在一起。去掉不影响三阶导数的至多二次多项式后，定义
$$
F_0(t)
=\frac16\kappa_{abc}t^at^bt^c
+\sum_{\beta\ne0}n_\beta^{(0)}
\operatorname{Li}_3(q^\beta).
$$
这里
$$
\operatorname{Li}_3(q^\beta)
=\sum_{d=1}^{\infty}\frac{q^{d\beta}}{d^3}
$$
把相应 curve contribution 的 multiple covers 一并编码。

**命题 20.9（Yukawa coupling 的 instanton 展开）.** 在定义 20.8 的 large-radius
收敛区域内，或逐项按形式幂级数理解时，
$$
C_{abc}(t)
\coloneqq\partial_a\partial_b\partial_cF_0(t)
=\kappa_{abc}
+\sum_{\beta\ne0}n_\beta^{(0)}
\beta_a\beta_b\beta_c
\frac{q^\beta}{1-q^\beta}.
$$

**证明.** 对 $x=\sum_a\beta_at^a$，逐项微分绝对收敛级数：
$$
\frac{d}{dx}\operatorname{Li}_s(e^x)
=\operatorname{Li}_{s-1}(e^x).
$$
连续三次使用该恒等式，并用
$\operatorname{Li}_0(q)=q/(1-q)$，得到每个 curve class 的显示项；三次微分
$\kappa_{def}t^dt^et^f/6$ 给出 $\kappa_{abc}$。$\square$

公式本身是生成函数的代数推论；把 A-model path integral 的 instanton sector
严格识别为 virtual fundamental class 上的 Gromov--Witten invariants，以及把
$n_\beta^{(0)}$ 解释为整数 BPS invariants，需要外部数学构造或 Gopakumar--Vafa
物理输入。两层陈述不能只因写在同一公式中就视为同一证明。

**例 20.10（resolved conifold）.** 对局部 Calabi--Yau
$$
X=\mathcal O(-1)\oplus\mathcal O(-1)\longrightarrow\mathbb P^1,
$$
唯一 primitive compact curve class $\beta=[\mathbb P^1]$ 的
$n_\beta^{(0)}=1$，其 multiples 不引入新的 primitive genus-zero invariants。因此
忽略 noncompact geometry 依赖的 classical polynomial，
$$
F_0^{\mathrm{inst}}(Q)
=\sum_{d=1}^{\infty}\frac{Q^d}{d^3}
=\operatorname{Li}_3(Q),
\qquad Q=q^\beta,
$$
而三阶 flat-coordinate derivative 为 $Q/(1-Q)$。A-model 把每个 degree 的
multiple-cover contribution 放在 $Q$ 展开系数中；mirror B-model 则从 periods
满足的 Picard--Fuchs equation 求得同一函数。这是“曲线模空间积分 $\leftrightarrow$
period differential equation”的具体计算接口。

## 20.4 从 worldsheet modularity 到算术增长

弦的 one-loop worldsheet 是 torus，所以 partition function 不能任意依赖
$\tau$。在 supersymmetric sector 中加入 $(-1)^F$ 后，非 BPS pairs 抵消，留下的
holomorphic generating function 往往具有更强的 Jacobi 或 modular transformation
law；Fourier coefficients 因而受到远超一般幂级数的约束。

**定义 20.11（elliptic genus）.** 对具有离散谱的 compact unitary
$N=(2,2)$ SCFT，定义 Ramond--Ramond elliptic genus
$$
\phi(\tau,z)
=\operatorname{Tr}_{\mathrm{RR}}
\left[
(-1)^{F_L+F_R}
y^{J_0}
q^{L_0-c_L/24}
\bar q^{\,\bar L_0-c_R/24}
\right],
\qquad
q=e^{2\pi i\tau},\quad y=e^{2\pi iz}.
$$
右移动 non-ground states 在 supersymmetric pairs 中抵消；当 trace 收敛且没有
continuum boundary contribution 时，$\phi$ 与 $\bar\tau$ 无关。

**外部输入定理 20.12（elliptic genus 的 Jacobi 变换）.** 若上述 SCFT 的
$U(1)$ charge lattice、spectral flow 与 anomalies 满足标准 integrality 条件，则
$\phi(\tau,z)$ 是带相应 multiplier 的 weak Jacobi form。对整数 weight $k$、index
$m$ 的情形，
$$
\phi\left(\frac{a\tau+b}{c\tau+d},\frac{z}{c\tau+d}\right)
=(c\tau+d)^k
\exp\left(\frac{2\pi imcz^2}{c\tau+d}\right)
\phi(\tau,z),
$$
并且
$$
\phi(\tau,z+\lambda\tau+\mu)
=e^{-2\pi im(\lambda^2\tau+2\lambda z)}\phi(\tau,z),
\qquad \lambda,\mu\in\mathbb Z.
$$
Calabi--Yau sigma model 的标准归一化给出 $k=0$、
$m=c_L/6=\dim_{\mathbb C}X/2$；奇复维时需保留半整数 index/multiplier 的细节。
非紧 target、连续谱或 mock completion 会修改“holomorphic weak Jacobi form”这一
简单结论。

即使不引入 chemical potential，最基本的 oscillator Fock space 在补回零点能后
也会产生经典模形式。

**命题 20.13（24 个 bosonic oscillators 的生成函数）.** 令 $d_N$ 是由
$24$ 种 bosonic creation operators $\alpha_{-n}^i$ 在总 level $N$ 生成的态数，
不施加额外 gauge、charge 或 level-matching 条件。则
$$
Z_{24}(q)
\coloneqq\sum_{N=0}^{\infty}d_Nq^N
=\prod_{n=1}^{\infty}(1-q^n)^{-24}
=\frac{q}{\Delta(\tau)},
$$
其中
$$
\eta(\tau)=q^{1/24}\prod_{n\ge1}(1-q^n),\qquad
\Delta(\tau)=\eta(\tau)^{24}.
$$
因此按 weight $-12$ 变换的是
$$
q^{-1}Z_{24}(q)=\frac1{\Delta(\tau)},
$$
而 $Z_{24}$ 是去掉零点能后的 level generating series，本身不具有该简单 modular
transformation law。前几项为
$$
Z_{24}(q)=1+24q+324q^2+O(q^3).
$$

**证明.** 对每个 mode number $n$ 和 species $i$，occupation number
$r_{n,i}\ge0$ 的生成函数是
$\sum_{r\ge0}q^{nr}=(1-q^n)^{-1}$。对全部 $n,i$ 相乘得到第一式。
Dedekind eta 的乘积定义给出
$\Delta=q\prod_{n\ge1}(1-q^n)^{24}$，故得到第二式。
Level $1$ 只有 $24$ 个 single-particle states；level $2$ 包含 $24$ 个
$\alpha_{-2}^i$ states 和
$\binom{24+2-1}{2}=300$ 个两个 $\alpha_{-1}$ 的对称 states，总数为 $324$。
$\square$

**外部输入定理 20.14（系数的 leading growth）.** 对命题 20.13 的 $d_N$，
Hardy--Ramanujan/Rademacher 方法或 $c=24$ Cardy theorem 给出
$$
\log d_N=4\pi\sqrt N+O(\log N),
\qquad N\to\infty.
$$
在 heterotic 或 D-brane BPS 问题中，只有当 level $N$ 已由 charges 固定、右移动
sector 处于相应 supersymmetric ground state，且 projection 与 zero modes 已处理时，
这个增长才能解释为物理 BPS degeneracy 或 index 的 leading entropy。裸
$Z_{24}$ 只计数 oscillator states。

这里可以看出数论信息的实际作用：模变换把 $\tau\to0$ 的高温区关联到
$\tau\to i\infty$ 的低能 polar data，从少量低阶系数控制大 level 的指数增长。
这不是“整数序列看起来很特别”，而是 worldsheet 大微分同胚对生成函数施加的函数
方程。

## 20.5 受保护指数为何能够跨越不同描述

前几节比较的量之所以能从弱耦合或大体积带到另一种描述，关键不在 duality 这个
名称，而在所比较对象是否为 index。普通能级在连续变形下可以移动；index 只记录
未被 boson--fermion 配对消去的差。

**定义 20.15（supersymmetric index）.** 设 $\mathbb Z_2$-graded Hilbert space
$\mathcal H=\mathcal H_{\bar0}\oplus\mathcal H_{\bar1}$ 上有闭的奇算符 $Q$，
$$
Q^2=0,\qquad
H=\frac12\{Q,Q^\dagger\}\ge0,
$$
并假设 $(-1)^Fe^{-\beta H}$ 为 trace class。定义
$$
I(\beta)=\operatorname{Tr}_{\mathcal H}
\left((-1)^Fe^{-\beta H}\right).
$$

**命题 20.16（离散谱下的 index 不变性）.** 在定义 20.15 的假设下，若正能谱离散
且有限重，所有正能 states 都在 $Q,Q^\dagger$ 作用下组成 boson--fermion pairs，
则
$$
I(\beta)
=\dim\ker(H|_{\mathcal H_{\bar0}})
-\dim\ker(H|_{\mathcal H_{\bar1}})
$$
与 $\beta$ 无关。对保持 grading、Fredholm 性和 supersymmetry 的连续变形，该整数
局部不变。

**证明.** 若 $H|\psi\rangle=E|\psi\rangle$ 且 $E>0$，则
$$
\|\!Q\psi\!\|^2+\|\!Q^\dagger\psi\!\|^2=2E\|\psi\|^2,
$$
所以 $|\psi\rangle$ 不可能同时被 $Q,Q^\dagger$ 湮灭。$Q+Q^\dagger$ 与 $H$
对易并翻转 fermion parity，因而在每个 $E>0$ eigenspace 上给出等维的
bosonic/fermionic 配对，其 supertrace 为零。只有 $E=0$ states 留下，得到显示式。
在保持 Fredholm 性的连续变形中，零模只能成对进入或离开零点，所以两种 parity
维数之差不变。$\square$

若 target 非紧、存在连续谱阈值、态从无穷远流入，或跨过 marginal-stability wall，
上述 Fredholm/trace-class 假设可能失效。此时 elliptic genus 可能需要非全纯
completion，BPS index 也可能发生 wall crossing。几何 index、worldsheet trace 和
时空 BPS count 只有在边界条件与 ensemble 一致时才是同一个量。

## 20.6 贯穿案例：K3、elliptic genus 与 D1--D5

K3 sigma model 把前三类接口放在一个紧致例子中。K3 的 Hodge diamond 给出
Euler characteristic
$$
\chi(K3)=24.
$$
对应 $N=(4,4)$ SCFT 的 elliptic genus 是 weight $0$、index $1$ 的 weak Jacobi
form，其 polar/constant 部分为
$$
\phi_{K3}(\tau,z)
=2y+20+2y^{-1}+O(q).
$$
取 $z=0$ 得
$$
\phi_{K3}(\tau,0)=24.
$$
左边是 RR Hilbert space 的 supersymmetric trace，右边是几何 Euler characteristic；
命题 20.16 解释了为何沿 K3 sigma-model moduli space 改变 metric 与 $B$-field 时
这个数保持不变。

**外部输入定理 20.17（DMVV symmetric-product formula）.** 写
$$
\phi_{K3}(\tau,z)
=\sum_{m\ge0,\,\ell\in\mathbb Z}c(m,\ell)q^my^\ell.
$$
则 symmetric products 的 orbifold elliptic genera 满足
$$
\sum_{N=0}^{\infty}p^N
\phi_{\operatorname{Sym}^N(K3)}(\tau,z)
=
\prod_{\substack{n>0\\m\ge0\\\ell\in\mathbb Z}}
\left(1-p^nq^my^\ell\right)^{-c(nm,\ell)}.
$$
该等式来自 permutation-orbifold twisted sectors 与 commuting-pair sum；这里把
完整 orbifold CFT/几何证明作为外部输入。

在 D1--D5 系统的适当 K3 charge sector 中，红外二维理论位于
$\operatorname{Sym}^N(K3)$ 模空间的相应分支。于是同一组系数有四种互相校准的
解释：K3 cohomology 给出 seed polar data，二维 QFT 把它写成 elliptic genus，
DMVV 乘积把多弦 twisted sectors 组织为 Jacobi generating function，大 charge
Fourier coefficients 再给出第十七章黑洞指数的熵增长。等号比较的是受保护 index；
center-of-mass multiplet、continuum、charge shift 与从 index 到绝对 degeneracy 的
升级仍需按第十七章的条件分别处理。

这些计算说明，跨领域接口的可靠单位是“同一对象的两种可核对构造”。拉伸弦与
Higgs mass 由参数归一化相接，Dirac/Dolbeault index 把几何特征类变成净手征数，
topological string 把曲线 classes 变成生成函数，modularity 再从 polar data 控制
大 level 增长。每次转译都保留明确假设，因此可以用于计算；一旦去掉低能截断、
compactness、supersymmetric protection 或 modular anomaly 条件，相应等号也必须
重新检验。

## 练习

**练习 20.1.** 两张平行 D$p$-branes 相距 $L$。从 Dirichlet classical solution
计算其对 $L_0$ 的贡献，并验证最低 GSO-allowed NS vector 的质量为
$L/(2\pi\alpha')$。

**练习 20.2.** 设 genus-$g$ Riemann surface 上的 line bundle $L$ 满足
$\deg L>2g-2$。用 Riemann--Roch 与 Serre duality 证明
$h^0(X,L)=\deg L+1-g$，并解释这个等式为何只固定净手征零模而非完整相互作用。

**练习 20.3.** 从
$F_0^{\mathrm{inst}}(Q)=\operatorname{Li}_3(Q)$ 出发，计算前三次
$t=\log Q$ 导数，并展开 $Q/(1-Q)$ 的前四项；说明这些项如何记录 multiple covers。

**练习 20.4.** 直接按 oscillator partitions 计算 $Z_{24}(q)$ 的 $q^3$ 系数，并把
贡献分成 mode partitions $3$、$2+1$ 与 $1+1+1$ 三类。

**练习 20.5.** 逐项指出命题 20.16 的证明在哪些地方使用了离散谱、trace-class
与 Fredholm 假设，并说明非紧 sigma model 的 continuum 为什么可能产生额外边界项。
