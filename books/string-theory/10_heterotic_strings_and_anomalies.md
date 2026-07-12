# 第十章：heterotic strings、current algebra 和 anomaly cancellation

## 本章目标

Heterotic string 把右移动 superstring 与左移动 bosonic string 组合在同一个闭弦理论中。本章目标是说明：

1. 为什么左移动内部 CFT 需要 central charge $16$；
2. 为什么十维 gauge lattice 被限制为 $E_8\times E_8$ 或 $\operatorname{Spin}(32)/\mathbb Z_2$；
3. gauge bosons 如何由 current algebra 产生；
4. Green-Schwarz anomaly cancellation 如何限制十维理论。

## 依赖前置知识

需要第三章 CFT、第八章 RNS 超弦、附录 C 的 Lie algebra 与 anomaly polynomial 语言。

## 10.1 Heterotic 构造

**定义 10.1（heterotic worldsheet CFT）.** Heterotic string 的 right-moving sector 取十维 RNS superstring，left-moving sector 取 bosonic string。十个 target coordinates $X^\mu$ 同时出现在左右两边；额外左移动 internal CFT 的 central charge 必须为 $16$。

**命题 10.2（central charge accounting）.** 十维 heterotic string 的左移动 internal sector 必须满足
$$
c_L^{\mathrm{int}}=16.
$$

**证明.** 左移动 bosonic string 需要 matter central charge $26$ 才能抵消 $bc$ ghosts。十个非紧坐标贡献 $c=10$，故 internal left-moving CFT 必须贡献 $26-10=16$。右移动 RNS sector 中十个 $X^\mu$ 与十个 $\psi^\mu$ 贡献 $c=15$，正好抵消 RNS ghosts。$\square$

**定义 10.3（lattice realization）.** 最简单的 heterotic internal sector 由 $16$ 个左移动 compact bosons 构成，紧化在 even unimodular lattice $\Lambda$ 上。

内部 lattice CFT 的 states 由 oscillator excitations 与 lattice momenta $P\in\Lambda$ 张成，其 character 含有 theta function
$$
\Theta_\Lambda(\tau)=\sum_{P\in\Lambda}q^{P^2/2},
\qquad q=e^{2\pi i\tau}.
$$

**外部输入定理 10.4（十六维 even unimodular lattices）.** 正定十六维 even unimodular lattices 只有两类：
$$
\Gamma_{E_8}\oplus\Gamma_{E_8},
\qquad
\Gamma_{\operatorname{Spin}(32)/\mathbb Z_2}.
$$
对应的十维 heterotic gauge groups 为
$$
E_8\times E_8,\qquad \operatorname{Spin}(32)/\mathbb Z_2.
$$

**使用边界.** 该分类属于 lattice theory。正文只使用分类结论和它对 modular invariance 的后果。

## 10.2 Heterotic 质量公式和 gauge bosons

**定义 10.5（内部 lattice momentum）.** 设 $P\in\Lambda$ 为左移动内部 momentum。左移动 Virasoro 零模含有项 $P^2/2$。

**命题 10.6（heterotic level matching）.** Heterotic string 物理态满足
$$
N_L+\frac{P^2}{2}-1=N_R-a_R,
$$
其中 right-moving NS sector 中 $a_R=1/2$，R sector 中 $a_R=0$。

**推导说明（标准物理口径）.** 左移动 bosonic sector 的 normal ordering constant 为 $1$，右移动 RNS sector 的 normal ordering constant 为 $a_R$。闭弦物理条件要求左右 $L_0$ 约束给出同一个 spacetime mass，并且 $L_0-\tilde L_0=0$；消去共同的非紧动量项即得。$\square$

**命题 10.7（gauge bosons）.** 十维 heterotic gauge bosons 来自形如
$$
J^a(z)\,\tilde\psi^\mu(\bar z)e^{ik\cdot X(z,\bar z)}
$$
的顶点算子，其中 $J^a$ 是左移动 current，$\tilde\psi^\mu$ 是右移动 NS fermion。

**推导说明（标准物理口径）.** 右移动部分 $\tilde\psi^\mu e^{ikX}$ 是 massless vector 的 NS 顶点。左移动 current $J^a$ 具有 weight $1$，取代开弦 Chan-Paton matrix，给出 gauge algebra index。整体 conformal weights 为 $(1,1)$，并且 $k^2=0$。$\square$

**例 10.7A（root states）.** 若 $P\in\Lambda$ 满足 $P^2=2$，则内部 vertex
$$
J_P(z)=:e^{iP\cdot Y(z)}:
$$
具有 weight $1$，对应 gauge algebra 的 root generator。Cartan generators 来自
$$
i\partial Y^I(z).
$$
因此 root lattice 与 Cartan currents 一起生成十维 gauge algebra。

## 10.3 Current algebra

**定义 10.8（affine Lie algebra OPE）.** Kac-Moody currents $J^a(z)$ 满足
$$
J^a(z)J^b(w)\sim
\frac{k\delta^{ab}}{(z-w)^2}
+\frac{if^{ab}_{\ \ c}J^c(w)}{z-w}.
$$
这里 $k$ 是 level。

**命题 10.9（level-one simply-laced current algebra）.** Lattice realization 中，长度平方为 $2$ 的 lattice vectors 给出 roots，并生成 level-one simply-laced current algebra。

**推导说明（标准物理口径）.** 内部 compact boson 的 vertex $e^{iP\cdot Y}$ 在 $P^2=2$ 时具有 left conformal weight $1$，可作为 current。其 OPE 的 singular terms 由 lattice inner product 控制，并复现对应 root system 的 Lie bracket。$\square$

## 10.4 Modular invariance 与 gauge group 限制

**命题 10.10（lattice realization 的 modular 条件）.** 在定义 10.3 的
$16$ 个纯左移 compact bosons realization 中，若内部 lattice character 要与右移
RNS spin-structure sum 组合成 modular-invariant torus integrand，则积分 lattice
$\Lambda$ 必须为 even 且 self-dual，亦即 unimodular。

**推导说明（标准物理口径）.** 先把有限维 Poisson summation 应用于 rank $r$
lattice Gaussian。若 $\Lambda^*$ 是 dual lattice，$\operatorname{vol}(\Lambda)$ 是
covolume，则
$$
\Theta_\Lambda(-1/\tau)
=\frac{(-i\tau)^{r/2}}{\operatorname{vol}(\Lambda)}
\Theta_{\Lambda^*}(\tau).
$$
因此 $S$ 变换若不引入新的 momentum cosets，就要求
$\Lambda=\Lambda^*$，等价于 $\operatorname{vol}(\Lambda)=1$。另一方面
$$
\Theta_\Lambda(\tau+1)
=\sum_{P\in\Lambda}e^{\pi iP^2}q^{P^2/2};
$$
要使相位不依赖 $P$，必须有 $P^2\in2\mathbb Z$。$\eta^{16}$ 在 $T$ 下仍有由
chiral central charge 决定的统一 multiplier；该 multiplier 与右移 RNS/ghost
因子一起抵消，不能要求内部 character 单独 invariant。Poisson summation 与
theta 变换是精确数学步骤，把 chiral characters 组合为 string measure 则是
one-loop CFT 输入。$\square$

**命题 10.10A（partition function 的 lattice 因子）.** 左移动内部 lattice 对
torus chiral character 的贡献形如
$$
Z_\Lambda(\tau)=\frac{\Theta_\Lambda(\tau)}{\eta(\tau)^{16}}.
$$
Modular invariance 要求 $Z_\Lambda$ 与非紧坐标、ghosts 及右移动 RNS
spin-structure sum 组合后在 $SL(2,\mathbb Z)$ 下不变；$Z_\Lambda$ 本身可带
central-charge multiplier。

**推导说明（标准物理口径）.** $16$ 个 compact chiral bosons 的 oscillator 部分给出 $\eta(\tau)^{-16}$，零模动量求和给出 $\Theta_\Lambda$。$\eta$ 与 theta function 的 modular transformation 决定 even unimodular 条件。$\square$

## 10.5 Anomaly cancellation

十维 $N=1$ supergravity coupled to super Yang-Mills 是 chiral theory，存在 gauge 与 gravitational anomalies。String consistency 要求这些 anomalies 被抵消。

**外部输入定理 10.11（heterotic Green--Schwarz factorization）.** 对十维
$N=1$ supergravity coupled to super Yang--Mills 的标准手征场内容，one-loop
局部 anomaly 由十二形式 $I_{12}$ 的 descent 给出。消去不可约 gravitational
$\operatorname{tr}R^6$ 项先要求 $\dim\mathfrak g=496$；若只用同一个 NS--NS
two-form 作 Green--Schwarz cancellation，还要求剩余 polynomial 因式分解为
$$
I_{12}=X_4X_8.
$$
对定义 10.3 的两种 heterotic lattice 所产生的 gauge algebra
$\mathfrak{so}(32)$ 与 $\mathfrak e_8\oplus\mathfrak e_8$，所需 trace identities
成立；对应 heterotic 全局群通常写作
$$
\operatorname{Spin}(32)/\mathbb Z_2,\qquad E_8\times E_8.
$$

在标准规范下，因式分解的低阶因子含有
$$
X_4=\operatorname{tr}R^2-\operatorname{Tr}F^2
$$
的形式；$X_8$ 是由 $\operatorname{tr}R^4$、$(\operatorname{tr}R^2)^2$、gauge traces 及混合项组成的八形式。

**使用边界.** 本书不重做 chiral determinant 的 index/descent 计算，也不把
factorization 误写成所有十维低能 gauge algebra 的完整分类。这里的结论是：前文
lattice 构造选出的两个 heterotic 理论通过局部 perturbative anomaly 检验。局部
$I_{12}$ factorization 单独不证明 global anomaly 消失，也不替代 worldsheet
modular invariance、factorization 或 tadpole 条件。

**定义 10.12（modified field strength）.** Green-Schwarz mechanism 中，NS-NS two-form $B$ 的 field strength 被修正为
$$
H=dB-\frac{\alpha'}4(\omega_{\mathrm{YM}}-\omega_{\mathrm L}),
$$
并满足 Bianchi identity
$$
dH=\frac{\alpha'}4\left(\operatorname{tr}R\wedge R-\operatorname{Tr}F\wedge F\right).
$$

**命题 10.13（$B$-field variation 抵消 anomaly 的结构）.** 若 $I_{12}=X_4X_8$ 且 $X_4$ 与 $dH$ 中的 characteristic form 相同，则加入
$$
S_{\mathrm{GS}}\supset \int B\wedge X_8
$$
可通过 $B$ 的 gauge/Lorentz transformation 抵消 one-loop anomaly。

**推导说明（标准物理口径）.** Gauge 或 Lorentz transformation 下 Chern-Simons forms 的 variation 为 exact form，使 $H$ invariant 需要 $B$ 同时变换。于是 $\int B\wedge X_8$ 的 variation 给出 descent formalism 中与 anomaly polynomial $X_4X_8$ 对应的 anomaly，符号和规范化由 string one-loop amplitude 固定。$\square$

**注 10.14（trace convention）.** $\operatorname{tr}$ 与 $\operatorname{Tr}$ 的区别是 anomaly cancellation 中最容易出错的 convention 之一。本书把 $\operatorname{tr}$ 用于 Lorentz fundamental trace，把 $\operatorname{Tr}$ 用于 gauge trace 的规范化形式；具体群表示的换算放入附录 C。

**注 10.15（必要条件与充分条件）.** Central-charge cancellation 是世界面局部条件，
even self-dual lattice 给出本章 realization 的 genus-one modular 条件，
$I_{12}=X_4X_8$ 是十维低能局部 anomaly 条件。三者相互支持但逻辑上不同；任意一条
都不能单独推出完整 perturbative heterotic string 的存在与一致性。

## 本章小结

Heterotic string 的严密主线是 central charge matching、even unimodular lattice、current algebra 和 anomaly cancellation。它不是任意把 gauge theory 加到 closed string 上；十维一致性把 gauge group 强烈限制为 $E_8\times E_8$ 与 $\operatorname{Spin}(32)/\mathbb Z_2$ 对应结构。

## 练习

**练习 10.1.** 解释为什么左移动额外自由度需要 $16$ 维 lattice。

**练习 10.2.** 验证 $P^2=2$ 的内部 lattice vertex 具有 left conformal weight $1$。

**练习 10.3.** 说明 $\Theta_\Lambda(\tau)/\eta(\tau)^{16}$ 中 theta function 和 eta function 分别来自哪些自由度。
