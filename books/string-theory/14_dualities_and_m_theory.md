# 第十四章：S-duality、U-duality、M-theory 和 brane web

## 本章目标

本章给出 string dualities 的概念分层。T-duality 是 perturbative worldsheet CFT 中可见的等价；S-duality、U-duality 和 M-theory 通常是非微扰物理原则，依靠 BPS spectra、低能 supergravity、brane dynamics 和大量检验支撑。

## 依赖前置知识

需要第七章 T-duality、第九至十二章超弦和 D-branes，以及第十三章紧化接口。

## 14.1 Duality 的状态分层

**定义 14.1（duality）.** Duality 是两个表面上不同的理论描述之间的等价，要求 physical observables、spectrum、correlation functions 或 protected quantities 在适当 dictionary 下匹配。

**注 14.2（本书状态约定）.** 本书按以下层级书写 duality：

1. Perturbative theorem：可在 worldsheet CFT 或低能有效理论中直接证明。
2. External input：已有标准理论证明，但证明不属于本书主线。
3. Physical conjecture：由多重检验支持，但不作为数学定理使用。

## 14.2 S-duality

**定义 14.3（S-duality）.** S-duality 指把耦合常数映到倒数或 fractional linear transform 的等价关系。基本例子为
$$
g_s\longleftrightarrow \frac1{g_s}.
$$

**物理猜想 14.4（type IIB $SL(2,\mathbb Z)$）.** Type IIB string theory 具有 $SL(2,\mathbb Z)$ duality，作用在 axio-dilaton
$$
\tau=C_0+ie^{-\Phi}
$$
上：
$$
\tau\mapsto\frac{a\tau+b}{c\tau+d},
\qquad
\begin{pmatrix}a&b\\c&d\end{pmatrix}\in SL(2,\mathbb Z).
$$

**命题 14.5（IIB field doublets）.** 在 type IIB S-duality 下，NS-NS two-form $B_2$ 与 R-R two-form $C_2$ 组成 doublet；fundamental strings 与 D1-branes 组成 $(p,q)$ strings。

**证明草图.** 低能 type IIB supergravity 的 equations of motion 可写成 $SL(2,\mathbb R)$ covariant form，量子 charge quantization 将其限制为 $SL(2,\mathbb Z)$。$B_2,C_2$ 与其电荷源 F1/D1 因而按 doublet 变换。完整量子等价是物理猜想 14.4 的一部分。$\square$

**物理猜想 14.6（type I/heterotic $SO(32)$ duality）.** Type I string theory 与 $SO(32)$ heterotic string 由 S-duality 相关，耦合满足
$$
g_{\mathrm I}\sim \frac1{g_{\mathrm{het}}}.
$$

## 14.3 M-theory 与 type IIA 强耦合

**物理猜想 14.7（IIA strong coupling limit）.** Type IIA string theory 的强耦合极限由十一维 M-theory 在圆上紧化描述，并满足
$$
R_{11}=g_s\ell_s,\qquad
\ell_{11}=g_s^{1/3}\ell_s,
$$
其中 $\ell_s=\sqrt{\alpha'}$。

**命题 14.8（D0-branes as KK modes）.** Type IIA D0-branes 的质量与十一维圆上的 Kaluza-Klein momentum spectrum 匹配。

**证明.** D0-brane 质量为
$$
M_{D0}=\frac1{g_s\ell_s}.
$$
十一维半径为 $R_{11}=g_s\ell_s$ 的圆上，单位 KK momentum 的质量为
$$
M_{KK}=\frac1{R_{11}}=\frac1{g_s\ell_s}.
$$
两者相等。$\square$

**命题 14.9（M-branes 的约化）.** M2-brane 与 M5-brane 在 $S^1_{11}$ 上约化时给出 IIA 中的 fundamental string、D2-brane、D4-brane 和 NS5-brane。

**证明草图.** M2 若包裹 $S^1_{11}$，其 worldvolume 降一维，得到 IIA fundamental string；不包裹则为 D2-brane。M5 若包裹圆，得到 D4-brane；不包裹则为 NS5-brane。张力标度在 $\ell_{11}=g_s^{1/3}\ell_s$ 与 $R_{11}=g_s\ell_s$ 下匹配。$\square$

## 14.4 U-duality

**物理猜想 14.10（U-duality）.** Type II string theory 在 torus $T^d$ 上紧化后具有离散 U-duality group $E_{d(d)}(\mathbb Z)$ 的作用，统一 T-duality 与 S-duality。

**注 14.11.** U-duality 的连续版本出现在低能 maximal supergravity 的 classical equations 中；量子 theory 只保留 charge lattice 的 arithmetic subgroup。

## 14.5 Brane web 和 protected tests

**定义 14.12（protected quantity）.** Protected quantity 是由 supersymmetry、topology 或 anomaly 限制而不随 coupling 连续改变的量，例如 BPS index、anomaly coefficient、charge lattice pairing。

**命题 14.13（duality 检验原则）.** 若两个 dual descriptions 正确，则它们必须匹配 protected spectra、charge lattice、anomaly data 和低能有效作用中的受保护耦合。

**证明草图.** Duality 是同一量子理论的两种描述。Protected quantities 在连续改变 coupling 时不跳变，因此可在弱耦合区域计算并延拓到另一描述。若这些量不匹配，则 duality dictionary 不可能成立。$\square$

## 14.6 张力匹配例子

**命题 14.14（M2 wrapped on circle gives F1 tension）.** 若 M2-brane 包裹十一维圆 $S^1_{11}$，则所得 IIA string 的张力与 fundamental string tension 匹配：
$$
T_{\mathrm{F1}}=\frac1{2\pi\alpha'}.
$$

**证明草图.** M2-brane 张力为
$$
T_{\mathrm{M2}}=\frac1{(2\pi)^2\ell_{11}^3}.
$$
包裹半径 $R_{11}=g_s\ell_s$ 的圆后，
$$
T_{\mathrm{wrapped}}
=2\pi R_{11}T_{\mathrm{M2}}
=\frac{R_{11}}{2\pi\ell_{11}^3}.
$$
用 $\ell_{11}=g_s^{1/3}\ell_s$ 得
$$
T_{\mathrm{wrapped}}=\frac1{2\pi\ell_s^2}
=\frac1{2\pi\alpha'}.
$$
$\square$

**命题 14.15（D3-brane 的 S-duality 自洽性）.** Type IIB D3-brane 在 $SL(2,\mathbb Z)$ 下映到自身，其 worldvolume $\mathcal N=4$ gauge theory 的 electric-magnetic duality 与 bulk S-duality 相容。

**证明草图.** D3-brane 耦合于 self-dual five-form，其 R-R charge 在 IIB S-duality 下不与 NS-NS charge 混合成不同维度 brane。Worldvolume gauge coupling 由 axio-dilaton 控制，因此 $SL(2,\mathbb Z)$ 作用为四维 gauge theory 的 Montonen-Olive 型 duality。完整量子等价仍是物理猜想的一部分。$\square$

## 本章小结

Duality 是 string theory 从多个微扰展开走向非微扰结构的核心机制。本书把 T-duality、S-duality、U-duality 和 M-theory 分层处理：可证明处给证明，依赖外部理论处标明输入，非微扰整体等价保持为物理猜想。

## 练习

**练习 14.1.** 比较 T-duality 与 S-duality 的耦合常数行为。

**练习 14.2.** 用 D0-brane 质量推导 $R_{11}=g_s\ell_s$。

**练习 14.3.** 用 M2-brane 张力推导 fundamental string tension。

