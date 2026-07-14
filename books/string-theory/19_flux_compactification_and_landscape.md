# 第十九章：flux compactifications、moduli stabilization 和 landscape 边界

无 flux 的 Calabi--Yau 紧化留下连续 massless moduli，因而还没有选出离散真空。
允许量子化的 $p$-form flux 后，内部拓扑整数进入四维 superpotential；在 type IIB
orientifold 的可控近似中，GVW 项通常固定 complex structure 与 axio-dilaton，却因
no-scale identity 暂时不固定 Kahler moduli。Tadpole、warping、backreaction 与量子
修正决定这种结论能否维持。以下在第十三章紧化、第十一章有效作用与第十四章 duality
口径上，推导 flux quantization、GVW F-terms 和 tadpole constraint，再比较 KKLT/
large-volume 等 Kahler stabilization 机制。所谓 landscape 或 swampland 只在明确
控制参数与猜想状态后讨论，不代替具体解的存在性和稳定性分析。

## 19.1 Flux quantization

**定义 19.1（flux compactification）.** Flux compactification 指在内部空间上打开 $p$-form field strength 的 cohomology class：
$$
\frac1{(2\pi\ell_s)^{p-1}}\int_{\Gamma_p}F_p\in\mathbb Z
$$
或相应 shifted quantization lattice，其中 $\Gamma_p\in H_p(X,\mathbb Z)$。

**命题 19.2（flux data 离散化 potential）.** Flux quantum numbers 是离散数据；在固定 flux sector 中，它们作为参数进入低维有效势能。

**证明.** Flux quantization 把 field strength 的 harmonic cohomology component 限制在 integral lattice 上。将高维 action 中的 $|F_p|^2$ 项在内部空间积分，得到依赖 moduli 和整数 flux quanta 的四维 scalar potential。$\square$

## 19.2 Type IIB GVW superpotential

**定义 19.3（IIB three-form flux）.** Type IIB Calabi-Yau orientifold compactification 中定义
$$
G_3=F_3-\tau H_3,
\qquad
\tau=C_0+ie^{-\Phi}.
$$

**外部输入定理 19.4（Gukov-Vafa-Witten superpotential）.** Type IIB flux compactification 的 complex structure moduli 与 axio-dilaton superpotential 为
$$
W_{\mathrm{GVW}}=\int_X \Omega\wedge G_3.
$$

**命题 19.5（complex structure stabilization 的接口）.** F-term equations
$$
D_iW_{\mathrm{GVW}}=0
$$
可固定 complex structure moduli 和 axio-dilaton，但在 tree-level no-scale 结构中通常不固定全部 Kahler moduli。

**推导说明（标准物理口径）.** $\Omega$ 依赖 complex structure，$G_3$ 依赖 $\tau$，故 $W_{\mathrm{GVW}}$ 对这些 fields 非平凡。Kahler moduli 在 tree-level Kähler potential 中出现，但 superpotential 不依赖它们；no-scale identity 使其势能方向在最低阶保持平坦。$\square$

## 19.3 Tadpole cancellation

**定义 19.6（D3 tadpole）.** IIB/F-theory compactification 中，three-form flux 携带 D3-brane charge：
$$
N_{\mathrm{flux}}
\propto
\int_X H_3\wedge F_3.
$$
一致紧化要求它与 localized sources 和 curvature contributions 满足 tadpole cancellation。

**命题 19.7（tadpole constraint 限制 flux choices）.** Tadpole cancellation 给 flux lattice 中可取整数点施加上界，因此受控模型中的 flux choices 是有限或在给定约束下有限可枚举的。

**推导说明（标准物理口径）.** $N_{\mathrm{flux}}$ 是 flux quanta 的二次型。若总 tadpole bound 固定为 $L$，则要求 $N_{\mathrm{flux}}\le L$。在正定或适当物理允许区域内，满足该二次约束的 lattice points 有限。$\square$

**例 19.7A（F-theory D3 tadpole）.** 在 F-theory on Calabi-Yau fourfold $Y_4$ 的常见规范中，D3 tadpole condition 写作
$$
N_{D3}+\frac12\int_{Y_4}G_4\wedge G_4
=\frac{\chi(Y_4)}{24}.
$$
这给出 flux choices 的上界，并约束可加入的 mobile D3-branes 数量。

## 19.4 Kahler moduli stabilization

**定义 19.8（nonperturbative superpotential）.** Kahler moduli 可通过 Euclidean D-brane instantons 或 gaugino condensation 产生非微扰项，例如
$$
W=W_0+Ae^{-aT}.
$$

**注 19.9（KKLT 与 LVS）.** KKLT 和 Large Volume Scenario 是两类常见机制。它们依赖 flux 先固定 complex structure 和 dilaton，再由 nonperturbative effects 与 $\alpha'$ corrections 处理 Kahler moduli。具体 vacuum 的存在性和控制条件必须逐模型检查。

**命题 19.9A（no-scale potential）.** 若 tree-level superpotential 不依赖 Kahler moduli，且 Kahler potential 满足 no-scale identity，则四维 F-term potential 中 Kahler moduli 方向在最低阶不产生势能。

**推导说明（标准物理口径）.** 四维 $\mathcal N=1$ supergravity potential 为
$$
V=e^K\left(K^{I\bar J}D_IW D_{\bar J}\overline W-3|W|^2\right).
$$
若 $W$ 与 Kahler moduli $T^a$ 无关，且
$$
K^{a\bar b}K_aK_{\bar b}=3,
$$
则 Kahler sector 的正项抵消 $-3|W|^2$，留下 complex structure/dilaton F-terms。$\square$

## 19.5 GKP supersymmetry 条件

**命题 19.13（ISD flux condition）.** Type IIB warped Calabi-Yau flux compactification 中，保持四维 Poincare invariance 和 supersymmetry 的三形式 flux 满足 imaginary self-dual 条件
$$
*_6G_3=iG_3
$$
并且 supersymmetric Minkowski 解要求 $G_3$ 的 Hodge type 为 primitive $(2,1)$。

**推导说明（标准物理口径）.** 十维 gravitino/dilatino supersymmetry variations 将 flux 的 $SU(3)$ representation 分量投影。未破缺 supersymmetry 排除 $(3,0)$ 和 non-primitive 分量，保留 primitive $(2,1)$；对应地 $G_3$ 满足 ISD。完整推导依赖 type IIB supersymmetry variations。$\square$

## 19.6 Landscape 边界

**原则 19.14（landscape statement 的分层）.** Landscape 讨论必须区分：

1. 已构造的具体 compactification data；
2. 低能有效理论中的临界点；
3. 在 large volume、small coupling、低曲率下受控的 vacuum；
4. 依赖未控制修正的推测性区域。

**原则 19.15（控制条件）.** 本书接受的 flux compactification 陈述必须至少说明：

1. flux quantization；
2. tadpole cancellation；
3. moduli stabilization mechanism；
4. $g_s$ 与 $\alpha'$ corrections 的控制；
5. backreaction 与 scale separation 的假设。

## 19.7 Swampland 接口

**注 19.16.** Swampland program 试图区分可由 quantum gravity 完成的低能有效理论与不可能的低能理论。本书不展开 swampland conjectures，只把它们作为第二十章外部接口。任何 swampland 陈述必须标明是 conjecture、evidence 还是 theorem。

No-scale 计算把结论的层级划得很清楚：量子化 flux 产生离散数据，GVW
superpotential 在 tree level 固定一部分 moduli，Kahler directions 则需
$\alpha'$、nonperturbative effects 或其他 ingredients 才可能被提升。每一步都要再
检查 tadpole、backreaction、scale separation 与 metastability；仅写出一个形式
potential 还不是受控真空。Landscape 是这些离散选择的统计问题，swampland 是关于
量子引力可完成性的猜想集合，二者都不能越过上述解方程与误差估计。

## 练习

**练习 19.1.** 说明 flux quantization 为什么使连续 moduli potential 依赖离散数据。

**练习 19.2.** 解释为什么 GVW superpotential 通常先固定 complex structure moduli 而不是 Kahler moduli。

**练习 19.3.** 用 no-scale identity 说明 tree-level flux potential 为什么不固定 Kahler moduli。
