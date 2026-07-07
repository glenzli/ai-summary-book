# 第九章：Green-Schwarz 形式、type II strings 和 spacetime supersymmetry

## 本章目标

本章把第八章的 RNS 描述转换为 target-space supersymmetry 的语言。核心问题是：

1. type IIA 与 type IIB 如何由左右 Ramond chirality 区分；
2. massless type II spectrum 如何组织为十维 supergravity multiplets；
3. Green-Schwarz formalism 如何显式呈现 spacetime supersymmetry；
4. kappa symmetry 如何移除一半 fermionic degrees of freedom。

本章不把十维 supergravity 完整分类作为独立目标；只保留 string theory 主线所需的谱、对称性和有效理论接口。

## 依赖前置知识

需要第八章 RNS 超弦、附录 E 的十维 spinor 约定，以及第十一章低能有效作用中的 supergravity 接口。

## 9.1 Type II superstrings

**定义 9.1（type II theories）.** Type II superstrings 是 closed oriented superstrings。RNS 语言中，左右移动 sectors 均为 ten-dimensional RNS matter theory，并分别做 GSO projection。

左右 Ramond ground states 的 chirality 区分两种理论：

1. type IIA：left 与 right Ramond ground states 手征性相反；
2. type IIB：left 与 right Ramond ground states 手征性相同。

**命题 9.2（IIA/IIB 的低能 supergravity 类型）.** Type IIA 的十维低能极限为非手征 $N=2$ supergravity；type IIB 的十维低能极限为手征 $N=2$ supergravity。

**证明草图.** R sector ground states 是十维 Majorana-Weyl spinors。IIA 中左右 spinors chirality 相反，因此两个 supersymmetry generators chirality 相反，理论非手征。IIB 中左右 chirality 相同，因此两个 supersymmetry generators chirality 相同，理论手征。低能极限只保留 massless string states，正好组织为相应 $N=2$ supergravity multiplets。$\square$

**命题 9.3（R-R potentials 的次数 parity）.** 在民主形式记号中，type IIA 的 R-R field strengths 为偶次数，R-R potentials 为奇次数；type IIB 的 R-R field strengths 为奇次数，R-R potentials 为偶次数，并且五形式场强满足 self-duality condition。

**证明草图.** R-R states 来自左右 Ramond spinors 的张量积。十维 Clifford algebra 给出 spinor bilinear 与 antisymmetric forms 的对应。相反 chirality 的 spinor bilinear 分解为奇次数 potentials，相同 chirality 的 spinor bilinear 分解为偶次数 potentials。IIB 的中间次数五形式由 chirality 和 field equation 共同施加 self-duality。$\square$

## 9.2 Massless spectrum

**定义 9.4（NS-NS 与 R-R sectors）.** Type II massless closed-string states 分为
$$
\mathrm{NS\text{-}NS}\oplus
\mathrm{NS\text{-}R}\oplus
\mathrm{R\text{-}NS}\oplus
\mathrm{R\text{-}R}.
$$

**命题 9.5（type II massless bosons）.** NS-NS sector 包含
$$
g_{\mu\nu},\qquad B_{\mu\nu},\qquad \Phi.
$$
R-R sector 包含 differential-form gauge potentials：

- IIA：$C_1,C_3$，民主记号中还包括其 Hodge dual potentials；
- IIB：$C_0,C_2,C_4$，其中 $F_5$ 受 self-duality constraint。

**证明草图.** NS-NS sector 与玻色闭弦第一激发层相同，张量分解给出 symmetric traceless、antisymmetric 和 trace。R-R sector 由左右 spinor ground states 的张量积给出；用 gamma matrices 把 spinor bilinear 展开为 forms，再按 chirality projection 选择允许的次数。$\square$

**注 9.6（fermionic fields）.** NS-R 与 R-NS sectors 给出 gravitini 和 dilatini。它们的 chirality pattern 与 IIA/IIB 的 supersymmetry generators 一致。

**定义 9.6A（massless field table）.** Type II massless bosonic fields 可按下表组织：

| sector | IIA | IIB |
|---|---|---|
| NS-NS | $g_{\mu\nu},B_2,\Phi$ | $g_{\mu\nu},B_2,\Phi$ |
| R-R potentials | $C_1,C_3$ | $C_0,C_2,C_4^+$ |
| dual field strengths | $F_2,F_4,F_6,F_8$ | $F_1,F_3,F_5,F_7,F_9$ |

其中 $C_4^+$ 表示其 field strength $F_5$ 满足 self-duality constraint。

**命题 9.6B（degrees of freedom matching）.** Type II massless bosonic 与 fermionic on-shell degrees of freedom 相等。

**证明草图.** 十维 massless little group 为 $SO(8)$。NS-NS sector 给出 $8_v\otimes8_v$ 的分解；R-R sector 给出左右 spinor 表示的张量积。GSO projection 选择 chirality 后，bosonic sectors 的总维数与 NS-R/R-NS fermionic sectors 的总维数匹配。这是 spacetime supersymmetry 的谱层面检验。$\square$

## 9.3 Green-Schwarz target superspace

**定义 9.7（target superspace coordinates）.** Green-Schwarz formalism 使用 target superspace coordinates
$$
Z^M=(X^\mu,\theta^A_\alpha),
$$
其中 $A=1,2$ 标记两份十维 Majorana-Weyl spinor。IIA 中 $\theta^1,\theta^2$ chirality 相反；IIB 中二者 chirality 相同。

**定义 9.8（GS action 的结构）.** 平坦背景中的 Green-Schwarz action 由 kinetic term 和 Wess-Zumino term 组成：
$$
S_{\mathrm{GS}}=S_{\mathrm{kin}}+S_{\mathrm{WZ}}.
$$
它具有：

1. worldsheet diffeomorphism；
2. Weyl symmetry；
3. global target-space supersymmetry；
4. local fermionic kappa symmetry。

更具体地，平坦 superspace 中定义 supersymmetric one-form
$$
\Pi_a^\mu=\partial_aX^\mu
-i\bar\theta^A\Gamma^\mu\partial_a\theta^A.
$$
Kinetic term 具有形式
$$
S_{\mathrm{kin}}
=-\frac1{2\pi\alpha'}\frac12
\int d^2\sigma\sqrt{-h}\,h^{ab}\Pi_a^\mu\Pi_{b\mu}.
$$
Wess-Zumino term 的精确符号依赖 IIA/IIB chirality convention，其作用是同时保证 target-space supersymmetry 与 kappa symmetry。

**外部输入定理 9.9（GS action 的 kappa symmetry）.** 十维平坦背景中，GS action 的 Wess-Zumino term 可选取为使 action 在 kappa transformation 下不变。Kappa symmetry 消去 $\theta$ 中一半 fermionic components，使物理 transverse bosons 与 fermions 数目匹配。

**使用边界.** 本书不证明 kappa-symmetry variation 的完整 gamma-matrix 恒等式；只使用其自由度计数和 light-cone gauge 后的谱等价。

## 9.4 Light-cone gauge 自由度计数

**命题 9.10（GS light-cone 自由度匹配）.** 在十维平坦背景中，light-cone gauge 下 GS string 的物理横向 bosonic degrees of freedom 为 $8$，fermionic degrees of freedom 也为 $8$。

**证明草图.** Worldsheet diffeomorphism 和 Virasoro constraints 消去 $X^\pm$，留下 $D-2=8$ 个 transverse coordinates $X^i$。十维 Majorana-Weyl spinor 初始有 $16$ 个实分量，kappa symmetry 消去一半，light-cone condition 再给出 $8$ 个物理 fermionic components。$\square$

**注 9.11（RNS 与 GS 的互补性）.** RNS formalism 使 worldsheet superconformal symmetry 和 covariant quantization 更直接；GS formalism 使 spacetime supersymmetry 更直接，但 covariant quantization 较困难。Pure spinor formalism 可视为二者之间的现代桥梁，本书只在后续接口处提及。

**命题 9.11A（light-cone GS spectrum 的基态）.** Light-cone GS quantization 中，横向 bosons $X^i$ 与横向 fermions $S^a$ 的 oscillator Fock space 直接生成 type II supergravity massless multiplet。

**证明草图.** Light-cone gauge 中物理 fields 只带 $SO(8)$ transverse indices。Fermion zero modes 生成 $SO(8)$ Clifford algebra，其 ground-state representation 与 bosonic oscillator first levels 组合为 type II supergravity multiplet。IIA/IIB 的差异由左右 fermion zero-mode chirality 决定。$\square$

## 9.5 Supersymmetry algebra 与 BPS bound

**定义 9.12（central charges）.** Type II supersymmetry algebra 可含有由 extended objects 携带的 form-valued central charges。这些 charges 与 D-branes、NS5-branes 和 fundamental strings 的存在相容。

**命题 9.13（BPS bound 的接口）.** 若态携带 brane charge $Z$，supersymmetry algebra 给出质量下界
$$
M\ge |Z|.
$$
饱和该界的态保持部分 supersymmetry，其质量和 charge 在连续耦合变化下受保护。

**证明草图.** Supersymmetry algebra 的 anticommutator $\{Q,Q^\dagger\}$ 是正算子。把 central charge 对角化后，正性给出 $M-|Z|\ge0$。饱和时某些 supercharges 湮灭该态，故得到 shortened multiplet。$\square$

**例 9.14（D$p$-brane 与 R-R charge）.** Type II 中 D$p$-brane 电耦合于 R-R potential $C_{p+1}$。因此 IIA 只允许偶维空间 branes $p=0,2,4,6,8$，IIB 只允许奇维空间 branes $p=-1,1,3,5,7,9$，与第 12 章 WZ coupling 的 form degree 相容。

**注 9.15（type IIB self-dual five-form）.** IIB supergravity 的 $F_5=*F_5$ 不能由普通 covariant action 直接无冗余推出。实践中常先写 pseudo-action，再在 equations of motion 层面施加 self-duality constraint。

## 本章小结

Type II strings 的核心区别是左右 Ramond chirality。RNS formalism 给出可量子化的 worldsheet 描述；GS formalism 显式呈现 spacetime supersymmetry，并通过 kappa symmetry 给出正确自由度计数。BPS 结构为 D-branes、duality 和黑洞微观计数提供后续主线。

## 练习

**练习 9.1.** 说明 type IIA 与 type IIB 的 R-R potential 次数 parity 为什么不同。

**练习 9.2.** 用 light-cone gauge 计数十维 GS string 的 transverse bosons 与 fermions。

**练习 9.3.** 根据 R-R potential 的 form degree 判断 IIA/IIB 中允许的 D-brane parity。

