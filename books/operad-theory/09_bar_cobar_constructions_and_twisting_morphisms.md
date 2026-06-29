# 第九章：Bar-cobar 构造与 twisting morphism

## 本章目标

本章建立同伦代数所需的 dg-operad 工具。核心目标是：

1. 固定链复形和 Koszul 符号约定。
2. 定义 dg 对称序列、dg-operad 和 dg-cooperad。
3. 定义 cofree conilpotent cooperad。
4. 定义 twisting morphism 及其 Maurer-Cartan 方程。
5. 构造 cobar functor $\Omega$ 和 bar functor $B$。
6. 证明 twisting morphism、cobar 映射和 bar 映射之间的自然对应。

本章的重点是定义和泛性质。Quillen 等价、Koszul 判别和具体同伦 operad 的 quasi-isomorphism 留到后续章节，并作为外部输入定理使用。

## 依赖前置知识

需要定义 6.6 的线性 operad、定义 8.4 和定义 8.16 的二次/Koszul 语言，以及链复形、张量积、微分和同调的基本知识。

## 9.1 链复形与符号约定

**约定 9.1.** 本章固定域 $k$。链复形采用同调次数：
$$
d:C_n\to C_{n-1}.
$$
若 $x$ 是齐次元素，其次数写作 $|x|$。链复形张量积的微分为
$$
d(x\otimes y)=d x\otimes y+(-1)^{|x|}x\otimes d y.
$$
对称 braiding 为
$$
\tau(x\otimes y)=(-1)^{|x||y|}y\otimes x.
$$
这些符号统称为 Koszul sign rule。

**定义 9.2.** 悬挂 $sC$ 与去悬挂 $s^{-1}C$ 定义为
$$
(sC)_n=C_{n-1},\qquad (s^{-1}C)_n=C_{n+1}.
$$
若 $x\in C$，则 $sx\in sC$ 的次数为 $|x|+1$，$s^{-1}x\in s^{-1}C$ 的次数为 $|x|-1$。微分由
$$
d(sx)=-s(dx),\qquad d(s^{-1}x)=-s^{-1}(dx)
$$
定义。

**命题 9.3.** $s$ 与 $s^{-1}$ 是互逆的链复形自等价，精确到典范自然同构。

**证明.** 在底层分次向量空间上，$s^{-1}sC$ 和 $C$ 有相同次数部分。微分计算为
$$
d(s^{-1}sx)=-s^{-1}d(sx)=s^{-1}s(dx),
$$
与 $C$ 上的微分一致。$ss^{-1}C$ 同理。$\square$

## 9.2 dg 对称序列、operad 与 cooperad

**定义 9.4.** 一个 dg 对称序列是函子
$$
M:\mathbf B_{\mathcal U}\to\mathbf{Ch}_k.
$$
dg 对称序列的代入乘积仍记为 $\circ$，由命题 6.7 的 arity 公式在 $\mathbf{Ch}_k$ 中解释：
$$
(M\circ N)(S)
=
\bigoplus_{\pi\in\operatorname{Part}(S)}
M(\operatorname{Bl}(\pi))
\otimes
\bigotimes_{B\in\operatorname{Bl}(\pi)}N(B),
$$
其中有限张量积使用约定 9.1 的微分和 braiding。

**命题 9.5.** dg 对称序列范畴连同 $\circ$ 与单位 $I_k$ 构成幺半范畴。

**证明.** 证明与命题 6.5 相同。唯一新增点是：在重排多个链复形张量因子时使用 Koszul braiding，因此所有分块重标号映射都是链映射。$\mathbf{Ch}_k$ 的对称幺半相干性保证结合约束和单位约束满足 Mac Lane 相干图。$\square$

**定义 9.6.** dg-operad 是幺半范畴
$$
(\operatorname{SymSeq}(\mathbf{Ch}_k),\circ,I_k)
$$
中的幺半对象。dg-cooperad 是该幺半范畴中的余幺半对象，即 dg 对称序列 $\mathcal C$ 配有分解映射
$$
\Delta:\mathcal C\to\mathcal C\circ\mathcal C
$$
和余单位
$$
\epsilon:\mathcal C\to I_k,
$$
满足余结合律和余单位律。

**定义 9.7.** 一个 augmented dg-operad 是 dg-operad $\mathcal P$ 连同 operad morphism
$$
\epsilon:\mathcal P\to I_k.
$$
其增广理想为
$$
\overline{\mathcal P}=\ker(\epsilon).
$$
一个 coaugmented dg-cooperad 是 dg-cooperad $\mathcal C$ 连同 cooperad morphism
$$
\eta:I_k\to\mathcal C.
$$
其 coaugmentation coideal 为
$$
\overline{\mathcal C}=\operatorname{coker}(\eta).
$$

本章默认 augmented operad 与 coaugmented cooperad 都是 reduced：
$$
\mathcal P(0)=0,\quad \mathcal P(1)\cong k\oplus\overline{\mathcal P}(1),
$$
并且 cooperad 有对应的 coaugmentation 分解。

## 9.3 Cofree conilpotent cooperad

**定义 9.8.** 设 $M$ 是 dg 对称序列。cofree conilpotent cooperad $\mathbb T^c(M)$ 定义为装饰有根树的 dg 对称序列：
$$
\mathbb T^c(M)(S)
=
\bigoplus_{[T]}
\bigotimes_{v\in V(T)}M(\operatorname{In}(v)),
$$
其中 $[T]$ 遍历有限 $S$-标号有根树的同构类；没有内部顶点的单位树给出 coaugmentation。若 $M(r)=0$，则含有 $r$ 输入顶点的树对直和没有贡献。树同构按输入边集合上的双射作用于装饰。

Cooperad 分解由切割内部边给出：一次切割若干内部边，把树分解为一个外层商树和若干内层子树；对应项落入
$$
\mathbb T^c(M)\circ\mathbb T^c(M).
$$

**命题 9.9.** $\mathbb T^c(M)$ 是 conilpotent coaugmented dg-cooperad。

**证明.** 余单位取只含一个顶点或单位树的相应投影，coaugmentation 由单位树给出。余结合律断言两次切割内部边与一次性记录两层切割给出同一结果；这是有限树中边子集分层的结合律。Conilpotence 来自顶点数滤过：一个有 $r$ 个内部顶点的树最多只能被非平凡分解 $r-1$ 次。微分逐顶点作用，和切割内部边交换，因此分解映射是链映射。$\square$

**命题 9.10.** $\mathbb T^c(M)$ 满足如下泛性质：若 $\mathcal C$ 是 conilpotent coaugmented dg-cooperad，则给出 cooperad morphism
$$
F:\mathcal C\to\mathbb T^c(M)
$$
等价于给出 dg 对称序列映射
$$
f:\overline{\mathcal C}\to M.
$$

**证明.** 给定 $F$，复合
$$
\overline{\mathcal C}\to\overline{\mathbb T^c(M)}\to M
$$
给出 $f$，其中第二个箭头投影到单顶点树部分。

反过来，给定 $f$，对 $c\in\overline{\mathcal C}$ 反复使用 cooperad 分解，得到有限和的多顶点张量；conilpotence 保证该过程在有限步后停止。然后对每个顶点应用 $f$，得到 $\mathbb T^c(M)$ 中的树装饰元素。余结合律保证这一定义与切割树的顺序无关，并且给出 cooperad morphism。两个构造互逆。$\square$

## 9.4 Infinitesimal composition 与 convolution pre-Lie 结构

**定义 9.11.** 对 dg 对称序列 $M,N$，定义 infinitesimal composition product $M\circ_{(1)}N$ 为 $M\circ N$ 中只允许恰有一个块被 $N$ 装饰、其他块由单位 $I_k$ 装饰的直和部分。直观上，$M\circ_{(1)}N$ 只记录“把一个 $N$-运算代入一个 $M$-运算的一个输入槽”。

若 $\mathcal P$ 是 dg-operad，其 infinitesimal composition 写作
$$
\gamma_{(1)}:\mathcal P\circ_{(1)}\mathcal P\to\mathcal P.
$$
若 $\mathcal C$ 是 dg-cooperad，其 infinitesimal decomposition 写作
$$
\Delta_{(1)}:\mathcal C\to\mathcal C\circ_{(1)}\mathcal C,
$$
即把完整分解投影到恰有一个非单位内层部分的直和。

**定义 9.12.** 设 $\mathcal C$ 是 dg-cooperad，$\mathcal P$ 是 dg-operad。对齐次映射
$$
f,g\in\operatorname{Hom}_{\mathbb S}(\mathcal C,\mathcal P)
$$
定义 convolution pre-Lie product
$$
f\star g
=
\gamma_{(1)}
\circ
(f\circ_{(1)}g)
\circ
\Delta_{(1)}.
$$
这里 $(f\circ_{(1)}g)$ 的符号由链复形张量积中的 Koszul rule 决定。

**命题 9.13.** graded commutator
$$
[f,g]=f\star g-(-1)^{|f||g|}g\star f
$$
使 $\operatorname{Hom}_{\mathbb S}(\mathcal C,\mathcal P)$ 成为 dg Lie algebra。

**证明.** 微分与 $\star$ 的相容性来自 $\gamma_{(1)}$ 与 $\Delta_{(1)}$ 是链映射。pre-Lie 恒等式来自两次 infinitesimal 代入的两类相对位置：嵌套和分离。嵌套项由 operad 结合律与 cooperad 余结合律匹配；分离项在交换两次代入后相同，并由 Koszul braiding 给出符号。pre-Lie algebra 的 graded commutator 满足 Jacobi 恒等式，因此得到 dg Lie algebra。$\square$

## 9.5 Twisting morphism

**定义 9.14.** 设 $\mathcal C$ 是 coaugmented dg-cooperad，$\mathcal P$ 是 augmented dg-operad。一个 twisting morphism 是次数 $-1$ 的 $\mathbb S$-模映射
$$
\alpha:\mathcal C\to\mathcal P
$$
满足：

1. $\alpha$ 在 coaugmentation 和 augmentation 上为零，即 $\alpha\eta=0$ 且 $\epsilon\alpha=0$；
2. Maurer-Cartan 方程
   $$
   \partial(\alpha)+\alpha\star\alpha=0,
   $$
   其中
   $$
   \partial(\alpha)=d_{\mathcal P}\alpha+ \alpha d_{\mathcal C}
   $$
   因为 $|\alpha|=-1$。

twisting morphism 集合记为
$$
\operatorname{Tw}(\mathcal C,\mathcal P).
$$

**命题 9.15.** 若 $\alpha$ 是 twisting morphism，则由 $\alpha$ 定义的 twisted composite differential 在 $\mathcal C\circ\mathcal P$ 上平方为零。

**证明.** 在 $\mathcal C\circ\mathcal P$ 上，总微分由内部微分加上用 $\alpha$ 把 cooperad 一次分解中的一个因子送入 operad 并复合的项组成。平方后分成三类：内部微分平方为零；内部微分与 twisting 项的反交换给出 $\partial(\alpha)$ 项；两个 twisting 项的复合给出 $\alpha\star\alpha$ 项。Maurer-Cartan 方程正说明后两类相加为零。符号由约定 9.1 的 Koszul rule 给出。$\square$

**说明 9.15.1.** 定义 I.11--定义 I.18 把本节的 convolution Lie algebra、Maurer-Cartan 方程、twisted composite product 和 Koszul complex 统一成一个严格约定。后续若使用 $\mathcal C\circ_\alpha\mathcal P$、$\mathcal P\circ_\alpha\mathcal C$ 或 $\mathcal P^¡\circ_\kappa\mathcal P$，默认采用这些定义。

## 9.6 Cobar 构造

**定义 9.16.** 设 $\mathcal C$ 是 conilpotent coaugmented dg-cooperad。其 cobar 构造定义为 quasi-free dg-operad
$$
\Omega\mathcal C
=
\left(\mathbb F(s^{-1}\overline{\mathcal C}),d=d_1+d_2\right).
$$
其中：

- $d_1$ 是由 $\overline{\mathcal C}$ 的内部微分诱导到自由 operad 上的导子；
- $d_2$ 是由 infinitesimal decomposition
  $$
  \overline{\mathcal C}\xrightarrow{\Delta_{(1)}}\mathcal C\circ_{(1)}\mathcal C
  $$
  经去悬挂后得到的二次导子。

更具体地，$d_2$ 在生成元 $s^{-1}c$ 上是有限和
$$
d_2(s^{-1}c)=
\sum \pm
(s^{-1}c')\circ_i(s^{-1}c''),
$$
其中 $\Delta_{(1)}(c)$ 的相应项把 $c$ 分解为 $c'$ 与 $c''$。符号由把去悬挂符号穿过张量因子的 Koszul rule 决定。

**命题 9.17.** $\Omega\mathcal C$ 是 dg-operad，即 $d^2=0$。

**证明.** 因为 $d$ 是自由 operad 上的导子，只需在生成元 $s^{-1}\overline{\mathcal C}$ 上检查。$d_1^2=0$ 来自 $\mathcal C$ 的微分平方为零。$d_1d_2+d_2d_1=0$ 来自 $\Delta_{(1)}$ 是链映射。$d_2^2=0$ 来自 cooperad 余结合律：对一个元素作两次 infinitesimal decomposition 的两种方式给出同一三层分解，符号相反，因此相消。$\square$

## 9.7 Bar 构造

**定义 9.18.** 设 $\mathcal P$ 是 augmented dg-operad。其 bar 构造定义为 quasi-cofree dg-cooperad
$$
B\mathcal P
=
\left(\mathbb T^c(s\overline{\mathcal P}),d=d_1+d_2\right).
$$
其中：

- $d_1$ 是由 $\overline{\mathcal P}$ 内部微分诱导的 coderivation；
- $d_2$ 是由 operad infinitesimal composition
  $$
  \mathcal P\circ_{(1)}\mathcal P\to\mathcal P
  $$
  经悬挂后得到的 coderivation，它把一条内部边连接的两个顶点收缩为一个顶点。

**命题 9.19.** $B\mathcal P$ 是 conilpotent coaugmented dg-cooperad。

**证明.** 底层 cooperad $\mathbb T^c(s\overline{\mathcal P})$ 已由命题 9.9 给出。$d_1$ 保持树形并逐顶点作用。$d_2$ 收缩一条内部边，因此降低顶点数。$d^2=0$ 的检查分为三部分：$d_1^2=0$；$d_1d_2+d_2d_1=0$ 来自 operad 乘法是链映射；$d_2^2=0$ 来自 operad 结合律，因为收缩两条相邻内部边的两种顺序给出同一复合且符号相反。Conilpotence 仍由顶点数滤过给出。$\square$

## 9.8 Bar-cobar 伴随与泛性质

**定理 9.20.** 对 conilpotent coaugmented dg-cooperad $\mathcal C$ 和 augmented dg-operad $\mathcal P$，存在自然双射
$$
\operatorname{Hom}_{\mathrm{dgOp}}(\Omega\mathcal C,\mathcal P)
\cong
\operatorname{Tw}(\mathcal C,\mathcal P)
\cong
\operatorname{Hom}_{\mathrm{dgCoop}}(\mathcal C,B\mathcal P).
$$

**证明.** 先看左侧。因为 $\Omega\mathcal C$ 的底层 graded operad 是自由 operad $\mathbb F(s^{-1}\overline{\mathcal C})$，一个 graded operad morphism
$$
F:\Omega\mathcal C\to\mathcal P
$$
等价于次数 $0$ 的映射
$$
s^{-1}\overline{\mathcal C}\to\overline{\mathcal P},
$$
也等价于次数 $-1$ 的映射 $\alpha:\overline{\mathcal C}\to\overline{\mathcal P}$。要求 $F$ 与微分相容，等价于在生成元上满足
$$
d_{\mathcal P}F=F(d_1+d_2).
$$
把该等式通过去悬挂翻译回 $\alpha$，正得到
$$
\partial(\alpha)+\alpha\star\alpha=0.
$$
因此左侧与 twisting morphism 自然对应。

右侧同理。由 cofree conilpotent cooperad 的泛性质，graded cooperad morphism
$$
G:\mathcal C\to B\mathcal P=\mathbb T^c(s\overline{\mathcal P})
$$
等价于次数 $0$ 的映射
$$
\overline{\mathcal C}\to s\overline{\mathcal P},
$$
也等价于次数 $-1$ 的映射 $\alpha:\overline{\mathcal C}\to\overline{\mathcal P}$。链映射条件等价于 $B\mathcal P$ 中 coderivation 的两部分与 $\mathcal C$ 的分解相容，展开后也是 Maurer-Cartan 方程。$\square$

**推论 9.21.** Cobar functor 是 bar functor 的左伴随：
$$
\Omega:\mathrm{Coop}^{\mathrm{conil}}_{\mathrm{dg}}
\rightleftarrows
\mathrm{Op}^{\mathrm{aug}}_{\mathrm{dg}}:B.
$$

**证明.** 定理 9.20 给出自然双射
$$
\operatorname{Hom}_{\mathrm{dgOp}}(\Omega\mathcal C,\mathcal P)
\cong
\operatorname{Hom}_{\mathrm{dgCoop}}(\mathcal C,B\mathcal P),
$$
这正是伴随定义。$\square$

## 9.9 Koszul twisting morphism

**定义 9.22.** 若 $\mathcal P=\mathcal P(E,R)$ 是二次 operad，其 Koszul twisting morphism
$$
\kappa:\mathcal P^¡\to\mathcal P
$$
是次数 $-1$ 的映射，在由 $sE$ 或相应悬挂生成的权重 $1$ 部分上等于自然投影到 $E\subseteq\mathcal P$，在其他权重上为零。精确定义依赖定义 8.11 的符号扭曲和定义 9.2 的悬挂约定。

**外部输入定理 9.23.** 二次 operad $\mathcal P$ Koszul 当且仅当由 $\kappa$ 诱导的 morphism
$$
\Omega\mathcal P^¡\to\mathcal P
$$
是 quasi-isomorphism。该定理是 operadic Koszul duality 的核心形式之一，完整证明依赖 bar-cobar 理论和 weight spectral sequence。Ginzburg--Kapranov classical core 已定位为 GK-3 与 GK-7；Fresse modern twisting weak equivalence criterion 已定位为 FRE-1--FRE-3。把本书 $\Omega\mathcal P^¡$ 记号逐字匹配到 Loday--Vallette/Fresse 书本 convention 仍需最终 locator。

**说明 9.24.** 定义 I.20 和命题 I.21 采用权重滤过解释该判别定理的输入：cobar differential 的二次部分增加顶点数，bar differential 的二次部分降低顶点数。最终版引用 Koszul 判别时，必须同时说明 reduced 假设、有限型假设、conilpotence 和所用 suspension convention。

## 本章小结

Bar-cobar 理论把“同伦代数结构”转化为 twisting morphism 的 Maurer-Cartan 方程。Cobar 构造 $\Omega\mathcal C$ 是自由 operad 上加由 cooperad 分解诱导的微分；bar 构造 $B\mathcal P$ 是 cofree cooperad 上加由 operad 乘法诱导的微分。核心泛性质是
$$
\operatorname{Hom}_{\mathrm{dgOp}}(\Omega\mathcal C,\mathcal P)
\cong
\operatorname{Tw}(\mathcal C,\mathcal P)
\cong
\operatorname{Hom}_{\mathrm{dgCoop}}(\mathcal C,B\mathcal P).
$$
后续 $A_\infty$、$L_\infty$ 和 $E_n$-operad 的定义都将使用这一机制。

## 练习

**练习 9.1.** 验证悬挂微分 $d(sx)=-s(dx)$ 确实满足 $d^2=0$。

**练习 9.2.** 对一个有三个顶点的装饰树，写出 cofree cooperad 分解中切割一条内部边和切割两条内部边分别得到的项。

**练习 9.3.** 展开 convolution product $f\star g$ 在一个两层树分解上的符号来源。

**练习 9.4.** 证明若 $\alpha=0$，Maurer-Cartan 方程退化为 $\partial(\alpha)=0$，并解释这为什么对应平凡 twisting。

**练习 9.5.** 对一个 augmented dg-operad $\mathcal P$，说明 bar differential 中 $d_2$ 为什么降低树的顶点数。
