# 第六章：Homotopy purity、Thom spaces 与 purity transformations

闭嵌入 `i:Z\hookrightarrow X` 的开补商 `X/(X\setminus Z)` 看似依赖整个环境
`X`，而局部几何提示它应只由 `Z` 附近的一阶法向信息决定。Homotopy purity 将这一直觉
精确化：在光滑正则情形，闭补商与法丛的 Thom 空间等价。稳定以后，同一法向数据又
以 Thom twist 的形式进入 `i^!`、Gysin 映射和交叉理论。

需要特别警惕的是，非稳定 homotopy purity、smooth purity、regular immersion 的
purity transformation 与 absolute purity 并不是同一条定理。本章先在 pointed
motivic spaces 中定义 Thom 空间并计算零丛和平凡丛，再逐一写出这些 purity 陈述的
对象类型与假设。法丛和 lci 态射的代数几何背景可查附录 E，六操作方差沿用第五章。

## 6.1 Thom spaces

**定义 6.1.** 设 `p:V\to X` 是 `X` 上向量丛，零截面为 `s:X\hookrightarrow V`。Thom space 定义为 pointed motivic space

$$
\operatorname{Th}_X(V)=V/(V\setminus s(X)).
$$

若上下文清楚，写作 `\operatorname{Th}(V)`。

**命题 6.2.** 对零向量丛 `0_X`，有自然等价

$$
\operatorname{Th}(0_X)\simeq X_+.
$$

**证明.** 零向量丛的总空间是 `X`，零截面为恒等嵌入，其补集为空。因此

$$
\operatorname{Th}(0_X)=X/\varnothing.
$$

在 pointed motivic spaces 中，把空子对象压缩到基点等于添加一个不相交基点，故得到 `X_+`。`\square`

**命题 6.3.** 若 `L` 是 `X` 上平凡线丛，则

$$
\operatorname{Th}(L)\simeq T\wedge X_+.
$$

**证明.** 平凡线丛总空间为 `X\times\mathbb A^1`，零截面的补为 `X\times\mathbb G_m`。因此

$$
\operatorname{Th}(L)=(X\times\mathbb A^1)/(X\times\mathbb G_m).
$$

由于 smash product 与 colimits 相容，该商等价于

$$
X_+\wedge(\mathbb A^1/\mathbb G_m)=X_+\wedge T.
$$

对称性给出 `T\wedge X_+`。`\square`

**外部输入定理 6.4（Thom direct sum formula）.** 对 `X` 上向量丛 `V,W`，有自然等价

$$
\operatorname{Th}(V\oplus W)\simeq \operatorname{Th}(V)\wedge_X\operatorname{Th}(W)
$$

在适当的相对 pointed motivic category 中成立；稳定化后给出 Thom twists 的可加性。

**注 6.5.** 本书后续在稳定范畴中把向量丛 `V` 的 Thom twist 写作 `\Sigma^V`。若 `V-W` 是虚向量丛，则写 `\Sigma^{V-W}=\Sigma^V\Sigma^{-W}`。

## 6.2 Homotopy purity

**定义 6.6.** 若 `i:Z\hookrightarrow X` 是闭嵌入，且 `Z`、`X` 光滑，记 `N_{Z/X}` 为法丛。

**外部输入定理 6.7（Morel--Voevodsky homotopy purity）.** 设 `S` 为本书
默认基概形，`i:Z\hookrightarrow X` 是 `\operatorname{Sm}_S` 中的闭嵌入；
亦即 `X`、`Z` 均 smooth finite type over `S`，且 `i` 为闭嵌入。则
`i` 为 regular immersion，法丛 `N_{Z/X}` 为 `Z` 上向量丛，并在
`\mathbf H_*(S)` 中有关于 smooth closed pairs 自然的等价

$$
X/(X\setminus Z)\simeq \operatorname{Th}(N_{Z/X}).
$$

**依赖源与边界.** Morel--Voevodsky, *A1-homotopy theory of schemes*,
Section 3, Theorem 2.23。该定理陈述的是 unstable pointed motivic spaces
中的商对象等价；它没有陈述 arbitrary closed immersions 的 `i^!` 公式。
Equivariant/stacky/log 变体需要各自外部输入。

**命题 6.8.** 若 `i:X\hookrightarrow X` 为恒等闭嵌入，则定理 6.7 化为 `X_+\simeq X_+`。

**证明.** 恒等闭嵌入的开补为空，故左侧为 `X/\varnothing\simeq X_+`。法丛为零向量丛 `0_X`，由命题 6.2，右侧为 `\operatorname{Th}(0_X)\simeq X_+`。`\square`

**命题 6.9.** 若 `i:S\hookrightarrow\mathbb A^1_S` 是零截面，则定理 6.7 给出

$$
\mathbb A^1/(\mathbb A^1\setminus0)\simeq T.
$$

**证明.** 左侧按定义就是 `T`。零截面的法丛是 `S` 上的平凡线丛。由命题 6.3，其 Thom space 为 `T\wedge S_+\simeq T`。`\square`

## 6.3 稳定 purity transformations

**外部输入定理 6.10（Thom twists 的可逆性）.** 对 `X` 上向量丛 `V`，
在 `\mathbf{SH}(X)` 中写

$$
\operatorname{Th}_X(V)=\Sigma_T^\infty \operatorname{Th}(V)
$$

其像是张量可逆对象，并定义自等价

$$
\Sigma^V(E)=\operatorname{Th}_X(V)\otimes_X E.
$$

该构造关于 direct sum 可加，并延拓为 K-theory space 上的 motivic
J-homomorphism；故虚向量丛 `v=[V]-[W]` 的
`\Sigma^v=\Sigma^V\Sigma^{-W}` 不依赖所选表示，精确地说其选择空间可缩。

**依赖源与边界.** Hoyois, Proposition 6.5、Corollary 6.7（取平凡群）给出
motivic spheres 的可逆性和稳定化口径；完整 K-theory coherence 属于
motivic J-homomorphism 的外部理论。只知道 `T` 可逆不足以逐点构造任意
非平凡向量丛的 coherent Thom twist。

**定义 6.11（virtual tangent class）.** 若 `f:X\to Y` 是 smoothable lci
separated morphism，记

$$
\tau_f=\langle L_f\rangle\in K(X)
$$

为其 virtual tangent class，采用 Deglise--Jin--Khan 的约定：若 `f`
smooth，则 `\tau_f=[T_f]`；若 `i` 是 regular closed immersion，则
`\tau_i=-[N_i]`；若 `f=p\circ i`，其中 `i` regular closed、`p` smooth，
则 `\tau_f=i^*[T_p]-[N_i]`。该 K-theory class 与 factorization 无关属于
外部 lci cotangent-complex 理论。

**外部输入定理 6.12（同伦三角层的 lci purity transformation）.** 对每个
smoothable lci separated finite-type morphism `f:X\to Y`，有三角函子之间的
自然变换

$$
\mathfrak p_f(E):\Sigma^{\tau_f}f^*E\longrightarrow f^!E,
\qquad E\in\operatorname{Ho}(\mathbf{SH}(Y)),
$$

其分量是 `\operatorname{Ho}(\mathbf{SH}(X))` 中的态射。随着 `f` 变化，
identity 与可复合 smoothable lci morphisms 的相容性，是 Proposition 2.5.4(i)
在 homotopy categories 上给出的交换方块；Remark 2.5.5 按该文的口径把它
组织为相应 `\mathbf{Tri}` 值逆变伪函子之间的自然变换；这里
`\mathbf{Tri}` 是以大三角范畴为对象、三角函子为 1-morphisms、可逆三角
自然变换为 2-morphisms 的 `(2,1)`-范畴。对 Tor-independent Cartesian
squares 的 transverse base change，则指 Proposition 2.5.4(ii) 的三角自然
变换交换方块。

这里不声明这些相容性已经提升为稳定 infinity-范畴值函子之间的
infinity-natural transformation，也不声明多重复合或 base-change pasting 的
higher coherence。对非 Tor-independent square，本定理尤其不提供无修正
base-change 公式。只有当方块中 `f` 及其拉回 `g` 都是 smoothable lci
separated finite-type morphisms，且 Deglise--Jin--Khan Paragraph 3.3.3 的
法丛单射具有向量丛商 `\xi` 时，Propositions 3.3.4、4.2.2 才分别对
fundamental classes 和带 unital associative commutative multiplication
系数的 Gysin maps 给出 excess-Euler 修正；这不是任意 Cartesian square 的
purity-transformation coherence。

**依赖源与边界.** Deglise--Jin--Khan, *Fundamental classes in motivic
homotopy theory*, Theorem 3.3.2 与 Theorem 4.1.4 构造 fundamental classes；
Paragraph 4.3.1 把它们写成 purity transformations；Proposition 2.5.4 给出
复合和 transverse base-change 相容。Remark 2.5.5 明确以 Proposition
2.5.4(i) 为依据，只在 homotopy categories 上把复合数据组织为逆变伪函子间
的自然变换；base change 则保留 Proposition 2.5.4(ii) 的交换方块。该 remark
还说明所期待的 infinity-category 层增强需要额外工作且该文不予完成。因此
本书只导入上述 `\operatorname{Ho}(\mathbf{SH})` 层的相容性；另补一手
enhancement 来源之前，不把它升级为 higher-coherent coefficient system。
定理也不声称该 morphism 对任意 lci `f` 和任意系数 `E` 可逆。

**外部输入定理 6.13（smooth purity）.** 若 `f:X\to Y` smooth 且
separated，相对切丛为 `T_f`，则 `\tau_f=[T_f]`，并且在稳定
infinity-范畴层有自然等价

$$
\mathfrak p_f:\Sigma^{T_f}f^*\xrightarrow{\sim}f^!.
$$

**依赖源.** Hoyois, Theorem 6.18(2)，其中 twist 由 Proposition 5.7 的
purity 识别；Deglise--Jin--Khan Paragraph 2.5.2 与 Paragraph 4.3.1 说明
smooth fundamental class 在 homotopy category 中给出该等价的同一三角影子。

**定义 6.14（relative 与 absolute purity）.** 对 6.12 中的 `f`，对象
`E\in\mathbf{SH}(Y)` 称为 **`f`-pure**，若 `\mathfrak p_f(E)` 在
`\operatorname{Ho}(\mathbf{SH}(X))` 中为同构。
特别地，若 `i:Z\hookrightarrow X` regular closed，法丛为 `N_i`，则总有
`\operatorname{Ho}(\mathbf{SH}(Z))` 中的 canonical morphism

$$
\mathfrak p_i(E):\Sigma^{-N_i}i^*E\longrightarrow i^!E,
$$

但“`E` 为 `i`-pure”才表示这条 map 可逆。

更一般地，设 `E\in\mathbf{SH}(S)`。称 `E` **满足绝对纯性**，若对每个
交换三角

$$
\begin{array}{c}
X\overset f\longrightarrow Y\\
\searrow p\qquad\swarrow q\\
S,
\end{array}
$$

其中 `f,p,q` 均 separated finite type，`f` smoothable lci，且 `X,Y`
regular，对象 `q^*E\in\mathbf{SH}(Y)` 都是 `f`-pure。按
Deglise--Jin--Khan Remark 4.3.12(i)，利用 purity transformation 在
homotopy-category 伪函子层的复合性，可等价地只检查 regular closed
immersions between regular schemes。

**命题 6.15（单位纯性能够推出的范围）.** 设
`i:Z\hookrightarrow X` regular closed。若 `\mathbb 1_X` 为 `i`-pure，
则每个 dualizable `E\in\mathbf{SH}(X)` 都为 `i`-pure。仅有单位等价

$$
i^!\mathbb 1_X\simeq\Sigma^{-N_i}\mathbb 1_Z
$$

并不在纯形式上给出整个函子之间的等价
`i^!\simeq\Sigma^{-N_i}i^*`；对非 dualizable 系数仍需另证。

**证明.** 先证明 dualizable `E` 的 canonical exchange map

$$
i^*E\otimes i^!\mathbb 1_X\longrightarrow i^!E
$$

为等价。记 `E^\vee` 为对偶。对任意 `C\in\mathbf{SH}(Z)`，依次用
duality、`i_!\dashv i^!`、`!`-projection formula 和再次 duality，得到

$$
\begin{aligned}
\operatorname{Map}_Z(C,i^*E\otimes i^!\mathbb 1_X)
&\simeq\operatorname{Map}_Z(C\otimes i^*E^\vee,i^!\mathbb 1_X)\\
&\simeq\operatorname{Map}_X(i_!(C\otimes i^*E^\vee),\mathbb 1_X)\\
&\simeq\operatorname{Map}_X(i_!C\otimes E^\vee,\mathbb 1_X)\\
&\simeq\operatorname{Map}_X(i_!C,E)\\
&\simeq\operatorname{Map}_Z(C,i^!E).
\end{aligned}
$$

Yoneda lemma 给出该 exchange map 为等价。Purity transformation 关于基系数
的 module action 相容，所以在 homotopy category 中 `\mathfrak p_i(E)` 识别为
`i^*E\otimes\mathfrak p_i(\mathbb 1_X)` 后接上述 exchange equivalence。
若单位的 purity morphism 在 homotopy category 中为同构，则
`\mathfrak p_i(E)` 也为同构。最后一句指出
证明使用了 `E^\vee`；没有 dualizability 时该论证不存在。`\square`

## 6.4 Gysin maps 的形式来源

**定义 6.16.** 设 `E` 是 motivic ring spectrum。对 smooth separated
morphism `f:X\to Y`，smooth purity 给出从 `f^*` 到 `f^!` 的 Thom twist
识别。
对一般 smoothable lci `f`，6.12 在 homotopy category 中给出 purity
morphism；结合三角层伴随可构造 Gysin 型映射，但只有在相应 coefficient 为
`f`-pure 时才能把该 map 当作 purity isomorphism。把 Thom twist 改写成纯
双次数还需要 `E` 的定向。

**命题 6.17（带切丛扭曲的 exceptional trace）.** 若 `f:X\to Y` proper
且 smooth，并且 `E\in\mathbf{SH}(Y)`，则有自然映射

$$
f_*\Sigma^{T_f}f^*E\longrightarrow E.
$$

**证明.** Smooth purity 给出
`\Sigma^{T_f}f^*E\simeq f^!E`，proper compatibility 给出
`f_*\simeq f_!`。因此左侧自然等价于 `f_!f^!E`，再应用伴随
`f_!\dashv f^!` 的 counit

$$
f_!f^!E\longrightarrow E
$$

即得所述 map。注意 `f^*\dashv f_*` 的 counit 类型是
`f^*f_*\to\operatorname{id}_{\mathbf{SH}(X)}`，并不产生
`f_*f^*E\to E`。若 `T_f=0`（例如 finite etale），上式才无扭曲；一般要
把 Thom twist 改写为双次数还需 orientation。`\square`

**命题 6.18.** 若向量丛 `V` 有 Thom class 使 `\operatorname{Th}(V)` 在某个 `E`-cohomology 理论中可定向，则 Thom isomorphism 是额外定向数据的后果，不是 Thom space 定义的形式后果。

**证明.** Thom space 的定义只给出对象 `V/(V\setminus X)`。Thom isomorphism 要求与系数理论 `E` 相关的类 `u\in E^{*,*}(\operatorname{Th}(V))` 使 cup product with `u` 诱导等价。该类的存在依赖 orientation；没有 orientation 时只能谈 Thom object 和 Thom twist，不能推出 cohomology 群同构。`\square`

## 6.5 失败模式

**命题 6.19.** 不能把 homotopy purity 的等价

$$
X/(X\setminus Z)\simeq\operatorname{Th}(N_{Z/X})
$$

直接替换为任意闭嵌入上的 `i^!\simeq\Sigma^{-N}i^*`。

**证明.** Homotopy purity 的假设是 `Z`、`X` 都 smooth over `S`，结论所在
范畴是 `\mathbf H_*(S)`，对象类型是商与 Thom space。`i^!` 只在稳定六操作
中定义。一般闭嵌入可能不是 regular immersion，因而没有向量丛法丛；即便
`i` regular，外部输入 6.12 也只在 homotopy category 中构造
`\Sigma^{-N_i}i^*\to i^!`。其可逆性是定义 6.14 的 coefficientwise purity
条件。两个陈述的假设、所在范畴和结论类型都不同，故不能互换。`\square`

## 6.6 从闭补商到 Thom 扭曲

Thom spaces 把向量丛转换为 pointed motivic spaces，homotopy purity 把 smooth
closed pair 的闭补商识别为法丛 Thom space。稳定六操作对 smoothable lci
separated 态射的 DJK purity 系统在本书只按
`\operatorname{Ho}(\mathbf{SH})` 伪函子层使用；smooth separated 情形另有
稳定 infinity-范畴层的 purity equivalence。对 regular closed immersion 和
一般 lci 态射则必须逐系数检查 `f`-purity。Absolute
purity 是 regular schemes 上一整族 coefficientwise 可逆性条件，不是
homotopy purity 的改写。Gysin maps、Euler classes 和 bivariant theory 还要
区分 transformation、isomorphism 与 orientation。

## 练习

**练习 6.1.** 证明零向量丛的 Thom space 为 `X_+`。

**练习 6.2.** 对平凡秩 `r` 向量丛，推导其 Thom space 与 `T^{\wedge r}\wedge X_+` 的关系。

**练习 6.3.** 说明 homotopy purity 中为什么需要法丛。

**练习 6.4.** 写出 smooth purity 在 separated etale morphism 情形下的形式。

**练习 6.5.** 解释 orientation 与 Thom space 定义之间的逻辑差异。
