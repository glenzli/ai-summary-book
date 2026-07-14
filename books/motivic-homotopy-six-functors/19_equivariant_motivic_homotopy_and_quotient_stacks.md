# 第十九章：Equivariant motivic homotopy 与 quotient stacks

同一个概形配上两个不同的群作用，忘掉作用以后可能完全相同，固定点、法丛表示和
transfer 却可能截然不同。Borel construction 只把作用编码进一个同伦商；genuine
equivariant theory 还要允许 representation spheres，并同时追踪各 stabilizer 的
信息。在代数几何中，quotient stack `[X/G]` 是保存这些数据的自然对象。

本章从代数群作用与 quotient stack 入手，比较 Borel 型和 genuine 型构造，再把
equivariant purity、固定点和 change of groups 放进六操作方差。第六至八章提供
普通 purity 与换基公式，代数群、linearly reductive 条件和 cdh 下降则在外部输入
中逐项声明。自由作用的好商会退化回 scheme-level theory，非自由作用则用
`G_m\curvearrowright\mathbb A^1` 的例子显示粗商遗失了什么。

## 19.1 Quotient stack 口径

**约定 19.1.** 固定 qcqs 概形 `B` 和其上的 flat finitely presented group
scheme `G`。称 `G` **tame**，若它 linearly reductive，且 `B` Nisnevich
局部具有 `G`-resolution property；后者表示每个有限生成 quasi-coherent
`G`-module 都是有限秩局部自由 `G`-module 的商。除非另行说明，本章再假设：
若 `G` 非有限，则 `B` 本身具有 `G`-resolution property；所有对象都是有限
表示的 `G`-quasi-projective `B`-schemes。若 `G` 离散，Hoyois 的定理允许把
对象类扩到 qcqs `G`-schemes。对带 `G`-作用的 `X`，商栈记为

$$
[X/G].
$$

**定义 19.2.** Equivariant motivic homotopy category 的 quotient stack 口径把 `G`-equivariant geometry 视为栈 `[X/G]` 上的 motivic homotopy theory。

**外部输入定理 19.3（EQ-19.3）.** 在约定 19.1 下存在 closed symmetric
monoidal stable infinity-category

$$
\mathbf{SH}_G(X)=\mathbf{SH}([X/G]).
$$

它由 equivariant motivic spaces 对基上表示球稳定化得到。对 smooth proper
`G`-morphism 有 ambidexterity；对 smooth closed pairs 有 equivariant
homotopy purity；开闭分解满足 gluing。

**精确来源与边界.** Marc Hoyois, *The six operations in equivariant motivic
homotopy theory*, Definition 2.26、Proposition 5.7、Theorem 6.9 与
Theorem 1.1（即 Theorem 6.18 和 Proposition 6.23）。仅有 linearly
reductive 性不足以替代 tameness 与 resolution hypotheses；正文也不把该理论
扩到任意 Artin stack 或任意非 representable 态射。

**命题 19.4.** 若 `G` 为平凡群，则 quotient stack 口径恢复非 equivariant motivic homotopy。

**证明.** 若 `G=1`，则 `[X/G]\simeq X`。定理 19.3 的对象、态射和六操作都限制到 schemes 上的普通 motivic spaces/spectra。因此恢复前文的 `\mathbf H(X)` 与 `\mathbf{SH}(X)` 口径。`\square`

## 19.2 Genuine 与 Borel 的区别

**定义 19.5.** Borel-equivariant theory 通常把 `G`-对象 `X` 送到混合商 `EG\times_GX` 或相应近似。Genuine equivariant theory 则保留稳定子和表示球等信息。

**命题 19.6.** Borel-equivariant theory 不足以恢复 genuine equivariant motivic homotopy 的全部信息。

**证明.** Borel construction 把 equivariant 信息通过自由 `G`-空间近似转换为非 equivariant quotient 信息。该过程通常不会保留 representation spheres、fixed-point data 和 genuine transfers。Genuine equivariant theory 的六操作和 purity 依赖 quotient stack 的 stabilizer 结构，因此不能由 Borel theory 无条件恢复。`\square`

## 19.3 Equivariant six operations

**外部输入定理 19.7（equivariant 六操作）.** 在约定 19.1 下，对每个
`G`-morphism 有 `f^*\dashv f_*`；对 smooth `f` 有
`f_\sharp\dashv f^*`；对 separated finite-type `f` 有
`f_!\dashv f^!`。这些函子满足 proper comparison、smooth purity、
extraordinary base change、projection formula 与 open-closed gluing。
此外，赋值 `X\mapsto\mathbf{SH}_G(X)` 对 `G`-equivariant cdh topology
满足范畴值下降。

**精确来源.** Hoyois, Theorem 1.1/Theorem 6.18 给出六操作，
Proposition 6.24 给出 cdh descent。后者是范畴值下降；某个上同调理论的 cdh
下降还要求其系数形成相应 cocartesian section，见该文 Corollary 6.25。

**命题 19.8.** Equivariant localization 对 invariant closed-open decomposition 成立。

**证明.** 若 `i:Z\hookrightarrow X` 是 `G`-invariant closed immersion，开补 `j:U\hookrightarrow X` 也带 `G`-作用。商后得到 closed-open pair `[Z/G]\hookrightarrow[X/G]\hookleftarrow[U/G]`。对 quotient stacks 应用 equivariant localization，得到相应 cofiber sequence。`\square`

## 19.4 Representation spheres

**定义 19.9.** 若 `V` 是 `G`-equivariant vector bundle 或 representation，其 Thom space 给出 equivariant sphere/twist，记为 `S^V` 或 `\Sigma^V`。

**命题 19.10.** Equivariant purity 中的 Thom twist 必须使用 equivariant normal bundle。

**证明.** 对 `G`-invariant closed immersion `Z\hookrightarrow X`，普通法丛 `N_{Z/X}` 自然带有 `G`-线性化。商栈上的 purity 记录 stabilizer 对法方向的作用；若忘记 equivariant structure，只保留底层向量丛，会丢失 representation sphere 信息。因此 Thom twist 必须是 equivariant normal bundle 的 Thom object。`\square`

## 19.5 Free actions 与 quotient schemes

**命题 19.11.** 若 `G` 在 `X` 上自由作用，且几何商 `X/G` 作为 scheme 存在并使 `X\to X/G` 为 `G`-torsor，则 quotient stack `[X/G]` 与 scheme `X/G` 表示的 stack 等价。

**证明.** 对任意测试对象 `T`，群胚 `\operatorname{Map}(T,[X/G])` 由 `G`-torsor `P\to T` 连同 equivariant map `P\to X` 给出。若 `X\to X/G` 为 `G`-torsor，则这样的数据等价于给出 `T\to X/G`，再取拉回 torsor `T\times_{X/G}X`。该构造给出群胚等价，故 stacks 等价。`\square`

**推论 19.12.** 在命题 19.11 的条件下，quotient-stack 口径给出自然等价

$$
\mathbf{SH}([X/G])\simeq\mathbf{SH}(X/G).
$$

**证明.** 由命题 19.11，`[X/G]\simeq X/G`。系数系统把基对象的等价送到
稳定范畴的等价，故得到所示比较。`\square`

**注 19.13.** 非自由作用时 `[X/G]` 保留 stabilizer 信息，而 coarse quotient 通常丢失该信息。因此 genuine equivariant theory 应使用 quotient stack，而不是只使用 coarse quotient。

## 19.6 Fixed points and isotropy

**定义 19.14.** 对子群 `H\subseteq G`，`H`-fixed locus `X^H` 是满足 `H` 逐点固定的子概形或子栈，若该对象存在。

**命题 19.15.** Genuine equivariant theory 中的 fixed-point information 不能由底层非 equivariant motivic spectrum 恢复。

**证明.** 底层非 equivariant functor 忘记 `G`-作用和所有 stabilizer 分层。两个不同 `G`-作用可能有同一底层 scheme 和同一底层 motivic spectrum，但 fixed loci、representation spheres 和 transfer data 不同。因此 fixed-point information 不由底层对象决定。`\square`

**例子 19.16.** `G_m` 作用在 `\mathbb A^1_k` 上：`t\cdot x=tx`。固定点
子概形为原点，原点在商栈 `[\mathbb A^1/G_m]` 中的 stabilizer 是整个
`G_m`；开轨道 `\mathbb G_m` 的 stabilizer 则平凡。另一方面，仿射范畴商为

$$
\mathbb A^1_k/\!/G_m
=\operatorname{Spec}k[x]^{G_m}=\operatorname{Spec}k,
$$

因为权一作用的不变量环只有 `k`。因此粗的仿射商同时遗失了 stabilizer 分层和
开轨道信息。

## 19.7 Change of groups

**定义 19.17.** 若 `\varphi:H\to G` 是代数群同态，则 restriction functor 把 `G`-对象视为 `H`-对象，记为

$$
\operatorname{Res}^G_H.
$$

在 quotient stack 口径下，它由态射

$$
[X/H]\longrightarrow [X/G]
$$

诱导。

**外部输入定理 19.18（restriction、induction 与 coinduction）.** 设
`H\subseteq G` 为闭子群，并假设商 `G/H` 存在、smooth 且
`G`-quasi-projective over `B`。结构态射

$$
p:G/H\longrightarrow B
$$

表示 schematic stack map `BH\to BG`，并有自然等价
`\mathbf{SH}_G(G/H)\simeq\mathbf{SH}_H(B)`。在该识别下，
`p^*` 是 restriction，`p_\sharp` 与 `p_*` 分别是 induction 与
coinduction。存在 Wirthmuller map

$$
p_\sharp\Sigma^{-\Omega_{G/H}}\longrightarrow p_*;
$$

若 `G/H` proper，则该 map 为等价。

**精确来源与边界.** Hoyois, Introduction, Section 1.4（Wirthmuller 与
Adams morphisms），其证明依赖 Theorem 1.1 的 smooth purity 和 proper
comparison。该来源没有在此处构造 norm functor；normed spectra 属于第十七章
的另一套有限 etale 乘法结构，不能由 subgroup restriction 自动推出。

**命题 19.19.** 若 `H=1`，restriction `\operatorname{Res}^G_1` 是忘却 equivariant structure 的 functor。

**证明.** 平凡群作用不包含非平凡 equivariance data。把 `G`-对象沿 `1\to G` 限制，只保留底层 scheme 或 spectrum，而忘记 `G`-action、stabilizer 和 representation-sphere 信息。`\square`

**命题 19.20.** 忘却 functor 的底层对象不足以恢复 fixed-point 信息。

**证明.** 忘却后两个不同 `G`-actions 可有同一底层对象。例如
`\mathbb A^1` 上的平凡作用与例子 19.16 的权一作用具有同一底层概形；前者的
`G_m`-固定点是整个 `\mathbb A^1`，后者只有原点。因此底层对象不能决定
fixed locus。这里断言的是信息不可恢复，不把它混写成某个具体忘却函子是否
检测所有等价的 conservativity 定理。`\square`

## 19.8 生成子与 equivariant cells

**定义 19.21.** Equivariant cells 通常由轨道型对象 `G/H_+`、representation spheres `S^V` 及其 motivic suspensions 生成。代数几何中应以 quotient stacks 或 homogeneous spaces 的存在性为前提。

**外部输入定理 19.22.** 在约定 19.1 下，若 `r:X\to B` 是结构态射，则
`\mathbf{SH}_G(X)` 由

$$
E^{-1}\otimes\Sigma^\infty Y_+,
\qquad Y\in\operatorname{Sm}^G_X,
\quad E\in r^*(\operatorname{Sph}_B),
$$

在 sifted colimits 下生成；这里 `\operatorname{Sph}_B` 由基 `B` 上的
equivariant vector-bundle spheres 组成。

**精确来源.** Hoyois, Proposition 6.4(2)。该命题给出 sifted-colimit
generation；若要进一步断言这些对象紧致，须另核对基、群和对象的有限性，不能把
“生成”自动改写成“紧生成”。

**注 19.23.** Equivariant compact generation 的生成子依赖允许的群和基。不能直接把拓扑 equivariant stable homotopy 中的 orbit category 公式逐字搬到代数几何语境。

## 19.9 Stabilizer 信息为何不可遗忘

Equivariant motivic homotopy theory 不是在普通 motivic theory 上附加一个群作用那么简单。Quotient stack、stabilizer、representation spheres 和 equivariant six operations 都是 genuine 结构。Hoyois 的六操作理论提供了标准入口；Borel 型理论只是某些线性化或近似版本。

## 练习

**练习 19.1.** 解释 `[X/1]\simeq X`。

**练习 19.2.** 比较 Borel-equivariant 与 genuine equivariant theory。

**练习 19.3.** 写出 invariant closed-open pair 的 localization sequence。

**练习 19.4.** 说明 equivariant normal bundle 在 purity 中的位置。

**练习 19.5.** 为什么 linearly reductive 假设会出现在 quotient stack 口径中？

**练习 19.6.** 证明自由作用且 torsor 商存在时 `[X/G]\simeq X/G`。

**练习 19.7.** 计算 `G_m` 作用在 `\mathbb A^1` 上的 fixed locus。

**练习 19.8.** 描述 `H\to G` 诱导的 `[X/H]\to[X/G]`。

**练习 19.9.** 解释为什么 equivariant cells 需要 representation spheres。
