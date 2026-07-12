# 第十章：Voevodsky motives、Cisinski-Deglise motives 与 HZ-modules

## 本章目标

本章解释 triangulated categories of motives 与 `H\mathbb Z`-modules 的关系。核心观点是：`H\mathbb Z` 把 `\mathbf{SH}(S)` 线性化，`H\mathbb Z`-modules 提供 motives 的稳定 infinity-categorical 模型；但 Voevodsky motives、Cisinski-Deglise motives 和 module categories 的比较依赖深外部定理。

## 依赖前置知识

需要 motivic cohomology、commutative algebra objects、module categories、monadicity、triangulated categories、six operations、Tate twists 和 finite correspondences 的基本背景。

## 10.1 HZ-modules

**定义 10.1.** 定义

$$
\operatorname{Mod}_{H\mathbb Z_S}(\mathbf{SH}(S))
$$

为 `\mathbf{SH}(S)` 中 `H\mathbb Z_S`-module objects 组成的 stable presentable infinity-category。

**命题 10.2.** `\operatorname{Mod}_{H\mathbb Z_S}(\mathbf{SH}(S))` 是 stable presentable infinity-category。

**证明.** `\mathbf{SH}(S)` 是 presentable stable symmetric monoidal infinity-category，且张量积分别保持小余极限。对其中的 commutative algebra object `H\mathbb Z_S`，module category presentable；其 limits 和 colimits 由遗忘函子创建。稳定性由底层范畴稳定且 module structure 与有限极限、余极限相容得到。`\square`

**定义 10.3.** 自由-遗忘伴随为

$$
H\mathbb Z_S\otimes -:\mathbf{SH}(S)
\rightleftarrows
\operatorname{Mod}_{H\mathbb Z_S}(\mathbf{SH}(S)):U.
$$

**命题 10.4.** 对 `E\in\mathbf{SH}(S)` 和 `M\in\operatorname{Mod}_{H\mathbb Z_S}`，有自然等价

$$
\operatorname{Map}_{\operatorname{Mod}_{H\mathbb Z_S}}
(H\mathbb Z_S\otimes E,M)
\simeq
\operatorname{Map}_{\mathbf{SH}(S)}(E,U(M)).
$$

**证明.** 这是自由 module 的泛性质。一个 `H\mathbb Z_S`-linear map 从自由 module `H\mathbb Z_S\otimes E` 到 `M` 等价于底层对象映射 `E\to U(M)`；`H\mathbb Z_S`-线性由自由性唯一延拓。`\square`

## 10.2 Motives of smooth schemes

**定义 10.5.** 对 `X\in\operatorname{Sm}_S`，定义其 `H\mathbb Z`-motive 为

$$
M_S(X)=H\mathbb Z_S\otimes\Sigma_T^\infty X_+
\in\operatorname{Mod}_{H\mathbb Z_S}(\mathbf{SH}(S)).
$$

**命题 10.6.** Motivic cohomology 可在 `H\mathbb Z`-modules 中表示为

$$
H^{p,q}(X,\mathbb Z)\simeq
\pi_0\operatorname{Map}_{\operatorname{Mod}_{H\mathbb Z_S}}
(M_S(X),\Sigma^{p,q}H\mathbb Z_S).
$$

**证明.** 由定义 10.5 和自由-遗忘伴随命题 10.4，右侧等价于

$$
\pi_0\operatorname{Map}_{\mathbf{SH}(S)}
(\Sigma_T^\infty X_+,\Sigma^{p,q}H\mathbb Z_S),
$$

这正是定义 9.2。`\square`

**定义 10.7.** Tate motive 写作

$$
\mathbb Z(q)=\Sigma^{2q,q}H\mathbb Z_S
$$

作为 `H\mathbb Z_S`-module。不同文献中 shift/Tate twist 约定不同；本书使用该稳定同伦双次数约定。

## 10.3 与 Voevodsky motives 的比较

**外部输入定理 10.8（DM-10.8，分层陈述）.** 下列两项处在不同的范畴
层级，不能互相替换。

1. 若 `k` 为 characteristic zero field，Röndigs--Ostvær 的 Theorem 1.1
   给出 monoidal triangulated equivalence

   $$
   \mathbf{DM}(k)\simeq
   \operatorname{Ho}\!\left(
   \operatorname{Mod}_{H\mathbb Z_k}(\mathbf{SH}(k))
   \right).
   $$

   这里左边和右边都是同伦范畴；该结论本身不包含高映射空间或
   presentability 数据。
2. 设 `k` 为域，`e` 为其 exponential characteristic。Elmanto--Kolderup
   Theorem 5.2 及其 finite-correspondence 特例 Corollary 5.3 在反演 `e`
   后给出 presentably symmetric monoidal stable infinity-categories 的等价

   $$
   \mathbf{DM}(k,\mathbb Z[1/e])
   \simeq
   \operatorname{Mod}_{H\mathbb Z_k[1/e]}
   (\mathbf{SH}(k)).
   $$

**精确来源.** Oliver Röndigs, Paul Arne Ostvaer, *Modules over motivic
cohomology*, Theorem 1.1，
`https://doi.org/10.1016/j.aim.2008.05.013`；Elden Elmanto, Hakon
Kolderup, *On Modules over Motivic Ring Spectra*, Theorem 5.2 and
Corollary 5.3，`https://arxiv.org/abs/1708.05651`。

**注 10.9.** 第一项不能仅凭“取 enhancement”升级为第二项；第二项也不能
去掉 `1/e`。有效 motives、一般基、其他 transfers 或整系数正特征版本均需
新的比较定理。

**命题 10.10.** 若定理 10.8 适用，则 Voevodsky motivic cohomology 的 Hom 公式与定义 9.2 相容。

**证明.** 在定理 10.8(1) 中取三角同伦范畴的 `Hom`，或在 10.8(2) 中先取
稳定 infinity-范畴的 mapping spectrum 再取 `\pi_0`。两种情形下，比较函子
都把 smooth scheme 的 motive 送到 `M_k(X)`，把 Tate object 送到
`\mathbb Z(q)`。命题 10.6 的自由 module 伴随遂把相应 `Hom` 识别为定义
9.2 的 motivic cohomology。证明只使用所选层级中的比较，不把三角 `Hom`
提升为未给出的 mapping-space 等价。`\square`

## 10.4 Cisinski-Deglise motivic categories

**高级外部输入 10.11（P1）.** Cisinski--Deglise 在其规定的基概形、系数和
motivic category 上构造 triangulated six operations、Tate twists、
localization 与 purity。这个一般基 package 不参与定理 10.8 的域上 P0
比较；使用时必须指定是 Beilinson motives、cdh motives 还是其他模型。

**约定 10.12.** 在本章 P0 主线中，`\mathbf{DM}(k)` 只表示定理 10.8 所用的
Voevodsky big motives，并始终附带相应系数。对一般 `S`，符号
`\mathbf{DM}(S)` 只有在先指定 Beilinson、cdh、effective 或 constructible
模型及其比较定理后才使用；本书不把这些模型按定义识别。

**命题 10.13.** 若 `\mathbf{DM}(S)\simeq\operatorname{Mod}_{H\mathbb Z_S}(\mathbf{SH}(S))`，则 `\mathbf{DM}(S)` 继承 stable presentable structure 和 `H\mathbb Z`-linear six operations。

**证明.** 等价把 module category 的 stable presentable 结构转移到 `\mathbf{DM}(S)`。若等价与六操作相容，则 `f^*, f_*, f_!, f^!` 在 module category 中的 `H\mathbb Z`-linear 结构也转移过去。相容性不是范畴等价自动给出的，必须包含在外部输入定理的假设中。`\square`

## 10.5 Effective motives 与稳定 motives

**定义 10.14.** Effective `H\mathbb Z`-motives 是由

$$
M_S(X)=H\mathbb Z_S\otimes\Sigma_T^\infty X_+
$$

在不允许负 Tate twist 的操作下生成的子范畴。稳定 motives 则允许所有 Tate twists `\mathbb Z(q)`，`q\in\mathbb Z`。

**命题 10.15.** 从 effective motives 到 stable motives 的过渡等价于形式上反演 Tate object。

**证明.** Effective 口径只允许非负 Tate twists。Stable 口径要求 `\mathbb Z(1)` 可逆，从而存在 `\mathbb Z(-1)`。把 effective category 中的 Tate suspension functor 形式反演，得到含所有正负 Tate twists 的稳定范畴。这正是从 effective motives 到 stable motives 的范畴论含义。`\square`

**注 10.16.** 文献中的 `DM^{eff}`、`DM`、constructible motives、Beilinson motives、cdh motives 不应混用。它们的生成子、系数、拓扑和六操作闭包条件不同。

## 10.6 Module monadicity

**命题 10.17.** 自由-遗忘伴随诱导的 monad 为

$$
T(E)=H\mathbb Z_S\otimes E.
$$

**证明.** 自由函子为 `F(E)=H\mathbb Z_S\otimes E`，遗忘函子为 `U`。合成 `UF` 在底层 `\mathbf{SH}(S)` 上就是 `H\mathbb Z_S\otimes E`。Monad multiplication 由 `H\mathbb Z_S` 的乘法

$$
H\mathbb Z_S\otimes H\mathbb Z_S\to H\mathbb Z_S
$$

给出，unit 由 `\mathbb 1_S\to H\mathbb Z_S` 给出。`\square`

**命题 10.18.** `H\mathbb Z`-module 是带有 action map

$$
H\mathbb Z_S\otimes M\to M
$$

且满足结合律和单位律的对象。

**证明.** 这是 commutative algebra object 上 module object 的定义。结合律要求两种

$$
H\mathbb Z_S\otimes H\mathbb Z_S\otimes M\to M
$$

的合成相同；单位律要求 `\mathbb 1_S\otimes M\to H\mathbb Z_S\otimes M\to M` 为 canonical identification。`\square`

## 10.7 Six operations on modules

**高级外部输入 10.19（P1）.** 对一个已选定且满足基变换、投影公式和
exceptional-functor 假设的 premotivic `H\mathbb Z`，其 module coefficient
system 可继承六操作；与某个一般基 `\mathbf{DM}(-)` 的六操作相容还需要
premotivic comparison theorem。该 package 不由定理 10.8 的单纤维范畴等价
推出，也不作为本章 P0 证明的输入。

**命题 10.20.** 若 `f^*H\mathbb Z_Y\simeq H\mathbb Z_X`，则 `f^*` 把 `H\mathbb Z_Y`-modules 送到 `H\mathbb Z_X`-modules。

**证明.** 设 `M` 是 `H\mathbb Z_Y`-module，action 为 `H\mathbb Z_Y\otimes M\to M`。对 `f^*` 作用并用强幺半性，得到

$$
f^*H\mathbb Z_Y\otimes f^*M\simeq f^*(H\mathbb Z_Y\otimes M)\to f^*M.
$$

若 `f^*H\mathbb Z_Y\simeq H\mathbb Z_X`，这就是 `H\mathbb Z_X` 在 `f^*M` 上的 action。结合律和单位律由 `f^*` 的函子性保持。`\square`

## 10.8 Linearization 的边界

**命题 10.21.** 从 `\mathbf{SH}(S)` 到 `H\mathbb Z`-modules 的函子

$$
H\mathbb Z_S\otimes -:\mathbf{SH}(S)\to\operatorname{Mod}_{H\mathbb Z_S}
$$

一般不是保守的。

**证明.** 保守性要求若 `H\mathbb Z_S\otimes E\simeq0` 则 `E\simeq0`。自由 module functor 是 extension of scalars，它只检测 `H\mathbb Z` 可见的信息。稳定 motivic homotopy 中存在不由 motivic cohomology 检测的高同伦、K-theoretic、cobordism 或 torsion phenomena。因此保守性不能从定义推出；若在特殊子范畴成立，必须另行证明。`\square`

**注 10.22.** 这就是 motives 与 stable motivic homotopy 的重要差异：`\mathbf{DM}` 是线性化世界，`\mathbf{SH}` 保留 sphere spectrum 及其非线性稳定同伦信息。

## 10.9 本章小结

`H\mathbb Z`-modules 为 motives 提供稳定同伦中的线性模型。域上比较分成
characteristic zero 的 monoidal triangulated equivalence，和反演指数特征后
的 presentably symmetric monoidal stable infinity-categorical equivalence。
一般基六操作比较属于 P1；它不能由单纤维的三角等价形式推出。

## 练习

**练习 10.1.** 证明 module category 的自由-遗忘伴随。

**练习 10.2.** 用命题 10.4 推导命题 10.6。

**练习 10.3.** 解释 Tate motive `\mathbb Z(1)` 与双次数 shift 的关系。

**练习 10.4.** 举例说明为什么 `H\mathbb Z`-linearization 不应期望保守。

**练习 10.5.** 列出比较 `\mathbf{DM}(S)` 与 `H\mathbb Z`-modules 时必须检查的假设。

**练习 10.6.** 写出 `H\mathbb Z`-module action 的结合律图。

**练习 10.7.** 解释 effective motives 到 stable motives 的 Tate object 反演。
