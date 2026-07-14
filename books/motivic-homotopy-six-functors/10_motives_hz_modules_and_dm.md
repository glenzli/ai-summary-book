# 第十章：Voevodsky motives、Cisinski-Deglise motives 与 HZ-modules

将一个谱 `E` 张量上 `H\mathbb Z`，会只保留能够由 motivic cohomology 线性看见的
部分。由此得到的 module 范畴与传统 motives 极其接近，但“接近”必须区分层级：
同伦范畴的三角等价并不自动给出 mapping spectra 的等价，域上的比较也不能无条件
推广到一般基概形或整系数正特征。

本章先完全在 `\mathbf{SH}(S)` 内构造 `H\mathbb Z`-modules、自由-遗忘伴随和
光滑概形的 motive，再陈述 Voevodsky 与 Cisinski--Deglise 型范畴的外部比较。
第九章提供代表谱和 cohomology 公式，有限对应的几何来源留到第十四章。读者由此能
分清三件事：module 是内部定义，`DM` 是另一个构造，而二者的识别是一条带系数、
基和范畴层级假设的定理。

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

**高级外部输入 10.11.** Cisinski--Deglise 在其规定的基概形、系数和
motivic category 上构造 triangulated six operations、Tate twists、
localization 与 purity。这个一般基 package 不参与定理 10.8 的域上比较；
使用时必须指定是 Beilinson motives、cdh motives 还是其他模型。

**约定 10.12.** 在本章中，`\mathbf{DM}(k)` 只表示定理 10.8 所用的
Voevodsky big motives，并始终附带相应系数。对一般 `S`，符号
`\mathbf{DM}(S)` 只有在先指定 Beilinson、cdh、effective 或 constructible
模型及其比较定理后才使用；本书不把这些模型按定义识别。

**命题 10.13.** 若 `\mathbf{DM}(S)\simeq\operatorname{Mod}_{H\mathbb Z_S}(\mathbf{SH}(S))`，则 `\mathbf{DM}(S)` 继承 stable presentable structure 和 `H\mathbb Z`-linear six operations。

**证明.** 等价把 module category 的 stable presentable 结构转移到 `\mathbf{DM}(S)`。若等价与六操作相容，则 `f^*, f_*, f_!, f^!` 在 module category 中的 `H\mathbb Z`-linear 结构也转移过去。相容性不是范畴等价自动给出的，必须包含在外部输入定理的假设中。`\square`

## 10.5 Effective motives 与稳定 motives

**定义 10.14.** 记
`\operatorname{DM}^{eff}_{H\mathbb Z}(S)` 为
`\operatorname{Mod}_{H\mathbb Z_S}(\mathbf{SH}(S))` 中由对象

$$
M_S(X)(q)=M_S(X)\otimes\mathbb Z(q),
\qquad X\in\operatorname{Sm}_S,\quad q\geq0,
$$

生成的最小 full stable localizing subcategory。这里 localizing 表示对小余极限
和 cofiber 封闭；定义只额外闭合非负 Tate twists，并不预先要求
`(-)\otimes\mathbb Z(1)` 的逆保持在该子范畴中。

**命题 10.15（Tate 反演的精确像）.** 张量自函子
`(1)=(-)\otimes\mathbb Z(1)` 保持
`\operatorname{DM}^{eff}_{H\mathbb Z}(S)`，因而有 canonical exact
colimit-preserving functor

$$
\operatorname{DM}^{eff}_{H\mathbb Z}(S)[(1)^{-1}]
\longrightarrow
\operatorname{Mod}_{H\mathbb Z_S}(\mathbf{SH}(S)).
$$

它的 essential image 是由所有 `M_S(X)(q)`，其中
`X\in\operatorname{Sm}_S`、`q\in\mathbb Z`，生成的 localizing
subcategory。特别地，只有在这些对象生成整个 module category 时，该 functor
才是到整个 `\operatorname{Mod}_{H\mathbb Z_S}` 的等价。

**证明.** 对生成子有

$$
M_S(X)(q)(1)\simeq M_S(X)(q+1),
$$

且 `(1)` 保持小余极限和 cofiber，所以它保持定义 10.14 的 localizing
subcategory。自函子局部化的泛性质遂给出所示 canonical functor。局部化中
`(1)` 可逆，故每个 `M_S(X)(q)`、`q\in\mathbb Z` 都在其像中；反过来，
局部化由 effective 生成子及 `(1)` 的正负幂在小余极限和 cofiber 下生成，故
essential image 不会超出这些对象生成的 localizing subcategory。最后一句正是
essential image 等于整个目标的充要条件。`\square`

**注 10.16.** 定义 10.14 是本书内部的 `H\mathbb Z`-module effective
subcategory，不是把文献中的任意 `DM^{eff}(S)` 重新定义了一遍。要把命题
10.15 的 source 或 target 与 Voevodsky、Cisinski--Deglise 的模型识别，仍须
使用定理 10.8 一类的比较定理。Constructible motives、Beilinson motives 与
cdh motives 的生成子、系数、拓扑和六操作闭包也各不相同。

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

**高级外部输入 10.19.** 对一个已选定且满足基变换、投影公式和
exceptional-functor 假设的 premotivic `H\mathbb Z`，其 module coefficient
system 可继承六操作；与某个一般基 `\mathbf{DM}(-)` 的六操作相容还需要
premotivic comparison theorem。该 package 不由定理 10.8 的单纤维范畴等价
推出，且本章后续证明不调用这一一般基结论。

**命题 10.20.** 若 `f^*H\mathbb Z_Y\simeq H\mathbb Z_X`，则 `f^*` 把 `H\mathbb Z_Y`-modules 送到 `H\mathbb Z_X`-modules。

**证明.** 设 `M` 是 `H\mathbb Z_Y`-module，action 为 `H\mathbb Z_Y\otimes M\to M`。对 `f^*` 作用并用强幺半性，得到

$$
f^*H\mathbb Z_Y\otimes f^*M\simeq f^*(H\mathbb Z_Y\otimes M)\to f^*M.
$$

若 `f^*H\mathbb Z_Y\simeq H\mathbb Z_X`，这就是 `H\mathbb Z_X` 在 `f^*M` 上的 action。结合律和单位律由 `f^*` 的函子性保持。`\square`

## 10.8 Linearization 的边界

**命题 10.21（保守性的一个充分判据）.** 考虑自由 module 函子

$$
H\mathbb Z_S\otimes -:\mathbf{SH}(S)\to\operatorname{Mod}_{H\mathbb Z_S}
$$

若单位 `\mathbb 1_S` 属于由 `H\mathbb Z_S` 生成的 thick tensor ideal，则该函子
保守。Module 公理本身不蕴含这一生成条件，因而也不形式蕴含保守性。

**证明.** 设 `H\mathbb Z_S\otimes E\simeq0`，并令

$$
\mathcal I_E=\{A\in\mathbf{SH}(S)\mid A\otimes E\simeq0\}.
$$

张量积对有限余极限正合，故 `\mathcal I_E` 对 cofiber、retract 和有限直和封闭；
结合律还说明它是 tensor ideal。假设给出 `H\mathbb Z_S\in\mathcal I_E`，于是由
`H\mathbb Z_S` 生成的 thick tensor ideal 包含于 `\mathcal I_E`。若
`\mathbb 1_S` 属于该 ideal，则 `E\simeq\mathbb 1_S\otimes E\simeq0`。
这证明保守性。反之，自由 module 的定义只给出乘法与单位作用，并没有给出
`\mathbb 1_S` 的上述有限生成表达，所以不能仅由 module 形式主义断言保守。`\square`

**注 10.22.** `H\mathbb Z`-modules 是线性化世界，`\mathbf{SH}` 还保留球谱的
非线性稳定同伦信息。某个局部化、完成或受限子范畴上的保守性必须由类似命题 10.21
的生成判据或独立比较定理建立。

## 10.9 线性化得到什么，又忘掉什么

`H\mathbb Z`-modules 为 motives 提供稳定同伦中的线性模型。域上比较分成
characteristic zero 的 monoidal triangulated equivalence，和反演指数特征后
的 presentably symmetric monoidal stable infinity-categorical equivalence。
一般基六操作比较需要额外 premotivic 假设；它不能由单纤维的三角等价形式推出。

## 练习

**练习 10.1.** 证明 module category 的自由-遗忘伴随。

**练习 10.2.** 用命题 10.4 推导命题 10.6。

**练习 10.3.** 解释 Tate motive `\mathbb Z(1)` 与双次数 shift 的关系。

**练习 10.4.** 证明命题 10.21 中 `\mathcal I_E` 是 thick tensor ideal，并说明
为什么“`H\mathbb Z_S` 是代数对象”不足以推出 `\mathbb 1_S` 由它厚生成。

**练习 10.5.** 列出比较 `\mathbf{DM}(S)` 与 `H\mathbb Z`-modules 时必须检查的假设。

**练习 10.6.** 写出 `H\mathbb Z`-module action 的结合律图。

**练习 10.7.** 解释 effective motives 到 stable motives 的 Tate object 反演。
