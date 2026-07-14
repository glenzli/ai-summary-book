# 第三章：Tate sphere、P1-稳定化与 SH(S)

在非稳定范畴 `\mathbf H_*(S)` 中，悬挂一般不可逆，因而 cofiber sequence 还不能
自由地向两个方向延伸。代数几何又多出一个拓扑中没有的坐标：去掉原点的仿射直线
`\mathbb G_m`。把普通圆方向与这一 Tate 方向同时稳定，才得到能够容纳双分次同伦群、
Thom 扭曲和六操作的范畴。

具体地，我们从 pointed cofiber
`\mathbb A^1/(\mathbb A^1\setminus0)` 得到 Tate 球 `T`，比较它与
`\mathbb P^1/\infty` 以及 `S^{1,0}\wedge\mathbb G_m`，再利用对称幺半对象反演的
泛性质定义 `\mathbf{SH}(S)`。稳定化的存在、3-symmetry 和谱模型比较属于明确的
外部输入；一旦这些输入固定，普通悬挂可逆性与稳定性则由书内的张量可逆性论证推出。

## 3.1 Pointed smash product

**外部输入定理 3.1.** `\mathbf H(S)` 的 Cartesian monoidal structure 诱导 `\mathbf H_*(S)` 上的 smash product

$$
X\wedge Y=(X\times Y)/(X\vee Y),
$$

并使 `\mathbf H_*(S)` 成为 presentable symmetric monoidal infinity-category，且张量积分别保持小余极限。

**依赖源.** 附录 C 外部输入 C.4 给出 pointed smash construction；C.12
用 Hoyois Proposition 3.15、Lemma 6.3、Proposition 6.4(1) 与 Corollary 6.7
核查 motivic localization、3-symmetry 和稳定化模型。

**定义 3.2.** Simplicial circle 记为

$$
S^{1,0}=\Delta^1/\partial\Delta^1
$$

视为常值 pointed motivic space。

**定义 3.3.** `\mathbb G_m` 以单位截面 `1:S\to\mathbb G_m` 为基点。定义

$$
S^{1,1}=S^{1,0}\wedge\mathbb G_m.
$$

**定义 3.4.** Tate sphere 定义为

$$
T=\mathbb A^1/(\mathbb A^1\setminus0)=\mathbb A^1/\mathbb G_m
$$

在 `\mathbf H_*(S)` 中的对象。

## 3.2 Tate sphere 的基本等价

**命题 3.5.** 在 `\mathbf H_*(S)` 中有自然等价

$$
T\simeq S^{1,0}\wedge\mathbb G_m.
$$

**证明.** 在 pointed motivic spaces 中，`T` 是包含映射

$$
\mathbb G_m\hookrightarrow\mathbb A^1
$$

的 cofiber。第二章推论 2.10 给出 `\mathbb A^1\simeq *`。因此该 cofiber
等价于映射 `\mathbb G_m\to *` 的 cofiber。附录 C 命题 C.5 对 motivic
pointed category 给出
`\operatorname{cofib}(Y\to *)\simeq\Sigma Y\simeq S^{1,0}\wedge Y`。
取 `Y=\mathbb G_m`，得到
`T\simeq S^{1,0}\wedge\mathbb G_m`。`\square`

**注 3.6.** 等价 `T\simeq\mathbb P^1/\infty` 需要把 `\mathbb P^1` 分解为 `\mathbb A^1` 与无穷远点，并验证对应商对象。该证明将在纯性和 Thom spaces 章节中给出坐标级版本。

**定义 3.7.** 对 `p\ge q\ge0`，定义 motivic sphere

$$
S^{p,q}=(S^{1,0})^{\wedge(p-q)}\wedge(\mathbb G_m)^{\wedge q}.
$$

稳定化后允许 `p,q` 为任意整数，此时负指数表示相应可逆对象的逆。

## 3.3 稳定 motivic homotopy category

**外部输入定理 3.8（对称幺半 `T`-反演）.** 对本章的基概形 `S`，存在
presentably symmetric monoidal infinity-category `\mathbf{SH}(S)` 和保持
小余极限的强对称幺半函子

$$
\Sigma_T^\infty:\mathbf H_*(S)\longrightarrow\mathbf{SH}(S)
$$

满足以下泛性质：对任意 presentably symmetric monoidal infinity-category
`\mathcal C`，若 `F:\mathbf H_*(S)\to\mathcal C` 保持小余极限、为强对称
幺半函子并把 `T` 送为张量可逆对象，则 `F` 通过
`\Sigma_T^\infty` 在此类函子组成的 infinity-范畴中唯一因子化。这里“唯一”
意指因子化空间若非空则可缩。

此外，`T` 是 3-symmetric，该形式反演与 symmetric `T`-spectra 的底层
infinity-category 等价。所得 `\mathbf{SH}(S)` 是 stable；最后一项也可由
命题 3.13 的书内论证从前述输入推出。

**依赖源.** Robalo, *Noncommutative Motives I*, Proposition 4.10、
Corollary 4.24、Theorem 4.29；motivic 模型与 3-symmetry 见 Hoyois,
*The six operations in equivariant motivic homotopy theory*, Lemma 6.3、
Proposition 6.4(1) 与 Corollary 6.7（取平凡群）。Morel-Voevodsky、Jardine
提供早期模型范畴构造。附录 C 明确区分形式对象反演、谱模型与稳定性。

**定义 3.9.** `\mathbf{SH}(S)` 称为 `S` 上的 stable motivic homotopy category，或稳定 motivic homotopy infinity-范畴。单位对象写作

$$
\mathbb 1_S=\Sigma_T^\infty S_+.
$$

**定义 3.10.** 对 `X\in\operatorname{Sm}_S`，其 suspension spectrum 定义为

$$
\Sigma_T^\infty X_+\in\mathbf{SH}(S).
$$

**命题 3.11.** `T` 在 `\mathbf{SH}(S)` 中可逆。

**证明.** 这是定理 3.8 泛性质中稳定化的组成部分：`\mathbf{SH}(S)` 是通过迫使 `T` 可逆得到的 symmetric monoidal stabilization。因此 `\Sigma_T^\infty T` 有张量逆。`\square`

**命题 3.12.** `\Sigma_T^\infty` 保持小余极限。

**证明.** 这是外部输入定理 3.8 中结构函子的明确组成部分，也可由附录 C
的 presentable 对象反演构造读出。这里不能只从“把 `T` 送为可逆对象”
推出余极限保持性；后者已包含在所使用的泛性质的函子类型
`\operatorname{Fun}^{L,\otimes}` 中。`\square`

## 3.4 与经典稳定同伦论的差异

**命题 3.13.** 在 `\mathbf{SH}(S)` 中，`S^{1,0}` 与 `\mathbb G_m` 的像分别
张量可逆，且二者的张量积为 `T` 的像。因此对任意 `p,q\in\mathbb Z`，

$$
S^{p,q}:=(S^{1,0})^{\wedge(p-q)}\wedge
(\mathbb G_m)^{\wedge q}
$$

良定义；负指数表示 Picard infinity-groupoid 中的张量逆。

**证明.** 命题 3.5 给出 `T\simeq S^{1,0}\wedge\mathbb G_m`，命题 3.11
给出 `T` 的像可逆。附录 C 引理 C.11 说明 symmetric monoidal
infinity-category 中可逆张量积的每个因子都可逆，故前两项成立。整数次幂
随后可用张量积、单位和所选逆对象定义；逆对象的选择空间可缩，所以结果在
Picard infinity-groupoid 中不依赖任意选择。公式直接给出二者张量积为
`T`。注意这不证明两个 Picard 类“代数独立”，也不证明
`S^{1,0}\simeq\mathbb G_m`；这类关系需要额外定理。`\square`

**例子 3.14.** 当 `S=\operatorname{Spec}\mathbb C` 时，复 Betti realization 会把光滑复概形送到复解析空间，再送到拓扑 spaces。该 functor 与 `\mathbb A^1`-局部化相容，因为 `\mathbb C` 的仿射直线拓扑上可缩；稳定后给出到 classical spectra 的 realization functor。其完整构造和对称幺半性是外部输入，不在此例中证明。

**命题 3.15.** 不能仅由 `\mathbf{SH}(S)` 的定义推出某个 realization functor 是保守的。

**证明.** 保守性要求若 `R(E)\simeq0` 则 `E\simeq0`。稳定化的定义只给出 `R` 若存在必须反演 `T` 并保持相关结构；它不控制 `R` 的核。一个保持余极限的 functor 可以有非零核，因此保守性必须由额外几何或同伦论结果证明。`\square`

## 3.5 紧致、构造性与生成子

**定义 3.16.** 设 `\mathcal C` 为 presentable infinity-category。对象
`K\in\mathcal C` 称为 **紧致对象**（compact object），若
`\operatorname{Map}_{\mathcal C}(K,-)` 保持 `\mathbb U`-小 filtered
colimits。一组对象 `\mathcal G` 称为 **稳定生成子**，若包含
`\mathcal G` 且对等价和全部 `\mathbb U`-小余极限封闭的最小 full
subcategory 就是 `\mathcal C`。因为 `\mathcal C` stable，这个 closure
自动对整数 suspension、fibers 和 cofibers 封闭。

**外部输入定理 3.17（motivic 紧生成）.** 若 `S` 是 qcqs 概形；特别地，
若 `S` 属于本书默认有限型 Noetherian 基范畴，则 `\mathbf{SH}(S)` 由对象

$$
\Sigma^{p,q}\Sigma_T^\infty X_+,\qquad X\in\operatorname{Sm}_S,\ p,q\in\mathbb Z
$$

组成的一集合紧致对象生成。可只取 smooth affine `X` 的一组小代表。

**依赖源与边界.** Hoyois, *The six operations in equivariant motivic homotopy
theory*, Proposition 6.4(2)-(3)，取平凡群；该命题给出 smooth suspension
spectra 的紧致性和稳定化后的生成性。整数球移位为张量可逆自等价，故保持
紧致性。扩展到非 qcqs 基、stacky 系数系统或其他 topology 时不得沿用本
定理而不另核查。

**定义 3.18.** `\mathbf{SH}(S)` 的 **几何构造性子范畴**
`\mathbf{SH}_c(S)` 是定理 3.17 中生成子在有限 cofibers、整数 suspension
和 retracts 下的最小 full subcategory，即这些生成子的 thick closure。
这是几何生成定义；它与“紧致”不是同义词定义。

**外部输入定理 3.19（构造性等于紧致性）.** 在定理 3.17 的假设下，

$$
\mathbf{SH}_c(S)=\mathbf{SH}(S)^\omega.
$$

**依赖源与边界.** 这是定理 3.17 与一般 compact-generation 定理的合成：
由一集合紧致对象生成的 stable presentable infinity-category，其全部紧致
对象正是生成子的 idempotent-complete thick closure。几何输入来自 Hoyois
Proposition 6.4(2)-(3)；一般范畴论输入见 Lurie, *Higher Topos Theory*,
Propositions 5.3.5.11、5.5.7.8 与 *Higher Algebra*, Proposition
1.4.4.1。前两项把 compactly generated category 识别为紧致子范畴的
Ind-completion，后一项控制 stable colimits 与 compactness。应用时先取
生成子的有限 cofiber/suspension closure，再作 idempotent completion；其
Ind-category 由生成性与 `\mathbf{SH}(S)` 等价，故紧致对象恰为该 thick
closure。本书不在缺少紧生成定理的扩展口径中作此识别。

**命题 3.20（生成子检测等价）.** 若 `\mathcal G` 是 stable presentable
infinity-category `\mathcal C` 的一组生成子，则态射 `u:E\to F` 是等价，
当且仅当对所有 `G\in\mathcal G` 和 `n\in\mathbb Z`，

$$
\operatorname{Map}_{\mathcal C}(\Sigma^nG,u)
$$

为等价。

**证明.** 令 `K` 为 `u` 的 fiber。`u` 为等价当且仅当 `K\simeq0`。若所有映射空间条件成立，则 `\operatorname{Map}(\Sigma^nG,K)` 对所有 `G,n` 可缩。令

$$
\mathcal L=\{A\in\mathcal C\mid \operatorname{Map}_{\mathcal C}(A,K)\simeq *\}.
$$

该 full subcategory 对平移、cofiber 和小余极限封闭，因为 `\operatorname{Map}(-,K)` 把 colimit 变为 limit。它包含所有生成子及其平移，故由生成性得 `\mathcal L=\mathcal C`。取 `A=K`，得 `\operatorname{Map}(K,K)` 可缩，于是恒等态射同伦于零，故 `K\simeq0`。反向由等价态射的 fiber 为零直接得到。`\square`

## 3.6 两个悬挂坐标与稳定范畴

稳定 motivic homotopy theory 通过对称幺半地反演 Tate sphere
`T=\mathbb A^1/(\mathbb A^1-0)` 得到。`T` 与
`S^{1,0}\wedge\mathbb G_m` 的等价及可逆张量积因子引理说明两个坐标在
`\mathbf{SH}(S)` 中分别可逆。`\mathbf{SH}(S)` 的对象反演、谱模型比较和
紧生成性是有精确假设的外部输入；稳定性及若干形式后果在书内从这些输入
推出。默认范围内构造性对象等于紧致对象，但两者的定义仍须区分。

## 练习

**练习 3.1.** 在 pointed spaces 中证明 `\operatorname{cofib}(Y\to *)\simeq\Sigma Y`。

**练习 3.2.** 说明 `\mathbb G_m` 的基点为什么取单位截面。

**练习 3.3.** 用命题 3.5 写出 `S^{2,1}` 与 `T` 的关系。

**练习 3.4.** 证明在 symmetric monoidal category 中，若 `A\otimes B`
可逆，则 `A` 与 `B` 分别可逆；并指出该结论若只有非交换 monoidal
structure 时需要怎样修改。

**练习 3.5.** 证明命题 3.20 中的 full subcategory `\mathcal L` 对 cofiber 封闭。
