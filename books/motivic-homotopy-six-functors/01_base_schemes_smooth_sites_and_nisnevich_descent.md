# 第一章：基概形、光滑站点与 Nisnevich descent

一个几何不变量若只按光滑概形逐点赋值，还不能称为几何理论：在开集或 etale 邻域
上得到的局部数据必须能够粘合。Zariski 拓扑对许多代数问题过粗，而 etale 拓扑又会
改变点的剩余域；Nisnevich 覆盖恰好要求每个点在某个 etale 邻域中以相同剩余域提升，
因而能同时看见局部几何与点值信息。这一条件将决定后文 deformation、purity 和
localization 可以怎样局部检验。

我们固定有限维 Noetherian 基概形 `S`，先在 `\operatorname{Sm}_S` 的小骨架上构造
空间值预层，再用覆盖筛和 Cech 神经表达下降。所需的概形、Yoneda 嵌入与
presentable infinity-范畴知识分别由附录 E、A 补足；本章会把每个层论结论落实到
elementary Nisnevich square，而不是把“满足下降”当作未展开的口号。

## 1.1 基概形与光滑站点

**约定 1.1.** 本章固定 `\mathbb U`-小有限维 Noetherian 概形 `S`。记

$$
\operatorname{Sm}_S
$$

为 `S` 上光滑有限型概形范畴的一个 `\mathbb U`-小骨架。对象写作
`X\to S`，态射为 `S`-态射。骨架存在性和选择独立性见命题 A.3 与注 A.4。

**定义 1.2.** `\operatorname{Sm}_S` 上的 presheaf of spaces 是函子

$$
F:\operatorname{Sm}_S^{op}\longrightarrow\mathcal S_{\mathbb U}.
$$

这些对象组成 presentable infinity-范畴

$$
\mathcal P(\operatorname{Sm}_S)=\operatorname{Fun}(\operatorname{Sm}_S^{op},\mathcal S).
$$

**命题 1.3.** `\mathcal P(\operatorname{Sm}_S)` 是 presentable infinity-范畴，且由 representables 在小余极限下生成。

**证明.** `\operatorname{Sm}_S` 是 `\mathbb U`-小。对它应用外部输入定理
A.5（HTT Theorem 5.1.5.6 与 Corollary 5.1.5.8），即得 presentability、
Yoneda 生成性和相应高阶相干。`\square`

**定义 1.4.** Yoneda 嵌入记为

$$
y:\operatorname{Sm}_S\longrightarrow \mathcal P(\operatorname{Sm}_S),\qquad
X\longmapsto h_X=\operatorname{Map}_{\operatorname{Sm}_S}(-,X).
$$

**命题 1.5.** 对任意 presheaf `F` 和 `X\in\operatorname{Sm}_S`，有自然等价

$$
\operatorname{Map}_{\mathcal P(\operatorname{Sm}_S)}(h_X,F)\simeq F(X).
$$

**证明.** 这是 infinity-categorical Yoneda lemma。由于 `h_X` 是 `X` 表示的 presheaf，自然变换 `h_X\to F` 等价于 `F` 在 `X` 上的值。`\square`

## 1.2 Nisnevich topology

**定义 1.6.** 一个 `\operatorname{Sm}_S` 中的有限族

$$
\{U_i\to X\}_{i\in I}
$$

称为 Nisnevich covering family，若每个 `U_i\to X` 为 etale，且对每个点 `x\in X`，存在 `i` 和点 `u\in U_i` 映到 `x`，使得剩余域扩张

$$
\kappa(x)\longrightarrow\kappa(u)
$$

为同构。

**命题 1.7.** 定义 1.6 给出 `\operatorname{Sm}_S` 上的 Grothendieck topology。

**证明.** 恒等覆盖满足条件，取 `u=x`。设 `\{U_i\to X\}` 为覆盖，
`Y\to X` 任意。Etale morphisms 对 base change 稳定。给定 `y\in Y`，令
`x` 为其像，取 `u\in U_i` 使 `\kappa(u)\simeq\kappa(x)`。纤维积在
`(u,y)` 上的 residue algebra 为

$$
\kappa(u)\otimes_{\kappa(x)}\kappa(y)\simeq\kappa(y),
$$

故有一点 `v\in U_i\times_XY` 位于 `y` 上方且
`\kappa(v)\simeq\kappa(y)`。这证明 base-change 稳定性。若每个 `U_i` 又有
有限 Nisnevich cover `\{V_{ij}\to U_i\}`，则先选
`\kappa(x)\simeq\kappa(u)`，再选 `\kappa(u)\simeq\kappa(v)`；复合给出
所需 residue-field isomorphism，且有限族的有限复合仍有限。故三条
Grothendieck-topology 公理成立。

通常允许任意 covering families 也给出同一 topology：`X` Noetherian
因而 quasi-compact。更精确地，对 etale map `U_i\to X`，令 `W_i\subset X`
为存在 `u\in U_i` 且 `\kappa(u)\simeq\kappa(x)` 的点 `x` 所成的 splitting
locus。Etale 邻域引理说明：这样的 `u` 在缩小 `x` 的开邻域后给出局部
截面，所以 `W_i` 为开集。Nisnevich 条件说 `(W_i)` 覆盖 `X`；由
quasi-compactness 可取有限子覆盖，相应的有限子族仍满足剩余域提升条件。
本书使用有限族，以保证 `\coprod_iU_i` 仍在 `\operatorname{Sm}_S` 中。
`\square`

**定义 1.8.** `\operatorname{Shv}_{Nis}(\operatorname{Sm}_S)` 是满足 Nisnevich descent 的 space-valued sheaves 组成的 full subcategory。

**约定.** 这里是 Cech sheaves，不默认 hypercomplete。若后文需要
hyperdescent，将显式写
`\operatorname{Shv}_{Nis}(\operatorname{Sm}_S)^\wedge`。

**命题 1.9.** `\operatorname{Shv}_{Nis}(\operatorname{Sm}_S)` 是 presentable infinity-范畴，且包含反射局部化

$$
L_{Nis}:\mathcal P(\operatorname{Sm}_S)\rightleftarrows
\operatorname{Shv}_{Nis}(\operatorname{Sm}_S):i.
$$

**证明.** 对 `\mathbb U`-小站点 `\operatorname{Sm}_S` 应用外部输入定理
A.8，即 HTT Proposition 6.2.2.7。该定理同时给出反射性、accessibility、
left exactness 和 presentability；其局部对象由覆盖筛
`R\hookrightarrow h_X` 的一集合 maps 检测。`\square`

## 1.3 Descent 条件

**定义 1.10.** 对 Nisnevich covering family `U=\coprod_iU_i\to X`，其 Cech nerve 是增广 simplicial object

$$
U_\bullet\to X,\qquad
U_n=U\times_X\cdots\times_XU
$$

其中右侧含 `n+1` 个 `U`。

**命题 1.11.** presheaf `F` 是 Nisnevich sheaf 当且仅当对每个 Nisnevich covering family，映射

$$
F(X)\longrightarrow \lim_{\Delta}F(U_\bullet)
$$

为等价，并且该条件对覆盖筛相容。

**证明.** sheaf 条件在 infinity-范畴中表示为对覆盖筛 `R\subset h_X`，映射 `F(X)\to\operatorname{Map}(R,F)` 为等价。若覆盖由族 `U\to X` 给出，则覆盖筛由 Cech nerve 的几何实现生成；映射到 `F` 把几何实现变为极限，得到右侧。反向地，若对所有覆盖族的 Cech nerve 满足该极限条件，并且对细化相容，则对覆盖筛也满足同一映射空间条件。`\square`

**定义 1.12.** 一个 elementary Nisnevich square 是 Cartesian 方块

$$
\begin{array}{c}
V'\longrightarrow V\\
\downarrow\qquad\downarrow p\\
U\overset{j}{\longrightarrow}X
\end{array}
$$

其中 `j` 是开嵌入，`p` 是 etale，且诱导的闭补映射

$$
V\setminus V'\longrightarrow X\setminus U
$$

为同构。

**命题 1.13.** 若 `F` 是 Nisnevich sheaf，则对每个 elementary Nisnevich square，方块

$$
\begin{array}{c}
F(X)\longrightarrow F(U)\\
\downarrow\qquad\downarrow\\
F(V)\longrightarrow F(V')
\end{array}
$$

是 spaces 中的拉回方块。

**证明.** 附录 B 的外部输入定理 B.11(1) 断言：在 Nisnevich
infinity-topos 中，elementary distinguished square 的 representable sheaves
组成推出方块

$$
h_X\simeq h_U\mathop{\amalg}_{h_{V'}}h_V.
$$

对该推出方块应用反变函子
`\operatorname{Map}_{\operatorname{Shv}_{Nis}}(-,F)`。映射出余极限等于
映射空间的极限，Yoneda lemma 又给出
`\operatorname{Map}(h_Y,F)\simeq F(Y)`，故

$$
F(X)\simeq F(U)\times_{F(V')}F(V).
$$

这里不能只把覆盖 `U\amalg V\to X` 的 Cech 神经截断在一次交叠：
`V\times_XV` 一般不等于 `V`。Precisely 是 distinguished-square
excision 定理把完整的高阶下降数据压缩为上述同伦拉回。`\square`

**注 1.14.** 反方向，即把空集送到终对象并把每个 elementary
Nisnevich square 送到同伦拉回便推出 Nisnevich descent，也需要
cd-structure 的生成定理。正反两个方向及其来源统一列在附录 B 的定理 B.11。

## 1.4 例子与非例子

**例子 1.15.** 由 `X\in\operatorname{Sm}_S` 表示的 presheaf `h_X` 通常不是任意拓扑下自动 sheaf，但对 subcanonical topology 是 sheaf。Nisnevich topology 是 subcanonical 的，因此 `h_X` 是 Nisnevich sheaf。

**证明.** Nisnevich covering morphisms 是 etale morphisms，特别是概形范畴中的有效下降覆盖。概形态射对 etale 覆盖满足 descent，因此 representable presheaves 满足 Nisnevich sheaf 条件。`\square`

**例子 1.16.** 常值 presheaf `A` 未必是 Nisnevich sheaf；其 sheafification 才是常值 Nisnevich sheaf。若 `X` 非连通，常值 presheaf 不区分连通分支上的局部粘合，而 sheafification 会记录局部常值行为。

**命题 1.17.** 若 `F` 是 Nisnevich sheaf，且 `F(X)\to F(X\times_S\mathbb A^1_S)` 对所有 `X` 为等价，则 `F` 已满足第二章中的 `\mathbb A^1`-局部对象条件。

**证明.** 第二章将 `\mathbb A^1`-局部对象定义为对所有投影 `X\times\mathbb A^1\to X` 取映射空间后得到等价。由 Yoneda lemma，`\operatorname{Map}(X,F)\simeq F(X)`，`\operatorname{Map}(X\times\mathbb A^1,F)\simeq F(X\times\mathbb A^1)`。给定假设正是这些映射空间等价。`\square`

## 1.5 Nisnevich 方块留下的几何信息

Motivic homotopy theory 的第一层输入是小站点 `\operatorname{Sm}_S` 和 Nisnevich topology。Presheaves 提供自由 cocompletion，Nisnevich sheaves 通过 accessible left exact localization 得到。Descent 条件可由覆盖的 Cech nerve 表达；elementary Nisnevich squares 为后续局部化和 purity 提供几何基本块。

## 练习

**练习 1.1.** 证明 `\operatorname{Sm}_S` 对纤维积封闭。

**练习 1.2.** 写出 Nisnevich covering family 的点提升条件，并与 Zariski 覆盖比较。

**练习 1.3.** 对两个开集覆盖 `U,V\subset X`，把命题 1.11 写成通常的 sheaf gluing 条件。

**练习 1.4.** 验证 elementary Nisnevich square 中 `U\amalg V\to X` 是 Nisnevich covering family。

**练习 1.5.** 解释为什么 representable presheaf 的 Nisnevich sheaf 性不是纯形式 Yoneda 结论，而使用了 topology 的 subcanonical 性。
