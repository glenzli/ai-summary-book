# 第一章：基概形、光滑站点与 Nisnevich descent

## 本章目标

本章建立 motivic homotopy theory 的输入站点。核心对象不是所有概形，而是基概形 `S` 上的光滑有限型概形范畴 `\operatorname{Sm}_S`，并在其上赋予 Nisnevich topology。本章证明若干只依赖 sheaf theory 的基本命题，为第二章的 `\mathbb A^1`-局部化做准备。

## 依赖前置知识

需要概形、光滑态射、etale 态射、Grothendieck topology、presheaves、sheaves、spaces、Yoneda embedding 和 presentable infinity-categories。

## 1.1 基概形与光滑站点

**约定 1.1.** 本章固定有限维 Noetherian 概形 `S`。记

$$
\operatorname{Sm}_S
$$

为 `S` 上光滑有限型概形范畴的一个小骨架。对象写作 `X\to S`，态射为 `S`-态射。

**定义 1.2.** `\operatorname{Sm}_S` 上的 presheaf of spaces 是函子

$$
F:\operatorname{Sm}_S^{op}\longrightarrow\mathcal S.
$$

这些对象组成 presentable infinity-范畴

$$
\mathcal P(\operatorname{Sm}_S)=\operatorname{Fun}(\operatorname{Sm}_S^{op},\mathcal S).
$$

**命题 1.3.** `\mathcal P(\operatorname{Sm}_S)` 是 presentable infinity-范畴，且由 representables 在小余极限下生成。

**证明.** `\operatorname{Sm}_S` 是小 infinity-范畴。任意小 infinity-范畴 `C` 的 presheaf 范畴 `\operatorname{Fun}(C^{op},\mathcal S)` 是自由 cocompletion，故 presentable，并由 Yoneda 嵌入的对象在小余极限下生成。取 `C=\operatorname{Sm}_S` 即得结论。`\square`

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

**定义 1.6.** 一个 `\operatorname{Sm}_S` 中的族

$$
\{U_i\to X\}_{i\in I}
$$

称为 Nisnevich covering family，若每个 `U_i\to X` 为 etale，且对每个点 `x\in X`，存在 `i` 和点 `u\in U_i` 映到 `x`，使得剩余域扩张

$$
\kappa(x)\longrightarrow\kappa(u)
$$

为同构。

**命题 1.7.** 定义 1.6 给出 `\operatorname{Sm}_S` 上的 Grothendieck topology。

**证明.** 需要验证三点。恒等覆盖满足条件，因为取 `u=x`。若 `\{U_i\to X\}` 是覆盖且 `Y\to X` 是任意态射，则 `U_i\times_XY\to Y` 仍 etale；对 `y\in Y`，令 `x` 为其像，取 `u\in U_i` 使 `\kappa(x)\simeq\kappa(u)`。纤维积中存在点 `v` 位于 `u` 与 `y` 上方，并且 etale 态射的剩余域提升条件保持 Nisnevich 点提升性质，得到覆盖的稳定性。若 `\{U_i\to X\}` 是覆盖且每个 `U_i` 有覆盖 `\{V_{ij}\to U_i\}`，则对 `x\in X` 先选 `u\in U_i` 且 `\kappa(x)\simeq\kappa(u)`，再选 `v\in V_{ij}` 且 `\kappa(u)\simeq\kappa(v)`；合成给出 `\kappa(x)\simeq\kappa(v)`。故覆盖可复合。`\square`

**定义 1.8.** `\operatorname{Shv}_{Nis}(\operatorname{Sm}_S)` 是满足 Nisnevich descent 的 space-valued sheaves 组成的 full subcategory。

**命题 1.9.** `\operatorname{Shv}_{Nis}(\operatorname{Sm}_S)` 是 presentable infinity-范畴，且包含反射局部化

$$
L_{Nis}:\mathcal P(\operatorname{Sm}_S)\rightleftarrows
\operatorname{Shv}_{Nis}(\operatorname{Sm}_S):i.
$$

**证明.** `\operatorname{Sm}_S` 是小站点。space-valued sheaves 可作为 presheaf 范畴中满足一组 covering sieve descent 条件的局部对象。对小站点，该条件由一集合态射的局部化给出，因此 sheaf subcategory 是 accessible left exact localization 的本质像。accessible localization 的本质像 presentable，左伴随即 sheafification。`\square`

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

**证明.** 由 elementary square 的定义，`U\amalg V\to X` 是 Nisnevich covering family：`U` 覆盖开集中的点；闭补中的点由 `V\to X` 中剩余域同构的唯一提升覆盖。对覆盖 `U\amalg V\to X` 应用 Cech descent。由于 `U\times_XV\simeq V'`，而 `U\times_XU` 和 `V\times_XV` 中的退化项在 sheaf 条件的等化数据中分别表达两个限制的自相容性，得到 `F(X)` 是 `F(U)` 与 `F(V)` 在 `F(V')` 上的同伦拉回。`\square`

**注 1.14.** 反方向，即 elementary Nisnevich squares 连同空集条件生成 Nisnevich descent，需要 cd-structure 的完整理论。本章暂不使用该反向，后续附录 B 补全。

## 1.4 例子与非例子

**例子 1.15.** 由 `X\in\operatorname{Sm}_S` 表示的 presheaf `h_X` 通常不是任意拓扑下自动 sheaf，但对 subcanonical topology 是 sheaf。Nisnevich topology 是 subcanonical 的，因此 `h_X` 是 Nisnevich sheaf。

**证明.** Nisnevich covering morphisms 是 etale morphisms，特别是概形范畴中的有效下降覆盖。概形态射对 etale 覆盖满足 descent，因此 representable presheaves 满足 Nisnevich sheaf 条件。`\square`

**例子 1.16.** 常值 presheaf `A` 未必是 Nisnevich sheaf；其 sheafification 才是常值 Nisnevich sheaf。若 `X` 非连通，常值 presheaf 不区分连通分支上的局部粘合，而 sheafification 会记录局部常值行为。

**命题 1.17.** 若 `F` 是 Nisnevich sheaf，且 `F(X)\to F(X\times_S\mathbb A^1_S)` 对所有 `X` 为等价，则 `F` 已满足第二章中的 `\mathbb A^1`-局部对象条件。

**证明.** 第二章将 `\mathbb A^1`-局部对象定义为对所有投影 `X\times\mathbb A^1\to X` 取映射空间后得到等价。由 Yoneda lemma，`\operatorname{Map}(X,F)\simeq F(X)`，`\operatorname{Map}(X\times\mathbb A^1,F)\simeq F(X\times\mathbb A^1)`。给定假设正是这些映射空间等价。`\square`

## 1.5 本章小结

Motivic homotopy theory 的第一层输入是小站点 `\operatorname{Sm}_S` 和 Nisnevich topology。Presheaves 提供自由 cocompletion，Nisnevich sheaves 通过 accessible left exact localization 得到。Descent 条件可由覆盖的 Cech nerve 表达；elementary Nisnevich squares 为后续局部化和 purity 提供几何基本块。

## 练习

**练习 1.1.** 证明 `\operatorname{Sm}_S` 对纤维积封闭。

**练习 1.2.** 写出 Nisnevich covering family 的点提升条件，并与 Zariski 覆盖比较。

**练习 1.3.** 对两个开集覆盖 `U,V\subset X`，把命题 1.11 写成通常的 sheaf gluing 条件。

**练习 1.4.** 验证 elementary Nisnevich square 中 `U\amalg V\to X` 是 Nisnevich covering family。

**练习 1.5.** 解释为什么 representable presheaf 的 Nisnevich sheaf 性不是纯形式 Yoneda 结论，而使用了 topology 的 subcanonical 性。
