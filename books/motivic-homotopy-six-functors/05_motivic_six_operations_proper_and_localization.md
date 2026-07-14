# 第五章：Motivic 六操作、proper compatibility 与 localization

稳定化给出了每个基概形上的 `\mathbf{SH}(S)`，但几何计算要求这些范畴随基变化。
若 `X` 分成开部分 `U` 与闭补 `Z`，一个谱应当能由它在两部分上的限制及粘合数据恢复；
若 `f` proper，紧支撑推前与普通推前应当一致。两件事分别表现为 recollement 和
`f_!\simeq f_*`，它们都不是第三章稳定化泛性质的自动后果。

本章把 motivic 六操作 package 作为精确标注的外部输入，然后在该输入之上完成形式
推导。读者只需掌握第四章的伴随与 stable cofiber calculus，几何上则要区分 proper、
open、closed 和一般 separated 态射。这样，localization triangle 的每一项、每个
单位余单位和紧致性结论都有确定类型，也为下一章从闭补走向法丛的 purity 定理作准备。

## 5.1 Motivic 系数系统

**约定 5.1.** 固定附录 A 的 `\mathbb U`-小有限维 Noetherian 概形 `B`。
本章的基范畴 `\mathcal B` 是有限型 `B`-概形的范畴；它对有限纤维积封闭，
其对象均为 qcqs、有限维 Noetherian 概形。任意 `B`-态射记为
`f:X\to Y`；在这个基范畴中它自动有限型。Exceptional 态射类
`\mathcal E` 取 separated 态射；由 Nagata compactification，此类态射可
compactify。若改用任意 qcqs 概形、非 separated exceptional 态射或代数栈，
必须另行声明并重新核查外部输入。

**外部输入定理 5.2（Motivic 六操作的定义域与方差）.** 赋值

$$
S\longmapsto\mathbf{SH}(S)
$$

扩展为如下相干系数系统：

1. 对每个 `f:X\to Y`，有保持小余极限的强对称幺半正合函子
   `f^*:\mathbf{SH}(Y)\to\mathbf{SH}(X)` 及伴随 `f^*\dashv f_*`；
2. 若 `f` smooth，则另有 `f_\sharp\dashv f^*`；
3. 若 `f` separated，则有 `f_!\dashv f^!`；
4. 每个 `\mathbf{SH}(X)` 都是 closed presentably symmetric monoidal
   stable infinity-category，内部操作为 `\otimes_X` 与
   `\underline{\operatorname{Hom}}_X`；
5. 上述函子具有 identity、composition 和 base-change pasting 的
   infinity-categorical coherence。

这里 `f_!`、`f^!` 对非 separated `f` 不属于本章定理；`f_\sharp` 只在
smooth 情形定义，且不应与 `f_!` 混记。

**依赖源与边界.** Hoyois, *The six operations in equivariant motivic homotopy
theory*, Theorem 1.1 和 Theorem 6.18，取平凡群；该文在平凡群情形允许 qcqs
基，exceptional adjunction 对 compactifiable morphisms 定义，本书以
Noetherian Nagata compactification 把 separated finite-type 态射纳入该类。
Ayoub 给出非等变 motivic 原始构造。Drew--Gallauer Theorem 7.14 说明
universal coefficient-system 性质，但不单独证明本定理列出的全部六操作
相容式。

**定义 5.3.** 对 `S`，记

$$
\mathbb 1_S\in\mathbf{SH}(S)
$$

为 motivic sphere spectrum。若 `f:X\to Y`，则 `f^*` 强对称幺半，故有自然等价

$$
f^*\mathbb 1_Y\simeq\mathbb 1_X.
$$

**命题 5.4.** 对任意 `E,F\in\mathbf{SH}(Y)`，有自然等价

$$
f^*(E\otimes_YF)\simeq f^*E\otimes_Xf^*F.
$$

**证明.** 定理 5.2 把 `f^*` 作为六操作形式主义中的强对称幺半 pullback。强对称幺半函子的结构态射正给出单位和张量积的自然等价。`\square`

## 5.2 Proper compatibility

**定义 5.5.** 若 `f:X\to Y` 为 proper 态射，proper compatibility 是六操作形式主义中的自然等价

$$
f_!\simeq f_*.
$$

**外部输入定理 5.6.** Motivic 六操作满足 proper compatibility：对
proper `f`，有自然等价 `f_!\simeq f_*`，并与复合、base change 和
projection maps 相干。

**依赖源.** Hoyois, Theorem 6.18(1)；proper base change 与 proper
projection formula 分别见 Corollaries 6.10 与 6.11。Ayoub 给出非等变
版本。

**命题 5.7.** 若 `f:X\to Y` 与 `g:Y\to Z` 均 proper，则 `(gf)_!\simeq(gf)_*` 与 `g_!f_!\simeq g_*f_*` 相容。

**证明.** proper 态射在复合下封闭。六操作复合相干给出

$$
(gf)_!\simeq g_!f_!,\qquad (gf)_*\simeq g_*f_*.
$$

对 `f` 和 `g` 使用 proper compatibility 得到 `g_!f_!\simeq g_*f_*`。定理 5.6 要求 proper compatibility 与复合相干，因此这条等价与 `(gf)` 的 proper compatibility 一致。`\square`

**推论 5.8.** 若 `f` proper，则对任意
`A\in\mathbf{SH}(X)`、`B\in\mathbf{SH}(Y)`，ordinary projection map

$$
f_*A\otimes_YB\longrightarrow f_*(A\otimes_X f^*B)
$$

为等价；等价地，可写成

$$
f_*(A\otimes_X f^*B)\simeq f_*A\otimes_YB.
$$

**证明.** Motivic `!`-projection formula 是外部输入定理 4.17 的组成部分，
给出 `f_!(A\otimes f^*B)\simeq f_!A\otimes B`。由定理 5.6 的相干
proper comparison 把两个 `f_!` 替换为 `f_*`，得到 ordinary projection
map 的逆为等价，故 ordinary projection map 本身为等价。`\square`

## 5.3 Open immersion 与 extension by zero

**定义 5.9.** 对开嵌入 `j:U\hookrightarrow X`，`j_!` 称为 extension by zero。它是 `j^!` 的左伴随；在 motivic 六操作中还满足 `j^!\simeq j^*`。

**外部输入定理 5.10.** 对开嵌入 `j`，有自然等价

$$
j^!\simeq j^*
$$

并且 `j_!` fully faithful。由于 `j` etale，`T_j=0`；该等价也与第六章的
smooth purity 口径一致。进一步有 `j_!\simeq j_\sharp`，但这个比较依赖
smooth ambidexterity，不是两个左伴随符号的定义相等。

**命题 5.11.** 对开嵌入 `j:U\hookrightarrow X`，单位态射

$$
E\longrightarrow j^*j_!E
$$

为等价。

**证明.** `j_!` fully faithful 等价于伴随 `j_!\dashv j^!` 的单位 `\operatorname{id}\to j^!j_!` 为等价。由定理 5.10，`j^!\simeq j^*`，因此得到 `E\simeq j^*j_!E`。`\square`

**命题 5.12.** 若 `E\in\mathbf{SH}(X)` 且 `j^*E\simeq0`，则 `E` 支撑在闭补上，即在 localization recollement 下 `E\simeq i_*F` 对某个 `F\in\mathbf{SH}(Z)` 成立。

**证明.** 设 `i:Z\hookrightarrow X` 为闭补。由定理 5.2 的 localization recollement，`\mathbf{SH}(X)` 中被 `j^*` 杀掉的 full subcategory 等于 `i_*` 的本质像。故存在 `F` 使 `E\simeq i_*F`。`\square`

## 5.4 Closed immersion 与 localization cofiber sequence

**定义 5.13.** 对闭嵌入 `i:Z\hookrightarrow X` 和开补 `j:U\hookrightarrow X`，motivic localization triangle 指自然 cofiber sequence

$$
j_!j^*E\longrightarrow E\longrightarrow i_*i^*E.
$$

**外部输入定理 5.14（Motivic localization）.** 对任意闭嵌入
`i:Z\hookrightarrow X` 及其开补 `j:U\hookrightarrow X`，
`\mathbf{SH}(X)`、`\mathbf{SH}(U)` 和 `\mathbf{SH}(Z)` 组成
open-closed recollement。特别地，`i_*`、`j_!` fully faithful，并有自然
cofiber sequences

$$
j_!j^*E\longrightarrow E\longrightarrow i_*i^*E,
\qquad
i_*i^!E\longrightarrow E\longrightarrow j_*j^*E.
$$

**依赖源.** Hoyois, Theorem 6.18(4)-(5)，取平凡群。第一条 sequence 是
定义 5.13 使用的 localization sequence；第二条不能通过把第一条中的
`i^*` 字面替换成 `i^!` 得到，而是 recollement 的另一半。

**命题 5.15.** 对 `E=\mathbb 1_X`，有 cofiber sequence

$$
j_!\mathbb 1_U\longrightarrow\mathbb 1_X\longrightarrow i_*\mathbb 1_Z.
$$

**证明.** 由 `j^*` 和 `i^*` 强对称幺半，`j^*\mathbb 1_X\simeq\mathbb 1_U` 且 `i^*\mathbb 1_X\simeq\mathbb 1_Z`。代入定义 5.13。`\square`

**命题 5.16.** 若 `E\in\mathbf{SH}(X)` 且 `i^*E\simeq0`，则 counit

$$
j_!j^*E\longrightarrow E
$$

为等价。

**证明.** localization cofiber sequence 给出

$$
j_!j^*E\to E\to i_*i^*E.
$$

若 `i^*E\simeq0`，则第三项为零对象。稳定范畴中 cofiber 为零等价于第一箭头为等价。`\square`

## 5.5 支撑与 exceptional functors

**定义 5.17.** 对闭嵌入 `i:Z\hookrightarrow X`，定义带支撑对象的 full subcategory

$$
\mathbf{SH}_Z(X)=\{E\in\mathbf{SH}(X)\mid j^*E\simeq0\},
$$

其中 `j:X\setminus Z\hookrightarrow X` 是开补。

**命题 5.18.** `i_*:\mathbf{SH}(Z)\to\mathbf{SH}(X)` 诱导等价

$$
\mathbf{SH}(Z)\simeq\mathbf{SH}_Z(X).
$$

**证明.** 定理 5.14 的 recollement 包含 `i_*` fully faithful，并且其本质像正是被 `j^*` 杀掉的对象。定义 5.17 说明该本质像就是 `\mathbf{SH}_Z(X)`。`\square`

**注 5.19.** `i^!` 与 `i^*` 对闭嵌入一般不同。二者的关系由 purity 或 absolute purity 控制，需要第六章的 Thom twist 和法丛数据。

## 5.6 紧致性、构造性与六操作

**命题 5.20（伴随的紧致性判据）.** 设
`L:\mathcal C\rightleftarrows\mathcal D:R` 是 presentable
infinity-categories 间的伴随。若 `R` 保持 filtered colimits，则 `L` 把
紧致对象送到紧致对象。

**证明.** 设 `K\in\mathcal C` 紧致，`(D_i)_{i\in I}` 为 filtered diagram。
由伴随、`R` 的假设和 `K` 的紧致性，依次有

$$
\begin{aligned}
\operatorname{Map}_{\mathcal D}(LK,\operatorname*{colim}_iD_i)
&\simeq \operatorname{Map}_{\mathcal C}
   (K,R\operatorname*{colim}_iD_i)\\
&\simeq \operatorname{Map}_{\mathcal C}
   (K,\operatorname*{colim}_iRD_i)\\
&\simeq \operatorname*{colim}_i
   \operatorname{Map}_{\mathcal C}(K,RD_i)\\
&\simeq \operatorname*{colim}_i
   \operatorname{Map}_{\mathcal D}(LK,D_i).
\end{aligned}
$$

故 `LK` 紧致。`\square`

**外部输入定理 5.21（continuity）.** 在约定 5.1 下：

1. 对任意 `f:X\to Y`，右伴随 `f_*` 保持全部小余极限；
2. 对 separated `f`，exceptional pullback `f^!` 保持全部小余极限。

**依赖源与边界.** Hoyois, Proposition 6.4(4) 给出第 1 项；Corollary 6.19
对 compactifiable `f` 给出第 2 项。约定 5.1 的 separated 态射由 Nagata
定理 compactifiable。这里的结论强于只保持 filtered colimits，但它不表示
`f_*` 或 `f^!` 自动保持紧致对象。

**推论 5.22.** 在约定 5.1 下：

1. 任意 `f` 的 `f^*` 保持紧致对象；
2. smooth `f` 的 `f_\sharp` 保持紧致对象；
3. separated `f` 的 `f_!` 保持紧致对象；
4. proper `f` 的 `f_*` 保持紧致对象。

在定理 3.19 的范围内，把“紧致”换成“几何构造性”仍成立。

**证明.** 第 1 项对伴随 `f^*\dashv f_*` 使用命题 5.20 和定理 5.21(1)。
第 2 项对 `f_\sharp\dashv f^*` 使用命题 5.20；`f^*` 本来就在
`\operatorname{Pr}^L` 中，故保持 filtered colimits。第 3 项对
`f_!\dashv f^!` 使用定理 5.21(2)。若 `f` proper，定理 5.6 给出
`f_*\simeq f_!`，第 4 项随之成立。最后由定理 3.19 的
`\mathbf{SH}_c=\mathbf{SH}^\omega` 得构造性版本。`\square`

**注 5.23.** 推论 5.22 没有断言任意 `f_*` 或任意 `f^!` 保持紧致对象。
“一个函子保持小余极限”与“它把紧致对象送到紧致对象”是不同性质；后者
通常通过其右伴随的 filtered-colimit continuity 检验。其他系数理论中的
constructibility preservation 也可能需要 quasi-excellence、系数可逆性或
有限 Tor-dimension 等附加假设。

**例子 5.24（仿射直线的原点与开补）.** 令
`X=\mathbb A^1_S`，`i:S\hookrightarrow X` 为零截面，
`j:\mathbb G_{m,S}\hookrightarrow X` 为其开补。定理 5.14 对单位谱给出

$$
j_!\mathbb 1_{\mathbb G_m}\longrightarrow
\mathbb 1_{\mathbb A^1}\longrightarrow i_*\mathbb 1_S.
$$

对该序列作用 `j^*`，由 `j^*j_!\simeq\operatorname{id}`、
`j^*i_*=0`，得到
`\mathbb 1_{\mathbb G_m}\xrightarrow{\simeq}
\mathbb 1_{\mathbb G_m}\to0`。作用 `i^*` 则由
`i^*j_!=0`、`i^*i_*\simeq\operatorname{id}` 得到
`0\to\mathbb 1_S\xrightarrow{\simeq}\mathbb 1_S`。因此三项分别记录
开部分、整个仿射直线和原点支撑；第六章的 purity 会进一步把第三项的
exceptional 信息识别为零截面法线丛的 Thom twist。

## 5.7 开闭粘合与 proper 推前

Motivic 六操作把 `\mathbf{SH}(-)` 组织成几何系数系统：`f^*\dashv f_*`
对所有默认态射存在，`f_!\dashv f^!` 只对 separated 态射使用。
Proper compatibility 允许对 proper 态射把 `f_!` 与 `f_*` 识别；open-closed
localization 把对象分解为开部分和闭支撑部分。Continuity 定理与伴随判据
精确控制若干操作何时保持紧致/构造性对象。第六章将区分 smooth purity、
lci purity transformation 与 coefficientwise absolute purity。

## 练习

**练习 5.1.** 证明 proper morphisms 在复合下封闭，并说明该事实在命题 5.7 中的位置。

**练习 5.2.** 对开嵌入 `j`，从 `j_!` fully faithful 推出 `j^*j_!\simeq\operatorname{id}`。

**练习 5.3.** 用 localization cofiber sequence 证明命题 5.16。

**练习 5.4.** 解释为什么 `i^*E\simeq0` 与 `j^*E\simeq0` 表示不同支撑条件。

**练习 5.5.** 写出 `E=\Sigma_T^\infty X_+` 时命题 5.15 的形式。

**练习 5.6.** 用命题 5.20 证明：若 `f` proper，则 `f_*` 保持紧致对象；
指出证明中 properness 被使用的唯一位置。
