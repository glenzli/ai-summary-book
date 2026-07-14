# 第二十三章：可表现 $\infty$-范畴、可达局部化与 $\operatorname{Pr}^L$

大多数高阶范畴不能由对象清单掌握，却常能由一小族 $\kappa$-紧对象经滤过余极限生成。Presentable $\infty$-范畴把这种可达性与全部小余极限结合起来，并可表示为预层 $\infty$-范畴的可达局部化。以保持余极限的函子为态射得到 $\operatorname{Pr}^L$，它为高阶 topos、高阶代数和导出几何提供共同工作环境。本章把第十二章的普通理论提升到映射空间口径，并证明伴随函子与局部化的关键判据。

读者需要 quasi-category、映射空间、极限/余极限和 left exact localization。紧性与可达性始终相对于明确的正则基数；凡使用 adjoint functor theorem 或 Ind-completion 的普适性质，都会列出小性与余极限保持条件。

## 23.1 预层 $\infty$-范畴与 Yoneda

**定义 23.1.** 设 $C$ 为小 $\infty$-范畴。其 space 值预层 $\infty$-范畴定义为

$$
\mathcal P(C)=\operatorname{Fun}(C^{op},\mathcal S).
$$

Yoneda 嵌入写作

$$
j:C\to\mathcal P(C),\qquad x\mapsto\operatorname{Map}_C(-,x).
$$

**外部输入定理 23.2（$\infty$-Yoneda）.** 对任意 $F\in\mathcal P(C)$ 与 $x\in C$，存在自然等价

$$
\operatorname{Map}_{\mathcal P(C)}(j(x),F)\simeq F(x).
$$

特别地，$j$ 全忠实。证明使用 quasi-category 的映射空间、straightening 或 simplicial presheaf 模型，是第二章 Yoneda 引理的高阶版本。

**命题 23.3.** $\mathcal P(C)$ 的小极限和小余极限逐点计算。

**证明.** 函子 $\infty$-范畴 $\operatorname{Fun}(C^{op},\mathcal S)$ 的极限和余极限由目标 $\mathcal S$ 中的极限和余极限逐点给出。更具体地，若 $D:K\to\mathcal P(C)$ 是图形，则候选余极限 $L$ 定义为

$$
L(c)=\operatorname*{colim}_{k\in K}D(k)(c).
$$

对任意 $F\in\mathcal P(C)$，映射空间满足

$$
\operatorname{Map}_{\mathcal P(C)}(L,F)
\simeq
\lim_{c\in C}\operatorname{Map}_{\mathcal S}(L(c),F(c)).
$$

代入 $L(c)$ 的逐点余极限，并使用 $\mathcal S$ 中余极限的映射空间判别，可得

$$
\operatorname{Map}_{\mathcal P(C)}(L,F)
\simeq
\lim_{k\in K^{op}}\operatorname{Map}_{\mathcal P(C)}(D(k),F).
$$

这正是 $L$ 为余极限的判别。极限同理，或直接由函子范畴极限逐点构造。$\square$

## 23.2 紧对象、可达性与 Ind 完备化

**定义 23.4.** 设 $\kappa$ 为正则基数，$C$ 有 $\kappa$-滤过余极限。对象 $x\in C$ 称为 $\kappa$-紧或 $\kappa$-compact，若

$$
\operatorname{Map}_C(x,-):C\to\mathcal S
$$

保持 $\kappa$-滤过余极限。

**命题 23.5.** 对小 $\infty$-范畴 $C$，每个可表预层 $j(c)$ 是 $\mathcal P(C)$ 中的紧对象。

**证明.** 设 $F_i:I\to\mathcal P(C)$ 为滤过图形。由 $\infty$-Yoneda 和逐点余极限，

$$
\operatorname{Map}_{\mathcal P(C)}(j(c),\operatorname*{colim}_{i\in I}F_i)
\simeq
(\operatorname*{colim}_{i\in I}F_i)(c)
\simeq
\operatorname*{colim}_{i\in I}F_i(c).
$$

再次用 Yoneda，

$$
F_i(c)\simeq\operatorname{Map}_{\mathcal P(C)}(j(c),F_i).
$$

合并得到

$$
\operatorname{Map}_{\mathcal P(C)}(j(c),\operatorname*{colim}_iF_i)
\simeq
\operatorname*{colim}_i\operatorname{Map}_{\mathcal P(C)}(j(c),F_i),
$$

即 $j(c)$ 紧。$\square$

**定义 23.6.** $\infty$-范畴 $C$ 称为 $\kappa$-可达，若：

1. $C$ 有 $\kappa$-滤过余极限；
2. 存在一个由 $\kappa$-紧对象组成的小全子范畴 $C^\kappa$；
3. 每个对象都是 $C^\kappa$ 中对象的 $\kappa$-滤过余极限。

若对某个正则基数 $\kappa$ 成立，则称 $C$ 可达。

**定义 23.7.** 小 $\infty$-范畴 $C$ 的 $\operatorname{Ind}_\kappa$-完备化记为

$$
\operatorname{Ind}_\kappa(C)\subseteq\mathcal P(C),
$$

是由可表预层在 $\mathcal P(C)$ 中经 $\kappa$-滤过余极限生成的全子 $\infty$-范畴。

**外部输入定理 23.8.** $\infty$-范畴 $D$ 是 $\kappa$-可达，当且仅当存在小 $\infty$-范畴 $C$，使 $D$ 等价于 $\operatorname{Ind}_\kappa(C)$ 的一个可达反射子范畴。该定理是第十二章 Ind 刻画的 $\infty$-范畴版本。

## 23.3 Presentable $\infty$-categories

**定义 23.9.** $\infty$-范畴 $C$ 称为 presentable，若它可达并且有所有小余极限。

**例子 23.10.** $\mathcal S$、$\mathcal P(C)$、$\operatorname{Sh}_\infty(\mathcal C,J)$、$\mathbf{Sp}$、以及环谱 $R$ 的模范畴 $\operatorname{Mod}_R$ 都是 presentable $\infty$-categories。

其中 $\operatorname{Sh}_\infty(\mathcal C,J)$ 是 $\mathcal P(\mathcal C)$ 的 left exact accessible localization；$\mathbf{Sp}$ 可由 pointed spaces 稳定化得到；$\operatorname{Mod}_R$ 是高阶代数中代数对象的模范畴。

**外部输入定理 23.11（presentable $\infty$-范畴伴随函子定理）.** 若 $C,D$ 是 presentable $\infty$-categories，则函子

$$
F:C\to D
$$

是左伴随，当且仅当它保持所有小余极限。等价地，$F$ 有右伴随当且仅当 $F$ 保持小余极限。

对偶方向中，右伴随保持小极限；若一个函子保持小极限并满足适当可达性，则它是右伴随。该定理是高阶范畴论中最常用的工具之一。

**命题 23.12.** 若 $F:C\to D$ 是 presentable $\infty$-categories 之间的左伴随，则 $F$ 保持所有 $\kappa$-滤过余极限。

**证明.** 左伴随保持所有小余极限。$\kappa$-滤过余极限是小余极限的一类，因此 $F$ 保持它们。$\square$

## 23.4 可达局部化与局部对象

**定义 23.13.** 设 $C$ 为 $\infty$-范畴。局部化函子由函子

$$
L:C\to C
$$

和自然变换 $\eta:\operatorname{id}_C\to L$ 组成，满足 $L\eta$ 与 $\eta L$ 都是等价。对象 $X$ 若满足 $\eta_X:X\to LX$ 为等价，则称为 $L$-局部对象。$L$-局部对象组成全子范畴 $C_L\subseteq C$。

**命题 23.14.** $C_L\hookrightarrow C$ 有左伴随，左伴随由 $L$ 给出。

**证明.** 对局部对象 $Y$，需证明

$$
\operatorname{Map}_{C_L}(LX,Y)\simeq\operatorname{Map}_C(X,Y)
$$

自然成立。由于 $Y$ 局部，$\eta_Y:Y\to LY$ 为等价。局部化恒等式 $L\eta_X\simeq\eta_{LX}$ 说明 $LX$ 已局部。映射 $X\to Y$ 预复合 $\eta_X:X\to LX$ 给出

$$
\operatorname{Map}_C(LX,Y)\to\operatorname{Map}_C(X,Y).
$$

对该映射施加 $L$，并使用 $LY\simeq Y$ 与 $L\eta_X$ 等价，可构造逆映射。两侧复合由 $\eta$ 的幂等性同伦给出恒等。因此 $L$ 表示到局部对象全子范畴的反射。$\square$

**定义 23.15.** 若 $C$ presentable，且局部化 $L:C\to C$ 是可达函子，则称 $C_L$ 为 $C$ 的 accessible localization。

**外部输入定理 23.16.** Presentable $\infty$-category 的 accessible localization 仍 presentable。反过来，presentable $\infty$-category 可等价地刻画为某个预层 $\infty$-范畴 $\mathcal P(C)$ 的 accessible localization。

**定义 23.17.** 给定一族态射 $S$，对象 $Z$ 称为 $S$-局部，若对每个 $f:A\to B$ 属于 $S$，诱导映射

$$
\operatorname{Map}_C(B,Z)\to\operatorname{Map}_C(A,Z)
$$

是等价。若存在局部化 $L$ 使局部对象正是 $S$-局部对象，则称 $L$ 为 $S$ 的 Bousfield localization。

**例子 23.18.** Sheaf 化是预层范畴的局部化：局部对象是满足覆盖下降的预层。高阶 topos 是预层 $\infty$-范畴的 left exact accessible localization；谱范畴中的许多同调局部化则是稳定 presentable $\infty$-category 中的 Bousfield localizations。

## 23.5 Left exact localization 与稳定局部化

**定义 23.19.** Presentable $\infty$-category $C$ 的局部化 $L:C\to C$ 称为 left exact，若 $L$ 保持有限极限。

**命题 23.20.** 若 $L:\mathcal P(C)\to\mathcal P(C)$ 是 left exact accessible localization，则局部对象全子范畴 $\mathcal P(C)_L$ 是 $\infty$-topos。

**证明.** 由定义 21.6，$\infty$-topos 等价地是 spaces 预层 $\infty$-范畴的 left exact accessible localization。$\mathcal P(C)$ 正是 $C$ 上的 spaces 预层 $\infty$-范畴，故结论直接由定义得到。$\square$

**定义 23.21.** 若 $C$ 是稳定 presentable $\infty$-category，局部化 $L:C\to C$ 称为 exact localization，若它是正合函子，即保持有限极限和有限余极限。

**命题 23.22.** 稳定 presentable $\infty$-category 中，保持有限余极限的局部化自动保持有限极限；因此 exact localization 可用保持有限余极限判别。

**证明.** 稳定 $\infty$-范畴中有限极限和有限余极限可由零对象、纤维、余纤维和悬挂互相表达。若 $L$ 保持有限余极限，则它保持零对象和余纤维。由稳定性，纤维可写为某个余纤维的环路或等价的拉回-推出方块；$L$ 与悬挂相容，因此保持纤维。故 $L$ 保持有限极限。$\square$

## 23.6 $\operatorname{Pr}^L$ 与 $\operatorname{Pr}^R$

**定义 23.23.** $\operatorname{Pr}^L$ 表示 presentable $\infty$-categories 与保持小余极限的函子组成的 $\infty$-范畴。$\operatorname{Pr}^R$ 表示 presentable $\infty$-categories 与右伴随函子组成的 $\infty$-范畴。

由外部输入定理 23.11，$\operatorname{Pr}^L$ 中的态射正是左伴随，$\operatorname{Pr}^R$ 中的态射正是右伴随。

**外部输入定理 23.24.** 取右伴随给出等价

$$
(\operatorname{Pr}^L)^{op}\simeq\operatorname{Pr}^R.
$$

此外，$\operatorname{Pr}^L$ 有闭对称幺半结构，张量积由保持余极限的双变量函子的泛性质刻画：

$$
\operatorname{Fun}^L(C\otimes D,E)
\simeq
\operatorname{Fun}^{L,L}(C\times D,E).
$$

该结构是高阶代数中“presentable 幺半 $\infty$-categories”的背景。

**例子 23.25.** 若 $C$ 是 presentable 幺半 $\infty$-category 且张量积分别保持余极限，则对 $E_1$-代数 $A\in C$，模范畴 $\operatorname{LMod}_A(C)$ 仍是 presentable。该结论是第二十二章模 $\infty$-范畴存在性定理的环境。

## 23.7 局部等价与反射子范畴

**定义 23.26.** 设 $L:C\to C$ 为局部化。态射 $f:X\to Y$ 称为 $L$-等价，若

$$
Lf:LX\to LY
$$

是等价。

**命题 23.27（局部等价的映射空间判别）.** 态射 $f:X\to Y$ 是 $L$-等价，当且仅当对每个 $L$-局部对象 $Z$，预复合诱导

$$
\operatorname{Map}_C(Y,Z)\to\operatorname{Map}_C(X,Z)
$$

是等价。

**证明.** 因为 $Z$ 局部，反射伴随给出自然等价

$$
\operatorname{Map}_C(Y,Z)\simeq\operatorname{Map}_C(LY,Z),
\qquad
\operatorname{Map}_C(X,Z)\simeq\operatorname{Map}_C(LX,Z).
$$

在这些等价下，预复合 $f$ 对应预复合 $Lf$。若 $Lf$ 是等价，则所有映射空间上的预复合都是等价。反过来，取 $Z=LY$ 与 $Z=LX$，上述条件说明 $Lf$ 在局部对象全子范畴中由 Yoneda 判别为等价。$\square$

**命题 23.28.** 若 $C$ 有小极限，则局部对象全子范畴 $C_L$ 的小极限由 $C$ 中的小极限计算。

**证明.** 设 $D:K\to C_L$ 为图形，令 $M=\lim_K D$ 在 $C$ 中计算。对任意对象 $X\in C$，

$$
\operatorname{Map}_C(X,M)
\simeq
\lim_{k\in K}\operatorname{Map}_C(X,D_k).
$$

由于每个 $D_k$ 局部，

$$
\operatorname{Map}_C(X,D_k)\simeq\operatorname{Map}_C(LX,D_k).
$$

因此

$$
\operatorname{Map}_C(X,M)
\simeq
\operatorname{Map}_C(LX,M),
$$

这正是命题 23.14 中反射伴随给出的局部对象判别，因此 $M$ 局部。于是 $M$ 是 $C_L$ 中的极限。$\square$

**推论 23.29.** Accessible localization 的包含函子 $C_L\hookrightarrow C$ 保持并创建小极限；局部化函子 $L:C\to C_L$ 作为左伴随保持小余极限。

**证明.** 前半由命题 23.28；后半是左伴随保持余极限。$\square$

## 23.8 预层、生成与可达局部化

Presentable $\infty$-categories 是现代高阶范畴论中处理“大”同伦范畴的标准框架。预层 $\infty$-范畴是自由生成对象；accessible localization 把预层范畴裁剪成 sheaves、local objects、稳定局部化和模范畴；$\operatorname{Pr}^L$ 则把这些大范畴及其左伴随组织成高阶代数可操作的环境。

## 练习

**练习 23.1.** 用 $\infty$-Yoneda 证明 $j:C\to\mathcal P(C)$ 全忠实。

**练习 23.2.** 证明 $\mathcal P(C)$ 的终对象逐点为 $\mathcal S$ 的终对象。

**练习 23.3.** 解释为什么可表预层 $j(c)$ 是紧对象。

**练习 23.4.** 比较第十二章的 $\kappa$-紧对象与定义 23.4。

**练习 23.5.** 写出 $\operatorname{Ind}_\kappa(C)$ 在 $\mathcal P(C)$ 中的生成方式。

**练习 23.6.** 说明 $\mathcal S$ 为什么是 presentable $\infty$-category。

**练习 23.7.** 用定理 23.11 判断保持小余极限的函子 $F:C\to D$ 是否有右伴随。

**练习 23.8.** 对局部化 $L$，证明若 $X$ 局部，则 $LX$ 局部。

**练习 23.9.** 写出 $S$-局部对象的定义，并说明它是映射空间条件。

**练习 23.10.** 解释 sheaf 化为什么可看成 Bousfield localization。

**练习 23.11.** 说明 left exact localization 与 $\infty$-topos 的关系。

**练习 23.12.** 在稳定 $\infty$-范畴中，为什么有限余极限能控制有限极限？

**练习 23.13.** 写出 $\operatorname{Pr}^L$ 和 $\operatorname{Pr}^R$ 的对象与态射。

**练习 23.14.** 解释 $(\operatorname{Pr}^L)^{op}\simeq\operatorname{Pr}^R$ 的含义。

**练习 23.15.** 说明 $\operatorname{Pr}^L$ 的张量积泛性质。

**练习 23.16.** 解释为什么第二十二章中要求张量积分别保持余极限。

**练习 23.17.** 证明若 $f$ 是等价，则它是任意局部化 $L$ 的 $L$-等价。

**练习 23.18.** 用命题 23.27 说明 $S$-局部化中的局部等价可由所有 $S$-局部对象检测。

**练习 23.19.** 证明两个局部对象的乘积仍是局部对象。
