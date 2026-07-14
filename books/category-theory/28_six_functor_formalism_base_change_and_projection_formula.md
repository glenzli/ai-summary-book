# 第二十八章：六操作形式主义、基变换与投影公式

几何上的拉回、直接像、紧支撑直接像和异常拉回并非六个互不相关的函子；它们由伴随、基变换、投影公式、局部化与对偶性组成一套相干系统。六操作形式主义把每个几何对象 $X$ 关联到稳定闭幺半 $\infty$-范畴 $\mathcal D(X)$，并为态射 $f:X\to Y$ 提供

$$
f^*,\quad f_*,\quad f_!,\quad f^!,\quad -\otimes_X-,\quad \underline{\operatorname{Hom}}_X(-,-).
$$

核心不在单个函子的存在，而在这些函子之间的相干性。本章先抽象规定适用的态射类别，再从 mate correspondence 解释基变换，从闭结构推出投影公式的型，并把局部化三角与 Verdier 对偶放进同一框架。Sheaf、etale sheaf、motivic sheaf、$D$-module 和 quasi-coherent sheaf 版本只在来源明确的假设下作为实例。

本章汇合伴随、闭幺半结构、Kan 延拓、topos、Cartesian fibration、presentable/stable $\infty$-范畴、Beck--Chevalley 与 dg enhancement。每个公式都会声明方形的性质、态射是否 proper/smooth 及使用 $*$ 还是 $!$，避免把不同几何理论中的六操作无条件合并。

## 28.1 六操作资料的抽象形式

**定义 28.1.** 设 $\mathcal B$ 是带有有限拉回的 $\infty$-范畴，称为基范畴。一个稳定系数系统是一个反变函子

$$
\mathcal D:\mathcal B^{op}\to\operatorname{CAlg}(\operatorname{Pr}^L_{\operatorname{st}})
$$

其中 $\operatorname{Pr}^L_{\operatorname{st}}$ 表示稳定 presentable $\infty$-范畴和保持小余极限的正合函子组成的 $\infty$-范畴，$\operatorname{CAlg}$ 表示闭对称幺半结构中的交换代数对象。

对 $X\in\mathcal B$，写 $\mathcal D(X)$；对 $f:X\to Y$，写

$$
f^*:\mathcal D(Y)\to\mathcal D(X).
$$

**定义 28.2.** 若每个 $f^*$ 保持小余极限，则由 presentable 伴随函子定理有右伴随

$$
f^*:\mathcal D(Y)\rightleftarrows\mathcal D(X):f_*.
$$

称 $f_*$ 为直接像或普通推前。

**定义 28.3.** 一个六操作形式主义是在稳定系数系统上额外指定：

1. 对合适态射 $f:X\to Y$ 的左伴随
   $$
   f_!:\mathcal D(X)\rightleftarrows\mathcal D(Y):f^!;
   $$
2. 每个 $\mathcal D(X)$ 中的闭幺半结构
   $$
   -\otimes_X-,\qquad \underline{\operatorname{Hom}}_X(-,-);
   $$
3. 对 Cartesian 方块
   $$
   \begin{array}{c}
   X'\xrightarrow{g'}X\\
   \downarrow f'\quad\downarrow f\\
   Y'\xrightarrow{g}Y
   \end{array}
   $$
   的 Beck-Chevalley 等价；
4. 投影公式、单位相干、复合相干和局部化相干。

**注 28.4.** 这一章不把六操作当作一个已经自动存在的结构。具体几何语境中，$f_!$、$f^!$、基变换和投影公式的存在性是深定理，必须逐理论证明。

## 28.2 伴随与基本函子关系

**命题 28.5.** 若 $f^*$ 是闭对称幺半的左伴随，则它保持单位对象和张量积：

$$
f^*\mathbb 1_Y\simeq\mathbb 1_X,\qquad
f^*(A\otimes_YB)\simeq f^*A\otimes_Xf^*B.
$$

**证明.** 闭对称幺半函子按定义配有这些结构态射，并且在强幺半情形它们为等价。六操作形式主义要求 $f^*$ 为强对称幺半左伴随，因此得到两个自然等价。$\square$

**命题 28.6.** 设 $f^*\dashv f_*$。存在自然态射

$$
f_*\,\underline{\operatorname{Hom}}_X(f^*A,B)\to
\underline{\operatorname{Hom}}_Y(A,f_*B).
$$

若 $A$ dualizable 且投影公式对 $A^\vee$ 成立，则该态射为等价。

**证明.** 对任意 $C\in\mathcal D(Y)$，由闭结构和伴随有自然等价链

$$
\operatorname{Map}_Y(C,f_*\underline{\operatorname{Hom}}_X(f^*A,B))
\simeq
\operatorname{Map}_X(f^*C,\underline{\operatorname{Hom}}_X(f^*A,B))
$$

$$
\simeq
\operatorname{Map}_X(f^*C\otimes f^*A,B)
\simeq
\operatorname{Map}_X(f^*(C\otimes A),B)
\simeq
\operatorname{Map}_Y(C\otimes A,f_*B)
$$

$$
\simeq
\operatorname{Map}_Y(C,\underline{\operatorname{Hom}}_Y(A,f_*B)).
$$

Yoneda 给出所需自然态射；若 $A$ dualizable，内部 Hom 可写为 $A^\vee\otimes-$，再用投影公式把比较态射识别为等价。$\square$

## 28.3 基变换

**定义 28.7.** 对 Cartesian 方块

$$
\begin{array}{c}
X'\xrightarrow{g'}X\\
\downarrow f'\quad\downarrow f\\
Y'\xrightarrow{g}Y
\end{array}
$$

普通基变换态射是

$$
g^*f_*\to f'_*g'^*.
$$

非常基变换态射是

$$
g^*f_!\to f'_!g'^*.
$$

若相应态射为等价，则称该方块满足普通基变换或非常基变换。

**命题 28.8.** 普通基变换态射可由伴随单位和余单位自然构造。

**证明.** 从 $g^*f_*$ 出发，插入伴随单位 $\operatorname{id}\to f'_*f'^*$：

$$
g^*f_*\to f'_*f'^*g^*f_*.
$$

因方块 Cartesian，有相干等价 $f'^*g^*\simeq g'^*f^*$。于是得到

$$
g^*f_*\to f'_*g'^*f^*f_*.
$$

再用伴随余单位 $f^*f_*\to\operatorname{id}$，得到

$$
g^*f_*\to f'_*g'^*.
$$

自然性来自单位、余单位和伪函子相干。$\square$

**注 28.9.** 与命题 28.8 不同，仅有四个伴随和方块的伪函子相干并不足以强制产生非常基变换态射。一个六操作形式主义还需给出 exchange transformation

$$
g^*f_!\longrightarrow f'_!g'^*
$$

及其与复合、单位和 mate 对应的相容性。在具体 sheaf 理论中，该态射通常由紧化或支撑条件构造；它何时为等价是几何定理，而非纯形式伴随演算的结论。

**外部输入定理 28.10（两种标准基变换定理）.** 对局部紧、有限维并带有限分层的空间及其可构造 sheaves，若 $f$ proper，则命题 28.8 的比较给出

$$
g^*Rf_*\simeq Rf'_*g'^*.
$$

对 scheme 的 étale 导出范畴，若 $f$ proper 且系数为与各剩余特征互素的 torsion ring，则同一 proper base change 等价成立。非常推前 $f_!$、motivic sheaves 与 $D$-modules 各有相应基变换定理，但其可分离有限型、可构造性或特征假设必须按所用形式主义另行陈述；本章不把这些版本合并为一个无条件定理。

## 28.4 投影公式

**定义 28.11.** 对 $f:X\to Y$，投影公式是自然态射

$$
f_!(A\otimes_X f^*B)\to f_!A\otimes_Y B
$$

为等价。普通推前版本为

$$
f_*(A\otimes_X f^*B)\to f_*A\otimes_Y B.
$$

**命题 28.12.** 若 $f_!$ 是 $\mathcal D(Y)$-线性函子，其中 $\mathcal D(X)$ 通过 $f^*$ 成为 $\mathcal D(Y)$-模范畴，则投影公式成立。

**证明.** $\mathcal D(Y)$-线性意味着对 $A\in\mathcal D(X)$ 和 $B\in\mathcal D(Y)$，结构等价

$$
f_!(A\otimes_X f^*B)\simeq f_!A\otimes_YB
$$

是该模函子结构的一部分。该等价相容于 $\mathcal D(Y)$ 的张量结合律和单位律，因此正是投影公式。$\square$

**命题 28.13.** 若投影公式对 $f$ 和 $g$ 成立，则在复合 $gf$ 的六操作相干下，投影公式对 $gf$ 成立。

**证明.** 设 $f:X\to Y$、$g:Y\to Z$。对 $A\in\mathcal D(X)$ 和 $C\in\mathcal D(Z)$，

$$
(gf)_!(A\otimes f^*g^*C)\simeq
g_!f_!(A\otimes f^*g^*C).
$$

对 $f$ 用投影公式得

$$
g_!(f_!A\otimes g^*C).
$$

再对 $g$ 用投影公式得

$$
g_!f_!A\otimes C\simeq (gf)_!A\otimes C.
$$

这些等价由六操作复合相干保证自然。$\square$

## 28.5 Proper pushforward、open extension 与局部化

**定义 28.14.** 若基范畴中指定一类 proper morphisms，六操作形式主义通常要求 proper $f$ 满足自然等价

$$
f_!\simeq f_*.
$$

称为 proper pushforward compatibility。

**命题 28.15.** 若 $f$ proper 且非常基变换对 $f_!$ 成立，则普通基变换对 $f_*$ 成立。

**证明.** 由 proper compatibility，$f_!\simeq f_*$；拉回后的 $f'$ 仍 proper 时也有 $f'_!\simeq f'_*$。非常基变换等价

$$
g^*f_!\simeq f'_!g'^*
$$

在这些识别下即为

$$
g^*f_*\simeq f'_*g'^*.
$$

$\square$

**定义 28.16.** 对开嵌入 $j:U\hookrightarrow X$，$j_!$ 称为 extension by zero。对闭嵌入 $i:Z\hookrightarrow X$，$i_*$ 通常是闭支撑推前。

**外部输入定理 28.17（Recollement 局部化）.** 在经典 sheaf 型六操作理论中，若 $j:U\hookrightarrow X$ 是开嵌入，$i:Z\hookrightarrow X$ 是闭补，则存在自然余纤维序列

$$
j_!j^*K\to K\to i_*i^*K
$$

以及对偶形式

$$
i_*i^!K\to K\to j_*j^*K.
$$

这些序列组织为 recollement 结构。

**命题 28.18.** 若 recollement 序列成立，则 $K\simeq0$ 当且仅当 $j^*K\simeq0$ 且 $i^*K\simeq0$。

**证明.** 若 $K\simeq0$，则 $j^*K\simeq0$ 且 $i^*K\simeq0$，因为函子保持零对象。反过来，若 $j^*K\simeq0$ 且 $i^*K\simeq0$，则余纤维序列

$$
j_!j^*K\to K\to i_*i^*K
$$

两端为零，因此中间项 $K$ 为零。$\square$

## 28.6 Verdier 对偶与 exceptional pullback

**定义 28.19.** 若 $p_X:X\to *$ 是终对象上的结构态射，定义 dualizing object

$$
\omega_X=p_X^!\mathbb 1_*.
$$

Verdier duality functor 为

$$
\mathbb D_X(K)=\underline{\operatorname{Hom}}_X(K,\omega_X).
$$

**命题 28.20.** 若 $K$ 在 $\mathcal D(X)$ 中 dualizable，则自然映射

$$
\mathbb D_X(K)\otimes_X L\to\underline{\operatorname{Hom}}_X(K,L\otimes_X\omega_X)
$$

在 $L=\mathbb 1_X$ 时给出 $\mathbb D_X(K)\simeq K^\vee\otimes_X\omega_X$。

**证明.** $K$ dualizable 意味着

$$
\underline{\operatorname{Hom}}_X(K,M)\simeq K^\vee\otimes_XM
$$

对所有 $M$ 自然成立。取 $M=\omega_X$ 得

$$
\mathbb D_X(K)=\underline{\operatorname{Hom}}_X(K,\omega_X)\simeq K^\vee\otimes_X\omega_X.
$$

更一般的映射由闭结构的张量-Hom 伴随给出。$\square$

**外部输入定理 28.21（Verdier 对偶公式）.** 在合适可构造或紧性假设下，有自然等价

$$
f^!\mathbb D_Y(K)\simeq \mathbb D_X(f^*K),
$$

并且 proper $f$ 满足

$$
\mathbb D_Y f_*K\simeq f_*\mathbb D_XK.
$$

光滑或局部 complete intersection 态射还具有 purity 公式，把 $f^!$ 表示为 $f^*$ 后张量相对 dualizing object 并平移。

## 28.7 六操作与 equipment 的关系

**命题 28.22.** 六操作中的基变换方块可视为 equipment 中 exact square 的高阶版本。

**证明.** 在第二十五章中，equipment 的 exact square 表示 companion/conjoint 沿方块复合时给出的比较 $2$-态射为同构。六操作形式主义把对象 $X$ 送到范畴 $\mathcal D(X)$，把态射 $f$ 送到伴随对 $f^*\dashv f_*$ 或 $f_!\dashv f^!$。Cartesian 方块上的比较态射

$$
g^*f_*\to f'_*g'^*,\qquad g^*f_!\to f'_!g'^*
$$

正是“沿两条路径重索引并推前”之间的高阶 Beck-Chevalley 比较。因此 exact square 的思想在稳定 presentable $\infty$-范畴值语境中表现为基变换等价。$\square$

**注 28.23.** 这解释了为什么本书先建立 profunctor、equipment 和 indexed categories，再进入六操作。六操作不是孤立的几何技术，而是范畴化的 base change 与 adjunction calculus。

## 28.8 形式相干的低维检查

**命题 28.24.** 对恒等态射 $\operatorname{id}_X:X\to X$，六操作资料中的

$$
\operatorname{id}_X^*,\quad \operatorname{id}_{X*},\quad \operatorname{id}_{X!},\quad \operatorname{id}_X^!
$$

都与恒等函子相容；相应基变换和投影公式退化为恒等比较。

**证明.** 稳定系数系统是反变函子，因此恒等态射被送到恒等函子。若 $f_!,f^!$ 作为额外六操作资料与复合相干，则恒等态射的左、右 adjoint 也由恒等函子给出。基变换方块若全为恒等，命题 28.8 的单位-余单位复合就是伴随三角恒等式，故为恒等；投影公式则为

$$
A\otimes_XB\simeq A\otimes_XB.
$$

$\square$

**命题 28.25.** 若 $f:X\to Y$ 与 $g:Y\to Z$ 都满足 proper compatibility，则复合 $gf$ 也满足 proper compatibility。

**证明.** Proper compatibility 给出 $f_!\simeq f_*$ 与 $g_!\simeq g_*$. 由六操作的复合相干，

$$
(gf)_!\simeq g_!f_!,\qquad (gf)_*\simeq g_*f_*.
$$

代入前两个等价得到

$$
(gf)_!\simeq g_!f_!\simeq g_*f_*\simeq(gf)_*.
$$

$\square$

**命题 28.26.** 若 recollement 序列

$$
j_!j^*K\to K\to i_*i^*K
$$

成立，则 $\ker(j^*)$ 由闭部分的本质像生成：若 $j^*K\simeq0$，则 $K\simeq i_*i^*K$。

**证明.** 在该余纤维序列中，若 $j^*K\simeq0$，则第一项 $j_!j^*K\simeq0$。因此余纤维序列退化为

$$
0\to K\to i_*i^*K,
$$

在稳定范畴中推出 $K\simeq i_*i^*K$。$\square$

## 28.9 六操作的相干骨架

六操作形式主义由稳定闭幺半 presentable $\infty$-范畴的参数化系统、伴随对、基变换、投影公式、proper compatibility、recollement 和 Verdier 对偶构成。它把 topos 和 sheaf 理论中的局部-整体原则，与稳定同伦论、Morita 理论和高阶 base change 相连。具体几何理论中的六操作存在性是大型外部输入；本章提供的是可检查的抽象结构和相干关系。

## 练习

**练习 28.1.** 定义稳定系数系统 $\mathcal D:\mathcal B^{op}\to\operatorname{CAlg}(\operatorname{Pr}^L_{\operatorname{st}})$。

**练习 28.2.** 解释为什么 $f^*$ 保持小余极限时有右伴随 $f_*$。

**练习 28.3.** 列出六操作中的六个操作。

**练习 28.4.** 证明强对称幺半的 $f^*$ 保持单位对象和张量积。

**练习 28.5.** 构造普通基变换态射 $g^*f_*\to f'_*g'^*$。

**练习 28.6.** 构造非常基变换态射 $g^*f_!\to f'_!g'^*$ 的伴随描述。

**练习 28.7.** 写出投影公式。

**练习 28.8.** 证明若 $f_!$ 是 $\mathcal D(Y)$-线性的，则投影公式成立。

**练习 28.9.** 证明投影公式对复合封闭。

**练习 28.10.** 说明 proper $f$ 下 $f_!\simeq f_*$ 如何把非常基变换转为普通基变换。

**练习 28.11.** 对开闭分解 $U\hookrightarrow X\hookleftarrow Z$，写出 recollement 的两个余纤维序列。

**练习 28.12.** 用 recollement 证明 $j^*$ 和 $i^*$ 联合保守。

**练习 28.13.** 定义 dualizing object 和 Verdier duality functor。

**练习 28.14.** 若 $K$ dualizable，证明 $\mathbb D_X(K)\simeq K^\vee\otimes\omega_X$。

**练习 28.15.** 解释六操作基变换与 equipment 中 Beck-Chevalley 条件的关系。

**练习 28.16.** 对恒等态射验证投影公式退化为恒等同构。

**练习 28.17.** 证明 proper compatibility 对复合封闭。

**练习 28.18.** 若 $j^*K=0$，用 recollement 证明 $K\simeq i_*i^*K$。
