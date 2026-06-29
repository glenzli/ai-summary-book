# 第十四章：站点、sheaf 与 Grothendieck topos

## 本章目标

本章定义 Grothendieck 拓扑、站点、sheaf、sheaf 化和 Grothendieck topos，并给出 Giraud 定理作为外部输入。

## 依赖前置知识

需要预层、极限、余极限、可表函子、正合范畴和局部化思想。

## 14.1 Grothendieck 拓扑

**定义 14.1.** 设 $\mathcal C$ 为小范畴。Grothendieck 拓扑 $J$ 给每个对象 $U$ 指定一类筛（sieves）$S\subseteq\mathcal C/U$，称为覆盖筛，满足：

1. 最大筛 $\mathcal C/U$ 覆盖 $U$；
2. 若 $S$ 覆盖 $U$ 且 $f:V\to U$，则拉回筛 $f^*S$ 覆盖 $V$；
3. 若 $S$ 覆盖 $U$，且 $R$ 是 $U$ 上筛，满足对每个 $f:V\to U$ 属于 $S$，$f^*R$ 覆盖 $V$，则 $R$ 覆盖 $U$。

二元组 $(\mathcal C,J)$ 称为站点（site）。

## 14.2 sheaf 条件

**定义 14.2.** 预层 $F:\mathcal C^{\operatorname{op}}\to\mathbf{Set}$ 称为 sheaf，若对每个对象 $U$ 和每个覆盖筛 $S$，限制映射

$$
F(U)\to\operatorname{Nat}(S,F)
$$

是双射。这里 $S$ 被视为 $yU$ 的子预层。

**命题 14.3.** 若拓扑由覆盖族 $\{U_i\to U\}$ 生成，则 sheaf 条件可写成等化子

$$
F(U)\to\prod_iF(U_i)
\rightrightarrows
\prod_{i,j}F(U_i\times_U U_j)
$$

在相应拉回存在且覆盖族形式足够稳定时成立。

**证明.** 令 $S$ 为覆盖族 $\{U_i\to U\}$ 生成的筛。一个自然变换 $S\to F$ 等价于给每个箭头 $V\to U$ 属于 $S$ 指定元素 $F(V)$，并且对 $S$ 中态射满足限制相容。若覆盖族由拉回稳定的基生成，则这样的数据由 $s_i\in F(U_i)$ 决定，并且相容性正是

$$
s_i|_{U_i\times_U U_j}=s_j|_{U_i\times_U U_j}
$$

对所有 $i,j$ 成立。因此 $\operatorname{Nat}(S,F)$ 可识别为等化子

$$
\prod_iF(U_i)
\rightrightarrows
\prod_{i,j}F(U_i\times_U U_j).
$$

sheaf 条件要求

$$
F(U)\to\operatorname{Nat}(S,F)
$$

是双射。把上述识别代入，就得到命题中的等化子条件。反过来，若所有基覆盖族满足该等化子条件，则对由这些覆盖族生成的覆盖筛，匹配族可逐级由基覆盖粘合并由唯一性保证与细化无关，故筛版本 sheaf 条件成立。$\square$

## 14.3 subcanonical 拓扑

**定义 14.4.** 若每个可表预层 $yU$ 都是 sheaf，则称拓扑 $J$ 是 subcanonical。

**例子 14.5.** 拓扑空间开集范畴上的通常开覆盖拓扑是 subcanonical；可表预层由开集 $V$ 给出 $U\mapsto\operatorname{Hom}(U,V)$，sheaf 条件反映连续映射可由开覆盖上的相容映射唯一粘合。

## 14.4 sheaf 化

**定义 14.6.** 预层 $F$ 称为 separated，若对每个覆盖筛 $S$，映射

$$
F(U)\to\operatorname{Nat}(S,F)
$$

是单射。也就是说，截面若局部相等，则全局相等。sheaf 条件是在 separated 的基础上再要求每个匹配族存在全局粘合。

**构造 14.7（plus 构造）.** 对预层 $F$，定义预层 $F^+$：

$$
F^+(U)=\operatorname*{colim}_{S\in J(U)}\operatorname{Nat}(S,F),
$$

其中 $J(U)$ 是 $U$ 上覆盖筛按包含排序所得范畴。元素可理解为某个覆盖上的匹配族，两个匹配族在共同细化后相等则被识别。

**外部输入命题 14.8.** plus 构造把预层送到 separated 预层；若 $F$ 已 separated，则 $F^+$ 是 sheaf。在许多标准站点口径下，sheaf 化可由

$$
aF\simeq F^{++}
$$

给出。证明的技术核心是检查覆盖筛的细化、传递性和匹配族粘合。

**外部输入定理 14.9（sheaf 化）.** 对任意小站点 $(\mathcal C,J)$，包含函子

$$
i:\operatorname{Sh}(\mathcal C,J)\hookrightarrow\widehat{\mathcal C}
$$

有左伴随

$$
a:\widehat{\mathcal C}\to\operatorname{Sh}(\mathcal C,J),
$$

称为 sheaf 化。并且 $a$ 保持有限极限。

该定理证明需要 plus construction 或局部对象反射理论，本书在此作为输入。

## 14.5 Grothendieck topos

**定义 14.10.** Grothendieck topos 是与某个小站点的 sheaf 范畴

$$
\operatorname{Sh}(\mathcal C,J)
$$

等价的范畴。

**命题 14.11.** 每个预层范畴 $\widehat{\mathcal C}$ 是 Grothendieck topos。

**证明.** 取 $\mathcal C$ 上平凡拓扑，即只有最大筛覆盖。此时每个预层自动满足 sheaf 条件，故

$$
\operatorname{Sh}(\mathcal C,J_{\mathrm{triv}})=\widehat{\mathcal C}.
$$

$\square$

**定义 14.12.** Grothendieck topoi 之间的几何态射

$$
f:\mathcal E\to\mathcal F
$$

由一对伴随函子

$$
f^*:\mathcal F\rightleftarrows\mathcal E:f_*
$$

组成，其中 $f^*$ 左伴随于 $f_*$，并且 $f^*$ 保持有限极限。$f^*$ 称为 inverse image，$f_*$ 称为 direct image。

**例子 14.13.** sheaf 化伴随

$$
a:\widehat{\mathcal C}\rightleftarrows\operatorname{Sh}(\mathcal C,J):i
$$

给出几何态射

$$
\operatorname{Sh}(\mathcal C,J)\to\widehat{\mathcal C}
$$

的 inverse image 部分 $a$，因为定理 14.9 说明 $a$ 左正合。

**外部输入定理 14.14（Giraud 定理）.** 一个范畴是 Grothendieck topos，当且仅当它满足一组内在公理：有小余极限、有限极限、余极限与拉回有适当相容性、等价关系有效，并有小生成族等。

本书后续只使用该定理的方向性解释；完整陈述和证明见 Johnstone、Mac Lane-Moerdijk 和 SGA 4。

## 14.6 极限、反射与几何态射

**命题 14.15.** 包含函子

$$
i:\operatorname{Sh}(\mathcal C,J)\hookrightarrow\widehat{\mathcal C}
$$

创建所有小极限。也就是说，sheaf 的图形若在预层范畴中取极限，则所得预层仍是 sheaf，且它就是 sheaf 范畴中的极限。

**证明.** 设 $F_\alpha$ 是 sheaf 图形，并令 $F=\lim_\alpha iF_\alpha$ 为预层范畴中的极限。对覆盖筛 $S$，有

$$
F(U)\cong\lim_\alpha F_\alpha(U),
$$

且自然变换集合满足

$$
\operatorname{Nat}(S,F)\cong
\operatorname{Nat}\left(S,\lim_\alpha F_\alpha\right)
\cong
\lim_\alpha\operatorname{Nat}(S,F_\alpha),
$$

因为 $\operatorname{Nat}(S,-)$ 是函子范畴中的 Hom 函子，保持极限。每个 $F_\alpha$ 是 sheaf，故 $F_\alpha(U)\to\operatorname{Nat}(S,F_\alpha)$ 是双射。取极限后得到

$$
F(U)\cong\operatorname{Nat}(S,F).
$$

所以 $F$ 是 sheaf。极限锥的泛性质在预层范畴中成立，而包含函子全忠实，因此同一锥也给出 sheaf 范畴中的极限。$\square$

**命题 14.16（sheaf 化的反射泛性质）.** 对任意预层 $F$ 和 sheaf $G$，sheaf 化伴随给出自然双射

$$
\operatorname{Sh}(\mathcal C,J)(aF,G)
\cong
\widehat{\mathcal C}(F,iG).
$$

因此从 $F$ 到 sheaf 的任意态射唯一经过单位 $F\to iaF$ 分解。

**证明.** 这是定理 14.9 中伴随 $a\dashv i$ 的 Hom 形式。由于 $i$ 是全忠实，单位 $F\to iaF$ 正是把 $F$ 反射到 sheaf 子范畴的普遍箭头。$\square$

**命题 14.17.** Grothendieck topoi 与几何态射组成范畴。

**证明.** 恒等几何态射由恒等伴随给出，其 inverse image 是恒等函子，显然保持有限极限。若

$$
f:\mathcal E\to\mathcal F,\qquad g:\mathcal F\to\mathcal G
$$

为几何态射，则复合的 inverse image 定义为

$$
(g f)^*=f^*\circ g^*:\mathcal G\to\mathcal E.
$$

它是左伴随的复合，其右伴随为 $g_*\circ f_*$。并且有限极限由 $g^*$ 保持后再由 $f^*$ 保持，所以 $(g f)^*$ 左正合。结合律和单位律来自函子复合的结合律和单位律。$\square$

## 14.7 本章小结

站点把覆盖关系从拓扑空间推广到一般范畴。sheaf 是满足覆盖粘合条件的预层。Grothendieck topos 是 sheaf 范畴，是集合范畴的广义版本，也是几何、逻辑和高阶 topos 的入口。Sheaf 子范畴是预层范畴的左正合反射子范畴；其极限由预层极限创建，几何态射则用左正合 inverse image 组织 topoi 之间的结构保持映射。

## 练习

**练习 14.1.** 对拓扑空间 $X$ 的开集范畴，写出开覆盖产生的筛。

**练习 14.2.** 证明常值预层不一定是 sheaf。

**练习 14.3.** 在 subcanonical 站点上，说明 Yoneda 嵌入如何落入 sheaf 范畴。

**练习 14.4.** 证明 sheaf 范畴中的有限极限由预层有限极限创建，并指出定理 14.9 的作用。

**练习 14.5.** 查阅 Giraud 定理的完整公理列表，并与定义 14.10 比较。

**练习 14.6.** 解释 separated 预层与 sheaf 的区别。

**练习 14.7.** 用 plus 构造语言说明匹配族如何成为 $F^+(U)$ 的元素。

**练习 14.8.** 证明若 $F$ 是 sheaf，则 $F$ 一定 separated。

**练习 14.9.** 写出几何态射中 $f^*$ 与 $f_*$ 的伴随方向，并说明为何要求 $f^*$ 左正合。

**练习 14.10.** 对平凡拓扑，说明 plus 构造不改变预层。

**练习 14.11.** 证明 sheaf 范畴中的二元积由预层二元积逐点给出。

**练习 14.12.** 用命题 14.16 证明：若 $F$ 已经是 sheaf，则单位 $F\to iaF$ 是同构。

**练习 14.13.** 证明两个几何态射的复合仍满足 inverse image 左正合。
