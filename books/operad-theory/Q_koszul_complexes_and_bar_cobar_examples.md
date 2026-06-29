# 附录 Q：Koszul complex、bar-cobar 谱序列与计算样例

本附录补充定义 8.4--定义 8.16、定义 9.14--定理 9.20 和定义 I.11--命题 I.21 中仍偏抽象的部分：如何在低权重和低 arity 中看见 Koszul complex、bar-cobar differential 和谱序列。完整 Koszul 判别仍是外部输入；本附录只证明可直接由定义推出的计算。

## Q.1 二次 operad 的权重分解

设 $k$ 为域，链复形采用同调分次。设 $E$ 是集中在 arity $2$ 的对称序列，$\mathbb F(E)$ 是自由 operad。按顶点数定义权重：
$$
\mathbb F(E)=\bigoplus_{r\ge0}\mathbb F^{(r)}(E).
$$
其中
$$
\mathbb F^{(0)}(E)=I,\qquad
\mathbb F^{(1)}(E)=E.
$$

**命题 Q.1.** 若 $E$ 集中在 arity $2$，则 $\mathbb F^{(r)}(E)$ 由有 $r$ 个二元顶点的树给出，故只在 arity $r+1$ 非零。

**证明.** 每个顶点有两个输入。设树有 $r$ 个内部顶点、$l$ 个叶。对有根树，边数计数给出
$$
\sum_{v\in V(T)}\operatorname{in}(v)=l+r-1.
$$
左侧等于 $2r$，故 $l=r+1$。自由 operad 的权重 $r$ 部分正由 $r$ 个生成元装饰的树组成，所以只在 arity $r+1$ 出现。$\square$

**定义 Q.2.** 一个二次 operad 写作
$$
\mathcal P=\mathcal P(E,R)=\mathbb F(E)/(R),
$$
其中
$$
R\subset \mathbb F^{(2)}(E).
$$
权重 $2$ 是关系所在层，对应有两个二元顶点、三个叶的树。

## Q.2 权重 $2$ 的关系空间

在非对称情形中，若 $E=k\cdot\mu$ 由一个二元生成元生成，则
$$
\mathbb F^{(2)}(E)(3)
=
k\{\mu\circ_1\mu,\ \mu\circ_2\mu\}.
$$

**定义 Q.3.** 非对称 associative operad $\operatorname{Ass}_{ns}$ 是
$$
\operatorname{Ass}_{ns}=\mathbb F_{ns}(\mu)/(r),
$$
其中
$$
r=\mu\circ_1\mu-\mu\circ_2\mu.
$$

**命题 Q.4.** $\operatorname{Ass}_{ns}$-代数正是非含单位结合代数。

**证明.** 一个 $\mathbb F_{ns}(\mu)$-代数等价于给定一个二元运算
$$
m:A\otimes A\to A.
$$
关系 $r=0$ 在 endomorphism operad 中的像为
$$
m(m(a,b),c)-m(a,m(b,c)).
$$
因此商 operad 的代数等价于满足结合律的二元运算，即非含单位结合代数。$\square$

**说明 Q.5.** 含单位结合代数需要 arity $0$ 单位或非齐次关系；二次 reduced Koszul 理论先处理非含单位核心，再通过 augmented/unital 版本恢复单位。

## Q.3 二次对偶的低阶形状

设 $E^\vee$ 表示线性对偶并包含 operadic suspension 所要求的符号扭转。二次对偶 cooperad $\mathcal P^¡$ 的 cogenerators 来自 $sE^\vee$，corelations 来自 $s^2R^\perp$。

**命题 Q.6.** 在二元二次情形，$\mathcal P^¡$ 的权重 $0,1,2$ 部分满足：
$$
(\mathcal P^¡)^{(0)}=I,\qquad
(\mathcal P^¡)^{(1)}=sE^\vee,
$$
而权重 $2$ 部分是 $\mathbb T^c(sE^\vee)^{(2)}$ 中由 $R^\perp$ 选出的子商，具体方向依赖采用 cooperad 还是 cooperad 子对象模型。

**证明.** 二次对偶 cooperad 按定义由 cogenerators $sE^\vee$ 与 corelations $s^2R^\perp$ 生成。权重 $0$ 是 coaugmentation 单位，权重 $1$ 是 cogenerators。权重 $2$ 正是第一次出现 corelations 的层。$\square$

**警告 Q.7.** 文献中 $\mathcal P^!$、$\mathcal P^¡$、$\mathcal P^{\ash}$ 的 suspension 和 dual convention 不同。本书把 operad 对偶写作 $\mathcal P^!$，cooperad 对偶写作 $\mathcal P^¡$；进入具体符号时必须回到定义 E.11 和定义 I.11--定义 I.18。

## Q.4 Koszul twisting morphism 的低权重行为

设 $\kappa:\mathcal P^¡\to\mathcal P$ 为 Koszul twisting morphism。

**定义 Q.8.** $\kappa$ 在权重 $1$ 上由 cogenerator 与 generator 的配对给出：
$$
sE^\vee\longrightarrow E.
$$
在权重 $0$ 和权重 $\ge2$ 上为零。

**命题 Q.9.** 右 Koszul complex
$$
K_r(\mathcal P)=\mathcal P^¡\circ_\kappa\mathcal P
$$
的 twisting differential 只作用于 $\mathcal P^¡$ 中被 infinitesimal decomposition 分出的一个权重 $1$ cogenerator。

**证明.** Twisting differential 的定义为
$$
\mathcal P^¡
\xrightarrow{\Delta_{(1)}}
\mathcal P^¡\circ_{(1)}\mathcal P^¡
\xrightarrow{\operatorname{id}\circ_{(1)}\kappa}
\mathcal P^¡\circ_{(1)}\mathcal P
\to
\mathcal P^¡\circ\mathcal P.
$$
由于 $\kappa$ 在权重 $1$ 外为零，只有 infinitesimal decomposition 中内层因子权重为 $1$ 的项存活。$\square$

**推论 Q.10.** $K_r(\mathcal P)$ 的 differential 降低 cooperad 权重 $1$，并把该权重转移为右侧 operad 的一个复合操作。

## Q.5 非对称 Ass 的 Koszul complex 形状

本节只讨论形状，不证明完整 exactness。令 $\operatorname{Ass}_{ns}$ 为定义 Q.3 的非对称 associative operad。它的 Koszul dual cooperad 在非对称 reduced convention 下与 coassociative cooperad 对应，记为 $\operatorname{coAss}_{ns}$。

**外部输入定理 Q.11.** $\operatorname{Ass}_{ns}$ 是 Koszul；等价地，
$$
\operatorname{coAss}_{ns}\circ_\kappa\operatorname{Ass}_{ns}\to I
$$
是 quasi-isomorphism。

**低阶形状 Q.12.** 在 arity $1$，
$$
K_r(\operatorname{Ass}_{ns})(1)\cong k
$$
集中在单位层。

在 arity $2$，只有一个二元生成层，complex 的非单位部分由
$$
s\mu^\vee\otimes \mathbf 1
\quad\text{和}\quad
\mathbf 1\otimes\mu
$$
类型的项组成；twisting differential 把前者送到后者，符号由 suspension convention 决定。

在 arity $3$，树形项对应两种括号：
$$
(\mu\circ_1\mu),\qquad(\mu\circ_2\mu).
$$
Koszul differential 的边界正检测二者在 $\operatorname{Ass}_{ns}$ 中被关系
$$
\mu\circ_1\mu-\mu\circ_2\mu
$$
识别。

**说明 Q.13.** 上述描述解释 Koszul complex 如何“解析单位”：arity $>1$ 的同调应消失，arity $1$ 保留单位。但这种消失不是由低阶形状自动推出，而是 Ass Koszul 性的内容。

## Q.6 Bar construction 的低权重 differential

设 $\mathcal P$ 是 augmented dg-operad。Bar construction
$$
B\mathcal P=\mathbb T^c(s\overline{\mathcal P})
$$
的 differential 为
$$
d=d_{\mathrm{int}}+d_{\mathrm{bar}}.
$$

**命题 Q.14.** $d_{\mathrm{bar}}$ 在二顶点树上由 operad composition 给出：
$$
s p\ \circ_i\ s q
\longmapsto
\pm s(p\circ_i q).
$$

**证明.** Bar differential 的二次部分由收缩一条内部边定义。二顶点树只有一条内部边；收缩该边正是把两个顶点装饰按对应 slot 作 operad partial composition。悬挂因子移过张量因子产生符号。$\square$

**命题 Q.15.** 在三顶点树上，$d_{\mathrm{bar}}^2=0$ 等价于 operad partial composition 的结合律加 Koszul 符号抵消。

**证明.** 对三顶点树连续收缩两条内部边有两种顺序。若两条边嵌套，两个顺序对应 operad 的嵌套结合律；若两条边分离，两个顺序对应交换两个收缩操作并产生 Koszul 反号。每个最终一顶点树项出现两次且符号相反，故和为零。$\square$

## Q.7 Cobar construction 的低权重 differential

设 $\mathcal C$ 是 coaugmented conilpotent dg-cooperad。Cobar construction
$$
\Omega\mathcal C=\mathbb F(s^{-1}\overline{\mathcal C})
$$
的 differential 为
$$
d=d_{\mathrm{int}}+d_{\mathrm{cobar}}.
$$

**命题 Q.16.** $d_{\mathrm{cobar}}$ 在一个生成元 $s^{-1}c$ 上由 infinitesimal decomposition 给出：
$$
d_{\mathrm{cobar}}(s^{-1}c)
=
\sum \pm (s^{-1}c')\circ_i(s^{-1}c'')
$$
其中
$$
\Delta_{(1)}(c)=\sum c'\circ_i c''.
$$

**证明.** Cobar construction 的二次 differential 首先对 cooperad 元素作 infinitesimal decomposition，然后对每个分量 desuspend，并作为 derivation 延拓到自由 operad。公式正是该定义在 generator 上的写法。$\square$

**命题 Q.17.** $d_{\mathrm{cobar}}^2=0$ 的二次部分由 cooperad 余结合律给出。

**证明.** 对 $c$ 连续作两次 infinitesimal decomposition。两种分解顺序对应先分解外层再分解内层，或先分解内层再分解外层。Cooperad 余结合律识别这些分量；desuspension 和 derivation 符号使相同项成对抵消。$\square$

## Q.8 Bar-cobar counit 的低权重形式

设 $\mathcal P$ 是 augmented dg-operad。Bar-cobar counit
$$
\epsilon:\Omega B\mathcal P\to\mathcal P
$$
在 generators 上由
$$
s^{-1}s\overline{\mathcal P}\to\overline{\mathcal P}
$$
给出，在多顶点生成树上由 operad composition 递归给出。

**命题 Q.18.** $\epsilon$ 是 dg-operad morphism 的低权重检查归结为 bar differential 与 $\mathcal P$ 的 composition 相容。

**证明.** 在权重 $1$ 上，$\epsilon$ 是 desuspension-suspension 的抵消，与内部微分相容。权重 $2$ 上，$\Omega B\mathcal P$ 的 differential 包含把 bar 中二顶点树收缩为一顶点的项；$\epsilon$ 作用后得到 $\mathcal P$ 中的 partial composition。另一方面，先用 $\epsilon$ 把两个生成元送入 $\mathcal P$ 再在 $\mathcal P$ 中复合，得到同一元素。更高权重由 derivation 性和 operad 结合律递归推出。$\square$

**外部输入定理 Q.19.** 在适当 conilpotent/reduced 假设下，bar-cobar counit
$$
\Omega B\mathcal P\to\mathcal P
$$
是 quasi-isomorphism 或 cofibrant resolution。Fresse 的 modern entry 已定位为 FRE-4；代数层 quasi-free/cofibrant replacement 已定位为 FRE-5。若最终排版要求 Loday--Vallette/Fresse 书本中 $\Omega B\mathcal P\to\mathcal P$ 的单一定理编号，则属于书目 convention translation，不是正文证明缺口。

## Q.9 谱序列页面的使用边界

设 $C$ 是带递增滤过 $F_pC$ 的链复形，且 differential 满足
$$
d(F_pC)\subseteq F_pC.
$$
可形成谱序列 $E^r_{p,q}$。

**说明 Q.20.** 在 bar-cobar 证明中，常用权重滤过：

1. $d_{\mathrm{int}}$ 保持权重；
2. bar differential 降低权重；
3. cobar differential 增加权重；
4. twisted differential 改变 cooperad/operad 权重分配但保持总 arity。

因此引用谱序列时必须说明是递增还是递减滤过，以及 differential 在第几页出现。

**警告 Q.21.** “谱序列退化”不是一个无条件短语。必须说明：

1. 收敛到哪个 filtered homology；
2. 是否强收敛；
3. 是否有 boundedness 或 complete/exhaustive 条件；
4. 页码 $E^r$ 的 convention。

## Q.10 小结

本附录给出以下可检查内容：

1. 二元二次 operad 的权重 $r$ 只在 arity $r+1$ 出现；
2. 非对称 associative operad 的关系是 $\mu\circ_1\mu-\mu\circ_2\mu$；
3. Koszul twisting morphism 只在权重 $1$ 非零；
4. Koszul differential 通过 infinitesimal decomposition 检测关系；
5. bar differential 是收缩内部边；
6. cobar differential 是展开 cooperad 分解；
7. bar-cobar counit 的低权重检查是 composition 相容性；
8. 完整 exactness、Koszul 性和 resolution 结论仍是外部输入。

这些计算应作为定义 8.16、定理 9.20 和定义 10.5--定义 10.10 中同伦代数构造的局部校验模板。
