# 附录 BF：Higher Groups、Deloopings 与 Classifying Types

本附录补入 HoTT 中群论和同伦论的结构性桥梁：群作为 loop space，delooping，classifying type，torsor，principal bundle 和自等价群。它连接第十一章基本群、第十二章合成同伦论、附录 Y 的上同调以及附录 AJ 的模态/截断。

## BF.1 Loop group

**定义 BF.1（pointed connected type）。** 一个 pointed type 是 $(B,b_0)$。若 $\|B\|_0$ 可收缩，则称 $B$ connected。

**定义 BF.2（loop group）。** 对 pointed type $(B,b_0)$，其 loop type
$$
\Omega B\coloneqq(b_0=b_0)
$$
带有由路径复合给出的群状结构：
$$
e\coloneqq\mathsf{refl}_{b_0},\qquad
p\cdot q,\qquad
p^{-1}.
$$
若 $\Omega B$ 是集合，则它是普通群。

**命题 BF.3（loop group laws，书内证明核）。** $\Omega B$ 的单位律、结合律和逆元律由路径代数给出。

**证明.** 单位律、结合律和逆元律分别是附录 A/D 中路径复合的标准引理；若 $\Omega B$ 是集合，则群律所在路径类型为命题，因而群结构相干唯一。$\square$

## BF.2 Delooping

**定义 BF.4（delooping）。** 群 $G$ 的 delooping 是 pointed connected type $(BG,\ast)$，配备群同构
$$
\Omega BG\cong G.
$$

**定义 BF.5（higher group）。** 一个 higher group 可定义为 pointed connected type $B$，其底层群状对象为 $\Omega B$。若 $\Omega B$ 为 $(n-1)$-type，则 $B$ 是 $n$-group 的 delooping。

**例 BF.6（圆的 delooping）。** 圆 $\mathbb S^1$ 是整数群 $\mathbb Z$ 的 delooping：
$$
\Omega(\mathbb S^1,\mathsf{base})\simeq\mathbb Z.
$$
群同构证明见附录 V。

## BF.3 Classifying type of automorphisms

**定义 BF.7（自等价群）。** 对类型 $A$，定义
$$
\mathsf{Aut}(A)\coloneqq(A\simeq A).
$$
它在复合下形成 higher group；若 $A$ 是集合且自等价类型为集合，则得到普通群。

**定义 BF.8（classifying type of $A$）。** 定义
$$
B\mathsf{Aut}(A)\coloneqq\sum_{X:\mathcal U}\|X=A\|_0
$$
或等价地取 $A$ 在 universe 中的 connected component。

**命题 BF.9（loop of classifying type，证明核）。** 在单值性下，
$$
\Omega(B\mathsf{Aut}(A),(A,|\mathsf{refl}|))\simeq\mathsf{Aut}(A).
$$

**证明.** $B\mathsf{Aut}(A)$ 的基点 loop 是保持截断见证的对象路径。由于第二分量是命题截断，其路径分量为命题且不贡献额外结构；第一分量给出 $A=A$。由单值性，
$$
(A=A)\simeq(A\simeq A).
$$
合成得结论。$\square$

## BF.4 Torsors

**定义 BF.10（右 $G$-作用）。** 对集合群 $G$ 和集合 $X$，右作用是函数
$$
\rho:X\times G\to X
$$
满足
$$
x\cdot e=x,\qquad (x\cdot g)\cdot h=x\cdot(gh).
$$

**定义 BF.11（$G$-torsor）。** $G$-torsor 是带右 $G$-作用的集合 $X$，满足：

1.  $X$ 仅仅 inhabited：$\|X\|$；
2.  对任意 $x,y:X$，类型
    $$
    \sum_{g:G}(x\cdot g=y)
    $$
    可收缩。

第二条表达作用自由且传递。

**命题 BF.12（标准 torsor）。** $G$ 以右乘作用于自身形成 $G$-torsor。

**证明.** inhabited 由单位元给出。给定 $x,y:G$，取 $g=x^{-1}y$，则 $xg=y$。唯一性由群消去律：若 $xg=y$ 且 $xh=y$，左乘 $x^{-1}$ 得 $g=h$。$\square$

## BF.5 Classifying type and torsors

**定理 BF.13（$BG$ classifies torsors，证明架构 / 外部输入）。** 对集合群 $G$ 的 delooping $BG$，有等价
$$
BG\simeq\mathsf{Tors}_G
$$
其中右侧为 $G$-torsor 类型的 connected component。

**证明架构.** 从点 $b:BG$ 取路径 torsor
$$
(\ast=b)
$$
并以 loop group $\Omega BG\cong G$ 作用。反向从 torsor $X$ 构造其 classifying point。自由传递性给出两方向互逆。完整证明需要 truncation、univalence 和作用相干。

**推论 BF.14（principal bundle classification，接口）。** 对类型 $X$，principal $G$-bundle over $X$ 由映射
$$
X\to BG
$$
分类。fiber over $x$ 是相应的 $G$-torsor。

**验证状态.** 这是 HoTT 中 classifying space 的标准合成形式。本书将其作为后续 bundle/cohomology 章节接口；逐行展开需选择具体 torsor 定义和 HIT delooping。

## BF.6 Actions as families over $BG$

**定义 BF.15（action by univalence）。** 若 $BG$ 是 $G$ 的 delooping，则 $G$ 在类型 $A$ 上的作用可由类型族
$$
E:BG\to\mathcal U
$$
及基点 fiber $E(\ast)=A$ 表示。沿 loop 的 transport 给出
$$
G\simeq\Omega BG\to(A\simeq A).
$$

**命题 BF.16（transport action law，证明核）。** 路径复合的 transport 等式给出作用的乘法律。

**证明.** 对 $p,q:\Omega BG$，transport 满足
$$
\mathsf{transport}^E(p\cdot q)
=
\mathsf{transport}^E(q)\circ\mathsf{transport}^E(p)
$$
方向依赖本书路径复合约定。该公式由路径归纳证明。$\square$

## BF.7 Cohomological viewpoint

**定义 BF.17（非阿贝尔一上同调接口）。** 对群 $G$，可把
$$
H^1(X;G)
$$
理解为映射类型 $\|X\to BG\|_0$，即 principal $G$-bundles 的同构类。

**对比 BF.18.** 附录 Y 的 $H^n(X;A)$ 使用阿贝尔群 $A$ 和 EM 型 $K(A,n)$。当 $n=1$ 且 $G$ 非阿贝尔时，classifying type $BG$ 仍可分类 torsor，但不形成普通阿贝尔群值上同调。

## BF.8 Delooping 与分类的边界

Transport action 只需带基点类型 $BG$；“$BG$ 分类所有 $G$-torsor”则还需要具体 delooping 构造和分类定理。一般 $BG$ 的 HIT、非阿贝尔上同调以及它与谱或 EM 型上同调的比较在本书中不是内部定理，只有在相应外部输入明确给出后才能使用。
