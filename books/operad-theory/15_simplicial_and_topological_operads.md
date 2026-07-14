# 第十五章：Simplicial operad 与 topological operad

Little cubes 的每个 arity 不是集合，而是带连续路径和高阶同伦的配置空间；只取其点集会丢失 operad 最重要的信息。可以直接在紧生成空间中保留这些配置，也可以取奇异复形，把连续同伦转成 simplicial 方向的组合数据。几何实现与奇异复形虽给出 Quillen equivalence，但要把它们逐 arity 提升到 operad，还需确认有限积、单位和 transferred 模型结构相容。本章沿这条比较研究 topological 与 simplicial operad，并以 little cubes、chains 以及只有一元运算的 simplicial categories 检查各层结构实际保留了什么。

## 15.1 空间型底范畴

**约定 15.1.** 本章中
$$
\mathbf{sSet}=\operatorname{Fun}(\Delta^{\operatorname{op}},\mathbf{Set}_{\mathcal U})
$$
表示 simplicial sets 范畴，采用 Kan-Quillen 模型结构：weak equivalences 为弱同伦等价，cofibrations 为 monomorphisms，fibrant objects 为 Kan complexes。

**约定 15.2.** $\mathbf{Top}$ 表示 compactly generated weak Hausdorff spaces 范畴，采用标准 Quillen 模型结构：weak equivalences 为弱同伦等价，fibrations 为 Serre fibrations。

**外部输入定理 15.3（Quillen equivalence）.** 几何实现与奇异复形给出 Quillen equivalence
$$
|-|:\mathbf{sSet}\rightleftarrows\mathbf{Top}:\operatorname{Sing}.
$$

更具体地，若 $K$ 是 simplicial set，$X$ 是 topological space，则有自然同构
$$
\mathbf{Top}(|K|,X)\cong\mathbf{sSet}(K,\operatorname{Sing}X).
$$

**命题 15.4.** $\mathbf{sSet}$ 是 cartesian closed 对称幺半模型范畴，幺半积为笛卡儿积。

**证明.** $\mathbf{sSet}$ 中有限积逐 simplicial degree 计算。对 simplicial sets $K,L$，内部 Hom 由
$$
\underline{\operatorname{Hom}}(K,L)_n=\mathbf{sSet}(K\times\Delta^n,L)
$$
定义，给出 cartesian closed 结构。Kan-Quillen 模型结构中 cofibrations 是 monomorphisms。若 $i:A\to B$ 和 $j:C\to D$ 为 monomorphisms，则 pushout-product
$$
(B\times C)\coprod_{A\times C}(A\times D)\to B\times D
$$
逐 degree 是集合的包含映射，因此是 monomorphism。若 $i$ 或 $j$ 为 trivial cofibration，则 pushout-product 是 anodyne extension；这是 Kan-Quillen 模型结构的 pushout-product axiom。单位对象 $\Delta^0$ cofibrant，unit axiom 成立。$\square$

**命题 15.5.** $\mathbf{Top}$ 在 compactly generated weak Hausdorff 口径下是 cartesian closed 对称幺半模型范畴。

**证明.** Compactly generated weak Hausdorff spaces 的乘积和 mapping space 给出 cartesian closed 结构。标准 Quillen 模型结构满足 pushout-product axiom 与 unit axiom；这些属于空间模型范畴的基础定理。$\square$

命题 15.5 的模型结构部分依赖经典拓扑模型范畴理论；此处只记录可用结论。

## 15.2 Simplicial operads

**定义 15.6.** Simplicial symmetric sequence 是函子
$$
X:\mathbf B_{\mathcal U}\to\mathbf{sSet}.
$$
等价地，它是带右 $\Sigma_n$ 作用的 simplicial sets $X(n)$ 的族。

**定义 15.7.** Simplicial operad 是 $\operatorname{SymSeq}(\mathbf{sSet})$ 中关于代入乘积 $\circ$ 的幺半对象。

展开地说，它包含：

1. 每个有限集 $S$ 上的 simplicial set $\mathcal O(S)$；
2. 对每个有限集映射 $f:S\to T$ 的复合映射
   $$
   \mathcal O(T)\times\prod_{t\in T}\mathcal O(f^{-1}(t))\to \mathcal O(S);
   $$
3. 单位点 $\Delta^0\to\mathcal O(\{*\})$；
4. 与可复合有限集映射、双射重标号相容的结合律、单位律和等变性。

这里的乘积是 simplicial sets 的笛卡儿积；$f$ 的空纤维保留 nullary operation spaces。

**定义 15.8.** Simplicial operad morphism $f:\mathcal O\to\mathcal P$ 是 operad morphism in $\mathbf{sSet}$。称 $f$ 为 entrywise weak equivalence，若每个 $n$ 上
$$
f(n):\mathcal O(n)\to\mathcal P(n)
$$
是 Kan-Quillen weak equivalence。

**外部输入定理 15.9（simplicial operad 转移；BM-1）.** 在 Kan--Quillen $\mathbf{sSet}$ 的 cartesian model structure 中，单位 $\Delta^0$ cofibrant，$\operatorname{Ex}^{\infty}$ 给出保有限积的 fibrant replacement，$\Delta^1$ 连同端点与对角给出 commutative Hopf interval。故 Berger--Moerdijk Theorem 3.1（BM-1）适用，$\operatorname{Op}(\mathbf{sSet})$ 存在 transferred 模型结构，weak equivalences 和 fibrations 逐 arity 检测。

**说明 15.10.** 因为 $\mathbf{sSet}$ 中所有对象 cofibrant，许多 cofibrancy 假设比链复形情形更温和。但 operad 代数的 rectification 仍需检查 fixed point、coinvariant 或等变条件；它不是单由“所有对象 cofibrant”推出。

## 15.3 Topological operads

**定义 15.11.** Topological symmetric sequence 是函子
$$
X:\mathbf B_{\mathcal U}\to\mathbf{Top}.
$$

**定义 15.12.** Topological operad 是 $\operatorname{SymSeq}(\mathbf{Top})$ 中关于 cartesian product 诱导的代入乘积的幺半对象。

换言之，对每个有限集映射 $f:S\to T$ 有连续映射
$$
\mathcal O(T)\times\prod_{t\in T}\mathcal O(f^{-1}(t))\to\mathcal O(S),
$$
且这些映射满足 operad 公理；空纤维对应 nullary operation spaces。

**定义 15.13.** Topological operad $\mathcal O$ 称为 well-pointed，若单位包含
$$
*\to\mathcal O(1)
$$
是 $\mathbf{Top}$ 中的 cofibration。称 $\mathcal O$ 为 $\Sigma$-free，若每个 $\Sigma_n$ 在 $\mathcal O(n)$ 上自由作用。

**说明 15.14.** Well-pointed 条件常用于 $W$-construction 和代数同伦理论。$\Sigma$-free 条件常用于避免对称群稳定子带来的等变同伦问题。

**外部输入定理 15.15（topological operad 转移；BM-1）.** 取 compactly generated weak Hausdorff spaces 的标准 cofibrantly generated cartesian model structure。单位点 cofibrant，所有对象 fibrant，$[0,1]$ 连同端点与对角给出 commutative Hopf interval；因此 BM-1 给出 $\operatorname{Op}(\mathbf{Top})$ 的 transferred 模型结构，weak equivalences 和 fibrations 逐 arity 检测。若改用其他 convenient category of spaces，必须重新检查 BM-1 的单位、小性、fibrant replacement 与 interval 假设。

**命题 15.16.** 若 $\mathcal O$ 是 topological operad，则
$$
\operatorname{Sing}\mathcal O
$$
逐 arity 定义为 $(\operatorname{Sing}\mathcal O)(S)=\operatorname{Sing}(\mathcal O(S))$，并自然成为 simplicial operad。

**证明.** 奇异复形函子 $\operatorname{Sing}:\mathbf{Top}\to\mathbf{sSet}$ 保有限积：
$$
\operatorname{Sing}(X\times Y)_n=\mathbf{Top}(|\Delta^n|,X\times Y)
\cong \mathbf{Top}(|\Delta^n|,X)\times\mathbf{Top}(|\Delta^n|,Y).
$$
因此 topological operad 沿任意 $f:S\to T$ 的复合映射
$$
\mathcal O(T)\times\prod_{t\in T}\mathcal O(f^{-1}(t))\to\mathcal O(S)
$$
经 $\operatorname{Sing}$ 后给出
$$
\operatorname{Sing}\mathcal O(T)\times\prod_{t\in T}\operatorname{Sing}\mathcal O(f^{-1}(t))\to\operatorname{Sing}\mathcal O(S).
$$
单位、结合律和等变性由 $\operatorname{Sing}$ 的函子性和保积性质保持。故 $\operatorname{Sing}\mathcal O$ 是 simplicial operad。$\square$

**命题 15.17.** 若 $\mathcal P$ 是 simplicial operad，则逐 arity 几何实现
$$
|\mathcal P|(S)=|\mathcal P(S)|
$$
自然成为 topological operad。

**证明.** 几何实现与 $\operatorname{Sing}$ 伴随，并且在 compactly generated spaces 中与有限积相容。对每个有限集映射 $f:S\to T$，simplicial operad 的复合
$$
\mathcal P(T)\times\prod_{t\in T}\mathcal P(f^{-1}(t))\to\mathcal P(S)
$$
几何实现后给出连续映射
$$
|\mathcal P(T)\times\prod_{t\in T}\mathcal P(f^{-1}(t))|\to|\mathcal P(S)|.
$$
利用有限积相容性，把左端识别为
$$
|\mathcal P(T)|\times\prod_{t\in T}|\mathcal P(f^{-1}(t))|.
$$
由几何实现的函子性，单位、结合律和等变性保持。$\square$

**外部边界 15.18（operad-level realization--Sing comparison）.** 命题 15.16--命题 15.17 已在内部构造逐 arity 伴随
$$
|-|:\operatorname{Op}(\mathbf{sSet})\rightleftarrows \operatorname{Op}(\mathbf{Top}):\operatorname{Sing}
$$
这个构造只给出底层伴随。要断言它在定理 15.9 与 15.15 的 transferred 模型结构间为 Quillen equivalence，还需要 operad-category 的 change-of-base theorem，并核对单位、cofibrancy 和 monoidal comparison。控制 transported operads 之代数范畴的结果不能未经翻译替代 operad-category 比较；因此本节只使用外部输入定理 15.3 的底范畴 Quillen equivalence，不声称上述 operad-level 伴随已经是 Quillen equivalence。

## 15.4 Little cubes operad

**定义 15.19.** 对整数 $d\ge1$，little $d$-cubes operad $\mathcal C_d$ 定义如下。当 $n=0$ 时，$\mathcal C_d(0)$ 是一点空间，对应空 cube 族。当 $n\ge1$ 时，$\mathcal C_d(n)$ 是所有 $n$ 个两两内部不交的仿射嵌入
$$
c_i:[0,1]^d\hookrightarrow[0,1]^d,\qquad 1\le i\le n,
$$
组成的空间，其中每个 $c_i$ 形如
$$
c_i(t_1,\ldots,t_d)=(a_{i1}t_1+b_{i1},\ldots,a_{id}t_d+b_{id}),
$$
且 $0<a_{ij}\le1$，$0\le b_{ij}\le1-a_{ij}$。

对称群 $\Sigma_n$ 通过重排 cubes 作用。单位为恒等嵌入 $[0,1]^d\to[0,1]^d$。

**定义 15.20.** 若
$$
c=(c_1,\ldots,c_n)\in\mathcal C_d(n),\qquad
d_i=(d_{i1},\ldots,d_{ik_i})\in\mathcal C_d(k_i),
$$
则 operad 复合定义为
$$
c\circ(d_1,\ldots,d_n)
=\big(c_i\circ d_{ij}\big)_{1\le i\le n,\,1\le j\le k_i}
\in\mathcal C_d(k_1+\cdots+k_n),
$$
其中输出按 blocks 顺序排列。

这里允许 $k_i=0$；此时 $d_i$ 是唯一空 cube 族，外层第 $i$ 个 cube 不产生输出 cube。这是 little-cubes 模型中的 nullary substitution。

**命题 15.21.** $\mathcal C_d$ 是 topological operad。

**证明.** 首先，仿射嵌入的参数 $(a_{ij},b_{ij})$ 给出 $\mathcal C_d(n)$ 作为欧氏空间中由不等式切出的子空间，因此 composition maps 的连续性可逐坐标检查。若 $c_i(t)=A_it+b_i$ 且 $d_{ij}(t)=A_{ij}t+b_{ij}$，则
$$
(c_i\circ d_{ij})(t)=A_iA_{ij}t+(A_ib_{ij}+b_i),
$$
其参数是原参数的多项式表达，因此连续。

结合律来自函数复合的结合律：
$$
(c_i\circ d_{ij})\circ e_{ijr}=c_i\circ(d_{ij}\circ e_{ijr}).
$$
单位律来自恒等嵌入作为函数复合单位。对称群等变性来自重排 indexed cubes 与上述复合公式相容。故 $\mathcal C_d$ 是 topological operad。$\square$

**外部输入定理 15.22（May recognition principle）.** 合适连通性和基点条件下，$\mathcal C_d$-spaces 刻画 $d$-fold loop spaces up to group completion。

该定理已在第十章作为外部输入出现；此处强调其 operad 是 topological operad，而非 simplicial 或 dg-operad。若要进入 $\mathbf{sSet}$ 或 $\mathbf{Ch}_k$，需分别取 $\operatorname{Sing}$ 或 chains。

## 15.5 从拓扑到链：chains on spaces

设 $k$ 是交换环。奇异链函子
$$
C_\*(-;k):\mathbf{Top}\to\mathbf{Ch}_k
$$
不是严格保笛卡儿积的强对称幺半函子；它通过 Eilenberg-Zilber 映射给出 lax symmetric monoidal 结构。

**外部输入定理 15.23（Eilenberg-Zilber）.** 存在自然 chain maps
$$
C_\*(X;k)\otimes C_\*(Y;k)\to C_\*(X\times Y;k)
$$
和
$$
C_\*(X\times Y;k)\to C_\*(X;k)\otimes C_\*(Y;k),
$$
它们在同伦意义下互为逆，并与对称性和结合性满足相干关系。

**命题 15.24.** 若 $\mathcal O$ 是 topological operad，则 $C_\*(\mathcal O;k)$ 自然给出 dg-operad，前提是选择了相干的 Eilenberg-Zilber lax monoidal 结构。

**证明.** 对每个有限集映射 $f:S\to T$，topological operad 复合给出
$$
\mathcal O(T)\times\prod_{t\in T}\mathcal O(f^{-1}(t))\to\mathcal O(S).
$$
先用 Eilenberg-Zilber lax monoidal map 得到
$$
C_\*(\mathcal O(T);k)\otimes\bigotimes_{t\in T} C_\*(\mathcal O(f^{-1}(t));k)
\to
C_\*\left(\mathcal O(T)\times\prod_{t\in T}\mathcal O(f^{-1}(t));k\right),
$$
再对复合映射取 chains，得到 dg-operad 的复合。单位由点的奇异 $0$-simplex 给出。结合律和等变性依赖 Eilenberg-Zilber 结构的相干性；这正是定理 15.23 中相干关系的用途。$\square$

**说明 15.25.** 这一步解释了为什么 $C_\*(\mathcal C_d;k)$ 是 $E_d$ dg-operad 的标准来源。但若讨论形式性，例如
$$
C_\*(\mathcal C_d;k)\simeq H_\*(\mathcal C_d;k),
$$
还需要额外的域、特征和模型结构假设；这些不由本章自动推出。

## 15.6 Simplicial categories as unary colored operads

**定义 15.26.** Simplicial category 是 enriched category over $\mathbf{sSet}$。它由对象集 $C$、mapping simplicial sets
$$
\mathcal A(x,y)
$$
和复合映射
$$
\mathcal A(y,z)\times\mathcal A(x,y)\to\mathcal A(x,z)
$$
组成。

**命题 15.27.** 对象集为 $C$ 的 simplicial categories 等价于只有 unary operations 的 $C$-colored simplicial operads。

**证明.** 给定 simplicial category $\mathcal A$，定义 colored operad $\mathcal O_\mathcal A$：
$$
\mathcal O_\mathcal A(c_1,\ldots,c_n;c)=
\begin{cases}
\mathcal A(c_1,c),& n=1,\\
\varnothing,& n\ne1.
\end{cases}
$$
Unary operad 复合正是 enriched category 的复合，单位正是 identity morphisms。反向地，给定只有 unary operations 的 colored simplicial operad，令
$$
\mathcal A(x,y)=\mathcal O(x;y).
$$
Operad 的单位和 unary composition 给出 enriched category 的单位和复合。两个构造在对象、mapping simplicial sets 和结构映射上互逆。$\square$

**说明 15.28.** 因此 simplicial categories 是 colored simplicial operads 的一维特例。后续 dendroidal sets 把 simplicial sets 视为“线性树”上的 presheaves，并把 operads 视为“所有树”上的 presheaves；这就是从 category nerve 到 dendroidal nerve 的动机。

## 15.7 三种实现各自保留的结构

Little cubes 的拓扑 operad保留真实配置空间，$\operatorname{Sing}$ 把它转为 simplicial operation spaces，chains 再把空间同伦压到 dg 层。前两者在相应 transferred 模型结构与幺半相容假设下由 Quillen 理论比较；chains 通常只给弱幺半或派生层面的信息，不能无条件反向恢复拓扑。只有一元运算的 simplicial operad正是 simplicial category，这条一维特例提示下一章的推广：把线性字符串换成有根树，便能同时记录任意多输入复合。

## 练习

**练习 15.1.** 证明 $\operatorname{Sing}$ 保有限积。

**练习 15.2.** 给出 $\mathcal C_1(2)$ 的显式参数空间，并描述 $\Sigma_2$ 作用。

**练习 15.3.** 验证 little cubes operad 的单位律。

**练习 15.4.** 设 $\mathcal O$ 是 discrete simplicial operad，即每个 $\mathcal O(n)$ 为常值 simplicial set。证明 $\mathcal O$ 等价于集合值 operad。

**练习 15.5.** 把一个普通小范畴写成只有 unary operations 的 colored operad，并说明其 simplicial nerve 与后续 dendroidal nerve 的关系。
