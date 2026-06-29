# 第四章：自由 operad、生成元与关系

## 本章目标

本章构造自由 operad，并给出“生成元与关系”的严格含义。核心思想是：自由 operad 的元素是由生成运算装饰的有根树；operad 代入是树的 grafting；关系是与重标号和代入稳定相容的 operadic congruence。

## 依赖前置知识

需要第一章的对称 operad、第三章的树代入思想，以及商集和泛性质。

## 4.1 非对称自由 operad

**定义 4.1.** 设 $E=\{E(n)\}_{n\ge0}$ 是非对称序列。定义非对称序列 $\mathbb F_{\mathrm{ns}}(E)$ 如下。$\mathbb F_{\mathrm{ns}}(E)(n)$ 是所有 arity 为 $n$ 的平面有根树 $T$ 连同顶点装饰
$$
\xi_v\in E(\operatorname{in}(v))
$$
的同构类集合。单位树没有内部顶点，给出 $\mathbb F_{\mathrm{ns}}(E)(1)$ 中的单位元素。

**定义 4.2.** 对两个装饰平面树 $T$ 和 $T_1,\ldots,T_n$，其中 $T$ 有 $n$ 个叶，定义 grafting
$$
T(T_1,\ldots,T_n)
$$
为把 $T_i$ 的根边接到 $T$ 的第 $i$ 个叶上所得平面树。顶点装饰由原装饰继承。

**命题 4.3.** $\mathbb F_{\mathrm{ns}}(E)$ 连同 grafting 是非对称 operad。

**证明.** 单位树 graft 到任一叶不改变树，任一树 graft 到单位树的唯一叶也不改变树，因此单位律成立。结合律要求比较
$$
T(T_1,\ldots,T_n)(S_{11},\ldots,S_{n k_n})
$$
与
$$
T(T_1(S_{11},\ldots,S_{1k_1}),\ldots,
T_n(S_{n1},\ldots,S_{nk_n})).
$$
两边的底层平面树由同一批树按同一叶-根邻接关系粘合而成，平面顺序也相同；装饰逐顶点继承，因此两边是同一个装饰树同构类。$\square$

**定理 4.4.** $\mathbb F_{\mathrm{ns}}$ 是从非对称序列到非对称 operad 的自由函子。换言之，对任意非对称 operad $\mathcal P$，有自然双射
$$
\operatorname{Op}_{\mathrm{ns}}(\mathbb F_{\mathrm{ns}}(E),\mathcal P)
\cong
\operatorname{Seq}_{\mathrm{ns}}(E,U\mathcal P).
$$

**证明.** 给定 operad morphism $\Phi:\mathbb F_{\mathrm{ns}}(E)\to\mathcal P$，限制到单顶点树得到非对称序列态射 $E\to U\mathcal P$。

反过来，给定非对称序列态射 $f:E\to U\mathcal P$，对一个 $E$-装饰树 $T$，把每个顶点装饰 $\xi_v$ 替换为 $f(\xi_v)\in\mathcal P(\operatorname{in}(v))$，再用第三章命题 3.15 的树求值定义
$$
\widehat f(T)=\operatorname{ev}_{\mathcal P}(T).
$$
该值与收缩顺序无关。grafting 后求值等于先求各输入树的值再在外树中代入，这是树收缩定义和 operad 结合律的直接结果，所以 $\widehat f$ 是 operad morphism。单顶点树保证 $\widehat f$ 限制为 $f$，而任一装饰树由单顶点树经 grafting 生成，因此这样的 $\widehat f$ 唯一。$\square$

## 4.2 对称自由 operad 的树公式

**定义 4.5.** 设 $S$ 是有限集。一个 $S$-标号有根树是有限有根树 $T$，其叶集合 $\operatorname{Leaf}(T)$ 配有双射
$$
\lambda:S\cong\operatorname{Leaf}(T).
$$
内部顶点 $v$ 的输入边集合记为 $\operatorname{In}(v)$。树同构必须保持根、叶标号和输入边-输出边 incidence。

**定义 4.6.** 设 $E:\mathbf B_{\mathcal U}\to\mathbf{Set}_{\mathcal U}$ 是对称序列。一个 $E$-装饰 $S$-标号有根树是 $S$-标号有根树 $T$ 连同每个内部顶点上的元素
$$
\xi_v\in E(\operatorname{In}(v)).
$$
若树同构 $\theta:T\to T'$ 把 $\operatorname{In}(v)$ 双射到 $\operatorname{In}(\theta v)$，则装饰按 $E$ 的函子性被送到对应装饰。

**定义 4.7.** 定义对称序列 $\mathbb F(E)$：
$$
\mathbb F(E)(S)
=
\{\text{$E$-装饰 $S$-标号有根树}\}/\cong.
$$
若 $\varphi:S\to S'$ 是双射，则 $\mathbb F(E)(\varphi)$ 把叶标号 $\lambda:S\cong\operatorname{Leaf}(T)$ 改为
$$
S'\xrightarrow{\varphi^{-1}}S\xrightarrow{\lambda}\operatorname{Leaf}(T).
$$

**定义 4.8.** 设 $\pi$ 是 $S$ 的分块。给定
$$
T\in\mathbb F(E)(\operatorname{Bl}(\pi)),
\qquad
T_B\in\mathbb F(E)(B)\quad(B\in\operatorname{Bl}(\pi)),
$$
定义代入树
$$
T\big((T_B)_{B\in\operatorname{Bl}(\pi)}\big)
$$
为把 $T$ 中标号为 $B$ 的叶替换为树 $T_B$；新叶集合由各 $B$ 的叶标号给出，所以总叶标号为 $S$。装饰由所有树的原装饰继承。

**命题 4.9.** $\mathbb F(E)$ 连同定义 4.8 的树代入是对称 operad。

**证明.** 重标号函子性来自叶标号双射的复合。单位是单边无顶点树，叶集合为单点集。把单位树代入任一叶不改变树，任一树代入单位树也不改变树，因此单位律成立。

结合律比较两种三层代入：先把 $T_{B,C}$ 代入 $T_B$ 再代入 $T$，或者先把 $T_B$ 代入 $T$ 后再一次性代入所有 $T_{B,C}$。两边得到的有根树有同一顶点集合、同一边粘合关系、同一根、同一 $S$-叶标号和同一顶点装饰。因此它们代表同一个同构类。双射重标号只改变叶标号，不改变粘合；所以代入与对称序列结构自然相容。$\square$

**定理 4.10.** $\mathbb F$ 是从对称序列到对称 operad 的自由函子。对任意对称 operad $\mathcal O$，自然双射
$$
\operatorname{Op}(\mathbb F(E),\mathcal O)
\cong
\operatorname{SymSeq}(E,U\mathcal O)
$$
成立。

**证明.** operad morphism 限制到单顶点树给出对称序列态射 $E\to U\mathcal O$。

反过来，设 $f:E\to U\mathcal O$ 是对称序列态射。对一个 $E$-装饰 $S$-标号树 $T$，先把每个顶点装饰 $\xi_v$ 替换为 $f(\xi_v)\in\mathcal O(\operatorname{In}(v))$。然后从叶向根逐步收缩内部边：当顶点 $v$ 的输出边接入顶点 $w$ 的某个输入边时，用 $\mathcal O$ 的有限集分块代入把 $v$ 的值代入 $w$ 的相应输入。对称 operad 结合律保证不同可收缩边的选择给出同一元素
$$
\widehat f(T)\in\mathcal O(S).
$$
树同构只通过输入边集合的双射改变局部装饰；因为 $f$ 是自然变换且 $\mathcal O$ 的代入对双射自然，$\widehat f(T)$ 与代表元选择无关。树 grafting 与 operad 代入相容，所以 $\widehat f:\mathbb F(E)\to\mathcal O$ 是 operad morphism。唯一性同样由每棵装饰树可由单顶点树经 operad 代入生成得到。$\square$

## 4.3 生成元、关系与商 operad

**定义 4.11.** 设 $\mathcal O$ 是 operad。一个 operadic congruence 是一族等价关系
$$
\sim_S\ \subseteq\ \mathcal O(S)\times\mathcal O(S)
$$
满足：

1. 对任意双射 $\varphi:S\to T$，若 $x\sim_S y$，则 $\mathcal O(\varphi)(x)\sim_T\mathcal O(\varphi)(y)$。
2. 对任意分块 $\pi$，若外层 $x\sim_{\operatorname{Bl}(\pi)}y$ 且每个块上 $x_B\sim_B y_B$，则
   $$
   \mu(x;(x_B))\sim_S\mu(y;(y_B)).
   $$

**命题 4.12.** 若 $\sim$ 是 $\mathcal O$ 上的 operadic congruence，则逐 arity 商
$$
(\mathcal O/\!\sim)(S)=\mathcal O(S)/\!\sim_S
$$
带有唯一 operad 结构，使得商映射
$$
q:\mathcal O\to\mathcal O/\!\sim
$$
是 operad morphism。

**证明.** 重标号由
$$
[x]\mapsto[\mathcal O(\varphi)(x)]
$$
定义，条件 1 保证良定义。代入由
$$
[x];([x_B])\mapsto[\mu(x;(x_B))]
$$
定义，条件 2 保证良定义。单位是 $[\mathbf 1]$。operad 公理在商中成立，因为它们在 $\mathcal O$ 中成立并且商映射保持等式。唯一性来自 $q$ 逐 arity 满射。$\square$

**定义 4.13.** 设 $E$ 是对称序列，$R$ 是一族成对元素
$$
r_0,r_1\in\mathbb F(E)(S_r).
$$
由 $R$ 生成的 operadic congruence 是包含所有 $r_0\sim r_1$ 且满足定义 4.11 的最小 congruence。由生成元 $E$ 和关系 $R$ 给出的 operad 定义为
$$
\langle E\mid R\rangle
=
\mathbb F(E)/{\sim_R}.
$$

**证明存在性.** 所有包含 $R$ 的 operadic congruence 的交仍是 operadic congruence，因为等价关系、重标号稳定性和代入稳定性都对任意交封闭。因此最小者存在。$\square$

**命题 4.14.** 对任意 operad $\mathcal O$，给出 operad morphism
$$
\langle E\mid R\rangle\to\mathcal O
$$
等价于给出对称序列态射 $f:E\to U\mathcal O$，使得每一条关系 $r_0=r_1$ 在 $\mathcal O$ 中成立。

**证明.** 由定理 4.10，$f$ 唯一延拓为 $\widehat f:\mathbb F(E)\to\mathcal O$。该延拓穿过商 $\mathbb F(E)\to\langle E\mid R\rangle$ 当且仅当 $\widehat f$ 把 $\sim_R$-等价元素送到相等元素。因为 $\mathcal O$ 中的相等关系是 operadic congruence，所以只需检查生成关系 $R$。$\square$

## 4.4 Ass 与 Com 的表示

**例 4.15.** 结合 operad $\operatorname{Ass}$ 可由一个二元生成元 $m$ 和一个零元生成元 $e$ 表示。这里 $m$ 位于 arity $2$，$e$ 位于 arity $0$。关系为
$$
m\circ_1 m=m\circ_2 m,
$$
以及
$$
m\circ_1 e=\mathbf 1,\qquad m\circ_2 e=\mathbf 1.
$$
这些关系表达结合律和左右单位律。注意在对称 operad 中，$m$ 的非平凡置换给出反向乘法；$\operatorname{Ass}$ 不加入 $m=(12)\cdot m$ 的关系。

**命题 4.16.** 例 4.15 的表示给出的 operad 的代数范畴等价于幺半群范畴。

**证明.** 由命题 4.14，一个该表示的代数是在集合 $X$ 上选择二元运算 $m_X:X^2\to X$ 和元素 $e_X\in X$，使得三条关系在 $\operatorname{End}_X$ 中成立。第一条关系在 $\operatorname{End}_X(3)$ 中正是
$$
m_X(m_X(x,y),z)=m_X(x,m_X(y,z)).
$$
后两条关系正是
$$
m_X(e_X,x)=x,\qquad m_X(x,e_X)=x.
$$
这就是幺半群结构。同态条件也正是保持乘法和单位。$\square$

**例 4.17.** 交换 operad $\operatorname{Com}$ 可在例 4.15 的生成元和关系基础上再加入交换关系
$$
(12)\cdot m=m.
$$

**命题 4.18.** 例 4.17 的表示给出的 operad 的代数范畴等价于交换幺半群范畴。

**证明.** 命题 4.16 已给出幺半群结构。新增关系在 $\operatorname{End}_X(2)$ 中表示
$$
m_X(y,x)=m_X(x,y),
$$
即交换律。反过来，任何交换幺半群满足结合、单位和交换关系，因此由命题 4.14 给出唯一代数结构。$\square$

## 本章小结

自由 operad 的严格模型是装饰有根树。非对称情形使用平面树；对称情形使用叶标号树和输入边集合的重标号。生成元与关系的精确定义不是“写若干公式”本身，而是在自由 operad 上取由这些公式生成的 operadic congruence。这个观点将在 Lie、Poisson、Gerstenhaber、BV 和 Koszul 对偶章节中反复使用。

## 练习

**练习 4.1.** 证明非对称自由 operad 中的单位树确实是 grafting 的左右单位。

**练习 4.2.** 写出由一个三元生成元 $t\in E(3)$ 生成的自由非对称 operad 在 arity $5$ 中的三个不同树形元素。

**练习 4.3.** 对称自由 operad 中，解释为什么同一棵未标号树的不同叶标号通常给出不同元素。

**练习 4.4.** 证明 operadic congruence 的任意交仍是 operadic congruence。

**练习 4.5.** 用生成元与关系写出没有单位的结合 operad，并说明它与本书默认的 $\operatorname{Ass}$ 有何不同。

