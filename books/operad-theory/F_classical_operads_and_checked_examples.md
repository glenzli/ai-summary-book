# 附录 F：经典 operad 的逐项验算

## 本附录目标

本附录把第一章和第六章中快速出现的经典例子展开为可检查的数学对象。重点不是增加新主题，而是验证以下四类数据：

1. 对称序列的 arity 数据和对称群作用。
2. operad 复合与单位。
3. 代数对象的泛性质或生成元关系。
4. arity $0$ 运算在含单位版本中的作用。

除非特别说明，本附录固定交换环 $R$，张量积为 $\otimes_R$，线性 operad 取值于 $\mathbf{Mod}_R$。集合值 operad 的线性化记为 $R[-]$。

## F.1 集合值线性化的检查

**定义 F.1.** 设 $\mathcal O$ 是集合值 operad。其线性化 $R[\mathcal O]$ 定义为
$$
R[\mathcal O](S)=R[\mathcal O(S)],
$$
其中右边是由集合 $\mathcal O(S)$ 生成的自由 $R$-模。若
$$
\gamma_\pi:\mathcal O(\operatorname{Bl}(\pi))\times\prod_{B\in\operatorname{Bl}(\pi)}\mathcal O(B)\to\mathcal O(S)
$$
是 $\mathcal O$ 沿分块 $\pi$ 的复合，则 $R[\mathcal O]$ 的复合定义为 $\gamma_\pi$ 的 $R$-多线性延拓：
$$
R[\mathcal O(\operatorname{Bl}(\pi))]
\otimes
\bigotimes_B R[\mathcal O(B)]
\longrightarrow
R[\mathcal O(S)].
$$

**命题 F.2.** $R[\mathcal O]$ 是 $R$-线性 operad。

**证明.** 需要检查三件事。

第一，双射作用是函子性的。若 $S\xrightarrow{\varphi}T\xrightarrow{\psi}U$ 是双射，则 $\mathcal O(\psi\varphi)=\mathcal O(\psi)\mathcal O(\varphi)$。自由 $R$-模函子把等式送为线性映射等式，所以 $R[\mathcal O]$ 是 $R$-模值对称序列。

第二，复合与重标号相容。该相容性在 $\mathcal O$ 中对基元素成立。由于 $R[\mathcal O(\operatorname{Bl}(\pi))]\otimes\bigotimes_B R[\mathcal O(B)]$ 由纯张量基生成，多线性延拓后仍成立。

第三，结合律和单位律成立。任取三层分块和每个顶点的基元素，两种复合结果在 $\mathcal O$ 中相等；线性化后，纯张量基上的等式推出整个张量积上的等式。单位律同理由 $\mathcal O$ 的单位律在线性化后得到。$\square$

**命题 F.3.** 设 $A$ 是 $R$-模。给出 $R[\mathcal O]$-代数结构等价于给出一族映射
$$
\theta_S:\mathcal O(S)\to \operatorname{Hom}_R(A^{\otimes S},A)
$$
使其关于双射、单位和分块复合相容。

**证明.** 一个 $R$-线性 operad morphism
$$
R[\mathcal O]\to \operatorname{End}_A
$$
在每个有限集 $S$ 上是 $R$-线性映射
$$
R[\mathcal O(S)]\to \operatorname{Hom}_R(A^{\otimes S},A).
$$
由自由 $R$-模的泛性质，它等价于集合映射 $\theta_S$。operad morphism 的等变性、单位和乘法相容条件逐 arity 展开后，正是命题中的三类相容条件。反向由这些相容条件唯一线性延拓得到 operad morphism。$\square$

## F.2 结合 operad 与张量代数

**定义 F.4.** 对有限集 $S$，令 $\operatorname{Lin}(S)$ 表示 $S$ 上全序的集合。结合 operad 的集合值模型为
$$
\operatorname{Ass}(S)=\operatorname{Lin}(S).
$$
若 $\pi$ 是 $S$ 的分块，$\alpha$ 是块集合 $\operatorname{Bl}(\pi)$ 上的全序，且每个块 $B$ 带全序 $\beta_B$，则复合
$$
\alpha\big(\beta_B\big)_{B\in\operatorname{Bl}(\pi)}
$$
定义为 $S$ 上的字典序：先比较元素所在块在 $\alpha$ 中的顺序；若在同一块，再用该块中的 $\beta_B$ 比较。

**命题 F.5.** $\operatorname{Ass}$ 是含 arity $0$ 的集合值 operad。

**证明.** 双射把全序推前，因此 $S\mapsto\operatorname{Lin}(S)$ 是对称序列。单位是单点集上的唯一全序。

检查结合律。设 $S$ 带三层分块：最外层块集合为 $P$，每个 $p\in P$ 内有中层块集合 $Q_p$，每个 $q\in Q_p$ 内有底层集合 $S_q$。给定 $P$ 的全序、每个 $Q_p$ 的全序和每个 $S_q$ 的全序。无论先把 $Q_p$ 与 $S_q$ 合成，还是先把 $P$ 与 $Q_p$ 合成，最终在 $S$ 上得到的比较规则都是：
$$
x<y
$$
当且仅当下列第一条可判定差异成立：

1. $x$ 和 $y$ 所在的最外层块在 $P$ 中有序；
2. 它们在同一个最外层块内，但所在中层块在对应 $Q_p$ 中有序；
3. 它们在同一个中层块内，并由 $S_q$ 上的全序比较。

因此两种复合得到同一全序。单位律来自单点全序在字典序比较中不改变原有顺序。$\square$

**命题 F.6.** Schur functor $S_{R[\operatorname{Ass}]}$ 自然同构于张量代数函子
$$
T(V)=\bigoplus_{n\ge0}V^{\otimes n}.
$$

**证明.** 对每个 $n$，$\operatorname{Lin}([n])$ 在 $\Sigma_n$ 作用下是自由传递的集合。固定标准全序 $1<\cdots<n$ 后，有右 $\Sigma_n$-集合同构
$$
\operatorname{Lin}([n])\cong \Sigma_n.
$$
于是
$$
R[\operatorname{Lin}([n])]\otimes_{R[\Sigma_n]}V^{\otimes n}
\cong
R[\Sigma_n]\otimes_{R[\Sigma_n]}V^{\otimes n}
\cong
V^{\otimes n}.
$$
把 $n$ 上求直和得到
$$
S_{R[\operatorname{Ass}]}(V)
=
\bigoplus_{n\ge0}R[\operatorname{Ass}(n)]\otimes_{R[\Sigma_n]}V^{\otimes n}
\cong
\bigoplus_{n\ge0}V^{\otimes n}.
$$
自然性来自 coinvariants 张量积的泛性质。$\square$

**命题 F.7.** $R[\operatorname{Ass}]$-代数等价于含单位结合 $R$-代数。

**证明.** 由命题 F.6，$R[\operatorname{Ass}]$ 的 Schur monad 底层函子是 $T$。operad 乘法给出的 monad 乘法
$$
T(T(V))\to T(V)
$$
把张量词的词再串接为一个张量词；单位 $\operatorname{id}\to T$ 把 $V$ 放入长度 $1$ 部分。一个 monad 代数 $T(A)\to A$ 等价于含单位结合乘法：长度 $0$ 部分 $R\to A$ 给出单位，长度 $2$ 部分 $A\otimes A\to A$ 给出乘法，monad 结合律强制任意长度词按任意加括号求值相同。反过来，含单位结合代数可用迭代乘法定义 $T(A)\to A$，结合律和单位律保证它是 monad 代数。$\square$

## F.3 交换 operad 与对称代数

**定义 F.8.** 交换 operad 的集合值模型为
$$
\operatorname{Com}(S)=\{*\}
$$
对每个有限集 $S$ 均为单点集，所有双射作用为恒等，所有复合为唯一映射。

**命题 F.9.** $\operatorname{Com}$ 是含 arity $0$ 的集合值 operad。

**证明.** 每个需要检查的图的源和目标都是单点集之间的函数。两个平行函数必相等，因此双射函子性、复合自然性、结合律和单位律全部成立。$\square$

**命题 F.10.** Schur functor $S_{R[\operatorname{Com}]}$ 自然同构于对称代数函子
$$
\operatorname{Sym}_R(V)=\bigoplus_{n\ge0}(V^{\otimes n})_{\Sigma_n}.
$$

**证明.** 因为 $\operatorname{Com}(n)$ 是单点且 $\Sigma_n$ 作用平凡，
$$
R[\operatorname{Com}(n)]\cong R
$$
为平凡右 $R[\Sigma_n]$-模。因此
$$
R[\operatorname{Com}(n)]\otimes_{R[\Sigma_n]}V^{\otimes n}
\cong
R\otimes_{R[\Sigma_n]}V^{\otimes n}
\cong
(V^{\otimes n})_{\Sigma_n}.
$$
对所有 $n$ 求直和即得公式。$\square$

**命题 F.11.** $R[\operatorname{Com}]$-代数等价于含单位交换 $R$-代数。

**证明.** 一个 $R[\operatorname{Com}]$-代数是一个 $R$-模 $A$ 连同 operad morphism
$$
R[\operatorname{Com}]\to\operatorname{End}_A.
$$
arity $0$ 的唯一元素给出 $u:R\to A$，记 $1_A=u(1)$。arity $2$ 的唯一元素给出乘法 $m:A\otimes A\to A$。

对称群 $\Sigma_2$ 在 $\operatorname{Com}(2)$ 上作用平凡，所以等变性给出
$$
m(a,b)=m(b,a).
$$
三元分块对应的 operad 结合律给出
$$
m(m(a,b),c)=m(a,m(b,c)).
$$
把 arity $0$ 元素代入二元运算的左输入或右输入，operad 单位关系给出
$$
m(1_A,a)=a,\qquad m(a,1_A)=a.
$$
故得到含单位交换 $R$-代数。

反过来，若 $A$ 是含单位交换 $R$-代数，则对每个有限集 $S$ 定义
$$
\mu_S:A^{\otimes S}\to A
$$
为所有输入的乘积；当 $S=\varnothing$ 时取 $1_A$。交换律保证 $\mu_S$ 不依赖于对 $S$ 的排序，结合律保证分块代入相容，单位律保证 operad 单位相容。于是得到唯一的 $R[\operatorname{Com}]$-代数结构。$\square$

## F.4 Endomorphism operad 的复合公式

**定义 F.12.** 设 $V$ 是 $R$-模。定义线性 endomorphism operad
$$
\operatorname{End}_V(S)=\operatorname{Hom}_R(V^{\otimes S},V).
$$
这里 $V^{\otimes S}$ 表示按有限集 $S$ 张量得到的对象；选择 $S$ 的一个全序可把它识别为 $V^{\otimes |S|}$，不同选择之间由对称幺半结构给出置换同构。

若 $\pi$ 是 $S$ 的分块，$f\in\operatorname{End}_V(\operatorname{Bl}(\pi))$，且 $g_B\in\operatorname{End}_V(B)$，定义复合
$$
f\circ_\pi(g_B)_B\in\operatorname{End}_V(S)
$$
为以下映射：
$$
V^{\otimes S}
\cong
\bigotimes_{B\in\operatorname{Bl}(\pi)}V^{\otimes B}
\xrightarrow{\otimes_B g_B}
V^{\otimes\operatorname{Bl}(\pi)}
\xrightarrow{f}
V.
$$

**命题 F.13.** $\operatorname{End}_V$ 是 $R$-线性 operad。

**证明.** 双射 $S\to T$ 的作用由 $V^{\otimes T}\cong V^{\otimes S}$ 的重标号同构前合成得到。单位是 $\operatorname{id}_V\in\operatorname{Hom}_R(V,V)$。

检查结合律。给定三层分块，复合映射总是按如下顺序作用在 $V^{\otimes S}$ 上：先在最小块内应用最内层运算，再把每个中层块的输出张量起来应用中层运算，最后应用最外层运算。两种加括号方式只改变中间张量积同构的括号和重排。$\mathbf{Mod}_R$ 的对称幺半相干性保证这些重排同构相同，所以两种复合得到同一 $R$-线性映射。单位律由 $\operatorname{id}_V$ 的复合单位性质给出。$\square$

**命题 F.14.** 若 $\mathcal P$ 是 $R$-线性 operad，则 $\mathcal P$-代数等价于 operad morphism $\mathcal P\to\operatorname{End}_V$。

**证明.** 这是定义 6.12 的展开。Schur monad 代数结构
$$
\bigoplus_{n\ge0}\mathcal P(n)\otimes_{R[\Sigma_n]}V^{\otimes n}\to V
$$
由一族等变 $R$-线性映射
$$
\mathcal P(n)\to\operatorname{Hom}_R(V^{\otimes n},V)
$$
给出。monad 乘法相容正是 operad 乘法在 endomorphism operad 中保持复合；monad 单位相容正是 operad 单位送到 $\operatorname{id}_V$。$\square$

## F.5 Lie operad 的生成元关系检查

**定义 F.15.** 令 $\mathbb F(E)$ 是由一个二元生成元
$$
b\in E(2)
$$
生成的自由 $R$-线性 operad，其中 $\Sigma_2$ 作用由符号表示给出：
$$
(12)\cdot b=-b.
$$
令 $J$ 为由 Jacobi 元素
$$
b\circ_1 b+
(b\circ_1 b)\cdot(123)+
(b\circ_1 b)\cdot(132)
$$
生成的 operadic ideal。定义
$$
\operatorname{Lie}_R=\mathbb F(E)/J.
$$

这里 $(123)$ 和 $(132)$ 表示对三个叶标号的循环置换；该公式是
$$
[[x,y],z]+[[y,z],x]+[[z,x],y]
$$
在自由 operad arity $3$ 中的坐标表达。

**命题 F.16.** $\operatorname{Lie}_R$-代数等价于带一个 $R$-双线性括号并满足反对称与 Jacobi 关系的 $R$-模。

**证明.** 由自由 operad 的泛性质，一个 $\mathbb F(E)$-代数等价于在 $V$ 上给出一个二元 $R$-线性运算
$$
[-,-]:V\otimes V\to V
$$
并满足由 $E(2)$ 的 $\Sigma_2$ 作用给出的反对称关系
$$
[y,x]=-[x,y].
$$
该代数结构下降到商 operad $\mathbb F(E)/J$ 当且仅当 $J$ 中每个关系在 $\operatorname{End}_V$ 中为零。Jacobi 生成元在 $\operatorname{End}_V(3)$ 中的像正是
$$
[[x,y],z]+[[y,z],x]+[[z,x],y].
$$
因此下降条件正是 Jacobi 恒等式。反向由同样的泛性质给出。$\square$

**警告 F.17.** 若 $2$ 在 $R$ 中不可逆，反对称关系不推出 $[x,x]=0$。若要采用 alternating Lie algebra 约定，需要把 $[x,x]=0$ 加为额外关系。此时得到的 operad 与上面的 $\operatorname{Lie}_R$ 在一般底环上可能不同；在 $2$ 可逆的底环上二者相同。

**外部输入定理 F.18.** 若 $R$ 是特征 $0$ 的域，则 $\operatorname{Lie}_R$ 的 arity $n$ 部分可由多种等价组合模型描述，例如 Lyndon word 或 rooted binary tree 的适当商；自由 Lie 代数嵌入张量代数的 primitive 元素中。本文不在此证明这些结构定理。它们应依赖 Reutenauer、Loday-Vallette 或标准 Lie 组合代数资料源。

## F.6 Poisson operad 的关系检查

**定义 F.19.** 普通 Poisson operad $\operatorname{Pois}_R$ 是由两个二元生成元
$$
m,\ell\in \operatorname{Pois}_R(2)
$$
生成的 $R$-线性 operad，关系如下：

1. $m$ 对称：
   $$
   (12)\cdot m=m.
   $$
2. $m$ 结合并有单位。
3. $\ell$ 反对称：
   $$
   (12)\cdot\ell=-\ell.
   $$
4. $\ell$ 满足 Jacobi 关系。
5. Leibniz 关系：
   $$
   \ell\circ_2 m
   =
   m\circ_1 \ell + (m\circ_2 \ell)\cdot(12),
   $$
   其代数表达为
   $$
   \{x,yz\}=\{x,y\}z+y\{x,z\}.
   $$

**命题 F.20.** $\operatorname{Pois}_R$-代数等价于含单位 Poisson $R$-代数。

**证明.** 一个 $\operatorname{Pois}_R$-代数给出两个二元运算：
$$
xy:=m_A(x,y),\qquad \{x,y\}:=\ell_A(x,y).
$$
关系 1 和 2 使 $(A,m_A,1)$ 成为含单位交换 $R$-代数。关系 3 和 4 使 $(A,\ell_A)$ 成为 Lie 型括号对象。关系 5 在 $\operatorname{End}_A(3)$ 中展开为
$$
\{x,yz\}=\{x,y\}z+y\{x,z\}.
$$
因此括号对第二变量是导子。由反对称性可推出它对第一变量也满足相应导子公式：
$$
\{xy,z\}=x\{y,z\}+y\{x,z\}.
$$
反过来，含单位 Poisson $R$-代数的乘法和括号满足上述所有生成元关系，所以由生成元关系 operad 的泛性质得到唯一 $\operatorname{Pois}_R$-代数结构。$\square$

**定义 F.21.** $n$-Poisson operad $\operatorname{Pois}_n$ 是分次版本：乘法为次数 $0$ 的交换结合乘法，括号的同调次数为 $n-1$ 或上同调次数为 $1-n$，具体取决于分次约定。Leibniz 与 Jacobi 关系必须带 Koszul signs，并应以附录 E 的悬挂约定为准。

**外部输入定理 F.22.** 对 little $n$-cubes operad $\mathcal C_n$，其同调 operad 与 $n$-Poisson operad 同构：
$$
H_\*(\mathcal C_n;R)\cong \operatorname{Pois}_n
$$
在适当系数和 convention 下成立。该结论不是由本附录的生成元关系证明推出，而是 Cohen-May 传统中的外部拓扑定理；正文引用时必须保留来源和 convention。

## F.7 arity $0$ 与单位的不可省略性

**命题 F.23.** 在含 arity $0$ 的 $\operatorname{Ass}$-代数中，空全序对应的元素是乘法单位。

**证明.** 设 $A$ 是 $\operatorname{Ass}$-代数。空集上的唯一全序给出元素 $e\in A$。二元标准全序给出乘法 $m$。

考虑集合 $\{1\}$ 的分块，其中外层块集合有两个块，一个块为空集，一个块为 $\{1\}$。在 $\operatorname{Ass}$ 中，把空全序和单点全序代入二元全序，得到单点全序。代数结构保持复合，所以
$$
m(e,a)=a.
$$
把空块放在右侧同理得到
$$
m(a,e)=a.
$$
因此 $e$ 是双侧单位。$\square$

**命题 F.24.** 在含 arity $0$ 的 $\operatorname{Com}$-代数中，arity $0$ 的唯一运算给出交换乘法的单位。

**证明.** 与命题 F.23 相同，但不需要区分空块在左或右，因为 $\operatorname{Com}(2)$ 的 $\Sigma_2$ 作用平凡，二者由等变性相同。operad 单位关系给出 $ea=a$，交换性给出 $ae=a$。$\square$

**警告 F.25.** 若删去 arity $0$，则 $\operatorname{Ass}_{>0}$-代数编码非必含单位的结合代数，$\operatorname{Com}_{>0}$-代数编码非必含单位的交换代数。把含单位和非含单位版本混用，会导致自由代数、bar construction 和 reduced Koszul duality 中的增广理想定义不一致。

## F.8 自由代数的三个基本例子

**命题 F.26.** $R[\operatorname{Ass}]$ 上的自由代数为张量代数：
$$
F_{\operatorname{Ass}}(V)=T(V)=\bigoplus_{n\ge0}V^{\otimes n}.
$$

**证明.** 命题 F.6 已说明 $R[\operatorname{Ass}]$ 的 Schur functor 是 $T$，命题 F.7 说明其 monad 乘法是词串接。张量代数 $T(V)$ 带串接乘法。给定含单位结合代数 $A$ 和线性映射 $f:V\to A$，定义
$$
\widetilde f(v_1\otimes\cdots\otimes v_n)=f(v_1)\cdots f(v_n),
$$
长度 $0$ 张量 $1_R$ 送到 $1_A$。结合律保证该公式给出代数同态；任何代数同态 $T(V)\to A$ 在长度 $1$ 部分必须等于 $f$，且在所有词上由乘法保持性强制为上述公式。故泛性质成立。$\square$

**命题 F.27.** $R[\operatorname{Com}]$ 上的自由代数为对称代数：
$$
F_{\operatorname{Com}}(V)=\operatorname{Sym}_R(V).
$$

**证明.** 命题 F.10 给出 Schur functor 的底层公式。对称代数的乘法由张量连接后取 coinvariants 定义：
$$
(V^{\otimes p})_{\Sigma_p}\otimes (V^{\otimes q})_{\Sigma_q}
\to
(V^{\otimes(p+q)})_{\Sigma_{p+q}}.
$$
这一定义良好，因为在前 $p$ 个因子或后 $q$ 个因子内部置换，不改变目标 coinvariants 中的类。给定交换含单位 $R$-代数 $A$ 和线性映射 $f:V\to A$，公式
$$
v_1\cdots v_n\longmapsto f(v_1)\cdots f(v_n)
$$
因 $A$ 的交换律而通过 coinvariants，因 $A$ 的结合律和单位律而成为代数同态。唯一性由代数同态必须保持乘法和单位推出。$\square$

**外部输入定理 F.28.** 在适当底环条件下，自由 Lie 代数可由张量代数中的 Lie words 或 primitive 元素模型描述。该模型的证明依赖 PBW、Shirshov-Witt 或 Lyndon basis 技术，本书不把它作为基础证明链的一部分。

## F.9 小结表

| operad | arity 数据 | 代数 | 自由代数 |
| --- | --- | --- | --- |
| $\operatorname{Ass}$ | 有限集上的全序 | 含单位结合代数 | 张量代数 |
| $\operatorname{Com}$ | 单点对称序列 | 含单位交换代数 | 对称代数 |
| $\operatorname{Lie}_R$ | 二元反对称生成元加 Jacobi 商 | Lie 型代数 | 外部输入模型 |
| $\operatorname{Pois}_R$ | 交换乘法加 Lie 括号加 Leibniz | Poisson 代数 | 由生成元关系给出 |
| $\operatorname{End}_V$ | $\operatorname{Hom}_R(V^{\otimes S},V)$ | 作为分类对象 | 不适用 |

本附录的作用是把经典例子从“名称列表”升级为可验证对象。后续正文若引用这些例子，应优先引用本附录中已经证明的部分；若引用 PBW、Cohen-May 或 $E_n$ 同调结论，应继续标为外部输入定理。

