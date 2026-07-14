# 第一章：对称序列、代入乘积与 operad

只给出集合族 $X(0),X(1),X(2),\ldots$ 还不能表达多元运算怎样换名，更不能说明把若干运算代入另一个运算后为何与分步代入一致。真正需要记录的是有限输入集的双射作用，以及一个有限集映射的每条纤维如何承载内层运算。特别地，空纤维迫使我们正面处理 arity $0$，不能把“分块”偷偷理解成满射。本章从有限集群胚构造代入乘积，再以幺半对象定义 operad；随后用 endomorphism operad 把抽象代入落实为集合上的函数复合。有限集、笛卡尔积、函子、自然变换和幺半范畴的基本语言足以跟随全部构造。

## 1.1 有限集群胚与对称序列

**定义 1.1.** 令 $\mathbf B_{\mathcal U}$ 表示如下群胚：

- 对象是 $\mathcal U$-小有限集；
- 态射是双射；
- 复合是函数复合。

一个集合值对称序列（symmetric sequence in sets）是一个函子
$$
X:\mathbf B_{\mathcal U}\to\mathbf{Set}_{\mathcal U}.
$$
对有限集 $S$，集合 $X(S)$ 的元素称为以 $S$ 为输入集合的抽象运算。

**例 1.2.** 若 $X:\mathbf B_{\mathcal U}\to\mathbf{Set}_{\mathcal U}$ 是对称序列，则对每个 $n\ge 0$，集合 $X([n])$ 带有 $\Sigma_n=\operatorname{Aut}([n])$ 的左作用：
$$
\sigma\cdot x=X(\sigma)(x).
$$
反过来，若给定集合族 $X_n$ 及 $\Sigma_n$ 左作用，则可重建一个对称序列。对 $|S|=n$ 定义
$$
\widetilde X(S)
=
\operatorname{Bij}([n],S)\times_{\Sigma_n}X_n,
$$
其中 $\operatorname{Bij}([n],S)$ 的右作用为 $b\cdot\sigma=b\circ\sigma$，平衡关系为
$$
(b\circ\sigma,x)\sim(b,\sigma\cdot x).
$$
若 $\varphi:S\to T$ 是双射，则
$$
\widetilde X(\varphi)[b,x]=[\varphi\circ b,x].
$$

**证明.** 左作用公理由 $X(\operatorname{id})=\operatorname{id}$ 和
$X(\sigma\tau)=X(\sigma)X(\tau)$ 给出。反向公式良定义，因为
$$
[\varphi b\sigma,x]=[\varphi b,\sigma x].
$$
它保持恒等态射与复合，故定义函子。若最初给定函子 $X$，则对任意 $|S|=n$，映射
$$
[b,x]\longmapsto X(b)(x)
$$
给出自然同构
$$
\operatorname{Bij}([n],S)\times_{\Sigma_n}X([n])\xrightarrow{\ \cong\ }X(S),
$$
因为 $X(b\sigma)(x)=X(b)(X(\sigma)(x))$，且任取一个 $b:[n]\to S$ 即可写出逆映射 $y\mapsto[b,X(b^{-1})(y)]$；平衡关系说明该逆映射与 $b$ 的选择无关。若最初给定作用族，则对 $S=[n]$，映射
$$
[b,x]\longmapsto b\cdot x
$$
给出 $\widetilde X([n])\cong X_n$。于是从函子取骨架值再重建，或从作用族重建再取骨架值，都自然同构于恒等构造。若改用右作用，本书固定
$$
x\cdot\sigma=\sigma^{-1}\cdot x,
$$
不能在后续 coinvariants 公式中省略这个逆元。$\square$

**定义 1.3.** 记
$$
\operatorname{SymSeq}_{\mathcal U}
=
\operatorname{Fun}(\mathbf B_{\mathcal U},\mathbf{Set}_{\mathcal U})
$$
为集合值对称序列范畴。态射是自然变换。

## 1.2 分块与代入乘积

**定义 1.4.** 设 $S$ 为有限集。$S$ 的一个分块 $\pi$ 是有限个非空子集组成的集合 $\operatorname{Bl}(\pi)$，满足
$$
S=\coprod_{B\in\operatorname{Bl}(\pi)}B.
$$
块集合 $\operatorname{Bl}(\pi)$ 本身是有限集。

**定义 1.4.1（纤维分解群胚）.** 令 $\operatorname{Fib}(S)$ 为如下群胚：

- 对象是函数 $f:S\to T$，其中 $T$ 是 $\mathcal U$-小有限集；
- 从 $f:S\to T$ 到 $f':S\to T'$ 的态射是满足 $f'=u\circ f$ 的双射 $u:T\to T'$。

这里不要求 $f$ 满射。若 $t\notin f(S)$，则 $f^{-1}(t)=\varnothing$；这样的空纤维记录将由 nullary operation 填入的外层输入槽。

$\operatorname{Fib}(S)$ 是 essentially $\mathcal U$-small：目标限制为标准有限集 $[k]$（$k\ge0$）所得满子群胚是其骨架。下文所有以 $\operatorname{Fib}(S)$ 为指标的 colimit 都在这个骨架上计算；任意有限目标 $T$ 的写法只用于保持公式无坐标。因此该 colimit 仍取值于 $\mathbf{Set}_{\mathcal U}$。

**定义 1.5.** 设 $X,Y\in\operatorname{SymSeq}_{\mathcal U}$。对有限集 $S$，定义
$$
(X\circ Y)(S)
=
\operatorname*{colim}_{(f:S\to T)\in\operatorname{Fib}(S)}
X(T)\times\prod_{t\in T}Y(f^{-1}(t)).
$$
具体地，一个元素由四元组
$$
(T,f,x,(y_t)_{t\in T})
$$
表示，其中 $x\in X(T)$ 且 $y_t\in Y(f^{-1}(t))$。若 $u:T\to T'$ 是双射，则施加关系
$$
(T,f,x,(y_t))
\sim
(T',uf,X(u)x,(y'_{t'})),
\qquad y'_{u(t)}=y_t.
$$

若 $\varphi:S\to S'$ 是双射，则把上述代表元送到
$$
\left(T,f\varphi^{-1},x,
\big(Y(\varphi|_{f^{-1}(t)})(y_t)\big)_{t\in T}\right).
$$
因为 $(f\varphi^{-1})^{-1}(t)=\varphi(f^{-1}(t))$，每个 $Y$-分量都有正确类型。

**说明 1.5.1（分块公式的适用边界）.** 若 $Y(\varnothing)=\varnothing$，含空纤维的项为空，只有满射 $f:S\twoheadrightarrow T$ 有贡献。满射的非空纤维给出 $S$ 的分块，因而有自然同构
$$
(X\circ Y)(S)
\cong
\coprod_{\pi\in\operatorname{Part}(S)}
X(\operatorname{Bl}(\pi))\times
\prod_{B\in\operatorname{Bl}(\pi)}Y(B).
$$
若 $Y(\varnothing)\ne\varnothing$，该分块公式不成立。

**命题 1.6.** 定义 1.5 的规则定义了函子
$$
X\circ Y:\mathbf B_{\mathcal U}\to\mathbf{Set}_{\mathcal U}.
$$

**证明.** 若代表元沿目标双射 $u:T\to T'$ 改写，再沿 $\varphi$ 重标号源，则所得代表元仍由同一个 $u$ 联系；故定义 1.5 的源重标号尊重 colimit 关系。恒等双射给出恒等函数。对双射
$S\xrightarrow{\varphi}S'\xrightarrow{\psi}S''$，有
$$
f\varphi^{-1}\psi^{-1}=f(\psi\varphi)^{-1},
$$
且纤维限制满足
$$
\psi|_{\varphi(f^{-1}(t))}\circ\varphi|_{f^{-1}(t)}
=(\psi\varphi)|_{f^{-1}(t)}.
$$
由 $Y$ 的函子性，两次重标号等于沿 $\psi\varphi$ 一次重标号。因此 $X\circ Y$ 是 $\mathbf B_{\mathcal U}$ 上的函子。$\square$

**定义 1.7.** 单位对称序列 $I$ 定义为
$$
I(S)=
\begin{cases}
\{*\}, & |S|=1,\\
\varnothing, & |S|\ne 1.
\end{cases}
$$
对双射的作用由唯一可能的函数给出。

**命题 1.8.** 对称序列范畴 $\operatorname{SymSeq}_{\mathcal U}$ 连同 $\circ$ 和 $I$ 构成幺半范畴。

**证明.** 对有限集 $S$，考虑二层有限集映射
$$
S\xrightarrow{g}U\xrightarrow{p}T
$$
及装饰
$$
x\in X(T),\qquad
y_t\in Y(p^{-1}(t)),\qquad
z_u\in Z(g^{-1}(u)).
$$
同时对 $U,T$ 的双射取商，双射必须与 $g,p$ 交换。记所得集合为 $D_{X,Y,Z}(S)$。

展开 $((X\circ Y)\circ Z)(S)$ 时，先得到 $g:S\to U$ 和各 $z_u$，再用 $p:U\to T$ 表示 $(X\circ Y)(U)$；故得到自然双射
$$
((X\circ Y)\circ Z)(S)\cong D_{X,Y,Z}(S).
$$
展开 $(X\circ(Y\circ Z))(S)$ 时，先得到复合 $pg:S\to T$；对每个 $t\in T$，限制
$$
g_t:(pg)^{-1}(t)\longrightarrow p^{-1}(t)
$$
表示该纤维上的 $(Y\circ Z)$-元素。反过来，各 $g_t$ 的目标不交并成 $U$，并给出 $p:U\to T$。因此也有自然双射
$$
(X\circ(Y\circ Z))(S)\cong D_{X,Y,Z}(S).
$$
两式复合给出结合约束；构造只使用复合、限制和不交并，所以关于 $S,X,Y,Z$ 自然。

对右单位，$(X\circ I)(S)$ 中只有每个纤维都是单点集的 $f:S\to T$ 有贡献，即 $f$ 必须是双射。映射
$$
[T,f,x]\longmapsto X(f^{-1})(x)
$$
给出 $(X\circ I)(S)\cong X(S)$。对左单位，$(I\circ X)(S)$ 中只有 $|T|=1$ 的目标有贡献；唯一函数 $S\to T$ 的纤维是整个 $S$，故得到 $(I\circ X)(S)\cong X(S)$。当 $S=\varnothing$ 时这两段论证仍成立：右单位使用双射 $\varnothing\to\varnothing$，左单位使用函数 $\varnothing\to\{*\}$，所以没有遗漏 arity $0$。

最后检查相干性。四个对称序列的任一加括号方式都展开为三层映射
$$
S\longrightarrow U_1\longrightarrow U_2\longrightarrow U_3
$$
及各层纤维上的装饰，并对逐层双射取商。五边形的两条路径在这个共同表示上都是恒等映射，故五边形交换。三角形的两条路径都插入再删除由单位强制的双射层或单点目标层，因而也在共同表示上相同。故 $\operatorname{SymSeq}_{\mathcal U}$ 连同 $\circ$ 与 $I$ 是幺半范畴。$\square$

## 1.3 Operad 的定义

**定义 1.9.** 一个集合值对称 operad 是幺半范畴
$$
\big(\operatorname{SymSeq}_{\mathcal U},\circ,I\big)
$$
中的幺半对象。也就是说，它由对称序列 $\mathcal O$、乘法自然变换
$$
\mu:\mathcal O\circ\mathcal O\to\mathcal O
$$
和单位自然变换
$$
\eta:I\to\mathcal O
$$
组成，并满足结合图和单位图交换。

**展开 1.10.** 对有限集映射 $f:S\to T$，一个外层运算
$$
o\in\mathcal O(T)
$$
和每个纤维上的内层运算
$$
o_t\in\mathcal O(f^{-1}(t)),
$$
乘法 $\mu$ 给出代入结果
$$
\mu_f(o;(o_t)_{t\in T})\in\mathcal O(S).
$$
这里允许 $f^{-1}(t)=\varnothing$，此时 $o_t$ 是 nullary operation。Operad 的结合律断言：对可复合映射 $S\to U\to T$，按任一层次顺序代入结果相同。单位律断言插入 arity $1$ 的单位运算不改变结果。

**定义 1.11.** operad morphism $F:\mathcal O\to\mathcal P$ 是对称序列态射，使得下列两个条件成立：
$$
F\circ\mu_{\mathcal O}
=
\mu_{\mathcal P}\circ(F\circ F),
\qquad
F\circ\eta_{\mathcal O}=\eta_{\mathcal P}.
$$
这里第一个等式是在自然变换 $\mathcal O\circ\mathcal O\to\mathcal P$ 中的等式。

## 1.4 Endomorphism operad

**定义 1.12.** 设 $X$ 为 $\mathcal U$-小集合。定义对称序列 $\operatorname{End}_X$：
$$
\operatorname{End}_X(S)=\mathbf{Set}_{\mathcal U}(X^S,X),
$$
其中 $X^S$ 是从 $S$ 到 $X$ 的函数集。若 $\varphi:S\to T$ 是双射，则
$$
\operatorname{End}_X(\varphi):\operatorname{End}_X(S)\to\operatorname{End}_X(T)
$$
把 $f:X^S\to X$ 送到
$$
X^T\xrightarrow{\varphi^\*}X^S\xrightarrow{f}X,
$$
其中 $\varphi^\*(a)=a\circ\varphi$。

**命题 1.13.** $\operatorname{End}_X$ 具有自然的 operad 结构。

**证明.** 设 $f:S\to T$ 是有限集映射。给定
$$
F:X^T\to X,
\qquad
G_t:X^{f^{-1}(t)}\to X\quad(t\in T),
$$
定义
$$
h:X^S\to X
$$
如下。对 $a:S\to X$，令
$$
h(a)=F\big(t\mapsto G_t(a|_{f^{-1}(t)})\big).
$$
这定义了代入映射
$$
\operatorname{End}_X(T)
\times
\prod_{t\in T}\operatorname{End}_X(f^{-1}(t))
\to
\operatorname{End}_X(S).
$$

若某个纤维为空，则 $G_t:X^\varnothing\to X$ 正是一个常量，所以上式也覆盖 nullary substitution。

单位由恒等函数 $\operatorname{id}_X:X\to X$ 给出，视为 $\operatorname{End}_X(\{*\})$ 的元素。结合律来自函数复合的结合律：对 $S\to U\to T$，两种代入顺序都先在最小纤维上求值，再在 $U\to T$ 的纤维上求值，最后应用 $X^T\to X$。自然性来自限制映射与双射重标号的相容。$\square$

**定义 1.14.** 设 $\mathcal O$ 是 operad。一个集合值 $\mathcal O$-代数是集合 $X$ 连同 operad morphism
$$
\alpha:\mathcal O\to\operatorname{End}_X.
$$

等价地，它给出对每个有限集 $S$ 的动作映射
$$
\mathcal O(S)\times X^S\to X,
\qquad
(o,a)\mapsto o\cdot a,
$$
并且这些映射对双射重标号、operad 代入和单位运算相容。

**命题 1.15.** 定义 1.14 中的 operad morphism 口径与动作映射口径等价。

**证明.** 给定 $\alpha:\mathcal O\to\operatorname{End}_X$，令
$$
o\cdot a=\alpha_S(o)(a).
$$
自然性说明双射重标号相容，保持乘法说明代入相容，保持单位说明单位运算作用为恒等。

反过来，给定相容的动作映射，对每个 $o\in\mathcal O(S)$ 定义函数
$$
\alpha_S(o):X^S\to X,\qquad a\mapsto o\cdot a.
$$
重标号相容性给出 $\alpha$ 的自然性；代入相容性和单位相容性给出 $\alpha$ 是 operad morphism。两个构造互逆。$\square$

## 1.5 交换 operad 与结合 operad

**定义 1.16.** 交换 operad $\operatorname{Com}$ 定义为
$$
\operatorname{Com}(S)=\{*\}
$$
对每个有限集 $S$ 成立。所有重标号作用都是唯一函数，所有代入映射也都是唯一函数。

**命题 1.17.** 集合值 $\operatorname{Com}$-代数等价于交换幺半群。

**证明.** 设 $X$ 是 $\operatorname{Com}$-代数。因为 $\operatorname{Com}([2])$ 是单点集，得到一个二元运算
$$
m:X\times X\to X.
$$
因为 $\operatorname{Com}(\varnothing)$ 是单点集，得到一个元素
$$
e:X^0\to X,
$$
即 $X$ 的一个指定元素 $e$。operad 单位和代入相容性给出
$$
m(e,x)=x,\qquad m(x,e)=x,
$$
以及
$$
m(m(x,y),z)=m(x,m(y,z)).
$$
双射 $(1\ 2):[2]\to[2]$ 的自然性给出
$$
m(x,y)=m(y,x).
$$
因此 $X$ 是交换幺半群。

反过来，若 $(X,m,e)$ 是交换幺半群，则对任意有限集 $S$ 定义
$$
\alpha_S(*):X^S\to X,\qquad
(x_s)_{s\in S}\mapsto \prod_{s\in S}x_s,
$$
其中空乘积定义为 $e$。交换律保证该表达式与 $S$ 的枚举无关。对任意 $f:S\to T$，先在各纤维相乘再对 $T$ 相乘，与直接对 $S$ 相乘相同；空纤维贡献 $e$。结合律和单位律因而给出全部代入相容性。因此得到 operad morphism
$$
\operatorname{Com}\to\operatorname{End}_X.
$$
两种构造互逆。$\square$

**定义 1.18.** 结合 operad $\operatorname{Ass}$ 定义如下。对有限集 $S$，令 $\operatorname{Ass}(S)$ 为 $S$ 上全序关系的集合。若 $\varphi:S\to T$ 是双射，则 $\operatorname{Ass}(\varphi)$ 把 $S$ 上全序推到 $T$ 上全序。

对映射 $f:S\to T$，给定 $T$ 上的外层全序以及每个纤维 $f^{-1}(t)$ 上的内层全序，定义 $S$ 上的字典序：先比较 $f(s),f(s')$ 的外层顺序；若二者相等，再在共同纤维中比较 $s,s'$。空纤维有唯一全序且不贡献元素。单位是单点集上的唯一全序。

**命题 1.19.** $\operatorname{Ass}$ 是 operad，且集合值 $\operatorname{Ass}$-代数等价于幺半群。

**证明.** 对可复合映射 $S\xrightarrow{g}U\xrightarrow{p}T$，任意两个元素先按其 $T$-像比较；像相同时再按其 $U$-像比较；两者仍相同时才在 $g$-纤维内比较。两种加括号方式都给出这一字典序，故结合律成立。插入恒等映射或单点目标不改变比较规则，故单位律成立。

若 $X$ 是 $\operatorname{Ass}$-代数，取 $[2]$ 上顺序 $1<2$ 得到二元运算 $m(x,y)$，取空集上的唯一顺序得到元素 $e$。对三个元素的两种分块
$$
\{\{1,2\},\{3\}\},\qquad \{\{1\},\{2,3\}\}
$$
代入相容性给出结合律；单点顺序给出单位律。因此 $X$ 是幺半群。

反过来，若 $(X,m,e)$ 是幺半群，则对 $S$ 上全序 $s_1<\cdots<s_n$ 定义
$$
\alpha_S(s_1<\cdots<s_n)(x_s)_{s\in S}
=
x_{s_1}\cdots x_{s_n},
$$
空序列的值定义为 $e$。幺半群结合律和单位律保证该构造与 operad 代入相容，双射重标号相容由推前全序的定义给出。于是得到 $\operatorname{Ass}$-代数。两种构造互逆。$\square$

## 1.6 与 arity 写法的关系

**命题 1.20.** 有限集口径的集合值对称 operad 等价于如下 arity 数据：

- 集合族 $\mathcal O(n)$，$n\ge 0$；
- 每个 $\mathcal O(n)$ 上的 $\Sigma_n$ 作用；
- 单位元素 $\mathbf 1\in\mathcal O(1)$；
- 对所有 $n,k_1,\ldots,k_n\ge0$ 的代入映射
  $$
  \mathcal O(n)\times\mathcal O(k_1)\times\cdots\times\mathcal O(k_n)
  \to
  \mathcal O(k_1+\cdots+k_n),
  $$
  满足单位、结合和对称群等变公理。

**证明.** 从有限集口径到 arity 口径，取 $\mathcal O(n)=\mathcal O([n])$，对称群作用由双射函子性给出。代入映射来自函数
$$
[k_1+\cdots+k_n]\longrightarrow[n]
$$
其第 $i$ 个纤维是相应的连续 $k_i$-元素子集；允许 $k_i=0$，此时该纤维为空。单位来自 $\eta:I\to\mathcal O$ 在单点集上的分量。

反向构造可用例 1.2 的平衡积完成，不需要逐个有限集任意选基。对称群等变公理保证代入映射通过平衡关系；单位和结合公理正是可复合有限集映射的单位与结合公理。两个方向在骨架限制下互逆。$\square$

## 1.7 从运算族到可代入对象

有限集口径同时解决了三个容易互相缠绕的问题：双射给出输入重标号，任意有限集映射的纤维给出内层 arity，映射复合则给出代入结合律。空纤维保留零元运算，因而 $\operatorname{Com}$ 与 $\operatorname{Ass}$ 的代数分别确实带幺元，而不只是无幺乘法。Endomorphism operad 又说明这些抽象数据最终作用在何处。下一章不再改变 operad 本身，而要研究固定 $\mathcal O$ 后全部 $\mathcal O$-代数如何组成范畴，以及一个集合怎样自由地产生这样的代数。

## 练习

**练习 1.1.** 直接用定义 1.5 证明 $I\circ X\cong X$ 和 $X\circ I\cong X$，并分别检查 $S=\varnothing$。

**练习 1.2.** 对集合 $X$，写出 $\operatorname{End}_X([0])$、$\operatorname{End}_X([1])$、$\operatorname{End}_X([2])$ 的含义。

**练习 1.3.** 验证 $\operatorname{Com}$ 中 arity $0$ 的唯一元素在代数中确实给出幺元，而不是任意常数。

**练习 1.4.** 设 $\operatorname{Ass}(S)$ 为 $S$ 上全序集合。对一个含四个元素的有限集，取一个两层分块，手写两种加括号代入得到的全序，并验证它们相同。

**练习 1.5.** 把命题 1.20 中的 $\Sigma_n$ 作用改成右作用约定，写出代入等变公式中需要出现的块置换。
