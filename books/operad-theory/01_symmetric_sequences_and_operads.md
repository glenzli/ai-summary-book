# 第一章：对称序列、代入乘积与 operad

## 本章目标

本章给出集合值对称 operad 的基础定义。核心路线是：

1. 把“带对称群作用的 arity 族”改写为有限集群胚上的函子。
2. 用有限集分块定义对称序列的代入乘积。
3. 把 operad 定义为代入乘积下的幺半对象。
4. 构造 endomorphism operad，并由此定义 operad 的代数。
5. 验证两个基本例子：交换 operad 和结合 operad。

## 依赖前置知识

需要有限集、分块、笛卡尔积、函子、自然变换和幺半范畴的定义。

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
反过来，若给定集合族 $X_n$ 及 $\Sigma_n$ 左作用，则在选定有限集骨架 $[n]$ 后可重建一个对称序列，重建只在等价意义下唯一。

**证明.** 第一部分由函子性给出：恒等双射作用为恒等函数，复合双射作用等于函数复合。反向构造需要对每个有限集 $S$ 选择一个双射 $[|S|]\cong S$，再用 $\Sigma_{|S|}$ 作用检查不同选择给出同构的结果。由于选择不是典范的，有限集口径比 arity 口径更适合作为定义。$\square$

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

**定义 1.5.** 设 $X,Y\in\operatorname{SymSeq}_{\mathcal U}$。定义对称序列 $X\circ Y$ 如下：对有限集 $S$，
$$
(X\circ Y)(S)
=
\coprod_{\pi\in\operatorname{Part}(S)}
X(\operatorname{Bl}(\pi))\times
\prod_{B\in\operatorname{Bl}(\pi)}Y(B),
$$
其中 $\operatorname{Part}(S)$ 是 $S$ 的所有分块构成的集合。

若 $\varphi:S\to T$ 是双射，分块 $\pi$ 被送到分块 $\varphi_\*\pi$，其块为 $\varphi(B)$。于是 $\varphi$ 诱导双射
$$
\operatorname{Bl}(\pi)\to\operatorname{Bl}(\varphi_\*\pi),\qquad B\mapsto\varphi(B),
$$
并且每个限制 $\varphi|_B:B\to\varphi(B)$ 诱导
$$
Y(B)\to Y(\varphi(B)).
$$
这些映射合起来定义 $(X\circ Y)(\varphi):(X\circ Y)(S)\to(X\circ Y)(T)$。

**命题 1.6.** 定义 1.5 的规则定义了函子
$$
X\circ Y:\mathbf B_{\mathcal U}\to\mathbf{Set}_{\mathcal U}.
$$

**证明.** 恒等双射保持每个分块和每个块不变，因此诱导恒等函数。若 $S\xrightarrow{\varphi}T\xrightarrow{\psi}U$ 是双射，则 $(\psi\circ\varphi)_*\pi=\psi_*(\varphi_*\pi)$，块集合上的双射和块内限制也按函数复合相容。由 $X$ 与 $Y$ 的函子性，先作用 $\varphi$ 再作用 $\psi$ 与作用 $\psi\circ\varphi$ 给出同一函数。$\square$

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

**证明.** 先构造结合约束。对有限集 $S$，一个元素
$$
((x;(y_B)_{B\in\pi});(z_C)_{C\in\rho_B})
\in ((X\circ Y)\circ Z)(S)
$$
按定义等价于以下数据：

- $S$ 的一个粗分块 $\pi$；
- 每个粗块 $B$ 的一个细分块 $\rho_B$；
- 外层元素 $x\in X(\operatorname{Bl}(\pi))$；
- 中层元素 $y_B\in Y(\operatorname{Bl}(\rho_B))$；
- 底层元素 $z_C\in Z(C)$。

由这些数据构造 $S$ 的分块
$$
\rho=\coprod_{B\in\operatorname{Bl}(\pi)}\rho_B,
$$
其块集合为所有细块 $C$。同时，$\operatorname{Bl}(\rho)$ 被分成子集
$$
\operatorname{Bl}(\rho_B)\subseteq\operatorname{Bl}(\rho),
\qquad B\in\operatorname{Bl}(\pi),
$$
并且这些子集的集合与 $\operatorname{Bl}(\pi)$ 典范同构。于是同一组数据也等价于
$$
(X\circ(Y\circ Z))(S)
$$
的元素：外层仍为 $x$，每个粗块 $B$ 上的内层元素为
$$
(y_B;(z_C)_{C\in\operatorname{Bl}(\rho_B)})
\in (Y\circ Z)(B).
$$
这给出函数
$$
a_{X,Y,Z,S}:((X\circ Y)\circ Z)(S)\to (X\circ(Y\circ Z))(S).
$$
反函数从右侧数据中取出 $S$ 的细分块 $\rho$，再取出 $\operatorname{Bl}(\rho)$ 的分块；后者把细块集合分成若干族，每一族的并集给出 $S$ 的一个粗块。两种构造互逆，因此 $a_{X,Y,Z,S}$ 是双射。若 $\varphi:S\to T$ 是双射，$\varphi$ 把粗分块、细分块、块集合上的分块和所有限制双射同时推前；这两个构造只使用这些推前操作。因此 $a_{X,Y,Z,S}$ 关于 $S$ 自然，并且关于 $X,Y,Z$ 的自然性来自函子值上直接作用分量。于是得到自然同构
$$
a_{X,Y,Z}:(X\circ Y)\circ Z\cong X\circ(Y\circ Z).
$$

再构造单位约束。对任意有限集 $S$，
$$
(X\circ I)(S)
=
\coprod_{\pi\in\operatorname{Part}(S)}
X(\operatorname{Bl}(\pi))\times\prod_{B\in\operatorname{Bl}(\pi)}I(B).
$$
乘积项非空当且仅当每个块 $B$ 都是单点集；此时 $\pi$ 是离散分块，且 $\operatorname{Bl}(\pi)\cong S$ 由单点块 $B=\{s\}$ 送到 $s$。所以得到自然双射 $(X\circ I)(S)\cong X(S)$。同理，
$$
(I\circ X)(S)
=
\coprod_{\pi\in\operatorname{Part}(S)}
I(\operatorname{Bl}(\pi))\times\prod_{B\in\operatorname{Bl}(\pi)}X(B)
$$
非空当且仅当 $\operatorname{Bl}(\pi)$ 是单点集，即 $\pi$ 是单块分块 $\{S\}$；于是得到自然双射 $(I\circ X)(S)\cong X(S)$。这些双射组成自然同构
$$
X\circ I\cong X,\qquad I\circ X\cong X.
$$

最后检查相干性。五边形图的任一边都把四层分块数据
$$
S\longrightarrow\text{第 1 层块}\longrightarrow\text{第 2 层块}
\longrightarrow\text{第 3 层块}
$$
送到同一个完全展开的数据：所有最细块、最细块集合上的逐层分块，以及 $W,Z,Y,X$ 四个对称序列在相应层上的元素。因为两条路径只是以不同括号顺序执行同一个“拉平多层分块”操作，所以五边形交换。三角形图同样化为插入单点分块和随后删除单点分块；两条路径都把二层数据送回同一二层数据。因此 $\operatorname{SymSeq}_{\mathcal U}$ 连同 $\circ$ 与 $I$ 是幺半范畴。$\square$

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

**展开 1.10.** 对有限集 $S$，一个分块 $\pi$，一个外层运算
$$
o\in\mathcal O(\operatorname{Bl}(\pi))
$$
和每个块上的内层运算
$$
o_B\in\mathcal O(B),
$$
乘法 $\mu$ 给出代入结果
$$
\mu_\pi(o;(o_B)_{B\in\operatorname{Bl}(\pi)})\in\mathcal O(S).
$$
operad 的结合律断言：若 $S$ 先分成粗块，再把每个粗块细分，则“先在细块内代入再在粗块间代入”与“一次性按两层分块代入”结果相同。单位律断言 arity $1$ 的单位运算对代入不起作用。

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

**证明.** 设 $\pi$ 是 $S$ 的分块。给定
$$
f:X^{\operatorname{Bl}(\pi)}\to X,
\qquad
g_B:X^B\to X\quad(B\in\operatorname{Bl}(\pi)),
$$
定义
$$
h:X^S\to X
$$
如下。对 $a:S\to X$，令 $a|_B:B\to X$ 为限制，并令
$$
h(a)=f\big(B\mapsto g_B(a|_B)\big).
$$
这定义了代入映射
$$
\operatorname{End}_X(\operatorname{Bl}(\pi))
\times
\prod_B\operatorname{End}_X(B)
\to
\operatorname{End}_X(S).
$$

单位由恒等函数 $\operatorname{id}_X:X\to X$ 给出，视为 $\operatorname{End}_X(\{*\})$ 的元素。结合律来自函数复合的结合律：三层分块时，无论先对最内层块求值还是先把分块拉平，最终对 $a:S\to X$ 计算得到的元素都是同一个逐层表达式。自然性来自限制映射与双射重标号的相容。$\square$

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
其中空乘积定义为 $e$。交换律保证该表达式与 $S$ 的枚举无关，结合律和单位律保证它与分块代入相容。因此得到 operad morphism
$$
\operatorname{Com}\to\operatorname{End}_X.
$$
两种构造互逆。$\square$

**定义 1.18.** 结合 operad $\operatorname{Ass}$ 定义如下。对有限集 $S$，令 $\operatorname{Ass}(S)$ 为 $S$ 上全序关系的集合。若 $\varphi:S\to T$ 是双射，则 $\operatorname{Ass}(\varphi)$ 把 $S$ 上全序推到 $T$ 上全序。

代入定义为：给定块集合 $\operatorname{Bl}(\pi)$ 上的全序，以及每个块 $B$ 上的全序，先按外层全序排列块，再在每个块内按内层全序排列元素，由此得到 $S$ 上的全序。单位是单点集上的唯一全序。

**命题 1.19.** $\operatorname{Ass}$ 是 operad，且集合值 $\operatorname{Ass}$-代数等价于幺半群。

**证明.** operad 结合律是全序字的代入结合律：三层分块时，最终得到的是同一个按外层、中层、内层词典顺序排列的元素列表。单位律由单点全序给出。

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

**证明.** 从有限集口径到 arity 口径，取 $\mathcal O(n)=\mathcal O([n])$，对称群作用由双射函子性给出，代入映射来自分块
$$
[k_1+\cdots+k_n]=[k_1]\coprod\cdots\coprod[k_n]
$$
及块集合与 $[n]$ 的指定识别。单位来自 $\eta:I\to\mathcal O$ 在单点集上的分量。

反向构造需要选择每个有限集与标准有限集 $[n]$ 的双射；对称群等变公理保证不同选择给出一致的重标号。operad 的单位、结合和等变公理正是有限集代入公理在标准分块上的表达。$\square$

## 本章小结

本章建立了本书的基础定义：对称序列是有限集群胚上的集合值函子；代入乘积由有限分块给出；operad 是代入乘积下的幺半对象。endomorphism operad 把抽象运算解释为集合上的具体多元运算，因此 operad 代数就是到 endomorphism operad 的 morphism。交换 operad 和结合 operad 分别编码交换幺半群和幺半群。

## 练习

**练习 1.1.** 直接用分块公式证明 $I\circ X\cong X$，并写出自然同构在有限集 $S$ 上的具体函数。

**练习 1.2.** 对集合 $X$，写出 $\operatorname{End}_X([0])$、$\operatorname{End}_X([1])$、$\operatorname{End}_X([2])$ 的含义。

**练习 1.3.** 验证 $\operatorname{Com}$ 中 arity $0$ 的唯一元素在代数中确实给出幺元，而不是任意常数。

**练习 1.4.** 设 $\operatorname{Ass}(S)$ 为 $S$ 上全序集合。对一个含四个元素的有限集，取一个两层分块，手写两种加括号代入得到的全序，并验证它们相同。

**练习 1.5.** 把命题 1.20 中的 $\Sigma_n$ 作用改成右作用约定，写出代入等变公式中需要出现的块置换。
