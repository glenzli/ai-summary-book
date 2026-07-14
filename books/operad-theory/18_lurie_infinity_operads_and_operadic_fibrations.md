# 第十八章：Lurie-style infinity-operads 与 operadic fibrations

一列颜色 $(X_1,\ldots,X_n)$ 可以沿投影只保留某个分量，也可以沿 $\langle n\rangle\to\langle1\rangle$ 参与一个真正的 $n$ 元运算。Lurie 的模型把这两种行为都放在 $N(\mathbf{Fin}_*)$ 上：inert 态射选择或遗忘分量，active 态射把保留的输入组织成输出。关键顺序是先 inert、后 active；它决定哪些边必须有 coCartesian 提升，也决定 category of operators 如何复合。本章从这一分解构造 Lurie-style infinity-operad，逐项核对 ordinary operad 的算子范畴，并保持与 dendroidal inner Kan 模型的区别。跨模型等价仍作为带适用范围的外部输入。

## 18.1 有基点有限集与 inert--active 分解

**定义 18.1.** 令 $\mathbf{Fin}_*$ 为有限有基点集合范畴。其标准对象记为
$$
\langle n\rangle=\{*,1,\ldots,n\},
$$
其中 $*$ 为基点。态射保持基点。

**定义 18.2.** 态射 $f:\langle m\rangle\to\langle n\rangle$ 称为 inert，若对每个 $1\le j\le n$，集合
$$
f^{-1}(j)
$$
恰有一个元素。

态射 $f$ 称为 active，若
$$
f^{-1}(*)=\{*\}.
$$

**例 18.3.** 对每个 $1\le i\le n$，投影
$$
\rho^i:\langle n\rangle\to\langle 1\rangle
$$
由 $\rho^i(i)=1$ 且 $\rho^i(j)=*$ for $j\ne i$ 定义。它是 inert。

**例 18.4.** 唯一的 multiplication map
$$
\mu_n:\langle n\rangle\to\langle 1\rangle,\qquad \mu_n(i)=1
$$
for $1\le i\le n$，是 active。

**命题 18.5（inert--active 分解）.** 每个态射 $f:\langle m\rangle\to\langle n\rangle$ 可分解为先 inert、后 active：
$$
\langle m\rangle\xrightarrow{i_f}\langle k\rangle\xrightarrow{a_f}\langle n\rangle.
$$
此分解在中间对象的唯一保持基点同构意义下唯一。

**证明.** 令
$$
S=\{x\in\{1,\ldots,m\}:f(x)\ne *\},
\qquad k=|S|,
$$
并选择双射 $u:S\xrightarrow{\cong}\{1,\ldots,k\}$。定义
$$
i_f(x)=
\begin{cases}
u(x),&x\in S,\\
*,&x\notin S,
\end{cases}
\qquad i_f(*)=*.
$$
对每个 $1\le y\le k$，有
$$
i_f^{-1}(y)=\{u^{-1}(y)\},
$$
故 $i_f$ inert。再定义
$$
a_f(u(x))=f(x)\quad(x\in S),
\qquad a_f(*)=*.
$$
每个 $u(x)$ 都被送到非基点，所以 $a_f^{-1}(*)=\{*\}$，即 $a_f$ active。逐点可见 $a_fi_f=f$。

若另有分解 $f=a'i'$，其中 $i':\langle m\rangle\to\langle k'\rangle$ inert、$a':\langle k'\rangle\to\langle n\rangle$ active，则
$$
x\in S\quad\Longleftrightarrow\quad i'(x)\ne *.
$$
事实上，$i'(x)=*$ 蕴含 $f(x)=*$；反之若 $i'(x)\ne *$，active 性保证 $a'i'(x)\ne *$。Inert 性又使 $i'$ 在 $S$ 与 $\{1,\ldots,k'\}$ 之间诱导双射。因此存在唯一保持基点双射
$$
\theta:\langle k\rangle\xrightarrow{\cong}\langle k'\rangle,
\qquad \theta(u(x))=i'(x),
$$
满足 $\theta i_f=i'$ 与 $a'\theta=a_f$。这给出所述唯一性。$\square$

**例 18.6.** 设 $f:\langle4\rangle\to\langle2\rangle$ 满足
$$
f(1)=1,\qquad f(2)=*,\qquad f(3)=2,\qquad f(4)=1.
$$
取 $u(1)=1,u(3)=2,u(4)=3$，则分解为
$$
\langle4\rangle\xrightarrow{i_f}\langle3\rangle
\xrightarrow{a_f}\langle2\rangle,
$$
其中 $i_f$ 把 $2$ 送到基点并依次保留 $1,3,4$，而
$$
a_f(1)=1,\qquad a_f(2)=2,\qquad a_f(3)=1.
$$
第一步只遗忘一个分量，第二步才把两个保留分量聚合到第一个输出。

**警告 18.6.1.** 分解顺序依赖基范畴与方差。本书始终在 $\mathbf{Fin}_*$ 本身采用命题 18.5 的“先 inert、后 active”约定，并令 $\rho^i:\langle n\rangle\to\langle1\rangle$ 为 inert。转到 $\mathbf{Fin}_*^{\operatorname{op}}$ 或 co-operadic 记号时不得照抄此顺序。

## 18.2 CoCartesian edges 与 inert transport

**定义 18.7.** 设 $p:X\to S$ 是 simplicial sets 的 inner fibration。边 $e:x\to y$ in $X$ 称为 $p$-coCartesian，若对每个对象 $z\in X$，诱导的 mapping space 方块使
$$
\operatorname{Map}_X(y,z)\to
\operatorname{Map}_X(x,z)\times_{\operatorname{Map}_S(p(x),p(z))}
\operatorname{Map}_S(p(y),p(z))
$$
成为弱等价。

这是 quasi-category 中 coCartesian edge 的模型化定义；完整表述需要使用 slice quasi-categories。

**定义 18.8.** 若 $p:\mathcal O^\otimes\to N(\mathbf{Fin}_*)$ 是 inner fibration，且对每个 inert morphism $i:\langle m\rangle\to\langle k\rangle$ 与每个 $X\in\mathcal O^\otimes_{\langle m\rangle}$，都存在一条以 $X$ 为源的 $p$-coCartesian edge
$$
X\longrightarrow i_!X
$$
lying over $i$，则称 $p$ admits inert coCartesian lifts。对一般 $f=a_fi_f$，此条件只提升第一步 $i_f$；第二步 active 态射 $a_f$ 所在的边编码运算，定义不要求它 coCartesian。

## 18.3 Lurie-style infinity-operad

**定义 18.9.** Lurie-style infinity-operad 是 inner fibration
$$
p:\mathcal O^\otimes\to N(\mathbf{Fin}_*)
$$
满足以下条件：

1. $p$ admits inert coCartesian lifts。
2. 对每个 $n\ge0$，由 inert projections $\rho^i:\langle n\rangle\to\langle1\rangle$ 诱导的函子
   $$
   \mathcal O^\otimes_{\langle n\rangle}\to
   \big(\mathcal O^\otimes_{\langle1\rangle}\big)^n
   $$
   是 equivalence of infinity-categories。
3. 对任意 $X\in\mathcal O^\otimes_{\langle m\rangle}$、$Y\in\mathcal O^\otimes_{\langle n\rangle}$ 和 $\alpha:\langle m\rangle\to\langle n\rangle$，选取 $Y\to Y_i$ lying over $\rho^i$ 的 inert coCartesian edges，则由后复合诱导的自然映射
   $$
   \operatorname{Map}^{\alpha}_{\mathcal O^\otimes}(X,Y)
   \to
   \prod_{i=1}^n
   \operatorname{Map}^{\rho^i\alpha}_{\mathcal O^\otimes}(X,Y_i)
   $$
   是 homotopy equivalence。

这里
$$
\operatorname{Map}^{\alpha}_{\mathcal O^\otimes}(X,Y)
:=
\operatorname{hofib}_{\alpha}\!\left(
\operatorname{Map}_{\mathcal O^\otimes}(X,Y)
\longrightarrow
\operatorname{Map}_{N(\mathbf{Fin}_*)}(\langle m\rangle,\langle n\rangle)
\right)
$$
是 lying over $\alpha$ 的 mapping space 同伦纤维。

**说明 18.10.** 条件 2 说 fiber over $\langle n\rangle$ 是 $n$ 个颜色对象的乘积；当 $n=0$ 时，空乘积是终端 infinity-category，所以 $\mathcal O^\otimes_{\langle0\rangle}$ 可缩。条件 3 说映到 $n$ 个输出分量的 morphism space 可由各分量上的 morphism spaces 重构。命题 18.5 先用 inert 部分选出实际参与运算的输入，再由 active 部分把它们送入输出；特别地，active morphisms over $\langle n\rangle\to\langle1\rangle$ 编码 $n$ 输入运算。

**定义 18.11.** $\mathcal O^\otimes_{\langle1\rangle}$ 称为 $\mathcal O$ 的 underlying infinity-category of colors，记作
$$
\mathcal O.
$$

## 18.4 Symmetric monoidal infinity-categories

**定义 18.12.** Symmetric monoidal infinity-category 是 Lurie-style infinity-operad
$$
p:\mathcal C^\otimes\to N(\mathbf{Fin}_*)
$$
使得 $p$ 是 coCartesian fibration。

这比一般 infinity-operad 更强：不仅 inert morphisms 有 coCartesian lifts，所有 morphisms in $\mathbf{Fin}_*$ 都有 coCartesian lifts。

**定义 18.13.** 在 symmetric monoidal infinity-category $\mathcal C^\otimes$ 中，$n$ 元 tensor product 由 active map
$$
\mu_n:\langle n\rangle\to\langle1\rangle
$$
的 coCartesian pushforward 给出：
$$
(X_1,\ldots,X_n)\mapsto \mu_{n,!}(X_1,\ldots,X_n).
$$

**命题 18.14.** Tensor product 在 infinity-category 意义下 associative and symmetric up to coherent equivalence。

**证明.** CoCartesian fibration 的 pushforward 沿 base morphisms 的复合满足 functoriality up to contractible choice。$\mathbf{Fin}_*$ 中 active maps 的不同分解对应不同括号和置换方式；coCartesian transport 把这些分解送到 $\mathcal C$ 中的等价。Higher simplices in $N(\mathbf{Fin}_*)$ 记录分解之间的相干关系，因此 tensor product 不是只给出单个二元函子，而是给出全体相干的多元张量。$\square$

## 18.5 Algebras over infinity-operads

**定义 18.15.** 设
$$
\mathcal O^\otimes\to N(\mathbf{Fin}_*),\qquad
\mathcal C^\otimes\to N(\mathbf{Fin}_*)
$$
是 Lurie-style infinity-operads。一个 $\mathcal O$-algebra in $\mathcal C$ 是 over $N(\mathbf{Fin}_*)$ 的 map of infinity-operads
$$
A:\mathcal O^\otimes\to\mathcal C^\otimes
$$
即保持 inert coCartesian edges 的 simplicial set map。

所有 $\mathcal O$-algebras 组成的 infinity-category 记为
$$
\operatorname{Alg}_{\mathcal O}(\mathcal C).
$$

**例 18.16.** 若 $\mathcal O^\otimes=N(\mathbf{Fin}_*)$ 作为 terminal commutative infinity-operad，则
$$
\operatorname{Alg}_{\mathcal O}(\mathcal C)
$$
是 $\mathcal C$ 中 commutative algebra objects 的 infinity-category，通常记作 $\operatorname{CAlg}(\mathcal C)$。

**例 18.17.** Associative algebra objects 可由 non-symmetric associative infinity-operad 或相应的 $\Delta^{\operatorname{op}}$-monoidal 模型定义。把 associative 与 commutative 都强行放入同一个 $\mathbf{Fin}_*$ 口径会遮蔽 non-symmetric 信息；本书后续会在需要时区分。

## 18.6 由 ordinary operad 到 Lurie-style infinity-operad

设 $\mathcal P$ 是 small colored operad。可以构造其 category of operators
$$
\mathcal P^\otimes\to\mathbf{Fin}_*
$$
如下：

1. 对象是二元组 $\big(\langle n\rangle;(c_1,\ldots,c_n)\big)$，其中 $c_i$ 是 $\mathcal P$ 的颜色；其投影为 $\langle n\rangle$。
2. 从 $(c_1,\ldots,c_m)$ 到 $(d_1,\ldots,d_n)$ lying over $\alpha:\langle m\rangle\to\langle n\rangle$ 的 morphism 是运算族 $(p_j)_{1\le j\le n}$，其中
   $$
   p_j\in
   \mathcal P\big((c_i)_{i\in\alpha^{-1}(j)};d_j\big).
   $$
   若 $\alpha^{-1}(j)=\varnothing$，这里使用 $\mathcal P(\varnothing;d_j)$ 中的 nullary operation；满足 $\alpha(i)=*$ 的源颜色不进入任何 $p_j$。
3. 恒等态射 lying over $\operatorname{id}_{\langle m\rangle}$ 是单位族 $(\mathbf1_{c_i})_{i=1}^m$。
4. 若 $(\alpha,(p_j))$ 后接 $(\beta,(q_\ell))$，则复合 lying over $\beta\alpha$ 的第 $\ell$ 个运算为
   $$
   r_\ell
   =
   q_\ell\big((p_j)_{j\in\beta^{-1}(\ell)}\big)
   \in
   \mathcal P\big((c_i)_{i\in(\beta\alpha)^{-1}(\ell)};e_\ell\big).
   $$
   此处使用分解
   $$
   (\beta\alpha)^{-1}(\ell)
   =
   \coprod_{j\in\beta^{-1}(\ell)}\alpha^{-1}(j)
   $$
   及相应的对称重标号。Operad 的单位律、结合律和等变性分别保证恒等态射、复合结合律和该公式独立于列举纤维的选择。

**外部输入定理 18.18（category-of-operators nerve；HA-OP-1）.** Nerve
$$
N(\mathcal P^\otimes)\to N(\mathbf{Fin}_*)
$$
是 Lurie-style infinity-operad。该构造把 strict colored operad 嵌入 Lurie-style infinity-operads。

**例 18.18.1（普通 operad 的完整算子范畴计算）.** 设 $\mathcal P$ 是允许 arity $0$ 的单色普通对称 operad，唯一颜色记为 $c$。于是 $\mathcal P^\otimes$ 在 $\langle n\rangle$ 上只有一个对象，简记为 $c^n$；但 lying over 同一个基映射的态射通常有许多个。

取
$$
\alpha:\langle3\rangle\to\langle2\rangle,
\qquad
\alpha(1)=\alpha(3)=1,\quad \alpha(2)=2,
$$
以及
$$
\beta:\langle2\rangle\to\langle1\rangle,
\qquad
\beta(1)=\beta(2)=1.
$$
二者都是 active。选取
$$
p\in\mathcal P(\{1,3\}),
\qquad
q\in\mathcal P(\{1,2\}),
$$
并记 $\mathbf1\in\mathcal P(\{2\})$ 为按单元素集合重标号后的单位。则
$$
(\alpha;(p,\mathbf1)):c^3\longrightarrow c^2,
\qquad
(\beta;q):c^2\longrightarrow c
$$
是 $\mathcal P^\otimes$ 中的态射。它们的复合 lying over $\beta\alpha=\mu_3$，其唯一运算是
$$
q(p,\mathbf1)\in\mathcal P(\{1,2,3\}).
$$
选择标准次序书写输入时，这个元素作用为
$$
(x_1,x_2,x_3)\longmapsto q\big(p(x_1,x_3),x_2\big).
$$
若再与第三个算子复合，算子范畴中的两种加括号方式相等，恰好是 $\mathcal P$ 的代入结合律。

现在考察 inert 投影
$$
\rho^1:\langle2\rangle\to\langle1\rangle,
\qquad
\rho^1(1)=1,\quad \rho^1(2)=*.
$$
态射
$$
(\rho^1;\mathbf1):c^2\longrightarrow c
$$
只保留第一个颜色，第二个源位置不进入任何 operad 运算。它在投影 $\mathcal P^\otimes\to\mathbf{Fin}_*$ 下是 coCartesian：给定 $\gamma:\langle1\rangle\to\langle r\rangle$，从 $c$ 出发 lying over $\gamma$ 的运算族，与从 $c^2$ 出发 lying over $\gamma\rho^1$ 的运算族逐项相同，因为位置 $2$ 在后一基映射下仍送往基点。这个逐项双射正是 ordinary category 中 coCartesian 态射的泛性质。一般 inert map 的提升同样由被选中颜色上的单位运算组成。

最后取 $\mathcal P=\operatorname{Com}$。每个有限输入集上恰有一个运算，包括空输入，所以上述每个 $\alpha$ 上恰有一个态射；因此
$$
\operatorname{Com}^\otimes\cong\mathbf{Fin}_*
$$
over $\mathbf{Fin}_*$。这也解释了例 18.16 中 terminal commutative infinity-operad 的来源。

**说明 18.19.** 这里的 $\mathcal P^\otimes$ 不是定义 16.14 的 dendroidal nerve。前者是 over $N(\mathbf{Fin}_*)$ 的 quasi-category，后者是 presheaf on $\Omega$。二者通过模型比较相连，而不是逐项相等。

## 18.7 与 dendroidal 模型的比较

**外部输入定理 18.20（open dendroidal--Lurie comparison；HHM-1--HHM-5）.** Heuts--Hinich--Moerdijk 在 open/no-constants 语境中，经 simplicial operads、dendroidal sets、forest sets、marked open forest sets 与 Lurie preoperads 构造 Quillen-equivalence zig-zag。具体定位为 Theorems 2.4.1、2.5.1、2.5.3、Corollary 2.5.4 和 Theorem 5.3.14；结论是模型范畴及其 underlying infinity-categories 的等价，不是逐对象相等。

**说明 18.21.** 本书默认允许 arity $0$，而外部输入定理 18.20 只覆盖 open/no-constants 子理论。因此含 nullary operations 的对象不得未经处理套用该 zig-zag；必须先限制到 open 部分或另引覆盖 constants 的比较定理。即使在 open 情形，也不能把 dendroidal inner horn filler 直接替换为 Lurie operadic fibration 条件；跨模型结论必须作为比较等价下的不变量传递。

**命题 18.22.** 在外部输入定理 18.20 的 open/no-constants 范围内，若一个结论只依赖 infinity-operad 的同伦不变量，并且已知在该比较等价下保持，则可在两个模型之间转移。

**证明.** 比较定理给出相应 homotopy theories 的等价。Homotopy invariant 的结论可表述为目标 infinity-category 中的等价不变性质。等价函子反映并保持等价，因此该性质在两边对应对象之间传递。$\square$

**说明 18.23.** 规则 M.18 给出 strict operad、dendroidal nerve、category of operators nerve、Lurie-style infinity-operad 和模型范畴中代数对象之间的允许依赖路径，警告 M.19 列出禁止捷径。后续凡跨模型移动，默认按照这些规则执行。

## 18.8 分量投影与真正运算

命题 18.5 的顺序把两种结构清楚分开：inert 部分先丢弃或选择分量，并要求 coCartesian 提升；active 部分随后把保留输入组织成输出，其 mapping spaces 才承载多元运算。Lurie-style infinity-operad 的 Segal 条件由各 $\rho^i$ 重构对象与态射的输出分量，symmetric monoidal infinity-category 则额外提升所有基态射。例 18.18.1 表明普通 operad 的单位恰好给出 inert coCartesian edges，而 operad 代入给出 active 态射的复合。Dendroidal nerve 与 category-of-operators nerve 仍位于不同模型中，含 nullary operations 时尤其不能越过外部比较定理的适用范围。

## 练习

**练习 18.1.** 验证投影 $\rho^i:\langle n\rangle\to\langle1\rangle$ 是 inert。

**练习 18.2.** 写出 $\mu_2:\langle2\rangle\to\langle1\rangle$ 对 tensor product 的作用。

**练习 18.3.** 在例 18.18.1 中改取 $\rho^2:\langle3\rangle\to\langle1\rangle$，逐项验证由单位给出的提升满足 ordinary coCartesian 泛性质。

**练习 18.4.** 解释为什么 symmetric monoidal infinity-category 的 tensor product 不是单独一个严格二元函子。

**练习 18.5.** 说明 dendroidal nerve $N_d(\mathcal P)$ 与 category of operators nerve $N(\mathcal P^\otimes)$ 所在范畴不同。
