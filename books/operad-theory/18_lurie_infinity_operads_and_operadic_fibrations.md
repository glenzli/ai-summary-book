# 第十八章：Lurie-style infinity-operads 与 operadic fibrations

本章介绍 Lurie-style infinity-operad。它不是 dendroidal set 的另一种记号，而是以 quasi-categories over $N(\mathbf{Fin}_*)$ 描述多输入运算的模型。核心思想是：$\mathbf{Fin}_*$ 的 inert morphisms 负责投影到各个输入或输出分量，active morphisms 负责真正的多输入运算。

本章只给出定义、基本例子和与 dendroidal 模型的比较边界。完整比较定理作为外部输入。

## 18.1 有基点有限集与 active-inert 分解

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

**命题 18.5.** 每个态射 $f:\langle m\rangle\to\langle n\rangle$ 可分解为 active 后接 inert：
$$
\langle m\rangle\xrightarrow{a}\langle k\rangle\xrightarrow{i}\langle n\rangle.
$$

**证明.** 令
$$
R=f^{-1}(*)\setminus\{*\}.
$$
设 $r=|R|$，取
$$
K=\{*,1,\ldots,n,\epsilon_x\ (x\in R)\}.
$$
这是一个有限有基点集合；选定一个保持基点的双射 $K\cong\langle k\rangle$ 后可把 $K$ 记作 $\langle k\rangle$。
定义 $i:\langle k\rangle\to\langle n\rangle$ 为
$$
i(j)=j\quad(1\le j\le n),\qquad i(\epsilon_x)=*,\qquad i(*)=*.
$$
则对每个 $1\le j\le n$，$i^{-1}(j)=\{j\}$，故 $i$ inert。

定义 $a:\langle m\rangle\to\langle k\rangle$ 为
$$
a(*)=*,
$$
并对非基点 $x$ 令
$$
a(x)=
\begin{cases}
f(x),& f(x)\ne *,\\
\epsilon_x,& f(x)=*.
\end{cases}
$$
于是 $a^{-1}(*)=\{*\}$，故 $a$ active。逐点检查得 $ia=f$。$\square$

**警告 18.6.** Active-inert 分解的方向依赖采用 $\mathbf{Fin}_*$ 还是其 opposite，以及采用 operadic convention 还是 co-operadic convention。本书在 Lurie-style 定义中使用 $\mathbf{Fin}_*$ 上的 inert morphisms $\rho^i:\langle n\rangle\to\langle1\rangle$。

## 18.2 CoCartesian edges 与 inert transport

**定义 18.7.** 设 $p:X\to S$ 是 simplicial sets 的 inner fibration。边 $e:x\to y$ in $X$ 称为 $p$-coCartesian，若对每个对象 $z\in X$，诱导的 mapping space 方块使
$$
\operatorname{Map}_X(y,z)\to
\operatorname{Map}_X(x,z)\times_{\operatorname{Map}_S(p(x),p(z))}
\operatorname{Map}_S(p(y),p(z))
$$
成为弱等价。

这是 quasi-category 中 coCartesian edge 的模型化定义；完整表述需要使用 slice quasi-categories。

**定义 18.8.** 若 $p:\mathcal O^\otimes\to N(\mathbf{Fin}_*)$ 是 inner fibration，且每个 inert morphism $\alpha:\langle m\rangle\to\langle n\rangle$ 与每个 $X\in\mathcal O^\otimes_{\langle m\rangle}$ 都有 $p$-coCartesian lift
$$
X\to \alpha_!X,
$$
则称 $p$ admits inert coCartesian lifts。

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
3. 对任意 $X\in\mathcal O^\otimes_{\langle m\rangle}$、$Y\in\mathcal O^\otimes_{\langle n\rangle}$ 和 $\alpha:\langle m\rangle\to\langle n\rangle$，选取 $Y\to Y_i$ over $\rho^i$ 的 inert coCartesian edges，则自然映射
   $$
   \operatorname{Map}^{\alpha}_{\mathcal O^\otimes}(X,Y)
   \to
   \prod_{i=1}^n
   \operatorname{Map}^{\rho^i\alpha}_{\mathcal O^\otimes}(X,Y_i)
   $$
   是 homotopy equivalence。

这里 $\operatorname{Map}^{\alpha}$ 表示 lying over $\alpha$ 的 mapping space fiber。

**说明 18.10.** 条件 2 说 fiber over $\langle n\rangle$ 是 $n$ 个颜色对象的乘积。条件 3 说映到 $n$ 个输出分量的 morphism space 可由各分量上的 morphism spaces 重构。Active morphisms over $\langle n\rangle\to\langle1\rangle$ 编码 $n$ 输入运算。

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

1. over $\langle n\rangle$ 的对象是 $n$ 元颜色列 $(c_1,\ldots,c_n)$；
2. 从 $(c_1,\ldots,c_m)$ 到 $(d_1,\ldots,d_n)$ over $\alpha:\langle m\rangle\to\langle n\rangle$ 的 morphism 由对每个 $1\le j\le n$ 的运算
   $$
   (c_i)_{\alpha(i)=j}\to d_j
   $$
   给出；
3. 复合由 $\mathcal P$ 的 operad composition 给出。

**外部输入定理 18.18.** Nerve
$$
N(\mathcal P^\otimes)\to N(\mathbf{Fin}_*)
$$
是 Lurie-style infinity-operad。该构造把 strict colored operad 嵌入 Lurie-style infinity-operads。

**说明 18.19.** 这里的 $\mathcal P^\otimes$ 不是定义 16.14 的 dendroidal nerve。前者是 over $N(\mathbf{Fin}_*)$ 的 quasi-category，后者是 presheaf on $\Omega$。二者通过模型比较相连，而不是逐项相等。

## 18.7 与 dendroidal 模型的比较

**外部输入定理 18.20（Heuts-Hinich-Moerdijk 型比较）.** Dendroidal infinity-operads 与 Lurie-style infinity-operads 之间存在模型比较。更精确地，适当的 dendroidal operadic model structure 与 Lurie preoperads / marked simplicial sets 模型之间通过 Quillen equivalence 或等价的 infinity-category 相联系。

**说明 18.21.** 该比较说明两种模型描述同一同伦理论，但它不允许在证明中把 dendroidal inner horn filler 直接替换为 Lurie operadic fibration 条件。任何跨模型使用的定理都必须说明经过比较等价传递。

**命题 18.22.** 若一个结论只依赖 infinity-operad 的同伦不变量，并且已知在 dendroidal 模型与 Lurie 模型的比较等价下保持，则可在两个模型之间转移。

**证明.** 比较定理给出相应 homotopy theories 的等价。Homotopy invariant 的结论可表述为目标 infinity-category 中的等价不变性质。等价函子反映并保持等价，因此该性质在两边对应对象之间传递。$\square$

**说明 18.23.** 规则 M.18 给出 strict operad、dendroidal nerve、category of operators nerve、Lurie-style infinity-operad 和模型范畴中代数对象之间的允许依赖路径，警告 M.19 列出禁止捷径。后续凡跨模型移动，默认按照这些规则执行。

## 18.8 本章小结

Lurie-style infinity-operad 是 over $N(\mathbf{Fin}_*)$ 的 inner fibration，inert morphisms 控制分量投影，active morphisms 控制多输入运算。Symmetric monoidal infinity-category 是更强的 coCartesian fibration。Algebras over an infinity-operad 是保持 inert structure 的 operad maps。Dendroidal model 与 Lurie model 比较是外部输入定理；本书后续会在使用时明确所处模型。

## 练习

**练习 18.1.** 验证投影 $\rho^i:\langle n\rangle\to\langle1\rangle$ 是 inert。

**练习 18.2.** 写出 $\mu_2:\langle2\rangle\to\langle1\rangle$ 对 tensor product 的作用。

**练习 18.3.** 对 one-colored ordinary operad $\mathcal P$，描述 $\mathcal P^\otimes$ over $\langle2\rangle$ 的对象和 morphisms。

**练习 18.4.** 解释为什么 symmetric monoidal infinity-category 的 tensor product 不是单独一个严格二元函子。

**练习 18.5.** 说明 dendroidal nerve $N_d(\mathcal P)$ 与 category of operators nerve $N(\mathcal P^\otimes)$ 所在范畴不同。
