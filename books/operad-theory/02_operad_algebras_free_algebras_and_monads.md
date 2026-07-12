# 第二章：Operad 代数、自由代数与单子

## 本章目标

本章把第一章的“到 endomorphism operad 的 morphism”推进为代数范畴和自由代数理论。核心目标是：

1. 定义 $\mathcal O$-代数同态和范畴 $\operatorname{Alg}_{\mathcal O}(\mathbf{Set})$。
2. 构造 operad morphism 诱导的限制标量函子。
3. 给出集合值自由 $\mathcal O$-代数的显式商公式。
4. 证明自由-遗忘伴随。
5. 把 $\mathcal O$-代数识别为一个 finitary monad 的代数。

## 依赖前置知识

需要第一章的对称 operad、endomorphism operad 和 operad 代数定义，以及范畴、函子、伴随和 monad 的基本语言。

## 2.1 代数同态

**定义 2.1.** 设 $\mathcal O$ 是集合值对称 operad。若 $(X,\alpha)$ 与 $(Y,\beta)$ 是 $\mathcal O$-代数，一个 $\mathcal O$-代数同态
$$
f:(X,\alpha)\to(Y,\beta)
$$
是函数 $f:X\to Y$，使得对任意有限集 $S$、任意 $o\in\mathcal O(S)$ 和任意 $a:S\to X$，有
$$
f\big(\alpha_S(o)(a)\big)
=
\beta_S(o)(f\circ a).
$$

**命题 2.2.** $\mathcal O$-代数和 $\mathcal O$-代数同态构成范畴，记为
$$
\operatorname{Alg}_{\mathcal O}(\mathbf{Set}_{\mathcal U}).
$$

**证明.** 恒等函数 $\operatorname{id}_X:X\to X$ 满足
$$
\operatorname{id}_X(\alpha_S(o)(a))=\alpha_S(o)(\operatorname{id}_X\circ a).
$$
若 $f:(X,\alpha)\to(Y,\beta)$ 与 $g:(Y,\beta)\to(Z,\gamma)$ 是同态，则
$$
(g\circ f)(\alpha_S(o)(a))
=g(\beta_S(o)(f\circ a))
=\gamma_S(o)(g\circ f\circ a),
$$
所以 $g\circ f$ 仍是同态。函数复合的结合律给出范畴公理。$\square$

**定义 2.3.** 遗忘函子
$$
U_{\mathcal O}:\operatorname{Alg}_{\mathcal O}(\mathbf{Set}_{\mathcal U})
\to \mathbf{Set}_{\mathcal U}
$$
把 $(X,\alpha)$ 送到 $X$，把同态送到底层函数。

## 2.2 限制标量

**定义 2.4.** 设 $\varphi:\mathcal O\to\mathcal P$ 是 operad morphism。定义限制标量函子
$$
\varphi^\*:\operatorname{Alg}_{\mathcal P}(\mathbf{Set}_{\mathcal U})
\to
\operatorname{Alg}_{\mathcal O}(\mathbf{Set}_{\mathcal U})
$$
如下。若 $\mathcal P$-代数由 $\beta:\mathcal P\to\operatorname{End}_X$ 给出，则 $\varphi^\*(X,\beta)$ 是 $\mathcal O$-代数
$$
\mathcal O\xrightarrow{\varphi}\mathcal P\xrightarrow{\beta}\operatorname{End}_X.
$$
同态的底层函数保持不变。

**命题 2.5.** 定义 2.4 的规则定义函子，并且对 operad morphism
$$
\mathcal O\xrightarrow{\varphi}\mathcal P\xrightarrow{\psi}\mathcal Q
$$
有严格等式
$$
(\psi\circ\varphi)^\*=\varphi^\*\circ\psi^\*.
$$

**证明.** 若 $f:X\to Y$ 是 $\mathcal P$-代数同态，则对 $o\in\mathcal O(S)$，
$$
f\big(\beta_S(\varphi_S(o))(a)\big)
=
\beta'_S(\varphi_S(o))(f\circ a),
$$
所以 $f$ 也是限制后的 $\mathcal O$-代数同态。恒等与复合保持不变，因此得到函子。第二个等式是 operad morphism 复合的结合律在 endomorphism operad 中的直接表达。$\square$

## 2.3 自由 $\mathcal O$-代数的底层集合

为了给出显式公式，本节使用标准有限集 $[n]=\{1,\ldots,n\}$。记 $\mathcal O(n)=\mathcal O([n])$。

**定义 2.6.** 设 $A$ 是集合。定义集合
$$
T_{\mathcal O}(A)
=
\left(
\coprod_{n\ge 0}\mathcal O(n)\times A^{[n]}
\right)\big/\sim,
$$
其中等价关系 $\sim$ 由以下关系生成：对任意 $\sigma\in\Sigma_n$、$o\in\mathcal O(n)$ 和 $a:[n]\to A$，
$$
(\mathcal O(\sigma)(o),a)\sim(o,a\circ\sigma).
$$
等价类记为
$$
[o;a_1,\ldots,a_n],
$$
其中 $a_i=a(i)$。当 $n=0$ 时写作 $[o;]$。

**解释 2.7.** $T_{\mathcal O}(A)$ 的元素是“一个抽象 $n$ 元运算 $o$，填入 $A$ 中的 $n$ 个元素”所得的形式表达式。商关系说明：若同时重标号运算的输入和被填入的元素，表达式不变。

**定义 2.8.** 定义函数
$$
\iota_A:A\to T_{\mathcal O}(A)
$$
如下。令 $\mathbf 1\in\mathcal O(1)$ 为 operad 单位，则
$$
\iota_A(a)=[\mathbf 1;a].
$$

**命题 2.8.1（有限集 coend 公式）.** 定义 2.6 的集合自然同构于 coend
$$
T_{\mathcal O}(A)
\cong
\int^{S\in\mathbf B_{\mathcal U}}\mathcal O(S)\times A^S,
$$
其中 $A^S=\mathbf{Set}_{\mathcal U}(S,A)$。对双射 $\sigma:S\to T$，$\mathcal O$ 在第一变量中用协变作用
$$
\mathcal O(\sigma):\mathcal O(S)\to\mathcal O(T),
$$
而 $A^{(-)}$ 在第二变量中用预复合
$$
A^T\to A^S,\qquad b\mapsto b\circ\sigma.
$$

**证明.** 由命题 A.7，群胚 $\mathbf B_{\mathcal U}$ 等价于骨架
$$
\coprod_{n\ge0}B\Sigma_n.
$$
Coend 在等价群胚上不变。因此命题 2.8.1 中的 coend 等价于对每个 $n$ 取
$$
\mathcal O(n)\times A^{[n]}
$$
并按 $\Sigma_n$ 作用取 orbit。由 coend 的定义，$\sigma\in\Sigma_n$ 生成关系
$$
(\mathcal O(\sigma)(o),a)\sim(o,a\circ\sigma),
$$
这正是定义 2.6 的关系。对所有 $n$ 求余和即得定义 2.6 的集合。自然性来自 coend 的泛性质：函数 $h:A\to B$ 诱导 $A^S\to B^S$，并与所有预复合映射交换。$\square$

**说明 2.8.2.** 命题 2.8.1 是本章的安全公式。骨架写法
$$
\left(
\coprod_{n\ge0}\mathcal O(n)\times A^{[n]}
\right)\big/\sim
$$
只是选择 $[n]$ 后的坐标表达。涉及 colored operad、线性 Schur functor 或模型范畴中自由代数时，应优先把相应公式理解为 coend 或 indexed coproduct，而不是非自然的序列商。

## 2.4 自由代数结构

**定义 2.9.** 设 $p\in\mathcal O(S)$，且给定 $S$-indexed 元素族
$$
x_s\in T_{\mathcal O}(A),\qquad s\in S.
$$
对每个 $s$ 选择代表元
$$
x_s=[o_s;a_s:T_s\to A],
$$
其中 $T_s$ 是有限集，$o_s\in\mathcal O(T_s)$。令
$$
T=\coprod_{s\in S}T_s
$$
并令
$$
q:T\longrightarrow S
$$
为不交并的典范投影，所以 $q^{-1}(s)$ 是带标签的 $T_s$ 副本；允许 $T_s=\varnothing$，此时 $s$ 仍是目标中的一个输入槽，只是该槽的纤维为空。
令
$$
a:T\to A
$$
为在每个 $T_s$ 上等于 $a_s$ 的函数。定义
$$
p\cdot(x_s)_{s\in S}
=
\left[
\mu_q\big(p;(o_s)_{s\in S}\big);
a:T\to A
\right],
$$
其中 $\mu_q$ 是 $\mathcal O$ 沿有限集映射 $q$ 的 operad 复合。

若选择 $S=[k]$ 和 $T_s=[n_s]$（$n_s\ge0$），这一定义就是坐标公式
$$
p\cdot(x_1,\ldots,x_k)
=
\big[
\gamma(p;o_1,\ldots,o_k);
a^1_1,\ldots,a^1_{n_1},\ldots,a^k_{n_k}
\big].
$$

**引理 2.10.** 定义 2.9 中的动作与代表元选择无关。

**证明.** 只需检查一个输入 $x_s$ 的代表元沿双射改变的情形。设
$$
\theta:T_s\to T'_s
$$
是双射，并把代表元替换为
$$
[o'_s;a'_s:T'_s\to A]
$$
其中
$$
o'_s=\mathcal O(\theta)(o_s),
\qquad
a_s=a'_s\circ\theta.
$$
令
$$
\Theta:
T=\coprod_{t\in S}T_t\longrightarrow
T'=\left(\coprod_{t\ne s}T_t\right)\coprod T'_s
$$
为在 $T_s$ 上等于 $\theta$、在其他 summands 上为恒等的双射。令 $q:T\to S$ 与 $q':T'\to S$ 为典范投影，则 $q'\Theta=q$。Operad 复合的自然性给出
$$
\mu_{q'}\big(p;(o'_s,o_t)_{t\ne s}\big)
=
\mathcal O(\Theta)\,
\mu_q\big(p;(o_s,o_t)_{t\ne s}\big).
$$
记右端被 $\mathcal O(\Theta)$ 作用前的元素为
$u=\mu_q(p;(o_s,o_t)_{t\ne s})\in\mathcal O(T)$。
同时新的输入函数 $a':T'\to A$ 满足
$$
a=a'\circ\Theta.
$$
由命题 2.8.1 的 coend 关系，
$$
(\mathcal O(\Theta)(u),a')\sim(u,a),
$$
所以两种代表元给出同一元素。多个输入同时重标号时逐个应用同一代表元无关性论证。

在骨架写法中，引理 2.10 的论证变为如下特殊情形：若第 $i$ 个输入由 $\tau\in\Sigma_{n_i}$ 重标号，则在商集中有
$$
[o_i;a^i_1,\ldots,a^i_{n_i}]
=
[\mathcal O(\tau)(o_i);a^i_{\tau^{-1}(1)},\ldots,a^i_{\tau^{-1}(n_i)}],
$$
这是定义 2.6 的关系按序列记号改写。把 $o_i$ 替换为 $\mathcal O(\tau)(o_i)$ 时，operad 代入的等变性说明总运算
$$
\gamma(p;o_1,\ldots,\mathcal O(\tau)(o_i),\ldots,o_k)
$$
等于把
$$
\gamma(p;o_1,\ldots,o_i,\ldots,o_k)
$$
按只作用于第 $i$ 个输入块的块内置换重标号。商关系同时把输入序列按同一块内置换重排，因此两者代表同一个等价类。多个输入同时重标号时逐一应用该论证。外层 $p$ 的重标号同理使用块置换等变性。$\square$

**命题 2.11.** $T_{\mathcal O}(A)$ 连同定义 2.9 的动作是 $\mathcal O$-代数。

**证明.** 引理 2.10 保证动作良定义。单位律：若 $\mathbf 1\in\mathcal O(1)$，则
$$
\mathbf 1\cdot[o;a_1,\ldots,a_n]
=
[\gamma(\mathbf 1;o);a_1,\ldots,a_n]
=
[o;a_1,\ldots,a_n],
$$
其中最后一步使用 operad 单位律。

结合律：设 $p\in\mathcal O(k)$，$q_i\in\mathcal O(n_i)$，其中 $n_i\ge0$，且每个输入又写成
$$
y_{ij}=[r_{ij};a^{ij}_1,\ldots,a^{ij}_{m_{ij}}].
$$
这里 $m_{ij}\ge0$。即使某个 $n_i$ 或 $m_{ij}$ 为零，空纤维仍保留相应外层槽。
先用 $q_i$ 作用再用 $p$ 作用，得到的外层 operad 元素是
$$
\gamma\big(p;\gamma(q_1;r_{11},\ldots,r_{1n_1}),\ldots,
\gamma(q_k;r_{k1},\ldots,r_{kn_k})\big).
$$
先把 $p$ 与所有 $q_i$ 代入，再一次性作用于所有 $y_{ij}$，得到的外层 operad 元素是
$$
\gamma\big(\gamma(p;q_1,\ldots,q_k);
r_{11},\ldots,r_{kn_k}\big).
$$
这两个元素由 operad 结合律相等；底层 $A$-输入序列在两种计算中都是按同一块顺序连接的序列。因此 $\mathcal O$-代数结合律成立。等变性由定义 2.6 的商关系和 operad 等变性给出。$\square$

## 2.5 自由-遗忘伴随

**定理 2.12.** 构造
$$
A\mapsto T_{\mathcal O}(A)
$$
定义函子
$$
F_{\mathcal O}:\mathbf{Set}_{\mathcal U}\to
\operatorname{Alg}_{\mathcal O}(\mathbf{Set}_{\mathcal U}),
$$
并且存在自然伴随
$$
F_{\mathcal O}\dashv U_{\mathcal O}.
$$

**证明.** 若 $h:A\to B$ 是函数，定义
$$
T_{\mathcal O}(h)([o;a_1,\ldots,a_n])
=
[o;h(a_1),\ldots,h(a_n)].
$$
该定义与商关系相容，因此给出函数。它保持 $\mathcal O$-动作，因为动作公式只改变运算代入，不改变对底层输入逐点施加函数的顺序。所以 $F_{\mathcal O}$ 是函子。

现在设 $(X,\alpha)$ 是 $\mathcal O$-代数。对任意函数 $u:A\to X$，定义
$$
\bar u:T_{\mathcal O}(A)\to X,
\qquad
\bar u([o;a_1,\ldots,a_n])
=
\alpha_{[n]}(o)(u(a_1),\ldots,u(a_n)).
$$
若代表元被 $\sigma\in\Sigma_n$ 重标号，则 $\mathcal O$-代数动作的自然性给出
$$
\alpha(\mathcal O(\sigma)o)(u(a_1),\ldots,u(a_n))
=
\alpha(o)(u(a_{\sigma(1)}),\ldots,u(a_{\sigma(n)})),
$$
所以 $\bar u$ 良定义。代入相容性说明 $\bar u$ 是 $\mathcal O$-代数同态。

并且
$$
\bar u(\iota_A(a))
=
\bar u([\mathbf 1;a])
=u(a)
$$
其中最后一步使用 $\mathcal O$-代数的单位律。因此 $\bar u\circ\iota_A=u$。

反过来，若 $g:T_{\mathcal O}(A)\to X$ 是 $\mathcal O$-代数同态，则 $u=g\circ\iota_A:A\to X$ 决定 $g$。事实上，
$$
[o;a_1,\ldots,a_n]
=
o\cdot(\iota_A(a_1),\ldots,\iota_A(a_n))
$$
在自由代数中成立，所以
$$
g([o;a_1,\ldots,a_n])
=
\alpha(o)(g\iota_A(a_1),\ldots,g\iota_A(a_n)).
$$
这正是由 $u$ 构造的 $\bar u$。于是有自然双射
$$
\operatorname{Alg}_{\mathcal O}(F_{\mathcal O}A,X)
\cong
\mathbf{Set}_{\mathcal U}(A,U_{\mathcal O}X),
$$
即 $F_{\mathcal O}\dashv U_{\mathcal O}$。$\square$

## 2.6 关联的 monad

**定义 2.13.** 令
$$
\mathbb T_{\mathcal O}=U_{\mathcal O}F_{\mathcal O}
$$
为 $\mathbf{Set}_{\mathcal U}$ 上的自函子。伴随 $F_{\mathcal O}\dashv U_{\mathcal O}$ 给出 monad
$$
(\mathbb T_{\mathcal O},\eta,\mu),
$$
其中 $\eta_A:A\to\mathbb T_{\mathcal O}(A)$ 是定义 2.8 的 $\iota_A$，乘法
$$
\mu_A:\mathbb T_{\mathcal O}\mathbb T_{\mathcal O}(A)\to\mathbb T_{\mathcal O}(A)
$$
由自由代数的 $\mathcal O$-代数结构给出。

**命题 2.14.** $\operatorname{Alg}_{\mathcal O}(\mathbf{Set}_{\mathcal U})$ 与 monad $\mathbb T_{\mathcal O}$ 的 Eilenberg-Moore 代数范畴同构。

**证明.** 若 $(X,\alpha)$ 是 $\mathcal O$-代数，定义
$$
\theta_X:\mathbb T_{\mathcal O}(X)\to X,
\qquad
\theta_X([o;x_1,\ldots,x_n])=\alpha(o)(x_1,\ldots,x_n).
$$
单位公理对应 $\mathcal O$-单位作用为恒等；乘法公理对应 $\mathcal O$-代入结合律。

反过来，若 $\theta:\mathbb T_{\mathcal O}(X)\to X$ 是 $\mathbb T_{\mathcal O}$-代数结构，则定义
$$
\alpha(o)(x_1,\ldots,x_n)
=
\theta([o;x_1,\ldots,x_n]).
$$
monad 单位和乘法公理分别给出 operad 单位律和代入结合律。两种构造在对象和同态上互逆。$\square$

**命题 2.15.** monad $\mathbb T_{\mathcal O}$ 保持滤过余极限。

**证明.** 设 $(A_i)_{i\in I}$ 是滤过图，$A=\operatorname*{colim}_i A_i$。由命题 2.8.1，
$$
\mathbb T_{\mathcal O}(A)
\cong
\int^{S\in\mathbf B_{\mathcal U}}\mathcal O(S)\times A^S.
$$
对固定有限集 $S$，任意函数 $S\to A$ 的像有限，因此它因子通过某个 $A_i$。若两个因子化 $S\to A_i$ 与 $S\to A_j$ 在 $A$ 中相等，由 $S$ 有限和 $I$ 滤过性，它们在某个共同后继 $A_k$ 中相等。因此
$$
\operatorname*{colim}_i A_i^S\cong A^S.
$$
现在 coend 在 $\mathbf{Set}$ 中由余和再取由双射生成的等价关系得到。$\mathbf B_{\mathcal U}$ 的每个连通分支等价于有限群 $\Sigma_n$ 的单对象群胚；固定 $n$ 时的 orbit quotient 是有限群作用的商。滤过余极限与有限极限形式的等式检测、余和以及有限群 orbit 商相容，故
$$
\operatorname*{colim}_i
\int^{S}\mathcal O(S)\times A_i^S
\cong
\int^{S}\mathcal O(S)\times
\operatorname*{colim}_i A_i^S
\cong
\int^{S}\mathcal O(S)\times A^S.
$$
这就是
$$
\operatorname*{colim}_i\mathbb T_{\mathcal O}(A_i)
\cong
\mathbb T_{\mathcal O}(A).
$$
因此 $\mathbb T_{\mathcal O}$ 保持滤过余极限。$\square$

## 2.7 两个自由代数例子

**命题 2.16.** $\operatorname{Com}$ 的自由代数 $\mathbb T_{\operatorname{Com}}(A)$ 是 $A$ 上的有限重集集合，乘法为重集并。

**证明.** 因为 $\operatorname{Com}(n)$ 是单点集，所以
$$
\mathbb T_{\operatorname{Com}}(A)
\cong
\coprod_{n\ge0}A^n/\Sigma_n.
$$
右侧正是 $A$ 中元素的有限无序列，即有限重集。operad 代入把一个外层唯一运算和若干内层唯一运算送到唯一运算；在输入上就是把若干有限重集并在一起。因此自由 $\operatorname{Com}$-代数是有限重集幺半群。$\square$

**命题 2.17.** $\operatorname{Ass}$ 的自由代数 $\mathbb T_{\operatorname{Ass}}(A)$ 是 $A$ 上的有限列表集合，乘法为列表连接。

**证明.** $\operatorname{Ass}(S)$ 是 $S$ 上全序集合。给定 $o\in\operatorname{Ass}(S)$ 和 $a:S\to A$，全序 $o$ 把 $S$ 排成 $s_1<\cdots<s_n$，于是得到列表
$$
(a(s_1),\ldots,a(s_n)).
$$
若同时重标号 $S$、全序和函数 $a$，该列表不变。因此
$$
\coprod_{S}\operatorname{Ass}(S)\times A^S/\text{重标号}
$$
与有限列表集合双射。$\operatorname{Ass}$ 的代入是全序字代入，所以自由代数乘法是列表连接。$\square$

## 本章小结

本章证明了集合值 operad 不只是编码运算符号；它给出集合范畴上的 finitary monad。自由 $\mathcal O$-代数由公式
$$
\mathbb T_{\mathcal O}(A)
=
\int^{S\in\mathbf B_{\mathcal U}}\mathcal O(S)\times A^S
$$
描述；选择骨架 $[n]$ 后才得到
$$
\left(
\coprod_{n\ge0}\mathcal O(n)\times A^n
\right)\big/\Sigma_n\text{兼容关系}.
$$
商关系正是“输入重标号不改变形式运算值”。$\operatorname{Com}$ 和 $\operatorname{Ass}$ 的自由代数分别是有限重集和有限列表。

## 练习

**练习 2.1.** 证明定义 2.6 中的关系确实生成等价关系，并把它写成 $\Sigma_n$ 作用的 orbit set。

**练习 2.2.** 直接验证 $\bar u:T_{\mathcal O}(A)\to X$ 是 $\mathcal O$-代数同态。

**练习 2.3.** 写出 $\mathbb T_{\operatorname{Com}}(\{a,b\})$ 中次数不超过 $3$ 的全部元素。

**练习 2.4.** 对 $\operatorname{Ass}$，说明为什么 $(a,b)$ 与 $(b,a)$ 通常不是同一个自由代数元素。

**练习 2.5.** 设 $\varphi:\operatorname{Ass}\to\operatorname{Com}$ 是遗忘顺序的 operad morphism。描述限制标量函子 $\varphi^\*$ 对一个交换幺半群做了什么。
