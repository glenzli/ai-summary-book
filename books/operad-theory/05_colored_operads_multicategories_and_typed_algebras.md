# 第五章：Colored operad、多范畴与带类型代数

单色 operad 默认所有输出都能送入所有输入槽，这对许多结构过于宽松。范畴中的箭头只有在源、靶匹配时才能复合；双模上的左作用、右作用和底层乘法也分别具有不同类型。把类型组成颜色集 $C$ 后，一个运算必须记录输入颜色函数 $S\to C$ 与输出颜色，而代入只有在内层输出逐槽匹配外层输入时才有定义。本章把第一章的有限集代入逐字推广到这种带类型轮廓，并比较 colored operad 与 symmetric multicategory。范畴、函子、作用和代数同态会成为可直接验算的例子。

## 5.1 带颜色的输入轮廓

**定义 5.1.** 固定一个 $\mathcal U$-小集合 $C$，称其元素为颜色（colors）。一个 $C$-轮廓（$C$-profile）是三元组
$$
(S,\kappa;c),
$$
其中 $S$ 是有限集，$\kappa:S\to C$ 是输入颜色函数，$c\in C$ 是输出颜色。

一个从 $(S,\kappa;c)$ 到 $(T,\lambda;d)$ 的同构是双射 $\varphi:S\to T$，满足
$$
d=c,\qquad \lambda\circ\varphi=\kappa.
$$
所有 $C$-轮廓及其同构构成群胚，记为 $\mathbf B_C$。

**定义 5.2.** 一个 $C$-colored symmetric sequence 是函子
$$
X:\mathbf B_C\to\mathbf{Set}_{\mathcal U}.
$$
其在轮廓 $(S,\kappa;c)$ 上的值写作
$$
X(S,\kappa;c).
$$
态射是自然变换。所得范畴记为
$$
\operatorname{SymSeq}_C.
$$

**例 5.3.** 当 $C=\{*\}$ 是单点集时，$\mathbf B_C$ 与第一章的有限集群胚 $\mathbf B_{\mathcal U}$ 等价。因此单色 colored symmetric sequence 就是普通对称序列。

## 5.2 Colored substitution product

**定义 5.4.** 设 $X,Y\in\operatorname{SymSeq}_C$。定义 $X\circ_C Y$ 如下。对轮廓 $(S,\kappa;c)$，
$$
(X\circ_CY)(S,\kappa;c)
=
\operatorname*{colim}_{(f:S\to T)\in\operatorname{Fib}(S)}
\coprod_{\delta:T\to C}
X(T,\delta;c)
\times
\prod_{t\in T}
Y(f^{-1}(t),\kappa|_{f^{-1}(t)};\delta(t)).
$$
这里 $\delta(t)$ 是纤维 $f^{-1}(t)$ 上内层运算的输出颜色，同时也是外层运算对应输入槽的颜色。函数 $f$ 不要求满射；空纤维允许把 nullary $Y$-operation 代入外层输入槽。

若 $\varphi:(S,\kappa;c)\to(S',\lambda;c)$ 是 $\mathbf B_C$ 中的同构，则把 $f:S\to T$ 改为 $f\varphi^{-1}:S'\to T$，并在每个纤维上用 $\varphi$ 的限制作用。目标双射同时重标号 $T$ 和 $\delta$，再对该关系取 colimit。

若对每个输出颜色 $d$ 都有 $Y(\varnothing,\varnothing;d)=\varnothing$，则只有满射有贡献，公式才退化为按非空 colored 分块求 coproduct。

**定义 5.5.** 单位 colored symmetric sequence $I_C$ 定义为
$$
I_C(S,\kappa;c)=
\begin{cases}
\{*\}, & S=\{s\}\text{ 且 }\kappa(s)=c,\\
\varnothing, & \text{否则}.
\end{cases}
$$

**命题 5.6.** $\operatorname{SymSeq}_C$ 连同 $\circ_C$ 和 $I_C$ 构成幺半范畴。

**证明.** 两种加括号都展开为可复合映射
$$
S\xrightarrow{g}U\xrightarrow{p}T,
$$
颜色函数 $\epsilon:U\to C$、$\delta:T\to C$，以及装饰
$$
x\in X(T,\delta;c),
$$
$$
y_t\in Y(p^{-1}(t),\epsilon|_{p^{-1}(t)};\delta(t)),
\qquad
z_u\in Z(g^{-1}(u),\kappa|_{g^{-1}(u)};\epsilon(u)).
$$
对 $U,T$ 的双射取商时同步重标号 $\epsilon,\delta$。从 $(X\circ_CY)\circ_CZ$ 到这组数据是直接展开；从 $X\circ_C(Y\circ_CZ)$ 到这组数据则把各外层纤维中的目标集合不交并成 $U$。两构造互逆并尊重颜色。

对右单位 $X\circ_CI_C$，单位因子强制 $f$ 为双射且 $\delta(f(s))=\kappa(s)$；对左单位 $I_C\circ_CX$，外层单位强制 $T$ 为单点且其唯一颜色为 $c$。两者分别给出 $X(S,\kappa;c)$，包括 $S=\varnothing$。四重代入展开为同一三层映射与两层中间颜色数据，所以五边形交换；插入、删除上述单位层给出三角形。$\square$

## 5.3 Colored operad

**定义 5.7.** 一个 $C$-colored operad 是幺半范畴
$$
(\operatorname{SymSeq}_C,\circ_C,I_C)
$$
中的幺半对象。也就是说，它由 $C$-colored symmetric sequence $\mathcal P$、乘法
$$
\mu:\mathcal P\circ_C\mathcal P\to\mathcal P
$$
和单位
$$
\eta:I_C\to\mathcal P
$$
组成，并满足结合律和单位律。

**展开 5.8.** 给定有限集映射 $f:S\to T$、中间颜色 $\delta:T\to C$、外层运算
$$
p\in\mathcal P(T,\delta;c)
$$
和各纤维上的内层运算
$$
p_t\in\mathcal P(f^{-1}(t),\kappa|_{f^{-1}(t)};\delta(t)),
$$
乘法给出
$$
\mu_f(p;(p_t)_{t\in T})\in\mathcal P(S,\kappa;c).
$$
空纤维上的 $p_t$ 是 nullary operation，其输出颜色仍必须是 $\delta(t)$。Operad 结合律断言对可复合映射 $S\to U\to T$，多层带颜色代入的结果不依赖先代入哪一层；单位律断言每个颜色 $c$ 有一个一元恒等运算
$$
\mathbf 1_c\in\mathcal P(\{*\},*\mapsto c;c)
$$
并且只在颜色匹配时作为单位。

**定义 5.9.** 若 $\mathcal P,\mathcal Q$ 是同一颜色集合 $C$ 上的 colored operad，一个 morphism $\mathcal P\to\mathcal Q$ 是保持乘法和单位的 colored symmetric sequence 态射。

## 5.4 Colored endomorphism operad 与代数

**定义 5.10.** 设 $A=(A_c)_{c\in C}$ 是 $C$-indexed 集合族。定义 colored endomorphism operad
$$
\operatorname{End}_A(S,\kappa;c)
=
\mathbf{Set}_{\mathcal U}\left(\prod_{s\in S}A_{\kappa(s)},A_c\right).
$$
轮廓同构通过重排有限乘积输入给出作用。

代入如下。设 $f:S\to T$ 是有限集映射，$\delta:T\to C$。给定
$$
F:\prod_{t\in T}A_{\delta(t)}\to A_c
$$
和
$$
G_t:\prod_{s\in f^{-1}(t)}A_{\kappa(s)}\to A_{\delta(t)},
$$
定义
$$
h:\prod_{s\in S}A_{\kappa(s)}\to A_c
$$
为
$$
h((a_s)_{s\in S})
=
F\big((G_t((a_s)_{s\in f^{-1}(t)}))_{t\in T}\big).
$$
若 $f^{-1}(t)=\varnothing$，则 $G_t$ 从空乘积 $1$ 出发，等价于选择 $A_{\delta(t)}$ 中的常量；所以该公式确实实现 nullary substitution。

**命题 5.11.** $\operatorname{End}_A$ 是 $C$-colored operad。

**证明.** 单位 $\mathbf 1_c$ 是恒等函数 $A_c\to A_c$。对可复合映射 $S\xrightarrow{g}U\xrightarrow{p}T$，两种复合都把 $(a_s)$ 先送入各 $g$-纤维上的 $Z$-层函数，再把所得元素送入各 $p$-纤维上的 $Y$-层函数，最后送入 $T$-指标的外层函数；因此结合律就是函数复合的结合律。颜色函数 $U\to C$ 与 $T\to C$ 保证每个中间输出正好具有下一层所需颜色，空纤维处的空乘积不改变论证。重标号自然性来自有限乘积坐标重排与纤维限制的相容性。$\square$

**定义 5.12.** 设 $\mathcal P$ 是 $C$-colored operad。一个集合值 $\mathcal P$-代数是 $C$-indexed 集合族 $A=(A_c)_{c\in C}$ 连同 colored operad morphism
$$
\alpha:\mathcal P\to\operatorname{End}_A.
$$

**展开 5.13.** 等价地，$\mathcal P$-代数给出每个运算
$$
p\in\mathcal P(S,\kappa;c)
$$
的具体函数
$$
\alpha(p):\prod_{s\in S}A_{\kappa(s)}\to A_c,
$$
并且这些函数对输入重标号、colored operad 代入和颜色单位相容。

**定义 5.14.** $\mathcal P$-代数同态 $F:A\to B$ 是函数族
$$
F_c:A_c\to B_c,\qquad c\in C,
$$
使得对任意 $p\in\mathcal P(S,\kappa;c)$ 和任意输入族 $(a_s)$，有
$$
F_c\big(\alpha_A(p)((a_s))\big)
=
\alpha_B(p)((F_{\kappa(s)}(a_s))_{s\in S}).
$$

## 5.5 Symmetric multicategory

**定义 5.15.** 一个 symmetric multicategory $\mathcal M$ 由以下数据组成：

- 对象集合 $\operatorname{Ob}(\mathcal M)$；
- 对任意有限集 $S$、对象族 $x:S\to\operatorname{Ob}(\mathcal M)$ 和对象 $y$，有集合
  $$
  \mathcal M((x_s)_{s\in S};y);
  $$
- 对任意对象 $x$，有单位元素 $\operatorname{id}_x\in\mathcal M((x);x)$；
- 对任意有限集映射 $f:S\to T$、外层多态射
  $$
  F\in\mathcal M((y_t)_{t\in T};z)
  $$
  和各纤维上的内层多态射
  $$
  G_t\in\mathcal M((x_s)_{s\in f^{-1}(t)};y_t),
  $$
  有复合
  $$
  F\circ(G_t)_{t\in T}
  \in
  \mathcal M((x_s)_{s\in S};z),
  $$
  满足重标号自然性、结合律和单位律。空纤维允许 nullary 多态射进入外层输入槽。

**命题 5.16.** $C$-colored operad 与对象集合固定为 $C$ 的 symmetric multicategory 是同一类数据。

**证明.** 给定 $C$-colored operad $\mathcal P$，定义
$$
\mathcal M((\kappa(s))_{s\in S};c)=\mathcal P(S,\kappa;c).
$$
colored operad 对全部有限集映射的单位和代入就是 multicategory 的单位和复合。

反过来，给定对象集合为 $C$ 的 symmetric multicategory $\mathcal M$，令
$$
\mathcal P(S,\kappa;c)=\mathcal M((\kappa(s))_{s\in S};c).
$$
重标号、单位和全部纤维复合由 $\mathcal M$ 的结构给出。两种构造在定义层面互逆；特别地，空输入轮廓与空纤维在两边对应同一 nullary 数据。$\square$

**说明 5.16.1.** 附录 K 给出 colored 轮廓群胚的骨架（命题 K.2）、colored substitution 的 coend 口径（定义 K.4--命题 K.6）、自由 colored operad 的树公式（定义 K.7--命题 K.9），以及代数同态、模和双模的生成元关系模型（定义 K.10--命题 K.15）。使用 colored operad 编码多对象代数系统时，默认采用附录 K 的这些约定。

## 5.6 基本例子

**例 5.17.** 任一小范畴 $\mathcal C$ 给出 colored operad $\mathcal P_{\mathcal C}$。颜色集合为 $\operatorname{Ob}(\mathcal C)$，并定义
$$
\mathcal P_{\mathcal C}(S,\kappa;c)=
\begin{cases}
\mathcal C(\kappa(s),c), & S=\{s\},\\
\varnothing, & |S|\ne1.
\end{cases}
$$
单位和代入由 $\mathcal C$ 的恒等态射与复合给出。

**命题 5.18.** $\mathcal P_{\mathcal C}$-代数等价于函子
$$
\mathcal C\to\mathbf{Set}_{\mathcal U}.
$$

**证明.** 一个 $\mathcal P_{\mathcal C}$-代数给出每个对象 $c$ 的集合 $A_c$，以及每个态射 $f:x\to y$ 的函数 $A_x\to A_y$。colored operad 单位律给出恒等态射作用为恒等函数，代入结合律给出复合态射作用为复合函数。因此得到函子。反向构造由同一公式给出。$\square$

**例 5.19.** 令颜色集合 $C=\{L,X,R\}$。可以定义一个 colored operad 编码“一个幺半群 $A$、一个幺半群 $B$、一个带左 $A$-作用和右 $B$-作用的集合 $X$”的数据：颜色 $L$ 上放左幺半群乘法，颜色 $R$ 上放右幺半群乘法，颜色 $X$ 上放来自 $L,X,R$ 的动作
$$
L,X\to X,\qquad X,R\to X,
$$
并加入结合、单位和左右动作交换关系。严格地说，这个 operad 可由第四章生成元关系方法的 colored 版本构造。若要编码双模或双线性作用，需要在线性 colored operad 或 enriched colored operad 中工作。

**例 5.20.** 一个带有一个指定代数同态的结构也可由 colored operad 编码。取颜色 $A,B$，在 $A$ 和 $B$ 上各放一个 $\operatorname{Ass}$ 型乘法与单位，再加入一元生成元 $f:A\to B$，并施加关系
$$
f(m_A(x,y))=m_B(f(x),f(y)),\qquad f(e_A)=e_B.
$$
于是代数正是两个幺半群及一个幺半群同态。

## 5.7 颜色把可复合性写进语法

颜色把“能否代入”从附加说明变成结构的一部分：有限集映射仍控制输入分块，但每个内层输出还必须等于相应外层输入颜色。由此，普通范畴成为只有一元运算的 colored operad，作用、双模与代数同态则需要真正的多输入轮廓。与 symmetric multicategory 的等价说明这不是另一套对象，而是同一数据的幺半写法和多箭头写法。

这套类型系统随后会向两个方向扩张。第六章把每个运算集合线性化；第七章允许一个运算有多个输出。若再进入 enriched 或模型范畴语境，还必须另外验证张量、coinvariants 与 admissibility，不能由本章的集合值构造自动推出。

## 练习

**练习 5.1.** 当 $C$ 是单点集时，逐项证明定义 5.4 退化为第一章的代入乘积。

**练习 5.2.** 对定义 5.10 的 colored endomorphism operad，写出可复合映射 $S\to U\to T$ 时结合律对应的函数等式，并允许两层都有空纤维。

**练习 5.3.** 给定小范畴 $\mathcal C$，验证例 5.17 的 $\mathcal P_{\mathcal C}$ 中没有非一元运算。

**练习 5.4.** 用 colored operad 的生成元关系写出“一个集合 $X$、一个幺半群 $M$ 和一个右作用 $X\times M\to X$”。

**练习 5.5.** 说明为什么普通单色 operad 不能直接编码“一个同态 $A\to B$”，而 colored operad 可以。
