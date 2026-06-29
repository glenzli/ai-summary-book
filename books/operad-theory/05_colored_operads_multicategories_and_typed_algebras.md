# 第五章：Colored operad、多范畴与带类型代数

## 本章目标

本章把单色 operad 推广到带颜色或带类型的 operad。颜色集合记录对象类型；一个运算不再只有 arity，而是有若干输入颜色和一个输出颜色。核心目标是：

1. 定义 $C$-colored symmetric sequence。
2. 用有限集分块定义 colored substitution product。
3. 把 $C$-colored operad 定义为幺半对象。
4. 定义 colored endomorphism operad 和 colored algebra。
5. 说明 colored operad 与 symmetric multicategory 的等价关系。
6. 给出范畴、带作用集合和代数态射的基本例子。

## 依赖前置知识

需要第一章的有限集口径、对称序列、代入乘积和 endomorphism operad。需要知道范畴由对象、态射、恒等态射和复合组成。

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
\coprod_{\pi\in\operatorname{Part}(S)}
\coprod_{\delta:\operatorname{Bl}(\pi)\to C}
X(\operatorname{Bl}(\pi),\delta;c)
\times
\prod_{B\in\operatorname{Bl}(\pi)}
Y(B,\kappa|_B;\delta(B)).
$$
这里 $\delta(B)$ 是块 $B$ 上内层运算的输出颜色，同时也是外层运算对应输入槽的颜色。

若 $\varphi:(S,\kappa;c)\to(T,\lambda;c)$ 是 $\mathbf B_C$ 中的同构，则 $\varphi$ 推前分块，并把块 $B$ 送到 $\varphi(B)$。颜色函数 $\delta$ 被送到
$$
\varphi_\*\delta(\varphi(B))=\delta(B).
$$
结合 $X$ 和 $Y$ 的函子性，得到 $(X\circ_CY)(\varphi)$。

**定义 5.5.** 单位 colored symmetric sequence $I_C$ 定义为
$$
I_C(S,\kappa;c)=
\begin{cases}
\{*\}, & S=\{s\}\text{ 且 }\kappa(s)=c,\\
\varnothing, & \text{否则}.
\end{cases}
$$

**命题 5.6.** $\operatorname{SymSeq}_C$ 连同 $\circ_C$ 和 $I_C$ 构成幺半范畴。

**证明.** 证明与命题 1.8 相同，但需要记录颜色。一个元素
$$
((x;(y_B));(z_D))
\in ((X\circ_CY)\circ_CZ)(S,\kappa;c)
$$
等价于以下数据：

- $S$ 的粗分块 $\pi$；
- 每个粗块 $B$ 的细分块 $\rho_B$；
- 粗块输出颜色函数 $\delta:\operatorname{Bl}(\pi)\to C$；
- 细块输出颜色函数 $\epsilon_B:\operatorname{Bl}(\rho_B)\to C$；
- 外层元素 $x\in X(\operatorname{Bl}(\pi),\delta;c)$；
- 中层元素 $y_B\in Y(\operatorname{Bl}(\rho_B),\epsilon_B;\delta(B))$；
- 底层元素 $z_D\in Z(D,\kappa|_D;\epsilon_B(D))$。

把所有细块组成 $S$ 的分块 $\rho$，并把 $\operatorname{Bl}(\rho)$ 按粗块分块。细块颜色由各 $\epsilon_B$ 拼接得到。于是同一数据也给出
$$
(X\circ_C(Y\circ_CZ))(S,\kappa;c)
$$
的元素。该对应有反构造，且关于轮廓同构自然，所以给出结合约束。单位约束来自单点块，并且颜色条件 $\kappa(s)=c$ 保证单位只连接相同颜色的输入和输出。五边形与三角形仍然是多层分块拉平次序无关性，加上颜色函数拼接的结合律。$\square$

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

**展开 5.8.** 给定分块 $\pi$、块输出颜色 $\delta:\operatorname{Bl}(\pi)\to C$、外层运算
$$
p\in\mathcal P(\operatorname{Bl}(\pi),\delta;c)
$$
和内层运算
$$
p_B\in\mathcal P(B,\kappa|_B;\delta(B)),
$$
乘法给出
$$
\mu_\pi(p;(p_B))\in\mathcal P(S,\kappa;c).
$$
operad 结合律断言多层带颜色代入的结果不依赖先代入哪一层；单位律断言每个颜色 $c$ 有一个一元恒等运算
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

代入如下。设 $\pi$ 是 $S$ 的分块，$\delta:\operatorname{Bl}(\pi)\to C$。给定
$$
f:\prod_{B\in\operatorname{Bl}(\pi)}A_{\delta(B)}\to A_c
$$
和
$$
g_B:\prod_{s\in B}A_{\kappa(s)}\to A_{\delta(B)},
$$
定义
$$
h:\prod_{s\in S}A_{\kappa(s)}\to A_c
$$
为
$$
h((a_s)_{s\in S})
=
f\big((g_B((a_s)_{s\in B}))_{B\in\operatorname{Bl}(\pi)}\big).
$$

**命题 5.11.** $\operatorname{End}_A$ 是 $C$-colored operad。

**证明.** 单位 $\mathbf 1_c$ 是恒等函数 $A_c\to A_c$。结合律是函数复合的结合律：三层分块时，对输入族 $(a_s)$ 的最终求值表达式相同。颜色条件保证每个中间函数的输出集合正好是外层函数所需的输入集合。重标号自然性来自有限乘积坐标重排与限制到块的相容性。$\square$

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
- 对任意分块 $\pi$、外层多态射
  $$
  f\in\mathcal M((y_B)_{B\in\operatorname{Bl}(\pi)};z)
  $$
  和内层多态射
  $$
  g_B\in\mathcal M((x_s)_{s\in B};y_B),
  $$
  有复合
  $$
  f\circ(g_B)_{B\in\operatorname{Bl}(\pi)}
  \in
  \mathcal M((x_s)_{s\in S};z),
  $$
  满足重标号自然性、结合律和单位律。

**命题 5.16.** $C$-colored operad 与对象集合固定为 $C$ 的 symmetric multicategory 是同一类数据。

**证明.** 给定 $C$-colored operad $\mathcal P$，定义
$$
\mathcal M((\kappa(s))_{s\in S};c)=\mathcal P(S,\kappa;c).
$$
colored operad 的单位和代入就是 multicategory 的单位和复合。

反过来，给定对象集合为 $C$ 的 symmetric multicategory $\mathcal M$，令
$$
\mathcal P(S,\kappa;c)=\mathcal M((\kappa(s))_{s\in S};c).
$$
重标号、单位和复合由 $\mathcal M$ 的结构给出。两种构造在定义层面互逆。$\square$

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

## 本章小结

Colored operad 是单色 operad 的带类型版本。颜色约束在代入时要求：内层运算的输出颜色必须等于外层输入槽的颜色。它与 symmetric multicategory 是同一数据。这个语言可以同时编码范畴、函子、带作用集合、代数同态和多对象代数系统；在线性或 enriched 版本中，它还能编码模、双模和双线性结构。因此 colored operad 是第七章 PROP/properad、附录 K enriched colored operad 和第十八章 infinity-operad 的必要中间层。

Enriched colored operad 的模型结构不由本章定义自动给出；需要附录 G 的模型范畴假设，并需使用警告 K.19 与外部输入定理 K.20 中的 admissibility 边界。

## 练习

**练习 5.1.** 当 $C$ 是单点集时，逐项证明定义 5.4 退化为第一章的代入乘积。

**练习 5.2.** 对定义 5.10 的 colored endomorphism operad，写出三层分块时结合律对应的函数等式。

**练习 5.3.** 给定小范畴 $\mathcal C$，验证例 5.17 的 $\mathcal P_{\mathcal C}$ 中没有非一元运算。

**练习 5.4.** 用 colored operad 的生成元关系写出“一个集合 $X$、一个幺半群 $M$ 和一个右作用 $X\times M\to X$”。

**练习 5.5.** 说明为什么普通单色 operad 不能直接编码“一个同态 $A\to B$”，而 colored operad 可以。
