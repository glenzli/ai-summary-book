# 第十二章：Brace operad 与 Hochschild cochains

## 本章目标

本章给出 Deligne 猜想的一条具体 operadic 路线：Hochschild cochains 上的 brace operations。目标是：

1. 定义 Hochschild cochain complex。
2. 定义 cup product、insertion 和 Gerstenhaber bracket。
3. 定义 brace operations。
4. 说明 brace identities 与 brace operad。
5. 陈述 brace operad 与 $E_2$-operad 的弱等价作为外部输入。

## 依赖前置知识

需要第十一章的 Gerstenhaber 结构和 Deligne 猜想。本章先写普通未分次结合代数的 cochain 公式；分次版本采用定义 E.18--定义 E.23 的 suspended Hochschild braces 约定。

## 12.1 Hochschild cochains

**约定 12.1.** 本章中 $A$ 是未分次结合 $k$-代数。Hochschild cochains 为
$$
C^n(A,A)=\operatorname{Hom}_k(A^{\otimes n},A),\qquad n\ge0.
$$
一个 $f\in C^m(A,A)$ 的 cohomological degree 为 $m$，其 brace degree 定义为
$$
|f|_{\mathrm{br}}=m-1.
$$

**定义 12.2.** Hochschild differential $\delta:C^n(A,A)\to C^{n+1}(A,A)$ 定义为
$$
\begin{aligned}
(\delta f)(a_1,\ldots,a_{n+1})
&=
a_1 f(a_2,\ldots,a_{n+1})\\
&\quad+\sum_{i=1}^{n}(-1)^i
f(a_1,\ldots,a_i a_{i+1},\ldots,a_{n+1})\\
&\quad+(-1)^{n+1}f(a_1,\ldots,a_n)a_{n+1}.
\end{aligned}
$$

**命题 12.3.** $\delta^2=0$。

**证明.** 展开 $\delta(\delta f)$。每一项对应在长度 $n+2$ 的输入序列中执行两次操作：左乘、合并相邻输入、右乘。任意两次操作的结果都以相反符号出现两次。例如先合并 $(a_i,a_{i+1})$ 再合并所得序列中的下一对，与先合并 $(a_{i+1},a_{i+2})$ 再合并前一对给出同一乘积项，符号相反；涉及首尾乘法的项也由结合律成对相消。因此 $\delta^2=0$。$\square$

## 12.2 Cup product 与 insertion

**定义 12.4.** 对 $f\in C^m(A,A)$、$g\in C^n(A,A)$，cup product 定义为
$$
(f\smile g)(a_1,\ldots,a_{m+n})
=
f(a_1,\ldots,a_m)g(a_{m+1},\ldots,a_{m+n}).
$$

**定义 12.5.** 对 $f\in C^m(A,A)$、$g\in C^n(A,A)$ 和 $0\le i\le m-1$，定义 insertion
$$
f\circ_i g\in C^{m+n-1}(A,A)
$$
为
$$
\begin{aligned}
&(f\circ_i g)(a_1,\ldots,a_{m+n-1})\\
&=
f(a_1,\ldots,a_i,
g(a_{i+1},\ldots,a_{i+n}),
a_{i+n+1},\ldots,a_{m+n-1}).
\end{aligned}
$$
定义总 insertion
$$
f\circ g
=
\sum_{i=0}^{m-1}(-1)^{(n-1)i}f\circ_i g.
$$

**定义 12.6.** Gerstenhaber bracket 在 cochains 上定义为
$$
[f,g]
=
f\circ g-(-1)^{(m-1)(n-1)}g\circ f.
$$

**命题 12.7.** 在 cohomology 上，cup product 和 bracket 诱导 Gerstenhaber algebra 结构。

**证明.** Insertion 的偏复合满足非对称 operad 的偏复合恒等式，带符号总 insertion 因此给出 graded pre-Lie product。其 graded commutator 满足 graded Jacobi 恒等式。Hochschild differential 可写为与乘法 cochain $\mu\in C^2(A,A)$ 的 bracket：
$$
\delta f=[\mu,f]
$$
在标准符号约定下成立。由结合律 $[\mu,\mu]=0$ 得 $\delta^2=0$。Cup product 与 bracket 的 Leibniz 相容性在 cohomology 上成立，边界项由 brace 公式控制。因此得到 Gerstenhaber algebra。$\square$

## 12.3 Brace operations

**定义 12.8.** 设 $f\in C^m(A,A)$，$g_j\in C^{n_j}(A,A)$，$1\le j\le r$。Brace operation
$$
f\{g_1,\ldots,g_r\}
\in
C^{m-r+n_1+\cdots+n_r}(A,A)
$$
定义为所有按顺序把 $g_1,\ldots,g_r$ 插入 $f$ 的输入槽中的和：
$$
\begin{aligned}
&f\{g_1,\ldots,g_r\}(a_1,\ldots,a_N)\\
&=
\sum (-1)^\epsilon
f(a_1,\ldots,
g_1(a_{i_1+1},\ldots,a_{i_1+n_1}),
\ldots,
g_r(a_{i_r+1},\ldots,a_{i_r+n_r}),
\ldots,a_N),
\end{aligned}
$$
其中
$$
N=m-r+n_1+\cdots+n_r.
$$
求和遍历所有互不重叠且按顺序出现的插入位置。未分次代数的标准符号可取
$$
\epsilon=\sum_{j=1}^r(n_j-1)i_j,
$$
其中 $i_j$ 是第 $j$ 个插入块之前未被插入块占据的输入数。若 $r=0$，定义 $f\{\}=f$。

**例 12.9.** 当 $r=1$ 时，
$$
f\{g\}=f\circ g.
$$
因此 brace operation 同时推广了 insertion 和 Gerstenhaber bracket 的构造。

**定义 12.10.** 一个 brace algebra 是分次向量空间 $B$，配有多元运算
$$
x\{y_1,\ldots,y_r\},
$$
满足 brace identity：
$$
\begin{aligned}
&(x\{x_1,\ldots,x_m\})\{y_1,\ldots,y_n\}\\
&=
\sum
\pm
x\{y_1,\ldots,
x_1\{y_{i_1},\ldots\},
\ldots,
x_m\{y_{i_m},\ldots\},
\ldots,y_n\},
\end{aligned}
$$
其中求和遍历把 $y_1,\ldots,y_n$ 按顺序分配到 $x_1,\ldots,x_m$ 的输入间隙和外部间隙的所有方式。符号由 brace degree 的 Koszul rule 给出。

**命题 12.11.** Hochschild cochains $C^\*(A,A)$ 连同定义 12.8 的 operations 构成 brace algebra。

**证明.** 左侧先把 $x_i$ 插入 $x$，再把 $y_j$ 插入所得表达式。每个 $y_j$ 最终要么落在某个 $x_i$ 内部，要么落在 $x$ 的外部空隙中。按这个最终位置分类，正得到右侧求和。符号由移动插入块经过已有输入槽时的 brace degree 贡献给出，和定义 12.8 的 $\epsilon$ 相容。$\square$

## 12.4 Brace operad

**定义 12.12.** Brace operad $\operatorname{Br}$ 是控制 brace algebras 的 dg-operad。它可用 rooted brace trees 描述：arity $r$ 的元素由一个根顶点和 $r$ 个标号输入顶点组成的有序插入树生成，operad 代入由把 brace tree 插入顶点并展开所有保持顺序的插入方式给出。

**命题 12.13.** Hochschild cochains 上的 brace operations 给出 dg-operad morphism
$$
\operatorname{Br}\to\operatorname{End}_{C^\*(A,A)}.
$$

**证明.** Brace identity 正是 $\operatorname{Br}$ 的 operad 代入关系在 endomorphism operad 中成立的断言。Hochschild differential 与 braces 的相容性由 $\delta=[\mu,-]$ 和 associativity $[\mu,\mu]=0$ 控制，因此该 morphism 与微分相容。$\square$

## 12.5 Deligne 猜想的 brace 路线

**外部输入定理 12.14.** Brace operad $\operatorname{Br}$ 与链级 $E_2$-operad 弱等价。更精确地，存在 dg-operad 的 zigzag quasi-isomorphisms
$$
\operatorname{Br}\simeq C_\*(E_2;k)
$$
或与等价的 little disks/cubes 链模型相连。

该定理有多种证明版本，涉及 McClure-Smith、Berger-Fresse、Kontsevich-Soibelman、Tamarkin 等工作。

**推论 12.15.** Hochschild cochains $C^\*(A,A)$ 具有自然的 $E_2$-algebra 结构，且在 cohomology 上诱导经典 Gerstenhaber algebra 结构。

**证明.** 命题 12.13 给出 $\operatorname{Br}$-algebra 结构。由外部输入定理 12.14，$\operatorname{Br}$ 是 $E_2$ 的链模型，因此该结构即为 $E_2$-algebra 结构。取 cohomology 后，brace 的一元插入给出 Gerstenhaber bracket，cup product 给出乘法；这与第十一章定理 11.11 的经典结构一致。$\square$

## 12.6 分次版本的 suspended 约定

**约定 12.16.** 若 $A$ 是 dg associative algebra，本书把分次 brace operations 定义在 suspended Hochschild cochains
$$
\widetilde C^p(A,A)=\underline{\operatorname{Hom}}\big((sA)^{\otimes p},sA\big)
$$
上。对 $F\in\widetilde C^p(A,A)$、$G_j\in\widetilde C^{q_j}(A,A)$，brace
$$
F\{G_1,\ldots,G_r\}
$$
按附录 E 定义 E.20 计算。未悬挂 cochain $f:A^{\otimes p}\to A$ 先送到
$$
\widetilde f=s\circ f\circ(s^{-1})^{\otimes p}
$$
再参与 brace 运算。

**命题 12.17.** 当所有输入集中在内部次数 $0$ 且忽略悬挂记号时，约定 12.16 退化为定义 12.8 的未分次 brace 公式。

**证明.** 若 $f\in C^m(A,A)$、$g_j\in C^{n_j}(A,A)$ 内部次数为 $0$，则
$$
|\widetilde g_j|=1-n_j.
$$
Suspended 输入 $sa$ 的次数为 $1$。附录 E 的符号指数
$$
\sum_j|\widetilde g_j|(|x_{c_1}|+\cdots+|x_{c_{i_j}}|)
$$
在模 $2$ 下等于
$$
\sum_j(n_j-1)i_j.
$$
这里 $c_1,\ldots,c_{p-r}$ 是未被插入块占据并直接进入外层 $f$ 的输入位置，$i_j$ 是第 $j$ 个插入块之前这样的输入个数。因此得到定义 12.8 的符号
$$
\epsilon=\sum_j(n_j-1)i_j.
$$
悬挂和去悬挂只改变 arity bookkeeping，不改变插入位置的求和集合。因此得到未分次 brace 公式。$\square$

**命题 12.18.** 在 dg 情形下，$\widetilde C^\*(A,A)$ 连同约定 12.16 的 operations 构成 brace algebra。

**证明.** 这是命题 E.22 的直接应用。该命题的证明只使用 suspended inputs 上的 Koszul sign rule 和插入位置的最终分类，因此适用于任意 dg associative algebra。$\square$

**说明 12.19.** Gerstenhaber bracket 的分次符号由 suspended bracket
$$
[F,G]_{\operatorname{sus}}=F\widetilde\circ G-(-1)^{|F||G|}G\widetilde\circ F
$$
给出。转回内部次数为 $0$ 的未分次 cochains 时，$|F|=1-m$、$|G|=1-n$，所以交换符号为 $(-1)^{(m-1)(n-1)}$，与定义 12.6 一致。

## 12.7 边界和后续

**警告 12.20.** Brace operad 作用给出的是 Hochschild cochains 上的链级结构。若只看 $HH^\*(A,A)$，会丢失大量高阶 brace operations 和同伦信息。

**说明 12.21.** 对 $A_\infty$-algebra、dg category、monoidal category 或 stable infinity-category，也有 Hochschild cochains 和 Deligne 型结构的推广。这些推广需要 colored operad、dg category 或 infinity-categorical Hochschild theory；本书后续只在建立相应模型后使用。

## 本章小结

Hochschild cochains 上的 insertion 给出 Gerstenhaber bracket，brace operations 组织所有高阶插入相干性。Brace operad 控制这些运算，并与链级 $E_2$-operad 弱等价。因此 Deligne 猜想可以理解为：Hochschild cochains 自然是 brace algebra，而 brace algebra 是一个具体的 $E_2$-algebra 模型。

## 练习

**练习 12.1.** 对 $f\in C^2(A,A)$、$g\in C^2(A,A)$，写出 $f\circ g$ 的两个插入项及其符号。

**练习 12.2.** 证明 $\delta f=[\mu,f]$，其中 $\mu$ 是乘法 cochain。

**练习 12.3.** 对 $f\{g,h\}$，写出所有可能插入位置。

**练习 12.4.** 用 brace identity 解释为什么两次插入的不同加括号方式给出相同总和。

**练习 12.5.** 说明推论 12.15 中哪一步使用了外部输入定理。
