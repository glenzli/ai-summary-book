# 第一章：依赖类型论的判断与结构规则

## 本章目标

本章建立后续 HoTT 所需的最小判断框架：语境、类型判断、项判断、judgmental equality、替换、宇宙、依赖函数类型和依赖对类型。我们只写内部规则，不使用函数外延性、单值性或高阶归纳类型。

## 依赖前置知识

本章依赖序章的基础口径和符号约定。读者只需熟悉普通函数、变量替换和基本逻辑推理。

## 1.1 判断

**定义 1.1.** 本书使用四类基本判断。

1.  $\Gamma\ \mathsf{ctx}$：$\Gamma$ 是合法语境。
2.  $\Gamma\vdash A:\mathcal U_i$：在语境 $\Gamma$ 中，$A$ 是第 $i$ 层宇宙中的类型。
3.  $\Gamma\vdash a:A$：在语境 $\Gamma$ 中，$a$ 是类型 $A$ 的项。
4.  $\Gamma\vdash a\equiv b:A$：在语境 $\Gamma$ 中，$a$ 与 $b$ judgmentally equal。

这些判断属于元语言。内部命题以后会表示为类型；但“$\Gamma\vdash a:A$ 成立”本身不是当前类型论中的一个项。

**定义 1.2.** 语境由有限变量列表组成：

$$
x_1:A_1,\ x_2:A_2(x_1),\ \ldots,\ x_n:A_n(x_1,\ldots,x_{n-1}).
$$

它的合法性要求每个 $A_k$ 在前面的语境中已经是类型。

**例 1.3.** 若 $A:\mathcal U_i$ 且 $B:A\to\mathcal U_j$，则

$$
x:A,\ y:B(x)
$$

是语境。这里 $B(x)$ 可以依赖于前一个变量 $x$。

## 1.2 结构规则

**规则 1.4（空语境）.** 空列表是合法语境，记为 $\cdot\ \mathsf{ctx}$。

**规则 1.5（语境扩张）.** 若 $\Gamma\ \mathsf{ctx}$ 且 $\Gamma\vdash A:\mathcal U_i$，则

$$
\Gamma,x:A\ \mathsf{ctx}.
$$

变量名要求相对于 $\Gamma$ 新鲜。

**规则 1.6（变量）.** 若 $\Gamma,x:A,\Delta\ \mathsf{ctx}$，则在该语境中有

$$
\Gamma,x:A,\Delta\vdash x:A.
$$

严格说，若 $\Delta$ 中后续变量依赖于 $x$，则 $A$ 在扩张后通过 weakening 仍被视为类型。

**规则 1.7（替换）.** 若

$$
\Gamma,x:A,\Delta\vdash J
$$

是一个判断，且 $\Gamma\vdash a:A$，则可以形成把 $a$ 替换给 $x$ 后的判断

$$
\Gamma,\Delta[a/x]\vdash J[a/x].
$$

这里 $J$ 可为类型判断、项判断或 judgmental equality 判断。

**命题 1.8（替换保持类型与项）.** 若 $\Gamma,x:A\vdash B:\mathcal U_j$ 且 $\Gamma\vdash a:A$，则 $\Gamma\vdash B[a/x]:\mathcal U_j$。若进一步 $\Gamma,x:A\vdash b:B$，则 $\Gamma\vdash b[a/x]:B[a/x]$。

**证明（书内规则说明）.** 这是替换规则在类型判断和项判断上的两个特例。第一部分取 $J$ 为 $B:\mathcal U_j$，第二部分取 $J$ 为 $b:B$。$\square$

## 1.3 Judgmental equality

**定义 1.9.** Judgmental equality $a\equiv b:A$ 是类型检查层面的相等。它不是一个需要项证明的类型，而是表达式按规则计算后被系统视为同一项。

本书要求 judgmental equality 至少满足以下结构性质：

1.  反身性：若 $\Gamma\vdash a:A$，则 $\Gamma\vdash a\equiv a:A$。
2.  对称性：若 $\Gamma\vdash a\equiv b:A$，则 $\Gamma\vdash b\equiv a:A$。
3.  传递性：若 $\Gamma\vdash a\equiv b:A$ 且 $\Gamma\vdash b\equiv c:A$，则 $\Gamma\vdash a\equiv c:A$。
4.  合同性：类型和项构造尊重 judgmental equality。
5.  替换稳定性：若 $a\equiv b:A$，则把 $a$ 或 $b$ 替入同一表达式得到 judgmentally equal 的结果。

**警告 1.10.** $a\equiv b:A$ 与 $a=_A b$ 不同。前者是判断，后者是类型。若 $p:a=_A b$，则 $p$ 是一条路径；它一般不使 $a$ 与 $b$ judgmentally equal。

## 1.4 宇宙

**规则 1.11（宇宙）.** 对每个自然数 $i$，有一个宇宙 $\mathcal U_i$。若 $\Gamma\ \mathsf{ctx}$，则

$$
\Gamma\vdash \mathcal U_i:\mathcal U_{i+1}.
$$

**约定 1.12.** 若 $\Gamma\vdash A:\mathcal U_i$，也说 $A$ 是一个 $i$-小类型。本书默认追踪宇宙层级；若公式中省略层级，是因为层级可由上下文恢复，而不是因为不存在大小问题。

**警告 1.13.** 本章不假设 $\mathcal U_i:\mathcal U_i$，否则会引入 Girard 悖论风险。也不默认 universe resizing。

## 1.5 依赖函数类型

**规则 1.14（$\Pi$ 形成）.** 若 $\Gamma\vdash A:\mathcal U_i$ 且 $\Gamma,x:A\vdash B(x):\mathcal U_j$，则

$$
\Gamma\vdash \prod_{x:A}B(x):\mathcal U_{\max(i,j)}.
$$

**规则 1.15（$\Pi$ 引入）.** 若 $\Gamma,x:A\vdash b(x):B(x)$，则

$$
\Gamma\vdash \lambda x.\,b(x):\prod_{x:A}B(x).
$$

**规则 1.16（$\Pi$ 消去）.** 若 $\Gamma\vdash f:\prod_{x:A}B(x)$ 且 $\Gamma\vdash a:A$，则

$$
\Gamma\vdash f(a):B(a).
$$

**规则 1.17（$\Pi$ 计算）.** 在上述条件下，

$$
(\lambda x.\,b(x))(a)\equiv b(a):B(a).
$$

这是 judgmental beta 规则。

**定义 1.18.** 若 $B$ 不依赖于 $x:A$，则 $\prod_{x:A}B$ 记为 $A\to B$。

**例 1.19.** 若 $A:\mathcal U_i$，则恒等函数定义为

$$
\mathsf{id}_A\coloneqq \lambda x.\,x:A\to A.
$$

对任意 $a:A$，由 $\Pi$ 计算规则有 $\mathsf{id}_A(a)\equiv a:A$。

**警告 1.20.** 本章不假设函数外延性。也就是说，从

$$
\prod_{x:A}(f(x)=g(x))
$$

不能在本章推出 $f=g$，除非以后引入函数外延性或相关原则。

## 1.6 依赖对类型

**规则 1.21（$\Sigma$ 形成）.** 若 $\Gamma\vdash A:\mathcal U_i$ 且 $\Gamma,x:A\vdash B(x):\mathcal U_j$，则

$$
\Gamma\vdash \sum_{x:A}B(x):\mathcal U_{\max(i,j)}.
$$

**规则 1.22（$\Sigma$ 引入）.** 若 $\Gamma\vdash a:A$ 且 $\Gamma\vdash b:B(a)$，则

$$
\Gamma\vdash (a,b):\sum_{x:A}B(x).
$$

**规则 1.23（$\Sigma$ 投影）.** 若 $\Gamma\vdash z:\sum_{x:A}B(x)$，则有

$$
\Gamma\vdash \mathsf{pr}_1(z):A,
$$

并且

$$
\Gamma\vdash \mathsf{pr}_2(z):B(\mathsf{pr}_1(z)).
$$

**规则 1.24（$\Sigma$ 计算）.** 若 $z\equiv(a,b)$，则

$$
\mathsf{pr}_1(a,b)\equiv a:A
$$

且

$$
\mathsf{pr}_2(a,b)\equiv b:B(a).
$$

**定义 1.25.** 若 $B$ 不依赖于 $x:A$，则 $\sum_{x:A}B$ 记为 $A\times B$。

**命题 1.26（非依赖二元积的消去）.** 若 $C:\mathcal U_k$，且在语境 $x:A,y:B$ 中有 $c(x,y):C$，则存在函数

$$
h:A\times B\to C
$$

使得 $h(a,b)\equiv c(a,b)$。

**证明.** 定义

$$
h\coloneqq \lambda z.\,c(\mathsf{pr}_1(z),\mathsf{pr}_2(z)).
$$

若 $z$ 是构造项 $(a,b)$，则由 $\Sigma$ 投影计算规则，

$$
h(a,b)\equiv c(\mathsf{pr}_1(a,b),\mathsf{pr}_2(a,b))\equiv c(a,b).
$$

这只使用 $\Pi$ 引入、$\Pi$ 计算和 $\Sigma$ 计算。$\square$

## 1.7 本章不使用的原则

本章的所有构造都只依赖判断规则、结构规则、宇宙规则、$\Pi$ 型和 $\Sigma$ 型。特别地，尚未使用：

- 恒等类型；
- path induction；
- 函数外延性；
- 单值性；
- 高阶归纳类型；
- 命题截断；
- classical logic。

这种限制是有意的。后续每增加一种原则，都要能指出它第一次进入本书的位置。

## 本章小结

本章建立了 HoTT 的语法基础：合法语境、类型、项、judgmental equality、替换、宇宙、依赖函数和依赖对。下一章将引入恒等类型；那时 $a=_A b$ 将成为内部类型，而不只是元层判断。

## 练习

**练习 1.1.** 在语境 $A:\mathcal U_i,\ B:A\to\mathcal U_j$ 中，写出函数
$$
\lambda z.\,\mathsf{pr}_1(z):\left(\sum_{x:A}B(x)\right)\to A
$$
的类型检查过程。

**练习 1.2.** 证明若 $\Gamma\vdash f:A\to B$ 且 $\Gamma\vdash a\equiv a':A$，则 $\Gamma\vdash f(a)\equiv f(a'):B$。说明使用了 judgmental equality 的哪条结构性质。

**练习 1.3.** 解释为什么 $\lambda x.\,f(x)\equiv f$ 不应在本章自动作为 judgmental equality 使用。

**练习 1.4.** 给出 $A\times B\to B\times A$ 的定义，并计算它作用在 $(a,b)$ 上的 judgmental normal form。
