# 第三章：基础归纳类型与命题作为类型

依赖函数和依赖对只告诉我们怎样组合已有类型，还没有给出自然数、真假值或“不可能”的具体对象。归纳类型用构造子规定数据从何而来，再用消去原则规定怎样使用这些数据。自然数的两个计算式因而不是对外部整数的描述，而是递归函数在 $0$ 与 $\mathsf{succ}$ 上的 judgmental 行为。

同一套机制也解释逻辑。空类型没有构造子，所以从其项可以消去到任意目标；和类型保留左右分支，所以一个析取证明携带实际选择。本章在第一、二章的 $\Pi$、$\Sigma$ 和路径规则上加入这些基础归纳类型，并由具体计算说明“命题作为类型”既表达真值，也保留证明数据。

## 3.1 空类型与单位类型

**规则 3.1（空类型形成）.** 在任意语境 $\Gamma$ 中，有
$$
\Gamma\vdash \mathbf 0:\mathcal U_0.
$$

**规则 3.2（空类型消去）.** 若 $\Gamma,x:\mathbf 0\vdash C(x):\mathcal U_i$，则有
$$
\Gamma,x:\mathbf 0\vdash \mathsf{exfalso}_C(x):C(x).
$$

空类型没有引入规则。

**规则 3.3（单位类型）.** 在任意语境 $\Gamma$ 中，有 $\Gamma\vdash \mathbf 1:\mathcal U_0$，并有唯一规范构造子
$$
\Gamma\vdash \star:\mathbf 1.
$$
其消去规则说：若 $\Gamma,x:\mathbf 1\vdash C(x):\mathcal U_i$，并且 $\Gamma\vdash c:C(\star)$，则可定义
$$
\Gamma,x:\mathbf 1\vdash \mathsf{ind}_{\mathbf 1}(c,x):C(x)
$$
并满足 $\mathsf{ind}_{\mathbf 1}(c,\star)\equiv c$。

**命题 3.4（单位类型有指定中心）.** 存在项
$$
(\star,\lambda x.\,\mathsf{ind}_{\mathbf 1}
(\mathsf{refl}_\star,x)):
\sum_{c:\mathbf 1}\prod_{x:\mathbf 1}(c=x).
$$
第四章把右侧类型记为 $\mathsf{isContr}(\mathbf 1)$。

**证明（书内证明）.** 第一分量取 $\star$。第二分量需给出
$$
\prod_{x:\mathbf 1}(\star=x).
$$
对 $x:\mathbf 1$ 作单位类型归纳。归纳后的目标为 $\star=\star$，取 $\mathsf{refl}_\star$。$\square$

## 3.2 和类型

**规则 3.5（和类型形成）.** 若 $\Gamma\vdash A:\mathcal U_i$ 且 $\Gamma\vdash B:\mathcal U_j$，则
$$
\Gamma\vdash A+B:\mathcal U_{\max(i,j)}.
$$

**规则 3.6（和类型引入）.** 若 $a:A$，则 $\mathsf{inl}(a):A+B$。若 $b:B$，则 $\mathsf{inr}(b):A+B$。

**规则 3.7（和类型消去）.** 若 $C:A+B\to\mathcal U_k$，并给出
$$
\prod_{a:A}C(\mathsf{inl}(a)),\qquad
\prod_{b:B}C(\mathsf{inr}(b)),
$$
则得到 $\prod_{z:A+B}C(z)$，并在两个构造子上满足对应 beta 计算规则。

**定义 3.8.** 否定定义为
$$
\neg A\coloneqq A\to\mathbf 0.
$$
合取可由 $\Sigma$ 型表示，析取可由和类型表示，存在量词可由 $\Sigma$ 型表示，全称量词可由 $\Pi$ 型表示。

**命题 3.9（爆炸律）.** 对任意类型 $A$，有函数 $\mathbf 0\to A$。

**证明（书内证明）.** 令 $x:\mathbf 0$，对 $x$ 使用空类型消去，目标族为常值族 $A$。$\square$

## 3.3 自然数

**规则 3.10（自然数形成与引入）.** 有类型 $\mathbb N:\mathcal U_0$，并有构造子
$$
0:\mathbb N,\qquad \mathsf{succ}:\mathbb N\to\mathbb N.
$$

**规则 3.11（自然数归纳）.** 若 $C:\mathbb N\to\mathcal U_i$，并给出
$$
c_0:C(0),\qquad c_s:\prod_{n:\mathbb N}(C(n)\to C(\mathsf{succ}(n))),
$$
则得到
$$
\mathsf{ind}_{\mathbb N}(C,c_0,c_s):\prod_{n:\mathbb N}C(n)
$$
并满足
$$
\mathsf{ind}_{\mathbb N}(C,c_0,c_s,0)\equiv c_0,
$$
$$
\mathsf{ind}_{\mathbb N}(C,c_0,c_s,\mathsf{succ}(n))
\equiv c_s(n,\mathsf{ind}_{\mathbb N}(C,c_0,c_s,n)).
$$

**定义 3.12（递归）.** 若 $C$ 不依赖于 $n$，自然数归纳给出递归原理。给定 $c_0:C$ 与 $s:C\to C$，可定义 $f:\mathbb N\to C$，使得
$$
f(0)\equiv c_0,\qquad f(\mathsf{succ}(n))\equiv s(f(n)).
$$

**定义 3.13（加法）.** 定义 $m+n$ 为对 $n$ 递归：
$$
m+0\equiv m,\qquad m+\mathsf{succ}(n)\equiv \mathsf{succ}(m+n).
$$

**命题 3.14（右零律）.** 对任意 $m:\mathbb N$，有 $m+0=m$。

**证明（书内证明）.** 这是加法定义的 judgmental 计算规则，因此取 $\mathsf{refl}_m$。$\square$

**命题 3.15（左零律）.** 对任意 $n:\mathbb N$，有 $0+n=n$。

**证明（书内证明）.** 对 $n$ 作自然数归纳。基步中 $0+0\equiv0$，取 $\mathsf{refl}_0$。归纳步中假设 $0+n=n$，目标为 $0+\mathsf{succ}(n)=\mathsf{succ}(n)$。左边按定义化为 $\mathsf{succ}(0+n)$，对归纳假设应用 $\mathsf{ap}_{\mathsf{succ}}$ 即得。$\square$

## 3.4 布尔类型与可判定性

**定义 3.16.** 布尔类型 $\mathbf 2$ 是有两个构造子的归纳类型：
$$
\mathsf{false}:\mathbf 2,\qquad \mathsf{true}:\mathbf 2.
$$
它的消去规则要求分别给出两个分支。

**定义 3.17.** 类型 $A$ 可判定，记为 $\mathsf{Dec}(A)$，若
$$
\mathsf{Dec}(A)\coloneqq A+\neg A.
$$

**警告 3.18.** 本书不默认每个类型都可判定，也不默认排中律
$$
\prod_{A:\mathcal U}(A+\neg A).
$$
若某章需要经典逻辑，必须显式声明。

## 3.5 命题作为类型

在命题作为类型的读法下，命题由类型表示，证明由项表示。下表给出本章建立的对应：

| 逻辑形式 | 类型构造 |
| --- | --- |
| $P\Rightarrow Q$ | $P\to Q$ |
| $P\wedge Q$ | $P\times Q$ |
| $P\vee Q$ | $P+Q$ |
| $\exists x:A,\ P(x)$ | $\sum_{x:A}P(x)$ |
| $\forall x:A,\ P(x)$ | $\prod_{x:A}P(x)$ |
| $\bot$ | $\mathbf 0$ |
| $\top$ | $\mathbf 1$ |

**警告 3.19.** 当前“命题”只是读法。第四章会正式定义“mere proposition”为任意两项相等的类型。并不是所有类型都应被看作命题；例如 $\mathbb N$ 有多个元素。

## 3.6 数据与命题的第一次分叉

空类型、单位类型、和类型与自然数表明，逻辑联结词在这里都有数据行为：存在证明含见证，析取证明含分支，自然数证明可被递归程序消费。与此同时，并非任意类型都只表达一个真值；$\mathbf 2$ 和 $\mathbb N$ 的不同构造子必须保持可区分。下一章会用路径空间正式刻画“至多一个证明”“集合式数据”和更高类型，从而把命题读法从解释提升为内部性质。

## 练习

**练习 3.1.** 定义乘法 $m\cdot n$，并证明 $m\cdot0=0$。

**练习 3.2.** 用和类型消去定义布尔取反函数，并计算它在两个构造子上的值。

**练习 3.3.** 证明 $\neg\neg\neg A\to\neg A$。

**练习 3.4.** 说明为什么 $A+\neg A$ 比 $\mathbf 2$ 带有更多证明信息。
