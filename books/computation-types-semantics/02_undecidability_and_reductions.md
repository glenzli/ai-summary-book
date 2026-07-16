# 第 2 章：不可判定性、归约与 Rice 定理

通用解释器使程序文本可以作为另一个程序的输入，于是“分析任意程序”的算法也必须接受自己的编码。对角停机集合把这种自应用压缩成一个不可判定问题，many-one 归约再把边界传递到其他性质。EI-1 提供程序解释，EI-2 把含参数的程序模板有效固化成编号；本章在这两个明确输入上完整证明停机问题与 Rice 定理，并逐项展示外延性为何是适用条件。

## 2.1 识别与停机

**定义 2.1（可识别）。** 集合 $A\subseteq\mathbb N$ 可识别，若存在机器 $M$，使
$$
x\in A\Rightarrow M(x)\downarrow1,\qquad
x\notin A\Rightarrow M(x)\text{ 不输出 }1.
$$
第二种情形允许输出别的值或发散。

**命题 2.1（双向识别推出判定）。** 若 $A$ 与 $\overline A$ 均可识别，则 $A$ 可判定。

**证明。** 在输入 $x$ 上交错模拟两个识别器：第 $s$ 轮各执行一步。由于
$x$ 恰属于 $A,\overline A$ 之一，相应识别器在有限步后输出 $1$。前者先输出时返回 $1$，后者先输出时返回 $0$。
该算法对每个输入终止并给出 $\chi_A$。证毕。

**定义 2.2（对角停机集合）。**
$$
K=\{e\in\mathbb N\mid\varphi_e(e)\downarrow\}.
$$

**定理 T2.1（$K$ 不可判定）。** 不存在计算 $\chi_K$ 的总计数器机。

**证明。** 假设总程序 $H$ 满足 $H(e)=1$ 当且仅当 $\varphi_e(e)\downarrow$。
由计数器机的条件跳转与程序拼接，构造程序 $D$：
$$
D(x)=
\begin{cases}
\uparrow,&H(x)=1,\\
0,&H(x)=0.
\end{cases}
$$
令 $d=\ulcorner D\urcorner$。若 $H(d)=1$，按 $H$ 的规格 $D(d)$ 应停机，但按 $D$ 的定义发散；
若 $H(d)=0$，按规格 $D(d)$ 应发散，但定义使其输出 $0$。两种可能都矛盾，故 $H$ 不存在。证毕。

**例 2.3（$K$ 的识别轨迹）。** 识别器在输入 $e$ 上运行 EI-1 的
$U(\langle e,e\rangle)$；若模拟配置
$c_0\to c_1\to\cdots\to c_t$ 且 $c_t$ halt，就输出 $1$。当
$e\in K$ 时该轨迹有限；当 $e\notin K$ 时模拟无限，因此正好识别 $K$。

## 2.2 Many-one 归约

**定义 2.4（many-one 归约）。** 若存在全可计算 $f:\mathbb N\to\mathbb N$ 使
$$
x\in A\Longleftrightarrow f(x)\in B,
$$
则写 $A\le_mB$。

**定理 T2.2（传递性）。** 若 $A\le_mB$ 且 $B\le_mC$，则 $A\le_mC$。

**证明。** 设 $f,g$ 分别见证两次归约。命题 1.8 给出全可计算 $g\circ f$，且
$x\in A\Leftrightarrow f(x)\in B\Leftrightarrow g(f(x))\in C$。证毕。

**命题 2.5（不可判定性沿归约传递）。** 若 $A\le_mB$ 且 $A$ 不可判定，则 $B$ 不可判定。

**证明。** 若 $B$ 有总判定器，在输入 $x$ 上先计算归约函数 $f(x)$ 再判定
$f(x)\in B$，便得到 $A$ 的总判定器，与假设矛盾。证毕。

**外部输入 EI-2（$s$-$m$-$n$ 参数定理）。** 对第 1 章的可接受程序枚举，存在全可计算函数
$s:\mathbb N^2\to\mathbb N$，使
$$
\varphi_{s(e,a)}(x)\simeq\varphi_e(\langle a,x\rangle).
$$
也就是说，程序模板的第一个输入可有效固化为程序编号。Kleene 第二递归定理不进入本书证明链，因而不属于 EI-2 的输入契约。

## 2.3 Rice 定理

**定义 2.6（外延索引性质）。** 集合 $S\subseteq\mathbb N$ 是外延的，若
$\varphi_e=\varphi_{e'}$ 推出 $e\in S\Leftrightarrow e'\in S$。它非平凡，若
$S\ne\varnothing$ 且 $S\ne\mathbb N$。令 $\bot$ 表示处处未定义偏函数。

**定理 T2.3（Rice 定理）。** 每个非平凡外延索引性质 $S$ 都不可判定。

**证明。** 先设 $\bot$ 不属于 $S$。由非平凡性，取 $p\in S$。给定 $x$，定义二元程序模板
$$
\psi(x,n)=
\begin{cases}
\varphi_p(n),&\varphi_x(x)\downarrow,\\
\uparrow,&\varphi_x(x)\uparrow.
\end{cases}
$$
其机器先用 EI-1 模拟 $\varphi_x(x)$；只有模拟 halt 后才运行 $P_p(n)$。模板有一个固定程序编号
$e_\psi$。由 EI-2，函数 $f(x)=s(e_\psi,x)$ 全可计算，且
$$
\varphi_{f(x)}=
\begin{cases}
\varphi_p,&x\in K,\\
\bot,&x\notin K.
\end{cases}
$$
外延性和 $p\in S,\bot\notin S$ 给出
$x\in K\Leftrightarrow f(x)\in S$，所以 $K\le_mS$。由 T2.1 与命题 2.5，$S$ 不可判定。

若 $\bot\in S$，则补集 $\overline S$ 仍外延、非平凡且不含 $\bot$；由上一段
$\overline S$ 不可判定。若 $S$ 可判定，则交换输出即可判定 $\overline S$，矛盾。两种情形穷尽，定理成立。证毕。

**例 2.7（把具体性质代入 Rice 证明）。** 令
$S_7=\{e\mid\exists n.\varphi_e(n)=7\}$。常值 $7$ 程序属于 $S_7$，处处发散程序不属于
$S_7$，故性质非平凡且外延。上述归约取 $p$ 为常值 $7$ 程序，得到
$$
x\in K\Longleftrightarrow f(x)\in S_7.
$$
这不是口头类比，而是一条由 $s$-$m$-$n$ 产生编号 $f(x)$ 的 many-one 归约。

性质“程序是否对所有输入停机”和“程序是否计算常零函数”同样外延且非平凡。相反，“程序文本少于
100 条指令”和“程序在固定输入 $0$ 上是否于 10 步内停机”依赖表示或步数，不是外延性质，Rice 定理不适用。

## 2.4 归约证明的输入边界

T2.1--T2.3 均在本章完整证明。T2.1 只使用第 1 章的程序枚举与复合封闭；T2.3
显式依赖 EI-1 的通用模拟和 EI-2 的参数固化。第二递归定理属于未展开的自指理论，不在本章或后续章节中调用。

## 练习

**练习 E2.1.** 证明 $K$ 可识别，并说明为什么该识别器不是判定器。

**练习 E2.2.** 构造全可计算 $f$，证明
$K\le_m\{e\mid\varphi_e(0)\downarrow\}$。

**练习 E2.3.** 判断“程序在固定输入 $0$ 上于 10 步内停机”是否受 Rice 定理约束，并给出一个判定算法。

**练习 E2.4.** 对性质“存在输入使输出为 $0$”逐项重做 T2.3 第一种情形的归约。
