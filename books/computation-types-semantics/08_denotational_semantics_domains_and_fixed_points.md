# 第 8 章：域、最小不动点与 while 的指称

操作语义保留程序运行的时间顺序，指称语义则希望把整个命令压缩为一个数学对象。循环造成的困难不是语法递归，而是部分性：对某些状态，有限执行结果从未出现。ω-cpo 把“已在有限步内得到的结果”排成近似链，最小不动点只收集链中实际出现的信息。本章证明 while 泛函连续，并用有限近似逐层重建大步推导，从而得到操作语义与指称语义的双向对应。

## 8.1 ω-cpo 与最小不动点

**定义 8.1（带底 ω-cpo）。** 偏序 $(D,\sqsubseteq)$ 是 ω-cpo，若每条递增链
$d_0\sqsubseteq d_1\sqsubseteq\cdots$ 有上确界 $\bigsqcup_nd_n$。若有最小元
$\bot$，称为带底 ω-cpo。

**定义 8.2（ω-连续）。** 函数 $f:D\to E$ 是 ω-连续的，若它单调且对每条递增链满足
$$
f\left(\bigsqcup_nd_n\right)=\bigsqcup_nf(d_n).
$$
本章只使用 ω-链，不把该术语扩张为对所有有向上确界的 Scott 连续性。

**定理 T8.1（Kleene 最小不动点）。** 若 $D$ 是带底 ω-cpo，且
$f:D\to D$ ω-连续，则
$$
\mu f=\bigsqcup_{n\in\mathbb N}f^n(\bot)
$$
存在，是 $f$ 的最小不动点。

**证明。** 因 $\bot\sqsubseteq f(\bot)$，单调性归纳给出
$f^n(\bot)\sqsubseteq f^{n+1}(\bot)$，故上确界 $d$ 存在。连续性给出
$$
f(d)=f\left(\bigsqcup_nf^n(\bot)\right)
=\bigsqcup_nf^{n+1}(\bot)=d;
$$
最后一个等号因为删去链首项 $\bot$ 不改变上确界。若 $e$ 是任意不动点，则
$\bot\sqsubseteq e$；由单调性归纳，
$f^n(\bot)\sqsubseteq f^n(e)=e$。取上确界得 $d\sqsubseteq e$。证毕。

## 8.2 偏状态变换域

**定义 8.3（图包含序）。** 对状态集合 $S$，令
$D=S\rightharpoonup S$。定义
$$
p\sqsubseteq q
\quad\Longleftrightarrow\quad
\forall\sigma,\tau.\ p(\sigma)=\tau\Rightarrow q(\sigma)=\tau.
$$
底元 $\bot_D$ 是处处未定义偏函数。

**定理 T8.2（偏函数域闭合）。** $(D,\sqsubseteq,\bot_D)$ 是带底 ω-cpo。

**证明。** 图包含显然自反、传递；若 $p\sqsubseteq q$ 且 $q\sqsubseteq p$，两图互相包含，故
$p=q$，所以它是偏序。处处未定义图包含于每个偏函数图。

给递增链 $p_0\sqsubseteq p_1\sqsubseteq\cdots$，令
$p=\bigcup_n p_n$（关系图的并）。若 $(\sigma,\tau)$ 与 $(\sigma,\tau')$ 都在该并中，
分别取所在层 $m,n$；不妨 $m\le n$，则 $p_m\sqsubseteq p_n$，两对都在函数图
$p_n$ 中，故 $\tau=\tau'$。所以 $p$ 是偏函数。每个 $p_n\sqsubseteq p$；若
$p_n\sqsubseteq q$ 对所有 $n$ 成立，则并图也包含于 $q$。因此
$p=\bigsqcup_np_n$。证毕。

**定义 8.4（命令指称）。** 使用第 7 章的状态与表达式解释。偏函数复合只在两个阶段都定义时有值：
$$
\llbracket\mathsf{skip}\rrbracket=\mathrm{id}_S,\qquad
\llbracket x:=a\rrbracket(\sigma)
=\sigma[x\mapsto\llbracket a\rrbracket\sigma],
$$
$$
\llbracket c_1;c_2\rrbracket
=\llbracket c_2\rrbracket\circ\llbracket c_1\rrbracket.
$$
条件按唯一布尔值选择相应分支。对
$W=\mathsf{while}\ b\ \mathsf{do}\ c$，令 $d=\llbracket c\rrbracket$，定义
$\Phi_{b,d}:D\to D$：
$$
\Phi_{b,d}(g)(\sigma)=
\begin{cases}
\sigma,&\llbracket b\rrbracket\sigma=\mathsf{false},\\
g(\tau),&\llbracket b\rrbracket\sigma=\mathsf{true}
 \text{ 且 }d(\sigma)=\tau,\\
\uparrow,&\llbracket b\rrbracket\sigma=\mathsf{true}
 \text{ 且 }d(\sigma)\uparrow.
\end{cases}
$$
第二行在 $g(\tau)\uparrow$ 时也无定义。定义
$\llbracket W\rrbracket=\mu\Phi_{b,d}$。

**引理 8.5（while 泛函 ω-连续）。** $\Phi_{b,d}$ 在 $D$ 上 ω-连续。

**证明。** 若 $g\sqsubseteq h$，逐状态分布尔值。假分支两者都返回 $\sigma$；真分支若
$d(\sigma)$ 无定义，两者都无定义；若 $d(\sigma)=\tau$，则
$g(\tau)$ 的每个已定义结果由 $g\sqsubseteq h$ 保留。因此 $\Phi(g)\sqsubseteq\Phi(h)$。

令 $g_0\sqsubseteq g_1\sqsubseteq\cdots$ 且 $g=\bigsqcup_ng_n$。仍逐状态讨论。
假分支中 $\Phi(g)(\sigma)=\sigma$，每个 $\Phi(g_n)(\sigma)=\sigma$，两边相等。
真分支且 $d(\sigma)$ 无定义时，两边都无定义。若 $d(\sigma)=\tau$，则
$$
\Phi(g)(\sigma)=g(\tau)
=\left(\bigsqcup_ng_n\right)(\tau)
=\left(\bigsqcup_n\Phi(g_n)\right)(\sigma),
$$
其中偏函数链的上确界是图并。所有状态情形已覆盖，故保持链上确界。证毕。

## 8.3 有限近似与操作充分性

**定义 8.6（计数循环推导）。** 对固定
$W=\mathsf{while}\ b\ \mathsf{do}\ c$，写
$\langle W,\sigma\rangle\Downarrow_k\tau$ 表示大步推导中恰执行 $k$ 次真分支：
$$
\frac{\llbracket b\rrbracket\sigma=\mathsf{false}}
{\langle W,\sigma\rangle\Downarrow_0\sigma},
$$
$$
\frac{\llbracket b\rrbracket\sigma=\mathsf{true}\quad
\langle c,\sigma\rangle\Downarrow\rho\quad
\langle W,\rho\rangle\Downarrow_k\tau}
{\langle W,\sigma\rangle\Downarrow_{k+1}\tau}.
$$
擦去下标恰得到定义 7.3 的 while 推导；反过来，每个有限 while 推导有唯一的真分支次数。

**引理 8.6（近似层刻画）。** 假设对循环体已有
$$
\llbracket c\rrbracket(\sigma)=\rho
\Longleftrightarrow
\langle c,\sigma\rangle\Downarrow\rho.
\tag{IH}
$$
则对每个 $n\ge0$，
$$
\Phi_{b,\llbracket c\rrbracket}^{\,n}(\bot_D)(\sigma)=\tau
\quad\Longleftrightarrow\quad
\exists k<n.\ \langle W,\sigma\rangle\Downarrow_k\tau.
$$

**证明。** 对 $n$ 归纳。$n=0$ 时左侧处处未定义，右侧没有自然数 $k<0$，同假。
设结论对 $n$ 成立，考察 $n+1$。

- 若布尔值为假，左侧按 $\Phi$ 定义返回 $\sigma$。右侧取 $k=0<n+1$，且计数规则给出
  $\Downarrow_0\sigma$。任何计数推导在假分支只能有 $k=0,\tau=\sigma$。
- 若布尔值为真且 $\llbracket c\rrbracket(\sigma)$ 无定义，由 (IH) 循环体没有终止大步推导；
  左侧无定义，右侧也不可能使用真分支规则。
- 若布尔值为真且 $\llbracket c\rrbracket(\sigma)=\rho$，左侧有值 $\tau$ 当且仅当
  $\Phi^n(\bot_D)(\rho)=\tau$。归纳假设把它等价为某个 $j<n$ 满足
  $\langle W,\rho\rangle\Downarrow_j\tau$；(IH) 给出
  $\langle c,\sigma\rangle\Downarrow\rho$。使用一次真分支规则，等价得到
  $\langle W,\sigma\rangle\Downarrow_{j+1}\tau$，且 $j+1<n+1$。

三种互斥状态情形穷尽。证毕。

**定理 T8.3（命令指称与大步语义双向一致）。** 对每个第 7 章命令，
$$
\llbracket c\rrbracket(\sigma)=\tau
\quad\Longleftrightarrow\quad
\langle c,\sigma\rangle\Downarrow\tau.
$$

**证明。** 对命令 $c$ 结构归纳。

- skip 与赋值：两边由各自定义给出同一状态。
- sequence：$\llbracket c_2\rrbracket\circ\llbracket c_1\rrbracket$ 在 $\sigma$ 上取
  $\tau$，当且仅当存在 $\rho$ 使两个指称分别取 $\rho,\tau$。两个结构归纳假设把它等价为
  两个大步前提，恰是 sequence 大步规则。
- conditional：布尔解释全定义且唯一。真、假两种情形分别使用所选分支的归纳假设，正好对应两条条件大步规则。
- while：由 T8.1、T8.2 和引理 8.5，
$$
\llbracket W\rrbracket
=\bigsqcup_n\Phi^n(\bot_D).
$$
图并在 $\sigma$ 上取 $\tau$ 当且仅当存在 $n$ 使
$\Phi^n(\bot_D)(\sigma)=\tau$。由引理 8.6，这又当且仅当存在 $k$ 使
$\langle W,\sigma\rangle\Downarrow_k\tau$；定义 8.6 擦去下标后恰是
$\langle W,\sigma\rangle\Downarrow\tau$。引理 8.6 的 (IH) 就是循环体的结构归纳假设。

五种命令构造均已覆盖。证毕。

**例 8.7（递减循环的近似表）。** 把状态限制为 $\sigma(x)\in\mathbb N$，令
$W=\mathsf{while}\ x>0\ \mathsf{do}\ x:=x-1$。记 $F=\Phi_{x>0,\llbracket x:=x-1\rrbracket}$。

| 近似 | 有定义的初值 | 结果 |
| --- | --- | --- |
| $F^0(\bot)$ | 无 | 无 |
| $F^1(\bot)$ | $x=0$ | $x=0$ |
| $F^2(\bot)$ | $x=0,1$ | $x=0$ |
| $F^3(\bot)$ | $x=0,1,2$ | $x=0$ |

例如从 $x=2$ 出发，
$F^3(\bot)(2)=F^2(\bot)(1)=F^1(\bot)(0)=0$，对应恰两次真分支的大步推导。

## 8.4 PCF 的外部语义边界

**外部输入 EI-8（PCF adequacy 与完全抽象）。** 对标准 PCF 及其 Scott 连续函数模型：
闭 ground-type 项的指称反映其终止数值观察（computational adequacy），但朴素 Scott 模型不完全抽象。
游戏语义的经典模型满足 PCF 上下文等价的完全抽象刻画。各陈述的 PCF 版本、观察预序与文献定位见
SOURCES.md；T8.3 不依赖这些结果。

**证明路线（不计作书内证明）。** Adequacy 通常用逻辑关系连接域中非底元素与有限求值；
不完全抽象由模型中的非顺序函数造成；游戏模型再以策略的可定义性与观察预序证明完全抽象。
这些高阶构造不由本章的偏状态函数域替代。

## 8.5 域语义证明的责任边界

T8.1--T8.3、偏函数域闭合、while 连续性和有限近似刻画均在本章完整证明。
EI-8 只用于第 10 章说明高阶语言的 adequacy/full-abstraction 差异。

## 练习

**练习 E8.1.** 采用如下替代约定：函数把每条非空递增 ω-链送到递增 ω-链，并保持其上确界。证明该约定已经蕴含单调性。

**练习 E8.2.** 计算例 8.7 的 $F^4(\bot)$，并写出从 $x=3$ 到 $x=0$ 的计数大步推导。

**练习 E8.3.** 证明偏函数复合分别对两个参数的递增 ω-链保持上确界。

**练习 E8.4.** 指出 T8.3 的 while 反向中为什么“最小”不动点不可替换为任意不动点。
