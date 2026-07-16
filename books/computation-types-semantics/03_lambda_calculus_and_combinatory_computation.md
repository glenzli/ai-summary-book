# 第 3 章：无类型 λ 演算、替换与归约

λ 演算把计算压缩为变量绑定和函数应用，同一个三行语法同时承载高阶函数、数据编码与通用计算。它的简洁会放大任何关于变量的含混：若替换捕获了自由变量，后续 β 归约、类型规则和抽象机模拟都会研究另一个对象。本章先证明新鲜变量选择不影响替换的 α-等价类，再把 β 合流性隔离为 EI-3，最后从该输入严格推出正规形唯一性并实际计算 Church 编码。

## 3.1 项、α-等价与捕获避免替换

**定义 3.1（原始项）。** 给定可数无限变量集 $\mathsf{Var}$，
$$
e::=x\mid\lambda x.e\mid e\,e.
$$
令 $\mathrm{FV}(e)$ 为自由变量集，$\mathrm{Var}(e)$ 为自由和受绑定变量的并集。两者按结构递归定义且均有限。

**定义 3.2（安全改名与 α-等价）。** 若
$y\notin\mathrm{Var}(e)$，记 $e\langle y/x\rangle$ 为把 $e$ 中由当前外层绑定者
$x$ 绑定的出现改成 $y$ 的结构改名。α-等价 $=_\alpha$ 是包含
$$
\lambda x.e=_\alpha\lambda y.e\langle y/x\rangle
\qquad(y\notin\mathrm{Var}(e))
$$
并对抽象、应用封闭的最小等价关系。

**定义 3.3（捕获避免替换）。** 变量和应用情形为
$$
x[x:=s]=s,\quad y[x:=s]=y\ (y\ne x),\quad
(e_1e_2)[x:=s]=e_1[x:=s]e_2[x:=s].
$$
抽象情形为
$$
(\lambda y.e)[x:=s]=
\begin{cases}
\lambda y.e,&y=x,\\
\lambda y.e[x:=s],&y\ne x,\ y\notin\mathrm{FV}(s),\\
\lambda z.(e\langle z/y\rangle[x:=s]),&y\ne x,\ y\in\mathrm{FV}(s),
\end{cases}
$$
其中第三行取
$z\notin\mathrm{Var}(e)\cup\mathrm{Var}(s)\cup\{x,y\}$。

**引理 3.4（新鲜改名交换）。** 设所有显示的新变量都不出现在原项中。

1. 两次针对不同绑定名的安全改名可交换。
2. 若 $y\ne x$ 且 $y\notin\mathrm{FV}(s)$，则
$$
(e\langle z/y\rangle)[x:=s]
=_\alpha(e[x:=s])\langle z/y\rangle.
$$
3. 若 $z,z'$ 都满足定义 3.3 的新鲜条件，则
$$
\lambda z.(e\langle z/y\rangle[x:=s])
=_\alpha
\lambda z'.(e\langle z'/y\rangle[x:=s]).
$$

**证明。** 前两项对 $e$ 结构归纳。变量情形直接检查变量是否等于被改名者；应用情形对两个子项使用归纳假设；抽象情形先把其绑定变量改成同时避开
$x,y,z,z'$ 和 $\mathrm{Var}(s)$ 的名字，再对主体使用归纳假设。由于每步排除的集合有限，新名字存在。

对第 3 项，再取同时新鲜的 $w$。按定义 3.2，把左、右两项的外层绑定者都改成 $w$。
第 1、2 项给出两边改名后的主体均 α-等价于
$e\langle w/y\rangle[x:=s]$，由传递性得到结论。证毕。

**定理 T3.1（替换在 α-商上良定义）。**

1. 定义 3.3 中不同的新鲜变量选择产生 α-等价结果。
2. 若 $e=_\alpha e'$ 且 $s=_\alpha s'$，则
$e[x:=s]=_\alpha e'[x:=s']$。

**证明。** 第 1 项对 $e$ 结构归纳。变量和应用没有新鲜选择，应用由两个归纳假设得到。抽象若绑定者等于
$x$，替换停止；若绑定者不在 $\mathrm{FV}(s)$，对主体用归纳假设；冲突情形正是引理 3.4(3)。

第 2 项先固定 $s$，对 $e=_\alpha e'$ 的生成推导归纳。等价关系的自反、对称、传递情形由
$=_\alpha$ 的对应性质得到；应用和抽象同余情形使用归纳假设。唯一的生成式是安全绑定改名
$\lambda y.e_0=_\alpha\lambda z.e_0\langle z/y\rangle$；把 $y,z$ 必要时共同改成一个避开
$x$ 与 $\mathrm{Var}(s)$ 的新变量，再由引理 3.4(2)得到替换后主体 α-等价。

最后对 $s=_\alpha s'$ 作归纳。变量位置 $x$ 直接使用该假设，其他变量不变；应用用两个归纳假设；
抽象先选一个同时避开 $s,s'$ 的共同新鲜绑定名，再对主体使用归纳假设。组合两段结论即得
$e[x:=s]=_\alpha e'[x:=s']$。证毕。

**例 3.5（捕获避免的实际一步）。**
$$
(\lambda x.\lambda y.x)\,y
\to_\beta(\lambda y.x)[x:=y]
=_\alpha\lambda z.y,
$$
其中 $z\notin\{x,y\}$。若写成 $\lambda y.y$，原来作为实参的自由 $y$ 被捕获；两个结果的自由变量集合分别是
$\{y\}$ 与 $\varnothing$，因此不 α-等价。

## 3.2 β-归约与合流性

**定义 3.6（β-归约）。** $\to_\beta$ 是包含
$$
(\lambda x.e)s\to_\beta e[x:=s]
$$
并在抽象体、应用左部和应用右部下封闭的最小关系。由 T3.1，它在 α-等价类上良定义。
自反传递闭包记为 $\to_\beta^*$。无可用 β 步的项称 β-正规形。

**外部输入 EI-3（Church-Rosser 合流性）。** 对定义 3.6 的完整 β-归约，若
$e\to_\beta^*a$ 且 $e\to_\beta^*b$，则存在 $c$ 使
$a\to_\beta^*c$ 且 $b\to_\beta^*c$，所有等式按 α-商理解。

**证明路线（不计作书内证明）。** 来源定义平行 β-归约，证明其 diamond 性及与
$\to_\beta^*$ 具有相同传递闭包，再推出合流性。完整平行替换引理和 diamond 情形由
SOURCES.md 中 EI-3 的定位承担。

**定理 T3.2（正规形唯一性）。** 若
$e\to_\beta^*n_1$、$e\to_\beta^*n_2$，且 $n_1,n_2$ 都是 β-正规形，则
$n_1=_\alpha n_2$。

**证明。** EI-3 给出 $c$ 使
$n_1\to_\beta^*c$ 且 $n_2\to_\beta^*c$。从正规形出发的多步归约只能是零步，所以
$n_1=_\alpha c=_\alpha n_2$。证毕。

**边界例 3.7（发散轨迹）。** 令
$\Omega=(\lambda x.x\,x)(\lambda x.x\,x)$，则
$$
\Omega\to_\beta
(\lambda x.x\,x)(\lambda x.x\,x)=\Omega\to_\beta\cdots.
$$
合流性不蕴含终止；该轨迹每一步都选择同一个外层 redex。

## 3.3 Church 数与可定义函数

**定义 3.8（Church 自然数）。**
$$
\overline n=\lambda f.\lambda x.f^n x,\qquad
f^0x=x,\quad f^{n+1}x=f(f^nx).
$$

**例 3.9（后继的归约轨迹）。** 令
$\mathsf{succ}=\lambda n.\lambda f.\lambda x.f(n\,f\,x)$。则
$$
\begin{aligned}
\mathsf{succ}\,\overline n
&\to_\beta\lambda f.\lambda x.f(\overline n\,f\,x)\\
&\to_\beta^*\lambda f.\lambda x.f(f^nx)
=_\alpha\overline{n+1}.
\end{aligned}
$$
中间多步先收缩 $\overline n\,f$，再收缩所得函数对 $x$ 的应用。

**命题 3.10（可定义函数对复合封闭）。** 若闭项 $F,G_1,\ldots,G_k$ 分别表示数值函数
$f,g_1,\ldots,g_k$，则
$h(\vec x)=f(g_1(\vec x),\ldots,g_k(\vec x))$ 由闭项表示。

**证明。** 取
$$
H=\lambda x_1\cdots x_n.
F\,(G_1x_1\cdots x_n)\cdots(G_kx_1\cdots x_n).
$$
对任意 Church 数输入 $\overline{\vec m}$，先由 β 归约的上下文闭包把每个
$G_i\overline{\vec m}$ 归约到 $\overline{g_i(\vec m)}$；再使用 $F$ 的表示性质，归约到
$\overline{f(g_1(\vec m),\ldots,g_k(\vec m))}$。$F,G_i$ 和全部输入绑定在 $H$ 中，所以 $H$ 闭。证毕。

## 3.4 λ 演算的输入边界

替换的选择无关和 α-相容性由 T3.1 完整证明；正规形唯一性由内部推理加 EI-3 得到。
λ 可定义函数与第 1 章计数器机函数类的双向等价属于 EI-1，不由若干 Church 编码例子代替。

## 练习

**练习 E3.1.** 重做例 3.5，并证明错误结果 $\lambda y.y$ 不与正确结果 α-等价。

**练习 E3.2.** 证明若 $x\notin\mathrm{FV}(e)$，则 $e[x:=s]=_\alpha e$。

**练习 E3.3.** 构造 Church 加法项，并写出它在 $\overline1,\overline2$ 上到
$\overline3$ 的完整 β 轨迹。

**练习 E3.4.** 说明 T3.2 的证明在哪一步使用“正规形”，并给出删去该假设后结论失败的例子。
