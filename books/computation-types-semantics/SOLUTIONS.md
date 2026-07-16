# 习题解答

本文件逐题回答正文与附录练习。解答可以调用题目之前的正文结果，但不承担正文主定理中缺失的证明责任。

## 第 0 章

**E0.1.** 取整数命令 $x:=1$。它在通常的算术类型规则下良构并终止，但三元组
$\{\mathsf{true}\}\ x:=1\ \{x=0\}$ 不语义有效，因为任一终态都满足 $x=1$。类型判断只约束语法和静态类别；后置条件属于程序逻辑层。缺失的是从类型系统到该规格的连接定理。

**E0.2.** 可以把三个元语言谓词分别写成
$$
\mathsf{HasType}\subseteq\mathsf{Ctx}\times\mathsf{Term}\times\mathsf{Type},
\qquad
\mathsf{Eval}\subseteq\mathsf{Term}\times\mathsf{Val},
$$
以及指称函数 $\mathsf{den}:\mathsf{Term}\rightharpoonup D$。于是
$\Gamma\vdash e:A$、$e\Downarrow v$ 和 $\llbracket e\rrbracket=d$ 分别是这三个不同类型的元语言断言。

**E0.3.** 一组测试只检查有限多个输入和有限运行前缀。Soundness 定理量化所有满足前提的程序、状态或推导，并由形式规则证明结论。除非另有完备测试定理，有限测试通过不能推出这个全称命题。

## 第 1 章

**E1.1.** 定义 1.4 给出
$$
\langle2,3\rangle=\frac{(2+3)(2+3+1)}2+3=18.
$$
恢复时寻找唯一 $s$ 使 $T_s\le18<T_{s+1}$。由于 $T_5=15$、$T_6=21$，有 $s=5$；
$b=18-T_5=3$，$a=s-b=2$。

**E1.2.** 若 $\chi_A,\chi_B$ 是总特征函数，则
$\chi_{\overline A}(x)=1-\chi_A(x)$，而
$\chi_{A\cap B}(x)=\chi_A(x)\chi_B(x)$。常数、减法和乘法均由有限计数器程序实现，所以补集和二元交可判定；对交集个数归纳即得有限交封闭。

**E1.3.** 令
$$
I_0=\mathsf{dec}(0,1,2),\qquad
I_1=\mathsf{inc}(0,3),\qquad
I_2=\mathsf{dec}(0,3,2),\qquad
I_3=\mathsf{halt}.
$$
零输入沿
$(0;0)\to(1;0)\to(3;1)$，输出 $1$。输入 $2$ 沿
$(0;2)\to(2;1)\to(2;0)\to(3;0)$，输出 $0$。一般正输入先被 $I_0$ 减一，再由 $I_2$ 清零，故该机计算 $[x=0]$。

**E1.4.** 对 $0\le k\le y$，第 $2k$ 个配置为
$$
C_{2k}=(0;x+k,y-k).
$$
若 $k<y$，随后两个配置为
$$
C_{2k+1}=(1;x+k,y-k-1),\qquad
C_{2k+2}=(0;x+k+1,y-k-1).
$$
这由 $k$ 归纳得到，并在每一步保持
$r_0+r_1=x+y$。到 $k=y$ 时得到 $(0;x+y,0)$，零分支再走一步到
$(2;x+y,0)$。因此运行恰用 $2y+1$ 步终止，halt 时 $r_0=x+y$。

## 第 2 章

**E2.1.** 在输入 $e$ 上运行 EI-1 的通用机模拟 $\varphi_e(e)$；模拟一旦停止就接受。因此 $K$ 可识别。若 $e\notin K$，模拟永不停止，所以它不是 $K$ 的总判定器；T2.1 说明不存在别的总判定器。

**E2.2.** 固定程序模板 $Q(x,n)$，它忽略 $n$ 并模拟 $\varphi_x(x)$。由 EI-2 得全可计算编号函数 $f$，使
$$
\varphi_{f(x)}(n)\simeq\varphi_x(x).
$$
于是 $x\in K$ 当且仅当 $\varphi_{f(x)}(0)\downarrow$，所以
$K\le_m\{e\mid\varphi_e(0)\downarrow\}$。

**E2.3.** 该性质依赖程序表示、固定输入和固定步数，不是外延性质，故 Rice 定理不适用。判定算法从程序在输入 $0$ 上的初始配置开始，用 T1.1 至多计算十次下一配置；若初态或这十步内到达 halt 就输出 $1$，否则输出 $0$。循环次数有常数界，所以算法总终止。

**E2.4.** 令
$$
S_0=\{e\mid\exists n.\ \varphi_e(n)=0\}.
$$
$S_0$ 外延、非平凡，并且 $\bot\notin S_0$；取 $p$ 为常零程序。给定 $x$，令模板先模拟
$\varphi_x(x)$，停止后才在输入 $n$ 上运行 $p$。EI-2 产生全可计算 $f$，满足
$$
\varphi_{f(x)}=
\begin{cases}
\varphi_p,&x\in K,\\
\bot,&x\notin K.
\end{cases}
$$
故 $x\in K\Longleftrightarrow f(x)\in S_0$。由 T2.1 和命题 2.5，$S_0$ 不可判定。

## 第 3 章

**E3.1.** 先把内层绑定变量改名：
$$
(\lambda x.\lambda y.x)\,y
=_\alpha(\lambda x.\lambda z.x)\,y
\to_\beta\lambda z.y,
$$
其中 $z\ne y$。错误结果 $\lambda y.y$ 没有自由变量，而
$\mathrm{FV}(\lambda z.y)=\{y\}$；alpha 等价保持自由变量集合，所以二者不 alpha 等价。

**E3.2.** 对 $e$ 结构归纳。变量情形中 $e=x$ 被假设排除，其余变量替换后不变。应用情形分别使用两个归纳假设。抽象情形先 alpha 改名绑定变量，使其不同于 $x$ 且不在 $\mathrm{FV}(s)$ 中，再对主体使用归纳假设；重新加上同一抽象即得原项的 alpha 等价类。

**E3.3.** 取
$$
\mathsf{add}=\lambda m.\lambda n.\lambda f.\lambda x.\,
m\,f\,(n\,f\,x).
$$
把 $\overline1=\lambda g.\lambda z.g z$、
$\overline2=\lambda g.\lambda z.g(gz)$ 代入，完整收缩为
$$
\begin{aligned}
\mathsf{add}\,\overline1\,\overline2
&\to(\lambda n.\lambda f.\lambda x.\overline1 f(nfx))\,\overline2\\
&\to\lambda f.\lambda x.\overline1 f(\overline2 f x)\\
&\to\lambda f.\lambda x.(\lambda z.fz)(\overline2 f x)\\
&\to\lambda f.\lambda x.f(\overline2 f x)\\
&\to\lambda f.\lambda x.f((\lambda z.f(fz))x)\\
&\to\lambda f.\lambda x.f(f(fx))=\overline3.
\end{aligned}
$$

**E3.4.** EI-3 先给出 $c$，使
$n_1\to_\beta^*c$ 且 $n_2\to_\beta^*c$。正规形假设正用于断言这两条多步归约只能是零步，从而
$n_1=_\alpha c=_\alpha n_2$。删去假设后，取
$e=(\lambda x.x)y$、$n_1=e$、$n_2=y$；有
$e\to_\beta^*n_1$ 和 $e\to_\beta^*n_2$，但 $n_1$ 与 $n_2$ 不 alpha 等价。

## 第 4 章

**E4.1.** 推导树为
$$
\frac{\displaystyle
  \frac{x:A\in x:A}{x:A\vdash x:A}\ \textsc{T-Var}}
{\varnothing\vdash\lambda x:A.x:A\to A}\ \textsc{T-Abs}.
$$

**E4.2.** 若 $x:A\vdash x\,x:B$，应用反演给出某个 $C$，使
$x:A\vdash x:C\to B$ 且 $x:A\vdash x:C$。变量反演分别给出
$A=C\to B$ 与 $A=C$，故 $C=C\to B$。有限简单类型的语法树不可能等于自己的真子树再接一个箭头，所以矛盾。

**E4.3.** 令 $V=\lambda f:\iota\to\iota.f$。例 4.5 的第一步是
$$
V\,((\lambda g:\iota\to\iota.g)I)
\to V\,I,
$$
其外层规则是 \textsc{E-App2}，前提是内层 \textsc{E-BetaV}；第二步
$V\,I\to I$ 直接用 \textsc{E-BetaV}。T4.3 先从
$\varnothing\vdash M:\iota\to\iota$ 得
$\varnothing\vdash V I:\iota\to\iota$，再得
$\varnothing\vdash I:\iota\to\iota$。

**E4.4.** 对第一条一步推导归纳。若末规则是 \textsc{E-App1}，第二条推导只能也是
\textsc{E-App1}：\textsc{E-App2} 或 \textsc{E-BetaV} 都要求左项是值，而值没有一步后继；对两个左子步使用归纳假设。若末规则是 \textsc{E-App2}，第二条要么同为
\textsc{E-App2}，此时对实参步骤使用归纳假设；它不能是 \textsc{E-App1}，也不能是
\textsc{E-BetaV}，后者要求本来正在步进的实参已经是值。若末规则是
\textsc{E-BetaV}，函数和实参都是值，第二条只能是同一个 beta-v 收缩。三种情形均给出 $e_1=e_2$。

## 第 5 章

**E5.1.** 先由 \textsc{Ctx-Empty} 得 $\vdash\varnothing\ \mathsf{ctx}$，再由宇宙形成得
$\varnothing\vdash\mathcal U\ \mathsf{type}$，故可形成
$A:\mathcal U$。在该上下文中 \textsc{Var} 给出
$A:\mathcal U\vdash A:\mathcal U$，\textsc{U-El} 给出
$A:\mathcal U\vdash A\ \mathsf{type}$，因而可形成
$A:\mathcal U,x:A$ 并得到 $x:A$。一次 $\Pi$-Intro 给出
$$
A:\mathcal U\vdash\lambda x.x:\Pi x:A.A.
$$
$\Pi x:A.A$ 在 $A:\mathcal U$ 中良构；以
$\mathcal U$ 为外层定义域再用一次 $\Pi$-Intro，得到
$$
\varnothing\vdash\lambda A.\lambda x.x:
\Pi A:\mathcal U.\Pi x:A.A.
$$
整个推导只使用 $\mathcal U\ \mathsf{type}$，没有使用被禁止的 $\mathcal U:\mathcal U$。

**E5.2.** $\Gamma\vdash a\equiv b:A$ 是元理论的判断等价，决定类型检查时哪些表达式按计算规则相同；
$\Gamma\vdash p:\mathsf{Id}_A(a,b)$ 是对象语言中一个可被传递和消去的项。当前系统没有等式反射规则，所以从后者不能推出前者；只能用 $J$ 对 $p$ 做依赖消去。

**E5.3.** 闭项为
$$
(\mathsf S(0),\mathsf{refl}_{\mathsf S(0)}):
\Sigma n:\mathsf{Nat}.
\mathsf{Id}_{\mathsf{Nat}}(n,\mathsf S(0)).
$$
两条投影计算是
$$
\pi_1(\mathsf S(0),\mathsf{refl}_{\mathsf S(0)})
\equiv\mathsf S(0),
\qquad
\pi_2(\mathsf S(0),\mathsf{refl}_{\mathsf S(0)})
\equiv\mathsf{refl}_{\mathsf S(0)}.
$$
第二式的类型经第一分量替换后正是
$\mathsf{Id}_{\mathsf{Nat}}(\mathsf S(0),\mathsf S(0))$。

**E5.4.** 按例 5.13 的递归方向，
$$
\begin{aligned}
\mathsf{plus}\,\overline2\,\overline1
&\equiv
(\lambda n.\mathsf{rec}_{k.\mathsf{Nat}}
  (\overline2;k,r.\mathsf S(r);n))\,\overline1
&&(\Pi\beta)\\
&\equiv
\mathsf{rec}_{k.\mathsf{Nat}}
  (\overline2;k,r.\mathsf S(r);\mathsf S(0))
&&(\Pi\beta)\\
&\equiv
\mathsf S(\mathsf{rec}_{k.\mathsf{Nat}}
  (\overline2;k,r.\mathsf S(r);0))
&&(\mathsf{Nat}\beta_S)\\
&\equiv\mathsf S(\overline2)
&&(\mathsf{Nat}\beta_0)\\
&=\overline3.
\end{aligned}
$$
前两条等价分别收缩 $\mathsf{plus}$ 对第一、第二实参的两个函数 beta-redex。

## 第 6 章

**E6.1.** 从 $\alpha;x:\alpha\vdash x:\alpha$ 用项抽象得到
$\alpha;\varnothing\vdash\lambda x:\alpha.x:\alpha\to\alpha$。因为
$\alpha\notin\mathrm{FTV}(\varnothing)$，再用 \textsc{T-TAbs} 得
$$
\varnothing;\varnothing\vdash
\Lambda\alpha.\lambda x:\alpha.x:
\forall\alpha.\alpha\to\alpha.
$$

**E6.2.** 对异常 monad，$(\eta_A)^*$ 在
$\mathsf{inl}(a)$ 上返回 $\mathsf{inl}(a)$，在
$\mathsf{inr}(\epsilon)$ 上返回同一异常，故等于
$\mathrm{id}_{A+E}$。其次
$f^*(\eta_A(a))=f^*(\mathsf{inl}(a))=f(a)$。对第三律分
$m\in A+E$：若 $m=\mathsf{inl}(a)$，两边都等于
$g^*(f(a))$；若 $m=\mathsf{inr}(\epsilon)$，两边都等于
$\mathsf{inr}(\epsilon)$。因此三条 Kleisli 三元组律全部成立。

**E6.3.** 因
$\mathsf{zero}:N_\mu$，有
$\mathsf{inr}\,\mathsf{zero}:1+N_\mu$，进而
$$
\mathsf{fold}(\mathsf{inr}\,\mathsf{zero}):N_\mu.
$$
所以题中整项类型为 $1+N_\mu$，并由同构递归类型唯一的计算规则一步得到
$$
\mathsf{unfold}(\mathsf{fold}(\mathsf{inr}\,\mathsf{zero}))
\to\mathsf{inr}\,\mathsf{zero}.
$$

**E6.4.** 类型 beta 步是
$(\Lambda\alpha.e_0)[B]\to e_0[\alpha:=B]$；它替换的是类型变量，并同时改变项中的类型标注和结论类型
$C[\alpha:=B]$。第 4 章 T4.2 只处理项变量替换
$e[x:=v]$，既没有类型上下文 $\Delta$，也不能改写类型标注。故 T6.1 的
\textsc{E-TBeta} 情形必须调用引理 6.4 的类型替换，而项 beta 情形才调用引理 6.5。

## 第 7 章

**E7.1.** 小步轨迹为
$$
\langle c_1;c_2,\sigma_0\rangle
\to\langle\mathsf{skip};c_2,\sigma_1\rangle
\to\langle c_2,\sigma_1\rangle
\to\langle\mathsf{skip},\sigma_2\rangle,
$$
其中 $c_1=c_2=(x:=x+1)$、
$\sigma_1(x)=1$、$\sigma_2(x)=2$。对应大步树是
$$
\frac{
 \langle c_1,\sigma_0\rangle\Downarrow\sigma_1
 \qquad
 \langle c_2,\sigma_1\rangle\Downarrow\sigma_2}
{\langle c_1;c_2,\sigma_0\rangle\Downarrow\sigma_2},
$$
两个前提各由赋值规则直接得到。

**E7.2.** 对第一条小步推导的末规则分类。赋值的更新状态由表达式解释唯一决定；while 只有唯一展开规则；条件规则因布尔解释只能取真或假而互斥。Sequence 中，若左命令可步进，只能使用上下文规则，并由归纳假设确定左后继；若左命令是 skip，只能使用 skip-sequence，因为 skip 无后继。两种 sequence 规则不可能同时适用。定义 7.2 的全部规则模式均已覆盖，所以后继配置唯一。

**E7.3.** 令
$w_x=(\lambda x.\lambda z.x,\varnothing)$、
$w_y=(\lambda y.y,\varnothing)$。完整轨迹为
$$
\begin{aligned}
&\langle(\lambda x.\lambda z.x)(\lambda y.y),\varnothing,\mathsf{mt}\rangle_E\\
\leadsto{}&
\langle\lambda x.\lambda z.x,\varnothing,
 \mathsf{arg}(\lambda y.y,\varnothing,\mathsf{mt})\rangle_E\\
\leadsto{}&
\langle\mathsf{arg}(\lambda y.y,\varnothing,\mathsf{mt}),w_x\rangle_R\\
\leadsto{}&
\langle\lambda y.y,\varnothing,\mathsf{fun}(w_x,\mathsf{mt})\rangle_E\\
\leadsto{}&
\langle\mathsf{fun}(w_x,\mathsf{mt}),w_y\rangle_R\\
\leadsto{}&
\langle\lambda z.x,[x\mapsto w_y],\mathsf{mt}\rangle_E\\
\leadsto{}&
\langle\mathsf{mt},(\lambda z.x,[x\mapsto w_y])\rangle_R.
\end{aligned}
$$
前四个后继的卸载项都 alpha 等于原应用；第五步卸载为对象步
$(\lambda x.\lambda z.x)(\lambda y.y)\to\lambda z.\lambda y.y$；最后的抽象返回不改变卸载项。最终闭包卸载为 $\lambda z.\lambda y.y$。

**E7.4.** 记 $W=\mathsf{while}\ b\ \mathsf{do}\ c$。布尔值为真时，对展开后命令的大步推导反演得到
$$
\frac{
 \llbracket b\rrbracket\sigma=\mathsf{true}
 \qquad
 \displaystyle
 \frac{\langle c,\sigma\rangle\Downarrow\rho
       \qquad\langle W,\rho\rangle\Downarrow\tau}
      {\langle c;W,\sigma\rangle\Downarrow\tau}}
{\langle\mathsf{if}\ b\ \mathsf{then}\ c;W\
 \mathsf{else}\ \mathsf{skip},\sigma\rangle\Downarrow\tau}.
$$
上式内层反演给出唯一中间状态 $\rho$。把同三个前提代入 while-true 规则，立即得到
$\langle W,\sigma\rangle\Downarrow\tau$，这就是引理 7.5 该分支所需的向后封闭。

## 第 8 章

**E8.1.** 若 $x\sqsubseteq y$，则
$x,y,y,\ldots$ 是非空递增 omega 链，上确界为 $y$。按题设替代约定，它的像
$f(x),f(y),f(y),\ldots$ 也必须是递增链；特别地 $f(x)\sqsubseteq f(y)$。故 $f$ 单调。保持上确界的等式随后给出
$$
f(y)=\bigsqcup(f(x),f(y),f(y),\ldots),
$$
但单调性已经由“像仍为链”这一明确条件得到。

**E8.2.** $F^4(\bot)$ 恰在 $x=0,1,2,3$ 时有定义，并都返回把 $x$ 置为 $0$ 的状态。从 $x=3$ 出发的计数推导依次使用三次真分支：
$$
\langle W,\sigma_3\rangle\Downarrow_3\sigma_0
$$
的三个循环体前提分别是
$\langle x:=x-1,\sigma_j\rangle\Downarrow\sigma_{j-1}$
（$j=3,2,1$），最内层以
$\llbracket x>0\rrbracket\sigma_0=\mathsf{false}$ 使用零次真分支规则。故引理 8.6 取 $n=4,k=3$。

**E8.3.** 固定偏函数 $q$，令 $p=\bigsqcup_np_n$。若
$(q\circ p)(\sigma)=\tau$，则存在 $\rho$ 使
$p(\sigma)=\rho$ 且 $q(\rho)=\tau$；图并定义给出某个 $n$ 已有
$p_n(\sigma)=\rho$，故该图对属于 $q\circ p_n$。反向包含显然，所以
$$
q\circ\Bigl(\bigsqcup_np_n\Bigr)=\bigsqcup_n(q\circ p_n).
$$
固定 $p$、对递增链 $(q_n)$ 使用同一论证：一旦
$(\bigsqcup_nq_n)(p(\sigma))$ 有值，该图对已经出现在某个 $q_n$ 中。因此
$(\bigsqcup_nq_n)\circ p=\bigsqcup_n(q_n\circ p)$。

**E8.4.** T8.3 使用
$\mu\Phi=\bigsqcup_n\Phi^n(\bot)$，所以每个指称结果都出现在某个有限近似层，并由引理 8.6 产生有限大步推导。任意不动点可能含有最小不动点之外的图对。具体取
$W=\mathsf{while}\ \mathsf{true}\ \mathsf{do}\ \mathsf{skip}$；其泛函满足
$\Phi(g)=g$，故每个偏状态函数都是不动点。选择
$g=\mathrm{id}_S$ 会虚构所有终止结果，但 $W$ 没有任何终止大步推导；最小不动点则是 $\bot_D$。因此反向证明不能用任意不动点。

## 第 9 章

**E9.1.** 赋值公理给出
$$
\{(x=1)[x:=x+1]\}\ x:=x+1\ \{x=1\},
$$
其前置即 $x+1=1$。背景整数算术证明
$x=0\Rightarrow x+1=1$，所以一次 \textsc{H-Consequence} 得目标三元组。

**E9.2.** 引入不被程序修改的逻辑常量 $x_0,y_0$，取前置
$x=x_0\land y=y_0\land y_0\ge0$。执行 $z:=x$ 后建立
$$
I\equiv x=x_0\land z+y=x_0+y_0\land y\ge0.
$$
在 $I\land y>0$ 下，顺序执行 $z:=z+1$、$y:=y-1$ 后有
$(z+1)+(y-1)=z+y$ 且 $y-1\ge0$，所以由两个赋值公理、sequence 和 consequence 得
$\{I\land y>0\}\ c\ \{I\}$。While 规则给出终态
$I\land y\le0$；与 $y\ge0$ 合并得
$y=0$、$z=x_0+y_0$。这完成部分正确性推导；终止性另由自然数变元 $y$ 严格下降证明。

**E9.3.** 对
$L=\mathsf{while}\ \mathsf{true}\ \mathsf{do}\ \mathsf{skip}$，
不存在 $\langle L,\sigma\rangle\Downarrow\tau$。因此定义 9.6 中对 $\tau$ 的蕴含前件恒假，
每个 $\sigma$ 都满足 $\mathrm{wlp}(L,Q)$，所以该断言等于
$\mathsf{true}$。总正确性的 weakest precondition 还要求终止，而没有任何初态满足这一要求，故等于 $\mathsf{false}$。

**E9.4.** 外层归纳对象是有限的 Hoare 语法推导
$\vdash_H\{P\}c\{Q\}$。其 while 末规则只有一个前提推导；外层归纳假设先把这个前提变成语义有效三元组。随后引理 9.4 启动第二个归纳：固定初态和终态，对有限大步推导
$\langle W,\sigma\rangle\Downarrow\tau$ 的高度归纳。False 末规则直接结束；True 末规则对严格更小的余下循环子推导使用内层归纳假设。两个归纳分别下降 Hoare 证明树和运行证明树，不能合并为同一个未说明的“结构归纳”。

## 第 10 章

**E10.1.** 包 $\mathcal A$ 的表达式片段采用带布尔常量的纯 STLC，所以其类型安全由 T4.5 的同样证明得到。指称观察可靠性失败使用两个闭项
$\mathsf{true}$ 与 $\mathsf{false}$：错误指称把它们映到同一元素，但布尔观察区分二者。

**E10.2.** 取两个上下文等价但语法树大小不同的项，例如对不观察源语法的语言，用一个可消去的恒等包装产生较大的项。扩张模型把语法树大小作为第二分量时，两项指称不同。因此被破坏的是
$$
e\approx_{\mathrm{ctx}}e'
\Longrightarrow
\llbracket e\rrbracket=\llbracket e'\rrbracket
$$
这一观察 completeness 方向；指称相等仍可蕴含观察相同，所以 soundness 方向可以保留。

**E10.3.** 令
$P(f)\equiv\exists x.\ f(x)\downarrow=0$。若两个程序计算同一偏函数，则该存在命题真值相同，所以它外延。常零函数满足 $P$，处处未定义函数和常一函数不满足 $P$，所以它非平凡。定义 10.6 的编号翻译因而允许 T10.2 直接排除其总判定器。

**E10.4.** 口号版本可以只要求：“每个部分递归函数都由某个 $L$ 程序表示。”这是纯存在性的表达完备，不给出从机器编号 $e$ 到 $L$ 程序编号的总可计算函数 $a(e)$，也未必给出反向的 $b(i)$。即使有一个判定 $L$ 程序性质的算法，没有 $a$ 就不能把它与输入 $e$ 有效合成为计数器机性质判定器，因此 T10.2 的归约步骤无法形成。定义 10.6 的双向有效翻译正是口号所缺的假设。

## 附录

**EA.1.** 图包含自反、传递。若偏函数 $p,q$ 的图互相包含，则图相等，因而 $p=q$，所以反对称。处处未定义偏函数的空图包含于每个图。

**EA.2.** 对表达式结构归纳。变量的变量集为单元素集，常量的变量集为空；每个有限元运算节点的变量集是有限多个子表达式变量集的有限并，故仍有限。

**EB.1.** 对闭值的类型推导反演。STLC 值语法只有
$\lambda x:A.e$；若它具有函数类型，抽象反演给出主体类型和相同的箭头定义域，故得到所需形状。若扩张值语法，必须按每个值构造子增加 canonical-forms 分支。

**EB.2.** 若 $(s,t)\in R_1\cup R_2$，它属于某个 $R_i$。$s$ 的每个转移由
$R_i$ 提供 $t$ 的匹配转移，后继对仍在 $R_i$，从而在并中；反方向同理。因此
$R_1\cup R_2$ 是双模拟。

**EC.1.** 若 $\Gamma\vdash e_1e_2:B$，应用语法头只能由
\textsc{T-App} 作为末规则产生；读取其两个前提，存在 $A$ 使
$\Gamma\vdash e_1:A\to B$ 且 $\Gamma\vdash e_2:A$。

**EC.2.** 未捕获异常可能既不是普通值，也没有普通求值步，原 progress 二分会把它误判为卡住。可把结论改成“值、可步进或未捕获异常”三分，或把异常构造列入结果语法；随后 preservation 也必须覆盖异常传播规则。
