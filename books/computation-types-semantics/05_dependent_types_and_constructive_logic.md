# 第 5 章：一个固定的 intensional MLTT 核心

“依赖类型论”不是单一形式系统。是否有 η 规则、等式反射、宇宙自包含或一般递归，都会改变正规化、判等与 canonicity。本章因此先固定一个具体的 intensional Martin-Löf type theory（MLTT）核心，再陈述结构定理。系统含 $\Pi$、$\Sigma$、恒等类型、自然数和一个 predicative Russell-style universe；不含函数外延、公理 K、等式反射或一般递归。

## 5.1 原始语法、上下文与判断

**定义 5.1（原始表达式）。** 类型与项共享一类原始表达式：
$$
\begin{aligned}
t,u,A,B ::= {}&x\mid\mathcal U\mid \Pi x:A.B\mid\lambda x.t\mid t\,u\\
&\mid\Sigma x:A.B\mid(t,u)\mid\pi_1t\mid\pi_2t\\
&\mid\mathsf{Id}_A(t,u)\mid\mathsf{refl}_t
  \mid J_{x,y,p.C}(x.d;a,b,q)\\
&\mid\mathsf{Nat}\mid 0\mid\mathsf S(t)
  \mid\mathsf{rec}_{n.C}(z;n,r.s;m).
\end{aligned}
$$
绑定位置是 $\Pi x:A.B$、$\Sigma x:A.B$、$\lambda x.t$、$J$ 的 $x,y,p$ 与分支中的 $x$，以及自然数递归器中的 $n,r$。原始表达式按 α-等价识别；替换同时、捕获避免地穿过所有构造子。

**定义 5.2（上下文）。** 原始上下文是互异变量声明序列
$x_1:A_1,\ldots,x_k:A_k$。它是否良构由规则判断，而不是由语法自动保证。若 $\Delta=y_1:B_1,\ldots,y_m:B_m$，则
$$
\Delta[x:=a]=y_1:B_1[x:=a],\ldots,y_m:B_m[x:=a].
$$
替换进入后段每个声明，因为这些类型可以依赖 $x$。

**定义 5.3（五类判断）。** 本章只有以下原始判断：
$$
\vdash\Gamma\ \mathsf{ctx},\qquad
\Gamma\vdash A\ \mathsf{type},\qquad
\Gamma\vdash a:A,
$$
$$
\Gamma\vdash A\equiv B\ \mathsf{type},\qquad
\Gamma\vdash a\equiv b:A.
$$
最后两类是判断等价（judgmental equality）。它们属于元语言，不是恒等类型的项。

**定义 5.4（上下文与变量规则）。**
$$
\frac{}{\vdash\varnothing\ \mathsf{ctx}}\;\textsc{Ctx-Empty}
\qquad
\frac{\vdash\Gamma\ \mathsf{ctx}\qquad\Gamma\vdash A\ \mathsf{type}
\qquad x\notin\mathrm{dom}(\Gamma)}
{\vdash\Gamma,x:A\ \mathsf{ctx}}\;\textsc{Ctx-Ext}
$$
$$
\frac{\vdash\Gamma\ \mathsf{ctx}\qquad x:A\in\Gamma}
{\Gamma\vdash x:A}\;\textsc{Var}.
$$

## 5.2 宇宙以及 $\Pi$、$\Sigma$、$\mathsf{Id}$、$\mathsf{Nat}$ 规则

下面的规则就是本章系统的规则边界。未列出的 extensionality、η 或反射原则均不存在。

**定义 5.5（一个 predicative universe）。**
$$
\frac{\vdash\Gamma\ \mathsf{ctx}}{\Gamma\vdash\mathcal U\ \mathsf{type}}\;\textsc{U-Form}
\qquad
\frac{\Gamma\vdash A:\mathcal U}{\Gamma\vdash A\ \mathsf{type}}\;\textsc{U-El}.
$$
没有规则 $\Gamma\vdash\mathcal U:\mathcal U$。因此宇宙本身不是自己的元素。

**定义 5.6（$\Pi$ 规则）。**
$$
\frac{\Gamma\vdash A\ \mathsf{type}\qquad\Gamma,x:A\vdash B\ \mathsf{type}}
{\Gamma\vdash\Pi x:A.B\ \mathsf{type}}\;\textsc{$\Pi$-Form}
$$
$$
\frac{\Gamma\vdash A:\mathcal U\qquad\Gamma,x:A\vdash B:\mathcal U}
{\Gamma\vdash\Pi x:A.B:\mathcal U}\;\textsc{$\Pi$-Small}
$$
$$
\frac{\Gamma,x:A\vdash b:B}{\Gamma\vdash\lambda x.b:\Pi x:A.B}\;\textsc{$\Pi$-Intro}
\qquad
\frac{\Gamma\vdash f:\Pi x:A.B\qquad\Gamma\vdash a:A}
{\Gamma\vdash f\,a:B[x:=a]}\;\textsc{$\Pi$-Elim}.
$$
其唯一原始计算等式是
$$
\Gamma\vdash(\lambda x.b)\,a\equiv b[x:=a]:B[x:=a].
\tag{$\Pi\beta$}
$$
本系统不采用判断性 $\Pi\eta$。

**定义 5.7（$\Sigma$ 规则）。** 形成规则与 $\Pi$ 相同地分为一般形成和小类型闭包：
$$
\frac{\Gamma\vdash A\ \mathsf{type}\qquad\Gamma,x:A\vdash B\ \mathsf{type}}
{\Gamma\vdash\Sigma x:A.B\ \mathsf{type}},
\qquad
\frac{\Gamma\vdash A:\mathcal U\qquad\Gamma,x:A\vdash B:\mathcal U}
{\Gamma\vdash\Sigma x:A.B:\mathcal U}.
$$
项规则为
$$
\frac{\Gamma\vdash a:A\qquad\Gamma\vdash b:B[x:=a]}
{\Gamma\vdash(a,b):\Sigma x:A.B}\;\textsc{$\Sigma$-Intro},
$$
$$
\frac{\Gamma\vdash p:\Sigma x:A.B}{\Gamma\vdash\pi_1p:A},
\qquad
\frac{\Gamma\vdash p:\Sigma x:A.B}{\Gamma\vdash\pi_2p:B[x:=\pi_1p]}.
$$
计算等式为
$$
\pi_1(a,b)\equiv a:A,
\qquad
\pi_2(a,b)\equiv b:B[x:=a].
\tag{$\Sigma\beta_1,\Sigma\beta_2$}
$$
不加入判断性 $\Sigma\eta$。

**定义 5.8（intensional 恒等类型）。**
$$
\frac{\Gamma\vdash A\ \mathsf{type}\qquad\Gamma\vdash a:A\qquad\Gamma\vdash b:A}
{\Gamma\vdash\mathsf{Id}_A(a,b)\ \mathsf{type}}\;\textsc{Id-Form},
$$
$$
\frac{\Gamma\vdash A:\mathcal U\qquad\Gamma\vdash a:A\qquad\Gamma\vdash b:A}
{\Gamma\vdash\mathsf{Id}_A(a,b):\mathcal U}\;\textsc{Id-Small},
$$
$$
\frac{\Gamma\vdash a:A}{\Gamma\vdash\mathsf{refl}_a:\mathsf{Id}_A(a,a)}\;\textsc{Id-Intro}.
$$
消去规则固定为路径归纳 $J$：
$$
\frac{
\begin{array}{c}
\Gamma,x:A,y:A,p:\mathsf{Id}_A(x,y)\vdash C\ \mathsf{type}\\
\Gamma,x:A\vdash d:C[y:=x,p:=\mathsf{refl}_x]\\
\Gamma\vdash a:A\quad\Gamma\vdash b:A\quad
\Gamma\vdash q:\mathsf{Id}_A(a,b)
\end{array}}
{\Gamma\vdash J_{x,y,p.C}(x.d;a,b,q):C[x:=a,y:=b,p:=q]}\;\textsc{Id-Elim}.
$$
计算等式为
$$
J_{x,y,p.C}(x.d;a,a,\mathsf{refl}_a)
\equiv d[x:=a]:C[x:=a,y:=a,p:=\mathsf{refl}_a].
\tag{$J\beta$}
$$
特别地，本系统没有“从 $q:\mathsf{Id}_A(a,b)$ 推出 $a\equiv b:A$”的等式反射规则。

**定义 5.9（自然数及依赖递归器）。**
$$
\frac{\vdash\Gamma\ \mathsf{ctx}}{\Gamma\vdash\mathsf{Nat}:\mathcal U},
\qquad
\frac{\vdash\Gamma\ \mathsf{ctx}}{\Gamma\vdash0:\mathsf{Nat}},
\qquad
\frac{\Gamma\vdash n:\mathsf{Nat}}{\Gamma\vdash\mathsf S(n):\mathsf{Nat}}.
$$
递归器规则为
$$
\frac{
\begin{array}{c}
\Gamma,n:\mathsf{Nat}\vdash C\ \mathsf{type}\qquad
\Gamma\vdash z:C[n:=0]\\
\Gamma,n:\mathsf{Nat},r:C\vdash s:C[n:=\mathsf S(n)]\qquad
\Gamma\vdash m:\mathsf{Nat}
\end{array}}
{\Gamma\vdash\mathsf{rec}_{n.C}(z;n,r.s;m):C[n:=m]}\;\textsc{Nat-Elim}.
$$
其中步进分支中的 $C$ 指 $C[n:=n]$。计算等式是
$$
\mathsf{rec}_{n.C}(z;n,r.s;0)\equiv z,
\tag{$\mathsf{Nat}\beta_0$}
$$
$$
\mathsf{rec}_{n.C}(z;n,r.s;\mathsf S(m))
\equiv s[n:=m,r:=\mathsf{rec}_{n.C}(z;n,r.s;m)].
\tag{$\mathsf{Nat}\beta_S$}
$$

**定义 5.10（判断等价与转换）。** 判断等价是满足下列条件的最小有类型关系：

1. 在 $\Gamma\vdash A\ \mathsf{type}$ 上，$A\equiv A$，并对称、传递；在 $\Gamma\vdash a:A$ 上，$a\equiv a:A$，并对称、传递。
2. α-等价项判断等价；定义 5.6--5.9 所列的全部 β 等式属于判断等价。
3. 原始表达式的每个构造子都保持判断等价。例如，若
   $\Gamma\vdash A\equiv A'\ \mathsf{type}$ 且在经转换识别的上下文 $\Gamma,x:A$ 中有
   $\Gamma,x:A\vdash B\equiv B'\ \mathsf{type}$，则
   $\Gamma\vdash\Pi x:A.B\equiv\Pi x:A'.B'\ \mathsf{type}$；应用、配对、投影、$J$ 和递归器同样逐参数同余。
4. 若 $\Gamma\vdash A\equiv B:\mathcal U$，则 $\Gamma\vdash A\equiv B\ \mathsf{type}$。
5. 转换规则为
$$
\frac{\Gamma\vdash a:A\qquad\Gamma\vdash A\equiv B\ \mathsf{type}}
{\Gamma\vdash a:B}\;\textsc{Conv},
\qquad
\frac{\Gamma\vdash a\equiv b:A\qquad\Gamma\vdash A\equiv B\ \mathsf{type}}
{\Gamma\vdash a\equiv b:B}\;\textsc{Eq-Conv}.
$$

“最小”排除了未声明的函数外延、proof irrelevance、UIP/K 和经典公理。替换按定义 5.1 的绑定结构递归，并满足构造子逐项代入；多变量替换从右向左、捕获避免地复合。

## 5.3 结构定理

只有在上述全部规则固定后，弱化和替换才有确定含义。

**定理 T5.1（弱化与依赖替换）。** 令 $J$ 表示定义 5.3 后四类判断中的任一种。

1. 若 $\Gamma\vdash J$ 且 $\vdash\Gamma,\Delta\ \mathsf{ctx}$，则 $\Gamma,\Delta\vdash J$。
2. 若 $\Gamma\vdash a:A$、$\vdash\Gamma,x:A,\Delta\ \mathsf{ctx}$ 且
   $\Gamma,x:A,\Delta\vdash J$，则
$$
\Gamma,\Delta[x:=a]\vdash J[x:=a].
$$
同一结论也适用于上下文形成判断本身。

**证明。** 两部分都对上下文形成、类型形成、项类型和两类等价推导作同时归纳。

先证单变量尾部弱化：若 $\Gamma\vdash J$、$\Gamma\vdash C\ \mathsf{type}$ 且 $z$ 新鲜，则 $\Gamma,z:C\vdash J$。逐类检查末规则如下。

- \textsc{Ctx-Empty} 与 \textsc{Ctx-Ext}：前者由一次扩张得到；后者对类型前提使用归纳假设后再用 \textsc{Ctx-Ext}。
- \textsc{Var}：原变量仍出现在扩张上下文中。
- \textsc{U-Form}、\textsc{U-El} 和 $\mathsf{Nat}$ 三条规则：上下文前提由扩张保持，项或小类型前提用归纳假设。
- $\Pi$ 与 $\Sigma$：形成、小类型闭包、引入和消去的每个前提分别使用归纳假设；遇到绑定变量时先 α-改名为不同于 $z$ 的变量，再重用同一规则。
- $\mathsf{Id}$：形成、小类型闭包和反身引入直接弱化所有前提；$J$ 的 motive、反身分支、两个端点与路径前提分别弱化后重用 \textsc{Id-Elim}。
- $\mathsf{Nat}$ 递归器：motive、零分支、步进分支和主参数四个前提分别弱化；绑定的 $n,r$ 先取新鲜名。
- 等价的自反、对称、传递、构造子同余和全部 β 生成规则：对其全部有类型前提使用归纳假设，再应用同一生成规则。\textsc{Conv} 与 \textsc{Eq-Conv} 分别弱化类型/等价两个前提。

这列举了定义 5.4--5.10 的全部规则模式，所以单变量弱化成立。对 $\Delta$ 的长度归纳、逐个追加声明，得到第 1 项。

再证替换。对最后一条推导规则作同时归纳；捕获风险通过先 α-改名绑定变量消除。

- 上下文规则：空后段不变；扩张末声明 $y:B$ 时，归纳假设先给出替换后的前缀合法，并给出其中的 $B[x:=a]$ 为类型，再用 \textsc{Ctx-Ext}。
- \textsc{Var}：若变量是 $x$，目标是把 $\Gamma\vdash a:A$ 沿 $\Delta[x:=a]$ 弱化，已由第 1 项得到；若变量在 $\Gamma$ 中，其声明不变；若变量在 $\Delta$ 中，替换后的上下文恰含声明 $y:B[x:=a]$。三种位置穷尽变量规则。
- 宇宙和 $\mathsf{Nat}$ 的无绑定规则：替换其前提后重用原规则。
- $\Pi$、$\Sigma$ 的形成、smallness、引入和消去：归纳假设分别作用于域、余族和项前提；由于替换与 $\Pi,\Sigma,\lambda$、应用、配对和投影逐构造子交换，结论正是原结论的替换。
- $\mathsf{Id}$ 形成与反身规则同理。对 $J$，分别替换 motive、分支、端点和路径；同时替换的结合律保证结论类型是
  $C[x:=a_0,y:=b_0,p:=q_0][x_0:=a]$ 的捕获避免形式，故可重用 \textsc{Id-Elim}。
- 对自然数递归器，分别替换 motive、零分支、步进分支和主参数。替换与
  $s[n:=m,r:=R]$ 的交换由绑定变量新鲜和捕获避免复合定义直接得到。
- 等价规则：等价关系的三条结构规则和构造子同余由归纳假设保持。每个 β 生成式替换后仍是同一 β 生成式的实例；例如
  $((\lambda y.b)c)[x:=a]=(\lambda y.b[x:=a])c[x:=a]$ 的右侧计算为
  $b[x:=a][y:=c[x:=a]]$，与 $b[y:=c][x:=a]$ α-等价。其余 $\Sigma\beta$、$J\beta$、$\mathsf{Nat}\beta$ 同理由同时替换结合律成立。\textsc{Conv} 与 \textsc{Eq-Conv} 对两个前提使用归纳假设。

所有原始规则模式均已覆盖，故第 2 项以及同步的上下文结论成立。证毕。

**定理 T5.2（Regularity）。**

1. 若 $\Gamma\vdash a:A$，则 $\vdash\Gamma\ \mathsf{ctx}$ 且 $\Gamma\vdash A\ \mathsf{type}$。
2. 若 $\Gamma\vdash a\equiv b:A$，则 $\Gamma\vdash a:A$、$\Gamma\vdash b:A$ 且 $\Gamma\vdash A\ \mathsf{type}$。
3. 若 $\Gamma\vdash A\equiv B\ \mathsf{type}$，则 $\Gamma\vdash A\ \mathsf{type}$ 且 $\Gamma\vdash B\ \mathsf{type}$。

**证明。** 对三类推导同时归纳。\textsc{Var} 的声明类型在其声明前缀中良构，由 T5.1 弱化到整个上下文。$\Pi$/$\Sigma$ 引入的结论类型由对应形成规则得到；消去结论中的族替换由 T5.1(2) 得到。$\mathsf{Id}$、$J$ 和自然数递归器的结论类型分别由其显式 motive/形成前提及替换定理得到。等价的自反、对称、传递和同余规则把所需类型作为前提携带；全部 β 规则的两端由相应引入/消去规则赋型。\textsc{Conv} 与 \textsc{Eq-Conv} 使用归纳假设和类型等价的两端 regularity。定义 5.4--5.10 没有其他末规则，三项遂同时成立。证毕。

## 5.4 三个可核对的推导与计算

**例 5.11（宇宙内多态恒等函数的推导）。** 令
$$
\mathsf{id}=\lambda A.\lambda x.x.
$$
由 $A:\mathcal U\vdash A:\mathcal U$ 和 \textsc{U-El} 得 $A:\mathcal U\vdash A\ \mathsf{type}$；于是
$$
\frac{
 \frac{A:\mathcal U,x:A\vdash x:A}
 {A:\mathcal U\vdash\lambda x.x:\Pi x:A.A}\;\textsc{$\Pi$-Intro}}
{\varnothing\vdash\lambda A.\lambda x.x:
 \Pi A:\mathcal U.\Pi x:A.A}\;\textsc{$\Pi$-Intro}.
$$
外层 $\Pi A:\mathcal U$ 是合法类型，因为 $\mathcal U$ 是类型；它不要求 $\mathcal U:\mathcal U$。

**例 5.12（依赖配对的完整见证）。** 目标类型是
$$
E=\Sigma n:\mathsf{Nat}.\mathsf{Id}_{\mathsf{Nat}}(n,0).
$$
由 $\varnothing\vdash0:\mathsf{Nat}$ 得
$\varnothing\vdash\mathsf{refl}_0:\mathsf{Id}_{\mathsf{Nat}}(0,0)$，因此 \textsc{$\Sigma$-Intro} 给出
$$
\varnothing\vdash(0,\mathsf{refl}_0):E.
$$
投影计算实际得到
$$
\pi_1(0,\mathsf{refl}_0)\equiv0:\mathsf{Nat},
\qquad
\pi_2(0,\mathsf{refl}_0)\equiv\mathsf{refl}_0:
\mathsf{Id}_{\mathsf{Nat}}(0,0).
$$

**例 5.13（由递归器定义加法并计算）。** 在
$m:\mathsf{Nat},k:\mathsf{Nat},r:\mathsf{Nat}$ 下有
$\mathsf S(r):\mathsf{Nat}$，故
$$
\mathsf{plus}=\lambda m.\lambda n.
\mathsf{rec}_{k.\mathsf{Nat}}(m;k,r.\mathsf S(r);n)
$$
具有类型 $\Pi m:\mathsf{Nat}.\Pi n:\mathsf{Nat}.\mathsf{Nat}$。令
$\overline1=\mathsf S(0)$、$\overline2=\mathsf S(\mathsf S(0))$，则判断等价轨迹为
$$
\begin{aligned}
\mathsf{plus}\,\overline1\,\overline2
&\equiv\mathsf{rec}(\overline1;k,r.\mathsf S(r);\mathsf S(\mathsf S(0)))\\
&\equiv\mathsf S(\mathsf{rec}(\overline1;k,r.\mathsf S(r);\mathsf S(0)))\\
&\equiv\mathsf S(\mathsf S(\mathsf{rec}(\overline1;k,r.\mathsf S(r);0)))\\
&\equiv\mathsf S(\mathsf S(\overline1)).
\end{aligned}
$$
每一步分别使用两次 $\mathsf{Nat}\beta_S$ 和一次 $\mathsf{Nat}\beta_0$。

## 5.5 命题即类型的精确范围

固定一阶直觉自然演绎片段，其原子命题被解释为小类型。定义翻译
$$
\begin{aligned}
\lVert P\Rightarrow Q\rVert&=\Pi \_:\lVert P\rVert.\lVert Q\rVert,\\
\lVert P\land Q\rVert&=\Sigma \_:\lVert P\rVert.\lVert Q\rVert,\\
\lVert\forall x:A.P(x)\rVert&=\Pi x:A.\lVert P(x)\rVert,\\
\lVert\exists x:A.P(x)\rVert&=\Sigma x:A.\lVert P(x)\rVert,\\
\lVert a=_A b\rVert&=\mathsf{Id}_A(a,b).
\end{aligned}
$$

量词规则采用标准特征变量条件：$\forall$-引入的对象变量不自由出现于任何未解除假设，$\exists$-消去的见证变量与其证明变量不自由出现于结论及其余未解除假设。等号消去采用可依赖于等号两端和等号证明的标准替换规则。

**定理 T5.3（自然演绎规则的类型保持翻译）。** 若直觉自然演绎在假设
$P_1,\ldots,P_n$ 下推出 $Q$，则在上下文
$p_1:\lVert P_1\rVert,\ldots,p_n:\lVert P_n\rVert$ 中可构造项
$q:\lVert Q\rVert$。

**证明。** 对自然演绎推导归纳，并逐一处理该片段的规则。

- 假设规则翻译为 \textsc{Var}。
- $\Rightarrow$-引入/消去分别翻译为 $\Pi$-Intro/$\Pi$-Elim。
- $\land$-引入翻译为 $\Sigma$ 配对；左右消去分别翻译为 $\pi_1,\pi_2$。非依赖情形中第二分量类型不依赖第一分量。
- $\forall$-引入在对象变量不自由出现于未解除假设的侧条件下翻译为 $\Pi$-Intro；$\forall$-消去翻译为应用和类型替换。
- $\exists$-引入把见证 $a$ 与归纳得到的 $P(a)$ 证明配对；$\exists$-消去把依赖配对送入一个由 $\Sigma$ 消去可定义的函数。具体地，对
  $p:\Sigma x:A.P(x)$ 使用 $\pi_1p,\pi_2p$ 代入归纳所得分支项。
- 等号反身性翻译为 $\mathsf{refl}$；等号消去翻译为 $J$。

这些正是所选自然演绎片段的全部规则。证毕。

该定理不包含排中律、双重否定消去、函数外延或 proof irrelevance。若把它们加入为常量，必须分别声明其类型和对计算性质的影响。

## 5.6 弱头正规化与 canonicity 的外部输入

**外部输入 EI-5（本章 MLTT 核心的弱头正规化与自然数 canonicity）。** 对定义 5.1--5.10 的系统，假定 $\mathcal U:\mathcal U$、等式反射、判断性 η、一般递归和额外公理均不存在。每个良类型项沿本章 β/递归器计算规则有限步归约到弱头正规形；并且若
$\varnothing\vdash n:\mathsf{Nat}$，则存在唯一 $k\in\mathbb N$ 使
$$
\varnothing\vdash n\equiv\mathsf S^k(0):\mathsf{Nat}.
$$
这里“唯一”是自然数 $k$ 的元语言唯一性。该输入不声称本章无 η 判断等价已有一个由所引文献直接给出的判定器；后文也不使用这项更强结论。向系统加入 quotient、univalence、高阶归纳类型或一般递归后，弱头正规化与 canonicity 均不能自动沿用。

**证明路线（不计作书内证明）。** 主来源用 reducibility/逻辑关系证明基本引理和弱头正规化，再由闭自然数项不可能是 neutral 推出 canonicity。来源系统还含空类型与 $\Pi/\Sigma$ 判断性 η；本书只把不依赖这些新增规则的弱头归约和自然数结论限制到其子系统。`SOURCES.md` 给出一个 universe、$\Pi/\Sigma/\mathsf{Id}/\mathsf{Nat}$、大消去以及规则差异的精确定位。

## 5.7 MLTT 元理论的责任边界

本章在完整列出核心系统后，内部证明 T5.1 的弱化/替换、T5.2 的 regularity 和 T5.3 的规则翻译。弱头正规化与自然数 canonicity 统一登记为 EI-5；后文不得把 EI-5 推广到未声明的类型论扩张，也不得从它额外推出本章判断等价的算法判定性。

## 练习

**练习 E5.1.** 重建例 5.11 的每个上下文形成与类型形成前提。

**练习 E5.2.** 解释 $\Gamma\vdash a\equiv b:A$ 与 $\Gamma\vdash p:\mathsf{Id}_A(a,b)$ 的区别，并指出本系统为何不能从后者推出前者。

**练习 E5.3.** 给出 $\Sigma n:\mathsf{Nat}.\mathsf{Id}_{\mathsf{Nat}}(n,\mathsf S(0))$ 的闭项及两个投影的计算。

**练习 E5.4.** 计算 $\mathsf{plus}\,\overline2\,\overline1$ 的完整判断等价链，并标注每一步 β 规则。
