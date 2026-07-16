# 第 4 章：简单类型、CBV 类型安全与正规化边界

本章固定一个最小而完整的简单类型 λ 演算（simply typed lambda calculus, STLC）。对象语言只有变量、带类型抽象和应用；动态语义固定为从左到右的 call-by-value（CBV）。本章证明的是这一个演算的类型安全，不把无类型 β 归约、call-by-name 或带常量的扩张悄悄混入证明。

## 4.1 原始语法、上下文与替换

**定义 4.1（类型、项和值）。** 给定非空基类型集合 $\mathsf{Base}$，类型和项由
$$
A,B ::= \iota\mid A\to B\quad(\iota\in\mathsf{Base}),
$$
$$
e ::= x\mid \lambda x:A.e\mid e\ e
$$
生成。应用左结合；值恰为 $v::=\lambda x:A.e$。类型构造子 $\to$ 是自由构造子，所以
$$
A_1\to A_2=B_1\to B_2\quad\Longrightarrow\quad A_1=B_1\text{ 且 }A_2=B_2.
$$

**定义 4.2（上下文约定）。** 上下文 $\Gamma$ 是从变量到类型的有限偏函数。写 $x:A\in\Gamma$ 表示 $\Gamma(x)=A$；$\Gamma,x:A$ 只在 $x\notin\mathrm{dom}(\Gamma)$ 时定义。上下文不记录声明次序，因此交换不需要单独规则。若 $\Gamma$ 是 $\Gamma'$ 的子映射，写 $\Gamma\subseteq\Gamma'$。

本章在 α-等价类上工作。绑定变量总可改名为避开当前证明中有限多个变量；等式未特别说明时均理解为 α-等价意义下的等式。

**定义 4.3（捕获避免替换）。** $e[x:=s]$ 对 $e$ 结构递归：
$$
x[x:=s]=s,\qquad y[x:=s]=y\ (y\ne x),\qquad
(e_1e_2)[x:=s]=e_1[x:=s]e_2[x:=s].
$$
对抽象，
$$
(\lambda y:A.e)[x:=s]=
\begin{cases}
\lambda y:A.e,&y=x,\\
\lambda y:A.e[x:=s],&y\ne x\text{ 且 }y\notin\mathrm{FV}(s),\\
\lambda z:A.(e[y:=z])[x:=s],&y\ne x\text{ 且 }y\in\mathrm{FV}(s),
\end{cases}
$$
第三行选择 $z\notin\mathrm{FV}(e)\cup\mathrm{FV}(s)\cup\{x,y\}$。第 3 章 T3.1 保证不同新鲜选择给出 α-等价结果，故该运算在 α-等价类上良定义。

**定义 4.4（类型规则）。** 类型判断是下列规则生成的最小关系：
$$
\frac{x:A\in\Gamma}{\Gamma\vdash x:A}\;\textsc{T-Var}
\qquad
\frac{\Gamma,x:A\vdash e:B}{\Gamma\vdash\lambda x:A.e:A\to B}\;\textsc{T-Abs}
$$
$$
\frac{\Gamma\vdash e_1:A\to B\qquad\Gamma\vdash e_2:A}
{\Gamma\vdash e_1e_2:B}\;\textsc{T-App}.
$$

**引理 4.1（自由变量受上下文控制）。** 若 $\Gamma\vdash e:A$，则 $\mathrm{FV}(e)\subseteq\mathrm{dom}(\Gamma)$。

**证明。** 对给定类型推导归纳。

- 末规则为 \textsc{T-Var} 时，$e=x$ 且 $x\in\mathrm{dom}(\Gamma)$。
- 末规则为 \textsc{T-App} 时，对两个前提使用归纳假设并取并集。
- 末规则为 \textsc{T-Abs} 时，前提给出 $\mathrm{FV}(e_0)\subseteq\mathrm{dom}(\Gamma)\cup\{x\}$，故 $\mathrm{FV}(\lambda x:A.e_0)=\mathrm{FV}(e_0)\setminus\{x\}\subseteq\mathrm{dom}(\Gamma)$。

三种类型规则均已覆盖。证毕。

**定理 T4.1（Weakening）。** 若 $\Gamma\vdash e:A$ 且 $\Gamma\subseteq\Gamma'$，则 $\Gamma'\vdash e:A$。

**证明。** 对 $\Gamma\vdash e:A$ 的推导归纳。

- \textsc{T-Var}：由 $x:A\in\Gamma$ 和子映射关系得 $x:A\in\Gamma'$，再用 \textsc{T-Var}。
- \textsc{T-App}：归纳假设分别给出 $\Gamma'\vdash e_1:B\to A$ 与 $\Gamma'\vdash e_2:B$，用 \textsc{T-App}。
- \textsc{T-Abs}：把绑定变量 α-改名，使 $x\notin\mathrm{dom}(\Gamma')$。由
  $\Gamma,x:B\subseteq\Gamma',x:B$，归纳假设给出 $\Gamma',x:B\vdash e_0:C$，再用 \textsc{T-Abs}。

三种末规则均已覆盖。证毕。

## 4.2 反演、唯一性与 canonical forms

**引理 4.2（类型反演）。**

1. 若 $\Gamma\vdash x:A$，则 $x:A\in\Gamma$。
2. 若 $\Gamma\vdash\lambda x:B.e:A$，则存在唯一 $C$，使 $A=B\to C$ 且 $\Gamma,x:B\vdash e:C$。
3. 若 $\Gamma\vdash e_1e_2:C$，则存在 $B$，使 $\Gamma\vdash e_1:B\to C$ 且 $\Gamma\vdash e_2:B$。

**证明。** 类型关系只由定义 4.4 的三条规则生成。变量、抽象、应用三种语法头分别只能由 \textsc{T-Var}、\textsc{T-Abs}、\textsc{T-App} 作为末规则产生；读取对应前提即得 1、2、3。第 2 项中 $C$ 的唯一性来自箭头类型构造子的单射性。证毕。

**引理 4.3（类型唯一性）。** 若 $\Gamma\vdash e:A$ 且 $\Gamma\vdash e:B$，则 $A=B$。

**证明。** 对 $e$ 结构归纳。

- $e=x$：由反演，两种类型都等于有限映射 $\Gamma(x)$。
- $e=\lambda x:C.e_0$：反演得 $A=C\to A_0$、$B=C\to B_0$，并有 $\Gamma,x:C\vdash e_0:A_0$ 与 $\Gamma,x:C\vdash e_0:B_0$。归纳假设给出 $A_0=B_0$。
- $e=e_1e_2$：反演得某些 $C,D$，使 $\Gamma\vdash e_1:C\to A$ 与 $\Gamma\vdash e_1:D\to B$。对 $e_1$ 的归纳假设给出 $C\to A=D\to B$；箭头单射性给出 $A=B$。

三种项构造均已覆盖。证毕。

**引理 4.4（Canonical forms）。** 若 $\varnothing\vdash v:A\to B$ 且 $v$ 是值，则存在 $x,e$，使 $v=\lambda x:A.e$ 且 $x:A\vdash e:B$。

**证明。** 由值语法，$v=\lambda x:C.e$。对其类型判断使用引理 4.2(2)，得到 $A\to B=C\to D$ 与 $x:C\vdash e:D$。箭头单射性给出 $C=A,D=B$。证毕。

**定理 T4.2（替换）。** 若 $\Gamma,x:A\vdash e:B$ 且 $\Gamma\vdash s:A$，则 $\Gamma\vdash e[x:=s]:B$。

**证明。** 对 $\Gamma,x:A\vdash e:B$ 的推导归纳。

- \textsc{T-Var}：设 $e=y$。若 $y=x$，则 $B=A$，目标正是前提 $\Gamma\vdash s:A$。若 $y\ne x$，则 $y:B\in\Gamma$，且 $y[x:=s]=y$，由 \textsc{T-Var} 得目标。
- \textsc{T-App}：末规则前提为 $\Gamma,x:A\vdash e_1:C\to B$ 和 $\Gamma,x:A\vdash e_2:C$。两次归纳假设给出 $\Gamma\vdash e_1[x:=s]:C\to B$、$\Gamma\vdash e_2[x:=s]:C$；用 \textsc{T-App}。
- \textsc{T-Abs}：把绑定变量 α-改名为 $y$，使 $y\ne x$、$y\notin\mathrm{FV}(s)$ 且 $y\notin\mathrm{dom}(\Gamma)$。末规则前提为
  $\Gamma,x:A,y:C\vdash e_0:D$。有限映射上下文可交换新鲜声明，故也可写作 $\Gamma,y:C,x:A\vdash e_0:D$。由 T4.1，$\Gamma,y:C\vdash s:A$。对前述主体推导使用归纳假设，得到 $\Gamma,y:C\vdash e_0[x:=s]:D$；\textsc{T-Abs} 给出
  $\Gamma\vdash\lambda y:C.e_0[x:=s]:C\to D$，这正是捕获避免替换后的项。

三种末规则均已覆盖。证毕。

## 4.3 从左到右的 CBV 动态语义

**定义 4.5（CBV 一步求值）。** 关系 $e\to e'$ 由以下三条规则生成：
$$
\frac{e_1\to e_1'}{e_1e_2\to e_1'e_2}\;\textsc{E-App1}
\qquad
\frac{e_2\to e_2'}{v_1e_2\to v_1e_2'}\;\textsc{E-App2}
$$
$$
(\lambda x:A.e)\,v\to e[x:=v]\;\textsc{E-BetaV}.
$$
第二条要求左项已经是值，第三条要求实参是值，因此求值次序被完全固定。称项 $e$ **卡住**，若它不是值且不存在 $e'$ 使 $e\to e'$。

**例 4.5（完整类型推导与求值轨迹）。** 固定基类型 $\iota$，令
$$
I=\lambda x:\iota.x,\qquad
M=(\lambda f:\iota\to\iota.f)\,((\lambda g:\iota\to\iota.g)\,I).
$$
内层恒等项的推导为
$$
\frac{x:\iota\vdash x:\iota}{\varnothing\vdash I:\iota\to\iota}\;\textsc{T-Abs}.
$$
同理，$\varnothing\vdash\lambda g:\iota\to\iota.g:(\iota\to\iota)\to(\iota\to\iota)$；两次 \textsc{T-App} 因而给出 $\varnothing\vdash M:\iota\to\iota$。CBV 轨迹必须先处理实参：
$$
M\to(\lambda f:\iota\to\iota.f)\,I\to I.
$$
第一步由 \textsc{E-App2} 包住内层 \textsc{E-BetaV}，第二步由 \textsc{E-BetaV}。每个中间项仍有类型 $\iota\to\iota$。

**定理 T4.3（Preservation）。** 若 $\Gamma\vdash e:A$ 且 $e\to e'$，则 $\Gamma\vdash e':A$。

**证明。** 对 $e\to e'$ 的推导归纳。

- \textsc{E-App1}：$e=e_1e_2$、$e'=e_1'e_2$ 且 $e_1\to e_1'$。由引理 4.2(3)，存在 $B$ 使 $\Gamma\vdash e_1:B\to A$、$\Gamma\vdash e_2:B$。归纳假设给出 $\Gamma\vdash e_1':B\to A$，再用 \textsc{T-App}。
- \textsc{E-App2}：$e=v_1e_2$、$e'=v_1e_2'$ 且 $e_2\to e_2'$。反演得到某个 $B$ 使 $\Gamma\vdash v_1:B\to A$、$\Gamma\vdash e_2:B$。归纳假设给出 $\Gamma\vdash e_2':B$，再用 \textsc{T-App}。
- \textsc{E-BetaV}：$e=(\lambda x:B.e_0)v$ 且 $e'=e_0[x:=v]$。对整个应用反演，得到某个 $C$，使
  $\Gamma\vdash\lambda x:B.e_0:C\to A$ 且 $\Gamma\vdash v:C$。再对抽象反演，得到某个 $D$，使
  $C\to A=B\to D$ 且 $\Gamma,x:B\vdash e_0:D$。箭头单射性给出 $C=B,D=A$；T4.2 因而给出 $\Gamma\vdash e_0[x:=v]:A$。

三条求值规则均已覆盖。注意 β 情形使用的是应用反演和抽象反演，而不是类型唯一性。证毕。

**定理 T4.4（Progress）。** 若 $\varnothing\vdash e:A$，则 $e$ 是值，或存在 $e'$ 使 $e\to e'$。

**证明。** 对类型推导归纳。

- \textsc{T-Var} 不可能为末规则，因为空上下文没有变量声明。
- \textsc{T-Abs} 的结论本身是值。
- \textsc{T-App}：有 $\varnothing\vdash e_1:B\to A$ 与 $\varnothing\vdash e_2:B$。对第一前提使用归纳假设。若 $e_1\to e_1'$，用 \textsc{E-App1}。否则 $e_1$ 是值；由引理 4.4，$e_1=\lambda x:B.e_0$。再对第二前提使用归纳假设。若 $e_2\to e_2'$，用 \textsc{E-App2}；否则 $e_2$ 是值，用 \textsc{E-BetaV}。

所有可能的类型末规则均已覆盖。证毕。

**定理 T4.5（多步保持与类型安全）。** 若 $\varnothing\vdash e:A$ 且 $e\to^*e'$，则 $\varnothing\vdash e':A$，并且 $e'$ 不会卡住。

**证明。** 对 $e\to^*e'$ 的长度归纳。零步时类型前提不变；若
$e\to e_1\to^*e'$，T4.3 给出 $\varnothing\vdash e_1:A$，归纳假设给出 $\varnothing\vdash e':A$。对最终类型判断使用 T4.4，$e'$ 是值或仍可步进，故不可能卡住。证毕。

## 4.4 正规化是独立的深结果

T4.5 排除的是可达的卡住项，并不从逻辑形式上排除无限求值。纯 STLC 还满足更强的强正规化，但其证明使用逻辑关系/可归约性方法，本书将它精确隔离为外部输入。

**外部输入 EI-4（纯 STLC 强正规化）。** 对定义 4.1 与 4.4 的纯 STLC，令 $\to_\beta$ 为可在任意项上下文中收缩 β-redex 的完整 β 归约。若 $\Gamma\vdash e:A$，则不存在无限序列
$$
e=e_0\to_\beta e_1\to_\beta e_2\to_\beta\cdots.
$$
因此定义 4.5 的 CBV 子关系也终止。该输入不参与 T4.3--T4.5。

**证明路线（不计作书内证明）。** 参考来源对每个类型 $A$ 定义可归约项集合 $\mathcal R_A$：基类型取强正规化项，箭头类型要求对每个 $u\in\mathcal R_A$ 都有 $t\,u\in\mathcal R_B$。关键工作是证明候选集对归约和中性展开封闭，再以类型推导归纳证明基本引理。完整责任由 `SOURCES.md` 中 EI-4 的定位承担。

**边界例 4.6（一般递归破坏正规化）。** 若扩张语言含
$\mathsf{fix}_A:(A\to A)\to A$ 及规则 $\mathsf{fix}_A\,f\to f(\mathsf{fix}_A\,f)$，则
$$
\mathsf{fix}_A(\lambda x:A.x)
\to(\lambda x:A.x)(\mathsf{fix}_A(\lambda x:A.x))
\to\mathsf{fix}_A(\lambda x:A.x)\to\cdots.
$$
每一步都保持类型 $A$，但轨迹无限；故 preservation、progress 与终止是不同结论。

## 4.5 类型安全与正规化的责任边界

T4.1--T4.5 均在本章完整证明。唯一外部输入 EI-4 是纯 STLC 的强正规化；它只支持终止性说明，不承担类型安全证明中的任何步骤。

## 练习

**练习 E4.1.** 写出 $\varnothing\vdash\lambda x:A.x:A\to A$ 的完整推导树。

**练习 E4.2.** 用引理 4.2 证明不存在类型 $A,B$ 使 $x:A\vdash x\,x:B$。

**练习 E4.3.** 为例 4.5 的两个求值步骤分别写出所用规则，并用 T4.3 核对中间项类型。

**练习 E4.4.** 证明本章 CBV 一步求值确定：若 $e\to e_1$ 且 $e\to e_2$，则 $e_1=e_2$。
