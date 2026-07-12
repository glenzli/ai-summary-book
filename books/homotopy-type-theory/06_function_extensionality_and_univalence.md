# 第六章：函数外延性、命题外延性与单值性

## 本章目标

本章引入函数外延性（function extensionality）、命题外延性和单值性（univalence）。重点是逐层声明宇宙，区分公理化 HoTT 中的路径计算与 cubical type theory 中依赖具体语法的归约规则，并完整推出沿单值性路径的 transport 定律。

## 依赖前置知识

本章依赖前五章，尤其是等价、路径代数、依赖 transport 和同伦层级。单值性从本章开始可用；之前章节不得逆向依赖它。本章继续采用第一章的非累积宇宙口径，不假设 resizing。

## 6.1 函数外延性

**定义 6.1（逐点作用）.** 设 $A:\mathcal U_i$、$B:A\to\mathcal U_j$，并令

$$
F\coloneqq\prod_{x:A}B(x):\mathcal U_{\max(i,j)}.
$$

对 $f,g:F$，有规范映射

$$
\mathsf{happly}_{f,g}:(f=g)\to\prod_{x:A}(f(x)=g(x)).
$$

它由对 $p:f=g$ 使用 J 定义；反身分支取 $\lambda x.\,\mathsf{refl}_{f(x)}$。因此

$$
\mathsf{happly}_{f,f}(\mathsf{refl}_f)
\equiv
\lambda x.\,\mathsf{refl}_{f(x)}.
$$

**公理 6.2（函数外延性）.** 在层级 $(i,j)$ 上，函数外延性断言：对上述每个 $A,B,f,g$，

$$
\mathsf{isEquiv}(\mathsf{happly}_{f,g}).
$$

从该等价的 fiber 中选出的逆函数记为

$$
\mathsf{funext}_{f,g}:
\left(\prod_{x:A}f(x)=g(x)\right)\to(f=g).
$$

等价数据还给出路径

$$
\mathsf{happly}(\mathsf{funext}(h))=h,
\qquad
\mathsf{funext}(\mathsf{happly}(p))=p.
$$

在本章的公理化口径中，这两式是 propositional computation，不是新增的 judgmental 规则。整个层级多态公理族位于 $\mathcal U_{\max(i,j)+1}$；这来自对 $A:\mathcal U_i$ 和宇宙值族 $B:A\to\mathcal U_j$ 的量化，而不是 resizing。

**验证状态。** 公理化 HoTT 可把函数外延性单列为上述公理；定理 6.11 记录 universe univalence 推出它的外部输入。CCHM cubical type theory 中，函数路径由区间方向逐点给出，函数外延性可在该对象语言内部证明；这是第十六章所述另一套基础规则，不能回填为第一至五章的 judgmental equality。

**命题 6.3（命题值函数类型是命题）.** 设 $A:\mathcal U_i$、$B:A\to\mathcal U_j$。若

$$
q:\prod_{x:A}\mathsf{isProp}(B(x)),
$$

则 $\prod_{x:A}B(x)$ 是命题。

**证明（书内证明，使用函数外延性）.** 给定 $f,g:\prod_{x:A}B(x)$。对每个 $x:A$，项 $q(x)$ 给出路径 $f(x)=g(x)$，故得到

$$
h\coloneqq\lambda x.\,q(x)(f(x))(g(x)):
\prod_{x:A}f(x)=g(x).
$$

取 $\mathsf{funext}(h):f=g$。由于 $f,g$ 任意，这正是 $\mathsf{isProp}(\prod_xB(x))$。附录 F.5 记录同一证明核。$\square$

## 6.2 命题外延性

**定义 6.4（命题外延性）.** 对 $P,Q:\mathcal U_i$，命题外延性是在假设 $\mathsf{isProp}(P)$ 与 $\mathsf{isProp}(Q)$ 下的映射

$$
(P\to Q)\to(Q\to P)\to(P=_{\mathcal U_i}Q).
$$

结论是宇宙 $\mathcal U_i$ 中两个元素的恒等类型，因而该结论本身位于 $\mathcal U_{i+1}$。

**命题 6.5（单值性推出命题外延性）.** 假设第 $i$ 层单值性。若 $P,Q:\mathcal U_i$ 都是命题，并且有 $f:P\to Q$ 与 $g:Q\to P$，则 $P=_{\mathcal U_i}Q$。

**证明（书内证明，使用单值性）.** 先证明 $f$ 是等价。对 $q:Q$，fiber

$$
\mathsf{fib}_f(q)\equiv\sum_{p:P}(f(p)=q)
$$

的中心取 $(g(q),\alpha_q)$，其中 $\alpha_q:f(g(q))=q$ 由 $Q$ 的命题性给出。给定任意 $(p,r):\mathsf{fib}_f(q)$，$P$ 的命题性给出 $s:g(q)=p$。由 $\Sigma$ 路径刻画，还需比较 fiber 第二分量；两者都位于 $Q$ 的路径类型，而命题的路径类型可收缩，故得到所需路径。于是每个 fiber 可收缩，得到 $e_f:P\simeq Q$。应用第 $i$ 层单值性的逆方向

$$
\mathsf{ua}_i(e_f):P=_{\mathcal U_i}Q.
$$

该证明没有使用函数外延性。附录 F.4 给出相同构造的展开。$\square$

## 6.3 从类型路径到等价

**定义 6.6（$\mathsf{idtoequiv}$ 及其层级）.** 固定 $i$，设 $A,B:\mathcal U_i$。首先定义

$$
\mathsf{idtofun}(p)
\coloneqq
\mathsf{transport}^{\lambda X:\mathcal U_i.\,X}(p)
:A\to B
$$

其中 $p:A=_{\mathcal U_i}B$。再对 $p$ 使用 J；反身分支使用恒等函数的等价性，得到

$$
\epsilon(p):\mathsf{isEquiv}(\mathsf{idtofun}(p)).
$$

定义

$$
\mathsf{idtoequiv}_{i,A,B}(p)
\coloneqq
(\mathsf{idtofun}(p),\epsilon(p)):A\simeq B.
$$

这里层级必须分开记录：

$$
A=_{\mathcal U_i}B:\mathcal U_{i+1},
\qquad
A\simeq B:\mathcal U_i,
$$

所以 $\mathsf{idtoequiv}_{i,A,B}$ 作为映射位于 $\mathcal U_{i+1}$。

**命题 6.7（反身路径上的计算）.** 有 judgmental equality

$$
\mathsf{idtoequiv}_{i,A,A}(\mathsf{refl}_A)
\equiv
\mathsf{idEquiv}_A.
$$

**证明.** $\mathsf{idtofun}(\mathsf{refl}_A)$ 由 transport 的 $\beta$-规则计算为恒等函数，等价性分量由 J 的 $\beta$-规则计算为恒等等价证明。因此整个依赖对 judgmentally 计算为 $\mathsf{idEquiv}_A$。$\square$

## 6.4 单值性

**公理 6.8（第 $i$ 层 universe univalence）.** 对任意 $A,B:\mathcal U_i$，映射

$$
\mathsf{idtoequiv}_{i,A,B}:
(A=_{\mathcal U_i}B)\to(A\simeq B)
$$

是等价。把这一层级多态公理族记为 $\mathsf{UA}_i$。由它选择的逆函数记为

$$
\mathsf{ua}_i:(A\simeq B)\to(A=_{\mathcal U_i}B).
$$

量化后的公理类型 $\mathsf{UA}_i$ 位于 $\mathcal U_{i+1}$。该陈述不比较分属不同宇宙层的类型；也不推出 $\mathcal U_i:\mathcal U_i$ 或任何 resizing 原则。

**计算原则 6.9（公理化口径的两条路径）.** 等价的两个逆律给出

$$
\beta_e:
\mathsf{idtoequiv}(\mathsf{ua}_i(e))=e
$$

以及

$$
\eta_p:
\mathsf{ua}_i(\mathsf{idtoequiv}(p))=p.
$$

在公理化 HoTT 中，$\beta_e$ 与 $\eta_p$ 是恒等类型中的项；不能把它们改写成 judgmental equality。

**命题 6.9.1（沿单值性路径的 transport）.** 设 $A,B:\mathcal U_i$、$e:A\simeq B$、$a:A$。在公理化 HoTT 中有路径

$$
\mathsf{transport}^{\lambda X:\mathcal U_i.\,X}
(\mathsf{ua}_i(e),a)
=
e.1(a).
$$

**证明（书内证明，使用单值性）.** 由计算原则 6.9 取

$$
\beta_e:\mathsf{idtoequiv}(\mathsf{ua}_i(e))=e.
$$

对该路径作用第一投影，得到函数路径

$$
\mathsf{ap}_{\mathsf{pr}_1}(\beta_e):
\mathsf{idtofun}(\mathsf{ua}_i(e))=e.1.
$$

再对这个函数路径应用 $\mathsf{happly}$ 并取点 $a$，得到

$$
\mathsf{idtofun}(\mathsf{ua}_i(e))(a)=e.1(a).
$$

按定义 6.6，左边 judgmentally 展开为题设中的 transport，故得到结论。此推导只用了 $\beta_e$、$\mathsf{ap}$ 和 $\mathsf{happly}$；没有把 $\mathsf{ua}_i(e)$ 变成 judgmental equality，也没有使用函数外延性。$\square$

**警告 6.10（内部路径与 cubical 归约）.** 单值性给出 $A=_{\mathcal U_i}B$ 的项，不给出 $A\equiv B:\mathcal U_i$。命题 6.9.1 因而是路径等式。特定 cubical type theory 通过区间、composition 和 Glue 定义单值性，并为 transport 给出额外归约行为；究竟哪些式子 judgmentally 计算，必须按该系统的语法逐条引用，不能从公理 6.8 单独推出。

## 6.5 单值性推出函数外延性

**定理 6.11（universe univalence 推出依赖函数外延性）.** 设 $A:\mathcal U_i$、$B:A\to\mathcal U_j$。若分类各 fiber $B(x)$ 的宇宙 $\mathcal U_j$ 满足 $\mathsf{UA}_j$，则定义 6.1 的每个 $\mathsf{happly}_{f,g}$ 都是等价。基底 $A$ 不必与 fibers 位于同一宇宙。

**验证状态：精确外部输入。** 本书不重证完整长证明。HoTT Book 第 4.9 节先在定理 4.9.4 中由 universe univalence 推出弱函数外延性，再在定理 4.9.5 中推出强依赖函数外延性；本书使用的精确版本和层级边界见附录 T。该外部输入只适用于公理 6.8 的 universe univalence，不能由 categorical、directed 或其他弱单值原则替代。

本书正文允许两种一致口径：可把函数外延性与单值性都显式列为第六章后的原则；也可只列单值性，并把定理 6.11 加入每个函数外延性结论的依赖链。

## 本章小结

本章逐层引入函数外延性和 universe univalence。函数外延性把逐点路径族提升为函数路径；单值性把同一小宇宙中的类型等价提升为宇宙路径。两者都产生 propositional equality，而不是 judgmental equality。命题 6.9.1 完整展示了如何从单值性的逆律推出结构 transport 的实际作用。

## 练习

**练习 6.1.** 用命题 2.9.2 和函数外延性证明常值族上的 transport 函数等于恒等函数，并指出哪一步首次使用函数外延性。

**练习 6.2.** 设 $P,Q:\mathcal U_i$ 是命题，证明 $(P\simeq Q)$ 是命题。

**练习 6.3.** 写出 $\mathsf{idtoequiv}$ 在 $\mathsf{refl}$ 上的完整 J 定义，并分别检查底层函数与等价性证明的计算。

**练习 6.4.** 说明为什么公理化单值性中的 $\beta_e$ 不应改写为 judgmental equality。

**练习 6.5.** 检查 $A=_{\mathcal U_i}B$、$A\simeq B$、$\mathsf{idtoequiv}_{i,A,B}$ 和 $\mathsf{UA}_i$ 的宇宙层级，并解释为什么这些层级不蕴含 resizing。
