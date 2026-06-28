# 第八章：正合性检测与第一层 Ext

## 本章目标

本章说明如何用极不连通测试对象检测凝聚阿贝尔群中的正合性，并给出 $\operatorname{Ext}$ 的第一层入口。目标不是计算 Scholze 讲义中的深层 Ext 公式，而是建立可继续做同调代数的基本机器。

## 依赖前置知识

需要第六章的极不连通空间正合取值，以及第七章的投射生成元。

## 8.1 正合性的 ED 检测

设

$$
A\xrightarrow{f}B\xrightarrow{g}C
$$

是 $\mathbf{CondAb}$ 中的复形，即 $g\circ f=0$。

**定理 8.1.** 上述复形在 $\mathbf{CondAb}$ 中正合，当且仅当对每个极不连通紧 Hausdorff 空间 $E$，阿贝尔群复形

$$
A(E)\xrightarrow{f_E}B(E)\xrightarrow{g_E}C(E)
$$

正合。

**证明.** 若原复形正合，则由定理 6.11，取值函子 $(-)(E)$ 正合，因此在每个 $E$ 上得到正合复形。

反过来，设每个极不连通 $E$ 上取值后正合。令

$$
K=\ker(g),\qquad I=\operatorname{im}(f)
$$

这里 $I$ 是 $f$ 的 sheaf-theoretic image。由于 $g\circ f=0$，有单态

$$
I\hookrightarrow K.
$$

要证明正合，只需证明商 sheaf

$$
Q=K/I
$$

为零。对任意极不连通 $E$，由取值正合性知

$$
I(E)=K(E),
$$

这里使用了第六章定理 6.11：取值函子 $(-)(E)$ 正合，因此它保持 image、kernel 和 quotient。故 $Q(E)=0$。现在取任意 $S\in\mathbf{CHaus}$ 和 $q\in Q(S)$。由定理 6.9，存在极不连通覆盖 $p:E\to S$。限制后

$$
p^*q\in Q(E)=0.
$$

由于 $Q$ 是 sheaf，且 $p:E\to S$ 是覆盖，若截面在覆盖上为零，则截面本身为零。因此 $q=0$。故 $Q=0$，从而 $I=K$，复形正合。证毕。

**注 8.2.** 这条定理是本书到目前为止最重要的实用结论之一。它说明：虽然凝聚阿贝尔群是 sheaf 范畴对象，但正合性可以在足够好的测试空间上逐点检查。

## 8.2 Hom 与取值

第七章已经证明：

$$
\operatorname{Hom}_{\mathbf{CondAb}}(\mathbb Z[\underline S],A)\cong A(S).
$$

当 $S=E$ 极不连通时，左边的源是投射对象。因此对任意短正合列

$$
0\to A'\to A\to A''\to 0
$$

应用

$$
\operatorname{Hom}(\mathbb Z[\underline E],-)
$$

仍正合。这与定理 6.11 是同一事实的两种表达。

## 8.3 Ext 的定义

因为 $\mathbf{CondAb}$ 有足够多的投射对象，可以定义右导出函子。

**定义 8.3.** 对 $A,B\in\mathbf{CondAb}$，取 $A$ 的投射分解

$$
\cdots\to P_2\to P_1\to P_0\to A\to 0.
$$

将其代入复形

$$
0\to
\operatorname{Hom}(P_0,B)
\to
\operatorname{Hom}(P_1,B)
\to
\operatorname{Hom}(P_2,B)
\to\cdots
$$

定义

$$
\operatorname{Ext}^n_{\mathbf{CondAb}}(A,B)
=
H^n(\operatorname{Hom}(P_\bullet,B)).
$$

标准同调代数表明，该定义与投射分解选择无关。

**例 8.4.** 若 $E$ 极不连通，则

$$
\operatorname{Ext}^n_{\mathbf{CondAb}}(\mathbb Z[\underline E],B)=0
\qquad n>0.
$$

**证明.** $\mathbb Z[\underline E]$ 是投射对象，因此可取长度为零的投射分解。高阶同调为零。证毕。

## 8.4 第一类计算：自由对象

设

$$
P=\bigoplus_\alpha \mathbb Z[\underline {E_\alpha}],
$$

其中每个 $E_\alpha$ 极不连通。则 $P$ 投射，并且

$$
\operatorname{Hom}(P,B)
\cong
\prod_\alpha B(E_\alpha).
$$

因此对这类对象，Hom 计算完全化为取值计算。

**命题 8.5.** 若

$$
0\to K\to P\to A\to 0
$$

是短正合列，且 $P$ 投射，则对任意 $B$ 有长正合列开头：

$$
0\to
\operatorname{Hom}(A,B)
\to
\operatorname{Hom}(P,B)
\to
\operatorname{Hom}(K,B)
\to
\operatorname{Ext}^1(A,B)
\to 0
$$

并且后面继续接

$$
\operatorname{Ext}^1(P,B)=0.
$$

**证明.** 对短正合列应用左正合函子 $\operatorname{Hom}(-,B)$ 的右导出长正合列。由于 $P$ 投射，$\operatorname{Ext}^1(P,B)=0$。证毕。

这给出计算 $\operatorname{Ext}^1$ 的标准方法：先找一个投射表示 $P\to A$，再研究核 $K$。

## 8.5 与普通阿贝尔群的差异

若 $M,N$ 是普通阿贝尔群，可把它们看作离散拓扑阿贝尔群并得到凝聚阿贝尔群 $\underline M,\underline N$。自然会问：

$$
\operatorname{Ext}^n_{\mathbf{CondAb}}(\underline M,\underline N)
$$

是否等于普通阿贝尔群范畴中的

$$
\operatorname{Ext}^n_{\mathbf{Ab}}(M,N)?
$$

这个问题不能随意回答。凝聚阿贝尔群范畴更大，投射对象不同，拓扑测试空间参与计算。因此即使在离散对象之间，凝聚范畴中的 Ext 也需要单独分析。

本书后续会在足够准备后讨论 Scholze 讲义中的具体 Ext 计算。当前阶段只建立定义和方法，不提前声明未证明公式。

## 8.6 本章小结

本章建立了三件事：

1. $\mathbf{CondAb}$ 中的正合性可由极不连通测试对象逐点检测。
2. $\mathbb Z[\underline E]$ 对极不连通 $E$ 是投射对象，Hom 计算等于取值。
3. $\mathbf{CondAb}$ 中的 $\operatorname{Ext}$ 可通过投射分解定义。

这些工具足以进入凝聚数学的真正同调代数部分，但还不足以计算深层例子。后续需要研究更具体的投射分解、固体阿贝尔群与解析结构。

## 练习

**练习 8.1.** 补全定理 8.1 中“若截面在覆盖上为零，则截面为零”的 sheaf 论证。

**练习 8.2.** 设 $E$ 极不连通。证明

$$
\operatorname{Ext}^n_{\mathbf{CondAb}}(\mathbb Z[\underline E],B)=0
$$

对 $n>0$ 成立。

**练习 8.3.** 设 $P=\bigoplus_\alpha \mathbb Z[\underline {E_\alpha}]$。证明

$$
\operatorname{Hom}(P,B)\cong \prod_\alpha B(E_\alpha).
$$

**练习 8.4.** 说明为什么不能在没有证明的情况下把 $\mathbf{CondAb}$ 中的 Ext 与普通 $\mathbf{Ab}$ 中的 Ext 直接等同。
