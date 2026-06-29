# 第三章：解析环的正式条件

## 本章目标

第一卷第十四章给出了解析环的入口。本章把该定义改写成第二卷可用的形式：先定义测度理论，再把 analytic condition 表述为一族派生 Hom 判别。

## 依赖

需要第一卷第十四章和第二卷第一章。

## 3.1 预解析环

令 $A$ 是凝聚交换环。

**定义 3.1.** $A$ 上的预解析结构由以下数据组成：

1. 对每个极不连通紧 Hausdorff 空间 $S$，给出凝聚 $A$-模
   $$
   \mathcal M[S].
   $$
2. 对每个连续映射 $f:S\to T$，给出 pushforward
   $$
   f_*:\mathcal M[S]\to\mathcal M[T],
   $$
   并满足函子性。
3. 对每个 $S$，给出 Dirac 映射
   $$
   A[\underline S]\to\mathcal M[S].
   $$
4. 对有限不交并有自然同构
   $$
   \mathcal M[S\sqcup T]\simeq\mathcal M[S]\times\mathcal M[T],
   \qquad
   \mathcal M[\varnothing]\simeq0.
   $$

组合 $(A,\mathcal M)$ 称为预解析环。

**注 3.2.** $\mathcal M[S]$ 应理解为 $S$ 上允许的 $A$-值测度对象。Dirac 映射把点质量测度嵌入允许测度。

## 3.2 解析局部对象

对每个极不连通 $S$，记

$$
K_S^{\mathcal M}
=
\operatorname{Cone}\left(A[\underline S]\to\mathcal M[S]\right)
$$

为测度扩张的 cone。

**定义 3.3.** 设 $C\in D(A)$。若对任意极不连通 $S$，

$$
R\operatorname{Hom}_A(\mathcal M[S],C)
\longrightarrow
R\operatorname{Hom}_A(A[\underline S],C)
$$

是同构，则称 $C$ 是 $(A,\mathcal M)$-解析复形。

等价地，

$$
R\operatorname{Hom}_A(K_S^{\mathcal M},C)\simeq0
$$

对所有 $S$ 成立。

**证明.** 对三角

$$
A[\underline S]\to\mathcal M[S]\to K_S^{\mathcal M}\to
$$

应用 $R\operatorname{Hom}_A(-,C)$，得到长三角。中间箭头为同构当且仅当第一项为零。证毕。

## 3.3 解析环

**输入定义 3.4（Scholze）.** 预解析环 $(A,\mathcal M)$ 称为解析环，如果上述局部对象形成反射稳定子范畴

$$
D(A,\mathcal M)\subset D(A),
$$

并且满足以下结构性质：

1. 包含函子有左伴随
   $$
   L_{(A,\mathcal M)}:D(A)\to D(A,\mathcal M).
   $$
2. $D(A,\mathcal M)$ 在极限、余极限和扩张下稳定。
3. 若 $A$ 交换，则 $D(A,\mathcal M)$ 继承与 localization 相容的闭对称幺半结构。
4. 对所有极不连通 $S$，$\mathcal M[S]$ 在解析化后扮演自由解析 $A$-模的角色。

本卷采用这个操作性表述。Scholze 讲义中的正式定义还跟踪集合论大小、动画环版本、以及 $\mathcal M$ 与乘法结构的相容性。

## 3.4 解析模

**定义 3.5.** 一个 ordinary 凝聚 $A$-模 $N$ 称为解析 $A$-模，如果 $N[0]$ 是定义 3.3 意义下的解析复形。

等价地，对所有极不连通 $S$，

$$
\operatorname{Hom}_A(\mathcal M[S],N)
\cong
\operatorname{Hom}_A(A[\underline S],N)
\cong
N(S).
$$

这些对象构成范畴

$$
(A,\mathcal M)\text{-}\mathbf{Mod}.
$$

## 3.5 solid 作为解析环

令 $A=\mathbb Z$，并取

$$
\mathcal M[S]=\mathbb Z^\square[S].
$$

则 $(\mathbb Z,\mathbb Z^\square)$ 是解析环的基本例子。其解析模正是 solid 阿贝尔群。

**命题 3.6.** 若 $C\in D(\mathbf{CondAb})$，则 $C$ 对解析环 $(\mathbb Z,\mathbb Z^\square)$ 解析，当且仅当 $C$ 是第二卷第一章中的 solid 复形。

**证明.** 把定义 3.3 中的 $A[\underline S]$ 和 $\mathcal M[S]$ 分别代入

$$
\mathbb Z[\underline S],
\qquad
\mathbb Z^\square[S].
$$

得到的 Hom 判别正是定义 1.1。证毕。

## 3.6 例子

**例 3.7.** 对有限离散 $S$，

$$
\mathbb Z^\square[S]\cong\mathbb Z[\underline S],
$$

因此 $K_S^{\mathbb Z^\square}\simeq0$。solid 条件只在无限 profinite 测试对象上有内容。

**例 3.8.** 对 $p$-进整数，可取

$$
\mathcal M[S]=\mathbb Z_p^\square[S].
$$

这给出 $p$-进完备方向的解析结构。它不是普通 $\mathbb Z_p$-模范畴，而是凝聚/solid 意义下的完备模范畴。

## 3.7 本章小结

解析环把“允许的测度对象”作为结构的一部分。解析模不是任意 $A$-模，而是对这些测度对象局部的对象。

附录 I 给出从预解析结构升级到 analytic ring 时需要检查的条件，并列出不能从普通完备化或任意测度赋值推出的失败模式。

## 练习

**练习 3.1.** 证明定义 3.3 与 $R\operatorname{Hom}_A(K_S^{\mathcal M},C)=0$ 的等价。

**练习 3.2.** 对有限离散 $S$，计算 $\mathcal M[S]$ 在 solid 例子中的形式。

**练习 3.3.** 解释为什么 Dirac 映射 $A[\underline S]\to\mathcal M[S]$ 是 analytic condition 的核心输入。

**练习 3.4.** 比较 solid 复形和一般解析复形的定义。
