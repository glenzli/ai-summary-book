# 第十四章：解析环

## 本章目标

solid 结构适合非阿基米德型完备化，但不足以覆盖实分析和更一般的解析几何。解析环（analytic ring）提供了更灵活的“测度理论”。本章给出 Scholze 讲义中的基本定义和例子。

## 依赖前置知识

需要凝聚环、凝聚模、solid 测度对象和派生 Hom。

## 14.1 测度理论

设 $A$ 是凝聚环。

**定义 14.1.** $A$ 上的测度理论（theory of measures）由以下数据组成：

1. 一个函子
   $$
   \mathcal M:\mathbf{ED}\to A\text{-}\mathbf{Mod},
   \qquad S\mapsto \mathcal M[S],
   $$
   其中 $\mathbf{ED}$ 表示极不连通紧 Hausdorff 空间范畴，$A\text{-}\mathbf{Mod}$ 表示凝聚阿贝尔群中的 $A$-模范畴。
2. 该函子把有限不交并变为乘积：
   $$
   \mathcal M[S\sqcup T]\cong \mathcal M[S]\times\mathcal M[T].
   $$
3. 自然的 Dirac 映射
   $$
   S\to \mathcal M[S].
   $$

直观上，$\mathcal M[S]$ 是 $S$ 上取值于 $A$ 的允许测度对象。

## 14.2 例子

**例 14.2.** $\mathbb Z^\square$-测度由

$$
S\mapsto \mathbb Z^\square[S]
$$

给出。

**例 14.3.** 对 $p$-进整数环 $\mathbb Z_p$，可定义

$$
\mathbb Z_{p}^{\square}[S]
=
\varprojlim_i \mathbb Z_p[S_i],
\qquad S=\varprojlim_iS_i.
$$

这是 $S$ 上的 $\mathbb Z_p$-值测度。

**例 14.4.** 若 $A$ 是有限生成 $\mathbb Z$-代数，可定义

$$
A^\square[S]
=
\varprojlim_i A[S_i]
$$

对极不连通 $S=\varprojlim_iS_i$。底层阿贝尔群可理解为 $A$-值测度。

## 14.3 解析环定义

**定义 14.5.** 解析环是凝聚环 $A$ 与测度理论 $\mathcal M$ 的组合，记作

$$
(A,\mathcal M),
$$

满足如下条件：对任意复形

$$
C:\cdots\to C_2\to C_1\to C_0\to 0
$$

若每个 $C_i$ 都是形如 $\mathcal M[T]$ 的对象的直和，其中 $T\in\mathbf{ED}$，则对任意 $S\in\mathbf{ED}$，自然映射

$$
R\operatorname{Hom}_A(\mathcal M[S],C)
\longrightarrow
R\operatorname{Hom}_A(A[\underline S],C)
$$

是同构。

**注 14.6.** 这个定义是 solid 定义的相对版本：把自由对象 $A[\underline S]$ 替换成允许的测度对象 $\mathcal M[S]$，并要求对由这些测度对象生成的复形有正确的派生 Hom 行为。

本章采用的是适合第一版教材的压缩表述。正式处理时还要同时跟踪集合论大小、动画环或导出环版本、以及 $\mathcal M$ 与 $A$ 的乘法相容性；这些技术条件不改变本章使用的核心判别式。

## 14.4 解析模

**定义 14.7.** 设 $(A,\mathcal M)$ 是解析环。一个 $A$-模 $N$ 称为 $(A,\mathcal M)$-模，如果对所有极不连通 $S$，自然映射

$$
\operatorname{Hom}_A(\mathcal M[S],N)
\longrightarrow
N(S)
$$

是同构。

这些对象构成 $A$-模范畴的全子范畴，记为

$$
(A,\mathcal M)\text{-}\mathbf{Mod}.
$$

**定理 14.8（Scholze）.** 若 $(A,\mathcal M)$ 是解析环，则：

1. $(A,\mathcal M)\text{-}\mathbf{Mod}$ 是良好的阿贝尔/导出范畴环境。
2. 包含到 $A$-模范畴的导出范畴有左伴随，可视为解析化。
3. 若 $A$ 交换，则导出范畴上有与解析化相容的对称幺半张量积。

**证明说明.** 这是 Scholze 讲义第七讲的基本结构定理。完整证明依赖 Lemma 5.10 和 analytic condition 的 Bousfield localization 解释。

## 14.5 与 solid 的关系

Scholze 讲义中指出：

$$
\mathbf{Solid}
=
\mathbb Z^\square\text{-}\mathbf{Mod}.
$$

也就是说，固体阿贝尔群是解析环 $\mathbb Z^\square$ 上的模。

这解释了为什么 analytic rings 是 solid 的推广：solid 处理的是 $\mathbb Z$ 上的非阿基米德型测度，而 analytic rings 允许换底环和换测度理论。

## 14.6 实数方向的警告

经典 Radon 测度给出的实数测度理论并不直接满足 analytic ring 条件。Scholze 讲义指出，处理实数需要更细的 $p$-凸或 liquid 型结构；特别是对 $0<p\le1$ 的某些测度理论，取极限

$$
\mathcal M_{<p}[S]=\varinjlim_{q<p}\mathcal M_q[S]
$$

可得到解析环结构。

本书不在第一版展开 liquid vector spaces，只记录：实分析方向不是 solid 的直接形式推广。

## 14.7 本章小结

本章给出了解析环的定义框架：

1. 测度理论指定极不连通测试对象上的允许测度模块。
2. 解析环要求测度对象对特定生成复形满足派生 Hom 判别。
3. solid 是解析环框架中的基本例子。
4. 实数与泛函分析需要更精细的 analytic/liquid 结构。

## 练习

**练习 14.1.** 检查 $\mathbb Z^\square$-测度理论把有限不交并变为乘积。

**练习 14.2.** 比较固体阿贝尔群定义与解析模定义中的 Hom 判别。

**练习 14.3.** 设 $A$ 为有限生成 $\mathbb Z$-代数。解释 $A^\square[S]=\varprojlim_iA[S_i]$ 为什么可视为 $A$-值测度。

**练习 14.4.** 说明为什么实数上的 Radon 测度理论不应被未经证明地当作 analytic ring。
