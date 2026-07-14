# 第十四章：解析环

solid 理论固定使用整值测度对象 $\mathbb Z^\square[S]$；一旦底环改变，或希望处理
实数函数空间，哪些无限 Dirac 组合应当收敛便不再由这套对象决定。普通 Radon 测度看似
自然，却不能自动满足所需的派生 Hom 条件。问题因此不是给凝聚环附加一个拓扑，而是
为每个极不连通测试空间指定允许的测度模，并要求这种指定与解析模的延拓性质相容。

第九、十章的凝聚环与模提供底层代数，第十二、十三章的 solid 测度给出首个模型，
第十一章的派生 Hom 则负责表达复形级条件。我们依 Scholze 的定义引入测度理论、解析
模与解析环，验证 solid 特例和有限生成环例子，并把实数方向的失败精确留给第二卷的
$p$-liquid 修正，而不把 Banach 完备化误认成解析化。

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
R\underline{\operatorname{Hom}}_A(\mathcal M[S],C)
\longrightarrow
R\underline{\operatorname{Hom}}_A(A[\underline S],C)
$$

是 $D(\mathbf{CondAb})$ 中的同构。这里使用内部派生 Hom；若再取终对象上的派生截面，
才得到普通派生映射复形。只写不带下划线的全局 $R\operatorname{Hom}$ 会丢失解析环公理
要求的凝聚参数，因而不是本定义的等价替代。

**注 14.6.** 这个定义是 solid 定义的相对版本：把自由对象 $A[\underline S]$ 替换成允许的测度对象 $\mathcal M[S]$，并要求对由这些测度对象生成的复形有正确的派生 Hom 行为。

本章采用的是适合第一卷的压缩表述。正式处理时还要同时跟踪集合论大小、动画环或导出环版本、以及 $\mathcal M$ 与 $A$ 的乘法相容性；这些技术条件不改变本章使用的核心判别式。

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

1. 存在 analytic 派生范畴 \(D(A,\mathcal M)\subset D(A)\)。
2. 包含函子有左伴随 \(L_{(A,\mathcal M)}:D(A)\to D(A,\mathcal M)\)，后文称为解析化。
3. 若 $A$ 交换，则 \(D(A,\mathcal M)\) 带有由 analytic kernel 张量理想性下降得到的对称幺半张量积。

**证明说明.** 第二卷输入定理 D.4 和附录 X 使用本结构定理的精确形式。本章只作为入口，不证明 analytic ring 公理推出 Bousfield localization 和张量相容。

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

第一卷不展开 liquid vector spaces，只记录：实分析方向不是 solid 的直接形式推广。第二卷会把 liquid/analytic 结构作为主题之一。

## 14.7 允许的测度决定解析模

测度理论把每个极不连通 $S$ 送到 $\mathcal M[S]$，Dirac 映射则把普通自由模与允许
测度相连；解析条件要求相应派生 Hom 判别在生成复形上成立。取
$\mathcal M[S]=\mathbb Z^\square[S]$ 恢复 solid，而有限生成离散环给出换底后的基本
模型。实数 Radon 测度的失败表明测度选择是实质结构而非装饰。第二卷将把这一条件写成
正式 localization，并以 $\mathcal M_{<p}$ 建立 liquid 方向；下一章先说明解析环如何
沿 rational localization 进入几何。

## 练习

**练习 14.1.** 检查 $\mathbb Z^\square$-测度理论把有限不交并变为乘积。

**练习 14.2.** 比较固体阿贝尔群定义与解析模定义中的 Hom 判别。

**练习 14.3.** 设 $A$ 为有限生成 $\mathbb Z$-代数。解释 $A^\square[S]=\varprojlim_iA[S_i]$ 为什么可视为 $A$-值测度。

**练习 14.4.** 说明为什么实数上的 Radon 测度理论不应被未经证明地当作 analytic ring。
