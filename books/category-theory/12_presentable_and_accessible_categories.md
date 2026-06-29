# 第十二章：可表现范畴与可达范畴

## 本章目标

本章引入正则基数、$\kappa$-滤过范畴、$\kappa$-紧对象、可达范畴和局部可表现范畴。这些概念是现代范畴论、模型范畴和 $\infty$-范畴中控制大小的核心工具。

## 依赖前置知识

需要小性约定、余极限、函子保持余极限和生成族。

## 12.1 正则基数与滤过性

**定义 12.1.** 基数 $\kappa$ 称为正则，若任意少于 $\kappa$ 个小于 $\kappa$ 的基数之和仍小于 $\kappa$。

**定义 12.2.** 小范畴 $\mathcal J$ 称为 $\kappa$-滤过，若对任意小于 $\kappa$ 的小范畴 $\mathcal I$ 和函子 $\mathcal I\to\mathcal J$，存在一个 cocone。$\omega$-滤过通常称为滤过。

**例子 12.3.** 有向偏序集作为范畴给出滤过范畴。有限子集偏序集 $\operatorname{Fin}(S)$ 按包含排序是滤过的。

## 12.2 紧对象

**定义 12.4.** 设 $\mathcal C$ 有 $\kappa$-滤过余极限。对象 $A\in\mathcal C$ 称为 $\kappa$-紧或 $\kappa$-presentable，若 Hom 函子

$$
\mathcal C(A,-):\mathcal C\to\mathbf{Set}
$$

保持 $\kappa$-滤过余极限。

**例子 12.5.** 在 $\mathbf{Set}$ 中，有限集合是 $\omega$-紧对象。更一般地，基数小于 $\kappa$ 的集合是 $\kappa$-紧对象。

**证明.** 设 $|A|<\kappa$，且 $X:\mathcal J\to\mathbf{Set}$ 是 $\kappa$-滤过图形。需证明自然映射

$$
\operatorname{colim}_{j\in\mathcal J}\mathbf{Set}(A,X_j)
\to
\mathbf{Set}\left(A,\operatorname{colim}_{j\in\mathcal J}X_j\right)
$$

是双射。

先证满射。给定函数

$$
f:A\to\operatorname{colim}_{j}X_j,
$$

对每个 $a\in A$，选择指标 $j_a$ 和元素 $x_a\in X_{j_a}$ 表示 $f(a)$。由于 $A$ 的元素数小于 $\kappa$，而 $\mathcal J$ 是 $\kappa$-滤过，存在对象 $j$ 和态射 $j_a\to j$ 同时支配所有 $j_a$。令 $y_a\in X_j$ 为 $x_a$ 的像，得到函数 $g:A\to X_j$。它在余极限中诱导 $f$，故满射。

再证单射。设 $g:A\to X_j$ 与 $h:A\to X_k$ 在 $\operatorname{colim}_jX_j$ 中诱导同一函数。对每个 $a\in A$，元素 $g(a)$ 与 $h(a)$ 在集合余极限中相等，故存在从 $j,k$ 出发的一段有限 zigzag 使二者在某个后继指标中相等。所有这些有限数据的总数仍小于 $\kappa$。由 $\kappa$-滤过性，可找到一个对象 $\ell$ 同时支配这些指标并使所有等式在 $X_\ell$ 中成立。于是 $g$ 与 $h$ 在 $\mathbf{Set}(A,X_\ell)$ 中有相同像，因此它们在左侧滤过余极限中相等。故映射单射。$\square$

## 12.3 可达范畴

**定义 12.6.** 范畴 $\mathcal C$ 称为 $\kappa$-可达，若：

1. $\mathcal C$ 有 $\kappa$-滤过余极限；
2. 存在一个小的 $\kappa$-紧对象全子范畴 $\mathcal C_\kappa$；
3. 每个对象都是 $\mathcal C_\kappa$ 中对象的 $\kappa$-滤过余极限。

若对某个正则基数 $\kappa$ 成立，则称 $\mathcal C$ 可达。

**定义 12.7.** 范畴 $\mathcal C$ 称为局部 $\kappa$-可表现，若它是 $\kappa$-可达且有所有小余极限。称局部可表现，若对某个 $\kappa$ 局部 $\kappa$-可表现。

**例子 12.8.** $\mathbf{Set}$、$\mathbf{Grp}$、$\mathbf{Ab}$、任意小范畴上的预层范畴都是局部可表现范畴。

## 12.4 小生成与 Ind 完备化

**定义 12.9.** 对小范畴 $\mathcal A$，其 $\operatorname{Ind}_\kappa$-完备化是由 $\mathcal A$ 的 $\kappa$-滤过形式余极限生成的范畴，典型实现为预层范畴中可表预层的 $\kappa$-滤过余极限闭包。

**外部输入定理 12.10.** 范畴 $\mathcal C$ 是 $\kappa$-可达，当且仅当它等价于某个小范畴 $\mathcal A$ 的 $\operatorname{Ind}_\kappa(\mathcal A)$ 的一个合适反射子范畴。局部可表现范畴可刻画为小范畴的自由余完备化经可达反射得到的范畴。

来源见 Adámek-Rosický、Borceux 和 Lurie 的 presentable category 章节。

## 12.5 可达函子

**定义 12.11.** 可达范畴之间的函子称为可达函子，若它保持某个正则基数 $\kappa$ 的 $\kappa$-滤过余极限。

**命题 12.12.** 左伴随 $F:\mathcal C\to\mathcal D$ 若定义在局部可表现范畴之间，则保持所有小余极限，因而保持所有 $\kappa$-滤过余极限，是可达函子的候选；但“可达”还要求选择与目标可达结构相容的基数。

**证明.** 左伴随保持余极限由定理 4.7 得到。关于基数相容性的精确陈述属于可达范畴理论的标准定理，本章只记录方向。$\square$

## 12.6 预层范畴与伴随函子定理

**命题 12.A.** 若 $\mathcal C$ 是小范畴，则预层范畴

$$
\widehat{\mathcal C}=\operatorname{Fun}(\mathcal C^{op},\mathbf{Set})
$$

是局部可表现范畴。更具体地，可表预层构成小生成族，并且任意预层是可表预层的典范小余极限。

**证明.** 余极限在预层范畴中逐点计算，因此 $\widehat{\mathcal C}$ 有所有小余极限。由第五章预层密度定理，对任意预层 $P$ 有典范同构

$$
P\cong\operatorname*{colim}_{(C,x)\in\int_{\mathcal C}P}yC.
$$

其中 $\int_{\mathcal C}P$ 是 $P$ 的元素范畴。由于 $\mathcal C$ 小且 $P(C)$ 为集合，该元素范畴小。可表预层 $yC$ 对逐点滤过余极限紧：由 Yoneda，

$$
\widehat{\mathcal C}(yC,\operatorname{colim}_jP_j)
\cong
(\operatorname{colim}_jP_j)(C)
\cong
\operatorname{colim}_jP_j(C)
\cong
\operatorname{colim}_j\widehat{\mathcal C}(yC,P_j).
$$

因此可表预层是 $\omega$-紧对象，并生成整个预层范畴。故 $\widehat{\mathcal C}$ 局部可表现。$\square$

**定义 12.B.** 局部可表现范畴 $\mathcal C$ 的一个小全子范畴 $\mathcal G$ 称为强生成子，若态射 $f:X\to Y$ 是同构当且仅当对所有 $G\in\mathcal G$，

$$
\mathcal C(G,f):\mathcal C(G,X)\to\mathcal C(G,Y)
$$

是双射。

**命题 12.C.** 若 $\mathcal C$ 局部 $\kappa$-可表现，则其 $\kappa$-紧对象的小骨架构成强生成子。

**证明.** 设 $\mathcal G$ 为 $\kappa$-紧对象小骨架。若 $f:X\to Y$ 在所有 $G\in\mathcal G$ 上诱导双射，则 $X$ 与 $Y$ 作为 $\mathcal G$ 中对象的 $\kappa$-滤过余极限时，所有来自紧生成对象的元素和关系都被 $f$ 完全检测。更具体地，任意 $G\to Y$ 由满性提升到 $G\to X$，这说明 $f$ 在由 $\mathcal G$ 生成的逗号图形上本质满；任意两个 $G\to X$ 若复合到 $Y$ 相同，则由单射性在后续滤过阶段相同。由于 $X,Y$ 均由这些紧对象滤过生成，$f$ 诱导的余极限比较为同构。反向显然。$\square$

**外部输入定理 12.D（局部可表现范畴的伴随函子定理）.** 设 $\mathcal C,\mathcal D$ 为局部可表现范畴。函子

$$
F:\mathcal C\to\mathcal D
$$

有右伴随，当且仅当 $F$ 保持所有小余极限并且是可达函子。对偶地，函子有左伴随的充分必要条件可用保持极限和可达性表述。

该定理是普通伴随函子定理在可表现范畴语境中的强形式；它解释了为什么 presentable $\infty$-category 中的左伴随通常被定义为保持余极限的可达函子。

## 12.7 本章小结

可表现范畴理论提供控制“大范畴”的方法：对象由小对象经滤过余极限生成，函子由其在小对象上的行为控制。该语言在模型范畴、Grothendieck 范畴和 presentable $\infty$-categories 中反复出现。

## 练习

**练习 12.1.** 证明有限集合是 $\mathbf{Set}$ 中的 $\omega$-紧对象。

**练习 12.2.** 说明为什么无限集合通常不是 $\omega$-紧对象。

**练习 12.3.** 证明预层范畴 $\widehat{\mathcal C}$ 由可表预层经小余极限生成。

**练习 12.4.** 查阅 locally finitely presentable category 的定义，并给出 $\mathbf{Grp}$ 的有限表现对象例子。

**练习 12.5.** 比较“生成族”和“紧生成”两个概念的差别。

**练习 12.6.** 证明可表预层 $yC$ 在预层范畴中保持滤过余极限的 Hom 函子。

**练习 12.7.** 用预层密度定理说明为什么可表预层构成生成族。

**练习 12.8.** 解释强生成子与“检测同构”的关系。

**练习 12.9.** 使用外部输入定理 12.D，说明为什么保持小余极限但不可达的函子不一定有右伴随。
