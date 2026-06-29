# 第十一章：Gerstenhaber、BV 与 Deligne 猜想

## 本章目标

本章介绍 Hochschild 理论和二维同伦代数之间的核心桥梁。目标是：

1. 定义 Gerstenhaber 代数和 Gerstenhaber operad。
2. 定义 BV 代数和 BV operad 的基本口径。
3. 说明 $E_2$-operad、Gerstenhaber operad 和 Hochschild cochains 之间的关系。
4. 陈述 Deligne 猜想及其 operadic 形式。
5. 区分同调层 Gerstenhaber 结构和链级 $E_2$-结构。

## 依赖前置知识

需要第六章的 Poisson operad、第十章的 $E_n$-operad，以及 dg 代数和 Hochschild cochains 的基本定义。本章涉及 Hochschild cohomology，默认使用上同调分次；与本书前面同调分次的转换由 $C^n=C_{-n}$ 完成。

## 11.1 分次约定

**约定 11.1.** 本章中若讨论 Hochschild cochains、Gerstenhaber bracket 或 BV operator，使用上同调分次。齐次元素 $x$ 的次数记为 $|x|$，微分次数为 $+1$。若把这些结构转回第九章的同调分次，需要把所有次数取相反数，并相应调整悬挂符号。

**警告 11.2.** Gerstenhaber bracket 在上同调约定中次数为 $-1$。因此它不是普通 graded Lie bracket，而是使 $G[1]$ 成为 graded Lie algebra 的括号。很多符号错误都来自忘记这一位移。

## 11.2 Gerstenhaber 代数

**定义 11.3.** 一个 Gerstenhaber 代数是分次向量空间 $G$，配有：

- 次数 $0$ 的结合乘法
  $$
  \cdot:G\otimes G\to G,
  $$
  使 $G$ 成为 graded commutative algebra，即
  $$
  a\cdot b=(-1)^{|a||b|}b\cdot a;
  $$
- 次数 $-1$ 的双线性括号
  $$
  \{-,-\}:G\otimes G\to G[-1],
  $$
  满足 shifted graded Lie 条件：
  $$
  \{a,b\}=-(-1)^{(|a|-1)(|b|-1)}\{b,a\},
  $$
  以及 shifted Jacobi 恒等式；
- Poisson-Leibniz 规则：
  $$
  \{a,b\cdot c\}
  =
  \{a,b\}\cdot c
  +(-1)^{(|a|-1)|b|}b\cdot\{a,c\}.
  $$

**定义 11.4.** Gerstenhaber operad $\operatorname{Ger}$ 是控制 Gerstenhaber 代数的 dg-operad。等价地，在特征 $0$ 上，
$$
\operatorname{Ger}\cong H_\*(E_2)
$$
按第十章的约定也写作 $\operatorname{Pois}_2$，但需注意上同调记号中 bracket 的次数写作 $-1$。

**命题 11.5.** Gerstenhaber 代数等价于 $\operatorname{Ger}$-代数。

**证明.** $\operatorname{Ger}$ 由一个交换乘法生成元和一个 shifted Lie bracket 生成元，以及交换结合、shifted Jacobi 和 Poisson-Leibniz 关系给出。给出 operad morphism
$$
\operatorname{Ger}\to\operatorname{End}_G
$$
正是给出这些结构映射，并要求它们在 $\operatorname{End}_G$ 中满足这些关系。$\square$

## 11.3 BV 代数

**定义 11.6.** 一个 BV 代数是 graded commutative algebra $(B,\cdot)$，配有次数 $-1$ 的线性算子
$$
\Delta:B\to B[-1],
$$
满足：

1. $\Delta^2=0$；
2. $\Delta$ 是二阶微分算子，即三元表达式
   $$
   \begin{aligned}
   &\Delta(abc)-\Delta(ab)c-(-1)^{|a|}a\Delta(bc)
   -(-1)^{(|a|-1)|b|}b\Delta(ac)\\
   &\quad+\Delta(a)bc
   +(-1)^{|a|}a\Delta(b)c
   +(-1)^{|a|+|b|}ab\Delta(c)
   \end{aligned}
   $$
   为 $0$。

由 $\Delta$ 定义括号
$$
\{a,b\}
=
(-1)^{|a|}
\big(\Delta(a b)-\Delta(a)b-(-1)^{|a|}a\Delta(b)\big).
$$

**命题 11.7.** BV 代数的导出括号使 $B$ 成为 Gerstenhaber 代数。

**证明.** 二阶条件说明导出括号对第二个变量满足 Poisson-Leibniz 规则；graded commutativity 给出对第一个变量的相应规则。$\Delta^2=0$ 展开后给出 shifted Jacobi 恒等式。反对称性由乘法的 graded commutativity 和导出括号公式直接计算得到。$\square$

**定义 11.8.** BV operad $\operatorname{BV}$ 是控制 BV 代数的 dg-operad。它可由交换乘法和一元算子 $\Delta$ 生成，并加入 $\Delta^2=0$ 与二阶微分算子关系。其代数正是定义 11.6 的 BV 代数。

**说明 11.9.** BV 结构比 Gerstenhaber 结构强：BV 算子 $\Delta$ 决定 Gerstenhaber bracket，但一般 Gerstenhaber algebra 不一定来自某个 BV operator。

## 11.4 Hochschild cohomology 上的 Gerstenhaber 结构

**定义 11.10.** 设 $A$ 是结合 $k$-代数。Hochschild cochains 定义为
$$
C^n(A,A)=\operatorname{Hom}_k(A^{\otimes n},A),
$$
配有 Hochschild differential $\delta:C^n(A,A)\to C^{n+1}(A,A)$。其 cohomology 记为
$$
HH^\*(A,A).
$$

**外部输入定理 11.11.** Gerstenhaber 构造给 $HH^\*(A,A)$ 一个自然 Gerstenhaber algebra 结构。乘法由 cup product 诱导，bracket 由 cochain insertion commutator 诱导。

该定理的链级公式将在第十二章给出；本章只记录其 operadic 角色。

## 11.5 Deligne 猜想

**外部输入定理 11.12.** Deligne 猜想的 operadic 形式断言：对任意结合代数 $A$，Hochschild cochain complex
$$
C^\*(A,A)
$$
自然带有链级 $E_2$-algebra 结构，并且该结构在 cohomology 上诱导的 Gerstenhaber algebra 正是定理 11.11 的经典结构。

**解释 11.13.** 定理 11.12 比“$HH^\*(A,A)$ 是 Gerstenhaber algebra”强得多。它说的是 cochain complex 上存在一个链级 operad 作用，该 operad 与 $E_2$ 弱等价。Gerstenhaber bracket 只是取 cohomology 后看到的阴影。

**说明 11.14.** Deligne 猜想有多种证明和加强形式，包括使用 little disks、brace operad、McClure-Smith 的 cosimplicial machinery、Tamarkin 的 formality 方法、Kontsevich-Soibelman 的 operadic 方法等。本书采用的路线是：

1. 第十二章构造 Hochschild cochains 上的 brace algebra。
2. Brace operad 与链级 $E_2$-operad 弱等价作为外部输入。
3. 因此得到 Hochschild cochains 上的 $E_2$-algebra 结构。

## 11.6 Framed $E_2$ 与 BV

**外部输入定理 11.15.** Framed little disks operad 的同调 operad 是 BV operad：
$$
H_\*(E_2^{\mathrm{fr}})\cong\operatorname{BV}.
$$
这里 framed 结构加入了每个小圆盘的旋转自由度。该旋转在同调上产生 BV operator。

**说明 11.16.** 在某些几何和拓扑语境中，Hochschild 或 string topology 型对象不仅有 $E_2$ 结构，还带有 framed $E_2$ 结构，从而在同调上产生 BV algebra。是否存在 BV 结构取决于额外的旋转、Calabi-Yau、cyclic 或 trace 型数据，不能从普通结合代数的 Hochschild cochains 自动推出。

## 本章小结

Gerstenhaber algebra 是带有交换乘法和次数 $-1$ Lie bracket 的 shifted Poisson algebra。BV algebra 通过一个平方为零的二阶算子产生 Gerstenhaber bracket。Deligne 猜想说明 Hochschild cochains 不只在 cohomology 上有 Gerstenhaber 结构，而是在链级有 $E_2$-algebra 结构。Framed $E_2$ 的同调给出 BV operad，但 BV 结构需要额外数据。

## 练习

**练习 11.1.** 验证定义 11.3 中 Poisson-Leibniz 规则的符号来自 bracket 次数 $-1$。

**练习 11.2.** 对 BV bracket 公式，证明若 $\Delta$ 是一阶导子，则导出括号为零。

**练习 11.3.** 设 $B$ 是 BV 代数，展开 $\Delta^2(ab)$ 并指出 Jacobi 恒等式的符号来源。

**练习 11.4.** 解释为什么 $HH^\*(A,A)$ 上的 Gerstenhaber 结构不等价于 $C^\*(A,A)$ 上的 $E_2$ 结构。

**练习 11.5.** 查阅 framed little disks 的定义，说明旋转自由度如何给出一元同调类。

