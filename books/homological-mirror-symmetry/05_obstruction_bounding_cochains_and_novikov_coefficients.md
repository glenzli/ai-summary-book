# 第五章：obstruction、bounding cochains、Novikov 系数与 curved $A_\infty$ 结构

## 本章目标

本章解释为什么 exact 情况之外的 Fukaya theory 需要 curvature、Novikov 系数和 bounding cochains。核心目标是给出 filtered curved $A_\infty$ 语言，并证明 Maurer-Cartan 方程如何把弯曲结构变回可用的 Floer differential。

## 依赖前置知识

需要第一章的 $A_\infty$ 范畴、第三章的 Floer cochains、第四章的 holomorphic polygon 计数。一般虚基本链技术作为外部输入。

## 5.1 Novikov 系数

**定义 5.1.** Novikov field $\Lambda$ 是形式和
$$
\sum_{i=0}^\infty a_iT^{\lambda_i},\qquad a_i\in k,\quad \lambda_i\in\mathbb R,\quad
\lambda_i\to+\infty,
$$
构成的域。其 valuation 定义为
$$
v\left(\sum_i a_iT^{\lambda_i}\right)=\inf\{\lambda_i\mid a_i\ne0\}.
$$
Novikov ring $\Lambda_{\ge0}$ 由 valuation 非负的元素构成，$\Lambda_{>0}$ 由 valuation 正的元素构成。

**解释 5.2.** $T^\lambda$ 记录 holomorphic curve 的 symplectic area。条件 $\lambda_i\to+\infty$ 保证按能量过滤的无穷和在 Novikov 拓扑下收敛。

**命题 5.3.** 若对每个能量上界 $E$，相关 holomorphic curve 模空间中只有有限多个面积 $\le E$ 的贡献，则以 $T^{\omega(u)}$ 加权的 polygon 计数在 $\Lambda$ 中收敛。

**证明.** 固定 $E$ 时，只有有限多个面积不超过 $E$ 的项，所以任意 valuation 截断只含有限和。随着面积趋向无穷，指数 $\omega(u)$ 趋向无穷，因此形式和满足 Novikov field 的定义。证毕。

## 5.2 curved $A_\infty$ 结构

**定义 5.4.** 一个 curved $A_\infty$ algebra 是分次 $\Lambda$-向量空间 $A$ 和次数 $2-d$ 的运算
$$
\mu^d:A^{\otimes d}\to A[2-d],\qquad d\ge0,
$$
满足包含 $d=0$ 项的 $A_\infty$ 方程。元素
$$
\mu^0\in A^2
$$
称为 curvature。

**解释 5.5.** 在 Fukaya theory 中，$\mu^0$ 来自边界落在同一 Lagrangian 上、带一个输出的 holomorphic disks。若 $\mu^0\ne0$，则 $\mu^1$ 一般不再平方为零。

**命题 5.6.** 对 curved $A_\infty$ algebra，低阶 $A_\infty$ 方程给出
$$
\mu^1(\mu^1(x))+\mu^2(\mu^0,x)\pm\mu^2(x,\mu^0)=0
$$
以及含更高项的相应公式。因此若 $\mu^0=0$，则 $\mu^1{}^2=0$。

**证明.** 把 curved $A_\infty$ 方程写在一个输入 $x$ 上。所有分解项包括先对空输入应用 $\mu^0$ 再与 $x$ 复合、先对 $x$ 应用 $\mu^1$ 再应用 $\mu^1$，以及符号由 suspension convention 决定的右侧 curvature 项。若 $\mu^0=0$，只剩 $\mu^1(\mu^1(x))=0$。证毕。

## 5.3 Maurer-Cartan 元与变形微分

**定义 5.7.** 设 $A$ 是 filtered curved $A_\infty$ algebra。一个 bounding cochain 是次数 $1$ 的元素
$$
b\in A^1\widehat\otimes\Lambda_{>0}
$$
满足 Maurer-Cartan 方程
$$
\sum_{d\ge0}\mu^d(b,\ldots,b)=0.
$$
更一般地，若右边等于 $W(b)e$，其中 $e$ 是单位，则称 $b$ 是 weak bounding cochain，$W(b)$ 称为 disk potential 或 obstruction value。这里采用 Fukaya 范畴常用的 $\mathbb Z/2$ 分次或等价地把 Novikov/势参数赋予补偿次数的约定；若坚持纯 $\mathbb Z$ 分次，则必须同时声明 $W(b)$ 的次数使 $W(b)e$ 与 $\mu_b^0$ 同次。

**定义 5.8.** 给定 bounding cochain $b$，定义变形后的运算
$$
\mu_b^d(x_d,\ldots,x_1)
=\sum_{r_0,\ldots,r_d\ge0}
\mu^{d+r_0+\cdots+r_d}(b^{r_d},x_d,b^{r_{d-1}},\ldots,x_1,b^{r_0}).
$$
Novikov 正 valuation 条件保证该和收敛。

**命题 5.9.** 若 $b$ 满足 Maurer-Cartan 方程，则 $\{\mu_b^d\}_{d\ge1}$ 构成非弯曲 $A_\infty$ 结构。特别地，
$$
(\mu_b^1)^2=0.
$$

**证明.** 把 $b$ 插入 curved $A_\infty$ 方程的所有空隙并对所有插入数求和。出现的所有项正是变形后运算的 $A_\infty$ 方程；其中没有外部输入的项由 Maurer-Cartan 方程消去，即变形 curvature
$$
\mu_b^0=\sum_{d\ge0}\mu^d(b,\ldots,b)
$$
为零。取一个输入的低阶方程得到 $(\mu_b^1)^2=0$。证毕。

**定义 5.10.** 两个对象 $(L_0,b_0)$、$(L_1,b_1)$ 的 deformed Floer complex 定义为
$$
CF^\ast((L_0,b_0),(L_1,b_1)),\qquad d=\mu^1_{b_0,b_1},
$$
其中左右边界上的 $b_0,b_1$ 插入由 polygon 边界标号决定。

## 5.4 Fukaya category 的对象扩张

**定义 5.11.** 在 filtered 口径下，Fukaya category 的对象不再只是 brane $\mathbb L$，而是二元组
$$
(\mathbb L,b)
$$
其中 $b$ 是 bounding cochain 或 weak bounding cochain。若采用 weak bounding cochains，则通常只允许 obstruction value 相同的对象之间形成同一个 fiber category。

**例 5.12.** 在 toric Fano 情况中，Lagrangian torus fibers 的 disk potential $W$ 是 Laurent polynomial。critical points of $W$ 给出 unobstructed 或 weakly unobstructed branes；相应 Floer cohomology 与 Jacobian ring 的分量相关。

**外部输入定理 5.13（FOOO filtered Fukaya theory）.** 对一般 compact Lagrangians，在适当相对 spin、Novikov、Kuranishi/virtual perturbation 和能量过滤假设下，可以构造 filtered curved $A_\infty$ structures、bounding cochain deformation theory 和相应 Floer cohomology。  
来源：Fukaya-Oh-Ohta-Ono 的 Lagrangian intersection Floer theory。

## 5.5 HMS 中的用途

在 HMS 中，obstruction theory 的作用至少有三类。

1. 对 compact 非 exact Lagrangians，必须用 $b$ 才能得到可用对象。
2. Toric mirror 中，disk potential $W$ 直接成为 Landau-Ginzburg B-side 的 potential。
3. Fukaya category 常按 potential value 分解，对应 matrix factorization category 或 singular fiber 上的 B-side 分类。

**命题 5.14.** 若两个 weak bounding cochains $b_0,b_1$ 的 obstruction values 不同，则它们不属于同一个非弯曲 Fukaya fiber category。在把所有 potential values 放入同一 curved 或 matrix-factorization 型模型时，对应 morphism object 在标准可逆曲率差假设下为 contractible。

**证明草图.** 低阶 curved 方程给出
$$
(\mu^1_{b_0,b_1})^2(x)=
(W(b_1)-W(b_0))\cdot x
$$
到符号约定下的单位作用。若只固定一个 fiber value $\lambda$，则要求 $W(b_i)=\lambda$ 才能得到平方为零的 morphism differential。若允许 curved/matrix-factorization 型 morphisms，且 $W(b_1)-W(b_0)$ 可逆，则可用该可逆曲率差构造 contracting homotopy，故对象间 morphism 在相应同伦范畴中消失。证毕。

## 本章小结

非 exact Fukaya theory 的核心新现象是 curvature $\mu^0$。Bounding cochains 是使 curvature 消失或变成标量单位的 Maurer-Cartan 解。Novikov 系数记录面积并保证能量过滤下的无穷和收敛。HMS 中的 Landau-Ginzburg potential 常由 disk counts 产生。

## 练习

**练习 5.1.** 验证 Novikov field 中两个收敛形式和的乘积仍满足指数趋向无穷条件。

**练习 5.2.** 展开 curved $A_\infty$ 方程在零个和一个输入上的低阶形式。

**练习 5.3.** 证明命题 5.9 中 $\mu_b^0=0$ 蕴含 $\mu_b^1{}^2=0$。

**练习 5.4.** 对 Laurent polynomial $W=x+x^{-1}$，计算其 critical points 和 Jacobian ring。
