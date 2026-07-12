# 第五章：obstruction、bounding cochains、Novikov 系数与 curved $A_\infty$ 结构

## 本章目标

本章解释为什么 exact 情况之外的 Fukaya theory 需要 curvature、Novikov 系数和 bounding cochains。核心目标是给出 filtered curved $A_\infty$ 语言，并证明 Maurer-Cartan 方程如何把弯曲结构变回可用的 Floer differential。

## 依赖前置知识

需要第一章的 $A_\infty$ 范畴、第三章的 Floer cochains、第四章的 holomorphic polygon 计数。一般虚基本链技术作为外部输入。

## 5.1 Novikov 系数

**定义 5.1.** 设 $k$ 是域，并固定加法子群
$\Gamma\subset\mathbb R$，赋予从 $\mathbb R$ 继承的全序。$\Gamma$-Novikov
field $\Lambda_{k,\Gamma}$ 的非零元素是形式和
$$
a=\sum_{i=0}^{N}a_iT^{\lambda_i},\qquad
a_i\in k^\times,\quad
\lambda_i\in\Gamma,\quad
\lambda_0<\lambda_1<\cdots,
$$
其中 $N\in\mathbb Z_{\ge0}\cup\{\infty\}$；若 $N=\infty$，要求
$\lambda_i\to+\infty$。等价地，指数 support 有下界，并且在每个
$(-\infty,C]$ 中只有有限多个点。把零元另行加入。若约定 support 外
$a_\lambda=0$，则加法与乘法定义为
$$
\begin{aligned}
a+b&=\sum_{\nu\in\Gamma}(a_\nu+b_\nu)T^\nu,\\
ab&=\sum_{\nu\in\Gamma}
\left(\sum_{\substack{\lambda,\mu\in\Gamma\\\lambda+\mu=\nu}}
a_\lambda b_\mu\right)T^\nu.
\end{aligned}
\tag{5.0}
$$
Support 的局部有限性保证 (5.0) 的每个内和有限，并保证结果仍满足同一
support 条件。定义 valuation
$$
v(a)=\min\{\lambda\in\Gamma:a_\lambda\ne0\}\quad(a\ne0),
\qquad v(0)=+\infty.
$$
当 $\Gamma=\mathbb R$ 时，$\Lambda_{k,\mathbb R}$ 称为 universal Novikov
field。本章固定 $\Gamma$ 并简写 $\Lambda=\Lambda_{k,\Gamma}$。Novikov
valuation ring 与其极大理想分别是
$$
\Lambda_{\ge0}=\{a\mid v(a)\ge0\},\qquad
\Lambda_{>0}=\{a\mid v(a)>0\};
$$
按 $v(0)=+\infty$ 的约定，两者都含零元。

**命题 5.1A.** $\Lambda_{k,\Gamma}$ 在 (5.0) 的加法与 Cauchy 乘法下是域，
并且
$$
v(ab)=v(a)+v(b),\qquad
v(a+b)\ge\min\{v(a),v(b)\}.
$$

**证明.** Support 的局部有限性保证每个固定指数的 Cauchy 系数只收到
有限项贡献，并在加法、乘法下保持。首项不能在乘积中消去，所以
$v(ab)=v(a)+v(b)$；加法只能消去最低项，故得到非 Archimedean
不等式。对 $a\ne0$，若它不只是一个 monomial，写
$$
a=a_0T^{\lambda_0}(1+h),\qquad v(h)=\delta>0.
$$
则 $v(h^m)\ge m\delta\to+\infty$，所以
$$
a^{-1}=a_0^{-1}T^{-\lambda_0}\sum_{m\ge0}(-h)^m
$$
按定义 5.1 的 support 条件收敛，并由逐个 valuation quotient 中的有限
几何级数计算满足 $aa^{-1}=1$。Monomial 情形直接取逆。证毕。

**解释 5.2.** $T^\lambda$ 记录 holomorphic curve 的 symplectic area。条件 $\lambda_i\to+\infty$ 保证按能量过滤的无穷和在 Novikov 拓扑下收敛。

**命题 5.3.** 若对每个能量上界 $E$，相关 holomorphic curve 模空间中只有有限多个面积 $\le E$ 的贡献，则以 $T^{\omega(u)}$ 加权的 polygon 计数在 $\Lambda$ 中收敛。

**证明.** 固定 $E$ 时，只有有限多个面积不超过 $E$ 的项，所以任意 valuation 截断只含有限和。随着面积趋向无穷，指数 $\omega(u)$ 趋向无穷，因此形式和满足 Novikov field 的定义。证毕。

## 5.2 curved $A_\infty$ 结构

**定义 5.4.** 一个 filtered curved $A_\infty$ algebra 是完备、分离的
filtered 分次 $\Lambda_{\ge0}$-module $A$，连同连续且保持能量过滤、次数为
$2-d$ 的运算
$$
\mu^d:A^{\otimes d}\to A[2-d],\qquad d\ge0,
$$
其 suspended Taylor family 满足定义 B.6 的过滤局部有限条件与 curved
$A_\infty$ 方程。元素
$$
\mu^0\in A^2
$$
称为 curvature。若去掉过滤与完备性，只能讨论每个固定 arity 的有限
恒等式，不能自动使用后文的 Maurer--Cartan 无穷和。

**解释 5.5.** 在 Fukaya theory 中，$\mu^0$ 来自边界落在同一 Lagrangian 上、带一个输出的 holomorphic disks。若 $\mu^0\ne0$，则 $\mu^1$ 一般不再平方为零。

**命题 5.6.** 令 $x=sa$。对 curved $A_\infty$ algebra，一个输入的
suspended 恒等式准确地是
$$
b_1b_1(x)+b_2(b_0,x)+(-1)^{|x|}b_2(x,b_0)=0.
$$
因此若 $\mu^0=0$，则 $\mu^1{}^2=0$。

**证明.** 这是推论 B.7 的单对象情形。允许内层 arity 为 $0$ 后，只有
把 $b_0$ 插在 $x$ 左、右两处以及复合 $b_1b_1$ 三种分解；右插入的
符号为 $(-1)^{|x|}$。若 $b_0=0$，只剩 $b_1^2=0$，desuspension 后得到
$\mu^1{}^2=0$。证毕。

## 5.3 Maurer-Cartan 元与变形微分

**定义 5.7.** 设 $A$ 是定义 5.4 的 filtered curved $A_\infty$ algebra。
一个 bounding cochain 的 unsuspended 数据是正过滤次数的次数 $1$ 元素
$$
b\in F^{>0}A^1
$$
；写 $\beta=sb\in F^{>0}(sA)^0$。Maurer--Cartan 方程定义为 suspended
等式
$$
\sum_{r\ge0}b_r(\beta^r)=0.
\tag{5.1}
$$
更一般地，若左端等于 $-s(W(b)e)$，其中 $e$ 是 strict unit，则称 $b$
是 weak bounding cochain，$W(b)$ 称为 disk potential 或 obstruction value。
负号来自 (B.7a)。这里采用 Fukaya 范畴常用的 $\mathbb Z/2$ 分次或等价地
把 Novikov/势参数赋予补偿次数的约定；若坚持纯 $\mathbb Z$ 分次，则必须
同时声明 $W(b)$ 的次数使 $W(b)e$ 与 $\mu_b^0$ 同次。

**定义 5.8.** 给定 $\beta=sb$，定义变形后的 suspended Taylor components
$$
b_d^\beta(x_d,\ldots,x_1)
=\sum_{r_0,\ldots,r_d\ge0}
b_{d+r_0+\cdots+r_d}
(\beta^{r_d},x_d,\beta^{r_{d-1}},\ldots,x_1,\beta^{r_0}).
\tag{5.2}
$$
Novikov 正 valuation 条件保证该和收敛。变形后的 $\mu_b^d$ 是
$b_d^\beta$ 按附录 B desuspend 所得的次数 $2-d$ 运算；(5.2) 而非一个
省略 Koszul signs 的 unsuspended 展开是本书的定义。

**命题 5.9.** 若 $b$ 满足 (5.1)，则
$\{\mu_b^d\}_{d\ge1}$ 构成非弯曲 $A_\infty$ 结构。特别地，
$$
(\mu_b^1)^2=0.
$$

**证明.** 先模去任意固定 valuation 阈值；正过滤次数保证所有相关和
变成有限和，所以可以把 $b$ 插入 curved $A_\infty$ 方程的所有空隙并
重排。出现的项正是变形后运算的 $A_\infty$ 方程；其中没有外部输入的
项是变形 suspended curvature
$$
b_0^\beta=\sum_{r\ge0}b_r(\beta^r)
$$
并由 (5.1) 为零；unsuspended curvature 是
$\mu_b^0=-s^{-1}b_0^\beta$。对所有阈值取逆极限，再取一个输入的
方程，得到 $(\mu_b^1)^2=0$。证毕。

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

**例 5.12.** 在满足标准 toric Fano Floer 构造假设时，Lagrangian torus
fibers 配以秩一局部系统给出 weakly unobstructed 对象，其 disk potential
$W$ 是 Laurent polynomial。Critical-point 方程是 deformed self-Floer
cohomology 可能非零的必要计算条件，并在相应外部定理下把这些对象与
$\operatorname{Jac}(W)$ 的局部分量联系起来。不能仅从 Laurent polynomial
的形式推出该 Floer 或 HMS 结论。

**外部输入定理 5.13（FOOO filtered curved algebra，窄版本）.** 设
$(M,\omega)$ 是 closed symplectic manifold，$L\subset M$ 是 compact、
oriented、relatively spin Lagrangian，并在特征零系数与 completed universal
Novikov ring 上工作。固定 tame almost-complex data、相对 spin/orientation
data、gapped energy filtration，并假设 FOOO 的 coherent Kuranishi/virtual
perturbation package。则可在 $L$ 的合适 completed chain model 上构造
unital、gapped、filtered curved $A_\infty$ algebra；不同辅助选择给出
filtered $A_\infty$ homotopy-equivalent models。对正 valuation 的
Maurer--Cartan 解，定义 5.8--5.9 的 self-twisted 运算收敛，并给出该
endomorphism complex 的 deformed Floer cohomology。

这里的结论只陈述一个 $L$ 的 endomorphism algebra。要同时处理多个
Lagrangians、定义对象间 morphisms 并得到 curved Fukaya category，还需
对所有 polygon moduli spaces 选择彼此相容的 virtual perturbations；这不是
从逐对象的 algebra 自动拼出的形式结论。来源：Fukaya--Oh--Ohta--Ono，
*Lagrangian Intersection Floer Theory: Anomaly and Obstruction*。

## 5.5 HMS 中的用途

在 HMS 中，obstruction theory 的作用至少有三类。

1. 对 compact 非 exact Lagrangians，必须用 $b$ 才能得到可用对象。
2. Toric mirror 中，disk potential $W$ 直接成为 Landau-Ginzburg B-side 的 potential。
3. Fukaya category 常按 potential value 分解，对应 matrix factorization category 或 singular fiber 上的 B-side 分类。

**命题 5.14（不同 potential values 的类型）.** 设两个 weakly
unobstructed 对象满足
$$
\mu^0_{b_i}=W_i e_i,
$$
其中 $e_i$ 是严格单位，并采用附录 I 的 $\mathbb Z/2$ curved 约定。令
$d=\mu^1_{b_0,b_1}$。则
$$
d^2=(W_1-W_0)\operatorname{id}.
$$
所以 $W_0=W_1$ 时才得到普通 morphism complex；$W_0\ne W_1$ 时该
数据是 potential $W_1-W_0$ 的 matrix factorization，而不是 cochain
complex。若进一步 $\operatorname{char}k\ne2$ 且 $W_1-W_0$ 可逆，则它在
相应 matrix-factorization homotopy category 中为零。

**证明.** 第一式是计算 I.8 的 desuspension。相同 value 时右端为零，
故定义 5.11 的 fiber category 有良定义 morphism complexes。不同 value
时，最后一句正是命题 I.11 应用于常数 $c=W_1-W_0$。证毕。

## 本章小结

非 exact Fukaya theory 的核心新现象是 curvature $\mu^0$。Bounding cochains 是使 curvature 消失或变成标量单位的 Maurer-Cartan 解。Novikov 系数记录面积并保证能量过滤下的无穷和收敛。HMS 中的 Landau-Ginzburg potential 常由 disk counts 产生。

## 练习

**练习 5.1.** 验证 Novikov field 中两个收敛形式和的乘积仍满足指数趋向无穷条件。

**练习 5.2.** 展开 curved $A_\infty$ 方程在零个和一个输入上的低阶形式。

**练习 5.3.** 证明命题 5.9 中 $\mu_b^0=0$ 蕴含 $\mu_b^1{}^2=0$。

**练习 5.4.** 对 Laurent polynomial $W=x+x^{-1}$，计算其 critical points 和 Jacobian ring。
