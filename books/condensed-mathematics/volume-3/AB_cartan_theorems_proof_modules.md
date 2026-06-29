# 附录 AB：Cartan A/B 的证明模块

## AB.0 目标

附录 V 把 Cartan A/B 当作输入并证明其形式后果。本附录进一步拆解 Cartan A/B 的经典证明依赖，说明完整证明需要哪些复分析模块。

本附录不重证 Weierstrass preparation、Oka coherence 或 Cousin 问题解定理；这些是经典多复变输入。书内证明从这些输入出发，推出 Cartan A/B。

## AB.1 解析局部代数输入

**输入定理 AB.1（Weierstrass preparation/division）.** 在收敛幂级数环

$$
\mathbb C\{z_1,\ldots,z_n\}
$$

中，对某一变量为 distinguished power series 的元素可写成单位乘以 Weierstrass polynomial，并有带余除法。

**输入定理 AB.2（Oka coherence）.** 复流形 $X$ 上结构层 $\mathcal O_X$ 是 coherent sheaf of rings。也就是说，局部有限生成的关系 sheaf 仍局部有限生成。

**推论 AB.3.** 若 $\mathcal F,\mathcal G$ 是相干解析层，则 kernel、image、cokernel 和 extension 仍相干。

**证明.** 相干层局部有有限表示。对有限表示之间的态射取 kernel/image/cokernel，可化为有限自由 $\mathcal O_X$-模之间态射的 kernel/image/cokernel。Oka coherence 保证关系 sheaf 有限生成，故结果相干。extension 情形由局部 presentation 的 horseshoe 构造得到。证毕。

## AB.2 Cousin 问题输入

设 $U$ 是 Stein 空间，$\mathfrak U=\{U_i\}$ 是 Stein 开覆盖。

**输入定理 AB.4（加性 Cousin 问题，coherent form）.** 设 $\mathcal F$ 是 Stein 空间 $U$ 上相干解析层。对任意 Čech 1-cocycle

$$
c_{ij}\in\mathcal F(U_i\cap U_j)
$$

若覆盖足够精细，则存在 $b_i\in\mathcal F(U_i)$ 使

$$
c_{ij}=b_j-b_i.
$$

等价地，在 Stein 空间上，相干层的局部 1-cocycle 可解。

**输入定理 AB.5（高阶 Cousin 消没）.** 对 Stein 空间上的相干层，所有高阶 Čech cocycle 可在 Stein 细化上解掉。

AB.4-AB.5 的经典证明使用 holomorphic convexity、Runge approximation、解 $\bar\partial$ 或 Oka-Weil 型逼近。

## AB.3 Cartan B

**定理 AB.6（Cartan B，由 Cousin 消没推出）.** 若 $U$ 是 Stein 空间，$\mathcal F$ 是相干解析层，则

$$
H^q(U,\mathcal F)=0
\qquad(q>0).
$$

**证明.** 取 Stein acyclic 基覆盖，并用 sheaf cohomology 的 Čech 细化描述：

$$
H^q(U,\mathcal F)
=
\varinjlim_{\mathfrak U}H^q(\check C^\bullet(\mathfrak U,\mathcal F)).
$$

输入定理 AB.5 说明每个 $q>0$ 的 Čech cocycle 在某个 Stein 细化上成为 coboundary。因此 direct limit 中每个 cohomology 类为零。故 $H^q(U,\mathcal F)=0$。证毕。

## AB.4 Cartan A

**输入定理 AB.7（局部嵌入与分离）.** Stein 空间有足够多全局全纯函数，使每个点附近可用全局函数给出局部坐标或局部嵌入。

**定理 AB.8（Cartan A，由 Cartan B 与 Oka coherence 推出）.** 若 $U$ 是 Stein 空间，$\mathcal F$ 是相干解析层，则 $\mathcal F$ 由全局截面生成。

**证明.** 固定 $x\in U$。由相干性，存在邻域 $V$ 和有限个局部截面 $s_1,\ldots,s_r\in\mathcal F(V)$ 生成 $\mathcal F|_V$。取闭解析子空间或相干商 sheaf 记录“在 $x$ 附近给定 jet 的局部截面能否由全局截面实现”的障碍。

具体地，对足够小的 Stein 邻域 $V$，考虑短正合列

$$
0\to\mathcal K\to\mathcal F\to\mathcal F/\mathcal K\to0
$$

其中 $\mathcal F/\mathcal K$ 支撑在 $x$ 的有限阶邻域上，并携带给定 germ 的有限 jet。由 AB.3，$\mathcal K$ 相干；由 Cartan B，

$$
H^1(U,\mathcal K)=0.
$$

全局截面长正合列于是给

$$
\Gamma(U,\mathcal F)\to\Gamma(U,\mathcal F/\mathcal K)
$$

满射。选择映到给定局部生成 germ 的全局截面，即得 $\mathcal F_x$ 由全局截面生成。证毕。

**注 AB.9.** 上述证明压缩了经典证明中的 jet interpolation 和局部嵌入步骤。AB.7 与 AB.1-AB.2 保证这些有限 jet quotient 是相干对象，Cartan B 负责消除提升障碍。

## AB.5 教材使用边界

在本书中：

1. Cartan B 的使用可追溯到 AB.4-AB.6。
2. Cartan A 的使用可追溯到 AB.7-AB.8。
3. 若要完全自足，需要补 Weierstrass preparation、Oka coherence、Runge approximation、Cousin 问题和 $\bar\partial$ 解估计。

这些内容足以构成一门多复变预备课程，不属于凝聚数学主线。

## 练习

1. 说明 Oka coherence 为什么是 kernel 保持相干的关键。
2. 用 Cartan B 证明短正合列全局截面右端满射。
3. 解释 Cartan A 的证明中为什么要引入有限 jet quotient。
4. 指出 AB.6 中 direct limit over refinements 的作用。
