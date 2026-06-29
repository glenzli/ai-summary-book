# 附录 AG：Runge、Cousin 与 Cartan B 的机制

## AG.0 目标

Cartan B 的证明核心是：Stein 空间上相干层的 Čech cocycle 可以通过逼近和分裂逐步解掉。本附录把 Runge approximation、Cousin problem 和 Cartan B 的关系写成教材式链条。

Runge approximation 和 Cousin 问题解定理作为经典输入。

## AG.1 Runge pair

**定义 AG.1.** Stein 空间中的开子集 $V\subset U$ 称为 Runge 子集，如果 $\mathcal O(U)$ 在 $\mathcal O(V)$ 中对 compact-open 拓扑稠密。

**输入定理 AG.2（Runge approximation）.** 若 $V\subset U$ 是 Runge pair，则任意 $f\in\mathcal O(V)$ 可在 $V$ 的紧子集上一致逼近为来自 $\mathcal O(U)$ 的函数。

**输入定理 AG.3（相干层 Runge approximation）.** 若 $V\subset U$ 是 Stein Runge pair，$\mathcal F$ 是 $U$ 上相干层，则

$$
\Gamma(U,\mathcal F)\to\Gamma(V,\mathcal F)
$$

的像在自然 Fréchet 拓扑中稠密。

## AG.2 二开 Cousin 分裂

设

$$
U=U_1\cup U_2
$$

是 Stein 开集的覆盖，且 $U_{12}$ Stein。

**输入定理 AG.4（二开加性 Cousin 分裂）.** 对 Stein 空间上相干层 $\mathcal F$，若 $c\in\Gamma(U_{12},\mathcal F)$ 是交叠截面，则在可允许的 Stein 细化后，存在

$$
b_1\in\Gamma(U_1,\mathcal F),\qquad b_2\in\Gamma(U_2,\mathcal F)
$$

使

$$
c=b_2|_{U_{12}}-b_1|_{U_{12}}.
$$

**命题 AG.5.** 在 AG.4 的假设下，二开覆盖的 Čech $H^1$ 为零。

**证明.** 二开覆盖的 1-cocycle 就是交叠上的截面 $c\in\Gamma(U_{12},\mathcal F)$。AG.4 说明它是 0-cochain $(b_1,b_2)$ 的 coboundary。证毕。

## AG.3 高阶消没

**输入定理 AG.6（Leray-Cartan covering refinement）.** Stein 空间存在 Stein 开覆盖基，使每个有限交 Stein，并且每个高阶 Čech cocycle 可通过有限次二开 Cousin 分裂和细化降为 coboundary。

**定理 AG.7（Cartan B 机制）.** 若 $U$ 是 Stein 空间，$\mathcal F$ 相干，则

$$
H^q(U,\mathcal F)=0
\qquad(q>0).
$$

**证明.** 取 Stein Leray 覆盖基。任意 sheaf cohomology 类可由某个 Stein 覆盖上的 Čech cocycle 表示。由 AG.6，经过 Stein 细化，该 cocycle 成为 coboundary。因此它在细化 direct limit 中为零。证毕。

## AG.4 Cartan A 的推导接口

Cartan A 可由 Cartan B 和有限 jet lifting 推出。具体地，给定 $x\in U$ 和 $\mathcal F_x$ 中的 germ，选择有限阶邻域 quotient $\mathcal Q$ 记录该 germ 的 jet。短正合列

$$
0\to\mathcal K\to\mathcal F\to\mathcal Q\to0
$$

中 $\mathcal K$ 相干。Cartan B 给

$$
H^1(U,\mathcal K)=0,
$$

从而

$$
\Gamma(U,\mathcal F)\to\Gamma(U,\mathcal Q)
$$

满射。取提升即可得到全局截面在 $x$ 处实现给定 germ。

## AG.5 与 Dolbeault 方法的关系

在光滑 Stein 域上，另一条证明 Cartan B 的路线是解 $\bar\partial$：

1. 用 Dolbeault resolution 计算相干层上同调；
2. 通过 $\bar\partial$ 解算子证明高阶 cohomology 消失；
3. 用相干层 resolution 把一般相干层化为有限自由层情形。

这条路线依赖 $L^2$ estimates 或 integral kernel estimates。Runge-Cousin 路线和 $\bar\partial$ 路线在结论上给出同一个 Cartan B。

## 练习

1. 写出二开覆盖的 Čech 1-coboundary 公式。
2. 说明 AG.5 为什么不能直接处理三重交叠上的 2-cocycle。
3. 解释 AG.7 中细化 direct limit 的作用。
4. 比较 Runge-Cousin 路线和 $\bar\partial$ 路线各自需要的分析输入。
