# 附录 V：Stein、Cartan 定理与相干层工具

## V.0 目标

卷三反复使用 Stein open sets、Cartan A/B 和相干层的局部有限性。本附录把这些工具整理成可引用的教材层级：

1. Stein 空间的定义和基本稳定性。
2. Cartan A/B 的精确输入形式。
3. 从 Cartan A 得到紧子集附近的有限生成。
4. 从 Cartan B 得到 acyclic cover 的 Čech 计算。
5. 相干层短正合列与上同调长正合列的使用规则。

Cartan A/B 本身仍作为经典复分析输入定理。其证明依赖 Oka-Weil 逼近、Cousin 问题和 $\bar\partial$ 方法，不在本附录重证。

## V.1 Stein 空间

**定义 V.1.** 复解析空间 $U$ 称为 Stein 空间，如果它满足以下性质：

1. $U$ 全纯凸：对每个紧集 $K\subset U$，其全纯凸包
   $$
   \widehat K=\{x\in U\mid |f(x)|\le \sup_K|f|,\ \forall f\in\mathcal O(U)\}
   $$
   仍为紧集。
2. $U$ 上全纯函数分离点。
3. 每点附近存在有限个全局全纯函数给出局部嵌入到某个 $\mathbb C^N$。

**输入定理 V.2（Stein 稳定性）.** 以下空间是 Stein：

1. $\mathbb C^n$ 的 polydisc 和 Stein 开子集。
2. Stein 空间的闭解析子空间。
3. Stein 空间中由有限多个全纯不等式给出的足够小开子集。
4. 射影空间标准仿射开集 $U_i=\{X_i\ne0\}\simeq\mathbb C^n$ 及其有限交。

本书使用 V.2 时，只需有限交仍为 Stein 的情形。

## V.2 Cartan A/B

**输入定理 V.3（Cartan A）.** 设 $U$ 是 Stein 空间，$\mathcal F$ 是 $U$ 上相干解析层。对每个 $x\in U$，自然映射

$$
\Gamma(U,\mathcal F)\otimes_{\mathbb C}\mathcal O_{U,x}
\to
\mathcal F_x
$$

的像生成 $\mathcal F_x$ 作为 $\mathcal O_{U,x}$-模。

**输入定理 V.4（Cartan B）.** 设 $U$ 是 Stein 空间，$\mathcal F$ 是 $U$ 上相干解析层。则

$$
H^q(U,\mathcal F)=0
\qquad(q>0).
$$

**推论 V.5（紧子集附近的有限全局生成）.** 设 $U$ 是 Stein 空间，$\mathcal F$ 是相干解析层，$K\subset U$ 是紧集。存在全局截面

$$
s_1,\ldots,s_r\in\Gamma(U,\mathcal F)
$$

和开邻域 $K\subset W\subset U$，使

$$
\mathcal O_W^{\oplus r}\to \mathcal F|_W,
\qquad
(a_i)\mapsto\sum_i a_is_i|_W
$$

为满射。

**证明.** 对每个 $x\in K$，由 Cartan A 取有限个全局截面，其 germ 生成 $\mathcal F_x$。满射性是 stalk 条件，因此在某个开邻域 $W_x$ 上仍成立。紧性给有限子覆盖 $W_{x_1},\ldots,W_{x_m}$。把这些点处选出的所有截面合并为有限族 $s_1,\ldots,s_r$，在 $W=\bigcup_jW_{x_j}$ 上生成。证毕。

**推论 V.6（Stein 上的相干层局部有限表示）.** 在 V.5 的设定中，若 $W'\Subset W$，则存在开邻域 $W'\subset W_1\subset W$ 和正合列

$$
\mathcal O_{W_1}^{\oplus m}
\to
\mathcal O_{W_1}^{\oplus r}
\to
\mathcal F|_{W_1}
\to0.
$$

**证明.** V.5 给出 $\mathcal O_W^{\oplus r}\to\mathcal F|_W$。其核 $\mathcal K$ 是相干层，因为相干层范畴对 kernel 封闭。再对紧集 $\overline{W'}\subset W$ 和相干层 $\mathcal K$ 应用 V.5，得到有限个生成元，并在较小邻域 $W_1$ 上给出所需表示。证毕。

## V.3 Acyclic 覆盖与 Čech 计算

**定义 V.7.** 开覆盖 $\mathfrak U=\{U_i\}_{i\in I}$ 对 sheaf $\mathcal F$ 称为 acyclic，如果每个有限交

$$
U_{i_0\cdots i_p}=U_{i_0}\cap\cdots\cap U_{i_p}
$$

满足

$$
H^q(U_{i_0\cdots i_p},\mathcal F)=0
\qquad(q>0).
$$

**命题 V.8（Stein 覆盖是相干层 acyclic 覆盖）.** 若 $X$ 有有限开覆盖 $\mathfrak U=\{U_i\}$，所有有限交 $U_{i_0\cdots i_p}$ 都是 Stein，且 $\mathcal F$ 是相干解析层，则 $\mathfrak U$ 对 $\mathcal F$ acyclic。

**证明.** 对每个有限交应用 Cartan B。证毕。

**定理 V.9（Čech 计算）.** 在命题 V.8 的假设下，自然映射

$$
H^q(\check C^\bullet(\mathfrak U,\mathcal F))
\to
H^q(X,\mathcal F)
$$

为同构。

**证明.** Čech-to-derived spectral sequence 给出

$$
E_1^{p,q}
=
\prod_{i_0<\cdots<i_p}H^q(U_{i_0\cdots i_p},\mathcal F)
\Rightarrow
H^{p+q}(X,\mathcal F).
$$

命题 V.8 使 $q>0$ 行为零，因此谱序列在 $E_2$ 页退化，并且 $q=0$ 行正是 Čech 复形。证毕。

## V.4 短正合列与 Cartan B

**命题 V.10.** 设 $U$ 是 Stein 空间，且

$$
0\to\mathcal F'\to\mathcal F\to\mathcal F''\to0
$$

是相干解析层短正合列。则全局截面列

$$
0\to\Gamma(U,\mathcal F')
\to\Gamma(U,\mathcal F)
\to\Gamma(U,\mathcal F'')
\to0
$$

正合。

**证明.** sheaf 全局截面左正合。其 cokernel 后接长正合列中的

$$
H^1(U,\mathcal F').
$$

Cartan B 给 $H^1(U,\mathcal F')=0$，故 $\Gamma(U,\mathcal F)\to\Gamma(U,\mathcal F'')$ 满射。证毕。

**推论 V.11（Stein 上由全局截面控制扩张）.** 在 V.10 的假设下，任意 $\mathcal F''$ 的全局截面可提升为 $\mathcal F$ 的全局截面。

这条推论是 Cartan B 在代数操作中的主要用法。

**证明.** V.10 断言映射
$\Gamma(U,\mathcal F)\to\Gamma(U,\mathcal F'')$ 满射；满射的定义正是
每个 $\mathcal F''$ 的全局截面都有一个 $\mathcal F$ 中的原像。证毕。

## V.5 与凝聚/analytic 语言的接口

当 $U$ 是 Stein 复空间时，$\Gamma(U,\mathcal F)$ 不只是向量空间。它带有自然 Fréchet 或 LF 型拓扑。进入 condensed/analytic 语言时，需要额外指定：

1. 该拓扑向量空间的凝聚化。
2. 限制映射的连续性。
3. Čech 微分作为连续线性映射。
4. 在 analytic/liquid 范畴中 totalization 与经典 Čech 复形的比较。

本附录只证明 sheaf cohomology 的经典形式计算；拓扑增强由第二卷 liquid 章节和第三卷 Dolbeault 章节负责。

## 练习

1. 证明 $\mathbb P^n$ 的标准覆盖所有有限交都是 Stein。
2. 用 V.10 证明 Stein 空间上相干层满射在全局截面上满射。
3. 设 $\mathfrak U$ 是二开 Stein 覆盖，写出 V.9 给出的 $H^1$ cokernel 公式。
4. 解释为什么 Cartan A 不直接给出整个非紧 Stein 空间上的有限个全局生成元。
