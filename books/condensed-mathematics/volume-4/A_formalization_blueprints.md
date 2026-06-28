# 附录 A：形式化蓝图

## A.1 状态分级

本附录把第四卷中适合形式化的命题拆成可实施的 lemma。为避免夸大现有形式化程度，采用三级标签：

1. **已形式化型**：主要依赖范畴、预层、sheaf、有限极限和等化子。
2. **可形式化路线**：数学证明清楚，但需要补齐拓扑、compact Hausdorff 或 derived category 库。
3. **输入定理型**：应作为外部数学定理接入，例如 Gleason 定理、Nöbeling 定理、solid tensor 的核心定理。

第四卷多数内容属于第二类；solid、analytic、liquid 的深层结构属于第三类。

## A.2 站点和覆盖的数据

形式化一个站点可拆成以下数据：

1. 一个小范畴 $\mathcal C$。
2. 对每个 $U\in\mathcal C$，给定覆盖筛或覆盖族集合 $J(U)$。
3. 最大筛覆盖。
4. 覆盖对拉回稳定。
5. 覆盖的传递性。

若采用覆盖族而非覆盖筛，还需要证明覆盖族生成的筛满足 Grothendieck topology 公理。对凝聚数学的有限覆盖站点，形式化时常把 covering family 先写成有限索引族

$$
(U_i\to U)_{i\in I}
$$

再证明它生成覆盖筛。

## A.3 匹配族与等化子

给定覆盖 $\mathcal U=\{U_i\to U\}$，定义

$$
\operatorname{Match}_F(\mathcal U)
=
\operatorname{Eq}
\left(
\prod_iF(U_i)
\rightrightarrows
\prod_{i,j}F(U_i\times_UU_j)
\right).
$$

形式化目标：

**Lemma A.3.1.** 对取值于具有有限极限范畴 $\mathcal E$ 的预层 $F$，覆盖 $\mathcal U$ 上的 sheaf 条件等价于自然映射

$$
F(U)\to\operatorname{Match}_F(\mathcal U)
$$

为同构。

证明拆分：

1. 构造限制映射 $F(U)\to\prod_iF(U_i)$。
2. 证明它落在等化子中。
3. 把等化子元素解释为相容截面族。
4. sheaf 的存在性给出满性。
5. sheaf 的唯一性给出单性。

在 Lean 风格中，关键不是计算公式，而是让 $U_i\times_UU_j$ 的投影和限制映射的复合保持 definitional 或可重写相等。

## A.4 可表 sheaf 的形式化路线

对于 compact Hausdorff 站点中的可表预层，证明目标是：

**Theorem A.4.1.** 若 $T\in\mathbf{CHaus}$，则 $h_T(S)=\operatorname{Hom}(S,T)$ 是 sheaf。

依赖 lemma：

1. 有限余并 $\coprod_iS_i$ 在 $\mathbf{CHaus}$ 中存在。
2. 若 $q:X\to Y$ 是 compact Hausdorff 空间之间的连续满射，则 $q$ 是商映射。
3. 对商映射 $q:X\to Y$，映射 $f:Y\to T$ 连续当且仅当 $f\circ q$ 连续。
4. 覆盖匹配条件等价于 $\coprod_iS_i$ 上的映射对等价关系常值。

其中第 2 条的证明为：compact 到 Hausdorff 的连续映射把闭集送到紧集，紧集在 Hausdorff 空间中闭，因此 $q$ 为闭满射，闭满射是商映射。

## A.5 基子站点比较

基子站点比较的形式化应避免一次性证明巨大定理。建议拆为：

**Lemma A.5.1（共同细化范畴非空）。** 对任意 $U\in\mathcal C$，$\mathcal D$-覆盖组成的范畴非空。

**Lemma A.5.2（共同细化范畴滤过）。** 任意两个 $\mathcal D$-覆盖存在共同细化。

**Lemma A.5.3（扩张对象定义良好）。** 对 $\mathcal D$-sheaf $H$，用

$$
\widetilde H(U)=
\varprojlim_{\mathcal U}
\operatorname{Match}_H(\mathcal U)
$$

定义的对象不依赖覆盖选择。

**Lemma A.5.4（扩张满足 sheaf 条件）。** $\widetilde H$ 是 $\mathcal C$ 上 sheaf。

**Theorem A.5.5。** 限制函子

$$
\operatorname{Sh}(\mathcal C)\to\operatorname{Sh}(\mathcal D)
$$

是范畴等价。

形式化风险点在于“覆盖的范畴”本身有 size 问题。教材正文把它隐藏在小站点假设下；机器证明应明确 universe 层级。

## A.6 Ext 和 Tor 的形式化边界

Ext/Tor 的基本代数形式化路线：

1. 在一般阿贝尔范畴中定义 chain complex。
2. 定义 projective object 和 projective resolution。
3. 证明 projective resolution 的同伦唯一性。
4. 定义 $\operatorname{Ext}^i$ 为 Hom 复形的同调。
5. 定义右正合双函子的左导出，得到 Tor。

凝聚数学特有输入：

1. $\mathbf{CondAb}$ 是 Grothendieck 阿贝尔范畴。
2. $\mathbf{CondAb}$ 有足够投射对象。
3. 对极不连通 $E$，$\mathbb Z[\underline E]$ 投射。
4. solid/analytic/liquid 局部化与导出张量的相容性。

前三条之外的 Ext/Tor 形式化基本是同调代数；第四条开始进入 condensed mathematics 的专门定理。

## A.7 推荐的最小形式化项目

一个可实际完成的最小项目可以只包含：

1. 小站点上的 sheaf 条件。
2. 有限覆盖的等化子形式。
3. Čech 微分 $d^2=0$。
4. 基子站点比较的抽象版本。
5. 一般阿贝尔范畴中的两项投射分解 Ext 公式。

这些内容已经足以覆盖第四卷多数“计算模板”的形式核心；深层 condensed 输入可先作为 axiom 或 theorem 参数引入。
