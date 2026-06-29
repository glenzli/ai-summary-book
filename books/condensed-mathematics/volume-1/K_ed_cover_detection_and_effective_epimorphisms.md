# 附录 K：ED 覆盖检测与有效满射

## K.0 目标

本附录把第一卷中反复使用的原则写成定理：

> 极不连通紧 Hausdorff 空间不仅提供投射对象，还能检测凝聚 sheaf 的单射、满射、同构和正合性。

这里的 ED 是 extremally disconnected 的缩写。证明只使用三类输入：

1. 每个紧 Hausdorff 空间有 ED 满射覆盖。
2. ED 空间关于紧 Hausdorff 满射是投射的。
3. sheaf 的 separated 条件和局部粘合条件。

Gleason 投射性本身仍是外部输入；本附录证明接受该输入后的形式后果。

## K.1 ED 覆盖

**约定 K.1.** 本附录中的 site 是第一卷使用的 compact Hausdorff site，覆盖族为有限联合满射族。若

$$
\{Y_i\to X\}_{i=1}^n
$$

是覆盖，则有限不交并

$$
Y=\coprod_{i=1}^nY_i
$$

仍是紧 Hausdorff 空间，且诱导映射 $Y\to X$ 是满射。因此可把有限覆盖等价地写成一个满射覆盖。

**输入定理 K.2（ED 覆盖存在性）.** 对任意紧 Hausdorff 空间 $X$，存在极不连通紧 Hausdorff 空间 $E$ 和连续满射

$$
p:E\to X.
$$

第一卷附录 J 构造了 $E_X=\operatorname{Stone}(\operatorname{RO}(X))$ 到 $X$ 的连续满射；$E_X$ 的投射性和一般 Gleason 定理仍作为输入。

**输入定理 K.3（ED 投射性）.** 若 $E$ 是 ED 紧 Hausdorff 空间，$q:Y\to Z$ 是紧 Hausdorff 满射，且 $f:E\to Z$ 连续，则存在连续映射 $\widetilde f:E\to Y$ 使

$$
q\circ\widetilde f=f.
$$

特别地，任意满射 $q:Y\to E$ 都有截面 $\sigma:E\to Y$。

## K.2 分离性检测

**引理 K.4（ED 覆盖检测截面相等）.** 设 $F$ 是集合值 sheaf，$X$ 是紧 Hausdorff 空间，$s,t\in F(X)$。若存在 ED 覆盖 $p:E\to X$ 使

$$
p^\ast s=p^\ast t\in F(E),
$$

则 $s=t$。

**证明.** $p:E\to X$ 是覆盖。sheaf 的 separated 性说明：两个截面若在一个覆盖上限制相同，则它们相等。证毕。

**推论 K.5（ED 检测零 sheaf）.** 设 $A$ 是阿贝尔群值 sheaf。若对所有 ED 空间 $E$ 都有

$$
A(E)=0,
$$

则 $A=0$。

**证明.** 对任意 $X$ 和任意 $a\in A(X)$，取 ED 覆盖 $p:E\to X$。由假设，$p^\ast a=0$。引理 K.4 用于阿贝尔群值 sheaf 的底层集合，得 $a=0$。证毕。

## K.3 满射在 ED 上变成逐点满射

一般 sheaf 满射不是逐对象满射；它只要求局部可提升。ED 投射性正好把这个局部条件变成逐点条件。

**命题 K.6（sheaf 满射的 ED 逐点满性）.** 设 $f:F\to G$ 是集合值 sheaf 的满射。若 $E$ 是 ED 空间，则

$$
f(E):F(E)\to G(E)
$$

是集合满射。

**证明.** 取 $y\in G(E)$。因为 $f$ 是 sheaf 满射，存在覆盖族 $\{Y_i\to E\}_{i=1}^n$，使每个 $y|_{Y_i}$ 都提升到某个 $x_i\in F(Y_i)$。

令 $Y=\coprod_iY_i$。则 $q:Y\to E$ 是紧 Hausdorff 满射。截面 $x_i$ 合成一个 $x_Y\in F(Y)$，满足

$$
f(x_Y)=q^\ast y.
$$

由 ED 投射性，$q$ 有截面 $\sigma:E\to Y$。令

$$
x=\sigma^\ast x_Y\in F(E).
$$

于是

$$
f(x)=\sigma^\ast f(x_Y)
=\sigma^\ast q^\ast y
=(q\circ\sigma)^\ast y
=y.
$$

故 $f(E)$ 满。证毕。

**命题 K.7（ED 逐点满性推出 sheaf 满射）.** 设 $f:F\to G$ 是集合值 sheaf 的态射。若对所有 ED 空间 $E$，映射

$$
f(E):F(E)\to G(E)
$$

都是满射，则 $f$ 是 sheaf 满射。

**证明.** 对任意 $X$ 和 $y\in G(X)$，取 ED 覆盖 $p:E\to X$。由假设，存在 $x_E\in F(E)$ 使

$$
f(x_E)=p^\ast y.
$$

这说明 $y$ 在覆盖 $E\to X$ 上局部可提升。因此 $f$ 是 sheaf 满射。证毕。

**定理 K.8（ED 检测 sheaf 满射）.** 集合值 sheaf 态射 $f:F\to G$ 是满射，当且仅当对所有 ED 空间 $E$，$f(E)$ 是集合满射。

**证明.** 由命题 K.6 和命题 K.7。证毕。

## K.4 单射与同构检测

**命题 K.9（ED 检测单射）.** 集合值 sheaf 态射 $f:F\to G$ 是单射，当且仅当对所有 ED 空间 $E$，$f(E)$ 是集合单射。

**证明.**

若 $f$ 是单射，则任意取值函子保持单射，所以 $f(E)$ 单。

反过来，设所有 $f(E)$ 单。对任意 $X$ 和 $s,t\in F(X)$，若 $f(s)=f(t)$，取 ED 覆盖 $p:E\to X$。则

$$
f(E)(p^\ast s)=p^\ast f(s)=p^\ast f(t)=f(E)(p^\ast t).
$$

由 $f(E)$ 单，$p^\ast s=p^\ast t$。引理 K.4 给出 $s=t$。故 $f$ 单。证毕。

**定理 K.10（ED 检测同构）.** 集合值 sheaf 态射 $f:F\to G$ 是同构，当且仅当对所有 ED 空间 $E$，$f(E)$ 是集合双射。

**证明.** sheaf 范畴中的同构等价于既是单射又是满射。由定理 K.8 和命题 K.9 得结论。证毕。

## K.5 阿贝尔 sheaf 的正合性检测

以下结果是第八章正合性检测的精确形式。

**定义 K.11.** 阿贝尔群值 sheaf 的复形

$$
A\xrightarrow{\alpha}B\xrightarrow{\beta}C
$$

在 $B$ 处正合，指

$$
\operatorname{im}(\alpha)=\ker(\beta)
$$

作为阿贝尔群值 sheaf 的子 sheaf 相等。

**定理 K.12（ED 检测正合性）.** 设

$$
A\xrightarrow{\alpha}B\xrightarrow{\beta}C
$$

是阿贝尔群值 sheaf 的复形，即 $\beta\alpha=0$。则它在 $B$ 处正合，当且仅当对所有 ED 空间 $E$，阿贝尔群复形

$$
A(E)\xrightarrow{\alpha(E)}B(E)\xrightarrow{\beta(E)}C(E)
$$

在 $B(E)$ 处正合。

**证明.**

若 sheaf 复形在 $B$ 处正合，则 $\ker\beta\to B$ 等于 $\operatorname{im}\alpha\to B$。由命题 K.6，满射 $A\to\operatorname{im}\alpha$ 在 ED 上逐点满；由有限极限逐点计算，$\ker\beta(E)=\ker(\beta(E))$。因此

$$
\operatorname{im}\alpha(E)=\ker\beta(E).
$$

反过来，设对所有 ED 空间取值后正合。令 $b\in B(X)$ 且 $\beta(b)=0$。取 ED 覆盖 $p:E\to X$。则

$$
\beta(E)(p^\ast b)=p^\ast\beta(b)=0.
$$

由 ED 取值正合，存在 $a_E\in A(E)$ 使

$$
\alpha(E)(a_E)=p^\ast b.
$$

这说明 $b$ 在覆盖 $E\to X$ 上局部属于 $\alpha$ 的像。因此 $b$ 给出 $\ker\beta\to\operatorname{coker}(\operatorname{im}\alpha\to\ker\beta)$ 的零截面。等价地，商 sheaf

$$
Q=\ker\beta/\operatorname{im}\alpha
$$

满足 $Q(E)=0$ 对所有 ED 空间成立。由推论 K.5，$Q=0$，于是 $\operatorname{im}\alpha=\ker\beta$。证毕。

**推论 K.13（短正合列检测）.** 阿贝尔群值 sheaf 列

$$
0\to A\to B\to C\to0
$$

短正合，当且仅当对所有 ED 空间 $E$，

$$
0\to A(E)\to B(E)\to C(E)\to0
$$

是阿贝尔群短正合列。

**证明.** 左端单射由命题 K.9 检测，中间正合由定理 K.12 检测，右端满射由定理 K.8 检测。证毕。

## K.6 与投射生成元的关系

记 $\mathbb Z[\underline E]$ 为由 ED 空间 $E$ 生成的自由凝聚阿贝尔群。第一卷第七章已经证明

$$
\operatorname{Hom}(\mathbb Z[\underline E],A)\cong A(E).
$$

定理 K.12 可改写为：

$$
A\to B\to C
$$

正合，当且仅当对所有 ED 空间 $E$，

$$
\operatorname{Hom}(\mathbb Z[\underline E],A)
\to
\operatorname{Hom}(\mathbb Z[\underline E],B)
\to
\operatorname{Hom}(\mathbb Z[\underline E],C)
$$

正合。

这说明 ED 自由对象不仅是投射对象，而且构成检测正合性的生成族。后续 Ext 与 Tor 的计算依赖的正是这个事实。

## K.7 练习

**练习 K.1.** 证明若 $p:E\to X$ 是覆盖，任意集合值 sheaf $F$ 的限制映射 $F(X)\to F(E)$ 是单射。

**练习 K.2.** 在命题 K.6 的证明中，解释为什么有限不交并 $Y=\coprod_iY_i$ 仍是紧 Hausdorff 空间。

**练习 K.3.** 证明若 $A(E)=0$ 对所有 ED 空间 $E$ 成立，则任意态射 $A\to B$ 都是零态射。

**练习 K.4.** 用定理 K.12 重新证明：若 $E$ 是 ED 空间，则取值函子 $(-)(E):\mathbf{CondAb}\to\mathbf{Ab}$ 是正合函子。
