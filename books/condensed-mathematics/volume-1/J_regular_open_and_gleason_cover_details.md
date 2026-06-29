# 附录 J：Regular open 代数与 Gleason cover 细节

## J.0 目标

附录 D 已经说明：对紧 Hausdorff 空间 $X$，regular open subsets 构成完备 Boolean 代数，且

$$
E_X=\operatorname{Stone}(\operatorname{RO}(X))
$$

给出一个极不连通紧 Hausdorff 空间，存在满射

$$
p:E_X\to X.
$$

本附录补齐这个构造中可直接证明的拓扑细节。Gleason 定理中“极不连通紧 Hausdorff 空间等价于 projective compact Hausdorff space”的投射性方向仍作为外部输入；但 Gleason cover 的 regular open 构造本身在这里写成完整证明。

## J.1 Regular open subsets

设 $X$ 是拓扑空间。开集 $U\subset X$ 称为 regular open，如果

$$
U=\operatorname{int}\overline U.
$$

记所有 regular open subsets 构成的集合为 $\operatorname{RO}(X)$。

**引理 J.1.** 若 $U,V\in\operatorname{RO}(X)$，则

$$
U\wedge V=U\cap V,
$$

$$
\neg U=\operatorname{int}(X\setminus U),
$$

$$
U\vee V=\operatorname{int}\overline{U\cup V}
$$

仍属于 $\operatorname{RO}(X)$。

**证明.** 对任意开集 $W$，集合 $\operatorname{int}\overline W$ 是 regular open，因为

$$
\operatorname{int}\overline{\operatorname{int}\overline W}
=
\operatorname{int}\overline W.
$$

这给出 $U\vee V$ 的 regular open 性。

交的情形：若 $U,V$ regular open，则 $U\cap V$ 开。需要证明

$$
\operatorname{int}\overline{U\cap V}=U\cap V.
$$

包含 $U\cap V\subset\operatorname{int}\overline{U\cap V}$ 对开集成立。反向，由

$$
\overline{U\cap V}\subset \overline U
$$

得到

$$
\operatorname{int}\overline{U\cap V}\subset\operatorname{int}\overline U=U.
$$

同理

$$
\operatorname{int}\overline{U\cap V}\subset V.
$$

故

$$
\operatorname{int}\overline{U\cap V}\subset U\cap V.
$$

于是交仍 regular open。

补的情形：令 $W=\operatorname{int}(X\setminus U)$。它开，且

$$
\operatorname{int}\overline W
\subset
\operatorname{int}(X\setminus U)
=W,
$$

因为 $\overline W\subset X\setminus U$。反向包含对开集成立。证毕。

**引理 J.2.** 若 $X$ 是任意拓扑空间，则 $\operatorname{RO}(X)$ 在上述运算下是 Boolean 代数。若 $X$ 有任意并，则 $\operatorname{RO}(X)$ 是完备 Boolean 代数，任意族 $(U_i)$ 的上确界为

$$
\bigvee_i U_i
=
\operatorname{int}\overline{\bigcup_iU_i}.
$$

**证明.** 交换律、结合律、吸收律和分配律可由 regular open 算子

$$
r(W)=\operatorname{int}\overline W
$$

与集合交并的关系逐项验证。补元满足

$$
U\cap\neg U=\varnothing
$$

以及

$$
U\vee\neg U
=
\operatorname{int}\overline{U\cup\operatorname{int}(X\setminus U)}
=X,
$$

因为 $U\cup\operatorname{int}(X\setminus U)$ 在 $X$ 中稠密。任意上确界公式由定义可知它是包含所有 $U_i$ 的最小 regular open 集；下确界由 De Morgan 公式定义。证毕。

## J.2 Stone 空间到 $X$ 的映射

现在设 $X$ 是紧 Hausdorff 空间，令

$$
B=\operatorname{RO}(X),
\qquad
E_X=\operatorname{Stone}(B).
$$

点 $\mathfrak u\in E_X$ 可等价看作 $B$ 上的超滤子。对 $U\in B$，Stone 基开闭集记为

$$
\widehat U=\{\mathfrak u\mid U\in\mathfrak u\}.
$$

**引理 J.3.** 对任意 $\mathfrak u\in E_X$，闭集族

$$
\{\overline U\mid U\in\mathfrak u\}
$$

有有限交性质，因此交非空。

**证明.** 若 $U_1,\ldots,U_n\in\mathfrak u$，则

$$
U_1\wedge\cdots\wedge U_n
=
U_1\cap\cdots\cap U_n
\in\mathfrak u.
$$

滤子不含 $0=\varnothing$，故该交非空。于是闭包交也非空。紧性给出全体闭集交非空。证毕。

**引理 J.4.** 上述交恰有一个点。

**证明.** 设 $x\ne y$ 都属于

$$
\bigcap_{U\in\mathfrak u}\overline U.
$$

由紧 Hausdorff 空间的正规性，取开集 $O$ 使

$$
x\in O,\qquad y\notin\overline O.
$$

令

$$
V=\operatorname{int}\overline O.
$$

则 $V\in\operatorname{RO}(X)$，$x\in V$，且可取 $O$ 足够小使 $y\notin\overline V$。由于 $\mathfrak u$ 是超滤子，$V\in\mathfrak u$ 或 $\neg V\in\mathfrak u$。

若 $V\in\mathfrak u$，则 $y\in\overline V$，矛盾。若 $\neg V\in\mathfrak u$，则 $x\in\overline{\neg V}$。但

$$
\overline{\neg V}
=
X\setminus V
$$

对 regular open $V$ 成立，而 $x\in V$，矛盾。故交至多一个点。结合引理 J.3，交恰有一个点。证毕。

**定义 J.5.** 定义

$$
p:E_X\to X
$$

为使

$$
\{p(\mathfrak u)\}
=
\bigcap_{U\in\mathfrak u}\overline U
$$

成立的映射。

## J.3 连续性与满射性

**命题 J.6.** 映射 $p:E_X\to X$ 连续。

**证明.** 设 $O\subset X$ 开。证明 $p^{-1}(O)$ 开。取 $\mathfrak u\in p^{-1}(O)$，令 $x=p(\mathfrak u)\in O$。由正规性，存在开集 $W$ 使

$$
x\in W,\qquad \overline W\subset O.
$$

令

$$
V=\operatorname{int}\overline W.
$$

则 $V\in\operatorname{RO}(X)$，$x\in V$，且 $\overline V\subset O$。若 $V\notin\mathfrak u$，则 $\neg V\in\mathfrak u$，从而

$$
x=p(\mathfrak u)\in\overline{\neg V}=X\setminus V,
$$

与 $x\in V$ 矛盾。因此 $V\in\mathfrak u$，即 $\mathfrak u\in\widehat V$。

若 $\mathfrak v\in\widehat V$，则 $p(\mathfrak v)\in\overline V\subset O$。故

$$
\mathfrak u\in \widehat V\subset p^{-1}(O).
$$

每个点都有这样的 Stone 基开邻域，所以 $p^{-1}(O)$ 开。证毕。

**命题 J.7.** 映射 $p:E_X\to X$ 满射。

**证明.** 固定 $x\in X$。令

$$
\mathcal F_x=\{U\in\operatorname{RO}(X)\mid x\in U\}.
$$

这是 proper filter：$X\in\mathcal F_x$，$\varnothing\notin\mathcal F_x$，并且有限交仍含 $x$。由超滤子引理，取超滤子 $\mathfrak u_x$ 包含 $\mathcal F_x$。对任意 $U\in\mathfrak u_x$，若 $x\notin\overline U$，则存在 regular open 邻域 $V$ 使

$$
x\in V,\qquad V\cap U=\varnothing.
$$

于是 $V\in\mathcal F_x\subset\mathfrak u_x$，但 $U,V\in\mathfrak u_x$ 且交为空，矛盾。故 $x\in\overline U$ 对所有 $U\in\mathfrak u_x$ 成立。因此

$$
x\in\bigcap_{U\in\mathfrak u_x}\overline U.
$$

按定义 J.5，$p(\mathfrak u_x)=x$。故 $p$ 满射。证毕。

## J.4 极不连通性

**命题 J.8.** $E_X$ 是极不连通紧 Hausdorff 空间。

**证明.** $B=\operatorname{RO}(X)$ 是完备 Boolean 代数。附录 D 命题 D.8 证明完备 Boolean 代数的 Stone 空间极不连通。Stone 空间本身紧 Hausdorff，故结论成立。证毕。

**定理 J.9（Gleason cover 的可证明部分）.** 对任意紧 Hausdorff 空间 $X$，空间

$$
E_X=\operatorname{Stone}(\operatorname{RO}(X))
$$

是极不连通紧 Hausdorff 空间，并且存在连续满射

$$
p:E_X\to X.
$$

**证明.** 极不连通性由命题 J.8，连续性由命题 J.6，满射性由命题 J.7。证毕。

## J.5 与 Gleason 投射性定理的边界

本附录没有证明以下定理：

> 极不连通紧 Hausdorff 空间在 $\mathbf{CHaus}$ 中关于满射投射。

这仍是 Gleason 定理的核心输入。第一卷使用的逻辑现在分成两层：

1. 本附录证明每个 $X\in\mathbf{CHaus}$ 有极不连通满射覆盖 $E_X\to X$。
2. Gleason 输入定理说明极不连通对象对满射有提升性质。

这样，第六至第八章中“用 ED 对象检测 sheaf 与正合性”的覆盖存在部分已经在书内闭合；只剩提升性质作为外部输入。

## 练习

**练习 J.1.** 证明若 $V$ regular open，则 $\overline{\neg V}=X\setminus V$。

**练习 J.2.** 在命题 J.6 中，说明如何由紧 Hausdorff 的正规性选取 $W$ 使 $\overline{\operatorname{int}\overline W}\subset O$。

**练习 J.3.** 对有限离散 $X$，计算 $\operatorname{RO}(X)$、$E_X$ 和 $p:E_X\to X$。

**练习 J.4.** 说明本附录为什么没有推出 ED 对象的投射性。
