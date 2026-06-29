# 附录 J：Serre 对偶的形式证明层

## J.0 目标

第三卷第五章把 Serre duality 标为输入定理，因为配对完美性需要 Hodge theory、椭圆正则性或 Clausen-Scholze 的相干对偶定理。本附录不重证完美性，而是证明以下形式部分：

1. 链级配对与微分相容时，诱导上同调配对。
2. 有限维复形中的 perfect pairing 等价于导出对偶同构。
3. $f_!\dashv f^!$ 的 counit 给出 trace map。
4. 在 $X\to *$ 的情形，Serre duality 可写成 derived Hom 公式。

这些步骤是教材中必须展开的线性代数和范畴论部分。

## J.1 链级配对与上同调配对

设 $C^\bullet,D^\bullet$ 是复向量空间上有界上链复形。设 $n$ 为整数。一个次数 $n$ 的链级配对是双线性映射族

$$
\langle-,-\rangle^i:
C^i\times D^{n-i}\to\mathbb C
$$

满足相容公式

$$
\langle d_Cx,y\rangle^{i+1}
=
(-1)^{i+1}\langle x,d_Dy\rangle^i.
$$

**命题 J.1.** 上述配对诱导良定义的上同调配对

$$
H^i(C^\bullet)\times H^{n-i}(D^\bullet)\to\mathbb C.
$$

**证明.** 若 $x$ 和 $y$ 都是 cocycle，则 $d_Cx=0$、$d_Dy=0$。若把 $x$ 改成 $x+d_Ca$，则

$$
\langle d_Ca,y\rangle
=
(-1)^i\langle a,d_Dy\rangle
=0.
$$

若把 $y$ 改成 $y+d_Db$，则

$$
\langle x,d_Db\rangle
=
(-1)^{i+1}\langle d_Cx,b\rangle
=0.
$$

因此配对只依赖上同调类。证毕。

## J.2 Perfect pairing 与导出对偶

设 $C^\bullet$ 是有限维有界复形。定义线性对偶复形

$$
(C^\vee)^k=\operatorname{Hom}_\mathbb C(C^{-k},\mathbb C)
$$

微分由

$$
d_{C^\vee}(\varphi)=(-1)^{k+1}\varphi\circ d_C
$$

给出，使得 $H^k(C^\vee)\cong H^{-k}(C)^\vee$。

**命题 J.2.** 次数 $n$ 的链级配对等价于复形态射

$$
C^\bullet\to (D^\bullet)^\vee[-n].
$$

**证明.** 给定配对，定义

$$
\Phi(x)(y)=\langle x,y\rangle
$$

其中 $x\in C^i$，$y\in D^{n-i}$。配对的符号相容正是 $\Phi$ 与微分交换的条件。反过来，任一复形态射 $\Phi$ 逐次给出 $C^i\to\operatorname{Hom}(D^{n-i},\mathbb C)$，即双线性配对。两种构造互逆。证毕。

**命题 J.3.** 若 $C^\bullet,D^\bullet$ 有界且各项有限维，则命题 J.2 中的态射为 quasi-isomorphism，当且仅当命题 J.1 的上同调配对对所有 $i$ 都 perfect。

**证明.** 对有限维复形，取上同调与线性对偶交换：

$$
H^i((D^\bullet)^\vee[-n])
\cong
H^{n-i}(D^\bullet)^\vee.
$$

因此 $H^i(\Phi)$ 正是映射

$$
H^i(C^\bullet)\to H^{n-i}(D^\bullet)^\vee
$$

由上同调配对诱导。它是同构当且仅当配对 perfect。逐 $i$ 成立即 $\Phi$ 是 quasi-isomorphism。证毕。

## J.3 Dolbeault-Serre 配对的形式检查

设 $X$ 是紧复流形，$\dim_\mathbb C X=n$，$E$ 是全纯向量丛。令

$$
C^\bullet=\Gamma(X,\mathcal A^{0,\bullet}(E)),
$$

$$
D^\bullet=\Gamma(X,\mathcal A^{0,\bullet}(E^\vee\otimes\omega_X)).
$$

定义

$$
\langle\alpha,\beta\rangle
=
\int_X \operatorname{ev}(\alpha\wedge\beta).
$$

**命题 J.4.** 该配对满足 J.1 的链级相容公式。

**证明.** 这正是附录 D 命题 D.3。Leibniz 规则给出

$$
\bar\partial\operatorname{ev}(\alpha\wedge\beta)
=
\operatorname{ev}(\bar\partial\alpha\wedge\beta)
+(-1)^i\operatorname{ev}(\alpha\wedge\bar\partial\beta).
$$

在紧无边界 $X$ 上积分 $\bar\partial$-exact 的 $(n,n-1)$ 或相应边界项为零。移项得到符号公式。证毕。

**输入定理 J.5（Serre perfectness）.** 上述配对在上同调上 perfect。

**本书不证明的部分.** perfectness 依赖椭圆正则性、Hodge theory 或 Clausen-Scholze 的相干对偶输入；本附录只证明 perfectness 一旦成立就等价于导出对偶同构。

**推论 J.6.** 接受输入定理 J.5 后，有导出同构

$$
R\Gamma(X,E)
\simeq
R\operatorname{Hom}_\mathbb C
(R\Gamma(X,E^\vee\otimes\omega_X),\mathbb C)[-n].
$$

**证明.** Dolbeault resolution 给出 $R\Gamma(X,E)\simeq C^\bullet$ 和 $R\Gamma(X,E^\vee\otimes\omega_X)\simeq D^\bullet$。命题 J.2 将积分配对视为复形态射

$$
C^\bullet\to(D^\bullet)^\vee[-n].
$$

由输入定理 J.5 和命题 J.3，该态射是 quasi-isomorphism。证毕。

## J.4 Trace map 与 $f^!$

设

$$
f:X\to *
$$

并在某个稳定闭幺半范畴中有两对伴随

$$
f^*\dashv f_*,
\qquad
f_!\dashv f^!.
$$

**定义 J.7.** trace map 是 counit

$$
\operatorname{Tr}_f:
f_!f^!\mathbf 1\to \mathbf 1.
$$

若 $f$ proper 且 $f_!=Rf_*$，这给出

$$
R\Gamma(X,f^!\mathbf 1)\to \mathbb C.
$$

**命题 J.8（Grothendieck duality 的 Hom 形式）.** 假设投影公式

$$
f_!(f^*A\otimes F)\simeq A\otimes f_!F
$$

成立。则对 $F\in\mathcal D$、$B\in\mathcal C$ 有自然等价

$$
f_*\mathcal Hom_\mathcal D(F,f^!B)
\simeq
\mathcal Hom_\mathcal C(f_!F,B).
$$

**证明.** 对任意 $A\in\mathcal C$，连续使用 $f^*\dashv f_*$、闭幺半伴随、$f_!\dashv f^!$、投影公式和 $\otimes\dashv\mathcal Hom$：

$$
\begin{aligned}
\operatorname{Map}_{\mathcal C}
(A,f_*\mathcal Hom_\mathcal D(F,f^!B))
&\simeq
\operatorname{Map}_{\mathcal D}
(f^*A,\mathcal Hom_\mathcal D(F,f^!B))\\
&\simeq
\operatorname{Map}_{\mathcal D}
(f^*A\otimes F,f^!B)\\
&\simeq
\operatorname{Map}_{\mathcal C}
(f_!(f^*A\otimes F),B)\\
&\simeq
\operatorname{Map}_{\mathcal C}
(A\otimes f_!F,B)\\
&\simeq
\operatorname{Map}_{\mathcal C}
(A,\mathcal Hom_\mathcal C(f_!F,B)).
\end{aligned}
$$

由 Yoneda 引理得到结论。证毕。

**推论 J.9（trace pairing）.** 取 $B=\mathbf 1$，命题 J.8 给出

$$
f_*\mathcal Hom(F,f^!\mathbf 1)
\simeq
\mathcal Hom(f_!F,\mathbf 1).
$$

等价地，有自然配对

$$
f_!F\otimes f_*\mathcal Hom(F,f^!\mathbf 1)\to\mathbf 1.
$$

**证明.** 第一式是命题 J.8 的特例。第二式是闭幺半范畴中对象与其内部 Hom 的 evaluation。证毕。

**推论 J.10（Serre duality 的六函子形式）.** 若 $F$ 是相干且满足有限性/完美性假设，则推论 J.9 的配对诱导

$$
R\operatorname{Hom}(f_!F,\mathbf 1)
\simeq
f_*\mathcal Hom(F,f^!\mathbf 1).
$$

**证明.** 命题 J.8 已给出更强的内部 Hom 等价。有限性保证两侧落在可由有限维复形表示的子范畴，perfectness 保证该等价对应的上同调配对为完美配对。证毕。

## J.5 对第三卷第五章的回填

1. 第五章的“配对与 $\bar\partial$ 相容”由命题 J.4 证明。
2. “完美性推出导出对偶同构”由命题 J.3 和推论 J.6 证明。
3. “trace map 来自 $f_!\dashv f^!$”由定义 J.7 说明。
4. “六函子形式的 Serre duality”由命题 J.8、推论 J.9 和推论 J.10 给出形式证明。

## 练习

**练习 J.1.** 检查命题 J.2 中对偶复形微分的符号。

**练习 J.2.** 设 $C^\bullet$ 只在次数 $0$ 非零。说明命题 J.3 退化为普通有限维线性代数中的 perfect pairing。

**练习 J.3.** 对 Riemann surface 和线丛 $L$，把推论 J.6 写成 $H^0/H^1$ 的两个对偶同构。

**练习 J.4.** 在命题 J.8 的证明中，写出投影公式使用的具体一步。
