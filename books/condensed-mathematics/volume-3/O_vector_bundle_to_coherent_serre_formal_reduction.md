# 附录 O：从向量丛对偶到相干层 Ext-Serre 形式

## O.0 目标

第五章先写向量丛形式的 Serre duality，再写相干层的 Ext 形式。本附录证明一个条件性命题：

> 若相干层 $\mathcal F$ 有有限局部自由 resolution，并且向量丛形式的 Serre duality 已知，则可推出 $\mathcal F$ 的 Ext-Serre duality。

这个命题不是完整的相干 Serre duality 定理。它只说明在有足够有限 resolution 的情形中，如何把向量丛对偶形式通过同调代数传递到相干层。一般相干层的全局有限 resolution、非光滑空间上的 dualizing complex、以及完美性仍是外部输入。

## O.1 有限局部自由 resolution

设 $X$ 是紧复流形，复维数为 $n$，$\omega_X$ 为 canonical bundle。

**定义 O.1.** 相干解析层 $\mathcal F$ 称为在本附录意义下有长度 $m$ 的有限局部自由 resolution，如果存在 exact sequence

$$
0\to E^{-m}\to E^{-m+1}\to\cdots\to E^0\to\mathcal F\to0
$$

其中每个 $E^{-i}$ 是全纯向量丛的截面层。

把 $E^\bullet$ 视为位于次数 $[-m,0]$ 的有界复形，则有 quasi-isomorphism

$$
E^\bullet\simeq\mathcal F.
$$

**输入说明 O.2.** 光滑代数几何中，regular scheme 上 coherent sheaf 局部有有限自由 resolution；解析流形上也有局部有限自由 resolution。全局有限 resolution 是否存在需要额外假设，本附录不声称自动存在。

## O.2 派生 Hom 的模型

**命题 O.3.** 若 $\mathcal F\simeq E^\bullet$ 如 O.1，则

$$
R\mathcal Hom_X(\mathcal F,\omega_X)
\simeq
\mathcal Hom_X(E^\bullet,\omega_X)
$$

其中右侧是项

$$
\mathcal Hom_X(E^{-i},\omega_X)
\cong
(E^{-i})^\vee\otimes\omega_X
$$

组成的对偶复形。

**证明.** 局部自由层可用于计算第一变量的派生 Hom：对局部自由有限秩 $E$，

$$
\mathcal Hom_X(E,-)
\cong
E^\vee\otimes -
$$

是 exact functor。因此有限局部自由 resolution 是计算 $R\mathcal Hom_X(-,\omega_X)$ 的可用 resolution，得到所示模型。证毕。

**推论 O.4.** 有自然同构

$$
\operatorname{Ext}^{n-i}_X(\mathcal F,\omega_X)
\cong
H^{n-i}\left(R\Gamma(X,\mathcal Hom_X(E^\bullet,\omega_X))\right).
$$

**证明.** 由 Ext 定义和命题 O.3：

$$
\operatorname{Ext}^{k}_X(\mathcal F,\omega_X)
=
H^k R\Gamma R\mathcal Hom_X(\mathcal F,\omega_X).
$$

取 $k=n-i$。证毕。

## O.3 向量丛 Serre 对偶的复形化

**输入定理 O.5（向量丛 Serre duality）.** 对每个全纯向量丛 $E$，有自然 perfect pairing

$$
H^q(X,E)
\times
H^{n-q}(X,E^\vee\otimes\omega_X)
\to
\mathbb C.
$$

等价地，

$$
R\Gamma(X,E)
\simeq
R\operatorname{Hom}_\mathbb C
\left(R\Gamma(X,E^\vee\otimes\omega_X),\mathbb C\right)[-n].
$$

**命题 O.6（有界复形版本）.** 设 $E^\bullet$ 是有界复形，每个 $E^a$ 是全纯向量丛。接受输入定理 O.5 后，有自然导出等价

$$
R\Gamma(X,E^\bullet)
\simeq
R\operatorname{Hom}_\mathbb C
\left(
R\Gamma(X,\mathcal Hom_X(E^\bullet,\omega_X)),
\mathbb C
\right)[-n].
$$

**证明.** 对每个单项复形 $E^a[-a]$，结论就是 O.5 加上 shift 相容性。令 $\mathcal T$ 为所有使该等价成立的有界向量丛复形构成的全子范畴。该性质对有限直和、shift 和 cone 封闭：这是因为 $R\Gamma$、$R\mathcal Hom$ 和线性对偶把 distinguished triangle 送到 distinguished triangle，并且有限维 perfect 复形对偶保持三角。

任意有界复形由其 stupid truncation 的有限层逐步从单项复形经 cone 拼成，因此属于 $\mathcal T$。证毕。

## O.4 Ext-Serre duality 的条件性推出

**定理 O.7（有限 resolution 情形的 Ext-Serre duality）.** 设 $\mathcal F$ 是有有限局部自由 resolution 的相干层，并假设向量丛 Serre duality O.5 对 resolution 中所有向量丛及其有限复形成立。则有自然同构

$$
H^i(X,\mathcal F)^\vee
\cong
\operatorname{Ext}^{n-i}_X(\mathcal F,\omega_X).
$$

**证明.** 取有限局部自由 resolution $E^\bullet\simeq\mathcal F$。由 O.6，

$$
R\Gamma(X,E^\bullet)
\simeq
R\operatorname{Hom}_\mathbb C
\left(
R\Gamma(X,\mathcal Hom_X(E^\bullet,\omega_X)),
\mathbb C
\right)[-n].
$$

左侧等价于 $R\Gamma(X,\mathcal F)$。右侧括号内由 O.3 等价于

$$
R\Gamma(X,R\mathcal Hom_X(\mathcal F,\omega_X)).
$$

取第 $i$ 个同调，并使用有限维 perfect 复形的恒等式

$$
H^i(R\operatorname{Hom}_\mathbb C(C,\mathbb C)[-n])
\cong
H^{n-i}(C)^\vee,
$$

得到

$$
H^i(X,\mathcal F)
\cong
\operatorname{Ext}^{n-i}_X(\mathcal F,\omega_X)^\vee.
$$

由 O.5 和有限复形假设，相关上同调与 Ext 群均有限维。两边再取有限维线性对偶，得到所需形式。证毕。

**边界 O.8.** 定理 O.7 的假设不可删除：

1. 若没有有限 resolution，不能用有限复形归纳。
2. 若上同调不是有限维，取双对偶可能改变对象。
3. 若 $X$ 非光滑，$\omega_X$ 可能需要替换为 dualizing complex。
4. 若 $X$ 非 proper，紧支条件和 $f_!$ 会进入公式。

## O.5 六函子语言中的同一证明

令 $f:X\to *$。若 $F$ 是 perfect object，则 duality 可写为

$$
R\operatorname{Hom}(Rf_*F,\mathbb C)
\simeq
Rf_*R\mathcal Hom(F,f^!\mathbb C).
$$

当 $F=\mathcal F$ 来自有限局部自由 resolution 时，$F$ 是 perfect；右侧的

$$
R\mathcal Hom(F,f^!\mathbb C)
$$

在光滑紧复流形情形中对应

$$
R\mathcal Hom(\mathcal F,\omega_X[n]).
$$

因此 O.7 是附录 J 命题 J.8 的具体复几何模型。

## O.6 练习

**练习 O.1.** 证明有限秩局部自由层 $E$ 满足 $\mathcal Hom(E,-)\cong E^\vee\otimes-$。

**练习 O.2.** 对两项 resolution $E^{-1}\to E^0\to\mathcal F\to0$，写出 $\mathcal Hom(E^\bullet,\omega_X)$ 的两个项和微分方向。

**练习 O.3.** 检查定理 O.7 中 shift $[-n]$ 对同调次数的影响。

**练习 O.4.** 说明为什么非 proper 情形中应把 $Rf_*$ 替换为带支撑条件的 $f_!$。
