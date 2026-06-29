# 附录 L：闭幺半局部化与内部 Hom

## L.0 目标

附录 K 证明了核为张量理想时，Bousfield 局部化继承对称幺半结构。本附录继续处理闭结构：

$$
\mathcal Hom(X,Y).
$$

这部分用于理解 solid/analytic 派生范畴中内部 Hom、投影公式和 $f^!$ 公式的类型边界。

本附录假设 $\mathcal C$ 是可展示稳定闭对称幺半范畴，张量积分别保持小余极限，并有内部 Hom

$$
\mathcal Hom_{\mathcal C}(-,-)
$$

满足

$$
\operatorname{Map}_{\mathcal C}(A\otimes B,C)
\simeq
\operatorname{Map}_{\mathcal C}(A,\mathcal Hom_{\mathcal C}(B,C)).
$$

设 $L:\mathcal C\to\mathcal C_{\operatorname{loc}}$ 是附录 K 中的幺半局部化，包含函子记为 $i$。

## L.1 局部对象的内部 Hom

**命题 L.1.** 若 $X,Y$ 是局部对象，则

$$
\mathcal Hom_{\mathcal C}(X,Y)
$$

也是局部对象。

**证明.** 设 $s:A\to B$ 是被局部化倒置的生成态射。需要证明

$$
\operatorname{Map}(B,\mathcal Hom(X,Y))
\to
\operatorname{Map}(A,\mathcal Hom(X,Y))
$$

是等价。由闭幺半伴随，该映射等同于

$$
\operatorname{Map}(B\otimes X,Y)
\to
\operatorname{Map}(A\otimes X,Y).
$$

附录 K 的幺半局部化判别说明 $s\otimes X$ 是局部等价。由于 $Y$ 是局部对象，上式为等价。证毕。

**推论 L.2.** 局部范畴 $\mathcal C_{\operatorname{loc}}$ 是闭对称幺半范畴，内部 Hom 可取为

$$
\mathcal Hom_{\operatorname{loc}}(X,Y)
=
\mathcal Hom_{\mathcal C}(iX,iY).
$$

**证明.** 命题 L.1 说明右侧仍在局部子范畴中。对局部对象 $Z,X,Y$，

$$
\operatorname{Map}_{\operatorname{loc}}(Z\otimes_{\operatorname{loc}}X,Y)
\simeq
\operatorname{Map}_{\mathcal C}(L(iZ\otimes iX),iY)
\simeq
\operatorname{Map}_{\mathcal C}(iZ\otimes iX,iY)
$$

因为 $iY$ 局部。闭伴随给

$$
\operatorname{Map}_{\mathcal C}(iZ,\mathcal Hom_{\mathcal C}(iX,iY))
\simeq
\operatorname{Map}_{\operatorname{loc}}(Z,\mathcal Hom_{\mathcal C}(iX,iY)).
$$

这正是内部 Hom 的泛性质。证毕。

## L.2 局部化与内部 Hom 的自然映射

对任意 $M,N\in\mathcal C$，有自然态射

$$
L\mathcal Hom_{\mathcal C}(M,N)
\to
\mathcal Hom_{\operatorname{loc}}(LM,LN).
$$

**构造 L.3.** 单位 $M\to iLM$ 和 $N\to iLN$ 给出映射

$$
\mathcal Hom_{\mathcal C}(iLM,iLN)
\to
\mathcal Hom_{\mathcal C}(M,iLN).
$$

同时 $N\to iLN$ 给出

$$
\mathcal Hom_{\mathcal C}(M,N)\to
\mathcal Hom_{\mathcal C}(M,iLN).
$$

经局部化并使用推论 L.2，可得到比较态射。该态射不总是等价；等价需要 $M$ 满足紧性、dualizability 或局部化相容条件。

**命题 L.4（dualizable 情形）.** 若 $M$ 是 dualizable 对象，则自然态射

$$
L\mathcal Hom_{\mathcal C}(M,N)
\to
\mathcal Hom_{\operatorname{loc}}(LM,LN)
$$

是等价。

**证明.** dualizable 给出

$$
\mathcal Hom_{\mathcal C}(M,N)\simeq M^\vee\otimes N.
$$

局部化是幺半的，因此

$$
L(M^\vee\otimes N)
\simeq
L(M^\vee)\otimes_{\operatorname{loc}}LN.
$$

又 $LM$ 在局部范畴中 dualizable，且对偶为 $L(M^\vee)$。于是右侧等于

$$
\mathcal Hom_{\operatorname{loc}}(LM,LN).
$$

证毕。

## L.3 闭结构与投影公式

设

$$
F:\mathcal C\rightleftarrows\mathcal D:G
$$

是闭对称幺半稳定范畴之间的伴随对，$F$ 是对称幺半左伴随。投影公式是自然态射

$$
F(X\otimes GY)\to F(X)\otimes Y
$$

为等价。

**命题 L.5.** 若 $Y=FW$ 属于 $F$ 的本质像，则有内部 Hom 比较公式

$$
G\mathcal Hom_{\mathcal D}(FW,Z)
\simeq
\mathcal Hom_{\mathcal C}(W,GZ)
$$

自然成立。

**证明.** 对任意 $X\in\mathcal C$，

$$
\operatorname{Map}_{\mathcal C}(X,G\mathcal Hom_{\mathcal D}(FW,Z))
\simeq
\operatorname{Map}_{\mathcal D}(FX,\mathcal Hom_{\mathcal D}(FW,Z))
$$

由伴随。闭结构给

$$
\operatorname{Map}_{\mathcal D}(FX\otimes FW,Z).
$$

因为 $F$ 是对称幺半左伴随，

$$
FX\otimes FW\simeq F(X\otimes W).
$$

于是上式同构于

$$
\operatorname{Map}_{\mathcal C}(X\otimes W,GZ)
\simeq
\operatorname{Map}_{\mathcal C}(X,\mathcal Hom_{\mathcal C}(W,GZ)).
$$

由 Yoneda 得结论。证毕。

**边界 L.6.** 若 $Y$ 不在 $F$ 的本质像中，或者没有可用的 duality/perfectness 假设，右伴随不必把 $\mathcal Hom_{\mathcal D}(Y,Z)$ 改写为 $\mathcal C$ 中的内部 Hom。几何中的 $f^!$ 公式必须逐条列出所需的 compactness、perfectness、properness 或有限 Tor-amplitude 假设。

## L.4 solid/analytic 中的使用规则

在 solid 或 analytic 派生范畴中，内部 Hom 公式使用时必须说明：

1. 所在闭幺半范畴是普通凝聚派生范畴、solid 派生范畴还是 analytic 模范畴。
2. 局部化是否为幺半局部化。
3. 被移入内部 Hom 的对象是否 dualizable、compact 或 perfect。
4. 比较态射是定义、形式等价，还是深层定理。

若这些条件不写清，公式

$$
R\mathcal Hom(M,N)
$$

可能在不同范畴中表示不同对象。

## 练习

1. 证明命题 L.1 中只需检查生成态射 $s\in\Sigma$。
2. 在有理化局部化中计算 $\mathcal Hom_{\mathbb Q}(M\otimes\mathbb Q,N\otimes\mathbb Q)$。
3. 给出一个非 dualizable 对象 $M$，说明 $\mathcal Hom(M,-)$ 不一定与小余极限相容。
4. 解释为什么 perfect complex 在许多几何范畴中满足命题 L.4 的假设。
