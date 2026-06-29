# 附录 K：幺半 Bousfield 局部化细节

## K.0 目标

卷二反复使用如下形式：

$$
L(M\otimes N),\qquad
L(M\otimes_A^LN),\qquad
L^\square(M\otimes^LN).
$$

这些表达只有在局部化与张量积相容时才有确定含义。本附录把所需的范畴论判别写成独立证明。深层输入不是形式范畴论，而是具体理论中“核为张量理想”或“指定 cone 张量后仍为局部零对象”的验证。

本附录固定一个可展示稳定对称幺半 $\infty$-范畴 $\mathcal C$，张量积分别保持小余极限。若读者使用模型范畴，可把本附录理解为其同伦范畴和导出映射空间上的陈述。

## K.1 由一组态射生成的局部化

设 $\Sigma$ 是 $\mathcal C$ 中的一组态射。

**定义 K.1.** 对象 $X$ 称为 $\Sigma$-局部，如果对每个 $s:A\to B$ 属于 $\Sigma$，

$$
\operatorname{Map}(B,X)\to\operatorname{Map}(A,X)
$$

是等价。态射 $f:M\to N$ 称为 $\Sigma$-等价，如果它对所有 $\Sigma$-局部对象诱导映射空间等价。

**输入定理 K.2（可展示局部化存在性）.** 对可展示 $\infty$-范畴 $\mathcal C$ 和一组态射 $\Sigma$，存在反射局部化

$$
L_\Sigma:\mathcal C\rightleftarrows\mathcal C_{\Sigma\operatorname{-loc}}:i
$$

使 $\mathcal C_{\Sigma\operatorname{-loc}}$ 正是 $\Sigma$-局部对象构成的全子范畴。单位 $M\to iL_\Sigma M$ 是 $\Sigma$-等价。

**命题 K.3.** 若 $\mathcal C$ 稳定，则 $\mathcal C_{\Sigma\operatorname{-loc}}$ 对有限极限、有限余极限和 shift 封闭，且 $L_\Sigma$ 是 exact 函子。

**证明.** 对每个 $s:A\to B$，函子

$$
X\mapsto \operatorname{fib}\bigl(\operatorname{Map}(B,X)\to\operatorname{Map}(A,X)\bigr)
$$

把有限极限变成有限极限。$\Sigma$-局部对象是这些 fiber 为 contractible 的对象，因此对有限极限封闭。稳定范畴中有限极限与有限余极限互相决定，所以也对有限余极限封闭。shift 封闭由

$$
\operatorname{Map}(B,X[1])\simeq \operatorname{Map}(B[-1],X)
$$

和 cofiber/fiber 的稳定性推出。

反射到稳定全子范畴的左伴随保持有限余极限；稳定范畴中 exact 等价于保持有限余极限和零对象。证毕。

## K.2 核与局部零对象

记

$$
\mathcal N_\Sigma=\{M\in\mathcal C\mid L_\Sigma M\simeq0\}.
$$

若 $s:A\to B$，记 $K_s=\operatorname{cofib}(s)$。

**命题 K.4.** $\mathcal N_\Sigma$ 是 localizing subcategory，且包含每个 $K_s$。并且态射 $f:M\to N$ 是 $\Sigma$-等价，当且仅当

$$
\operatorname{cofib}(f)\in\mathcal N_\Sigma.
$$

**证明.** $L_\Sigma$ 是 exact 且保持小余极限，因此其零对象原像对 shift、cofiber 和小余极限封闭。

对 $s:A\to B$，单位使 $L_\Sigma s$ 成为等价，所以

$$
L_\Sigma K_s\simeq\operatorname{cofib}(L_\Sigma A\to L_\Sigma B)\simeq0.
$$

最后，$f$ 是 $\Sigma$-等价当且仅当 $L_\Sigma f$ 是等价；在稳定范畴中，这等价于 $L_\Sigma\operatorname{cofib}(f)\simeq0$。证毕。

**命题 K.5.** $\mathcal N_\Sigma$ 是包含所有 $K_s$ 的最小 localizing subcategory 的局部化闭包。更精确地，若 $\mathcal T$ 是包含所有 $K_s$ 的 localizing subcategory，且 Verdier 商反射存在并以 $\mathcal T$ 为核，则该反射倒置 $\Sigma$。

**证明.** 第一部分由命题 K.4 给出包含关系。第二部分中，若 $K_s\in\mathcal T$，则在商中 $s$ 的 cofiber 为零，所以 $s$ 变为等价。因此商反射倒置所有 $\Sigma$。它的局部对象必满足 K.1 中的映射空间判别。证毕。

## K.3 幺半局部化判别

**定义 K.6.** $\mathcal N_\Sigma$ 称为张量理想，如果

$$
N\in\mathcal N_\Sigma,\ X\in\mathcal C
\quad\Longrightarrow\quad
N\otimes X\in\mathcal N_\Sigma.
$$

**定理 K.7（核为张量理想推出幺半局部化）.** 若 $\mathcal N_\Sigma$ 是张量理想，则 $\mathcal C_{\Sigma\operatorname{-loc}}$ 继承对称幺半结构，并且

$$
X\otimes_{\operatorname{loc}}Y
=
L_\Sigma(iX\otimes iY).
$$

局部化函子 $L_\Sigma:\mathcal C\to\mathcal C_{\Sigma\operatorname{-loc}}$ 是对称幺半函子。

**证明.** 先证明张量积能下降到局部化。若 $f:M\to N$ 是 $\Sigma$-等价，则命题 K.4 给出 $\operatorname{cofib}(f)\in\mathcal N_\Sigma$。因为张量保持 cofiber，

$$
\operatorname{cofib}(f\otimes X)
\simeq
\operatorname{cofib}(f)\otimes X
\in\mathcal N_\Sigma.
$$

故 $f\otimes X$ 仍是 $\Sigma$-等价。

因此表达式 $L_\Sigma(M\otimes N)$ 只依赖 $LM,LN$。若 $X,Y$ 是局部对象，定义

$$
X\otimes_{\operatorname{loc}}Y=L_\Sigma(X\otimes Y).
$$

结合律、交换律和单位约束由 $\mathcal C$ 中的相应约束经 $L_\Sigma$ 得到。coherence 图在 $\mathcal C$ 中交换，函子作用后仍交换。单位为 $L_\Sigma(\mathbb 1_{\mathcal C})$。

对任意 $M,N$，自然映射

$$
L_\Sigma M\otimes_{\operatorname{loc}} L_\Sigma N
\to
L_\Sigma(M\otimes N)
$$

由单位 $M\to L_\Sigma M$、$N\to L_\Sigma N$ 诱导；前述张量保持局部等价说明它是等价。证毕。

**推论 K.8（生成态射形式的判别）.** 若对每个 $s:A\to B$ 和每个 $X\in\mathcal C$，态射

$$
s\otimes X:A\otimes X\to B\otimes X
$$

是 $\Sigma$-等价，则 $\mathcal N_\Sigma$ 是张量理想。

**证明.** 每个 $K_s\otimes X\simeq \operatorname{cofib}(s\otimes X)$ 属于 $\mathcal N_\Sigma$。因为 $-\otimes X$ 保持 shift、cofiber 和小余极限，由 $K_s$ 生成的 localizing subcategory 张量后仍落入 $\mathcal N_\Sigma$。再用命题 K.5 的局部化闭包描述，得到整个核为张量理想。证毕。

## K.4 交换代数与模范畴

设 $A$ 是 $\mathcal C$ 中的交换代数对象。

**命题 K.9.** 在定理 K.7 的假设下，$L_\Sigma A$ 是局部范畴中的交换代数对象。若 $M$ 是 $A$-模，则 $L_\Sigma M$ 是 $L_\Sigma A$-模。

**证明.** 交换代数结构由乘法、单位和有限 coherence 图给出。对称幺半函子 $L_\Sigma$ 把这些数据送入局部范畴，并保持 coherence 图。模结构同理。证毕。

**定理 K.10（相对张量积与局部化交换）.** 若 $M,N$ 是 $A$-模，则自然等价

$$
L_\Sigma(M\otimes_A^LN)
\simeq
L_\Sigma M\otimes_{L_\Sigma A}^{L,\operatorname{loc}}L_\Sigma N
$$

成立。

**证明.** 相对派生张量积由双边 bar construction 给出：

$$
M\otimes_A^LN
\simeq
\left|M\otimes A^{\otimes\bullet}\otimes N\right|.
$$

$L_\Sigma$ 保持几何实现，且由定理 K.7 与张量积相容，所以

$$
L_\Sigma\left|M\otimes A^{\otimes\bullet}\otimes N\right|
\simeq
\left|L_\Sigma M\otimes_{\operatorname{loc}}
(L_\Sigma A)^{\otimes_{\operatorname{loc}}\bullet}
\otimes_{\operatorname{loc}}L_\Sigma N\right|.
$$

右侧即局部范畴中的相对 bar construction。证毕。

## K.5 solid 与 analytic 的代入方式

在 solid 理论中，$\Sigma$ 可取为 Dirac/free 对象到 solid 测度对象的态射族，其 cofiber 记为 $K_S$。形式证明只要求如下输入：

**输入定理 K.11（solid 核的张量理想性）.** 对所有 profinite $S$ 和所有凝聚阿贝尔群复形 $X$，

$$
K_S\otimes^L X
$$

在 solid localization 后为零。

接受 K.11 后，定理 K.7-K.10 给出：

$$
M\otimes^{L,\square}N
\simeq
L^\square(M\otimes^LN),
$$

以及 solid 环上模的相对张量积公式。

analytic ring 的情形相同，但 $\Sigma$ 由 analytic ring 指定的测试 cone 给出。真正困难在于验证这些 cone 在张量后仍属于局部化核；一旦该点成立，其余是本附录的形式范畴论。

## K.6 失败模式

若核不是张量理想，则以下表达式没有良定义：

$$
LM\otimes_{\operatorname{loc}}LN=L(M\otimes N).
$$

原因是 $M$ 可被局部等价 $M'$ 替换，但 $M\otimes N\to M'\otimes N$ 未必仍为局部等价。于是右侧依赖代表元，而不是只依赖局部对象 $LM,LN$。

这就是第二卷中反复强调“张量理想性不是装饰条件”的原因。

## 练习

1. 证明若 $\mathcal N_\Sigma$ 是张量理想，则局部对象 $X,Y$ 的内部 Hom 若存在，满足合适的局部性判别。
2. 在普通链复形范畴中，取有理化局部化 $L(-)=(-)\otimes\mathbb Q$。验证其核为张量理想，并写出 K.10 的具体形式。
3. 构造一个反射局部化的抽象例子，使核不是张量理想，并指出 $L(M\otimes N)$ 不能下降的具体位置。
4. 对 solid localization，说明为什么只检查有限 profinite $S$ 不足以推出 K.11。
