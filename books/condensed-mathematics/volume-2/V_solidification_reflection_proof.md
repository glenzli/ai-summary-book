# 附录 V：Solidification 反射存在性的证明模块

## V.0 目标

附录 Q 把 solidification 作为输入定理 Q.4。本附录开始把该输入拆成可证明的模块。目标是证明如下结论的形式部分：

> 由 Dirac-to-measure cone \(K_S\) 生成的局部对象构成反射局部子范畴，反射函子记为 \(L^\square\)。

完整证明包含两类成分：

1. 一般可展示稳定范畴中的集合生成局部化；
2. 凝聚阿贝尔群派生范畴中 \(K_S\) 的集合性、可访问性和小性控制。

本附录把第 1 类完整写出，把第 2 类化为可检查的大小和生成假设。

## V.1 集合生成的局部对象

令 \(\mathcal C\) 是可展示稳定 \(\infty\)-范畴，\(\Sigma\) 是一组态射。对 \(f:X\to Y\in\Sigma\)，记

$$
K_f=\operatorname{cofib}(f).
$$

**定义 V.1（\(\Sigma\)-局部对象）.** 对象 \(Z\in\mathcal C\) 称为 \(\Sigma\)-局部，若对每个 \(f:X\to Y\in\Sigma\)，诱导映射

$$
\operatorname{Map}(Y,Z)\to\operatorname{Map}(X,Z)
$$

是等价。

在稳定范畴中，这等价于

$$
\operatorname{Map}(K_f,Z)\simeq *.
$$

**证明.** cofiber sequence

$$
X\to Y\to K_f
$$

经 \(\operatorname{Map}(-,Z)\) 变成 fiber sequence

$$
\operatorname{Map}(K_f,Z)\to \operatorname{Map}(Y,Z)\to\operatorname{Map}(X,Z).
$$

中间箭头是等价当且仅当前一项可缩。证毕。

## V.2 反射局部化的一般定理

**输入定理 V.2（可展示范畴的集合生成局部化）.** 设 \(\mathcal C\) 是可展示 \(\infty\)-范畴，\(\Sigma\) 是一组态射。则 \(\Sigma\)-局部对象构成可展示反射子范畴

$$
\mathcal C_\Sigma\subset\mathcal C,
$$

包含函子有左伴随

$$
L_\Sigma:\mathcal C\to\mathcal C_\Sigma.
$$

若 \(\mathcal C\) 稳定，则 \(\mathcal C_\Sigma\) 稳定，且 \(L_\Sigma\) exact。

**说明.** 这是 presentable \(\infty\)-category 的一般 localization theorem。本书不重证高阶范畴论的小对象论证；后续只使用其稳定范畴后果。

**命题 V.3（稳定性后果）.** 在 V.2 的稳定情形下，\(\mathcal C_\Sigma\) 对 shift、fiber、cofiber、小极限和 retract 封闭。

**证明.** \(\Sigma\)-局部性由所有 \(\operatorname{Map}(K_f,-)\) 可缩检测。mapping functor 保持极限，并把 fiber/cofiber 序列送到 fiber/cofiber 序列。故这些操作保持局部性。证毕。

**命题 V.4（核的 localizing 性）.** \(\ker L_\Sigma\) 是 \(\mathcal C\) 的 localizing subcategory。

**证明.** \(L_\Sigma\) 是 exact 左伴随，因此保持 shift、cofiber 和小余极限。若 \(L_\Sigma X\simeq0\)，则 shift 后仍为零；若 \(X\to Y\to Z\) 是 cofiber sequence 且其中两项在核中，则第三项局部化后也为零；若 \(X_i\) 都在核中，则

$$
L_\Sigma(\operatorname*{colim}_iX_i)
\simeq
\operatorname*{colim}_iL_\Sigma X_i
\simeq0.
$$

证毕。

## V.3 核由 cones 生成

令 \(\mathcal N_\Sigma\) 为由 \(K_f\) 生成的 localizing subcategory。

**命题 V.5.** 有包含

$$
\mathcal N_\Sigma\subseteq\ker L_\Sigma.
$$

**证明.** 对 \(f:X\to Y\)，局部化后 \(L_\Sigma f\) 是等价，所以

$$
L_\Sigma K_f\simeq0.
$$

因此所有 \(K_f\) 在核中。由 V.4，核是 localizing subcategory，故包含由 \(K_f\) 生成的 \(\mathcal N_\Sigma\)。证毕。

**命题 V.6（核等于生成核的判别）.** 若对任意 \(M\in\ker L_\Sigma\) 和任意 \(\Sigma\)-局部对象 \(Z\)，有

$$
\operatorname{Map}(M,Z)\simeq *,
$$

且 \(\mathcal N_\Sigma\) 是所有对 \(\Sigma\)-局部对象左正交的对象，则

$$
\ker L_\Sigma=\mathcal N_\Sigma.
$$

**证明.** V.5 已给出 \(\mathcal N_\Sigma\subseteq\ker L_\Sigma\)。反向，若 \(M\in\ker L_\Sigma\)，则对局部 \(Z\)，由伴随

$$
\operatorname{Map}(M,Z)\simeq
\operatorname{Map}(L_\Sigma M,Z)\simeq *.
$$

按假设 \(M\in\mathcal N_\Sigma\)。证毕。

## V.4 应用于 solidification

取

$$
\mathcal C=D(\mathbf{CondAb}),
$$

\(\Sigma_\square\) 为所有 profinite \(S\) 的 Dirac-to-measure map

$$
\delta_S:\mathbb Z[\underline S]\to\mathbb Z^\square[S].
$$

**大小约定 V.7.** 固定 universe 后，只取一个小骨架中的 profinite \(S\)。因此 \(\Sigma_\square\) 是一组态射。

**输入定理 V.8（可展示性）.** \(D(\mathbf{CondAb})\) 是可展示稳定 \(\infty\)-范畴。

**定理 V.9（solidification 的反射存在性，范畴论部分）。** 在 V.7-V.8 下，\(\Sigma_\square\)-局部对象构成反射稳定子范畴

$$
D(\mathbf{CondAb})_{\Sigma_\square}\subset D(\mathbf{CondAb}),
$$

包含函子有 exact 左伴随

$$
L_{\Sigma_\square}:D(\mathbf{CondAb})\to D(\mathbf{CondAb})_{\Sigma_\square}.
$$

**证明.** 由 V.2 直接应用于 \(\mathcal C=D(\mathbf{CondAb})\) 和 \(\Sigma=\Sigma_\square\)。证毕。

**命题 V.10（与 \(K_S\)-正交定义一致）。** \(M\) 是 \(\Sigma_\square\)-局部对象，当且仅当对所有 profinite \(S\)，

$$
R\operatorname{Hom}(K_S,M)\simeq0.
$$

**证明.** 由 V.1，对每个 \(\delta_S\) 局部等价条件等价于其 cofiber \(K_S\) 对 \(M\) 正交。证毕。

## V.5 与 Scholze solidification 的识别

**输入定理 V.11（Scholze 识别定理）。** V.9 构造的局部范畴与 Scholze 的 solid 派生范畴 \(D_\square(\mathbb Z)\) 一致；局部化函子 \(L_{\Sigma_\square}\) 与 solidification \(L^\square\) 一致。

**书内已证部分.** V.9-V.10 已证明：存在一个由 Dirac-to-measure cone 定义的反射局部范畴。

**仍属输入的部分.** 该局部范畴具有 Scholze solid theory 所需的额外计算性质，例如 profinite 测度对象公式、投射生成族和与 solid tensor 的相容性。

## V.6 本附录闭包

**定理 V.12（solid 反射存在性的分解）。** solidification 反射存在性可拆成：

1. \(D(\mathbf{CondAb})\) 可展示稳定；
2. Dirac-to-measure maps 构成一组态射；
3. 集合生成局部化存在；
4. 局部对象等价于 \(K_S\)-正交对象；
5. Scholze solid 范畴与该局部范畴的识别。

其中 2、4 在书内证明；1、3 是一般高阶范畴论输入；5 是 Scholze solid theory 输入。

**证明.** V.7 给出 2，V.10 给出 4；V.8 和 V.2 给出 1、3；V.11 给出 5。证毕。

## 练习

1. 证明 V.3 中对 retract 封闭的细节。
2. 在稳定范畴中证明 \(f\) 被局部化为等价当且仅当 \(\operatorname{cofib}(f)\) 被局部化为零。
3. 说明为什么必须固定 profinite 空间的小骨架。
4. 解释 V.9 还不足以推出 solid tensor product。

