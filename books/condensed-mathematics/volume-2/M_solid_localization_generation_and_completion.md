# 附录 M：Solid localization 的生成、反射与完备化口径

## M.0 目标

solid 对象在正文中以输入定理出现。本附录把 solidification 的形式结构写成可逐步检查的 localization 语言，说明哪些部分是一般稳定范畴论，哪些部分是 Scholze 的实质输入。

本附录工作在

$$
D(\mathbf{CondAb})
$$

中。对 profinite 集合 $S$，记

$$
\mathbb Z[\underline S]\to\mathbb Z^\square[S]
$$

为 Dirac-to-solid-measure 映射，并记其 cofiber 为

$$
K_S.
$$

## M.1 Solid 局部对象

**定义 M.1.** 对象 $C\in D(\mathbf{CondAb})$ 称为 solid，如果对所有 profinite $S$，

$$
R\operatorname{Hom}(K_S,C)\simeq0.
$$

记所有 solid 对象构成的全子范畴为

$$
D_\square(\mathbb Z)\subset D(\mathbf{CondAb}).
$$

**命题 M.2.** $D_\square(\mathbb Z)$ 对 shift、fiber、cofiber 和小极限封闭。

**证明.** 条件由函子 $R\operatorname{Hom}(K_S,-)$ 检测。该函子保持极限，并把 fiber/cofiber sequence 送到 fiber/cofiber sequence。零对象、shift 和有限极限条件逐项检查即可。证毕。

## M.2 由 cone 生成的核

令 $\mathcal N_\square$ 为由所有 $K_S$ 生成的 localizing subcategory，即包含 $K_S$ 且对 shift、cofiber 和小余极限封闭的最小全子范畴。

**命题 M.3.** 若 $C$ 是 solid，则对任意 $N\in\mathcal N_\square$，

$$
R\operatorname{Hom}(N,C)\simeq0.
$$

**证明.** 固定 solid $C$。使 $R\operatorname{Hom}(N,C)\simeq0$ 的对象类对 shift、cofiber 和小余极限封闭，因为 $R\operatorname{Hom}(-,C)$ 把小余极限变成小极限，并在稳定范畴中保持 fiber/cofiber 序列。该对象类含所有 $K_S$，故含 $\mathcal N_\square$。证毕。

**输入定理 M.4（solid 反射存在性）.** 包含函子

$$
i:D_\square(\mathbb Z)\hookrightarrow D(\mathbf{CondAb})
$$

有左伴随

$$
L^\square:D(\mathbf{CondAb})\to D_\square(\mathbb Z).
$$

单位映射 $M\to iL^\square M$ 的 cofiber 属于 $\mathcal N_\square$。

**定义 M.5.** $L^\square M$ 称为 $M$ 的 solidification。

## M.3 泛性质

**命题 M.6.** 对 $M\in D(\mathbf{CondAb})$ 和 solid 对象 $C$，自然映射

$$
R\operatorname{Hom}(L^\square M,C)
\to
R\operatorname{Hom}(M,C)
$$

是等价。

**证明.** 这是 $L^\square\dashv i$ 的伴随泛性质。证毕。

**命题 M.7.** 态射 $f:M\to N$ 在 solidification 后成为等价，当且仅当

$$
\operatorname{cofib}(f)\in\mathcal N_\square.
$$

**证明.** 由 $L^\square$ exact，$L^\square f$ 是等价当且仅当 $L^\square\operatorname{cofib}(f)\simeq0$。核正是 $\mathcal N_\square$ 的局部化闭包；在 M.4 的输入下，单位 cofiber 与生成核给出该判别。证毕。

## M.4 张量理想输入

**输入定理 M.8（solid 核为张量理想）.** 若 $N\in\mathcal N_\square$ 且 $X\in D(\mathbf{CondAb})$，则

$$
N\otimes^L X\in\mathcal N_\square.
$$

等价地，对所有 profinite $S$ 和所有 $X$，

$$
K_S\otimes^LX
$$

在 solid localization 后为零。

**推论 M.9.** $D_\square(\mathbb Z)$ 继承闭对称幺半结构，且

$$
M\otimes^{L,\square}N
\simeq
L^\square(M\otimes^LN).
$$

**证明.** 这是附录 K 的幺半 Bousfield localization 判别在 $\mathcal N_\square$ 上的应用。证毕。

## M.5 完备化口径

solidification 可被理解为对所有 profinite 测试对象强制满足 measure-compatible descent 的完备化：

$$
\mathbb Z[\underline S]\to\mathbb Z^\square[S]
$$

被倒置后，普通 free 凝聚对象与 solid measure 对象在局部范畴中不可区分。

**边界 M.10.** 这种“完备化”不是拓扑阿贝尔群意义下的 Hausdorff completion。它是稳定范畴中的 localization，由一族 cone 生成；其效果由 $R\operatorname{Hom}(K_S,-)$ 的消没检测。

## M.6 使用清单

每次使用 solidification，需要标明：

1. 对象位于 $D(\mathbf{CondAb})$ 还是 $D_\square(\mathbb Z)$。
2. 是否使用了反射存在性 M.4。
3. 是否使用了张量理想输入 M.8。
4. 张量积是普通 $\otimes^L$ 还是 solid $\otimes^{L,\square}$。
5. 内部 Hom 是否在闭 solid 范畴中计算。

## 练习

1. 证明 M.2 中对 cofiber 封闭的细节。
2. 证明 M.3 中小余极限被 $R\operatorname{Hom}(-,C)$ 变为小极限。
3. 解释为什么 M.8 不能从普通张量积的右正合性推出。
4. 对有限离散 $S$，说明 $K_S\simeq0$。
