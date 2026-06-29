# 附录 Q：Solid 主定理包

## Q.0 目标

本附录把第二卷中分散的 solid 定义、输入定理和形式推论收束为一个主定理包。它的作用不是重证 Scholze 的 solid theory，而是给出版级教材需要的逻辑闭包：

1. 明确 solid 局部化的输入数据；
2. 说明哪些结论是一般稳定范畴论；
3. 证明接受 solid 输入后，solid 环、solid 模和 solid 张量积的形式结构无缺环；
4. 给出使用 solid theory 时必须检查的类型约束。

全篇工作在稳定 \(\infty\)-范畴或其同伦范畴口径下。若读者只使用三角范畴语言，可把 mapping space 改为派生 Hom 复形，把 limit/colimit 改为对应的导出极限/余极限。

## Q.1 Solid 数据

令

$$
\mathcal C=D(\mathbf{CondAb}).
$$

对 profinite 集合 \(S\)，有 Dirac-to-measure 态射

$$
\delta_S:\mathbb Z[\underline S]\longrightarrow \mathbb Z^\square[S],
$$

并记

$$
K_S=\operatorname{cofib}(\delta_S).
$$

令 \(\mathcal N_\square\subset \mathcal C\) 为由所有 \(K_S\) 生成的 localizing subcategory。

**定义 Q.1（solid 局部对象）.** 对象 \(M\in\mathcal C\) 称为 solid，若对所有 profinite \(S\)，

$$
R\operatorname{Hom}_{\mathcal C}(K_S,M)\simeq0.
$$

记 solid 对象的全子范畴为

$$
\mathcal C_\square=D_\square(\mathbb Z).
$$

**命题 Q.2（局部对象的稳定性）.** \(\mathcal C_\square\) 对 shift、fiber、cofiber、小极限和 retract 封闭。

**证明.** 对固定 \(S\)，函子 \(R\operatorname{Hom}(K_S,-)\) 保持极限，并把 fiber/cofiber 序列送到 fiber/cofiber 序列。零对象、shift、fiber、cofiber、小极限和 retract 条件逐项由 \(R\operatorname{Hom}(K_S,-)\simeq0\) 检测。对所有 \(S\) 取交即可。证毕。

**命题 Q.3（核对象对局部对象正交）.** 若 \(N\in\mathcal N_\square\) 且 \(M\in\mathcal C_\square\)，则

$$
R\operatorname{Hom}_{\mathcal C}(N,M)\simeq0.
$$

**证明.** 固定 \(M\)。满足 \(R\operatorname{Hom}(N,M)\simeq0\) 的 \(N\) 组成一个 localizing subcategory：它对 shift 和 cofiber 封闭，因为 \(R\operatorname{Hom}(-,M)\) 把 cofiber 变成 fiber；它对小余极限封闭，因为 \(R\operatorname{Hom}(-,M)\) 把小余极限变成小极限。该类含所有 \(K_S\)，故含由它们生成的 \(\mathcal N_\square\)。证毕。

## Q.2 Solid 输入定理

**输入定理 Q.4（solid 反射局部化）.** 包含函子

$$
i:\mathcal C_\square\hookrightarrow \mathcal C
$$

有左伴随

$$
L^\square:\mathcal C\longrightarrow\mathcal C_\square.
$$

对任意 \(M\in\mathcal C\)，单位态射

$$
\eta_M:M\to iL^\square M
$$

的 cofiber 属于 \(\mathcal N_\square\)，且

$$
\ker L^\square=\mathcal N_\square.
$$

**输入定理 Q.5（solid 核的张量理想性）.** 对任意 \(N\in\mathcal N_\square\) 和 \(X\in\mathcal C\)，有

$$
N\otimes_{\mathbb Z}^L X\in\mathcal N_\square.
$$

**输入定理 Q.6（profinite 测度对象的张量公式）.** 对 profinite 集合 \(S,T\)，存在自然等价

$$
\mathbb Z^\square[S]\otimes_{\mathbb Z}^{L,\square}\mathbb Z^\square[T]
\simeq
\mathbb Z^\square[S\times T].
$$

Q.4-Q.6 是 solid theory 的实质输入。后文只证明这些输入的形式后果。

## Q.3 Solid 张量积

**定义 Q.7（solid 张量）.** 对 \(M,N\in\mathcal C_\square\)，定义

$$
M\otimes_{\mathbb Z}^{L,\square}N
=
L^\square(iM\otimes_{\mathbb Z}^LiN).
$$

**定理 Q.8（solid 闭对称幺半结构）.** 在 Q.4-Q.5 下，\(\mathcal C_\square\) 成为闭对称幺半稳定范畴，单位为

$$
\mathbb Z^\square=L^\square(\mathbb Z),
$$

张量积为 Q.7。包含函子 \(i\) 是 lax symmetric monoidal，局部化函子 \(L^\square\) 是 symmetric monoidal localization。

**证明.** 普通派生张量使 \(\mathcal C\) 成为闭对称幺半稳定范畴。Q.5 说明局部化核 \(\mathcal N_\square\) 是张量理想。由幺半 Bousfield localization 判别，张量积可下降到局部对象子范畴，下降后的张量由

$$
L^\square(M\otimes^LN)
$$

给出。结合律、交换律和单位约束由 \(\mathcal C\) 中对应约束经 \(L^\square\) 传递；核为张量理想保证这些约束与局部等价相容。闭结构由右伴随

$$
R\underline{\operatorname{Hom}}_\square(M,N)
\simeq
R\underline{\operatorname{Hom}}_{\mathcal C}(iM,iN)
$$

再取局部对象判别得到：对 \(P\in\mathcal C_\square\)，

$$
\operatorname{Map}(P,R\underline{\operatorname{Hom}}_{\mathcal C}(iM,iN))
\simeq
\operatorname{Map}(P\otimes^{L,\square}M,N).
$$

故该内部 Hom 表示闭结构。证毕。

**推论 Q.9（solid 张量的局部等价不变性）.** 若 \(f:M\to M'\) 是 solid 局部等价，则对任意 \(N\in\mathcal C\)，

$$
L^\square(M\otimes^LN)\to L^\square(M'\otimes^LN)
$$

为等价。

**证明.** \(\operatorname{cofib}(f)\in\mathcal N_\square\)。其与 \(N\) 张量仍在 \(\mathcal N_\square\)，故局部化后为零。证毕。

## Q.4 Solid 环与模

**定义 Q.10（solid 环）.** solid 环是 \(\mathcal C_\square\) 中的交换代数对象。若 \(R\) 是 solid 环，记

$$
D_\square(R)=\operatorname{Mod}_R(\mathcal C_\square).
$$

**命题 Q.11（solid 模范畴的稳定性）.** \(D_\square(R)\) 是稳定范畴，极限和余极限由底层 \(\mathcal C_\square\) 中极限和余极限计算。

**证明.** 任意稳定 presentable 对称幺半范畴中，交换代数对象的模范畴仍稳定，forgetful functor 保持极限并由自由模函子生成余极限。这里使用 Q.8 给出的对称幺半稳定结构。证毕。

**命题 Q.12（相对 solid 张量）.** 对 \(M,N\in D_\square(R)\)，相对张量积可由 bar construction 计算：

$$
M\otimes_R^{L,\square}N
\simeq
\left|\,M\otimes^{L,\square}R^{\otimes^{L,\square}\bullet}\otimes^{L,\square}N\,\right|.
$$

**证明.** 这是任意幺半稳定范畴中模对象相对张量积的标准构造。Q.8 保证所用张量积存在并与几何实现相容。证毕。

## Q.5 生成元与计算口径

**输入定理 Q.13（solid 投射生成元）.** 对集合 \(I\)，对象

$$
P_I=\prod_I\mathbb Z
$$

的 solid 化或对应的 profinite 测度对象给出 \(\mathbf{Solid}\) 心脏中的投射生成族。对 profinite \(S\)，\(\mathbb Z^\square[S]\) 属于该生成族的闭包。

**命题 Q.14（生成元检验）.** 若 \(F,G:D_\square(\mathbb Z)\to\mathcal D\) 是保持小余极限的 exact functor，且在 Q.13 的生成族上自然变换 \(\alpha:F\to G\) 为等价，则 \(\alpha\) 是等价。

**证明.** 使 \(\alpha_X\) 为等价的对象 \(X\) 构成 localizing subcategory。该子范畴包含生成族，因此等于整个 \(D_\square(\mathbb Z)\)。证毕。

## Q.6 Solid 主闭包定理

**定理 Q.15（Solid 主闭包）。** 接受 Q.4-Q.6 与 Q.13 后，第二卷关于 solid theory 的以下结构在书内闭合：

1. solid 对象的局部对象定义；
2. solidification 的泛性质；
3. solid 张量积与闭对称幺半结构；
4. solid 环与 solid 模；
5. 相对 solid 张量积；
6. profinite 测度对象的基本乘法公式；
7. 生成元检验和类型检查。

**证明.** Q.1-Q.3 证明局部对象和核正交的形式部分。Q.4 给出反射局部化，Q.5 给出幺半下降所需的张量理想性，Q.8-Q.12 证明张量、环和模结构，Q.13-Q.14 给出生成元检验。Q.6 负责 profinite 测度对象的乘法公式。上述列表逐项由这些结果覆盖。证毕。

## Q.7 不能省略的假设

1. 只知道 \(D(\mathbf{CondAb})\) 有普通张量积，不足以推出 solid 张量积；必须使用 Q.5。
2. 只检查有限 profinite \(S\) 不足以推出 solidification；无限 profinite 对象携带真正测度信息。
3. \(P_I=\prod_I\mathbb Z\) 的乘积行为不是普通阿贝尔群张量积行为；它依赖 solid localization。
4. solid completion 不是拓扑阿贝尔群的 Hausdorff completion；它是稳定范畴局部化。

## 练习

1. 证明 Q.3 中的对象类对小余极限封闭。
2. 在 Q.8 中写出 associativity constraint 从普通派生张量下降到 solid 张量的步骤。
3. 证明 Q.9 的 cofiber 计算。
4. 对有限离散 \(S,T\)，检查 Q.6 与普通自由阿贝尔群张量公式相容。

