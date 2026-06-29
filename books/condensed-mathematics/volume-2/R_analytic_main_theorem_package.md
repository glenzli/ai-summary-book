# 附录 R：Analytic 主定理包

## R.0 目标

本附录把 analytic rings、analyticization、rational localization 和 descent 收束成一个主定理包。它补足第二卷正文中最容易松散的地方：analytic structure 不是“拓扑完备化”的类比，而是一套由测度对象、局部化和下降共同控制的派生范畴结构。

本附录固定一个凝聚交换环 \(A\)，并在

$$
D(A)
$$

中工作。

## R.1 Pre-analytic 数据

**定义 R.1（测度数据）.** 一个 pre-analytic datum 由以下数据组成：

1. 一类测试对象 \(\mathcal T\)；
2. 对每个 \(S\in\mathcal T\)，一个 \(A\)-模对象 \(\mathcal M[S]\)；
3. 自然的 Dirac 态射
   $$
   \delta_S:A[\underline S]\to\mathcal M[S].
   $$

记

$$
K_S^{\mathcal M}=\operatorname{cofib}(\delta_S).
$$

**定义 R.2（\(\mathcal M\)-局部对象）.** 对象 \(C\in D(A)\) 称为 \(\mathcal M\)-analytic，若对所有 \(S\in\mathcal T\)，

$$
R\operatorname{Hom}_A(K_S^{\mathcal M},C)\simeq0.
$$

记所有 \(\mathcal M\)-analytic 对象构成的全子范畴为

$$
D(A,\mathcal M)\subset D(A).
$$

**命题 R.3（analytic 局部对象的稳定性）.** \(D(A,\mathcal M)\) 对 shift、fiber、cofiber、小极限和 retract 封闭。

**证明.** 与 Q.2 相同，由所有 \(R\operatorname{Hom}_A(K_S^{\mathcal M},-)\) 同时检测。证毕。

## R.2 Analytic ring 输入

**输入定理 R.4（analytic 反射局部化）.** 若 \((A,\mathcal M)\) 是 Scholze 意义下的 analytic ring，则包含函子

$$
i:D(A,\mathcal M)\hookrightarrow D(A)
$$

有左伴随

$$
L_{(A,\mathcal M)}:D(A)\to D(A,\mathcal M).
$$

其核由 \(K_S^{\mathcal M}\) 生成。

**输入定理 R.5（analytic 核的张量理想性）.** \(L_{(A,\mathcal M)}\) 的核是 \(\otimes_A^L\)-张量理想。

**输入定理 R.6（analytic 闭结构）.** \(D(A,\mathcal M)\) 带有与 \(D(A)\) 相容的闭对称幺半结构。

R.6 可由 R.4-R.5 经幺半局部化推出；在实际引用中，Scholze 的 analytic ring 定理通常直接给出 R.4-R.6 的组合形式。

## R.3 Analyticization

**定义 R.7（analyticization）.** 对 \(M\in D(A)\)，称

$$
M^{\mathrm{an}}=L_{(A,\mathcal M)}M
$$

为 \(M\) 的 analyticization。

**命题 R.8（analyticization 泛性质）.** 若 \(N\in D(A,\mathcal M)\)，则自然映射

$$
R\operatorname{Hom}_{D(A,\mathcal M)}(M^{\mathrm{an}},N)
\to
R\operatorname{Hom}_{D(A)}(M,iN)
$$

为等价。

**证明.** 这是 \(L_{(A,\mathcal M)}\dashv i\) 的伴随泛性质。证毕。

**定理 R.9（analytic 张量）.** 在 R.4-R.5 下，\(D(A,\mathcal M)\) 的张量积由

$$
M\otimes_A^{L,\mathcal M}N
=
L_{(A,\mathcal M)}(iM\otimes_A^LiN)
$$

给出，并使 \(L_{(A,\mathcal M)}\) 成为 symmetric monoidal localization。

**证明.** 由 R.5 和幺半 Bousfield localization 判别。证毕。

**命题 R.10（解析化与相对张量）.** 若 \(B\) 是 \(D(A,\mathcal M)\) 中的交换代数对象，\(M,N\in D(B)\)，则

$$
L(M\otimes_B^LN)
\simeq
LM\otimes_{LB}^{L,\mathcal M}LN
$$

在 analytic 局部范畴中成立。

**证明.** 用 bar construction 表示相对张量积。\(L\) 保持余极限并由 R.9 与张量积相容，逐项局部化后得到右侧。证毕。

## R.4 Rational localization

令 \((A,A^+)\) 是离散 Huber pair，设 \(X=\operatorname{Spa}(A,A^+)\)。

**输入定理 R.11（Huber pair 的 analytic ring）.** \((A,A^+)\) 给出 analytic ring

$$
(A,A^+)^\square=(A,\mathcal M_{A,A^+}).
$$

**输入定理 R.12（rational localization）.** 若 \(U\subset X\) 是 rational subset，则存在 analytic ring

$$
(A_U,\mathcal M_U)
$$

以及 restriction functor

$$
\rho_U^\ast:D(A,\mathcal M)\to D(A_U,\mathcal M_U)
$$

与代数 rational localization 相容。

**输入定理 R.13（rational Čech descent）.** 若 \(X=\bigcup_{i=1}^n U_i\) 是有限 rational cover，则自然函子

$$
D(A,\mathcal M)\to
\operatorname{Tot}\bigl(D(A_{U_\bullet},\mathcal M_{U_\bullet})\bigr)
$$

为等价。

## R.5 Descent 的形式后果

**命题 R.14（对象粘合）.** 在 R.13 下，全局 analytic 模等价于 rational Čech nerve 上的 descent datum。

**证明.** R.13 给出全局范畴与 cosimplicial diagram 的 totalization 等价。totalization 的对象正是每个 \(U_i\) 上的对象、交叠上的等价和所有高阶 cocycle 相容。证毕。

**命题 R.15（态射空间下降）.** 在 R.13 下，对 \(M,N\in D(A,\mathcal M)\)，有

$$
\operatorname{Map}(M,N)
\simeq
\operatorname{Tot}\operatorname{Map}(M|_{U_\bullet},N|_{U_\bullet}).
$$

若覆盖的二截断已决定映射空间，则得到 equalizer 公式

$$
\operatorname{Map}(M,N)\simeq
\operatorname{Eq}
\left(
\prod_i\operatorname{Map}(M_i,N_i)
\rightrightarrows
\prod_{i,j}\operatorname{Map}(M_{ij},N_{ij})
\right).
$$

**证明.** mapping space 是范畴 totalization 中的 limit。写出 totalization 的 mapping object 即得。证毕。

**命题 R.16（perfect 性的 rational 检测）.** 假设 \(D(A,\mathcal M)\) 和所有 \(D(A_{U_I},\mathcal M_{U_I})\) 紧生成，且 restriction functor 保持 compact/perfect 对象。若 \(M|_{U_i}\) 均 perfect，并且 perfect 性满足 descent，则 \(M\) perfect。

**证明.** 这是 perfect 对象形成的子堆性质。R.13 把 \(M\) 写成局部 perfect 对象的 descent datum；perfect 子范畴在有限极限和等价下降下封闭，故 \(M\) 属于全局 perfect 子范畴。证毕。

## R.6 Analytic 主闭包定理

**定理 R.17（Analytic 主闭包）。** 接受 R.4-R.6 和 R.11-R.13 后，第二卷关于 analytic theory 的以下结构在书内闭合：

1. analytic 局部对象；
2. analyticization 的泛性质；
3. analytic 张量积与闭结构；
4. Huber pair 给出的 analytic ring；
5. rational localization；
6. rational Čech descent；
7. 对象、态射和 perfect 性的局部检测形式后果。

**证明.** R.1-R.3 给出局部对象形式。R.4-R.10 给出 analyticization、张量和相对张量。R.11-R.13 给出几何局部化输入。R.14-R.16 是接受 descent 后的形式推论。逐项对应上表。证毕。

## R.7 不能省略的假设

1. pre-analytic datum 不自动是 analytic ring；必须验证 R.4-R.6。
2. rational localization 不只是普通环局部化；它必须与测度对象和 analytic 模范畴相容。
3. descent 必须在范畴层面陈述；只写对象层 sheaf 条件不足以控制 mapping space。
4. analytic tensor 不等于普通 \(\otimes_A^L\)，而是 ordinary tensor 后再 analyticization。

## 练习

1. 证明 R.3。
2. 写出 R.9 中单位约束下降的证明。
3. 对二开 rational cover，写出 R.14 的 descent datum。
4. 解释为什么 R.16 需要 perfect 性本身满足 descent。

