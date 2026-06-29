# 附录 X：Analytic localization 的证明模块

## X.0 目标

附录 R 把 analytic ring localization 作为输入定理 R.4-R.6。本附录把该输入拆成证明模块。其目标是说明：

1. 哪些部分只是 solidification 的形式推广；
2. 哪些部分依赖 analytic ring 公理；
3. analytic tensor 和 analyticization 如何由局部化推出；
4. 剩余核心证明集中在何处。

## X.1 Analytic cone 与局部对象

设 \((A,\mathcal M)\) 是 pre-analytic datum。对测试对象 \(S\)，有

$$
\delta_S:A[\underline S]\to\mathcal M[S],
$$

并令

$$
K_S^{\mathcal M}=\operatorname{cofib}(\delta_S).
$$

令 \(\Sigma_{\mathcal M}\) 为所有 \(\delta_S\)。

**命题 X.1.** 若 \(D(A)\) 是可展示稳定范畴且 \(\Sigma_{\mathcal M}\) 是一组态射，则 \(\Sigma_{\mathcal M}\)-局部对象构成反射稳定子范畴。

**证明.** 与附录 V 的集合生成局部化相同，应用 V.2。证毕。

**命题 X.2.** \(\Sigma_{\mathcal M}\)-局部对象等价于满足

$$
R\operatorname{Hom}_A(K_S^{\mathcal M},C)\simeq0
$$

的对象。

**证明.** 与 V.1 相同，对 cofiber sequence

$$
A[\underline S]\to\mathcal M[S]\to K_S^{\mathcal M}
$$

取 \(R\operatorname{Hom}_A(-,C)\)。证毕。

## X.2 Analytic ring 公理的角色

若只给出 pre-analytic datum，X.1-X.2 已能构造一个形式局部化候选。但 Scholze 意义的 analytic ring 还要求该局部化有正确的张量、测度和几何行为。

**输入定理 X.3（analytic ring localization）。** 若 \((A,\mathcal M)\) 是 analytic ring，则 X.1 的局部化 \(L_{(A,\mathcal M)}\) 满足：

1. 核由 \(K_S^{\mathcal M}\) 生成；
2. 核是 \(\otimes_A^L\)-张量理想；
3. 局部范畴 \(D(A,\mathcal M)\) 是闭对称幺半稳定范畴；
4. \(\mathcal M[S]\) 在局部范畴中代表正确的 \(A\)-值测度对象。

**书内部分.** X.1-X.2 给出局部化候选和局部对象判别。

**外部部分.** 条件 1-4 的同时成立是 analytic ring 深层结构。

## X.3 Analyticization 泛性质

**定义 X.4.** 对 \(M\in D(A)\)，定义

$$
M^{\mathrm{an}}=L_{(A,\mathcal M)}M.
$$

**命题 X.5.** 对 analytic 对象 \(N\)，有自然等价

$$
R\operatorname{Hom}_{D(A,\mathcal M)}(M^{\mathrm{an}},N)
\simeq
R\operatorname{Hom}_{D(A)}(M,N).
$$

**证明.** 这是反射局部化的伴随泛性质。证毕。

## X.4 Analytic 张量

**命题 X.6.** 若 X.3 的核为张量理想，则 analytic 张量积

$$
M\otimes_A^{L,\mathcal M}N
=
L_{(A,\mathcal M)}(M\otimes_A^LN)
$$

给出 \(D(A,\mathcal M)\) 的对称幺半结构。

**证明.** 与 Q.8 或附录 K 的幺半 Bousfield localization 判别相同。核为张量理想保证替换代表元不改变局部化后的张量，普通张量的结合律、交换律和单位约束下降到局部范畴。证毕。

**命题 X.7（内部 Hom）。** 若 \(D(A)\) 是闭对称幺半，且 analytic localization 与闭结构相容，则

$$
R\underline{\operatorname{Hom}}_{\mathcal M}(M,N)
$$

由 \(D(A)\) 中内部 Hom 的 analytic local part 表示。

**证明.** 对 analytic 对象 \(P\)，要求

$$
\operatorname{Map}(P,R\underline{\operatorname{Hom}}_{\mathcal M}(M,N))
\simeq
\operatorname{Map}(P\otimes_A^{L,\mathcal M}M,N).
$$

右侧用 X.6 改写为普通张量后局部化的 mapping space，再由闭结构伴随得到 \(D(A)\) 中内部 Hom。取其局部化即可表示该函子。证毕。

## X.5 Analytic ring 与 solid ring 的关系

solidification 是 analytic localization 的原型。若取 \(A=\mathbb Z\) 且测度对象为

$$
\mathcal M[S]=\mathbb Z^\square[S],
$$

则 analytic cone \(K_S^{\mathcal M}\) 退化为 solid cone \(K_S\)。

**命题 X.8.** 在上述特例中，X.1-X.7 退化为 solidification 的形式结构。

**证明.** 将 \(A[\underline S]\) 代为 \(\mathbb Z[\underline S]\)，\(\mathcal M[S]\) 代为 \(\mathbb Z^\square[S]\)。局部对象、cone、局部化和张量下降的公式逐项与附录 V、W、Q 一致。证毕。

## X.6 核心剩余问题

analytic localization 的完全证明需要补以下内容：

1. 给定 \((A,\mathcal M)\)，证明 \(D(A)\) 可展示且测试对象集合可控；
2. 证明 \(K_S^{\mathcal M}\) 生成的局部化与 Scholze analytic module category 一致；
3. 证明 kernel 是张量理想；
4. 证明 \(\mathcal M[S]\) 的 functoriality 与张量、base change、rational localization 相容；
5. 在 Huber pair 情形证明 rational localization 满足 descent。

## X.7 本附录闭包

**定理 X.9（analytic localization 的分解）。** analytic localization 的证明分解为：

1. 集合生成局部化存在；
2. \(K_S^{\mathcal M}\)-正交给出局部对象；
3. analytic ring 公理识别该局部化为 \(D(A,\mathcal M)\)；
4. kernel 张量理想性给出 analytic tensor；
5. Huber/rational 结构给出几何 descent。

其中 1、2、4 的形式后果在书内证明；3、5 仍为 Scholze 输入或后续证明目标。

## 练习

1. 证明 X.2。
2. 证明 X.6 中张量代表元无关。
3. 在 \(A=\mathbb Z\) 的 solid 特例中写出所有 \(K_S^{\mathcal M}\)。
4. 解释为什么 pre-analytic datum 不足以推出 rational descent。

