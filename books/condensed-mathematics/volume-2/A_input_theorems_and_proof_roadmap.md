# 附录 A：输入定理与证明路线

## A.0 目标

第二卷多次使用 Scholze 讲义中的结构定理。本附录不把这些长证明重写成简短伪证明，而是列出每个输入定理的作用、依赖、证明路线和本卷使用范围。

## A.1 solid 派生 localization

**输入定理 A.1.** 包含函子

$$
D_{\square}(\mathbb Z)\hookrightarrow D(\mathbf{CondAb})
$$

有左伴随

$$
L^\square:D(\mathbf{CondAb})\to D_{\square}(\mathbb Z).
$$

**本卷使用位置.** 第一章定义派生 solidification；第二章定义派生 solid 张量积。

**证明路线.**

1. 对每个 profinite $S$，考虑
   $$
   \mathbb Z[\underline S]\to\mathbb Z^\square[S].
   $$
2. 令 $K_S$ 为 cone，solid 复形等价于对所有 $S$ 满足
   $$
   R\operatorname{Hom}(K_S,C)=0.
   $$
3. 在紧生成稳定范畴中，对由 $\{K_S\}$ 生成的局部化子范畴作 Bousfield localization。
4. 证明局部对象的心脏与 ordinary solid abelian groups 相容。

**风险点.** 第 3 步需要集合论控制和紧生成性；这不是第一卷的普通同调代数能直接推出的。

## A.2 solid 张量积

**输入定理 A.2.** $D_{\square}(\mathbb Z)$ 在

$$
M\otimes_{\mathbb Z}^{L,\square}N
=
L^\square(M\otimes_{\mathbb Z}^LN)
$$

下成为闭对称幺半范畴。

**本卷使用位置.** 第二章定义 solid 环、solid 模和相对 solid 张量积。

**证明路线.**

1. 证明局部化所杀掉的对象对张量稳定。
2. 检查单位对象是 $\mathbb Z^\square$。
3. 对生成元 $\mathbb Z^\square[S]$ 验证乘法公式
   $$
   \mathbb Z^\square[S]\otimes^{L,\square}\mathbb Z^\square[T]
   \simeq
   \mathbb Z^\square[S\times T].
   $$
4. 用生成元推广到整个范畴。

**风险点.** 普通张量积不保持无限乘积公式；必须使用 solidification 后的张量。

## A.3 analytic rings

**输入定理 A.3.** 解析环 $(A,\mathcal M)$ 的解析对象构成反射子范畴

$$
D(A,\mathcal M)\subset D(A),
$$

并带有解析化函子

$$
L_{(A,\mathcal M)}:D(A)\to D(A,\mathcal M).
$$

**本卷使用位置.** 第三、四章。

**证明路线.**

1. 对所有极不连通 $S$，构造 cone
   $$
   K_S^{\mathcal M}=\operatorname{Cone}(A[\underline S]\to\mathcal M[S]).
   $$
2. 把解析对象定义为对所有 $K_S^{\mathcal M}$ 局部的对象。
3. 用 Bousfield localization 构造左伴随。
4. 证明该 localization 与 $A$-模张量积相容。

**风险点.** 解析环定义包含的技术条件正是为了保证第 3、4 步成立；任意预解析结构不自动给出解析环。

## A.4 liquid 测度理论

**输入定理 A.4.** 对第二卷输入定理 D.5 指定的允许范围内的 $p$，$(\mathbb R,\mathcal M_{<p})$ 是解析环，并给出 $p$-liquid 实向量空间范畴。

**本卷使用位置.** 第五章。

**证明路线.**

1. 构造满足 $p$-型可求和条件的测度对象 $\mathcal M_p[S]$。
2. 对 $q<p$ 取 filtered colimit 得到 $\mathcal M_{<p}[S]$。
3. 验证 finite disjoint union、Dirac map 和张量相容性。
4. 验证 analytic condition，即对应 Hom 判别和 Bousfield localization。

**风险点.** 不能把 liquid 当作 Banach completion；关键是测度对象的解析环条件。

## A.5 离散 Huber pair 与 rational localization

**输入定理 A.5.** 离散 Huber pair $(A,A^+)$ 给出解析环 $(A,A^+)^\square$，并且 rational localization 与解析模范畴相容。

**本卷使用位置.** 第六章。

**证明路线.**

1. 对有限生成子 Huber pair 构造解析测度对象。
2. 通过 filtered colimit 扩展到一般离散 Huber pair。
3. 对 rational subset 构造局部化 Huber pair。
4. 验证解析化与局部化交换。
5. 建立 rational Cech 下降。

**风险点.** 这一步是从代数式解析环进入几何空间的关键；不能只用普通环局部化替代。

## A.6 紧支撑推前

**输入定理 A.6.** 对仿射有限型

$$
f:\operatorname{Spec}A\to\operatorname{Spec}\mathbb Z,
$$

存在

$$
f_!:D(A^\square)\to D(\mathbb Z^\square)
$$

并满足投影公式和右伴随 $f^!$ 的存在性。

**本卷使用位置.** 第七章。

**证明路线.**

1. 对 affine space 建立边界项控制。
2. 对一般有限型代数用 presentation 降到 affine space 和闭嵌入。
3. 证明 $f_!$ 保持直接和和紧对象。
4. 用 compact generation 得到右伴随。
5. 证明投影公式。

**风险点.** 非 proper 边界项是核心难点；普通 forgetful pushforward 不能替代 $f_!$。

## A.7 复几何目标

Clausen-Scholze 复几何讲义中的 finiteness、Serre duality、GAGA 和 Riemann-Roch 属于第三卷主题。第二卷只准备范畴语言。

## A.8 本附录小结

第二卷的证明完整性应这样理解：

1. localization、Hom 判别、类型检查、基本推论在卷内证明。
2. Scholze 的结构定理作为输入定理使用，并在本附录给出证明路线。
3. 复几何应用不在第二卷证明，交给第三卷。
