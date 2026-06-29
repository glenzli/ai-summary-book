# 附录 I：解析环公理检查表与失败模式

## I.0 目标

第三、四章把 analytic ring 写成由测度对象 $\mathcal M[S]$ 和 Dirac 映射

$$
A[\underline S]\to\mathcal M[S]
$$

控制的局部化理论。本附录补一个检查表：给定预解析结构后，哪些条件只是形式数据，哪些条件是真正的 analytic ring 输入，哪些错误类比会导致错误结论。

本附录不证明某个具体 $(A,\mathcal M)$ 是 analytic ring；它只整理“证明它是 analytic ring 时必须验证什么”。

## I.1 预解析数据

**定义 I.1.** 设 $A$ 是凝聚交换环。一个预解析结构包含以下数据：

1. 对每个测试空间 $S$，给出凝聚 $A$-模 $\mathcal M[S]$。
2. 对每个连续映射 $f:S\to T$，给出 $A$-线性 pushforward
   $$
   f_*:\mathcal M[S]\to\mathcal M[T].
   $$
3. 对每个 $S$，给出 Dirac 映射
   $$
   \delta_S:A[\underline S]\to\mathcal M[S].
   $$
4. 对复合映射 $S\xrightarrow fT\xrightarrow gU$，有
   $$
   (g\circ f)_*=g_*\circ f_*.
   $$
5. 对恒等映射有 $(\operatorname{id}_S)_*=\operatorname{id}_{\mathcal M[S]}$。

**命题 I.2（有限不交并相容）.** 若预解析结构满足

$$
\mathcal M[S\sqcup T]\simeq\mathcal M[S]\times\mathcal M[T],
\qquad
\mathcal M[\varnothing]\simeq0,
$$

则对有限离散 $S=\{s_1,\ldots,s_n\}$，有

$$
\mathcal M[S]\simeq\prod_{i=1}^n\mathcal M[\{s_i\}].
$$

**证明.** 对 $n$ 归纳。$n=0$ 是空集条件，$n=1$ 为恒等。若 $S=S'\sqcup\{s_n\}$，则

$$
\mathcal M[S]\simeq\mathcal M[S']\times\mathcal M[\{s_n\}],
$$

再用归纳假设。证毕。

**边界 I.3.** 有限不交并相容只控制有限离散测试对象。analytic ring 的内容主要出现在无限 profinite 或 ED 测试对象上；有限情形不能检测 solid/analytic 的深层差异。

## I.2 Cone 判别

对每个测试对象 $S$，定义

$$
K_S^\mathcal M=\operatorname{Cone}(A[\underline S]\xrightarrow{\delta_S}\mathcal M[S]).
$$

**命题 I.4（局部对象的两个表述）.** 对 $C\in D(A)$，以下条件等价：

1. 对所有 $S$，
   $$
   R\operatorname{Hom}_A(K_S^\mathcal M,C)\simeq0.
   $$
2. 对所有 $S$，
   $$
   R\operatorname{Hom}_A(\mathcal M[S],C)
   \to
   R\operatorname{Hom}_A(A[\underline S],C)
   $$
   是等价。

**证明.** 对 distinguished triangle

$$
A[\underline S]\to\mathcal M[S]\to K_S^\mathcal M\to
$$

应用 $R\operatorname{Hom}_A(-,C)$，得到 fiber sequence

$$
R\operatorname{Hom}_A(K_S^\mathcal M,C)
\to
R\operatorname{Hom}_A(\mathcal M[S],C)
\to
R\operatorname{Hom}_A(A[\underline S],C).
$$

第三个箭头为等价当且仅当前一项为零。证毕。

**命题 I.5（有限测试对象的空洞性）.** 若对有限离散 $S$，Dirac 映射 $\delta_S$ 是等价，则 $K_S^\mathcal M\simeq0$，该 $S$ 对 analytic condition 不施加额外限制。

**证明.** 等价态射的 cone 为零；零对象对任意 $C$ 的导出 Hom 为零。证毕。

## I.3 从预解析到解析环需要的输入

**检查表 I.6.** 要把预解析结构 $(A,\mathcal M)$ 升级为本书使用的 analytic ring，至少需要以下结论：

1. **反射性。** 由 $\{K_S^\mathcal M\}$ 定义的局部对象全子范畴
   $$
   D(A,\mathcal M)\subset D(A)
   $$
   的包含函子有左伴随。
2. **稳定性。** $D(A,\mathcal M)$ 对 shift、fiber/cofiber、极限和余极限封闭。
3. **张量理想性。** 由 $K_S^\mathcal M$ 生成的核是 $D(A)$ 中的张量理想。
4. **自由对象相容。** 解析化后，$\mathcal M[S]$ 具有自由解析 $A$-模的预期泛性质。
5. **几何局部化相容。** 对 Huber pair 或几何空间，rational localization 与 analytic localization 相容，并满足 Cech descent。
6. **大小控制。** 测试对象 $S$ 从一个集合大小控制的站点中选取，避免把局部化定义成 proper class 生成的问题。

其中 1、3、4、5 是 Scholze 理论中的实质输入；2 在 1 的 presentable stable setting 下是形式推论；6 是集合论技术条件。

**命题 I.7（反射性推出稳定性的一部分）.** 若 $D(A,\mathcal M)$ 是由条件

$$
R\operatorname{Hom}_A(K_S^\mathcal M,-)\simeq0
$$

定义的局部对象范畴，则它对极限、shift 和扩张封闭。

**证明.** 对固定 $K_S^\mathcal M$，函子 $R\operatorname{Hom}_A(K_S^\mathcal M,-)$ 保持极限。稳定范畴中 shift 与 fiber sequence 也被映射到对应的 shift 与 fiber sequence。零对象条件因此对这些操作封闭。证毕。

## I.4 失败模式

**失败模式 I.8（任意测度赋值不够）.** 给每个 $S$ 任意指定 $\mathcal M[S]$ 和 Dirac 映射，不足以推出 analytic ring。缺失点可能是：

1. 局部对象没有左伴随反射。
2. 反射存在但核不是张量理想。
3. 张量下降存在但不满足几何局部化。
4. $\mathcal M[S]$ 不满足自由解析模的泛性质。

**证明.** 四项分别对应检查表 I.6 的独立条件。任意数据只给出 cone $K_S^\mathcal M$ 和局部对象定义；它不自动提供左伴随、张量理想性或几何 descent。证毕。

**失败模式 I.9（普通完备化类比不够）.** 设 $A$ 是拓扑环。即使某个代数完备化 $\widehat A$ 存在，也不能由此推出 $(A,\mathcal M)$ 是 analytic ring。

**理由.** 完备化控制指定滤过系统或 Cauchy 条件；analytic ring 要控制所有测试对象 $S$ 上的测度对象 $\mathcal M[S]$，并要求 cone 生成的局部化与张量和几何覆盖相容。两者不是同一组条件。

**失败模式 I.10（逐点张量不够）.** 若 $M,N$ 是解析对象，普通张量 $M\otimes_A^LN$ 不一定已经解析；解析张量定义为

$$
M\otimes_{(A,\mathcal M)}^LN
=
L_{(A,\mathcal M)}(M\otimes_A^LN).
$$

**证明.** 若普通张量总保持解析对象，则 localization 核自动为张量理想。但张量理想性正是需要验证或引用的条件，不能从反射性推出。证毕。

## I.5 可安全使用的形式推论

一旦 $(A,\mathcal M)$ 已知是 analytic ring，本书可以安全使用：

1. 对解析对象 $C$，
   $$
   R\operatorname{Hom}_A(LM,C)\simeq R\operatorname{Hom}_A(M,C).
   $$
2. 对任意 $M$，
   $$
   L K_S^\mathcal M\simeq0.
   $$
3. 对解析对象 $M,N$，
   $$
   M\otimes_{(A,\mathcal M)}^LN
   =
   L(M\otimes_A^LN).
   $$
4. 若 rational Cech descent 是输入定理，则解析对象和态射可在 rational cover 上检测。

这些结论分别由附录 C、E、G 的形式命题证明。

## I.6 练习

**练习 I.1.** 对有限二点集 $S=\{0,1\}$，在有限不交并相容假设下写出 $\mathcal M[S]$。

**练习 I.2.** 证明若 $K_S^\mathcal M\simeq0$，则 $S$ 对局部对象条件无贡献。

**练习 I.3.** 举出检查表 I.6 中哪一项用于定义解析张量积。

**练习 I.4.** 说明为什么 rational Cech descent 不是 Bousfield localization 的形式推论，而是额外几何输入。
