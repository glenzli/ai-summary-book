# 附录 F：经典复几何输入定理的精确形式

## F.0 目标

第三卷主体章节使用 Dolbeault、Cartan、有限性、Serre duality、GAGA 和 Riemann-Roch。若只说“经典定理”或“Clausen-Scholze 输入”，颗粒太粗。本附录把这些输入定理精确写出，并说明哪些推论在本书内证明，哪些仍作为外部输入。

本附录采用以下约定：

1. $X$ 表示复流形或复解析空间；若涉及积分和 Serre duality，默认 $X$ 是紧复流形，复维数为 $n$。
2. $\mathcal O_X$ 为全纯函数层。
3. 若 $E$ 是全纯向量丛，$\mathcal O(E)$ 表示其全纯截面层。
4. $\mathcal A_X^{p,q}(E)$ 表示 $E$-值光滑 $(p,q)$-形式层。
5. $\omega_X=\Omega_X^n$ 表示 canonical bundle。

## F.1 Dolbeault resolution

**输入定理 F.1（Dolbeault lemma with coefficients）.** 设 $X$ 是复流形，$E$ 是全纯向量丛。复形

$$
0\to
\mathcal O(E)
\to
\mathcal A_X^{0,0}(E)
\xrightarrow{\bar\partial}
\mathcal A_X^{0,1}(E)
\xrightarrow{\bar\partial}
\cdots
\xrightarrow{\bar\partial}
\mathcal A_X^{0,n}(E)
\to0
$$

是 $\mathcal O(E)$ 的 resolution。换言之，该复形在正次数处 exact，零次 kernel 为全纯截面。

**本书不证明的部分.** 局部 $\bar\partial$-Poincaré lemma 所需的一变量 Cauchy-Green 基本解估计。其 sheaf 正合推导和 polydisc 同伦骨架见附录 R。

**本书证明的推论 F.2.** 若 $X$ 是 paracompact 复流形，则

$$
H^i(X,\mathcal O(E))
\cong
H^i(\Gamma(X,\mathcal A_X^{0,\bullet}(E)),\bar\partial).
$$

**证明.** 由输入定理 F.1，$\mathcal A_X^{0,\bullet}(E)$ 是 $\mathcal O(E)$ 的 resolution。每个 $\mathcal A_X^{0,q}(E)$ 是 fine sheaf，因为光滑 partition of unity 可逐点乘到光滑形式上。fine sheaf 在 paracompact 空间上 acyclic。因此该 resolution 是 acyclic resolution。sheaf cohomology 可由 acyclic resolution 的全局截面复形计算，得到同构。证毕。

附录 N 展开本段使用的 fine sheaf、Cech 同伦和 acyclic resolution 形式证明。

## F.2 Cartan 定理 A/B 与 Stein acyclicity

**输入定理 F.3（Cartan A）.** 若 $U$ 是 Stein 空间，$\mathcal F$ 是相干解析层，则 $\mathcal F$ 由全局截面生成。也就是说，对每个 $x\in U$，自然映射

$$
\Gamma(U,\mathcal F)\otimes_{\mathbb C}\mathcal O_{U,x}
\to
\mathcal F_x
$$

的像生成 $\mathcal F_x$。

**输入定理 F.4（Cartan B）.** 若 $U$ 是 Stein 空间，$\mathcal F$ 是相干解析层，则

$$
H^i(U,\mathcal F)=0
\qquad (i>0).
$$

**推论 F.5（Stein 覆盖计算相干上同调）.** 设 $X$ 有有限开覆盖 $\mathfrak U=\{U_i\}$，并且所有有限交

$$
U_{i_0\cdots i_p}=U_{i_0}\cap\cdots\cap U_{i_p}
$$

都是 Stein。若 $\mathcal F$ 是相干解析层，则 Čech 复形

$$
\check C^p(\mathfrak U,\mathcal F)
=
\prod_{i_0<\cdots<i_p}
\Gamma(U_{i_0\cdots i_p},\mathcal F)
$$

计算 $H^\bullet(X,\mathcal F)$。

**证明.** 由 Cartan B，所有有限交上的高阶上同调消失。因此覆盖 $\mathfrak U$ 对 $\mathcal F$ acyclic。Čech-to-derived spectral sequence

$$
E_1^{p,q}=
\prod_{i_0<\cdots<i_p}
H^q(U_{i_0\cdots i_p},\mathcal F)
\Rightarrow
H^{p+q}(X,\mathcal F)
$$

在 $q>0$ 行消失，故退化为 Čech 复形同调。证毕。

Cartan A/B 接受后的有限生成、短正合列全局截面正合性和 Stein acyclic 覆盖工具见附录 V。

## F.3 相干上同调有限性

**输入定理 F.6（Grauert finiteness / coherent finiteness）.** 若 $X$ 是紧复空间，$\mathcal F$ 是相干解析层，则

$$
\dim_{\mathbb C}H^i(X,\mathcal F)<\infty
$$

对所有 $i$ 成立，且非零项只可能出现在有限范围。

**向量丛情形的分析输入 F.7（Hodge-Fredholm 形式）.** 若 $X$ 是紧复流形，$E$ 是全纯向量丛，并选择 Hermitian 度量，则 Dolbeault Laplacian

$$
\Delta_{\bar\partial}
=
\bar\partial\bar\partial^\ast+\bar\partial^\ast\bar\partial
$$

是椭圆算子，其 harmonic forms 空间

$$
\mathcal H^{0,q}(X,E)=\ker\Delta_{\bar\partial}
$$

有限维，并且自然映射

$$
\mathcal H^{0,q}(X,E)
\to
H^q(\Gamma(X,\mathcal A_X^{0,\bullet}(E)),\bar\partial)
$$

为同构。

**推论 F.8.** 若接受 F.7，则向量丛 $E$ 的 $H^q(X,\mathcal O(E))$ 有限维。

**证明.** 由 F.2，$H^q(X,\mathcal O(E))$ 由 Dolbeault 全局截面复形计算；由 F.7，该同调同构于有限维 harmonic forms 空间。附录 L 给出 Fredholm-Hodge 输入推出有限维性的形式证明。证毕。

若相干层有全局有限局部自由 resolution，则向量丛有限性传播到相干层有限性；见附录 X。局部有限局部自由分解和其全局化边界见附录 W。

## F.4 Serre duality

**输入定理 F.9（Serre duality, vector bundle form）.** 设 $X$ 是紧复流形，$\dim_{\mathbb C}X=n$，$E$ 是全纯向量丛。则积分配对诱导完美配对

$$
H^q(X,\mathcal O(E))
\times
H^{n-q}(X,\mathcal O(E^\vee\otimes\omega_X))
\to
\mathbb C.
$$

等价地，

$$
H^q(X,\mathcal O(E))^\vee
\cong
H^{n-q}(X,\mathcal O(E^\vee\otimes\omega_X)).
$$

**本书可证明的相容性 F.10.** 在 Dolbeault 复形层面，配对

$$
(\alpha,\beta)\mapsto
\int_X\operatorname{tr}(\alpha\wedge\beta)
$$

与 $\bar\partial$ 微分相容，因此下降到 Dolbeault cohomology。

**证明.** 对合适次数的形式，由 Leibniz 规则

$$
\bar\partial(\alpha\wedge\beta)
=
\bar\partial\alpha\wedge\beta
 +(-1)^{|\alpha|}\alpha\wedge\bar\partial\beta.
$$

在紧无边界流形上，Stokes 定理给

$$
\int_X\bar\partial(\alpha\wedge\beta)=0.
$$

故若改变 $\alpha$ 或 $\beta$ 一个 $\bar\partial$-边界，积分配对不变，并且闭形式之间的配对只依赖同调类。证毕。

**本书不证明的部分.** F.9 的完美性，即配对非退化。这通常由 Hodge theory、椭圆正则性和 Hodge star 给出。

## F.5 GAGA

**输入定理 F.11（Serre GAGA）.** 设 $X$ 是 proper $\mathbb C$-scheme，$X^{an}$ 为其解析化。则解析化函子给出范畴等价

$$
\operatorname{Coh}(X)
\xrightarrow{\sim}
\operatorname{Coh}(X^{an}),
$$

并且对任意 $\mathcal F\in\operatorname{Coh}(X)$ 和所有 $i$，自然映射

$$
H^i(X,\mathcal F)
\to
H^i(X^{an},\mathcal F^{an})
$$

为同构。

**不可省略的假设.** properness 必须保留。非 proper 情形中解析侧会出现比代数侧更多的全纯函数。例如 $\mathbb A^1_\mathbb C$ 的解析化为 $\mathbb C$，其全纯函数远多于多项式函数。

## F.6 Grothendieck-Hirzebruch-Riemann-Roch

**输入定理 F.12（HRR，光滑紧复流形形式）.** 设 $X$ 是紧复流形，$E$ 是全纯向量丛。则

$$
\chi(X,E)
=
\sum_i(-1)^i\dim_{\mathbb C}H^i(X,\mathcal O(E))
=
\int_X\operatorname{ch}(E)\operatorname{td}(T_X).
$$

**本书可证明的低维例子 F.13.** 对 $X=\mathbb P^1$，$E=\mathcal O(d)$，

$$
\chi(\mathbb P^1,\mathcal O(d))=d+1.
$$

**证明.** 令 $H\in H^2(\mathbb P^1,\mathbb Z)$ 为超平面类，满足 $\int_{\mathbb P^1}H=1$。有

$$
\operatorname{ch}(\mathcal O(d))=e^{dH}=1+dH
$$

因为 $H^2=0$。又 $c_1(T_{\mathbb P^1})=2H$，故

$$
\operatorname{td}(T_{\mathbb P^1})
=1+\frac12c_1(T_{\mathbb P^1})
=1+H.
$$

于是

$$
\operatorname{ch}(\mathcal O(d))\operatorname{td}(T_{\mathbb P^1})
=(1+dH)(1+H)=1+(d+1)H.
$$

取顶次数积分得 $d+1$。证毕。

## F.7 凝聚/解析翻译的输入

**输入定理 F.14（Clausen-Scholze 复几何建模，拆分形式）.** 对本卷涉及的 compact complex manifolds 和 coherent analytic sheaves，存在 condensed/analytic 派生范畴中的对象和函子，使得：

1. $\mathcal O_X$、相干层和 Dolbeault 复形可提升到 analytic/liquid 语境。
2. Dolbeault resolution 在该语境中仍计算导出全局截面。
3. 有限性、Serre duality、GAGA 和 HRR 可用 $f_!,f^!$、trace 与相干对象表达。
4. 忘记 condensed/analytic 结构后，恢复经典定理的数值和配对。

**严格性说明.** F.14 不是一个可以在本书中用几行证明的定理。第三卷主体所有 condensed/analytic 复几何结论都应引用 F.14 的具体条款，而不是泛称“由 Clausen-Scholze”。

## F.8 本附录小结

第三卷可在书内完整证明的内容主要是：

1. Dolbeault resolution 推出 sheaf cohomology 的计算方式。
2. acyclic Stein 覆盖推出 Čech 计算。
3. Dolbeault 配对与微分相容。
4. $\mathbb P^1$ 上 Riemann-Roch 的低维计算。

仍作为输入的内容是：

1. 局部 $\bar\partial$ 可解性。
2. Cartan A/B。
3. 椭圆正则性与 Hodge theorem。
4. Serre duality 配对完美性。
5. GAGA。
6. HRR 的一般证明。
7. Clausen-Scholze 的 condensed/analytic 建模。
