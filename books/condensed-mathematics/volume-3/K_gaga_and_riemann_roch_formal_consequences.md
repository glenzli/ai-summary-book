# 附录 K：GAGA 与 Riemann-Roch 的形式推论

## K.0 目标

第三卷第六、七章把 GAGA 和 Riemann-Roch 作为输入定理。本附录证明接受这些输入后的形式推论：

1. abelian GAGA 等价推出 bounded coherent derived category 的等价。
2. 上同调比较可表达为 $R\Gamma$ 的自然同构。
3. Euler characteristic 在 GAGA 比较下保持。
4. Riemann-Roch 公式如何从 K-theory、Chern character 和 trace 的相容性推出。

本附录不证明 Serre GAGA 本身，也不构造 Chern character 或 Todd class。

## K.1 Abelian GAGA 到导出 GAGA

设 $X$ 是 proper $\mathbb C$-scheme，$X^{an}$ 为解析化。设

$$
\operatorname{an}:\operatorname{Coh}(X)\to\operatorname{Coh}(X^{an})
$$

是 exact equivalence。

**命题 K.1.** exact equivalence $\operatorname{an}$ 诱导三角范畴等价

$$
D^b(\operatorname{Coh}(X))
\simeq
D^b(\operatorname{Coh}(X^{an})).
$$

**证明.** exact functor 逐项作用于有界复形，得到

$$
K^b(\operatorname{Coh}(X))
\to
K^b(\operatorname{Coh}(X^{an})).
$$

因为 $\operatorname{an}$ exact，它保持 acyclic 复形和 quasi-isomorphism，因此下降到导出范畴。其 quasi-inverse 由 abelian equivalence 的 exact quasi-inverse 逐项作用得到。两侧复合与恒等 functor 自然同构，故得到三角等价。证毕。

**输入定理 K.2（GAGA cohomology comparison）.** 对任意 $\mathcal F\in\operatorname{Coh}(X)$，自然映射

$$
H^i(X,\mathcal F)\to H^i(X^{an},\mathcal F^{an})
$$

为同构。

**命题 K.3.** 输入定理 K.2 等价于导出全局截面比较

$$
R\Gamma(X,\mathcal F)
\simeq
R\Gamma(X^{an},\mathcal F^{an})
$$

在 $D^b(\mathbf C)$ 中对所有 coherent sheaf 成立，并对 $D^b(\operatorname{Coh}(X))$ 中对象逐三角延拓。

**证明.** 若 $R\Gamma$ 比较同构成立，取上同调即得到 K.2。反过来，对 sheaf $\mathcal F$，两侧是有界复向量空间复形；若所有上同调映射为同构，则该映射是 quasi-isomorphism。对有界复形对象，用 stupid filtration 或同调截断把对象分解为有限个 sheaf 的 shift；比较同构对三角满足二出三性质，因此延拓到整个 $D^b$。证毕。

## K.2 GAGA 保持 Euler characteristic

**定义 K.4.** 对 $E\in D^b_{\operatorname{coh}}(X)$，定义

$$
\chi(X,E)=
\sum_i(-1)^i\dim_\mathbb C H^i(R\Gamma(X,E)).
$$

properness 与 coherent finiteness 保证该和有限。

**命题 K.5.** 若 GAGA 的 $R\Gamma$ 比较成立，则

$$
\chi(X,E)=\chi(X^{an},E^{an}).
$$

**证明.** 由命题 K.3，

$$
R\Gamma(X,E)\simeq R\Gamma(X^{an},E^{an})
$$

在 $D^b(\mathbf C)$ 中同构。同构复形有同构上同调，因此各维数相同，交错和相同。证毕。

## K.3 K-theory 形式

设 $K_0(X)$ 为 coherent sheaves 的 Grothendieck group。短正合列

$$
0\to A\to B\to C\to0
$$

给关系

$$
[B]=[A]+[C].
$$

**命题 K.6.** Euler characteristic 定义出群同态

$$
\chi_X:K_0(X)\to\mathbb Z.
$$

**证明.** 短正合列给出上同调长正合列。有限维长正合列的交错维数和为零，因此

$$
\chi(B)=\chi(A)+\chi(C).
$$

故 $\chi$ 尊重 Grothendieck group 的关系，定义群同态。证毕。

**命题 K.7.** 若 $\operatorname{an}$ 是 exact equivalence，则它诱导同构

$$
K_0(X)\simeq K_0(X^{an}),
$$

并且 Euler characteristic 同态与该同构相容。

**证明.** exact equivalence 保持短正合列，故诱导 Grothendieck group 同构。Euler characteristic 的相容性由命题 K.5 得到。证毕。

## K.4 Riemann-Roch 的形式推出

设 $X$ 是紧复流形，$E$ 是全纯向量丛。Riemann-Roch 的输入包含：

1. Chern character
   $$
   \operatorname{ch}:K^0(X)\to H^{even}(X,\mathbb Q).
   $$
2. Todd class
   $$
   \operatorname{td}(T_X)\in H^{even}(X,\mathbb Q).
   $$
3. trace/integration map
   $$
   \int_X:H^{2\dim_\mathbb C X}(X,\mathbb Q)\to\mathbb Q.
   $$
4. 输入定理：Euler characteristic 与右侧 characteristic number 相等。

**命题 K.8.** 若 Riemann-Roch 输入定理对向量丛成立，并且两边对短正合列可加，则它诱导群同态恒等式

$$
\chi_X(-)
=
\int_X\operatorname{ch}(-)\operatorname{td}(T_X)
$$

作为 $K^0(X)\to\mathbb Q$ 的同态成立。

**证明.** 左侧由命题 K.6 可加。右侧中 $\operatorname{ch}$ 是 $K$-理论群同态，乘以固定类 $\operatorname{td}(T_X)$ 后再积分仍为群同态。若两个群同态在向量丛生成元上相同，则在整个由向量丛生成的 $K^0(X)$ 上相同。证毕。

**命题 K.9（$\mathbb P^1$ 检验）.** 对 $X=\mathbb P^1$ 和 $E=\mathcal O(d)$，

$$
\int_{\mathbb P^1}\operatorname{ch}(\mathcal O(d))\operatorname{td}(T_{\mathbb P^1})
=
d+1.
$$

**证明.** 令 $H\in H^2(\mathbb P^1,\mathbb Z)$ 为点类，满足 $\int_{\mathbb P^1}H=1$。有

$$
c_1(\mathcal O(d))=dH,
\qquad
\operatorname{ch}(\mathcal O(d))=e^{dH}=1+dH
$$

因为 $H^2=0$。又

$$
c_1(T_{\mathbb P^1})=2H,
\qquad
\operatorname{td}(T_{\mathbb P^1})=1+\frac12c_1(T_{\mathbb P^1})=1+H.
$$

相乘得

$$
(1+dH)(1+H)=1+(d+1)H.
$$

积分取 $H$ 的系数，得到 $d+1$。证毕。

**推论 K.10.** 附录 H 的 Čech 计算与 Riemann-Roch 在 $\mathbb P^1$ 上相容：

$$
\chi(\mathbb P^1,\mathcal O(d))=d+1.
$$

**证明.** 附录 H 已证明

$$
h^0(\mathbb P^1,\mathcal O(d))-h^1(\mathbb P^1,\mathcal O(d))=d+1
$$

对所有 $d\in\mathbb Z$ 成立。命题 K.9 给出 Riemann-Roch 右侧也等于 $d+1$。证毕。

## K.5 Condensed/analytic 表述中的位置

在 condensed/analytic 框架中，GAGA 和 Riemann-Roch 的形式推论可读作：

1. analytic 派生范畴提供 algebraic 和 analytic coherent theories 的共同目标。
2. $R\Gamma$ 比较是该共同目标中同一对象的不同模型之间的 quasi-isomorphism。
3. Euler characteristic 是 trace of identity 的数值影子。
4. Riemann-Roch 断言这个 trace 可由 characteristic classes 计算。

**边界说明.** 本附录没有构造 analytic 派生范畴，也没有证明 Chern character 与 trace 的相容性。这些仍是 Clausen-Scholze 和经典 Riemann-Roch 理论的输入。

## 练习

**练习 K.1.** 证明 exact equivalence 的 quasi-inverse 也是 exact。

**练习 K.2.** 用 long exact sequence 证明命题 K.6 中交错维数可加。

**练习 K.3.** 对 $d=-2,-1,0,1$ 分别比较附录 H 与命题 K.9 的结果。

**练习 K.4.** 说明为什么命题 K.8 不能替代 Riemann-Roch 输入定理本身。
