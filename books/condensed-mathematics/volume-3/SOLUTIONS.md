# 第三卷练习答案与教师手册补充

作者：Dr. Stochastic Parrot

## 使用说明

全书统一答案见 [../SOLUTIONS.md](../SOLUTIONS.md)。本文件补第三卷核心证明题，重点是有限性、Serre duality、GAGA 和 HRR 的闭包。

## 1. Čech 计算 \(\mathbb P^1\) 上 \(\mathcal O(d)\)

取覆盖 \(U_0=\{X_0\ne0\}\)、\(U_\infty=\{X_1\ne0\}\)，令 \(z=X_1/X_0\)。交集为 \(\mathbb C^\times\)。线丛 \(\mathcal O(d)\) 的转移函数为 \(z^d\)。

Čech 复形为

$$
\mathbb C[z]\oplus z^d\mathbb C[z^{-1}]
\to
\mathbb C[z,z^{-1}].
$$

若 \(d\ge0\)，cokernel 为零，kernel 为次数 \(\le d\) 的多项式，故

$$
h^0=d+1,\qquad h^1=0.
$$

若 \(d=-1\)，两者皆为零。若 \(d\le-2\)，kernel 为零，cokernel 由

$$
z^{-1},z^{-2},\ldots,z^{d+1}
$$

表示，维数为 \(-d-1\)。

## 2. 有限 resolution 传播有限性

设 \(\mathcal F\) 有有限局部自由 resolution

$$
0\to E^{-m}\to\cdots\to E^0\to\mathcal F\to0.
$$

若每个 \(H^q(X,E^i)\) 有限维，则 \(\mathcal F\) 的上同调有限维。

**详解。** 用 hypercohomology spectral sequence

$$
E_1^{p,q}=H^q(X,E^p)
\Rightarrow
\mathbb H^{p+q}(X,E^\bullet).
$$

每个 \(E_1^{p,q}\) 有限维，且 \(p\) 有界。每一页都是有限维空间的 subquotient。收敛后得到有限过滤，其 graded pieces 有限维，因此 abutment 有限维。由于 \(E^\bullet\to\mathcal F\) quasi-isomorphism，\(\mathbb H^n(X,E^\bullet)\cong H^n(X,\mathcal F)\)。

## 3. Serre duality 的 derived 形式

若接受

$$
R\Gamma(X,F)^\vee
\simeq
R\Gamma(X,R\mathcal Hom(F,\omega_X^\bullet)),
$$

则取 cohomology 得

$$
H^i(X,F)^\vee
\cong
H^{-i}(X,R\mathcal Hom(F,\omega_X^\bullet)).
$$

有限性保证 derived dual 的 cohomology 与 ordinary dual 相容。smooth 向量丛情形中，\(\omega_X^\bullet=\omega_X[n]\)，且

$$
R\mathcal Hom(E,\omega_X[n])\simeq E^\vee\otimes\omega_X[n],
$$

故得到经典公式

$$
H^i(X,E)^\vee\cong H^{n-i}(X,E^\vee\otimes\omega_X).
$$

## 4. GAGA 到导出等价

若

$$
\operatorname{Coh}(Y)\simeq\operatorname{Coh}(Y^{an})
$$

是 exact equivalence，则逐项作用在 bounded complex 上给 functor

$$
C^b(\operatorname{Coh}(Y))
\to
C^b(\operatorname{Coh}(Y^{an})).
$$

exactness 保持 acyclic complex，因此保持 quasi-isomorphism。于是下降到

$$
D^b_{\operatorname{coh}}(Y)
\simeq
D^b_{\operatorname{coh}}(Y^{an}).
$$

quasi-inverse 由原 abelian equivalence 的 quasi-inverse 逐项给出。

## 5. HRR 的有限性前提

HRR 左侧为

$$
\chi(X,E)=\sum_i(-1)^i\dim H^i(X,E).
$$

该表达式要求：

1. \(H^i(X,E)\) 有限维；
2. 只有有限多个 \(i\) 非零。

紧复流形上相干上同调有限性给第一项；cohomological dimension 有界给第二项。没有这些前提，左侧不是一个确定整数，HRR 公式无从陈述。

## 6. Clausen-Scholze 核心定理图谱

**AR.1.** AR.3 是 AR.4 和 AR.6 的共同前提。

**答案。** Serre duality 要把 derived dual 的 cohomology 解释成 ordinary dual，需要上同调有限维。HRR 左侧是 Euler characteristic，也需要每个上同调有限维且只有有限多个非零。因此 coherent finiteness 是 duality 和 Riemann-Roch 的共同基础。

**AR.2.** GAGA 的 properness 假设不可删。

**答案。** 非 proper 情形下代数全局函数和解析全局函数可严重不同，例如 \(\mathbb A^1_\mathbb C\) 的代数全局函数是 \(\mathbb C[z]\)，而解析化 \(\mathbb C\) 上全局 holomorphic functions 是所有整函数。解析化不再给 coherent sheaves 和上同调的等价比较。

**AR.3.** Dolbeault-liquid theorem 的三段式。

**答案。** classical theorem：Dolbeault resolution 计算 \(H^\bullet(X,E)\)，且紧情形 Dolbeault Fréchet 复形满足 Fredholm/Hodge 有限性。realization theorem：liquid realization 对相关 Fréchet 复形和闭值域短正合列 exact。formal consequence：realization 后的 cohomology 与 \(\mathcal L_p(H^\bullet(X,E))\) 同构，有限维时为 perfect liquid object。

**AR.4.** six functor 在本卷只能作为接口。

**答案。** 完整 six functor formalism 需要构造 \(f^\ast,Rf_\ast,f_!,f^!,\otimes,R\mathcal Hom\)、证明 base change、projection formula、trace/counit、proper/compact-support 比较和复合相容。本卷只证明接受这些输入后 Serre duality、projection formula 和 internal Hom 的形式后果，因此只能作为后续接口。
