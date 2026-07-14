# 第三章：Prismatic cohomology 与基础比较定理

Prismatic cohomology 的统一性只有在比较态射的源、靶和换基操作全部写清时才有内容。同一个 $\Delta_{R/A}$ 通过模掉 prism ideal、沿 Frobenius 换基、反演特定元素或取完备化，分别通向 Hodge--Tate、de Rham、crystalline 与 étale 信息；这些操作既不发生在同一系数范畴，也不能任意交换。本章以第二章的 bounded prism、structure sheaves 与 Frobenius 为起点，固定每个 specialization 的导出张量、滤过和 semilinear 结构。深比较定理作为精确外部输入，书内则完成态射类型、base change 与相互兼容性的形式推导。

## 3.1 Affine 记号

**约定 3.1.** 若 $X=\operatorname{Spf}(R)$ 是 affine $p$-adic formal scheme over $A/I$，则记
$$
\Delta_{R/A}=R\Gamma_\Delta(X/A).
$$
若 $R$ 是 $p$-completely smooth over $A/I$，则 $\mathbb L_{R/(A/I)}$ 可由 $\Omega^1_{R/(A/I)}$ 表示。

**定义 3.2.** Hodge-Tate specialization complex 定义为
$$
\overline\Delta_{R/A}=\Delta_{R/A}\otimes_A^L A/I.
$$
De Rham specialization complex 定义为
$$
\Delta^{\mathrm{dR}}_{R/A}=\phi_A^\ast\Delta_{R/A}
\widehat\otimes_A^L A/I,
$$
其中 $\phi_A^\ast\Delta_{R/A}=A\otimes_{A,\phi_A}^L\Delta_{R/A}$，并且
本式的帽号专指 tensor product 之后的 derived $p$-completion：
$$
C\widehat\otimes_A^LA/I
:=(C\otimes_A^LA/I)^{\wedge,L}_p.
$$
Hodge--Tate specialization 没有这个额外 completion 符号；这与
Bhatt--Scholze, Theorem 1.8 (2)--(3) 的两个公式一致。

**警告 3.3.** Hodge-Tate specialization 与 de Rham specialization 的操作
序列不同：前者直接 derived modulo $I$；后者先沿 $\phi_A$ pullback，再
derived modulo $I$ 并作 derived $p$-completion。后续公式若省略 Frobenius
twist 或 completion ideal，都必须视为未校验。

## 3.2 Hodge-Tate comparison

**外部输入定理 3.4（Hodge-Tate comparison）.** 令 $(A,I)$ 为 bounded
prism，令 $X$ 为 smooth $p$-adic formal scheme over $A/I$。在 $X_{\mathrm{et}}$
上，Hodge-Tate specialization 的 cohomology sheaves 满足
$$
\mathcal H^i(\overline\Delta_{X/A})
\cong \Omega^i_{X/(A/I)}\{-i\}.
$$
等价地，在递增 Postnikov（亦称 conjugate）filtration 的编号下，其
associated graded 为
$$
\operatorname{gr}_{i}^{\mathrm{conj}}\overline\Delta_{X/A}
\simeq
R\Gamma\left(X,\wedge^i\mathbb L_{X/(A/I)}\right)[-i]\{-i\}.
$$
若 $X$ smooth，则可写成
$$
\operatorname{gr}_{i}^{\mathrm{conj}}\overline\Delta_{X/A}
\simeq
R\Gamma\left(X,\Omega^i_{X/(A/I)}\right)[-i]\{-i\}.
$$
全局公式中的 $\overline\Delta_{X/A}$ 表示对上述 sheaf complex 取
$R\Gamma(X_{\mathrm{et}},-)$。来源为 Bhatt-Scholze, Theorem 4.11 与
Theorem 6.3。

**说明 3.5.** 本书采用
$M\{i\}=M\otimes_{A/I}(I/I^2)^{\otimes i}$，故定理中为 $\{-i\}$。
这一方向已与 Bhatt-Scholze v4, Theorem 1.8 (2) 及 Theorem 4.11 核对；
它不是未定的 dual convention。

**形式推论 3.6.** 若 $X$ proper smooth over $A/I$，且各 $R\Gamma(X,\Omega^i)$ 是 perfect complex，则 $\overline\Delta_{X/A}$ 是 $A/I$ 上的 perfect complex。

**证明.** 设 $d$ 为 $X/(A/I)$ 的最大相对维数；properness 使 $X$
quasi-compact，故可取有限的 $d$，并且 $\Omega^i=0$ 对 $i>d$。定理 3.4
给出的 filtration 因而只有 $0\le i\le d$ 个非零 graded pieces。Shift 和
invertible twist 保持 perfect，故每个 graded piece perfect。由
$$
F^{i+1}\overline\Delta_{X/A}\to
F^i\overline\Delta_{X/A}\to
\operatorname{gr}^i_{\mathrm{conj}}\overline\Delta_{X/A}
$$
逐级使用 perfect complexes 对 cones 的封闭性，得到
$\overline\Delta_{X/A}$ perfect。证毕。

## 3.3 De Rham comparison

**外部输入定理 3.7（de Rham comparison）.** 在定理 3.4 的假设下，
存在自然的 $E_\infty$-$A/I$-algebra 拟同构
$$
\phi_A^\ast R\Gamma_\Delta(X/A)\widehat\otimes_A^L A/I
\simeq
R\Gamma_{\mathrm{dR}}(X/(A/I)).
$$
这里 $\phi_A^\ast C=A\otimes_{A,\phi_A}^LC$，帽号表示 derived
$p$-completion；右侧 $R\Gamma_{\mathrm{dR}}$ 也按 $p$-completed de Rham
complex 解释。该同构与乘法和 differential graded algebra 结构相容。
来源为 Bhatt-Scholze, Corollary 15.4。它本身是
unfiltered comparison；若要保留 Hodge/Nygaard filtration，必须另行调用
relative Nygaard theorem，不能把 filtered compatibility 自动加入本陈述。

**警告 3.8.** 若去掉左侧的 $\phi_A^\ast$，通常得到的是 Hodge-Tate specialization 而不是 de Rham cohomology。这个差异是 prismatic theory 中最常见的公式错误之一。

## 3.4 Crystalline comparison

**外部输入定理 3.9（crystalline comparison）.** 令 $(A,(p))$ 为
crystalline prism，令 $X$ 为 smooth $p$-adic formal scheme over $A/p$。
则在 $D(A)$ 中有自然的 $\varphi$-equivariant 拟同构
$$
R\Gamma_{\mathrm{crys}}(X/A)
\simeq
R\Gamma_\Delta(X/A)\widehat\otimes_{A,\phi_A}^LA
=\left(\phi_A^\ast R\Gamma_\Delta(X/A)\right)^{\wedge,L}_p.
$$
两边均按 derived $p$-complete commutative $A$-algebras 解释。来源为
Bhatt-Scholze, Theorem 5.2。这是积分
comparison；结论没有 invert $p$。

**外部输入定理 3.10（Frobenius isogeny）.** 在定理 3.4 的假设下，
令
$$
C=R\Gamma_\Delta(X/A),\qquad
C^{(1)}=C\widehat\otimes_{A,\phi_A}^LA.
$$
命题 2.19 的 linearized Frobenius 在 derived $(p,I)$-completion 后给出
$C^{(1)}\to C$，并满足
$$
\varphi^{\mathrm{lin}}[1/I]:
C^{(1)}[1/I]
\xrightarrow{\sim}
C[1/I].
$$
若 $I=(d)$，则 $[1/I]$ 表示 $[1/d]$。来源为 Bhatt-Scholze,
Corollary 15.5。积分层的
$\varphi^{\mathrm{lin}}$ 一般不是同构；invert $I$ 也不等于 invert $p$。

## 3.5 Etale comparison

**外部输入定理 3.11（etale comparison, finite level）.** 令
$(A,(d))$ 为 oriented perfect prism，令 $R=A/(d)$ 为对应 perfectoid ring，
令 $X$ 为任意 $p$-adic formal scheme over $R$。记
$\Delta_{X/A}\in D(X_{\mathrm{et}},A)$ 为 prismatic sheaf，记
$$
\mu:X_{\eta,\mathrm{et}}\longrightarrow X_{\mathrm{et}}
$$
为 nearby-cycles map。则对每个 $n\ge1$，在
$D(X_{\mathrm{et}},\mathbf Z/p^n)$ 中有自然拟同构
$$
R\mu_*\mathbf Z/p^n
\simeq
\left(\Delta_{X/A}[1/d]/p^n\right)^{\varphi=1}.
\tag{3.1}
$$
右侧的 fixed points 是 sheaf complexes 中的
$\operatorname{fib}(\varphi-1)$，不是 cohomology sheaves 的 ordinary
invariants。这里 $\varphi$ 虽对 $A$ semilinear，却固定 $\mathbf Z_p$；
restriction of scalars 后 $\varphi-1$ 是 $\mathbf Z/p^n$-linear。无
orientation 时写 $[1/I]$。

若 $X=\operatorname{Spf}(S)$ 是 affine，Theorem 9.1 进一步给出全局推论
$$
R\Gamma_{\mathrm{et}}(\operatorname{Spec}(S[1/p]),\mathbf Z/p^n)
\simeq
\left(\Delta_{S/A}[1/d]/p^n\right)^{\varphi=1},
\qquad
\Delta_{S/A}=R\Gamma_\Delta(X/A).
\tag{3.2}
$$
来源为 Bhatt-Scholze, Theorem 9.1。对非 affine
$X$，由 (3.1) 取 derived global sections 只直接得到右侧 sheaf fixed
complex 的 hypercohomology。若要把它改写成 (3.2) 型的
$R\Gamma_\Delta(X/A)$ 公式，至少还须验证 canonical exchange map
$$
R\Gamma(X_{\mathrm{et}},\Delta_{X/A}/p^n)[1/d]
\longrightarrow
R\Gamma\left(X_{\mathrm{et}},(\Delta_{X/A}/p^n)[1/d]\right)
\tag{3.3}
$$
为拟同构；本定理不对一般 $X$ 自动断言该交换条件。

**警告 3.12.** 定理 3.11 不能简写为
$$
R\Gamma_\Delta(X/A)\cong R\Gamma_{\mathrm{et}}(X_\eta,\mathbf Z_p).
$$
左侧是 $A$-complex 带 Frobenius，右侧是 $\mathbf Z_p$-complex 带 Galois/pro-etale 信息；二者通过 invert $I$ 和 Frobenius fixed construction 才比较。

## 3.6 Base change

**外部输入定理 3.13（prismatic base change）.** 令
$(A,I)\to(B,IB)$ 为 bounded prisms 的态射，令 $X$ 为 smooth $p$-adic
formal scheme over $A/I$，并设
$$
Y=X\times_{\operatorname{Spf}(A/I)}\operatorname{Spf}(B/IB).
$$
定义
$$
C\widehat\otimes_A^LB
:=(C\otimes_A^LB)^{\wedge,L}_{(p,IB)}.
$$
则有自然拟同构
$$
R\Gamma_\Delta(X/A)\widehat\otimes_A^L B
\simeq
R\Gamma_\Delta(Y/B),
$$
并与 $E_\infty$-乘法结构相容。Prism ideal rigidity 保证任意 prism map 的
目标 ideal 就是 $IB$。来源为 Bhatt-Scholze, Corollary 4.12；complex-level theorem 不额外假设 $A\to B$ 有有限 ordinary
Tor-amplitude。

**形式推论 3.14.** 若 $R\Gamma_\Delta(X/A)$ 为 perfect $A$-complex，则 base change 后得到的 $R\Gamma_\Delta(Y/B)$ 为 perfect $B$-complex。

**证明.** 令 $C=R\Gamma_\Delta(X/A)$。Derived base change
$P=C\otimes_A^LB$ 是 perfect $B$-complex。因 $(B,IB)$ 是 bounded prism，
$B$ derived $(p,IB)$-complete。Derived complete $B$-complexes 构成 stable、
对 retract 封闭的 full subcategory；它包含 $B$，从而包含由有限个 shifts、
cones 和 retracts 从 $B$ 生成的所有 perfect complexes。因此 $P$ 已 derived
$(p,IB)$-complete，completion map $P\to P^{\wedge,L}_{(p,IB)}$ 为同构。
定理 3.13 于是把 $R\Gamma_\Delta(Y/B)$ 识别为 perfect complex $P$。证毕。

**警告 3.14A（cohomology 不自动逐次基变换）.** 定理 3.13 是 complexes
的 completed derived base-change theorem。它不自动给出
$$
H^n_\Delta(X/A)\otimes_AB
\cong H^n_\Delta(Y/B).
$$
先令
$$
P=R\Gamma_\Delta(X/A)\otimes_A^LB.
$$
只有另知 $P$ 已 derived $(p,IB)$-complete 时，定理 3.13 的 target 才等于
$P$。在这个额外完备性假设下，若 $B$ 对 $A$ flat，上述 cohomology base
change 由 Tor spectral sequence 得到；若 $R\Gamma_\Delta(X/A)$ bounded
且 $B$ 的 ordinary Tor-dimension 至多一，则命题 5.16 给出短正合列，并
可能出现
$\operatorname{Tor}_1^A(H^{n+1}_\Delta(X/A),B)$。没有该完备性假设时，
Tor spectral sequence 只计算 $H^n(P)$，不能越过 completion 直接计算
$H^n_\Delta(Y/B)$。

## 3.7 统一图式

**说明 3.15（操作层级表）.** 下列结果不能串成一个未标注的交换图：

| 层级 | 对 $R\Gamma_\Delta(X/A)$ 的操作 | 目标范畴 | 是否积分 |
| --- | --- | --- | --- |
| Hodge-Tate | $\otimes_A^LA/I$ | $D(A/I)$ | 是；保留 $p$-torsion |
| de Rham | $\phi_A^*(-)\widehat\otimes_A^LA/I$ | $D(A/I)$ | 是 |
| crystalline | $I=(p)$ 时 $\widehat\otimes_{A,\phi_A}^LA$ | $D(A)$ | 是；未 invert $p$ |
| Frobenius isogeny | completed twist $C^{(1)}$ 后 $[1/I]$ | $D(A[1/I])$ | 只反演 $I$；不等于 rationalization |
| etale finite level | $/p^n$、$[1/I]$、derived $\varphi=1$ | $D(\mathbf Z/p^n)$ | torsion-level |
| rational period comparison | 再 invert $p$ 并延标到 period ring | $D(B_{\mathrm{dR}})$ 等 | 否；属于第四章 |

特别地，$[1/I]$、$[1/p]$、$\otimes_A^LA/I$ 与 $/p^n$ 是四种不同操作。

## 3.8 比较态射的类型检查

**定义 3.16.** 本书把 comparison statement 分为四类：

1. **specialization isomorphism**：由 $R\Gamma_\Delta(X/A)$ 经 base change 得到目标 cohomology；
2. **completed base change**：derived tensor 后还要对指定理想 completion；
3. **Frobenius isogeny**：linearized Frobenius 只在指定 localization 后可逆；
4. **fixed-point comparison**：目标由 Frobenius fibre construction 得到。

Filtered comparison 和 rational period comparison 是额外属性或第五类接口；只有来源
明确声明时才加入。

**命题 3.17.** 定理 3.11 属于 fixed-point comparison；定理 3.7 属于
Frobenius-twisted specialization，而不是定理 3.11 的普通 base change 特例。

**证明.** De Rham comparison 的形式是
$$
\phi_A^\ast R\Gamma_\Delta(X/A)\widehat\otimes_A^L A/I
\simeq R\Gamma_{\mathrm{dR}}(X/(A/I))^{\wedge,L}_p,
$$
其 target 仍为 $A/I$-linear de Rham complex。Etale comparison 则先 modulo
$p^n$、invert $I$，再在 $D(\mathbf Z/p^n)$ 中取 $\varphi-1$ 的 fibre。
两者的系数范畴和操作序列均不同，故分类不同。证毕。

**警告 3.18.** 若一个证明把 fixed-point comparison 当作普通 base change comparison 使用，则它通常会丢失 derived fixed points 中的 cokernel 项。

## 3.9 一个对象的四种 specialization

本章定义了 Hodge-Tate 和 de Rham specialization，并把 crystalline、etale、
Frobenius isogeny 与 base change 写成类型完整的 Bhatt-Scholze 外部输入。
书内证明了有限 filtration 推 perfectness 以及 perfect complex 在 completed base
change 下无需再次 completion 的形式推论。积分、finite-level、after-$I$ 与
after-$p$ 的结论已分别标记。

## 练习

**练习 3.1.** 解释为什么 $\Delta_{R/A}\otimes_A^L A/I$ 与 $\phi_A^\ast\Delta_{R/A}\otimes_A^L A/I$ 不是同一个 construction。

**练习 3.2.** 在 $X$ proper smooth 且 $R\Gamma(X,\Omega^i)$ perfect 的假设下，补全形式推论 3.6 中“有限滤过保持 perfect”的证明。

**练习 3.3.** 写出定理 3.11 的错误简写版本，并指出每个对象的系数环和附加结构为什么不匹配。
