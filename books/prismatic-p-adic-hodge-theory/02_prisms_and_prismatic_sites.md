# 第二章：Prism、Cartier divisor 与 prismatic site

## 本章目标

本章给出 prism、有界 prism、oriented prism、基本例子和 relative prismatic site 的严格定义。目标是把 prismatic cohomology 的输入对象写清楚：不是一个单独的环，而是一个带 $\delta$-结构和 Cartier divisor 的完备算术 thickenings 构成的 site。

## 依赖前置知识

依赖第一章的 $\delta$-环、Frobenius lift、distinguished element、Witt vectors 和 Breuil-Kisin 型例子。需要熟悉 formal schemes、site、sheaf 和 faithfully flat topology 的基本语言。

## 2.1 Cartier divisor ideal

**定义 2.1.** 令 $A$ 为环。理想 $I\subset A$ 称为定义 Cartier divisor，如果 Zariski 局部存在 $d\in A$ 使得 $I=(d)$，且 $d$ 是 nonzerodivisor。若全局 $I=(d)$，则称该 Cartier divisor 被 $d$ 定向。

**说明 2.2.** 本书中的 prism 经常在 oriented 情形写作 $(A,d)$。这不是额外结构的省略，而是把 $I=(d)$ 的生成元纳入记号。不同生成元会改变某些 twist 的具体识别，因此最终版本必须记录生成元选择。

## 2.2 Prism

**定义 2.3.** 一个 prism 是二元组 $(A,I)$，其中：

1. $A$ 是 $\delta$-环；
2. $I\subset A$ 定义 Cartier divisor；
3. $A$ 是 derived $(p,I)$-complete；
4. $p\in I+\phi_A(I)A$。

若 $A/I$ 的 $p^\infty$-torsion 有界，即存在 $n$ 使得
$$
(A/I)[p^\infty]=(A/I)[p^n],
$$
则称 $(A,I)$ 为 bounded prism。

**定义 2.4.** Prism 态射 $(A,I)\to(B,J)$ 是保持 $\delta$-结构的环同态 $A\to B$，并满足 $I$ 的像落在 $J$ 中。

**命题 2.5.** Prism 态射与 Frobenius lift 相容：若 $f:(A,I)\to(B,J)$ 是 prism 态射，则
$$
f\circ\phi_A=\phi_B\circ f.
$$

**证明.** 因为 $f$ 保持 $\delta$，对任意 $x\in A$，
$$
f(\phi_A(x))=f(x^p+p\delta_A(x))
=f(x)^p+p\delta_B(f(x))
=\phi_B(f(x)).
$$
证毕。

**命题 2.6.** 若 $A$ 是 $\delta$-环，$d\in A$ 是 nonzerodivisor 且 distinguished，并且 $A$ derived $(p,d)$-complete，则 $(A,(d))$ 满足定义 2.3 中除 boundedness 外的 prism 条件。

**证明.** 理想 $(d)$ 由 nonzerodivisor 生成，因此定义 Cartier divisor。完备性是假设。由命题 1.6，模 $(d)$ 后 $p$ 属于 $\phi(d)$ 的像生成的理想，因此 $p\in(d,\phi(d))=(d)+\phi((d))A$。证毕。

**警告 2.7.** 命题 2.6 只给出 oriented prism 的常用充分条件。一般 prism 不必在全局由一个指定生成元表示；即使局部 principal，twist $M\{i\}$ 的全局写法也应使用 $I/I^2$。

## 2.3 基本例子

**例 2.8（crystalline prism）.** 令 $A$ 为 $p$-torsionfree、$p$-complete 的 $\delta$-环，并假设 $p$ 是 nonzerodivisor。则 $(A,(p))$ 是 prism，且它自动是 bounded prism：因为 $p$ 在 $A/p$ 上为零，故
$$
(A/p)[p^\infty]=(A/p)[p].
$$

验证如下：$(p)$ 由 nonzerodivisor 生成；$A$ 是 $(p)$-complete；且 $p\in(p)$。最后一项不需要使用 $\phi(p)$。

**例 2.9（perfect prism）.** 若 $R$ 为 perfectoid ring，则
$$
(A_{\inf}(R),\ker\theta)
$$
是 perfect prism。这是外部输入定理 1.13 的直接应用。

**例 2.10（Breuil-Kisin prism）.** 令 $\mathfrak S=W(k)[[u]]$，$\phi(u)=u^p$，$E(u)$ 为 uniformizer $\pi$ 的 Eisenstein polynomial。则
$$
(\mathfrak S,(E(u)))
$$
是 prism，且 $\mathfrak S/(E(u))\cong\mathcal O_K$。这是外部输入定理 1.17 与命题 1.16 的组合。

**例 2.11（$q$-crystalline prism）.** 令
$$
A=\mathbf Z_p[[q-1]],\qquad \phi(q)=q^p,\qquad [p]_q=1+q+\cdots+q^{p-1}.
$$
则 $(A,([p]_q))$ 是 prism。该例子是 prismatic theory 与 $q$-de Rham cohomology 的接口。

**警告 2.12.** 例 2.11 的 prism 条件依赖 $[p]_q$ 的 distinguished 性。本书当前只把它作为标准外部输入例子；附录 B 后续需要补全逐项计算。

## 2.4 Relative prismatic site

**定义 2.13.** 令 $(A,I)$ 为 bounded prism，令 $X$ 为 $p$-adic formal scheme over $\operatorname{Spf}(A/I)$。Relative prismatic site $(X/A)_\Delta$ 的对象是三元组
$$
((B,J),\alpha,\eta),
$$
其中 $(B,J)$ 是 prism，$\alpha:(A,I)\to(B,J)$ 是 prism 态射，且
$$
\eta:\operatorname{Spf}(B/J)\to X
$$
是 over $\operatorname{Spf}(A/I)$ 的态射。

态射 $((B,J),\alpha,\eta)\to((B',J'),\alpha',\eta')$ 是 prism 态射 $(B,J)\to(B',J')$，与 $A$-结构和到 $X$ 的态射相容。

**约定 2.14.** 本书采用 Bhatt-Scholze 的方向约定：prismatic site 的 underlying category 通常取上述 prism-probes category 的 opposite。为避免方向混乱，本书在定义 sheaf 时总是直接写出对象值。

**定义 2.15.** 在 $(X/A)_\Delta$ 上，结构层 $\mathcal O_\Delta$ 在对象 $((B,J),\alpha,\eta)$ 上的值为
$$
\mathcal O_\Delta(B,J)=B.
$$
Hodge-Tate specialization 层 $\overline{\mathcal O}_\Delta$ 的对象值为
$$
\overline{\mathcal O}_\Delta(B,J)=B/J.
$$

**定义 2.16.** 覆盖由对象自身 ideal 决定的 completely faithfully flat prism maps 给出。也就是说，对象族 $\{(B,J)\to(B_\lambda,J_\lambda)\}$ 是覆盖，如果相应环映射 $B\to\prod_\lambda B_\lambda$ 在 derived $(p,J)$-complete 意义下 faithfully flat，并与 prism 结构相容；目标侧 completion 则使用对应的 $(p,J_\lambda)$。底 prism $(A,I)$ 只提供结构映射，不应把所有对象的 complete flatness 统一写成 $(p,I)$。

**警告 2.17.** “Flat” 在 prismatic site 中不是普通离散 flatness 的无条件替代。若底层环非 noetherian 或存在 derived completion 现象，必须使用 completed flatness 或相应 derived flatness 口径。

## 2.5 Prismatic cohomology 的定义

**定义 2.18.** 在定义 2.13 的条件下，$X$ over $(A,I)$ 的 prismatic cohomology 定义为
$$
R\Gamma_\Delta(X/A)
=R\Gamma((X/A)_\Delta,\mathcal O_\Delta)\in D(A).
$$
其 Hodge-Tate specialization 定义为
$$
R\Gamma_{\mathrm{HT},\Delta}(X/A)
=R\Gamma((X/A)_\Delta,\overline{\mathcal O}_\Delta).
$$

**命题 2.19.** $R\Gamma_\Delta(X/A)$ 带有自然的 $\phi_A$-semilinear Frobenius endomorphism。

**证明草图.** 每个 prism object $(B,J)$ 的结构层值 $B$ 带 Frobenius lift $\phi_B$。命题 2.5 保证这些 Frobenius 与态射相容，因此给出 sheaf $\mathcal O_\Delta$ 到其 Frobenius pullback 的 semilinear endomorphism。对 derived global sections 取 $R\Gamma$ 得到结论。完整证明需要处理 sheafification 和 derived functor，与 Bhatt-Scholze 的 site-theoretic 构造一致。证毕。

## 2.6 Affine probes 的逐项读法

**说明 2.20.** 在 affine 情形 $X=\operatorname{Spf}(R)$，一个 prismatic probe 可写成
$$
(B,J,\alpha,\eta),
$$
其中 $\alpha:(A,I)\to(B,J)$ 是 prism morphism，而
$$
\eta:R\to B/J
$$
是 $p$-complete $A/I$-algebra map。于是 prismatic site 同时记录两类信息：$B$ 是 prism thickening，$R\to B/J$ 是该 thickening 在 special fibre 上落到 $X$ 的方式。

**命题 2.21.** 若 $X=\operatorname{Spf}(R)$，则对象 $((B,J),\alpha,\eta)$ 给出的结构层值与 $R$ 没有直接相等关系；只有 quotient $B/J$ 接收来自 $R$ 的映射。

**证明.** 定义 2.13 要求的是 $\operatorname{Spf}(B/J)\to X$，在 affine 情形反向给出 $R\to B/J$。结构层 $\mathcal O_\Delta(B,J)=B$ 是 thickening ring。除非另有 lift $R\to B$，否则 $R$ 不映入 $B$。证毕。

**警告 2.22.** 这正是 prismatic cohomology 非平凡的原因：site 中对象不是 $R$-algebras，而是其 modulo $J$ 后才映到 $R$ 的 prism thickenings。

## 本章小结

本章给出了 prism、有界 prism、基本例子和 relative prismatic site 的定义。Prismatic cohomology 是该 site 上结构层的 derived global sections。所有比较定理都发生在这个定义之后，不能作为定义本身的一部分。

## 练习

**练习 2.1.** 设 $d\in A$ 是 distinguished nonzerodivisor。逐项写出命题 2.6 中 $p\in(d,\phi(d))$ 的推导。

**练习 2.2.** 对 crystalline prism $(A,(p))$，说明定义 2.3 的第四条为什么自动成立。

**练习 2.3.** 写出 $(X/A)_\Delta$ 中对象到对象的态射所需满足的全部交换条件。
