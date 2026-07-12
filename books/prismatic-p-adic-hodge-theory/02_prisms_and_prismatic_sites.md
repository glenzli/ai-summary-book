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

这里的 derived $(p,I)$-complete 使用定义 A.2 的 Koszul tower，理想为
$(p)+I$。Bhatt-Scholze 的术语先区分 prism 与 bounded prism；依照本书
全局约定，后文未加修饰而进入 site 或 comparison theorem 的 prism 均为
bounded prism。

**定义 2.4.** Prism 态射 $(A,I)\to(B,J)$ 是保持 $\delta$-结构的环同态 $A\to B$，并满足 $I$ 的像落在 $J$ 中。

**外部输入引理 2.4A（prism ideal rigidity）.** 若
$(A,I)\to(B,J)$ 是 prisms 的态射，则事实上
$$
J=IB.
$$
这是 Bhatt-Scholze, Proposition 3.5（locator `BS-PRISM-DEF`）。证明使用
prism 条件和 Cartier divisor 结构，本书不重证。特别地，固定 base prism
后，probe ideal 不是独立变化的数据。

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

**例 2.8（crystalline prism）.** 令 $A$ 为 $p$-torsionfree、classically
$p$-adically complete 的 $\delta$-环。则 $(A,(p))$ 是 prism，且它自动是
bounded prism：因为 $p$ 在 $A/p$ 上为零，故
$$
(A/p)[p^\infty]=(A/p)[p].
$$

验证如下：$p$-torsionfreeness 说明 $(p)$ 由 nonzerodivisor 生成；命题
A.4 说明经典 $p$-完备性蕴含此处所需的 derived $p$-完备性；且
$p\in(p)$。最后一项不需要使用 $\phi(p)$。

**例 2.9（perfect prism）.** 若 $R$ 为 perfectoid ring，则
$$
(A_{\inf}(R),\ker\theta)
$$
是 perfect prism。这是外部输入定理 1.13 的直接应用。

**例 2.10（Breuil-Kisin prism）.** 令 $\mathfrak S=W(k)[[u]]$，$\phi(u)=u^p$，$E(u)$ 为 uniformizer $\pi$ 的 Eisenstein polynomial。则
$$
(\mathfrak S,(E(u)))
$$
是 bounded prism，且 $\mathfrak S/(E(u))\cong\mathcal O_K$。Prism 条件来自
外部输入定理 1.17 与命题 1.16；boundedness 来自
$\mathcal O_K$ 无 $p$-torsion 及命题 A.8。

**例 2.11（$q$-crystalline prism）.** 令
$$
A=\mathbf Z_p[[q-1]],\qquad \phi(q)=q^p,\qquad [p]_q=1+q+\cdots+q^{p-1}.
$$
则 $(A,([p]_q))$ 是 prism。该例子是 prismatic theory 与 $q$-de Rham cohomology 的接口。

**警告 2.12.** 例 2.11 的 prism 条件依赖 $[p]_q$ 的 distinguished 性。本书当前只把它作为标准外部输入例子；附录 B 后续需要补全逐项计算。

## 2.4 Relative prismatic site

**定义 2.13（relative prismatic probe category）.** 令 $(A,I)$ 为
bounded prism，令 $X$ 为 $p$-adic formal scheme over
$\operatorname{Spf}(A/I)$。定义 algebraic probe category
$\mathcal P_{X/A}$。它的对象是三元组
$$
((B,IB),\alpha,\eta),
$$
其中 $(B,IB)$ 是 bounded prism，
$\alpha:(A,I)\to(B,IB)$ 是 prism 态射，且
$$
\eta:\operatorname{Spf}(B/IB)\to X
$$
是 over $\operatorname{Spf}(A/I)$ 的态射。

在 $\mathcal P_{X/A}$ 中，从 $B$-probe 到 $C$-probe 的态射是 prism 态射
$(B,IB)\to(C,IC)$，并与 $A$-结构和到 $X$ 的态射相容。引理 2.4A 保证
目标 ideal 必为 $IC=(IB)C$。

**约定 2.14（site 的方向）.** Relative prismatic site 的 underlying
category 定义为
$$
(X/A)_\Delta=\mathcal P_{X/A}^{\mathrm{op}}.
$$
因此 site 中从 $C$-probe 到 $B$-probe 的箭头对应环方向的
$\delta$-$A$-algebra map $B\to C$。下文说 “$B\to C$ 给出 cover” 时，
指的是 site 中反方向的覆盖箭头。

**定义 2.15（结构预层）.** 在 $(X/A)_\Delta$ 上定义
$$
\mathcal O_\Delta(B,IB)=B.
$$
Hodge-Tate specialization 预层的对象值为
$$
\overline{\mathcal O}_\Delta(B,IB)=B/IB.
$$
限制映射由环映射 $B\to C$ 给出。它们分别是 commutative
$\delta$-$A$-algebra-valued 与 commutative $A/I$-algebra-valued
presheaves。

**定义 2.16（flat topology）.** Flat topology 由下列单覆盖生成：site
中的箭头从 $C$-probe 到 $B$-probe 是 cover，当且仅当对应的 prism map
$$
(B,IB)\longrightarrow(C,IC)
$$
使 $C$ 作为 $B$-complex 是 $(p,IB)$-completely faithfully flat，含义见
定义 A.13。特别地，模 $(p,IB)$ 后的映射必须 faithfully flat；这不是只
要求 ordinary $B\to C$ flat。

Bhatt-Scholze, Corollary 3.12 与 Definition 4.1（locator
`BS-PRISM-SITE`）是此处的外部输入：它们证明上述 covers 确实定义
Grothendieck topology，并证明定义 2.15 的两个 presheaves 都是 sheaves。
此外有自然同构
$$
\mathcal O_\Delta\otimes_A^LA/I\simeq
\overline{\mathcal O}_\Delta.
$$

**警告 2.17.** Cover 的 complete ideal 是 source probe ring $B$ 中的
$(p,IB)$；底 prism 上写作 $(p,I)$，经 $A\to B$ 延拓后才得到它。把 probe
ideal 当作任意 $J$，或只检查 $B/IB\to C/IC$ faithfully flat 而不检查
complete flatness，都会改变 site。

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

**命题 2.19（Frobenius 的类型）.** 令
$C=R\Gamma_\Delta(X/A)$。它带有自然的 $\phi_A$-semilinear map
$\varphi_C:C\to C$。等价地，它带有 $D(A)$ 中的 $A$-linear map
$$
\varphi_C^{\mathrm{lin}}:
\phi_A^\ast C=A\otimes_{A,\phi_A}^LC\longrightarrow C.
$$

**证明.** 每个 prism object $(B,IB)$ 的结构层值 $B$ 带 Frobenius lift
$\phi_B$。若 $f:(B,IB)\to(B',IB')$ 是 prism 态射，则它是 $\delta$-环态射，
故 $f\circ\phi_B=\phi_{B'}\circ f$。因此对象值上的 $\phi_B$ 组成结构预层
$\mathcal O_\Delta$ 的自然、乘法且 $\phi_A$-semilinear 变换。它与限制
映射相容，故在定义 2.16 的 sheaf 上仍有定义。对 underlying complexes of
abelian groups 取 derived global sections 得到 $\varphi_C$；等式
$\varphi_C(ac)=\phi_A(a)\varphi_C(c)$ 来自对象值上的同一等式。由
semilinear map 与 linearization 的自然对应，得到所述 $D(A)$-morphism。
证毕。

**警告 2.19A.** 命题 2.19 不断言
$\varphi_C^{\mathrm{lin}}$ 在积分层为同构。若 $X$ smooth，Bhatt-Scholze,
Corollary 15.5 才给出它在 invert $I$ 后成为同构；第三章把这一深结果标为
外部输入定理 3.10。

## 2.6 Affine probes 的逐项读法

**说明 2.20.** 在 affine 情形 $X=\operatorname{Spf}(R)$，一个 prismatic probe 可写成
$$
(B,IB,\alpha,\eta),
$$
其中 $\alpha:(A,I)\to(B,IB)$ 是 bounded prism morphism，而
$$
\eta:R\to B/IB
$$
是 $p$-complete $A/I$-algebra map。于是 prismatic site 同时记录两类信息：
$B$ 是 prism thickening，$R\to B/IB$ 是该 thickening 在 special fibre 上
落到 $X$ 的方式。

**命题 2.21.** 若 $X=\operatorname{Spf}(R)$，则对象
$((B,IB),\alpha,\eta)$ 给出的结构层值与 $R$ 没有直接相等关系；只有
quotient $B/IB$ 接收来自 $R$ 的映射。

**证明.** 定义 2.13 要求的是 $\operatorname{Spf}(B/IB)\to X$，在
affine 情形反向给出 $R\to B/IB$。结构层
$\mathcal O_\Delta(B,IB)=B$ 是 thickening ring。除非另有 lift
$R\to B$，否则 $R$ 不映入 $B$。证毕。

**警告 2.22.** 这正是 prismatic cohomology 非平凡的原因：site 中对象
不是 $R$-algebras，而是 modulo $IB$ 后才接收 $R$-algebra structure 的
prism thickenings。

## 本章小结

本章给出了 prism、有界 prism、基本例子和 relative prismatic site 的定义。Prismatic cohomology 是该 site 上结构层的 derived global sections。所有比较定理都发生在这个定义之后，不能作为定义本身的一部分。

## 练习

**练习 2.1.** 设 $d\in A$ 是 distinguished nonzerodivisor。逐项写出命题 2.6 中 $p\in(d,\phi(d))$ 的推导。

**练习 2.2.** 对 crystalline prism $(A,(p))$，说明定义 2.3 的第四条为什么自动成立。

**练习 2.3.** 写出 $(X/A)_\Delta$ 中对象到对象的态射所需满足的全部交换条件。
