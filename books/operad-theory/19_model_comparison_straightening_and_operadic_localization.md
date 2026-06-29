# 第十九章：模型比较、straightening 与 operadic localization

本章解释如何从严格模型转入 infinity-categorical 模型。前面已经出现三类对象：

1. 模型范畴中的 operads 与其代数；
2. dendroidal infinity-operads；
3. Lurie-style infinity-operads over $N(\mathbf{Fin}_*)$。

它们不是同一个定义。要在它们之间移动，必须使用 localization、straightening/unstraightening 和模型比较定理。本章的原则是：凡是跨模型的结论，都必须说明经过哪个 localization 或 Quillen equivalence。

## 19.1 Relative categories 与 Dwyer-Kan localization

**定义 19.1.** 一个 relative category 是一对
$$
(\mathcal C,\mathcal W),
$$
其中 $\mathcal C$ 是小范畴，$\mathcal W\subset\mathcal C$ 是含所有对象和恒等态射的 wide subcategory。$\mathcal W$ 中的态射称为 weak equivalences。

**定义 19.2.** Relative functor
$$
F:(\mathcal C,\mathcal W)\to(\mathcal D,\mathcal V)
$$
是函子 $F:\mathcal C\to\mathcal D$，满足 $F(\mathcal W)\subset\mathcal V$。

**定义 19.3.** Relative category 的 Dwyer-Kan localization 是一个 infinity-category
$$
L(\mathcal C,\mathcal W)
$$
连同函子
$$
\ell:\mathcal C\to L(\mathcal C,\mathcal W)
$$
满足以下 universal property：对任意 infinity-category $\mathcal E$，预复合 $\ell$ 给出等价
$$
\operatorname{Fun}\big(L(\mathcal C,\mathcal W),\mathcal E\big)
\simeq
\operatorname{Fun}_{\mathcal W}(\mathcal C,\mathcal E),
$$
右端表示把 $\mathcal W$ 中态射送到 equivalences in $\mathcal E$ 的函子构成的 full subcategory。

**外部输入定理 19.4（Dwyer-Kan）.** 每个 relative category admits Dwyer-Kan localization。若采用 hammock localization，则可得到 simplicial category
$$
L^H(\mathcal C,\mathcal W)
$$
其 homotopy coherent nerve 是 $L(\mathcal C,\mathcal W)$ 的一个模型。

**命题 19.5.** 若
$$
F:(\mathcal C,\mathcal W)\to(\mathcal D,\mathcal V)
$$
是 relative functor，则它诱导函子
$$
L(F):L(\mathcal C,\mathcal W)\to L(\mathcal D,\mathcal V).
$$

**证明.** 复合
$$
\mathcal C\xrightarrow{F}\mathcal D\xrightarrow{\ell_\mathcal D}L(\mathcal D,\mathcal V)
$$
把 $\mathcal W$ 中态射送到 equivalences，因为 $F(\mathcal W)\subset\mathcal V$ 且 $\ell_\mathcal D$ invert $\mathcal V$。由定义 19.3 的 universal property，存在唯一 up to contractible choice 的函子
$$
L(\mathcal C,\mathcal W)\to L(\mathcal D,\mathcal V)
$$
使三角图交换。$\square$

**定义 19.6.** Relative functor $F$ 称为 DK-equivalence，若诱导函子 $L(F)$ 是 equivalence of infinity-categories。

## 19.2 模型范畴的 underlying infinity-category

设 $\mathcal M$ 是模型范畴，$\mathcal W_\mathcal M$ 为弱等价类。

**定义 19.7.** $\mathcal M$ 的 underlying infinity-category 定义为
$$
\mathcal M_\infty=L(\mathcal M,\mathcal W_\mathcal M).
$$

若 $\mathcal M$ 不是小范畴，则在固定 universe 中取一个小的 full subcategory of representatives，或在更大 universe 中工作。

**外部输入定理 19.8.** 若 $\mathcal M$ 是 simplicial model category，并且 $\mathcal M^{cf}$ 是 cofibrant-fibrant objects 的 full simplicial subcategory，则
$$
N_\Delta(\mathcal M^{cf})\to \mathcal M_\infty
$$
是 equivalence of infinity-categories，其中 $N_\Delta$ 为 homotopy coherent nerve。

**命题 19.9.** Quillen adjunction
$$
F:\mathcal M\rightleftarrows\mathcal N:G
$$
诱导 adjunction of infinity-categories
$$
\mathbf L F:\mathcal M_\infty\rightleftarrows\mathcal N_\infty:\mathbf R G.
$$

**证明.** 左 Quillen 函子 $F$ 不必把所有弱等价送到弱等价，但它把 cofibrant objects 之间的 weak equivalences 送到 weak equivalences。取函子性 cofibrant replacement $Q:\mathcal M\to\mathcal M$。若 $u:X\to Y$ 是 weak equivalence，则 $Qu:QX\to QY$ 也是 weak equivalence，因为 $QX\to X$ 与 $QY\to Y$ 是 weak equivalences，且 weak equivalences 满足 two-out-of-three。对象 $QX,QY$ cofibrant，因此 $F(Qu)$ 是 $\mathcal N$ 中的 weak equivalence。故复合 $FQ$ 是 relative functor
$$
(\mathcal M,\mathcal W_\mathcal M)\to(\mathcal N,\mathcal W_\mathcal N)
$$
并诱导 $\mathbf L F$。同理，$GR$ 对函子性 fibrant replacement $R$ 诱导 $\mathbf R G$。模型范畴中的 derived adjunction unit/counit 通过 localization 诱导 infinity-categorical adjunction unit/counit。完整相干性依赖模型范畴 localization 定理。$\square$

**外部输入定理 19.10.** 若上面的 Quillen adjunction 是 Quillen equivalence，则
$$
\mathbf L F:\mathcal M_\infty\to\mathcal N_\infty
$$
是 equivalence of infinity-categories。

**说明 19.11.** 这就是“Quillen equivalent models present the same infinity-category”的精确含义。它不是说两个模型范畴同构，也不是说它们的对象逐个相等。

## 19.3 Straightening 与 coCartesian fibrations

设 $S$ 是 infinity-category。

**定义 19.12.** CoCartesian fibration over $S$ 是 inner fibration
$$
p:X\to S
$$
使得 $S$ 中每条边 $s\to t$ 和每个 $x\in X_s$ 都存在 $p$-coCartesian lift $x\to y$ over $s\to t$。

**外部输入定理 19.13（straightening/unstraightening）.** 对任意 infinity-category $S$，coCartesian fibrations over $S$ 的 infinity-category 等价于 functor infinity-category
$$
\operatorname{Fun}(S,\mathbf{Cat}_\infty).
$$
该等价把
$$
p:X\to S
$$
送到函子
$$
s\mapsto X_s
$$
并把 $S$ 中的边送到 coCartesian transport。

**定义 19.14.** 若 $p:X\to S$ 是 coCartesian fibration，则其 straightening 记为
$$
\operatorname{St}_S(X):S\to\mathbf{Cat}_\infty.
$$
若 $F:S\to\mathbf{Cat}_\infty$ 是函子，则其 unstraightening 记为
$$
\operatorname{Un}_S(F)\to S.
$$

**命题 19.15.** 对 $s\in S$，有 canonical equivalence
$$
\operatorname{St}_S(X)(s)\simeq X_s.
$$

**证明.** Straightening 的构造按定义把 $p$ 的 fiber 作为 $s$ 上的值。不同模型中的构造可能先取 fibrant replacement 或 marked simplicial set replacement，但定理 19.13 保证所得值与 homotopy fiber $X_s$ 等价。$\square$

## 19.4 Operadic straightening

设
$$
\mathcal O^\otimes\to N(\mathbf{Fin}_*)
$$
是 Lurie-style infinity-operad，$\mathcal C^\otimes\to N(\mathbf{Fin}_*)$ 是 symmetric monoidal infinity-category。

**定义 19.16.** $\mathcal C$ 中的 $\mathcal O$-algebra 可等价地看作 section
$$
s:\mathcal O^\otimes\to
\mathcal O^\otimes\times_{N(\mathbf{Fin}_*)}\mathcal C^\otimes
$$
of the projection
$$
\mathcal O^\otimes\times_{N(\mathbf{Fin}_*)}\mathcal C^\otimes\to\mathcal O^\otimes
$$
满足：$s$ 把 $\mathcal O^\otimes$ 中 inert morphisms 的 chosen coCartesian lifts 送到 $\mathcal C^\otimes$ 中的 inert coCartesian lifts。

**命题 19.17.** 定义 19.16 与第十八章的 operad map 定义等价。

**证明.** 一个 section $s$ 等价于给出 over $N(\mathbf{Fin}_*)$ 的 map
$$
A:\mathcal O^\otimes\to\mathcal C^\otimes
$$
因为 fiber product 的点是二元组 $(x,A(x))$，投影到第一因子为 $x$。Section 保持 inert coCartesian lifts 的条件正是 map of infinity-operads 的结构保持条件。因此二者给出相同的对象和 morphisms。$\square$

**外部输入定理 19.18（operadic straightening）.** 在适当 marked simplicial set 或 infinity-operad 模型中，$\mathcal O$-algebras in $\mathcal C$ 可由某个 coCartesian fibration of operadic families 的 sections 描述；该描述与 Lurie-style $\operatorname{Alg}_{\mathcal O}(\mathcal C)$ 等价。

**说明 19.19.** Operadic straightening 是 ordinary straightening 的 operad 相对版本。它不仅分类 $S\to\mathbf{Cat}_\infty$，还必须保留 inert morphisms 所编码的多输入分量结构。

## 19.5 Symmetric monoidal localization

设 $\mathcal M$ 是 symmetric monoidal model category，弱等价类为 $\mathcal W$。

**定义 19.20.** 称 tensor product preserves weak equivalences between cofibrant objects，若对 cofibrant objects $X,Y,X',Y'$ 和 weak equivalences
$$
X\to X',\qquad Y\to Y',
$$
诱导态射
$$
X\otimes Y\to X'\otimes Y'
$$
是 weak equivalence。

**外部输入定理 19.21（monoidal localization）.** 若 $\mathcal M$ 是足够良好的 symmetric monoidal model category，例如 combinatorial、left proper、满足 pushout-product/unit 条件，并且 tensor product preserves weak equivalences between cofibrant objects，则 localization
$$
\mathcal M_\infty=L(\mathcal M,\mathcal W)
$$
自然成为 symmetric monoidal infinity-category
$$
\mathcal M_\infty^\otimes\to N(\mathbf{Fin}_*).
$$
其 tensor product 由 $\mathcal M$ 的 derived tensor product 给出。

**命题 19.22.** 在定理 19.21 的假设下，若 $X,Y\in\mathcal M$，则
$$
X\otimes^{\mathbf L}Y\simeq QX\otimes QY
$$
in $\mathcal M_\infty$，其中 $QX,QY$ 为 cofibrant replacements。

**证明.** Derived tensor product 是 ordinary tensor product 的左导出。左导出在 localization 中通过先取 cofibrant replacement 再应用左 Quillen bifunctor 计算。若选择不同的 cofibrant replacements，它们由 weak equivalences under $X,Y$ 相连；定义 19.20 保证 tensor 后仍为 weak equivalences，因此在 $\mathcal M_\infty$ 中给出等价。$\square$

**警告 19.23.** 若 tensor product 不保持 cofibrant objects 之间的 weak equivalences，则 ordinary tensor product 不一定 descends to localization。此时不能直接把 $\mathcal M_\infty$ 视为 symmetric monoidal infinity-category。

## 19.6 Operadic localization of algebra categories

设 $\mathcal O$ 是 $\mathcal M$ 中的 operad，并假设 $\operatorname{Alg}_{\mathcal O}(\mathcal M)$ 有 transferred model structure。

**定义 19.24.** $\mathcal O$-代数模型范畴的 operadic localization 是 underlying infinity-category
$$
\operatorname{Alg}_{\mathcal O}(\mathcal M)_\infty
=L\big(\operatorname{Alg}_{\mathcal O}(\mathcal M),\mathcal W_{\operatorname{Alg}}\big),
$$
其中 $\mathcal W_{\operatorname{Alg}}$ 是底层 $\mathcal M$ 中逐颜色 weak equivalences。

若 $\mathcal O$ 同时给出 $\mathcal M_\infty$ 中的 infinity-operad $\mathcal O^{\operatorname{loc}}$，则可比较
$$
\operatorname{Alg}_{\mathcal O}(\mathcal M)_\infty
\quad\text{与}\quad
\operatorname{Alg}_{\mathcal O^{\operatorname{loc}}}(\mathcal M_\infty).
$$

**外部输入定理 19.25（algebra localization comparison）.** 在适当的 admissibility、cofibrancy、monoidal localization 和 rectification 假设下，存在 equivalence of infinity-categories
$$
\operatorname{Alg}_{\mathcal O}(\mathcal M)_\infty
\simeq
\operatorname{Alg}_{\mathcal O^{\operatorname{loc}}}(\mathcal M_\infty).
$$

**说明 19.26.** 该定理是许多“严格代数模型呈现 infinity-categorical algebra objects”的基础。它依赖假设；在一般底环上的 commutative dg algebras、非 cofibrant operads 或不良 monoidal model categories 中不能直接套用。

**命题 19.27.** 若 $\varphi:\mathcal O\to\mathcal P$ 是满足第十四章 rectification criterion 的 operad weak equivalence，则 localization 后有 equivalence
$$
\operatorname{Alg}_{\mathcal O}(\mathcal M)_\infty
\simeq
\operatorname{Alg}_{\mathcal P}(\mathcal M)_\infty.
$$

**证明.** Rectification criterion 给出 Quillen equivalence
$$
\varphi_!:\operatorname{Alg}_{\mathcal O}(\mathcal M)
\rightleftarrows
\operatorname{Alg}_{\mathcal P}(\mathcal M):\varphi^\*.
$$
由外部输入定理 19.10，Quillen equivalence induces equivalence of underlying infinity-categories。$\square$

## 19.7 比较图式

在良好假设下，严格 operad 代数、模型范畴和 infinity-operad 代数之间的关系可概括为：
$$
\begin{CD}
\operatorname{Alg}_{\mathcal O}(\mathcal M) @>{L}>> \operatorname{Alg}_{\mathcal O}(\mathcal M)_\infty\\
@V{\text{rectification}}VV @VV{\simeq}V\\
\operatorname{Alg}_{\mathcal P}(\mathcal M) @>{L}>> \operatorname{Alg}_{\mathcal P^{\operatorname{loc}}}(\mathcal M_\infty).
\end{CD}
$$

这里 $\mathcal P^{\operatorname{loc}}$ 表示由 $\mathcal P$ 经 localization 或 category-of-operators construction 得到的 infinity-operad，不是第十章中的 Koszul resolution $\mathcal P_\infty=\Omega\mathcal P^¡$。

此图式不是一个无条件定理；每条箭头都需要独立假设：

1. 左侧竖箭头需要 Quillen equivalence。
2. 上下横箭头是 Dwyer-Kan localization。
3. 右侧竖箭头需要 algebra localization comparison。
4. 若 $\mathcal P^{\operatorname{loc}}$ 取 Lurie-style 模型，还需要 dendroidal/Lurie 或 strict/category-of-operators 比较。

**警告 19.28.** “先取代数再 localization”与“先 localization 再取代数”不自动交换。它们交换正是定理 19.25 类型结果的内容。

## 19.8 本章小结

模型范畴通过 Dwyer-Kan localization 给出 underlying infinity-category。Quillen equivalence 在 localization 后成为 infinity-categorical equivalence。Straightening/unstraightening 把 coCartesian fibrations 与 category-valued functors 等价起来；operadic straightening 在此基础上加入 inert morphisms。Symmetric monoidal localization 和 algebra localization comparison 是把严格 operad 代数与 infinity-operad 代数连接起来的关键定理，但它们都需要 cofibrancy、admissibility 和 monoidal compatibility 假设。

## 练习

**练习 19.1.** 证明 relative functor 诱导 localized infinity-categories 之间的函子。

**练习 19.2.** 设 $\mathcal M$ 是模型范畴。解释为什么 homotopy category $\operatorname{Ho}(\mathcal M)$ 只记录 $\mathcal M_\infty$ 的 $1$-categorical truncation。

**练习 19.3.** 对 simplicial model category $\mathcal M$，说明为何只取 cofibrant-fibrant objects 仍能呈现 $\mathcal M_\infty$。

**练习 19.4.** 写出 derived tensor product $X\otimes^{\mathbf L}Y$ 的计算步骤。

**练习 19.5.** 给出一个场景，说明“先取代数再 localization”和“先 localization 再取代数”需要比较定理才能相等。
