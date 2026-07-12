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

**外部输入定理 19.4（infinity-categorical localization；DKR-1--DKR-2）.** DKR-1（Hinich, arXiv:1311.4128v4, Sections 1.1.2--1.1.3）用 marked simplicial sets 的 fibrant replacement 构造每个 relative category 的 infinity-categorical localization。经典 hammock localization 给出 simplicial category
$$
L^H(\mathcal C,\mathcal W)
$$
；在 DKR-2（Proposition 1.2.1）的 fibrancy 假设下，其 homotopy coherent nerve 与上述 localization 比较。后文只使用定义 19.3 的 universal property，不把两个模型视为字面相等。

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

这里的 localization 必须连同 universe 一起解释。若 $\mathcal M$ 是 $\mathcal U$-小范畴，就在 $\mathcal U$ 层使用定义 19.3。通常的模型范畴并不 $\mathcal U$-小；此时默认把 $(\mathcal M,\mathcal W_\mathcal M)$ 视为 $\mathcal V$-小 relative category，并在更大的 universe 中构造 localization；若它连 $\mathcal V$-小也不是，则继续进入 $\mathcal W$ 或另行指定更大的 universe。本书仍用同一符号 $\mathcal M_\infty$，但不声称它属于原来的小性层级。

不能仅因 $\mathcal M$ 很大就任取一个 $\mathcal U$-小 full subcategory 并称其为“representatives”。只有以下两类额外输入允许小化：

1. $\mathcal M$ 本质 $\mathcal U$-小，此时可取与 $\mathcal M$ 等价的 $\mathcal U$-小 full subcategory；
2. 已给出 $\mathcal U$-小 relative category $(\mathcal C,\mathcal V_\mathcal C)$ 及 relative functor
   $$
   (\mathcal C,\mathcal V_\mathcal C)\longrightarrow
   (\mathcal M,\mathcal W_\mathcal M)
   $$
   并证明它是 DK-equivalence，此时称 $(\mathcal C,\mathcal V_\mathcal C)$ 为一个小 DK presentation，且可用 $L(\mathcal C,\mathcal V_\mathcal C)$ 呈现 $\mathcal M_\infty$。

**外部输入定理 19.8（fibrant-cofibrant presentation；DKR-5--DKR-6）.** 若 $\mathcal M$ 是 simplicial model category，并且 $\mathcal M^{cf}$ 是 cofibrant-fibrant objects 的 full simplicial subcategory，则
$$
N_\Delta(\mathcal M^{cf})\to \mathcal M_\infty
$$
是 equivalence of infinity-categories，其中 $N_\Delta$ 为 homotopy coherent nerve。来源为 Hinich Propositions 1.3.4--1.3.5 对 fibrant/cofibrant subcategories 与 simplicial presentation 的比较。

本定理中的 $\mathcal M^{cf}$ 在一般情形仍是大 simplicial category；只取 cofibrant-fibrant objects 给出 DK-equivalent presentation，但不会自动给出小 presentation。其 nerve 与上式 localization 按定义 19.7 在同一个扩大的 universe 中解释。

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

**外部输入定理 19.10（Quillen-equivalence passage；DKR-7）.** 若上面的 Quillen adjunction 是 Quillen equivalence，则
$$
\mathbf L F:\mathcal M_\infty\to\mathcal N_\infty
$$
是 equivalence of infinity-categories。Hinich Proposition 1.5.1（DKR-7）构造 induced adjunction；Quillen equivalence 的 derived unit/counit criterion 使该 adjunction 成为 equivalence。

**说明 19.11.** 这就是“Quillen equivalent models present the same infinity-category”的精确含义。它不是说两个模型范畴同构，也不是说它们的对象逐个相等。

## 19.3 Straightening 与 coCartesian fibrations

设 $S$ 是 infinity-category。

**定义 19.12.** CoCartesian fibration over $S$ 是 inner fibration
$$
p:X\to S
$$
使得 $S$ 中每条边 $s\to t$ 和每个 $x\in X_s$ 都存在 $p$-coCartesian lift $x\to y$ over $s\to t$。

**外部输入定理 19.13（straightening/unstraightening；HTT-1）.** 对任意 infinity-category $S$，Lurie *Higher Topos Theory* Theorem 3.2.0.1 给出 coCartesian fibrations over $S$ 的 infinity-category 与 functor infinity-category 的等价
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

**命题 19.17.** 定义 19.16 与定义 18.15 的 operad map 定义等价。

**证明.** 一个 section $s$ 等价于给出 over $N(\mathbf{Fin}_*)$ 的 map
$$
A:\mathcal O^\otimes\to\mathcal C^\otimes
$$
因为 fiber product 的点是二元组 $(x,A(x))$，投影到第一因子为 $x$。Section 保持 inert coCartesian lifts 的条件正是 map of infinity-operads 的结构保持条件。因此二者给出相同的对象和 morphisms。$\square$

**外部输入定理 19.18（spaces-valued operadic straightening；PRA-4）.** 设 $\mathcal O^\otimes$ 是 Lurie infinity-operad。Pratali, arXiv:2501.05263v2, Theorem 5.1 给出 equivalence
$$
\operatorname{Left}^{\mathrm{opd}}_{\mathcal O^\otimes}
\simeq
\operatorname{Alg}_{\mathcal O^\otimes}(\mathcal S^\times),
$$
左侧是 $\mathcal O^\otimes$ 上的 operadic left fibrations，右侧是 spaces 中的 $\mathcal O$-algebras。该来源是 2025 preprint，本书将其作为 P1 locator；它不证明任意 symmetric monoidal $\mathcal C$-值代数的 straightening。

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

**外部输入定理 19.21（underlying symmetric monoidal infinity-category；HA-MON-1）.** 设 $\mathcal M$ 是 Lurie *Higher Algebra* 第 4.1.7 节意义下的 symmetric monoidal model category：单位 cofibrant，幺半结构 closed，且 tensor product 是 left Quillen bifunctor。令 $\mathcal M^c$ 为 cofibrant objects 的全子范畴，$\mathcal W^c$ 为其 weak equivalences。则
$$
N(\mathcal M^c)[(\mathcal W^c)^{-1}]
$$
自然成为 symmetric monoidal infinity-category，其底层 infinity-category 等价于 $\mathcal M_\infty=L(\mathcal M,\mathcal W)$，tensor product 是 derived tensor product。来源为 HA Proposition 4.1.7.4 与 Example 4.1.7.6（HA-MON-1）。若 $\mathcal M$ 还是相容的 simplicial symmetric monoidal model category，则 HA Corollary 4.1.7.16（HA-MON-2）用 fibrant-cofibrant objects 的 operadic nerve 给出显式模型
$$
\mathcal M_\infty^\otimes\to N(\mathbf{Fin}_*).
$$

**定位说明 19.21.1.** HA-MON-1--HA-MON-2 构造原模型范畴的 underlying symmetric monoidal infinity-category。White WHT-4 处理进一步作 Bousfield localization 时何时仍为 monoidal；若还要求该 localization 保持 operad 或 colored-operad algebra structures，则使用 WHT-1--WHT-3 和 WY-1--WY-3。这三个问题不能共用一个“monoidal localization”缩写。

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
该式沿用定义 19.7 的大小约定：代数模型范畴若不是 $\mathcal U$-小，localization 默认进入更大 universe；只有本质小性或一个已验证的 small DK presentation 才允许把它小化。

若 $\mathcal O$ 同时给出 $\mathcal M_\infty$ 中的 infinity-operad $\mathcal O^{\operatorname{loc}}$，则可比较
$$
\operatorname{Alg}_{\mathcal O}(\mathcal M)_\infty
\quad\text{与}\quad
\operatorname{Alg}_{\mathcal O^{\operatorname{loc}}}(\mathcal M_\infty).
$$

**外部输入定理 19.25（simplicial strict-to-infinity comparison；PSAR-5）.** 设 $\mathcal M$ 是 PSAR-5 所允许的 simplicial symmetric monoidal model category，$\mathcal O$ 是 simplicial colored operad，$\mathcal O$ admissible，并且 projective cofibrant replacement
$$
Q\mathcal O\longrightarrow\mathcal O
$$
满足来源要求的 symmetric flatness。则 Pavlov--Scholbach, arXiv:1410.5675v4, Theorem 7.11 给出 equivalence of infinity-categories
$$
\operatorname{Alg}_{\mathcal O}(\mathcal M)_\infty
\simeq
\operatorname{Alg}_{\mathcal O^{\operatorname{loc}}}(\mathcal M_\infty).
$$

**定位说明 19.25.1.** White WHT-1--WHT-3 和 White--Yau WY-1--WY-3 支撑的是 Bousfield localization preserves operad/colored-operad algebra structures 的模型范畴断言。它们不足以单独推出定理 19.25 的 infinity-categorical equivalence；该 equivalence 应按 P0 引用定位批次 9 中 PSAR-5--PSAR-6、HA-ALG-1--HA-ALG-3 或 P0 引用定位批次 10 中 DKR-7 的模型依赖输入使用。

**说明 19.26.** HA-ALG-1 与 HA-ALG-2 分别给出 associative 和 commutative algebra 的其他精确版本，不能替代任意 colored operad 的 PSAR-5 假设。对一般底环上的 commutative dg algebras、未验证 admissibility 的 operads 或不良 monoidal model categories，不能直接套用本条。

**命题 19.27.** 若 $\varphi:\mathcal O\to\mathcal P$ 是满足外部输入定理 14.26 的 operad weak equivalence，则 localization 后有 equivalence
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
\begin{array}{ccc}
\operatorname{Alg}_{\mathcal O}(\mathcal M) & \xrightarrow{L} & \operatorname{Alg}_{\mathcal O}(\mathcal M)_\infty\\
{\scriptstyle \mathrm{rectification}}\downarrow & & \downarrow{\scriptstyle \simeq}\\
\operatorname{Alg}_{\mathcal P}(\mathcal M) & \xrightarrow{L} & \operatorname{Alg}_{\mathcal P^{\operatorname{loc}}}(\mathcal M_\infty)
\end{array}
$$

这里 $\mathcal P^{\operatorname{loc}}$ 表示由 $\mathcal P$ 经 localization 或 category-of-operators construction 得到的 infinity-operad，不是定义 10.1 和定义 L.1 中的 Koszul resolution $\mathcal P_\infty=\Omega\mathcal P^¡$。

此图式不是一个无条件定理；每条箭头都需要独立假设：

1. 左侧竖箭头需要 Quillen equivalence。
2. 上下横箭头是 Dwyer-Kan localization。
3. 右侧竖箭头需要 algebra localization comparison。
4. 若 $\mathcal P^{\operatorname{loc}}$ 取 Lurie-style 模型，还需要 dendroidal/Lurie 或 strict/category-of-operators 比较。

**警告 19.28.** “先取代数再 localization”与“先 localization 再取代数”不自动交换。它们交换正是定理 19.25 类型结果的内容。

**说明 19.29.** 规则 M.18 把本章使用的 localization、dendroidal-Lurie 比较、category of operators nerve 和 algebra localization comparison 组织成允许路径。最终版若引用跨模型结论，应在正文旁标明使用规则 M.18 中哪一条路径，并检查警告 M.19 的禁止捷径。

**警告 19.29.1（constants 与 HHM 路径）.** HHM-1--HHM-5 在本书用于 dendroidal--Lurie 比较的 zig-zag 带 open/no-constants 限制。本书默认 operad 允许 arity $0$，所以默认对象不能未经处理沿该路径移动。Category-of-operators nerve 与 HA-OP locator 是另一条入口，也不由 HHM zig-zag 的缩写自动给出。

## 19.8 本章小结

模型范畴通过 Dwyer-Kan localization 给出 underlying infinity-category。Quillen equivalence 在 localization 后成为 infinity-categorical equivalence。Straightening/unstraightening 把 coCartesian fibrations 与 category-valued functors 等价起来；operadic straightening 在此基础上加入 inert morphisms。Symmetric monoidal localization 和 algebra localization comparison 是把严格 operad 代数与 infinity-operad 代数连接起来的关键定理，但它们都需要 cofibrancy、admissibility 和 monoidal compatibility 假设。

## 练习

**练习 19.1.** 证明 relative functor 诱导 localized infinity-categories 之间的函子。

**练习 19.2.** 设 $\mathcal M$ 是模型范畴。解释为什么 homotopy category $\operatorname{Ho}(\mathcal M)$ 只记录 $\mathcal M_\infty$ 的 $1$-categorical truncation。

**练习 19.3.** 对 simplicial model category $\mathcal M$，说明为何只取 cofibrant-fibrant objects 仍能呈现 $\mathcal M_\infty$，并解释该替换为什么不会自动把一个大模型范畴小化。

**练习 19.4.** 写出 derived tensor product $X\otimes^{\mathbf L}Y$ 的计算步骤。

**练习 19.5.** 给出一个场景，说明“先取代数再 localization”和“先 localization 再取代数”需要比较定理才能相等。
