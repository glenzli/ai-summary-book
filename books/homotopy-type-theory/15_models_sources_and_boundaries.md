# 第十五章：模型语义、可靠性与相对一致性

前十四章一直在对象语言内部工作：语境中形成类型，给出项，再用归纳或消去规则证明恒等类型中的结论。模型语义提出另一类问题：能否把这些语境、类型和项解释成某个外部数学范畴中的对象，并让替换与计算规则都成立？一个肯定答案可以说明规则彼此相容，也可以揭示路径和单值性在几何模型中的含义，但它不会自动产生新的对象语言项。

本章先写出模型至少要保存的数据，再比较集合、simplicial 与 cubical 三种语义现象。布尔类型的自等价会给出一个可计算反例：普通集合宇宙为什么不能验证单值性。随后，可靠性、模型存在、相对一致性和保守性会被分别定义；这些概念的区分为第十六章阅读具体元定理提供类型正确的框架。

## 15.1 一个依赖类型论模型保存什么

**定义 15.1（CwF 风格的语义数据）.** 一个 category with families 风格的模型至少包含以下数据。

1. 一个语境与替换组成的范畴 $\mathcal C$，其对象解释语境，态射
   $\sigma:\Delta\to\Gamma$ 解释“在语境 $\Delta$ 中给出语境 $\Gamma$ 各变量的替换项”，并有终对象 $1$ 解释空语境；
2. 对每个 $\Gamma:\mathcal C$，一个类型集合
   $\mathsf{Ty}(\Gamma)$；对每个 $A:\mathsf{Ty}(\Gamma)$，一个项集合
   $\mathsf{Tm}(\Gamma,A)$；
3. 对替换 $\sigma:\Delta\to\Gamma$，重索引
   $$
   A\longmapsto A[\sigma]:\mathsf{Ty}(\Delta),
   \qquad
   a\longmapsto a[\sigma]:\mathsf{Tm}(\Delta,A[\sigma]),
   $$
   且恒等与复合的重索引律严格成立；
4. 对 $A:\mathsf{Ty}(\Gamma)$，语境扩张 $\Gamma.A$、投影
   $p_A:\Gamma.A\to\Gamma$ 和通用项
   $q_A:\mathsf{Tm}(\Gamma.A,A[p_A])$，满足 comprehension 的泛性质。

该泛性质说：给定 $\sigma:\Delta\to\Gamma$ 与
$a:\mathsf{Tm}(\Delta,A[\sigma])$，存在唯一
$$
\langle\sigma,a\rangle:\Delta\to\Gamma.A
$$
使 $p_A\circ\langle\sigma,a\rangle=\sigma$，且
$q_A[\langle\sigma,a\rangle]=a$。这正是语境扩张与替换规则的语义版本。

**定义 15.2（解释一个具体对象理论）.** 若对象理论 $T$ 还包含 $\Pi$、$\Sigma$、恒等类型、宇宙、单值性或指定 HIT，则 $T$ 的模型是在定义 15.1 的数据上，再给出相应的语义结构，并证明其形成、引入、消去、计算和替换稳定性。只构造底层范畴而不验证这些结构，不能称为整个 $T$ 的模型。

**命题 15.3（可靠性模式）.** 固定对象理论 $T$ 及一个逐条验证其规则的模型 $M$。若判断 $J$ 在 $T$ 中有有限推导，则 $J$ 在 $M$ 中的解释成立；若推导末步断言 judgmental equality，则两个解释由模型指定的严格相等关系识别。

**证明（元语言归纳）.** 对 $J$ 的推导树作归纳。变量、替换和语境扩张由定义 15.1 的结构解释；每个类型形成子对应定义 15.2 中已经验证的一组结构。归纳步把前提判断的解释代入末条推理规则的语义有效性。计算规则的情形使用模型为该规则验证的严格等式。该证明是关于推导的元语言归纳，不是 $T$ 内部的一个项。$\square$

可靠性只沿“语法推导 $\Rightarrow$ 模型有效”方向工作。反向方向是 completeness，需要额外的初始语法模型或完整性定理，不能从定义 15.2 自动得到。

## 15.2 普通集合模型为什么看不见单值性

考虑标准的集合族解释：语境是集合，$\Gamma$ 上的类型是集合族
$A:\Gamma\to\mathsf{Set}$，项是截面
$a:\prod_{\gamma:\Gamma}A(\gamma)$，替换是预合成，语境扩张是依赖和
$$
\Gamma.A\coloneqq\sum_{\gamma:\Gamma}A(\gamma).
$$
它自然解释 $\Pi$ 与 $\Sigma$。若恒等类型用离散集合相等解释，则
$\mathsf{Id}_A(x,y)$ 在 $x=y$ 时为单位集合，否则为空集合。

**例 15.4（布尔自等价检测单值性失败）.** 设集合宇宙 $U$ 的元素是小集合的严格代码，代码的恒等类型按集合相等解释。布尔集合 $2=\{0,1\}$ 有至少两个自等价：恒等置换与交换 $0,1$ 的置换。因此
$$
(2\simeq2)
$$
至少有两个元素。另一方面，离散恒等解释下的
$$
(2=_U2)
$$
只有反身证明。规范映射
$$
\mathsf{idtoequiv}:(2=_U2)\to(2\simeq2)
$$
不可能满射，因而不是等价。

这个反例没有否定普通集合数学；它说明该严格代码宇宙不是第六章意义下的单值宇宙。若要让每个自等价都对应宇宙 loop，语义必须允许代码之间具有非平凡路径，而不能把 universe identity 压成离散相等。

## 15.3 Simplicial 语义提供的相对一致性

Simplicial set 模型把类型解释为具有 Kan lifting 结构的纤维，把恒等类型解释为合适的路径对象。一个 universe 可以分类一类小 Kan fibration；其 universe path 不再是严格代码相等，因而有可能记录类型等价。

**外部输入定理 15.5（单值 simplicial universe）.** 在带两个不可达基数的 ZFC 元理论中，Kapulkin--Lumsdaine 构造 contextual category 模型，解释论文所指定的 Martin--Löf 类型论，并使一个 universe 满足单值性。

**来源与未重证边界.** Kapulkin--Lumsdaine, *The Simplicial Model of Univalent Foundations (after Voevodsky)*, JEMS 23 (2021), DOI `10.4171/JEMS/1050`, arXiv:1211.2851v5。模型依赖 weakly universal Kan fibration 的构造、coherence 和 contextual-category 解释；本书不重证这些内容。该来源不同时证明本书全部 HIT、normalization、canonicity 或任意 universe hierarchy。

**推论 15.6（条件性相对一致性）.** 设 $M$ 是上述元理论，$T$ 是该模型实际解释的对象理论。若 $M$ 一致，则 $T$ 一致：
$$
\mathsf{Con}(M)\Longrightarrow\mathsf{Con}(T).
$$

**证明（元语言）.** 若 $T$ 推出空类型的闭项，可靠性命题 15.3 会把它解释成模型中空对象的全局元素，与模型的集合论构造矛盾。因此 $T$ 的矛盾推导会给出 $M$ 中的矛盾。结论必须保留对 $M$ 一致性的条件。$\square$

模型中的 simplicial map 或 homotopy 不是 $T$ 的语法项。要把外部构造反射回语法，还需要 reification、completeness 或 normalization 一类额外元定理。

## 15.4 Cubical 语义把哪些内容变成计算

Cubical type theory 不只更换模型，还更换对象语言。以 CCHM calculus 为例，语法加入区间 $\mathbb I$、面格、$\mathsf{Path}$、composition/filling 与 Glue。路径项可以沿区间变量求值，Glue 则把等价组织成 universe path，并与 composition 一起产生计算行为。

**外部输入定理 15.7（CCHM 的计算单值性接口）.** 在 CCHM 指定的 de Morgan cubical calculus 中，函数外延性可内部证明，Glue 给出 universe univalence，且该语法有构造性的 cubical set 语义。

**来源与未重证边界.** Cohen--Coquand--Huber--Mörtberg, *Cubical Type Theory: A Constructive Interpretation of the Univalence Axiom*, DOI `10.4230/LIPIcs.TYPES.2015.5`，尤其是 §§4、6、7.2、8。这里的 $\mathsf{Path}$ 是 cubical 原始构造，不是第二章归纳恒等类型的另一个记号。本书不把 CCHM 的归约规则添加到第六章的公理化 $\mathsf{ua}$。

**例 15.8（同一句 transport 公式的两种身份）.** 第六章从单值性的逆律证明
$$
\mathsf{transport}^{\lambda X.X}(\mathsf{ua}(e),a)=e.1(a).
$$
这是恒等类型中的路径。CCHM 中相应 Glue 构造具有由其语法规定的归约行为。若没有从公理化语法到 CCHM 语法的解释定理，就不能用后者把前式左、右两边宣布为本书基础语言中的 judgmental equality。

## 15.5 模型比较、翻译与保守性

**定义 15.9（语法翻译）.** 从理论 $T$ 到理论 $T'$ 的翻译应把 $T$ 的语境、类型、项与替换送到 $T'$ 中相应对象，并保持每条形成、引入、消去和计算规则。若它只保持可证路径而不保持 judgmental equality，必须另行标明较弱的保持口径。

**定义 15.10（保守扩张）.** 若 $T'$ 扩张 $T$，称 $T'$ 对一类旧语言判断 $\mathcal J$ 保守，若对每个 $J\in\mathcal J$，
$$
T'\vdash J\quad\Longrightarrow\quad T\vdash J.
$$
保守性是语法可导性的反向结论，比“$T'$ 有一个也能解释 $T$ 的模型”更强。

**命题 15.11（共同模型不足以证明保守性）.** 两个理论在同一数学范畴中都有可靠模型，不足以推出其中一个对另一个保守。

**证明.** 可靠性只把语法推导送到语义有效性。即使一个旧语言判断在共同模型中成立，也需要 completeness 才能反推出它在旧理论中可导；而共同模型假设没有提供 completeness。因此从这些前提无法构造定义 15.10 所需的反向蕴含。$\square$

比较 simplicial、cubical、two-level 或 directed 语法时，真正需要的因而是具体翻译、模型解释、初始性或保守性定理，而不是名称上的相似。第十六章将按这一标准阅读 canonicity、normalization 和若干 2026 年模型分离结果。

## 15.6 语义给出的解释与不给出的项

集合族模型准确解释依赖数据，却因离散 universe identity 而不能验证单值性；simplicial 模型用高阶路径给出条件性相对一致性；cubical 语言进一步把区间、composition 和 Glue 放进语法，从而研究计算单值性。三者回答的是不同问题。一个模型可以证明规则有语义，一个可靠性定理可以传递推导，一个 normalization 定理可以分析闭项计算，而保守性还需要把扩展语言中的旧结论送回基础语法。保持这些箭头的方向，才能避免用外部几何直觉替代内部证明。

## 练习

**练习 15.1.** 在集合族模型中写出替换
$A[\sigma]$ 与项替换 $a[\sigma]$，并验证两次替换的复合律。

**练习 15.2.** 完整证明例 15.4 中交换置换不同于恒等置换，并指出该论证使用了布尔构造子的哪条可区分性。

**练习 15.3.** 解释为什么一个 simplicial 模型中的路径对象不能直接充当第二章语法中的路径项。

**练习 15.4.** 分别写出“模型存在”“可靠性”“相对一致性”“保守性”的量词方向，并给出任意两者不能混同的理由。

**练习 15.5.** 对第六章的公理化单值性与 CCHM Glue 单值性，列出比较两者所需的最少语法翻译数据。
