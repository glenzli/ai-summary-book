# 附录 G：模型结构假设、admissibility 与 rectification 检查表

## 本附录目标

第十四章中许多结论不能只由“$\mathcal M$ 是对称幺半模型范畴”推出。本附录把常用假设拆成可检查的清单，供后续逐章引用。它不替代 Berger-Moerdijk、Hinich、Fresse、Pavlov-Scholbach 或 Lurie 的定理；大型存在性和 rectification 结论仍作为外部输入。

## G.1 三个不同问题

**定义 G.1.** 在对称幺半模型范畴 $\mathcal M$ 中，operad 同伦理论至少包含三个不同问题：

1. **Operad 模型结构问题。** $\operatorname{Op}(\mathcal M)$ 是否存在模型结构，并且弱等价、fibration 是否逐 arity 检测。
2. **固定 operad 的代数模型结构问题。** 对给定 operad $\mathcal O$，$\operatorname{Alg}_{\mathcal O}(\mathcal M)$ 是否存在由底层对象转移来的模型结构。
3. **Rectification 问题。** 若 $\varphi:\mathcal O\to\mathcal P$ 是 operad 弱等价，是否有 Quillen equivalence
   $$
   \varphi_!:\operatorname{Alg}_{\mathcal O}(\mathcal M)
   \rightleftarrows
   \operatorname{Alg}_{\mathcal P}(\mathcal M):\varphi^\*.
   $$

**警告 G.2.** 这三个问题没有形式蕴含关系。即使 operads 自身有 transferred 模型结构，也不自动推出每个 operad 的代数范畴 admissible；即使两个 operads 逐 arity 弱等价，也不自动推出其代数范畴 Quillen equivalent。

## G.2 基础假设包

**定义 G.3.** 设 $\mathcal M$ 是模型范畴。称 $\mathcal M$ 满足基础转移包 T0，若：

1. $\mathcal M$ 完备且余完备；
2. $\mathcal M$ cofibrantly generated，且生成 cofibrations、生成 trivial cofibrations 的源满足所需小性；
3. $\mathcal M$ 是闭对称幺半范畴；
4. pushout-product axiom 和 unit axiom 成立；
5. monoid axiom 成立。

**定义 G.4.** 称 $\mathcal M$ 满足本书的 BM-operad 转移包 T1，若除 T0 外还指定以下数据与性质：

1. 单位对象 cofibrant；
2. symmetric monoidal fibrant replacement；
3. Berger--Moerdijk 来源意义下的 commutative Hopf interval；
4. 对每个 $n$，$\mathcal M^{\Sigma_n}$ 有 projective 模型结构；
5. 生成映射满足 BM-1 所需小性。

T1 是用于外部输入定理 G.11 的一个充分包，不声称是必要条件。

**定义 G.5.** 固定 $\mathcal U$-小颜色集 $C$。称 $\mathcal M$ 满足 PSAR-colored admissibility 包 T2，若它满足 PSAR-1--PSAR-2 的来源假设：$\mathcal M$ 是 combinatorial 或 admissibly generated、tractable 的 symmetric monoidal model category，并且 symmetric h-monoidal。

**定义 G.6.** 设 $\mathcal M$ 满足 T2，且 $\varphi:\mathcal O\to\mathcal P$ 是 admissible $C$-colored operads 之间的态射。称 $(\mathcal M,\varphi)$ 满足 rectification 包 T3，若 $\varphi$ symmetric flat；更一般地，也可直接要求 PSAR-4 的 free-cofibrant-algebra comparison condition。

**说明 G.7.** T0 是 monoid 层面的常见入口；T1 是 BM-1 operad 自身模型结构的一个明确入口；T2 通过 PSAR-2 控制固定 colored operad 的代数范畴；T3 通过 PSAR-4 控制给定 operad 态射的 rectification。Pretty smallness、quasi-tractability 或其他文献版本只有在所引定理实际要求时加入，不能与上述词互换。

## G.3 自由代数中的对称幂风险

**命题 G.8.** 设 $\mathcal O$ 是单色 $R$-线性 operad。其自由代数函子具有形式
$$
F_{\mathcal O}(X)
\cong
\bigoplus_{n\ge0}
\mathcal O(n)\otimes_{R[\Sigma_n]}X^{\otimes n}.
$$
因此若 coinvariants 或对称幂不保持 quasi-isomorphism，则 transferred 模型结构或 rectification 可能失败。

**证明.** 这是第六章 Schur functor 公式在 $\mathcal O$ 上的应用。自由代数底层对象为 $S_{\mathcal O}(X)$，而 $S_{\mathcal O}$ 正是上述直和。若 $X\to Y$ 是 quasi-isomorphism，但某个
$$
X^{\otimes n}_{\Sigma_n}\to Y^{\otimes n}_{\Sigma_n}
$$
不是 quasi-isomorphism，则自由代数函子不保持相应的 trivial cofibration 候选，转移模型结构的小对象论证无法按标准方式推出弱等价保持。Rectification 中同样需要这些对称幂表达式保持 weak equivalence，否则 extension of scalars 不能在 derived 层面保持等价。$\square$

**例 G.9.** 若 $k$ 是特征 $0$ 的域，有限群代数 $k[\Sigma_n]$ 半单。此时 invariants 与 coinvariants exact，许多对称幂同调问题可由 Maschke 定理化简。

**警告 G.10.** 若 $R$ 是一般交换环或正特征域，$R[\Sigma_n]$ 通常不半单，coinvariants 不 exact。特别地，不能从 $E_\infty\to\operatorname{Com}$ 是逐 arity quasi-isomorphism 推出 $E_\infty$-algebras 与 strict commutative dg algebras 的同伦理论无条件相同。

## G.4 常见底范畴状态表

下表只给本书使用层面的状态，不替代文献定理。

| 底范畴 $\mathcal M$ | 常用模型结构 | operad/代数状态 | rectification 风险 |
| --- | --- | --- | --- |
| $\mathbf{sSet}$ | Kan-Quillen | simplicial operads 有良好模型结构，需处理 $\Sigma$-作用 | 与 infinity-operad 比较需外部定理 |
| compactly generated $\mathbf{Top}$ | weak homotopy equivalences | topological operads 可用 Berger-Moerdijk/Boardman-Vogt 技术 | 点集条件和 well-pointed 假设不可省 |
| $\mathbf{Ch}_k$, $\operatorname{char}k=0$ | projective quasi-isomorphism | dg-operads 与许多代数模型结构较稳定 | Com 与 $E_\infty$ rectification 常可用，但仍需 cofibrancy 假设 |
| $\mathbf{Ch}_R$, 一般 $R$ | projective quasi-isomorphism | associative 情形较好，commutative 与 general symmetric operads 更敏感 | 对称幂不 exact，$E_\infty$ 与 Com 不可无条件等价 |
| symmetric spectra / chain spectra | stable model structures | operads 和 structured ring spectra 有专门 admissibility 定理 | positive model structures、flatness 和 unit 条件需逐条说明 |

## G.5 Operad 模型结构检查表

若正文声明 $\operatorname{Op}(\mathcal M)$ 有 transferred 模型结构，必须至少记录：

1. $\mathcal M$ 的具体模型结构；
2. $\mathcal M$ 是否 cofibrantly generated 或 combinatorial；
3. 单位对象是否 cofibrant，若否，unit axiom 如何使用；
4. monoid axiom 或替代条件；
5. $\mathcal M^{\Sigma_n}$ 的 projective 模型结构；
6. 自由 operad 函子是否存在；
7. 弱等价和 fibration 是否逐 arity 创建；
8. 引用的外部定理及其版本。

**外部输入定理 G.11（operad transferred model structure；BM-1）.** 若 $\mathcal M$ 满足定义 G.4 的 T1，则 Berger--Moerdijk, arXiv:math/0206094v3, Theorem 3.1 给出 $\operatorname{Op}(\mathcal M)$ 的 transferred 模型结构，弱等价和 fibration 由底层对称序列逐 arity 检测。

这是外部存在性定理；本书只在命题 14.15 中证明其一旦存在便形成 Quillen adjunction。Fresse 或 Pavlov--Scholbach 的替代版本必须以各自的完整假设包另行引用，不能只写作者名。

## G.6 代数范畴 admissibility 检查表

若正文声明 $\mathcal O$ admissible，必须至少记录：

1. $\mathcal O$ 是单色还是 colored；
2. 颜色集是否 small；
3. $\mathcal O$ 是否 $\Sigma$-cofibrant、entrywise cofibrant、admissible 或 symmetric flat；
4. 自由 $\mathcal O$-代数函子是否保存生成 trivial cofibrations 产生的弱等价；
5. 代数范畴弱等价和 fibration 是否逐颜色检测；
6. 若 $\mathcal O=\operatorname{Com}$，底范畴是否满足 commutative monoid axiom 或等价替代条件。

**外部输入定理 G.12（colored admissibility；PSAR-2）.** 固定 $\mathcal U$-小颜色集 $C$。若 $\mathcal M$ 满足定义 G.5 的 T2，则 Pavlov--Scholbach, arXiv:1410.5675v4, Theorem 5.11 断言每个 $C$-colored symmetric operad admissible；其代数范畴的 weak equivalences 与 fibrations 由 $\mathcal M^C$ 创建。

Symmetric flatness 不是这条存在性定理在本书所用版本中的附加结论；它在 G.13 的 rectification 中出现。

## G.7 Rectification 检查表

若正文声明
$$
\varphi_!:\operatorname{Alg}_{\mathcal O}(\mathcal M)
\rightleftarrows
\operatorname{Alg}_{\mathcal P}(\mathcal M):\varphi^\*
$$
是 Quillen equivalence，必须至少记录：

1. $\mathcal O$ 和 $\mathcal P$ 均 admissible；
2. $\varphi:\mathcal O\to\mathcal P$ 是何种 weak equivalence；
3. $\mathcal O$ 是否 cofibrant、$\Sigma$-cofibrant 或 flat；
4. $\mathcal M$ 是否满足 symmetric flatness 或文献中的替代条件；
5. 代数对象是否需要 cofibrant replacement 后比较；
6. 结论是 Quillen equivalence、derived equivalence 还是 infinity-categorical equivalence。

**外部输入定理 G.13（rectification；PSAR-4）.** 若 $(\mathcal M,\varphi)$ 满足定义 G.6 的 T3，则 Pavlov--Scholbach, arXiv:1410.5675v4, Theorem 7.5 给出
$$
\varphi_!:\operatorname{Alg}_{\mathcal O}(\mathcal M)
\rightleftarrows
\operatorname{Alg}_{\mathcal P}(\mathcal M):\varphi^\*
$$
为 Quillen equivalence。若不用 symmetric flatness 这一充分条件，则必须逐字验证该定理的 free-cofibrant-algebra comparison condition。

**警告 G.14.** “$\mathcal O(n)\to\mathcal P(n)$ 对每个 $n$ 是 weak equivalence”只是 rectification 的输入之一，不是结论本身。结论还需要自由代数中的所有对称幂表达式同伦良好。

## G.8 正特征中的典型边界

**说明 G.15.** 在正特征中，$E_\infty$-algebra 通常携带由 $E_\infty$ 结构诱导的 power operations。Strict commutative dg algebra 强加严格交换乘法。二者的同伦理论是否等价取决于底范畴和模型结构；一般不能以 $E_\infty\to\operatorname{Com}$ 的逐 arity quasi-isomorphism 单独证明。

**命题 G.16（正特征中的内部障碍）.** 对 $k=\mathbb F_p$，存在 acyclic chain complex $C$，使自由严格交换 dg algebra 的正次数部分含非零同调；特别地，
$$
\operatorname{Sym}(0)\longrightarrow\operatorname{Sym}(C)
$$
不是 quasi-isomorphism。因此任何要求自由交换代数函子保持该 weak equivalence 的 transferred-structure 或 rectification 判据在 $\mathbf{Ch}_{\mathbb F_p}$ 中不满足。

**证明.** 命题 X.15 构造 $C=(k\cdot y\xrightarrow{d}k\cdot x)$，$|y|=2$、$|x|=1$、$dy=x$，并证明 $y^p$ 在 $\operatorname{Sym}^p(C)$ 中是非零同调类。推论 X.16 说明该类在 $\operatorname{Sym}(C)$ 中仍非零，而 $0\to C$ 是 trivial cofibration。最后一句只应用“该判据要求自由函子保持此 weak equivalence”的定义，不声称由这个例子单独分类所有可能模型结构。$\square$

**外部边界 G.17.** Power operations 给出的更强非 rectification 结论属于 Mandell、Hinich、Pavlov--Scholbach 或 structured ring spectra 文献。本书没有从命题 G.16 推出这些深结论；需要时必须另引精确版本。

## G.9 本附录小结

Operad 的模型范畴理论不是一个单一开关。每当正文使用“admissible”“rectification”“cofibrant resolution 可替代原 operad”或“代数范畴 Quillen equivalent”时，都必须说明底范畴、operad、弱等价、cofibrancy/flatness 和对称幂假设。缺少这些信息时，只能把结论标为外部输入定理或研究边界。
