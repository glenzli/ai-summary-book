# 附录 M：Dendroidal、Lurie 与模型范畴比较图

## 本附录目标

第十六至十九章涉及多种 infinity-operad 模型。本附录给出一张可检查的依赖图，说明哪些对象属于哪个模型，哪些箭头是定义，哪些箭头是外部输入定理。

本附录的基本原则：

1. Strict operad、simplicial operad、dendroidal set、Lurie-style infinity-operad 不是同一对象。
2. 跨模型移动必须通过 nerve、localization、Quillen adjunction、Quillen equivalence 或 explicit comparison functor。
3. 任何“等价”必须说明是在模型范畴、homotopy category 还是 infinity-category 中。

## M.1 四类模型

**定义 M.1.** 本书区分以下四类 operadic 模型。

| 模型 | 对象 | 弱等价/等价 |
| --- | --- | --- |
| Strict colored operads | 集合、simplicial sets、topological spaces 或 chain complexes 中的 colored operads | 取决于底范畴和 transferred model structure |
| Dendroidal sets | presheaves $\Omega^{op}\to\mathbf{Set}$ | Cisinski-Moerdijk operadic weak equivalences |
| Lurie-style infinity-operads | inner fibrations $\mathcal O^\otimes\to N(\mathbf{Fin}_*)$ 满足 inert 条件 | equivalence in corresponding preoperad/marked model |
| Operad algebras in model categories | $\operatorname{Alg}_{\mathcal O}(\mathcal M)$ | transferred weak equivalences, then localization |

**警告 M.2.** “$\mathcal O$ 是 infinity-operad”不是完整数学陈述。必须说明它是 dendroidal inner Kan object、Lurie-style operadic fibration，还是某个模型范畴中 operad 的 localization。

## M.2 Strict operad 的两个 nerve

**定义 M.3.** 对 ordinary colored operad $\mathcal P$，dendroidal nerve 定义为
$$
N_d(\mathcal P)_T=\operatorname{Hom}_{\operatorname{Operad}}(\Omega(T),\mathcal P).
$$
它是 dendroidal set。

**定义 M.4.** 对 ordinary colored operad $\mathcal P$，category of operators $\mathcal P^\otimes\to\mathbf{Fin}_*$ 的 nerve
$$
N(\mathcal P^\otimes)\to N(\mathbf{Fin}_*)
$$
是 Lurie-style 模型中的对象。

**命题 M.5.** $N_d(\mathcal P)$ 与 $N(\mathcal P^\otimes)$ 位于不同范畴，不能逐项相等。

**证明.** $N_d(\mathcal P)$ 是 $\Omega$ 上的 presheaf，其 $T$-simplices 由 $\Omega(T)\to\mathcal P$ 给出。$N(\mathcal P^\otimes)$ 是到 $N(\mathbf{Fin}_*)$ 的 simplicial set，其 $n$-simplices 是 category of operators 中的可复合箭头链。源范畴 $\Omega$ 与 $\Delta/\mathbf{Fin}_*$ 不同，取值对象也不同。因此没有逐项相等的意义。它们只能经比较构造联系。$\square$

## M.3 Dendroidal 模型内部

**定义 M.6.** Dendroidal infinity-operad 是 inner Kan dendroidal set。Strict operad 的 dendroidal nerve 有唯一 inner horn fillers；Moerdijk--Weiss 定位为 MW-4。

**外部输入定理 M.7.** Dendroidal sets 上存在 Cisinski-Moerdijk operadic model structure，其 fibrant objects 为 inner Kan dendroidal sets，cofibrations 为 normal monomorphisms。

**说明 M.8.** 在该模型结构中，operadic weak equivalence 不是逐树集合双射，也不是逐树同伦等价。它由模型结构或等价的 local object 条件定义。

## M.4 Lurie 模型内部

**定义 M.9.** Lurie-style infinity-operad 是 inner fibration
$$
\mathcal O^\otimes\to N(\mathbf{Fin}_*)
$$
满足 inert coCartesian lifts、Segal product condition 和 mapping space compatibility。

**外部输入定理 M.10.** Ordinary colored operad 的 category of operators nerve 给出 Lurie-style infinity-operad。

**说明 M.11.** Lurie 模型中的 active/inert 分解方向依赖 convention。本文统一使用第十八章的 $\mathbf{Fin}_*$ 约定：$\rho^i:\langle n\rangle\to\langle1\rangle$ 为 inert。

## M.5 Dendroidal-Lurie 比较

**外部输入定理 M.12（Heuts-Hinich-Moerdijk 比较）.** 适当的 dendroidal model for infinity-operads 与 Lurie-style preoperad/operadic fibration model 之间存在 Quillen equivalence 或等价的 homotopy theory comparison。

**使用规则 M.13.** 若要把 dendroidal 结论转到 Lurie 模型，必须检查：

1. 该结论在 dendroidal operadic weak equivalence 下不变；
2. 比较定理适用于所用 fibrant/cofibrant replacement；
3. 目标命题可表述为 underlying infinity-category 或 algebra infinity-category 中的等价不变性质；
4. 所用颜色、mapping spaces 和 completeness 条件在比较中被保留。

**警告 M.14.** Dendroidal inner horn filler 条件不能直接替换为 Lurie mapping-space 条件。二者对应需要外部比较定理。

## M.6 模型范畴中 operads 到 infinity-operads

**定义 M.15.** 设 $\mathcal M$ 是 symmetric monoidal model category，$\mathcal O$ 是 $\mathcal M$ 中 operad。若 $\mathcal M_\infty$ 存在 symmetric monoidal localization，则 $\mathcal O$ 可在适当条件下给出 $\mathcal M_\infty$ 中的 infinity-operad 或 algebra object，记作 $\mathcal O^{loc}$。

**外部输入定理 M.16.** 在 admissibility、cofibrancy 和 monoidal localization 条件下，有比较
$$
\operatorname{Alg}_{\mathcal O}(\mathcal M)_\infty
\simeq
\operatorname{Alg}_{\mathcal O^{loc}}(\mathcal M_\infty).
$$

**说明 M.17.** 这不是形式恒等式。左边先取严格 $\mathcal O$-代数模型范畴再 localization；右边先把底范畴和 operad 数据送到 infinity-categorical 语境再取代数。二者比较正是外部输入定理 M.16 的内容。White WHT-1--WHT-4 和 White--Yau WY-1--WY-3 只定位模型范畴中 Bousfield localization preserves operad/colored-operad algebra structures；它们可作为 M.16 的模型范畴 preservation 前置，但不能单独推出 M.16。

## M.7 允许的推理路径

**规则 M.18.** 以下推理路径允许使用，但必须引用相应定理：

1. Strict colored operad $\mathcal P$ 到 dendroidal set：
   $$
   \mathcal P\mapsto N_d(\mathcal P).
   $$
   这是定义；fully faithfulness 是外部输入或已证明的 strict Segal 性加强。
2. Strict colored operad $\mathcal P$ 到 Lurie-style object：
   $$
   \mathcal P\mapsto N(\mathcal P^\otimes).
   $$
   Category of operators nerve 是外部输入定理。
3. Dendroidal model 到 Lurie model：
   使用 Heuts-Hinich-Moerdijk 型比较。
4. Model category of operad algebras 到 infinity-category：
   使用 Dwyer-Kan localization。
5. Strict algebra model 到 infinity-operad algebra category：
   使用 algebra localization comparison；若只需要模型范畴中 localization preserves operad/colored-operad algebras，可使用 WHT-1--WHT-4 与 WY-1--WY-3。

## M.8 禁止的捷径

**警告 M.19.** 以下推理在本书中禁止：

1. 把 $N_d(\mathcal P)$ 与 $N(\mathcal P^\otimes)$ 当作相同对象。
2. 把 operad 的 aritywise weak equivalence 直接当作代数范畴 Quillen equivalence。
3. 把 $E_\infty$-algebra 与 strict commutative dg algebra 在一般底环上无条件等同。
4. 把 dendroidal inner Kan filler 的存在当作 Lurie-style coCartesian fibration 条件的定义。
5. 把 homotopy category equivalence 当作 infinity-category equivalence，除非已知 mapping spaces 相容。

## M.9 依赖图

本书第三部分的安全依赖图为：

$$
\begin{CD}
\operatorname{Op}_{strict} @>{N_d}>> \mathbf{dSet}\\
@V{(-)^\otimes}VV @VV{\text{HHM comparison}}V\\
\operatorname{Cat}_{/\mathbf{Fin}_*} @>{N}>> \operatorname{PreOp}_{Lurie}\\
@. @VV{\operatorname{Alg}_{(-)}(\mathcal C)}V\\
@. \mathbf{Cat}_\infty
\end{CD}
$$

模型范畴中的严格代数进入该图的路径为：
$$
\operatorname{Alg}_{\mathcal O}(\mathcal M)
\xrightarrow{L}
\operatorname{Alg}_{\mathcal O}(\mathcal M)_\infty
\xrightarrow[\text{external}]{\simeq}
\operatorname{Alg}_{\mathcal O^{loc}}(\mathcal M_\infty).
$$

每一个未标为定义的箭头都必须由外部输入定理支持。

## M.10 本附录小结

Infinity-operad theory 的主要风险不是缺少定义，而是跨模型混用定义。本书允许使用多个模型，但每次移动必须说明路径。Dendroidal sets 适合树形 Segal/horn 语言；Lurie-style infinity-operads 适合 algebra objects 和 symmetric monoidal infinity-categories；模型范畴语言适合构造和计算。三者之间的桥梁是比较定理，不是记号替换。
