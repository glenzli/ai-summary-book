# 附录 F：三角范畴和稳定 infinity-范畴翻译表

## 本附录目标

Motivic homotopy 的早期文献大量使用模型范畴和三角范畴，而本书主体使用 stable presentable infinity-categories。本附录给出翻译规则，防止把同伦范畴层面的同构误当作 infinity-范畴中的相干等价。

## 依赖前置知识

需要 stable infinity-categories、homotopy categories、triangulated categories、exact functors、cofiber sequences、adjunctions 和 mapping spectra。

## F.1 从 stable infinity-category 到 triangulated category

**外部输入定理 F.1（稳定 infinity-范畴的三角影子）.** 若 `\mathcal C` 是 stable infinity-category，则其 homotopy category `h\mathcal C` 有自然 triangulated category 结构。Distinguished triangles 是 `\mathcal C` 中 cofiber sequences 的像；Verdier 公理由 stable infinity-category 的 bicartesian square 演算给出。本书把该基础定理作为稳定范畴论输入，不在此重建全部 octahedral axiom。

**外部来源与边界.** Lurie, *Higher Algebra*, Theorem 1.1.2.14；稳定性满足
该定理假设见 Remark 1.1.2.15。该结果从 enhancement 产生 triangulation，
不声称任意抽象 triangulated category 都存在或唯一存在 stable
infinity-categorical enhancement。

**定义 F.2.** `\mathcal C` 中的 cofiber sequence

$$
X\to Y\to Z
$$

在 `h\mathcal C` 中给出 distinguished triangle

$$
X\to Y\to Z\to \Sigma X.
$$

**命题 F.3.** Stable infinity-category 中的 fiber sequence 与 cofiber sequence 等价。

**证明.** Stable infinity-category 定义要求 pointed、有限极限和有限余极限存在，且 pullback square 等价于 pushout square。由此任意 map 的 fiber 和 cofiber 通过同一 bicartesian square 相关，故 fiber sequence 和 cofiber sequence 是同一结构的两个方向。`\square`

## F.2 Exact functors

**定义 F.4.** Stable infinity-categories 之间的 functor 称为 exact，若保持有限极限；等价地保持有限余极限。

**命题 F.5.** Exact functor 诱导 triangulated functor。

**证明.** 设 `F:\mathcal C\to\mathcal D` exact。它保持零对象和有限
pushouts。对 `X\in\mathcal C`，suspension 由 pushout

$$
\begin{array}{c}
X\longrightarrow0\\
\downarrow\qquad\downarrow\\
0\longrightarrow\Sigma X
\end{array}
$$

定义；应用 `F` 得到 canonical equivalence
`F(\Sigma X)\simeq\Sigma F(X)`。若 `X\to Y\to Z` 是 cofiber sequence，
其定义 pushout 被 `F` 送为 pushout，故
`F(X)\to F(Y)\to F(Z)` 仍为 cofiber sequence。原 connecting map
`Z\to\Sigma X` 由同一个 pushout 的泛性质构造；自然性说明其像在上述
suspension equivalence 下正是
`F(Z)\to\Sigma F(X)`。所以 `hF` 保持 shift、connecting maps 和全部
distinguished triangles，是 triangulated functor。`\square`

**依赖源与边界.** Exactness 的 finite-limit 与 finite-colimit 定义等价见
Lurie, *Higher Algebra*, Proposition 1.1.4.1。这里从已给 stable
infinity-functor 得到 triangulated functor；不声称任意 triangulated functor
都存在 exact enhancement。

## F.3 Mapping spectra and Hom groups

**定义 F.6.** Stable infinity-category 中对象 `X,Y` 的 mapping spectrum 记为

$$
\operatorname{Map}_{\mathcal C}^{sp}(X,Y).
$$

Homotopy category 中的 Hom 群为

$$
\operatorname{Hom}_{h\mathcal C}(X,Y)=\pi_0\operatorname{Map}_{\mathcal C}(X,Y).
$$

**命题 F.7.** 对整数 `n`，

$$
\pi_n\operatorname{Map}_{\mathcal C}^{sp}(X,Y)
\simeq
\operatorname{Hom}_{h\mathcal C}(\Sigma^nX,Y).
$$

**证明.** Mapping spectrum 由等价
`\Omega^\infty\operatorname{Map}^{sp}_{\mathcal C}(X,Y)
\simeq\operatorname{Map}_{\mathcal C}(X,Y)` 刻画，并满足

$$
\operatorname{Map}^{sp}_{\mathcal C}(\Sigma X,Y)
\simeq\Omega\operatorname{Map}^{sp}_{\mathcal C}(X,Y).
$$

迭代该等价（负 `n` 时用 suspension 的逆）给出

$$
\pi_n\operatorname{Map}^{sp}(X,Y)
\simeq\pi_0\operatorname{Map}^{sp}(\Sigma^nX,Y).
$$

右侧是 mapping space 的连通分支集合，并由 stable enrichment 带有阿贝尔群
结构，正是 `\operatorname{Hom}_{h\mathcal C}(\Sigma^nX,Y)`。`\square`

## F.4 Adjunctions and enhancements

**命题 F.8.** Infinity-categorical adjunction 诱导 homotopy category adjunction。

**证明.** 若 `L\dashv R`，则有 mapping space 等价

$$
\operatorname{Map}(LX,Y)\simeq\operatorname{Map}(X,RY).
$$

取 `\pi_0` 得 Hom 集自然同构，故在 homotopy categories 上为伴随。`\square`

**注 F.9.** 反向不成立：homotopy category 上的伴随不一定提升为 infinity-categorical adjunction，因为缺少 mapping spaces 的相干信息。

## F.5 紧致性的翻译

**定义 F.10.** 设 `\mathcal C` 为 stable presentable infinity-category。
对象 `K` 在 infinity-categorical 意义下紧致，是指
`\operatorname{Map}_{\mathcal C}(K,-)` 保持 filtered colimits。对象 `K`
在 triangulated 意义下紧致，是指对每族 `(X_i)_{i\in I}`，canonical map

$$
\bigoplus_i\operatorname{Hom}_{h\mathcal C}(K,X_i)
\longrightarrow
\operatorname{Hom}_{h\mathcal C}\left(K,\bigoplus_iX_i\right)
$$

为同构。

**命题 F.11.** `K` 在 `\mathcal C` 中紧致，当且仅当它在
`h\mathcal C` 中紧致。

**证明.** 先设 `K` infinity-categorically compact。任意 coproduct
`\bigoplus_{i\in I}X_i` 是有限子集偏序 `\operatorname{Fin}(I)` 上有限
coproduct 的 filtered colimit。故

$$
\operatorname{Map}\left(K,\bigoplus_iX_i\right)
\simeq
\operatorname*{colim}_{J\in\operatorname{Fin}(I)}
\operatorname{Map}\left(K,\bigoplus_{j\in J}X_j\right).
$$

取 `\pi_0`；spaces 的 filtered colimits 与 `\pi_0` 交换，而 stable
category 中有限 coproduct 是 biproduct，便得到定义 F.10 的直和同构。

反之，设 `K` 在 `h\mathcal C` 中紧致。每个 `\Sigma^nK` 也紧致。对任意
对象族，canonical map of spectra

$$
\bigoplus_i\operatorname{Map}^{sp}(K,X_i)
\longrightarrow
\operatorname{Map}^{sp}\left(K,\bigoplus_iX_i\right)
$$

在第 `n` 个同伦群上正是 `\Sigma^nK` 的 triangulated compactness map，故
对所有 `n\in\mathbb Z` 为同构；由 spectra 的 Whitehead theorem，该 map
为等价。因此 exact functor `\operatorname{Map}^{sp}(K,-)` 保持 coproducts。

最后说明它保持全部小余极限。Lurie, *Higher Algebra*, Proposition
1.4.4.1(2) 说明 stable infinity-categories 间的 exact functor 保持全部小
余极限，当且仅当它保持小 coproducts；上段已验证后者。该判据也可直接由
simplicial replacement 展开：各层只用 coproduct，几何实现由有限
cofibers、coproducts 和 sequential colimit 组成，而 sequential colimit 是
`\operatorname{cofib}(1-\mathrm{shift}:\bigoplus X_n\to\bigoplus X_n)`。
Exactness 与 coproduct preservation 因而保持这些步骤。特别地，它保持
filtered colimits。`\Omega^\infty:\operatorname{Sp}\to\mathcal S` 保持
filtered colimits，所以 `\operatorname{Map}_{\mathcal C}(K,-)` 也保持
filtered colimits。故 `K` infinity-categorically compact。`\square`

## F.6 Six operations translation

**定义 F.12.** 文献中的 triangulated six functors

$$
f^*,f_*,f_!,f^!,\otimes,\underline{\operatorname{Hom}}
$$

若来自 stable infinity-categorical six functor formalism，则它们是取 homotopy category 后的影子。

**命题 F.13（已有高阶变换的可逆性检测）.** 设
`F,G:\mathcal C\to\mathcal D` 为 infinity-functors，
`\eta:F\to G` 是已经给定的 infinity-natural transformation。则 `\eta`
为自然等价，当且仅当 `h\eta:hF\to hG` 的每个分量在 `h\mathcal D` 中为
同构。

**证明.** Infinity-category 中一条边是等价，当且仅当它在 homotopy
category 中的像是同构。因此 `h\eta_X` 对每个 `X` 为同构，当且仅当
`\eta_X` 对每个 `X` 为等价。Functor infinity-category 中的等价逐对象
检测，所以这又等价于 `\eta` 为自然等价。`\square`

**推论 F.14.** 若 infinity-categorical base-change、projection-formula 或
purity transformation 已经构造，且其三角影子的每个分量已证明为同构，则
原 transformation 为等价。特别地，infinity-category 中的等价总在
homotopy category 中给出自然同构。

**证明.** 对指定 transformation 应用命题 F.13。`\square`

**注 F.15（不能反推的内容）.** “某两个 triangulated functors 存在一个
自然同构”与“某个已指定 infinity-natural transformation 的三角影子可逆”
不是同一陈述。前者没有指定 mapping spaces 上的 transformation，也没有
base-change pasting、projection formula 与复合的 higher coherence；它甚至
未必提升。因而早期文献只给 triangulated isomorphism 时，必须另有
enhancement theorem 或直接的 infinity-categorical construction，才能在本书
中当作 coherent six-functor equivalence 使用。

第六章定理 6.12 是这一边界的具体实例：Deglise--Jin--Khan Remark 2.5.5
只把复合数据组织成 `\operatorname{Ho}(\mathbf{SH})` 值逆变伪函子之间的
自然变换，Proposition 2.5.4(ii) 则给出 transverse base-change 交换方块；
该文明确不完成所期待的 infinity-category 层增强。因此这些三角影子不能被
升级为未另行证明的 higher coherence。

| stable infinity-category 陈述 | triangulated shadow | 不能从 shadow 单独恢复 |
| --- | --- | --- |
| cofiber sequence | distinguished triangle | cofiber square 的选择空间与相干 |
| mapping spectrum | 全部 `\operatorname{Hom}(\Sigma^nX,Y)` | 谱的乘法/高阶组合数据 |
| infinity-adjunction | homotopy-category adjunction | mapping-space 等价的相干 |
| natural equivalence with pasting | natural isomorphism | 多方块 Beck--Chevalley coherence |
| presentable localization | Verdier localization triangle | accessible localization 与 adjoints |

## F.7 本附录小结

Stable infinity-category 提供 mapping spectra、余极限和全部相干层级；
triangulated category 只保留其一阶同伦影子。已有 infinity-natural
transformation 的可逆性可以在三角影子中检测，但 bare triangulated
isomorphism 不能凭空产生 transformation 或 higher coherence。紧致性在已有
stable presentable enhancement 时可由命题 F.11 精确翻译；presentability、
constructibility 的几何生成口径和六操作相干仍不能只从抽象 triangulated
category 恢复。

## 练习

**练习 F.1.** 从 cofiber sequence 写出 distinguished triangle。

**练习 F.2.** 证明 exact functor 保持 distinguished triangles。

**练习 F.3.** 推导公式 `\pi_n Map(X,Y)\simeq Hom(\Sigma^nX,Y)`。

**练习 F.4.** 解释为什么 homotopy category 同构不足以保证 higher coherence。

**练习 F.5.** 找出一个六操作定理，说明其 triangulated 版本和 infinity-categorical 版本的差别。

**练习 F.6.** 在命题 F.11 中证明 sequential colimit 的 telescope cofiber
公式，并说明为什么它使 exact coproduct-preserving functor 保持 sequential
colimits。
