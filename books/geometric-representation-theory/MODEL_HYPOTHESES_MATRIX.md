# 模型假设矩阵

本文件固定 sheaves、D-modules、ind-schemes、stacks 和 convolution 的模型假设。任何正式章节调用相关结果时，应指向本矩阵或更细附录。

## 1. Sheaf theory 模型

| 模型 | 空间 | 系数 | 六函子 | Verdier duality | weights/purity | 典型用途 |
| --- | --- | --- | --- | --- | --- | --- |
| Betti constructible | complex algebraic variety 的 analytic space | characteristic zero field | 有 | 有 | 无自然 Frobenius weights | Riemann-Hilbert、Schubert IC |
| etale l-adic | finite type $k$-schemes, $\ell\ne\operatorname{char}k$ | $\mathbb Q_\ell$ 或有限扩张 | 有 | 有，含 Tate twist | 有，若 $k$ 有 Frobenius | trace functions、purity |
| mixed Hodge modules | complex algebraic varieties | $\mathbb Q$ 或 $\mathbb C$ | 有 | 有 | Hodge/weight filtration | decomposition theorem、purity |
| D-modules | smooth complex varieties 或 stacks | $\mathbb C$ | 有相应版本 | duality for D-modules | regular holonomic 经 RH 对应 | localization、Langlands |

**规则 1.1.** 一个证明不得在未给 comparison functor 的情况下从 Betti sheaves 切换到 l-adic sheaves。

**规则 1.2.** 若结论使用 purity 或 Frobenius trace，则必须采用 mixed/l-adic 模型；Betti constructible sheaves 不自带 weight formalism。

## 2. Perverse sheaf 假设

| 结论 | 最小假设 | 外部输入 |
| --- | --- | --- |
| perverse t-structure 存在 | finite-dimensional stratified complex variety 或合适 etale 模型 | BBD |
| simple perverse 分类 | 有限分层，strata 上 local systems 可控 | BBD |
| IC middle extension | locally closed smooth stratum，local system | BBD |
| decomposition theorem | complex algebraic proper map；输入默认 $\operatorname{IC}_X$；characteristic-zero coefficients | BBD 6.2.5；splitting 非 canonical |
| relative hard Lefschetz | projective map 与指定 $f$-ample class | BBD/Saito；不能只由 properness 推出 |
| semismall decomposition | proper surjective $f:Z\to Y$，$Z$ smooth pure-dimensional，stratified locally trivial，$2\dim f^{-1}(s)\le\dim Z-\dim S$ | BBD/de Cataldo--Migliorini |
| KL-IC 对应 | Schubert varieties，mixed setting 或 purity replacement | Kazhdan-Lusztig/BBD |

## 3. D-module 假设

| 结论 | 最小假设 | 注意 |
| --- | --- | --- |
| $\mathcal D_X$ 定义 | $X$ smooth over $\mathbb C$ | singular $X$ 需不同定义 |
| characteristic variety | coherent $\mathcal D_X$-module with good filtration | filtration independence 是外部输入 |
| holonomic | Bernstein inequality | 外部输入 |
| Riemann-Hilbert | regular holonomic D-modules | irregular 版本不在基础链 |
| Beilinson-Bernstein | $G/B$、twisted D-modules、regular dominant parameter | $\rho$ shift 必须锁定 |

## 4. Ind-scheme 和 affine Grassmannian 假设

| 对象 | 模型 | 正式使用前需要 |
| --- | --- | --- |
| $LG$ | functor/ind-scheme | sheafification topology |
| $L^+G$ | group functor | action representability |
| $\operatorname{Gr}_G$ | fpqc quotient represented by ind-projective ind-scheme；Betti model 取 reduction | Schubert exhaustion；nonreduced test-ring families 不进入 Betti category |
| $\operatorname{Fl}_G$ | ind-projective ind-scheme | Iwahori orbit decomposition |
| convolution Grassmannian | contracted-product ind-scheme；twisted external product 由 torsor descent | finite Schubert support 上的 proper restriction；一般 ind-map 不足 |

**规则 4.1.** $D^b_{L^+G,\mathrm{fs}}(\operatorname{Gr}_G,E)$ 只含 finite-dimensional support。定义 convolution 时必须给出包含输入和输出支撑的 finite-type stages，并用 closed base change 证明 stage independence。

**规则 4.2.** Properness 只给 $m_!\simeq m_\ast$；perverse t-exactness 另需 stratified-semismall estimate。Fusion commutativity、fiber functor tensor compatibility 和 root datum identification 是三个不同的 geometric Satake inputs。

## 5. Derived stack 和 Langlands 假设

Geometric Langlands 章节涉及 derived stacks、D-modules on stacks、IndCoh、singular support、renormalized categories 和 factorization categories。当前书稿只允许把这些作为边界对象。若后续定理化，必须新增专门模型附录。
