# 符号与约定

## 宇宙、小性与基范畴

- 固定 Grothendieck universes `\mathbb U\in\mathbb V`。“小”表示 `\mathbb U`-小；presentable 表示 `\mathbb U`-accessible、具有全部 `\mathbb U`-小余极限，并在 `\operatorname{Cat}_{\mathbb V}` 中讨论。
- `\mathcal S_{\mathbb U}` 表示 `\mathbb U`-小 spaces 的 infinity-范畴；不致混淆时简写为 `\mathcal S`。
- `\operatorname{Pr}^L` 表示 presentable infinity-范畴和保持小余极限函子组成的 infinity-范畴。
- `\operatorname{Pr}^L_{\operatorname{st}}` 表示稳定 presentable infinity-范畴和保持小余极限的正合函子。
- 单基章节的 `S` 为 `\mathbb U`-小有限维 Noetherian 概形。六操作章节固定此类概形 `B`，默认变化的基为有限型 `B`-概形；exceptional morphisms 为 separated morphisms。
- `\operatorname{Sm}_S` 表示 `S` 上光滑有限型概形的一个 `\mathbb U`-小骨架。
- `\operatorname{Sch}^{ft}_S` 表示 `S` 上有限型概形范畴，除非另有说明，态射在 `S` 上。

## Presheaves、sheaves 与 motivic spaces

- `\mathcal P(\operatorname{Sm}_S)=\operatorname{Fun}(\operatorname{Sm}_S^{op},\mathcal S)`。
- `\operatorname{Shv}_{Nis}(\operatorname{Sm}_S)` 表示 Nisnevich topology 下的 space-valued Cech sheaves，默认不 hypercomplete；hypercompletion 写作 `(-)^\wedge` 并须显式声明。
- `L_{Nis}` 表示 Nisnevich sheafification。
- `L_{\mathbb A^1}` 表示关于所有投影 `X\times_S\mathbb A^1_S\to X` 的 accessible localization。
- `\mathbf H(S)=L_{\mathbb A^1}\operatorname{Shv}_{Nis}(\operatorname{Sm}_S)`。
- `\mathbf H_*(S)` 表示带基点 motivic spaces。
- 若 `X\in\operatorname{Sm}_S`，同一符号 `X` 可表示其 Yoneda presheaf、Nisnevich sheafification 或在 `\mathbf H(S)` 中的像；正文必须在第一次使用时说明所在范畴。

## Spheres、稳定化与 SH

- `S^{1,0}` 表示 simplicial circle。
- `\mathbb G_m` 以单位截面 `1:S\to\mathbb G_m` 为基点。
- `S^{1,1}=S^{1,0}\wedge\mathbb G_m`。
- `T=\mathbb A^1/(\mathbb A^1\setminus 0)`，在默认口径下与 `\mathbb P^1/\infty` 和 `S^{1,1}` 等价。
- `S^{p,q}=(S^{1,0})^{\wedge(p-q)}\wedge(\mathbb G_m)^{\wedge q}` 在 `p\ge q\ge0` 时使用；一般双指标球用稳定范畴中的可逆 suspension 坐标定义。
- `\Sigma_T^\infty:\mathbf H_*(S)\to\mathbf{SH}(S)` 表示 `T`-悬挂谱函子。
- `\mathbf{SH}(S)` 表示 `T` 或 `\mathbb P^1` 稳定 motivic homotopy infinity-范畴。
- `\mathbb 1_S` 表示 `\mathbf{SH}(S)` 的单位，即 motivic sphere spectrum。
- `\mathcal C[A^{-1}]` 表示 presentable symmetric monoidal object-inversion；只有验证 3-symmetry 与谱模型比较后才写 `\operatorname{Sp}_A(\mathcal C)`。
- `\mathbf{SH}(S)^\omega` 表示紧致对象；`\mathbf{SH}_c(S)` 表示几何生成子的 thick closure。默认 qcqs 范围内二者由外部紧生成定理识别，但不是同义定义。

## 六操作

对态射 `f:X\to Y`，六操作写作

$$
f^*,\qquad f_*,\qquad f_!,\qquad f^!,\qquad -\otimes_X-,\qquad \underline{\operatorname{Hom}}_X(-,-).
$$

- `f^*` 默认为强对称幺半 pullback。
- `f_*` 是 `f^*` 的右伴随，存在性来自 presentability 和外部构造。
- `f_!` 与 `f^!` 是非常推前和非常拉回；默认只对 separated finite-type `B`-morphisms 声明。
- smooth `f` 的 `f_\sharp` 是 `f^*` 的左伴随；它不是六个基本符号之一，也不按定义等于 `f_!`。
- proper 态射的 `f_!\simeq f_*` 是 proper compatibility，不是定义。
- 对开嵌入 `j:U\hookrightarrow X`，`j_!` 表示 extension by zero。
- 对闭嵌入 `i:Z\hookrightarrow X`，`i_*` 通常 fully faithful；具体断言依赖 localization theorem。
- `!`-projection formula 的 map 写作 `f_!(A\otimes f^*B)\to f_!A\otimes B`；ordinary projection map 写作 `f_*A\otimes B\to f_*(A\otimes f^*B)`，二者方向不同。

## Purity

- smoothable lci separated `f` 的 virtual tangent class 写作
  `\tau_f=\langle L_f\rangle`；smooth 时为 `[T_f]`，regular closed
  immersion 时为 `-[N_f]`。
- purity morphism 写作 `\mathfrak p_f(E):\Sigma^{\tau_f}f^*E\to f^!E`；
  DJK 的复合数据在本书按 `\operatorname{Ho}(\mathbf{SH})` 值伪函子层使用，
  transverse base change 按同伦三角层的交换方块使用；smooth purity 则另有
  stable infinity-categorical equivalence。
- `E` 为 `f`-pure 表示该 morphism 在 homotopy category 中为同构；absolute purity 是 regular schemes 上一族 `f`-purity 条件，不是任意 regular immersion 的无条件定理。
- 对 proper smooth `f`，exceptional counit 的类型是
  `f_*\Sigma^{T_f}f^*E\to E`；无扭曲的 `f_*f^*E\to E` 不来自
  `f^*\dashv f_*` 的 counit。

## 证明标签

- “内部命题”表示可由本书此前定义和一般范畴论直接证明。
- “外部输入定理”表示使用已发表专著、论文或已核查预印本。
- “研究边界”表示截至 2026-07-11 已核查但暂不作为正文基础定理使用的前沿结果。
