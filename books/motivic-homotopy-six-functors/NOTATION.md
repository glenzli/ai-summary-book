# 符号与约定

## 宇宙、小性与基范畴

- 固定 Grothendieck universes `\mathbb U\in\mathbb V`。小范畴默认指 `\mathbb U`-小；presentable infinity-范畴默认在 `\mathbb V` 中讨论。
- `\mathcal S` 表示 spaces 的 infinity-范畴。
- `\operatorname{Pr}^L` 表示 presentable infinity-范畴和保持小余极限函子组成的 infinity-范畴。
- `\operatorname{Pr}^L_{\operatorname{st}}` 表示稳定 presentable infinity-范畴和保持小余极限的正合函子。
- 默认基概形 `S` 为有限维 Noetherian 概形；扩展到 qcqs 概形、代数栈、log 概形或解析栈时必须另行声明。
- `\operatorname{Sm}_S` 表示 `S` 上光滑有限型概形的一个 `\mathbb U`-小骨架。
- `\operatorname{Sch}^{ft}_S` 表示 `S` 上有限型概形范畴，除非另有说明，态射在 `S` 上。

## Presheaves、sheaves 与 motivic spaces

- `\mathcal P(\operatorname{Sm}_S)=\operatorname{Fun}(\operatorname{Sm}_S^{op},\mathcal S)`。
- `\operatorname{Shv}_{Nis}(\operatorname{Sm}_S)` 表示 Nisnevich topology 下的 space-valued sheaves；是否要求 hyperdescent 在相关章节中声明。
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

## 六操作

对态射 `f:X\to Y`，六操作写作

$$
f^*,\qquad f_*,\qquad f_!,\qquad f^!,\qquad -\otimes_X-,\qquad \underline{\operatorname{Hom}}_X(-,-).
$$

- `f^*` 默认为强对称幺半 pullback。
- `f_*` 是 `f^*` 的右伴随，存在性来自 presentability 和外部构造。
- `f_!` 与 `f^!` 是非常推前和非常拉回；默认只对六操作形式主义允许的态射类声明。
- proper 态射的 `f_!\simeq f_*` 是 proper compatibility，不是定义。
- 对开嵌入 `j:U\hookrightarrow X`，`j_!` 表示 extension by zero。
- 对闭嵌入 `i:Z\hookrightarrow X`，`i_*` 通常 fully faithful；具体断言依赖 localization theorem。

## 证明标签

- “内部命题”表示可由本书此前定义和一般范畴论直接证明。
- “外部输入定理”表示使用已发表专著、论文或已核查预印本。
- “研究边界”表示截至 2026-07-08 已核查但暂不作为正文基础定理使用的前沿结果。
