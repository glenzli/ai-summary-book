# 符号约定

本文档记录《凝聚数学讲义》的固定符号。后续章节不得随意更改。

## 集合论与范畴

- 固定一个 Grothendieck universe $\mathcal U$。若不特别说明，“集合”“范畴”“拓扑空间”均指 $\mathcal U$-小对象。
- $\mathbf{Set}$：集合范畴。
- $\mathbf{Ab}$：阿贝尔群范畴。
- $\mathbf{Top}$：拓扑空间范畴。
- $\mathbf{CHaus}$：紧 Hausdorff 空间范畴，态射为连续映射。
- $\mathbf{ProFin}$：profinite 集合范畴，即紧、Hausdorff、全不连通空间范畴。
- 对范畴 $\mathcal C$，记
  $$
  \widehat{\mathcal C}=\operatorname{Fun}(\mathcal C^{\operatorname{op}},\mathbf{Set})
  $$
  为 $\mathcal C$ 上的集合值预层范畴。
- Yoneda 嵌入记为
  $$
  y:\mathcal C\to \widehat{\mathcal C},\qquad
  U\mapsto h_U=\operatorname{Hom}_{\mathcal C}(-,U).
  $$

## 站点与 sheaf

- 若 $J$ 是 $\mathcal C$ 上的 Grothendieck 拓扑，记
  $$
  \operatorname{Sh}(\mathcal C,J)
  $$
  为 sheaf 范畴。
- 若覆盖族写作 $\{U_i\to U\}_{i\in I}$，本书默认 $I$ 是有限集，除非特别说明。
- 对覆盖族的 sheaf 条件常写为等化子：
  $$
  F(U)\longrightarrow \prod_i F(U_i)
  \rightrightarrows
  \prod_{i,j}F(U_i\times_U U_j).
  $$

## 凝聚对象

- $\mathbf{CondSet}$：凝聚集合范畴。
- $\mathbf{CondAb}$：凝聚阿贝尔群范畴。
- 对拓扑空间 $T$，其关联凝聚集合暂记为
  $$
  \underline T(S)=\operatorname{Cont}(S,T),
  \qquad S\in \mathbf{CHaus}.
  $$
- 对集合 $A$，若赋予离散拓扑，则 $\underline A$ 表示对应离散拓扑空间的凝聚集合。

## 证明用语

- “覆盖”默认指有限联合满射覆盖，除非所在站点另有说明。
- “满射”在 $\mathbf{CHaus}$ 中指底层集合上的满射连续映射。
- “商映射”指拓扑意义上的 quotient map。
