# 符号约定

本文档记录《凝聚数学讲义》的固定符号。后续章节不得随意更改。

## 集合论与范畴

- 固定不可数强极限基数 \(\kappa\) 与 Grothendieck universes
  \(\mathcal U\in\mathcal V\)，详见附录 A。测试空间底层集合基数小于
  \(\kappa\)，sheaf 值为 \(\mathcal U\)-小对象，相关范畴在 \(\mathcal V\) 中讨论。
- $\mathbf{Set}$：集合范畴。
- $\mathbf{Ab}$：阿贝尔群范畴。
- $\mathbf{Top}$：拓扑空间范畴。
- $\mathbf{CHaus}_\kappa$：\(\kappa\)-小紧 Hausdorff 空间的选定骨架；
  无下标 \(\mathbf{CHaus}\) 表示同一固定层级。
- $\mathbf{ProFin}_\kappa$：\(\kappa\)-小 profinite 集合骨架；无下标时同上。
- $\mathbf{ED}_\kappa$：\(\kappa\)-小极不连通紧 Hausdorff 空间骨架；无下标时同上。
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
- 若覆盖族写作 $\{U_i\to U\}_{i\in I}$，本书默认 $I$ 是有限集且允许
  \(I=\varnothing\)。在 condensed site 上空族只覆盖空对象，并强制
  \(F(\varnothing)=*\)（阿贝尔群值时为零群）。
- 对覆盖族的 sheaf 条件常写为等化子：
  $$
  F(U)\longrightarrow \prod_i F(U_i)
  \rightrightarrows
  \prod_{i,j}F(U_i\times_U U_j).
  $$

## 凝聚对象

- $\mathbf{CondSet}=\mathbf{CondSet}_\kappa$：本书固定层级的凝聚集合范畴。
- $\mathbf{CondAb}=\mathbf{CondAb}_\kappa$：本书固定层级的凝聚阿贝尔群范畴。
- $\mathbf{Solid}$：固体阿贝尔群范畴。
- 对拓扑空间 $T$，其关联凝聚集合暂记为
  $$
  \underline T(S)=\operatorname{Cont}(S,T),
  \qquad S\in \mathbf{CHaus}.
  $$
- 对集合 $A$，若赋予离散拓扑，则 $\underline A$ 表示对应离散拓扑空间的凝聚集合。

## Solid 与解析符号

- 对 profinite 集合 $S$，
  $$
  \mathbb Z^\square[S]=\varprojlim_i\mathbb Z[\underline{S_i}],
  \qquad S=\varprojlim_iS_i,\ S_i\text{ finite}.
  $$
- $\mathbb Z^\square$ 表示 $\underline{\mathbb Z}$ 的固化。
- $M^\square$ 表示凝聚阿贝尔群 $M$ 的固化。
- $\otimes^\square$ 表示固体张量积。
- $\otimes^{L,\square}$ 表示派生固体张量积。

## 证明用语

- “覆盖”默认指有限联合满射覆盖，除非所在站点另有说明。
- “满射”在 $\mathbf{CHaus}$ 中指底层集合上的满射连续映射。
- “商映射”指拓扑意义上的 quotient map。
