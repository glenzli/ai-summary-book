# 依赖图

本文件记录当前草稿的定义依赖和外部输入依赖，防止倒用高级定理。

## 1. 定义链

```text
stable infinity category
  -> spectra, finite spectra, compactness
  -> homology theory E_*X
  -> E-acyclic, E-local, L_E
  -> Bousfield class
  -> K(n), E(n), E_n
  -> L_n, L_K(n), M_n
  -> chromatic tower, fracture square
```

```text
complex orientation
  -> E^*(CP^\infty)=E^*[[x]]
  -> formal group law F_E
  -> p-series and height
  -> Quillen theorem for MU
  -> BP and p-typical coordinates
  -> E(n), K(n), BP<n>
```

```text
finite p-local spectra
  -> K(n)-homology
  -> type
  -> thick subcategories C_n
  -> v_n self-map
  -> telescope T(n)
  -> finite localization L_n^f
```

```text
Lubin-Tate deformation
  -> Morava E-theory E_n
  -> Morava stabilizer group G_n
  -> E_infinity action
  -> homotopy fixed points E_n^{hG_n}
  -> K(n)-local sphere
  -> descent spectral sequence
```

```text
K(n)-local descent
  -> continuous stabilizer cohomology
  -> chromatic splitting problem
  -> Gross-Hopkins duality
  -> Picard descent
  -> exotic Picard elements
```

```text
stable semiadditivity
  -> finite biproducts
  -> norm map for finite groups
  -> higher semiadditivity for pi-finite spaces
  -> semiadditive cardinality
  -> transchromatic character and integration
```

```text
BP_*BP Hopf algebroid
  -> comodules and Ext
  -> Adams-Novikov spectral sequence
  -> invariant ideals I_n
  -> chromatic spectral sequence
  -> Morava change of rings
  -> continuous group cohomology
```

```text
type n finite spectrum
  -> Hopkins-Smith periodicity
  -> v_n self-map
  -> telescope v^{-1}F
  -> T(n) Bousfield class
  -> finite localization L_n^f
  -> telescope comparison failure modes
```

## 2. 外部输入依赖

| 外部输入 | 被哪些章节使用 | 不得用于 |
| --- | --- | --- |
| Quillen theorem | 第二章、附录 A | 直接构造 $BP\langle n\rangle$ 的 structured quotient |
| Landweber exactness | 第二章、第三章、第八章 | 替代 $\mathbb E_\infty$ 精化 |
| DHS nilpotence | 第四章、第七章 | 任意非有限谱的检测 |
| Hopkins-Smith thick subcategory | 第四章、第五章 | 一般 presentable subcategory 分类 |
| Periodicity theorem | 第四章、第七章 | 证明 $T(n)=K(n)$ |
| Chromatic convergence | 第五章 | 一般谱的收敛 |
| Goerss-Hopkins-Miller | 第三章、第六章、第八章 | 由系数环自动推出结构 |
| Devinatz-Hopkins descent | 第六章 | 离散群上同调替代连续群上同调 |
| BHLŠ telescope counterexample | 第七章 | 否定所有特殊高度一或特殊对象结果 |
| Hahn-Wilson redshift | 第七章 | arbitrary $\mathbb E_1$ ring 的 redshift |
| Chromatic Nullstellensatz | 第七章 | 无条件 $K(n)$-local statement |
| Angelini-Knoll 2026 | 第七章、frontier audit | 进入基础证明链 |
| Higher semiadditivity | 第九章 | 普通谱范畴的 Tate vanishing |
| HKR character theory | 第九章 | 任意 $\pi$-finite 空间的 transchromatic integration |
| Gross-Hopkins duality | 第十章 | 未指定 convention 的悬挂公式 |
| Picard descent | 第十章 | Morava module 直接完成 Picard 分类 |
| Equivariant periodicity | 第十一章 | naive equivariant spectra |
| Motivic/synthetic reconstruction | 第十一章 | 任意基域或非 cellular 对象 |

## 3. 允许路径

- 从 $K(n)$ 系数环可证明 graded field 性，但不能推出 module category 分类，除非引用相应 module theorem。
- 从复定向可构造形式群律，但 universal 性必须引用 Quillen。
- 从 finite type 可进入 thick subcategory 和 periodicity，但必须引用 Hopkins-Smith。
- 从 $E_n$ 系数环可解释 Lubin-Tate 变形，但 $\mathbb E_\infty$ 结构和 $\mathbb G_n$ action 必须引用 Goerss-Hopkins-Miller。
- 从 chromatic tower 可讨论 convergence，但只有有限谱版本已作为外部输入登记。
- 从 0-semiadditivity 可推出有限直和/直积相同，但 higher semiadditivity 必须引用 chromatic 外部定理。
- 从 Picard 的 Morava module 可得到代数比较映射，但 exotic kernel 需要 descent theorem 和计算。

## 4. 禁止捷径

1. 不得把 $T(n)$-local 与 $K(n)$-local 直接等同。
2. 不得把 $L_n^f$ 与 $L_n$ 直接等同。
3. 不得把 $tmf$ 称为高度二 Morava E-theory。
4. 不得把 Adams-Novikov $E_\infty$ 页直接称为稳定同伦群。
5. 不得从 $BP_*/I$ 的环商自动推出 $\mathbb E_k$-ring quotient。
6. 不得对 profinite Morava stabilizer group 使用普通离散群上同调。
7. 不得把 2026 预印本结论用于基础证明链，除非完成 locator 和假设翻译。
8. 不得把 $E_\infty$ 谱序列页面直接当作同伦群。
9. 不得把 genuine equivariant spectra 替换成 naive group actions。
10. 不得把 motivic chromatic theory 写成普通 chromatic theory 的双分次复制。
