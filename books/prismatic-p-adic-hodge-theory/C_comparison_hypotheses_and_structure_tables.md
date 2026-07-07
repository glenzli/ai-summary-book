# 附录 C：比较定理假设表和结构保真表

## 本附录目标

本附录把全书涉及的 comparison theorem 按输入对象、假设、输出对象和保真结构列成表。它不新增数学定理；用途是防止把不同 comparison theorem 混写。

## C.1 基础 comparison 表

| 编号 | 输入 | 假设 | 输出 | 保真结构 | 状态 |
| --- | --- | --- | --- | --- | --- |
| HT | $R\Gamma_\Delta(X/A)\otimes_A^L A/I$ | $(A,I)$ bounded, $X$ smooth | conjugate filtration graded by $\Omega^i[-i]\{-i\}$ | pullback, multiplicative structure in suitable form | 外部输入 |
| dR | $\phi_A^\ast R\Gamma_\Delta(X/A)\otimes_A^L A/I$ | bounded prism, $X$ smooth | de Rham complex | Hodge filtration, cdga structure | 外部输入 |
| crys | $R\Gamma_\Delta(X/A)$ over $(A,(p))$ | crystalline prism, $X$ smooth over $A/p$ | crystalline cohomology/Frobenius descent | Frobenius | 外部输入 |
| etale | $R\Gamma_\Delta(X/A)[1/I]$ | perfect prism, $X$ smooth proper | $R\Gamma_{\mathrm{et}}(X_\eta,\mathbf Z/p^n)$ via $\varphi$-fixed | cup product, functoriality | 外部输入 |
| syntomic | $N^{\ge i}R\Gamma_\Delta$ and $\varphi_i$ | Nygaard filtration, range hypotheses | $\mathbf Z_p(i)$-type complexes | cup product, Tate twist | 外部输入 |

## C.2 Classical comparison 表

| Classical theorem | Geometry | Representation | Period ring | Extra structure |
| --- | --- | --- | --- | --- |
| Hodge-Tate | proper smooth $X/K$ | $H^n_{\mathrm{et}}(X_{\overline K},\mathbf Q_p)$ | $B_{\mathrm{HT}}$ | grading |
| de Rham | proper smooth $X/K$ | same | $B_{\mathrm{dR}}$ | filtration |
| crystalline | good reduction | same | $B_{\mathrm{cris}}$ | $\varphi$, filtration |
| semistable | semistable reduction | same | $B_{\mathrm{st}}$ | $\varphi$, $N$, filtration |

## C.3 积分 comparison 表

| Theory | Integral object | Base ring | Rational/specialization output | Main source |
| --- | --- | --- | --- | --- |
| BMS $A_{\inf}$ | $R\Gamma_{A_{\inf}}(\mathfrak X)$ | $A_{\inf}$ | de Rham, crystalline, etale | BMS1 |
| Breuil-Kisin | $R\Gamma_{\mathfrak S}(\mathfrak X)$ | $\mathfrak S=W(k)[[u]]$ | crystalline lattices/BK modules | BMS2 |
| Prismatic | $R\Gamma_\Delta(X/A)$ | prism $(A,I)$ | multiple specialization | Bhatt-Scholze |
| Prismatic $F$-crystal | finite locally free crystal + Frobenius | prismatic site | crystalline $G_K$-lattices | Bhatt-Scholze 2021/2023 |

## C.4 结构保真检查表

使用 comparison theorem 前必须回答：

1. 比较对象是否在同一个 derived category 中，若不是，桥接 functor 是什么。
2. 比较是否保持 filtration。
3. 比较是否保持 Frobenius 或 Frobenius-semilinear structure。
4. 比较是否保持 cup product。
5. 比较是否 functorial in $X$。
6. 比较是否 compatible with base change。
7. 是否需要 truncation range。
8. 是否需要 invert $p$、invert $I$、modulo $p^n$ 或 derived completion。

## C.5 不可替代表

| 不可替代对象 | 不能替代为 | 原因 |
| --- | --- | --- |
| $\overline\Delta_{X/A}$ | de Rham complex | 少了 Frobenius pullback |
| derived fixed points | ordinary fixed submodule | 丢失 cokernel/obstruction |
| BK module | filtered $\varphi$-module | 底环和 integral lattice 不同 |
| BKF module | Galois representation | 需要 comparison/classification functor |
| $F$-crystal | $F$-gauge | 后者含额外 filtration/gauge data |
| source-level locator | theorem locator | 缺 section/theorem/page 精度 |

## 本附录小结

Comparison theorem 的正式使用必须同时记录假设、目标、结构保真和 locator。表格中的每一行都应在最终版本中升级为精确引用。

