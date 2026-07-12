# 附录 C：比较定理假设表和结构保真表

## 本附录目标

本附录把全书涉及的 comparison theorem 按输入对象、假设、输出对象和保真结构列成表。它不新增数学定理；用途是防止把不同 comparison theorem 混写。

## C.1 基础 comparison 表

| 编号 | 输入 | 假设 | 输出 | 保真结构 | 状态 |
| --- | --- | --- | --- | --- | --- |
| HT | $R\Gamma_\Delta(X/A)\otimes_A^L A/I$ | $(A,I)$ bounded, $X$ smooth | Postnikov/conjugate graded pieces $R\Gamma(X,\Omega^i)[-i]\{-i\}$ | sheaf-level functoriality, multiplication；无额外 completion 符号 | BS Theorems 4.11, 6.3 |
| dR | $\phi_A^\ast R\Gamma_\Delta(X/A)\widehat\otimes_A^L A/I$ | bounded prism, $X$ smooth | derived $p$-completed de Rham complex | $E_\infty$/cdga；filtered refinement 需另引 Nygaard | BS Corollary 15.4 |
| crys | $R\Gamma_\Delta(X/A)\widehat\otimes_{A,\phi_A}^LA$ | crystalline prism $(A,(p))$, $X$ smooth | $R\Gamma_{\mathrm{crys}}(X/A)$ | integral Frobenius | BS Theorem 5.2 |
| $\varphi$-isogeny | $C^{(1)}=C\widehat\otimes_{A,\phi_A}^LA\to C$ | bounded prism, $X$ smooth | isomorphism only after $[1/I]$ | semilinear/linearized typing；completed Frobenius twist | BS Corollary 15.5 |
| etale sheaf | $(\Delta_{X/A}[1/I]/p^r)^{\varphi=1}$ | perfect prism, $X$ any $p$-adic formal scheme | $R\mu_*\mathbf Z/p^r$ on $X_{\mathrm{et}}$ | sheaf-level derived fixed fibre | BS Theorem 9.1 |
| etale global | $(\Delta_{S/A}[1/I]/p^r)^{\varphi=1}$ | perfect prism, $X=\operatorname{Spf}(S)$ affine | $R\Gamma_{\mathrm{et}}(\operatorname{Spec}(S[1/p]),\mathbf Z/p^r)$ | derived fixed fibre in $D(\mathbf Z/p^r)$；nonaffine 需另证 $R\Gamma$ 与 $I^{-1}$ 交换 | BS Theorem 9.1, affine corollary |
| base change | $R\Gamma_\Delta(X/A)\widehat\otimes_A^LB$ | map $(A,I)\to(B,IB)$ bounded, $X$ smooth | $R\Gamma_\Delta(X_B/B)$ | derived $(p,IB)$-completion | BS Corollary 4.12 |
| syntomic | $\operatorname{fib}(\varphi-\operatorname{can})$ on $\mathcal N^{\ge i}\widehat\Delta\{i\}$ | quasisyntomic model, $i\ge0$ | $\mathbf Z_p(i)$；mixed characteristic is first $\mathbf Z/p^r(i)$ with $\tau^{\le i}$, then derived-limited | cup product, Tate twist | BMS2 Theorems 1.12 (5), 1.15, 10.1 |

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
| BMS $A_{\inf}$ | perfect complex; every $H^i$ BKF | $A_{\inf}$ | derived de Rham/crystalline specializations; etale after $[1/\mu]$ | BMS1 Theorems 1.8, 14.3 |
| Breuil-Kisin | perfect complex; every $H^i$ BK module | $\mathfrak S=W(k)[[u]]$ | $A_{\inf}$、de Rham、crystalline exits along specified maps | BMS2 Theorem 1.2 |
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
9. Complex comparison 降到 $H^n$ 时，uncompleted derived tensor 是否已对
   指定理想完备，以及是否有 ordinary Tor-amplitude 或相邻次数
   torsionfreeness。

## C.5 不可替代表

| 不可替代对象 | 不能替代为 | 原因 |
| --- | --- | --- |
| $\overline\Delta_{X/A}$ | de Rham complex | 少了 Frobenius pullback |
| derived fixed points | ordinary fixed submodule | 丢失 cokernel/obstruction |
| BK module | filtered $\varphi$-module | 底环、$E(u)$-divisor 与 integral lattice 不同 |
| bare BK module | crystalline lattice | 缺 prismatic Cech descent 或 Kisin theorem 的 essential-image 假设 |
| general BKF module | finite free lattice pair | Fargues classification 只覆盖 finite free BKF modules |
| $[1/I]$ statement | $[1/p]$ statement | 前者 invert prism divisor，后者 rationalize in $p$ |
| $F$-crystal | $F$-gauge | 后者含额外 filtration/gauge data |
| source-level locator | theorem locator | 缺 section/theorem/page 精度 |

## 本附录小结

Comparison theorem 的正式使用必须同时记录假设、目标、结构保真和 locator。
核心 prismatic/BMS rows 已绑定 numbered statements；classical rows 的最终
教材源选择仍按附录 D 的真实 locator 等级处理。
