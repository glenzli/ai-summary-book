# 术语索引

本文档记录本书核心术语、首次系统出现位置和使用边界。

| 术语 | 英文 | 位置 | 边界 |
| --- | --- | --- | --- |
| $\delta$-环 | delta ring | 定义 1.1 | 不等同于任意 Frobenius map；需满足 $\delta$ 恒等式 |
| Frobenius lift | Frobenius lift | 命题 1.3 | $p$-torsionfree 时与 $\delta$-结构等价 |
| distinguished element | distinguished element | 定义 1.5 | 依赖 $\delta$-结构 |
| Koszul complex/tower | Koszul complex/tower | 定义 A.1 | 本书置于 cohomological degrees $[-r,0]$；completion tower 使用生成元幂 |
| derived $J$-completion | derived $J$-completion | 定义 A.2 | 一般不能未经 weak-proregular/noetherian 假设改写成 $R\varprojlim M/J^nM$ |
| bounded $p^\infty$-torsion | bounded $p$-power torsion | 定义 A.7 | 不等于 torsionfree；在 principal completion 中消除 negative derived-completion terms |
| $J$-complete flatness | $J$-complete flatness | 定义 A.13 | 测试 $J$-power-torsion modules；不同于 ordinary flatness |
| complete Tor-amplitude | complete Tor-amplitude | 定义 A.15 | 只控制 modulo $J$/power-torsion；不同于命题 5.16 的 ordinary Tor-dimension |
| completed base change | completed derived base change | 定理 3.13，警告 3.14A | Tor spectral sequence 先计算 uncompleted tensor；只有该 tensor 已对目标理想 derived complete 时才直接计算 comparison target |
| prism | prism | 定义 2.3 | 裸定义不包含 boundedness；本书 site/comparison 主线另行假设 bounded |
| bounded prism | bounded prism | 定义 2.3 | 要求 $A/I$ 的 $p^\infty$-torsion 有界 |
| prism ideal rigidity | prism ideal rigidity | 引理 2.4A | prism map $(A,I)\to(B,J)$ 自动满足 $J=IB$ |
| prismatic site | prismatic site | 定义 2.13--2.16 | site 是 probe category 的 opposite；covers 为 $(p,IB)$-completely faithfully flat prism maps |
| prismatic cohomology | prismatic cohomology | 定义 2.18 | derived global sections |
| semilinear Frobenius | semilinear Frobenius | 命题 2.19 | map $\varphi_A^*C\to C$ 才是 $A$-linear；积分时不自动为同构 |
| completed Frobenius twist | completed Frobenius twist | 符号表，定理 3.10, 7.3 | $C^{(1)}=C\widehat\otimes_{A,\varphi_A}^LA$；不同于只作 typing 的 ordinary $\varphi_A^*C$ |
| $L\eta$ | décalage functor | 定义 5.4，定理 5.6 | 非 exact；modulo ideal 由 Bockstein complex 控制，只与同一 ideal 的 completion 自动交换 |
| Hodge-Tate specialization | Hodge-Tate specialization | 定义 9.1 | 不等于 de Rham specialization |
| conjugate filtration | conjugate filtration | 定理 9.4 | 位于 Hodge-Tate specialization；按递增 Postnikov 编号 $\operatorname{Fil}^{\mathrm{conj}}_i=\tau^{\le i}$ |
| de Rham specialization | de Rham specialization | 定义 9.8 | 需要 Frobenius pullback 与 derived $p$-completion；comparison 本身 unfiltered |
| Nygaard filtration | Nygaard filtration | 定义 7.1, 定理 7.3 | derived relative filtration 位于 completed twist $C^{(1)}$；graded piece 为 $\tau^{\le i}\overline\Delta\{i\}$ |
| syntomic complex | syntomic complex | 定义 7.5, 11.7 | 通常为 fibre construction |
| prismatic crystal | prismatic crystal | 定义 6.1, I.4 | sheaf condition 加 crystal rigidity |
| prismatic $F$-crystal | prismatic F-crystal | 定义 6.4, I.9 | probe-dependent ideal sheaf $\mathcal I_\Delta$ 反演后 linearized Frobenius 为同构 |
| effective prismatic $F$-crystal | effective prismatic F-crystal | 定义 6.4, I.9 | Frobenius linearization 在反演前落入 lattice；不是一般 $F$-crystal 自动具有的性质 |
| Breuil-Kisin module | Breuil-Kisin module | 定义 12.4 | 有限生成 $\mathfrak S$-module，linearized Frobenius 只要求 invert $E(u)$ 后同构；finite projective/height 需另加 |
| Breuil-Kisin-Fargues module | Breuil-Kisin-Fargues module | 定义 12.7 | finite presentation，invert $\xi$ 后 Frobenius 同构且 $M[1/p]$ finite free；Fargues pair classification 只覆盖 finite free 子类 |
| integral/after-inversion/rational layer | integral/after-inversion/rational layer | §3.7, §12.7 | $[1/I]$、$[1/p]$ 与 coefficient-field extension 是不同操作，均不能自动恢复 integral lattice |
| derived fixed points | derived Frobenius fixed points | 定义 11.4 | 不是普通 fixed subgroup |
| $B$-admissibility | B-admissibility | 定义 4.5, J.6 | 依赖 Fontaine period ring formalism |
| prismatization | prismatization | 定义 8.2 | 研究边界，不替代 prismatic site |
| $F$-gauge | F-gauge | 定义框架 8.5 | 不等于 $F$-crystal |

## 使用规则

1. 第一次使用术语时优先给中文名和英文括注。
2. 若术语存在多个文献 convention，必须说明本书 convention。
3. 前沿术语只在研究边界章节使用，除非已补外部输入 locator。
