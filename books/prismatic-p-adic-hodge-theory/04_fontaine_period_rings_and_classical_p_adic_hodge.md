# 第四章：Fontaine 周期环与 classical $p$-adic Hodge theory

棱柱对象最终要解释哪些经典比较现象，必须先有一份不含糊的目标清单。对 $p$-进 Galois 表示 $V$，不同 Fontaine 周期环通过不变量
$D_B(V)=(V\otimes B)^{G_K}$ 检测 Hodge--Tate、de Rham、crystalline 或 semistable 性；“可容许”还要求维数没有在取不变量时丢失。本章固定局部域、Galois 作用、滤过与 Frobenius module 的全部类型，说明各周期环携带哪些结构，以及几何比较定理输出什么。周期环的完整构造作为外部输入；这里的任务是建立第五至七章必须回收的经典接口，而不是用名称代替 admissibility 条件。

## 4.1 $p$-adic Galois representations

**约定 4.1.** 本章令 $K$ 为特征 $0$ 的完全离散赋值域，剩余域 $k$ 特征为 $p$ 且完美。固定代数闭包 $\overline K$，令
$$
G_K=\operatorname{Gal}(\overline K/K).
$$

**定义 4.2.** 一个 $p$-adic representation of $G_K$ 是有限维 $\mathbf Q_p$-向量空间 $V$，配有连续群同态
$$
G_K\to\operatorname{Aut}_{\mathbf Q_p}(V).
$$
$G_K$-stable lattice 是有限生成 $\mathbf Z_p$-子模 $T\subset V$，满足 $T\otimes_{\mathbf Z_p}\mathbf Q_p=V$ 且 $G_KT=T$。

**例 4.3.** Tate twist $\mathbf Q_p(1)$ 是 $\mathbf Q_p$ 上的一维 representation，其 $G_K$-作用由 cyclotomic character $\chi_{\mathrm{cyc}}:G_K\to\mathbf Z_p^\times$ 给出。定义
$$
\mathbf Q_p(n)=\mathbf Q_p(1)^{\otimes n},\qquad
\mathbf Q_p(-n)=\operatorname{Hom}_{\mathbf Q_p}(\mathbf Q_p(n),\mathbf Q_p).
$$

## 4.2 Period rings and admissibility

**定义 4.4.** 一个 Fontaine period ring 在本章中指带连续 $G_K$-作用的拓扑 $\mathbf Q_p$-代数 $B$，通常还带有额外结构，例如 filtration、Frobenius $\varphi$ 或 monodromy $N$。令
$$
E=B^{G_K}.
$$
对 $p$-adic representation $V$，定义
$$
D_B(V)=(B\otimes_{\mathbf Q_p}V)^{G_K}.
$$

**定义 4.5.** 存在自然 $B$-线性映射
$$
\alpha_{B,V}:B\otimes_E D_B(V)\longrightarrow B\otimes_{\mathbf Q_p}V.
$$
若 $\alpha_{B,V}$ 为同构，且 $\dim_E D_B(V)=\dim_{\mathbf Q_p}V$，则称 $V$ 为 $B$-admissible。

**警告 4.6.** 定义 4.5 在实际使用时需要 $B$ 满足 Fontaine 的 regularity 条件，才能得到良好的维数不等式和 tensor stability。本章不把这些环论条件重证；它们属于 Fontaine 理论的外部输入。

## 4.3 四类基本周期环

**外部输入定义 4.7.** Fontaine 构造了如下周期环。

1. $B_{\mathrm{HT}}$：Hodge-Tate period ring，graded，基本形态为 $\bigoplus_{i\in\mathbf Z}\mathbf C_p(i)$。
2. $B_{\mathrm{dR}}$：de Rham period ring，complete discretely filtered field，带递减 filtration。
3. $B_{\mathrm{cris}}$：crystalline period ring，带 Frobenius $\varphi$ 和 filtration after scalar extension。
4. $B_{\mathrm{st}}$：semistable period ring，带 Frobenius $\varphi$、monodromy $N$ 和 filtration。

**定义 4.8.** 对 $p$-adic representation $V$，定义：

- $V$ 是 Hodge-Tate，如果它是 $B_{\mathrm{HT}}$-admissible；
- $V$ 是 de Rham，如果它是 $B_{\mathrm{dR}}$-admissible；
- $V$ 是 crystalline，如果它是 $B_{\mathrm{cris}}$-admissible；
- $V$ 是 semistable，如果它是 $B_{\mathrm{st}}$-admissible。

**外部输入定理 4.9.** 存在 full subcategories 的包含关系
$$
\operatorname{Rep}_{\mathrm{cris}}(G_K)
\subset
\operatorname{Rep}_{\mathrm{st}}(G_K)
\subset
\operatorname{Rep}_{\mathrm{dR}}(G_K)
\subset
\operatorname{Rep}_{\mathrm{HT}}(G_K)
\subset
\operatorname{Rep}_{\mathbf Q_p}(G_K).
$$
在一般情形中这些包含不应未经假设写成等号。

**说明 4.10.** 若 $K$ 的剩余域有限，de Rham representation 潜在 semistable 是 $p$-adic monodromy theorem 的内容，不是定义层事实。

## 4.4 几何比较定理接口

**外部输入定理 4.11（Hodge-Tate comparison）.** 令 $X$ 为 proper smooth variety over $K$。则 $H^n_{\mathrm{et}}(X_{\overline K},\mathbf Q_p)$ 是 Hodge-Tate representation，且其 Hodge-Tate graded pieces 与 algebraic de Rham cohomology 的 Hodge filtration associated graded 比较。

**外部输入定理 4.12（de Rham comparison）.** 在定理 4.11 的假设下，$H^n_{\mathrm{et}}(X_{\overline K},\mathbf Q_p)$ 是 de Rham representation，并存在 filtered $B_{\mathrm{dR}}$-linear comparison isomorphism
$$
B_{\mathrm{dR}}\otimes_K H^n_{\mathrm{dR}}(X/K)
\simeq
B_{\mathrm{dR}}\otimes_{\mathbf Q_p}H^n_{\mathrm{et}}(X_{\overline K},\mathbf Q_p).
$$

**外部输入定理 4.13（crystalline comparison）.** 若 $X$ proper smooth over $K$ 且有 good reduction，则 $H^n_{\mathrm{et}}(X_{\overline K},\mathbf Q_p)$ 是 crystalline representation，并与 special fibre 的 crystalline cohomology 通过 $B_{\mathrm{cris}}$ 比较。

**外部输入定理 4.14（semistable comparison）.** 若 $X$ proper smooth over $K$ 且有 semistable reduction，则 $H^n_{\mathrm{et}}(X_{\overline K},\mathbf Q_p)$ 是 semistable representation，并与 log-crystalline cohomology 通过 $B_{\mathrm{st}}$ 比较。

## 4.5 与 prismatic theory 的关系

**说明 4.15.** Classical theory 从 representation $V$ 出发，通过 period rings 构造线性代数对象 $D_B(V)$。Prismatic theory 的方向更几何：先对 integral model 或 formal scheme 构造 $R\Gamma_\Delta$，再由其 specialization 得到 de Rham、crystalline、etale 和 syntomic 信息。

**命题 4.16（形式兼容性要求）.** 若某个 prismatic comparison theorem 声称回收 de Rham comparison，则它必须至少给出以下数据：

1. 一个与 $R\Gamma_{\mathrm{dR}}(X/K)$ 比较的 filtered object；
2. 一个与 $R\Gamma_{\mathrm{et}}(X_{\overline K},\mathbf Q_p)$ 比较的 Frobenius fixed 或 period-inverted object；
3. 一个说明 filtration、Frobenius 和 Galois action 如何对应的 functorial statement。

**证明.** Classical de Rham comparison 定理 4.12 的结论不是裸向量空间同构，而是 filtered $B_{\mathrm{dR}}$-linear comparison。若 prismatic construction 只给出未滤过复形同构，则无法恢复 Hodge filtration；若只给出 de Rham specialization 而无 etale comparison，则无法恢复 Galois representation；若无 functorial statement，则不能比较 cup product、pullback 和 Galois action。因此三项均为回收 classical comparison 所必需。证毕。

## 4.6 Filtered $\varphi$-modules 的接口

**定义 4.17.** 一个 filtered $\varphi$-module 的基本模型是有限维 $K_0$-向量空间 $D$，配有 $\sigma$-semilinear Frobenius
$$
\varphi_D:D\to D
$$
使 linearization 为同构，并且 $D_K=D\otimes_{K_0}K$ 配有递减 filtration。

**说明 4.18.** Crystalline representation $V$ 的 $D_{\mathrm{cris}}(V)$ 是 filtered $\varphi$-module。其 filtration 通常来自 $B_{\mathrm{dR}}$ comparison，而 Frobenius 来自 $B_{\mathrm{cris}}$。

**警告 4.19.** Filtered $\varphi$-module 是 rational object。它不能记录一个 $\mathbf Z_p$-lattice 的所有积分信息。第十二章的 Breuil-Kisin 和 BKF modules 正是为保留积分结构而出现。

**命题 4.20.** 若两个 $G_K$-stable lattices $T_1,T_2\subset V$ 给出同一个 rational representation $V$，则 classical $D_{\mathrm{cris}}(V)$ 不能区分 $T_1$ 与 $T_2$。

**证明.** $D_{\mathrm{cris}}$ 的输入是 $V=T_i\otimes_{\mathbf Z_p}\mathbf Q_p$。若 $T_1$ 与 $T_2$ rationalization 相同，则输入相同，输出相同。区分 lattices 需要积分理论。证毕。

## 4.7 Classical comparison 的输入输出表

**说明 4.21.** 本书后续使用 classical theory 时，只使用下表中的结构接口。

| 比较 | 几何假设 | period ring | 线性代数输出 | 本书中的作用 |
| --- | --- | --- | --- | --- |
| Hodge-Tate | proper smooth | $B_{\mathrm{HT}}$ | graded Hodge-Tate decomposition | 对照第九章 Hodge-Tate specialization |
| de Rham | proper smooth | $B_{\mathrm{dR}}$ | filtered $K$-vector space | 对照 de Rham specialization and Hodge filtration |
| crystalline | good reduction | $B_{\mathrm{cris}}$ | filtered $\varphi$-module | 对照 crystalline prism and special fibre |
| semistable | semistable reduction | $B_{\mathrm{st}}$ | filtered $(\varphi,N)$-module | 作为 log/semistable boundary |

**说明 4.22.** 这张表也是防错表。若一个 prismatic statement 只给出 cohomology groups 而没有说明 filtration、Frobenius、monodromy 或 Galois action，它至多回收 comparison 的一部分。若一个 statement 声称是 integral comparison，却只在 $\mathbf Q_p$ 或 $B_{\mathrm{dR}}$ 层面成立，它不能替代 BMS 或 Breuil-Kisin 结构。

## 4.8 经典比较的目标范畴

本章把 classical $p$-adic Hodge theory 固定为 period rings 和 admissible representations 的理论。所有几何比较定理均为外部输入。后续 prismatic 章节不能把这些结论当作定义，而应说明 prismatic cohomology 如何通过 specialization 和 Frobenius fixed constructions 与它们相接。

## 练习

**练习 4.1.** 对 $V=\mathbf Q_p(n)$，写出 $D_{B_{\mathrm{HT}}}(V)$ 的预期 grading，并说明符号 convention 如何影响 $n$ 的正负。

**练习 4.2.** 解释为什么 crystalline representation 必然 de Rham 是定理而不是定义。

**练习 4.3.** 在定理 4.12 的公式中标出 filtration 位于哪一侧，以及 $G_K$ 作用位于哪一侧。
