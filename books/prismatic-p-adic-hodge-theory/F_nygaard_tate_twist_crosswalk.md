# 附录 F：Nygaard、Tate twist 与符号交叉表

## 本附录目标

Nygaard filtration 和 Tate twist 是本书最容易出现 indexing 错误的部分。本附录固定当前草稿 convention，并列出最终版本必须核对的源文献差异。

## F.1 本书 twist convention

**约定 F.1.** 对 prism $(A,I)$，若 $I/I^2$ 为 invertible $A/I$-module，定义
$$
M\{i\}=M\otimes_{A/I}(I/I^2)^{\otimes i}.
$$
负 twist 使用 dual invertible module。

**源码核查 F.2.** Bhatt-Scholze, arXiv:1905.08229 v4, `thm:A` 明确采用同一 convention：
$$
M\{i\}=M\otimes_{A/I}(I/I^2)^{\otimes i}.
$$
因此本书 Hodge-Tate comparison 的 twist convention 与该核心源一致。

**警告 F.3.** 若其他文献把 $\{1\}$ 定义为 $(I/I^2)^\vee$，则所有 Hodge-Tate graded formula 中的正负号都要反转。跨文献引用时应引用本附录，而不是在章节中临时改写。

## F.2 Nygaard convention

**工作公式 F.4.** 在 oriented prism $(A,d)$ 的 naive module 模型中，
$$
N^{\ge i}_{\mathrm{naive}}M=\{x\in M\mid \varphi(x)\in d^iM\}.
$$
Derived prismatic cohomology 中应替换为 filtered derived category 中的 Nygaard filtration。

**源码核查 F.5.** Bhatt--Scholze, Theorem 1.16（正文 Theorem 15.3，
源码 label `thmCagain`）在 affine smooth 情形把 Nygaard filtration 放在
completed Frobenius twist 后：
$$
\mathrm{Fil}^i_N R\Gamma_{\Prism}(X/A)^{(1)}
=
R\Gamma(X_{\mathrm{qsyn}},\mathrm{Fil}^i_N\Prism_{-/A}^{(1)}).
$$
其 graded piece 为
$$
\operatorname{gr}^i_N R\Gamma_{\Prism}(X/A)^{(1)}
\cong
\tau^{\le i}\overline{\Prism}_{R/A}\{i\}.
$$
Frobenius 分解为
$$
R\Gamma_{\Prism}(X/A)^{(1)}
\xrightarrow{\widetilde{\varphi}}
L\eta_I R\Gamma_{\Prism}(X/A)
\to
R\Gamma_{\Prism}(X/A),
$$
且 $\widetilde{\varphi}$ 是同构。

**交叉检查 F.6.** 使用 Nygaard filtration 时逐项检查：

1. filtration 是递增还是递减；
2. $N^{\ge i}$ 还是 $N^{\le i}$；
3. divided Frobenius 的目标是 $C$、$C\{i\}$ 还是 $C\{-i\}$；
4. 是否已 modulo $p^n$；
5. 是否有 truncation $\tau^{\le r}$；
6. 是否在 absolute prismatic site、relative prismatic site 或 quasisyntomic site 上。

## F.3 Syntomic convention

**公式 F.7（BMS2 quasisyntomic model）.** 对 quasisyntomic ring $A$，
BMS2, Theorem 1.12 (5) 给出
$$
\mathbf Z_p(i)(A)
=
\operatorname{fib}\left(
\varphi-\operatorname{can}:
\mathcal N^{\ge i}\widehat\Delta_A\{i\}
\longrightarrow
\widehat\Delta_A\{i\}
\right)
$$
in the completed filtered derived category。两张 map 的 target 相同；
$\varphi-1$ 是固定 canonical map 后的简写。

**警告 F.8.** 公式 F.7 不能通过把 $\widehat\Delta_A$ 替换成任意 relative
prismatic complex 来使用。Bhatt--Scholze 的 relative Nygaard theorem 位于
$C^{(1)}$，BMS2 的公式位于 quasisyntomic Nygaard-complete object；两者的
comparison 是外部输入。Modulo $p^r$ 与 nearby-cycles truncation 也必须
另行标记。

**源码入口 F.9.** BMS2, arXiv:1802.03261 v2, `eq:TateTwist` 给出模 $p$ 的入口：
$$
\mathbf Z/p\mathbf Z(i)(A)
=
\operatorname{hofib}\left(
\varphi_i-1:
\mathcal N^{\ge i}\widehat{\Prism}_A\{i\}/p
\to
\widehat{\Prism}_A\{i\}/p
\right).
$$
同一来源的 Theorem 1.15（mixed-characteristic proof 为 Theorem 10.1）把
$\mathbf Z_p(n)$ 在 characteristic $p$ smooth 情形与
$W\Omega^n_{\log}[-n]$ 关联；在 mixed characteristic smooth formal
$\mathcal O_C$ 情形，它对每个 $r$ 给出
$\mathbf Z/p^r(n)\simeq\tau^{\le n}R\psi_*\mathbf Z/p^r(n)$，连续
$\mathbf Z_p$ 版本由 compatible tower 的 derived inverse limit 得到。因此
第七、十一章均引用 locator `BMS2-SYN`。

## F.4 Hodge-Tate graded convention

**工作公式 F.10.** 本书当前写作
$$
\operatorname{gr}^{\mathrm{conj}}_i\overline\Delta_{X/A}
\simeq
R\Gamma(X,\wedge^i\mathbb L_{X/(A/I)})[-i]\{-i\}.
$$

**核查状态 F.11.** 对 Bhatt-Scholze v4 的 Hodge-Tate convention，本书的 $\{-i\}$ 写法已通过源码 label `thm:A` 核对。若某章使用其他来源的 conjugate filtration 或 Hodge-Tate graded formula，仍需显式说明该来源的 twist convention。

## F.5 最终核对队列

| Queue | 内容 | 优先级 |
| --- | --- | --- |
| NQ-1 | Bhatt-Scholze Hodge-Tate comparison 的 twist 正负号 | P0 |
| NQ-2 | Bhatt-Scholze Nygaard filtration notation | P0 |
| NQ-3 | BMS syntomic complex 的 fibre convention | P0 |
| NQ-4 | $p$-adic Tate twist $\mathbf Z_p(i)$ 与 $\{i\}$ 的对应 | P0 |
| NQ-5 | Carmeli-Feng spectral syntomic convention | P2/frontier |

## F.6 队列状态

| Queue | 当前状态 | locator |
| --- | --- | --- |
| NQ-1 | 已达 L3 | Bhatt--Scholze Theorems 4.11, 6.3 |
| NQ-2 | 已达 L3 | Bhatt--Scholze Theorem 1.16 / Theorem 15.3 |
| NQ-3 | 已达 L3 | BMS2 Theorem 1.12 (5), Theorem 1.15, Theorem 10.1 |
| NQ-4 | 部分核查 | BMS1 `cor:identtatetwist` 与 BMS2 `prop:breuilkisintwist` 可作为后续入口 |
| NQ-5 | 保持研究边界 | 不进入基础定理链 |

## 本附录小结

本附录已经完成 Bhatt--Scholze 核心 Hodge--Tate/Nygaard convention 与
BMS2 syntomic/products/nearby-cycles formulas 的 numbered-statement 核查，
并已把对象、twist、shift 与 truncation 吸收到第七、十一章。尚未闭合的是
不同 authors 的 Tate-twist normalization crosswalk，而不是上述主公式。
