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

**源码核查 F.5.** Bhatt-Scholze, arXiv:1905.08229 v4, `thmCagain` 在 affine smooth 情形把 Nygaard filtration 放在 Frobenius twist 后：
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

**工作公式 F.7.** 本书当前采用
$$
R\Gamma_{\mathrm{syn}}(X,\mathbf Z_p(i))
=
\operatorname{fib}\left(
N^{\ge i}R\Gamma_\Delta(X/A)
\xrightarrow{\varphi_i-1}
R\Gamma_\Delta(X/A)\{i\}
\right)
$$
作为 convention form。

**警告 F.8.** 公式 F.7 仍是本书内部 convention form。由于 Bhatt-Scholze 的 Nygaard 定理对 Frobenius twist、$\tau^{\le i}$ 和 $L\eta_I$ 有明确结构，最终 syntomic 公式必须根据所选来源决定是否使用 $p^i-\varphi$、$\varphi_i-1$、truncate 后 fibre 或 modulo $p^n$ 的版本。

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
同一来源的 `thm:main6` 把 $\mathbf Z_p(n)$ 在 characteristic $p$ smooth 情形与 logarithmic de Rham-Witt sheaves 关联，在 mixed characteristic smooth formal $\mathcal O_C$ 情形与截断 nearby cycles 关联。因此第十一章的 syntomic tower 应以 BMS2-SYN 作为最终核查入口。

## F.4 Hodge-Tate graded convention

**工作公式 F.10.** 本书当前写作
$$
\operatorname{gr}^i_{\mathrm{conj}}\overline\Delta_{X/A}
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
| NQ-1 | 已完成核心源核查 | Bhatt-Scholze v4 `thm:A` |
| NQ-2 | 已完成核心源核查 | Bhatt-Scholze v4 `thmCagain` |
| NQ-3 | 基础公式已吸收到第七、十一章；最终变体待 L3 | BMS2 `eq:TateTwist`, `thm:main6`, `thm:nearbycycles` |
| NQ-4 | 部分核查 | BMS1 `cor:identtatetwist` 与 BMS2 `prop:breuilkisintwist` 可作为后续入口 |
| NQ-5 | 保持研究边界 | 不进入基础定理链 |

## 本附录小结

本附录已经完成 Bhatt-Scholze 核心 Hodge-Tate/Nygaard convention 与 BMS2 syntomic/Tate twist 源码入口的核查，并已把基础公式吸收到第七章和第十一章。正式出版前仍需把 mod $p^r$、truncation 和 nearby cycles 的变体全部升级为 L3。
