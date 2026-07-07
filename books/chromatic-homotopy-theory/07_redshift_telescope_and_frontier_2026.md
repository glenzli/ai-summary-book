# 第七章：Telescope、redshift 与 2026 研究边界

## 本章目标

本章记录 2023 年之后 chromatic homotopy theory 的关键状态变化：telescope conjecture 的反例、redshift 的结构化证明、cyclotomic descent、higher semiadditivity 和 2026 年 BP<n> 相关前沿。内容按“已可作外部输入”和“前沿边界”分层。

## 依赖前置知识

需要第四章的 $v_n$ self-map 和 telescope、第五章的 $L_n^f/L_n$ 区别、第六章的 $K(n)$-local category。代数 K-theory、THH、TC 和 cyclotomic spectra 仅作接口说明。

## 7.1 Telescope conjecture 的新版口径

**定义 7.1.** 设 $F$ 是 type $n$ 有限谱，$v:\Sigma^dF\to F$ 是 $v_n$ self-map。其 telescope 记作
$$
T(n)=v^{-1}F.
$$
由 periodicity theorem，$T(n)$ 的 Bousfield 类在适当意义下不依赖选择。

**历史命题 7.2 (telescope conjecture).** 经典 telescope conjecture 断言 finite/telescopic localization 与 chromatic localization 在相应高度相合，例如 $L_n^f\simeq L_n$，或单层上 $T(n)$ 与 $K(n)$ 给出相同局部化。

**前沿输入 7.3 (Burklund-Hahn-Levy-Schlank).** 2023 年预印本证明，在每个素数 $p$ 且高度至少 $2$ 的相应层次，telescopic 与 chromatic localization 不同。特别是本书不得使用“$T(n)$-local 等于 $K(n)$-local”作为默认事实。

**警告 7.4.** 旧文献中在 telescope conjecture 假设下成立的陈述，进入本书时必须标明：

1. 是历史条件命题；
2. 是否只在高度 $1$ 或特殊对象上成立；
3. 是否已被新反例排除；
4. 是否仍有 finite localization 版本可用。

## 7.2 Redshift philosophy

**定义 7.5.** Redshift philosophy 指如下现象：代数 K-theory 往往把环谱的 chromatic height 提高一层。粗略地，若 $R$ 是高度 $n$ 的结构化环谱，则 $K(R)$ 应显示高度 $n+1$ 的周期性。

**警告 7.6.** 定义 7.5 不是定理。要成为定理必须指定：

- $R$ 是 $\mathbb E_1$、$\mathbb E_3$ 还是 $\mathbb E_\infty$ ring；
- “高度 $n$”采用 fp-type、Bousfield 类、$T(n)$-local 或 $K(n)$-local 哪一种定义；
- $K(R)$ 是 connective algebraic K-theory 还是 nonconnective K-theory；
- 是否取 $p$-completion 或 chromatic localization；
- 结论是非零检测、精确高度、bounded fiber 还是 Lichtenbaum-Quillen 型近似。

**外部输入 7.7 (Hahn-Wilson).** 对每个素数 $p$ 和高度 $n$，$BP\langle n\rangle$ 可配备特定 $\mathbb E_3$-$BP$ algebra structure，并且其 algebraic K-theory 显示高度 $n+1$ 的 redshift 行为。精确表述见 theorem locator。

**外部输入 7.8 (Chromatic Nullstellensatz).** Burklund-Schlank-Yuan 证明 $T(n)$-local $\mathbb E_\infty$-rings 中 Lubin-Tate theories 的 Nullstellensatz 型刻画和 nilpotence detection，并推出 arbitrary $\mathbb E_\infty$-rings 的 algebraic K-theory redshift 结果。正文使用前必须翻译其 $T(n)$-local 和 $\mathbb E_\infty$ 假设。

## 7.3 Cyclotomic redshift 与 descent

**定义 7.9.** Cyclotomic redshift 研究 algebraic K-theory、THH、TC、cyclotomic spectra 和 chromatic localization 的相互作用，尤其关注高度 $n$ 的 Galois/cyclotomic extension 经 $K$-theory 后是否变成高度 $n+1$ 的对应结构。

**外部输入 7.10.** Ben-Moshe-Carmeli-Schlank-Yanovski 证明 $T(n+1)$-localized algebraic K-theory 对某些 $\pi$-finite $p$-group actions 满足 descent，并把 $T(n)$-local Galois/cyclotomic extensions 送到 $T(n+1)$-local 对应结构。

**说明 7.11.** 定理 7.10 与 telescope 反例相互作用：某些 $K(n+1)$-localized 版本可有 hyperdescent，而 $T(n+1)$-localized 版本出现反例。这再次说明 $K(n)$ 与 $T(n)$ 不能混用。

## 7.4 Higher semiadditivity

**定义 7.12.** 一个 infinity-范畴的 higher semiadditivity 是关于 $\pi$-finite spaces 上 indexed limits 与 colimits 一致的结构。$K(n)$-local 和 $T(n)$-local spectra 的 higher semiadditivity 是现代 chromatic theory 的核心工具之一。

**外部输入 7.13.** Hopkins-Lurie、Carmeli-Schlank-Yanovski 等工作建立了 chromatic 局部范畴的 higher semiadditivity。Ben-Moshe 2025 预印本给出通过 redshift 和高度归纳的新证明路线。

**边界 7.14.** 本书当前不把 higher semiadditivity 用作基础证明工具。它将进入后续“transchromatic character 与 semiadditive integration”章节。

## 7.5 2026 年 BP<n> syntomic/K-theory 前沿

**前沿记录 7.15.** Angelini-Knoll 2026 年预印本计算 truncated Brown-Peterson spectra 的 MU-based syntomic cohomology，并声称对所有 $\mathbb E_1$ $MU$-algebra forms of $BP\langle n\rangle$ 解决 Lichtenbaum-Quillen、telescope 和 redshift 问题，同时给出若干显式 K-theory 计算。

**使用限制 7.16.** 记录 7.15 截至 2026-07-08 作为前沿预印本处理。进入正文定理链前必须完成：

1. 版本固定；
2. theorem number 和假设定位；
3. $\mathbb E_1$ $MU$-algebra form 与本书 $BP\langle n\rangle$ convention 的比较；
4. syntomic cohomology、TC、K-theory 和 redshift 结论之间的逻辑链检查。

## 7.6 Redshift 命题的标准拆解格式

**模板 7.17.** 一个可进入正文的 redshift 定理必须写成如下格式：

1. 固定素数 $p$ 和高度 $n$；
2. 指定 ring spectrum $R$ 的结构级别，例如 $\mathbb E_1$、$\mathbb E_3$ 或 $\mathbb E_\infty$；
3. 指定 $R$ 的高度条件，例如 fp-type $n$、$T(n)$-local、$K(n)$-local 或 $v_n$-periodic；
4. 指定 $K(R)$ 的版本和 completion；
5. 指定结论是 $T(n+1)$-非平凡性、$K(n+1)$-非平凡性、精确高度、Lichtenbaum-Quillen 型近似，还是 descent；
6. 指定是否依赖 telescope conjecture 或其反例。

**例 7.18.** “$BP\langle n\rangle$ redshifts to height $n+1$”不是本书可接受的最终陈述。可接受陈述必须说明采用 Hahn-Wilson 的 $\mathbb E_3$-$BP$ algebra structure，并说明结论使用的高度定义和 algebraic K-theory 局部化。

**命题 7.19.** 若两个 redshift 定理采用不同高度定义，则不能直接合并为一个定理。

**证明.** 高度定义可能分别由 $K(n)$-local 非零性、$T(n)$-local 非零性、fp-type、或 Bousfield 类给出。由于 $K(n)$ 与 $T(n)$ 在高度至少 $2$ 不可默认相同，不同定义之间没有无条件等价。因此结论不能直接合并。证毕。

## 7.7 Telescope 反例后的教材改写规则

**规则 7.20.** 旧教材中出现的句子“由 telescope conjecture 可把 $T(n)$ 换成 $K(n)$”必须改写为三部分：

1. 历史假设：若 telescope conjecture 在该高度成立，则可比较；
2. 当前状态：高度至少 $2$ 的一般等同性不能作为事实；
3. 本文使用：本书只在明确局部化类型下陈述结论。

**例 7.21.** 旧句子“finite localization equals chromatic localization”应改写为：
“$L_n^f$ 与 $L_n$ 的比较是 telescope conjecture 的内容。本文除非额外引用特定高度结果，不使用 $L_n^f\simeq L_n$。”

## 本章小结

现代 chromatic theory 的前沿已经不能按旧的“telescope 可能成立”世界观组织。高度至少 $2$ 时 telescopic 与 chromatic localization 的差异必须进入基本口径。Redshift 已从哲学发展为多组定理，但每个定理的结构化环谱假设、局部化类型和高度定义都不同。Higher semiadditivity 和 2026 年 syntomic/K-theory 结果应收录，但必须作为前沿边界分层处理。

## 练习

**练习 7.1.** 写出一个旧式陈述“$T(n)$-local 等于 $K(n)$-local”，并把它改写为符合 2026 状态的条件命题或错误警告。

**练习 7.2.** 对 redshift 命题列出至少五个必须指定的假设。

**练习 7.3.** 解释为什么 $K(n+1)$-localized descent 与 $T(n+1)$-localized descent 可能给出不同结论。
