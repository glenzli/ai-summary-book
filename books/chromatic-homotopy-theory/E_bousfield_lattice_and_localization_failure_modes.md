# 附录 E：Bousfield lattice 与局部化失败模式

## E.1 Bousfield 类比较

**定义 E.1.** 谱 $E$ 的 Bousfield 类 $\langle E\rangle$ 是其 acyclic 类
$$
\mathcal A_E=\{X\mid E\otimes X\simeq0\}
$$
所决定的等价类。

**约定 E.2.** 本书避免使用未说明方向的 Bousfield 偏序。若必须比较，直接写 acyclic implication：
$$
\mathcal A_E\subseteq \mathcal A_F
$$
或
$$
E\otimes X\simeq0\Rightarrow F\otimes X\simeq0.
$$

**失败模式 E.3.** 若只写 $\langle E\rangle\le \langle F\rangle$ 而不说明方向，后续 local object、acyclic 和 localization functor 的推理可能全部反向。

## E.2 $K(n)$、$E(n)$、$T(n)$

**失败模式 E.4.** 把 $K(n)$-local 和 $E(n)$-local 混用。$E(n)$ 保留高度 $\le n$ 信息；$K(n)$ 聚焦高度 $n$。二者的 local categories 和 fracture 行为不同。

**失败模式 E.5.** 把 $T(n)$-local 和 $K(n)$-local 混用。2023 年后的口径要求：除非在高度 $1$ 或特殊条件下有单独定理，不得使用 telescope conjecture 型等同。

**失败模式 E.6.** 把 $L_n^f$ 和 $L_n$ 混用。$L_n^f$ 是 finite/telescopic localization；$L_n=L_{E(n)}$ 是 chromatic localization。二者比较正是 telescope 问题的核心。

## E.3 有限、compact、dualizable

**失败模式 E.7.** 在不同范畴中混用 finite、compact 和 dualizable。

- 在 $\mathbf{Sp}_{(p)}$ 中，有限谱是由球谱有限构造得到的 compact-dualizable 对象。
- 在 $K(n)$-local category 中，compact 对象的描述不同。
- 在 module category 中，perfect module、compact module 和 dualizable module 需要按 ring spectrum 检查。

**检查 E.8.** 使用 chromatic convergence、type、thick subcategory theorem 或 $v_n$ self-map 前，应确认对象是有限 $p$-局部谱。

## E.4 Completion 与 localization

**失败模式 E.9.** 把 $p$-localization 和 $p$-completion 混用。$X_{(p)}$ 使非 $p$ 素数可逆；$X^\wedge_p$ 是 derived completion。它们在非有限或非有界对象上行为不同。

**失败模式 E.10.** 把 $K(n)$-localization 当成 ordinary completion。$L_{K(n)}$ 是 Bousfield localization，不是对某个普通理想的代数完备化。

## E.5 谱序列失败模式

**失败模式 E.11.** 把 $E_\infty$ 页等同于 abutment。同伦群还需要 extension data。

**失败模式 E.12.** 忽略收敛条件。Adams-Novikov、Morava descent、Tate spectral sequence 都可能需要有界性、完备性或有限性假设。

**失败模式 E.13.** 把连续群上同调换成离散群上同调。Morava stabilizer group 是 profinite group；其 cohomology 必须带拓扑。

## E.6 前沿表述失败模式

**失败模式 E.14.** 把 redshift 写成“$K$-theory 总是升高一层”的无条件定理。正式表述必须指定 ring structure、高度定义、局部化和 completion。

**失败模式 E.15.** 把 2026 预印本结果直接写入基础定理链。预印本可以记录，但必须通过版本、定理编号、假设翻译和独立引用检查。

## 本附录小结

Chromatic theory 的错误常不是公式算错，而是模型和局部化混用。本附录列出的失败模式应作为每章修订后的审稿清单。
