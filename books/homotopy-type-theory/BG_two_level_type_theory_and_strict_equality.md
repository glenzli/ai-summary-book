# 附录 BG：Two-Level Type Theory、Strict Equality 与半单纯形

普通 HoTT 的 identity type 是同伦路径，不适合表达某些元理论对象的严格相等。Two-level type theory, 2LTT, 在一个系统中同时保留 fibrant/HoTT 层和 strict/external 层。本附录给出 2LTT 的使用接口，并说明它如何支持 semisimplicial types、Reedy fibrant diagrams 和元理论构造。

## BG.1 两层类型

**输入 BG.1（pretype 与 fibrant type）。** 2LTT 有一层外部或 strict 类型，记为 pretype；其中一部分类型是 fibrant types，构成内部 HoTT 层。内部层支持路径、单值性和 HIT；外部层支持 strict equality。

**定义 BG.2（strict equality）。** 对外部层项 $a,b:A$，有 strict equality
$$
a\equiv_s b.
$$
它满足 UIP：
$$
\prod_{p,q:a\equiv_s b}(p=q).
$$

**规则 BG.3（路径与 strict equality 的边界）。** 若 $A$ fibrant，则仍有 HoTT identity type
$$
a=_A b.
$$
Strict equality $a\equiv_s b$ 与 path equality $a=_A b$ 不可混用，除非系统给出明确的 comparison map 或 reflection principle。

## BG.2 外部自然数与内部自然数

**输入 BG.4（exo-natural numbers）。** 外部层有自然数对象 $\mathbb N_s$，用于索引元理论构造。内部层有 HoTT 自然数 $\mathbb N$。

**公理 BG.5（cofibrant exo-nat / comparison，口径）。** 某些 2LTT 发展假设 $\mathbb N_s$ 足够好，使得可对外部自然数递归构造内部对象族；更强口径可要求 $\mathbb N_s$ 与内部 $\mathbb N$ 有比较等价。

**用途.** Semisimplicial type 到任意外部维度 $n:\mathbb N_s$ 的截断可在 2LTT 中递归定义，而在普通 HoTT 中只能对每个固定元语言自然数逐个写出。

## BG.3 半单纯形类型

**定义 BG.6（truncated semisimplicial type，接口）。** 对外部自然数 $n:\mathbb N_s$，$n$-truncated semisimplicial type 包含：

1.  顶点类型 $X_0$；
2.  边类型 $X_1(x_0,x_1)$；
3.  二单纯形、三单纯形，直到维度 $n$；
4.  面映射和所有 strict simplicial identities。

**命题 BG.7（strict identities 的作用）。** Simplicial identities 需要写成 strict equality，才能避免在每一维引入新的高阶路径相干。

**证明说明.** 若用 HoTT path equality 表达 face maps 的等式，则这些等式本身有路径，下一维又需要相干，导致无限塔。2LTT 的 strict equality 有 UIP，因此相干在 strict 层截断。

## BG.4 Reedy fibrant diagrams

**定义 BG.8（Reedy fibrant diagram，接口）。** 在 2LTT 中，可把 Reedy category 的对象、degree 和 matching object 定义在 strict 层，并要求内部图
$$
X:R^{op}\to\mathcal U_{\mathsf{fib}}
$$
满足 matching map 是 fibration 或相应 fibrancy 条件。

**用途.** Reedy fibrancy 使 semisimplicial types 和 complete Segal objects 的逐维构造可以在类型论内表达。

**事实 BG.9（2LTT 应用）。** Annenkov-Capriotti-Kraus-Sattler 发展了 2LTT 工具，包括 Reedy fibrant diagrams，并把它用于 semisimplicial types 和 $(\infty,1)$-category 结构。

## BG.5 Strictification 与保守性

**定义 BG.10（strictification 任务）。** 给定外部层严格数据，strictification 问题是构造内部 fibrant 对象，并证明其同伦性质不依赖 strict presentation。

**边界 BG.11（保守性不是对象语言定理）。** 2LTT 对 HoTT 的保守性、模型存在性和 strict equality 的语义是元理论结果。它不能在普通 HoTT 中作为函数调用使用。

**原则 BG.12（使用纪律）。** 本书若引用 2LTT 结果，必须说明：

1.  哪些对象在 strict 层；
2.  哪些对象在 fibrant/HoTT 层；
3.  strict equality 是否有 reflection；
4.  是否假设 exo-nat cofibrancy；
5.  结果是否已有完整证明或模型来源。

## BG.6 与 Rezk/Segal 和 QIIT 的关系

**关系 BG.13（Rezk/Segal）。** 附录 BB 的 Rezk/Segal object 若要在普通 HoTT 中以 semisimplicial 数据逐维定义，会遇到无限相干。2LTT 提供一种表达这些数据的严格外部层。

**关系 BG.14（QIIT）。** 附录 BC 的语法商 QIIT 常需要区分对象语言的 definitional equality 与元语言的相等。2LTT 可作为内部化元理论的框架，但不替代每个 QIIT 的初始性证明。

**关系 BG.15（directed/simplicial type theory）。** 2LTT 与附录 AS/AX 的 simplicial type theory 是不同路线：前者加入 strict 外部层；后者改变对象语言，使 directed hom 和 horn filler 成为内部结构。

## BG.7 两层语言的比较边界

Strict equality、外部自然数和 Reedy 图只有相对于一套具体 2LTT 语法与模型才有确定含义。本附录没有构造无限 semisimplicial/Rezk 对象，也没有给出它与附录 BB 的 synthetic $\infty$-category 语言之间的翻译；缺少该翻译时，两套接口不能互相替代。
