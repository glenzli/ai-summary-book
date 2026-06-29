# 附录 AN：Directed / Simplicial Type Theory 与高阶范畴接口

本附录把 directed HoTT、simplicial type theory 和结构同态原则纳入本书范围。它们不是普通 HoTT 的保守小扩展；其对象语言增加 directed path 或 hom 类型，因此必须与前文的 identity/path 口径区分。

## AN.1 Directed path

**输入 AN.1（directed hom 类型）.** Simplicial type theory 为类型 $A$ 引入 directed hom 类型
$$
\mathsf{hom}_A(a,b)
$$
其元素表示从 $a$ 到 $b$ 的有向态射，而不是对称路径。通常有恒等态射
$$
\mathsf{id}_a:\mathsf{hom}_A(a,a).
$$

**原则 AN.2（不可对称化）.** 从
$$
u:\mathsf{hom}_A(a,b)
$$
不能推出 $\mathsf{hom}_A(b,a)$，也不能直接推出 identity path $a=b$。普通 HoTT 的路径代数不自动适用于 directed hom。

## AN.2 Segal 类型

**定义 AN.3（Segal 条件，内在形式）.** 类型 $A$ 称为 Segal 类型，若 directed hom 可组合，且二单纯形的内 horn filler 唯一。简化地说，对 $a,b,c:A$，组合映射
$$
\mathsf{hom}_A(a,b)\times\mathsf{hom}_A(b,c)\to\mathsf{hom}_A(a,c)
$$
存在，并满足由唯一 horn filler 给出的结合相干。

**定理 AN.4（高维内 horn 相干，外部输入）.** 在把 simplicial type theory 表示为 HoTT 加一个 interval 型的口径下，若类型有唯一 $(2,1)$-horn fillers，则可导出所有 inner $(n,k)$-horn fillers 的唯一性。

**证明状态.** de Jong-Kraus-Ljungström 2026 通过 HoTT 中的 Leibniz adjunction 证明该结果。书内只引用其定理形态。

## AN.3 Directed univalence

**定义 AN.5（离散类型宇宙）.** 在 triangulated/simplicial type theory 中，设 $\mathcal S$ 为离散类型的宇宙。

**定理 AN.6（directed univalence）.** 对离散类型 $A,B:\mathcal S$，有等价
$$
\mathsf{hom}_{\mathcal S}(A,B)\simeq(A\to B).
$$

**含义.** 宇宙中的 directed hom 不是等价，而是普通函数。这与 HoTT 的 univalence
$$
(A=B)\simeq(A\simeq B)
$$
方向不同：identity 对应 equivalence，directed hom 对应 homomorphism/function。

**证明状态.** Gratzer-Weinberger-Buchholtz 2024/2026 构造了具有非平凡 hom 的类型并证明 $\mathcal S$ directed univalent。该结果是 simplicial HoTT 的基础例子，不是普通 HoTT 内部定理。

## AN.4 结构同态原则

**定义 AN.7（结构同态原则）.** 给定结构谓词或结构族
$$
P:\mathcal S\to\mathcal U,
$$
若 $P$ 在 directed hom 下 functorial，则任意函数 $f:A\to B$ 可诱导结构同态
$$
P(A)\to P(B)
$$
或相应 displayed directed map。

**对比 AN.8（SIP 与 directed SIP）.** 普通结构等同性原则说“等价的结构相等”；directed 结构同态原则说“同态沿结构 functorial”。前者依赖 univalence，后者依赖 directed univalence。

## AN.5 $\infty$-范畴的类型论构造

**输入 AN.9（$\infty$-category of spaces）.** Simplicial type theory 中可构造空间的 $\infty$-范畴，其对象为离散/空间型数据，态射由 directed hom 给出。

**输入 AN.10（$\infty$-category of $\infty$-categories）.** 2026 年 Gratzer-Weinberger-Buchholtz 构造了 $\infty$-范畴的 $\infty$-范畴，并在该口径下恢复 straightening-unstraightening。

**使用边界.** 这些结果显著扩展 HoTT 的高阶范畴覆盖，但它们工作在 simplicial/directed type theory 中，不可直接作为本书第十三章普通单值范畴论的定理。

## AN.6 Directed fibration

**定义 AN.11（cocartesian fibration，口径）.** 在 directed type theory 中，cocartesian fibration 是具有 cocartesian lift 的 directed family
$$
E\to B
$$
并满足 closure 和 composition 性质。

**事实 AN.12（LARI adjunction 与 initial sections）.** Lossin 2026 在 Riehl-Shulman 的 synthetic simplicial type theory 中研究 cocartesian fibrations，并使用 LARI adjunction 与 initial sections 的等价证明闭包性质。

**边界.** 该方向使用的 directed/simplicial 语法与公理化 HoTT 的 identity type 不同；引用时必须记录语法口径。

## AN.7 本附录的接口

1.  第十三、十四章的单值范畴论仍在普通 HoTT 中；AN 只提供 directed/simplicial 扩展入口。
2.  附录 AS 给出 directed/simplicial 对象语言的规则核。
3.  第十七章谈 higher category theory 时必须区分 univalent categories、Segal types、simplicial type theory 和 external $\infty$-category theory。
4.  若后续扩写 directed HoTT，应新增独立主章，而不是把 directed hom 混入第二章的 identity type。
