# 附录 AX：Directed / Simplicial Type Theory 的语义接口

附录 AS 给出 directed/simplicial 规则核。本附录补充语义接口，说明这些规则如何与模型论背景相连。

## AX.1 Simplicial 模型语义

**输入 AX.1（simplicial type theory 模型）.** Simplicial type theory 可在适当的 simplicial/covariant model structure 或 marked simplicial set 语义中解释。类型解释为高阶范畴对象，directed hom 解释为有向映射空间。

**解释 AX.2.** 普通 HoTT 的类型解释为 $\infty$-groupoids；simplicial type theory 的类型可解释为 $(\infty,1)$-category-like 对象。因此 directed hom 不应有自动逆。

**定理 AX.3（Segal 条件的语义）.** 若类型 $A$ 满足 Segal 条件，则其语义对象具有可组合态射和同伦结合的组合结构。

**证明状态.** 由 spine/horn filler 语义给出。附录 AS.7-AS.9 是对象语言口径；模型语义证明属于 Riehl-Shulman 与后续 simplicial type theory 元理论。

## AX.2 对象语言判断

**输入 AX.4（synthetic $\infty$-category 语法口径）.** 相关对象语言的核心对象包括：

1.  cubes/simplices 或 shape contexts；
2.  type families over shapes；
3.  extension types 或 horn filling judgments；
4.  directed equivalence、isomorphism、fibration 等结构。

**规则 AX.5（引用 directed/simplicial 结果）.** 若本书引用 directed/simplicial 结果，必须记录：

1.  所用对象语言和基础规则；
2.  shape primitives；
3.  extension 或 horn filling 口径；
4.  与附录 AS 规则的翻译；
5.  该结果是对象语言证明、模型论输入还是研究边界。

## AX.3 Cocartesian fibration

**定义 AX.6（cocartesian lift）.** 对 directed family
$$
p:E\to B
$$
和 arrow $u:\mathsf{hom}_B(b,b')$，在 $e:E_b$ 上的 cocartesian lift 是 arrow
$$
\bar u:\mathsf{hom}_E(e,e')
$$
覆盖 $u$，并满足对任意 $v:e\to z$ 覆盖 $u\cdot w$ 的 arrow，存在唯一 factorization through $\bar u$。

**定义 AX.7（cocartesian fibration）.** $p:E\to B$ 是 cocartesian fibration，若每个 $u$ 和 $e$ 都有 cocartesian lift，并且 cocartesian arrows 对 identity 和 composition 封闭。

**定理 AX.8（closure under composition，证明核）。** 两个可复合 cocartesian arrows 的复合仍 cocartesian。

**证明核.** 给定
$$
e\xrightarrow{\bar u}e'\xrightarrow{\bar v}e'',
$$
要证明 $\bar v\circ\bar u$ 的泛性质。任意覆盖复合底箭头的 arrow $w:e\to z$ 先由 $\bar u$ 的泛性质唯一分解为 $e'\to z$，再由 $\bar v$ 的泛性质唯一分解为 $e''\to z$。唯一性由两次唯一性复合得到。$\square$

## AX.4 Straightening-unstraightening 接口

**输入 AX.9（straightening-unstraightening）.** 在 simplicial type theory 中，cocartesian fibrations over $B$ 与 functors
$$
B\to\mathcal S
$$
之间存在等价。

**使用边界.** 这是高阶范畴语义中的核心定理。附录 AN 记录 $\infty$-category of $\infty$-categories 的来源；本书不在普通 HoTT 中重建其完整证明。

## AX.5 与普通单值范畴的比较

**命题 AX.10（离散底的退化）.** 若 directed base $B$ 是离散的，即 directed hom 仅由 identity 给出，则 cocartesian fibration over $B$ 退化为普通族
$$
B\to\mathcal U.
$$

**证明.** 因每个底 arrow 都等同于 identity，cocartesian lift 只需给 identity transport；composition 封闭自动退化为族的恒等 transport 相干。$\square$

**边界 AX.11.** 普通单值范畴的 displayed category 与 directed cocartesian fibration 有形式相似性，但基础对象语言不同。前者使用 Hom 集合和 isomorphism/path univalence，后者使用 directed hom 和 horn filler。
