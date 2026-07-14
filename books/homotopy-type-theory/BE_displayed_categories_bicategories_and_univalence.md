# 附录 BE：Displayed Categories、Bicategories 与高阶单值性

第十三、十四章给出一范畴层的单值范畴论。本附录补入 displayed categories、displayed bicategories 和 univalent bicategories。它们是在 HoTT 中避免“对象相等”硬编码的主要工具，也是高阶单值范畴论的重要组织方式。

## BE.1 Displayed category

**定义 BE.1（displayed category over $\mathcal C$）。** 设 $\mathcal C$ 为预范畴。一个 displayed category $\mathcal D$ over $\mathcal C$ 由以下数据组成：

1.  对每个 $c:\mathcal C$，有类型 $\mathcal D(c)$；
2.  对每个 $f:c\to c'$ 和 $x:\mathcal D(c)$、$y:\mathcal D(c')$，有 displayed morphism 类型
    $$
    x\xrightarrow{f}_{\mathcal D}y;
    $$
3.  displayed identity
    $$
    \mathsf{id}_x:x\xrightarrow{\mathsf{id}_c}_{\mathcal D}x;
    $$
4.  displayed composition
    $$
    (x\xrightarrow{f}y)\to(y\xrightarrow{g}z)\to(x\xrightarrow{g\circ f}z);
    $$
5.  displayed associativity 和单位律。

**解释.** displayed category 表示“在基范畴对象和态射上附加结构”。它避免先构造总对象类型再向基范畴投影，从而减少对象相等 transport。

## BE.2 Total category

**定义 BE.2（total category）。** displayed category $\mathcal D$ 的 total category $\int_{\mathcal C}\mathcal D$ 定义为：
$$
\left(\int_{\mathcal C}\mathcal D\right)_0
\coloneqq
\sum_{c:\mathcal C_0}\mathcal D(c),
$$
且
$$
\left(\int_{\mathcal C}\mathcal D\right)((c,x),(c',y))
\coloneqq
\sum_{f:\mathcal C(c,c')}(x\xrightarrow{f}_{\mathcal D}y).
$$
恒等和复合由基范畴与 displayed category 的恒等、复合逐分量给出。

**命题 BE.3（total category Hom 集合性，书内证明核）。** 若 $\mathcal C$ 的 Hom 是集合，且每个 displayed morphism 类型是集合，则 $\int_{\mathcal C}\mathcal D$ 的 Hom 是集合。

**证明.** Hom 是 $\Sigma$ 类型
$$
\sum_{f:\mathcal C(c,c')}(x\xrightarrow{f}_{\mathcal D}y).
$$
第一分量是集合；第二分量按假设在每个 $f$ 上是集合。由 $\Sigma$ 类型保持集合性得到结论。$\square$

## BE.3 Displayed univalence

**定义 BE.4（displayed isomorphism）。** 给定基同构 $i:c\cong c'$，对象 $x:\mathcal D(c)$ 与 $y:\mathcal D(c')$ 的 displayed isomorphism 是沿 $i$ 的 displayed morphism 和沿 $i^{-1}$ 的 displayed morphism，并满足 displayed 左右逆律。

**定义 BE.5（displayed univalence）。** displayed category $\mathcal D$ 是 displayed univalent，若对任意 $x:\mathcal D(c)$、$y:\mathcal D(c')$ 和基路径或基同构 $i$，transport 给出的映射
$$
(x=\mathsf{transport}_{\mathcal D}(p,y))
\to
\mathsf{dispIso}_i(x,y)
$$
是等价。

**定理 BE.6（total univalence，证明架构）。** 若 $\mathcal C$ 单值且 $\mathcal D$ displayed univalent，则 $\int_{\mathcal C}\mathcal D$ 单值。

**证明架构.** total category 中对象路径分解为基对象路径 $p:c=c'$ 和 fiber 中路径。由 $\mathcal C$ 单值性把 $p$ 等价为基同构 $i:c\cong c'$；由 displayed univalence 把 fiber 路径等价为沿 $i$ 的 displayed isomorphism。二者合并正是 total category 中的同构。

## BE.4 Displayed functor 与 fibration

**定义 BE.7（displayed functor）。** 若 $F:\mathcal C\to\mathcal C'$，displayed functor over $F$ 把
$$
x:\mathcal D(c)
$$
送到
$$
F^\sharp(x):\mathcal D'(F c)
$$
并把 displayed morphism over $f$ 送到 displayed morphism over $Ff$，且保持恒等和复合。

**定义 BE.8（cartesian displayed morphism）。** displayed morphism $\bar f:x\xrightarrow{f}y$ 是 cartesian，若对任意 $g:d\to c'$ 和 displayed morphism over $g$ 到 $y$，存在唯一分解过 $\bar f$ 的 displayed morphism over 相应基态射。

**定义 BE.9（displayed fibration）。** displayed category 是 fibration，若每个基态射 $f:c\to c'$ 和每个 $y:\mathcal D(c')$ 都有 cartesian lift。

**用途.** 这种定义避免直接说“函子 fiber 中对象相等”，更适合 HoTT，因为对象相等在非单值范畴中不是结构不变概念。

## BE.5 Bicategory

**定义 BE.10（bicategory，数据）。** 一个 bicategory $\mathcal B$ 由以下数据组成：

1.  对象类型 $\mathcal B_0$；
2.  对 $a,b:\mathcal B_0$，Hom category $\mathcal B(a,b)$；
3.  恒等 1-cell $\mathsf{id}_a:\mathcal B(a,a)$；
4.  水平复合函子
    $$
    \mathcal B(b,c)\times\mathcal B(a,b)\to\mathcal B(a,c);
    $$
5.  associator、left unitor、right unitor 的自然同构；
6.  pentagon 和 triangle coherence。

**定义 BE.11（local univalence）。** Bicategory $\mathcal B$ 是 locally univalent，若每个 Hom category $\mathcal B(a,b)$ 是单值范畴。

**定义 BE.12（global univalence）。** Locally univalent bicategory $\mathcal B$ 是 globally univalent，若对象路径
$$
a=b
$$
等价于 adjoint equivalence of objects
$$
a\simeq_{\mathcal B} b.
$$

**定义 BE.13（univalent bicategory）。** Bicategory 同时 locally univalent 和 globally univalent。

## BE.6 Displayed bicategory

**定义 BE.14（displayed bicategory，接口）。** 给定 bicategory $\mathcal B$，displayed bicategory 在每个对象 $b$ 上给出 fiber 对象，在每个 1-cell 上给出 displayed 1-cell，在每个 2-cell 上给出 displayed 2-cell，并配备 displayed identity、composition、associator、unitor 和 coherence。

**定理 BE.15（total bicategory univalence，外部输入）。** 若基 bicategory univalent，displayed bicategory 满足 displayed local/global univalence，则 total bicategory univalent。

**来源与边界.** Ahrens--Frumin--Maggesi--Veltri--van der Weide 的 univalent bicategories 工作给出 displayed bicategory 方法和多个 univalent bicategory 实例。本书把该定理作为外部输入，不在正文逐行重建全部 bicategory coherence。

## BE.7 典型实例

**例 BE.16（结构范畴）。** 群、环、模、拓扑结构、范畴带额外结构等通常可作为 displayed category over $\mathsf{Set}$ 或 over $\mathsf{Cat}$ 构造。若结构等同性原则已证明，则 total category 单值性由 BE.6 得出。

**例 BE.17（univalent categories 的 bicategory）。** 对象为单值范畴，1-cell 为函子，2-cell 为自然变换。适当定义下它构成 univalent bicategory。

**例 BE.18（monoidal categories）。** Monoidal category 可作为 displayed bicategory 或 displayed structure over categories 处理。Wullaert-Matthes-Ahrens 的相关工作证明了 univalent monoidal categories 的 bicategory 单值性和 Rezk completion。

## BE.8 一维与二维单值性的分界

Displayed category 可以逐 fiber 组织第十三章的一维结构；bicategory 还需要 associator、unitor 与全部二维 coherence。具体实例只有在 displayed univalence 已证明后才能调用 total univalence；与附录 BB 的高阶 Rezk object 比较还需要额外 nerve 或语义定理，不能由名称相近直接得到。
