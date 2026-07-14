# 附录 BB：Rezk 类型、Complete Segal 对象与合成无穷范畴

本附录把高阶范畴论的内部语言补成可审查接口。第十三、十四章处理的是 Hom 为集合的单值范畴；本附录处理 Rezk type、complete Segal object 和合成 $\infty$-范畴。它们需要 simplicial/directed type theory、半单纯形对象或额外模型输入，不能直接并入普通 identity type 规则。

## BB.1 Simplicial object 的数据

**定义 BB.1（simplex category 口径）.** 记 $\Delta$ 为有限非空有序集和保序映射构成的范畴。一个 simplicial object in types 是反变函子
$$
X:\Delta^{op}\to\mathcal U.
$$
记 $X_n\coloneqq X([n])$。面映射和退化映射由 $\Delta$ 中的余面与余退化诱导。

**边界 BB.2（普通 HoTT 中的相干问题）.** 在普通 HoTT 中直接定义完整 semisimplicial 或 simplicial type 会遇到无限相干数据。解决方式有三类：

1.  把 simplicial object 作为外部元语言对象；
2.  使用 two-level、simplicial 或 directed type theory；
3.  只对有限截断层级写出所需数据。

本附录采用第二种和第三种混合口径：定义写成数学接口，证明状态按外部输入或证明核标注。

## BB.2 Segal 条件

**定义 BB.3（spine map）.** 对 $n\ge2$，$n$-simplex 的 spine 是由连续边
$$
0\to1\to\cdots\to n
$$
组成的子对象。对 simplicial object $X$，spine 限制诱导 Segal map
$$
\mathsf{seg}_n:
X_n\to
X_1\times_{X_0}X_1\times_{X_0}\cdots\times_{X_0}X_1.
$$

**定义 BB.4（Segal object）.** $X$ 是 Segal object，若对每个 $n\ge2$，$\mathsf{seg}_n$ 是等价。

**解释.** $X_0$ 是对象类型，$X_1(x,y)$ 是从 $x$ 到 $y$ 的态射类型。Segal 条件说复合态射及其高阶相干由 simplex filler 唯一确定到可收缩选择。

**命题 BB.5（复合的存在与唯一性，证明核）.** 若 $X$ 是 Segal object，则对 $f:X_1(x,y)$ 与 $g:X_1(y,z)$，存在复合
$$
g\circ f:X_1(x,z)
$$
并且所有复合选择构成可收缩类型。

**证明.** 由 $n=2$ 的 Segal 等价，给定 $(f,g)$ 的 fiber 可收缩。其中心给出二单纯形，外边即复合；任意两个选择相等由 fiber 可收缩性给出。$\square$

**命题 BB.6（结合相干，证明核 / 外部输入）。** 若 $X$ 是 Segal object，则三重复合的两种括号方式由 $n=3$ 的 Segal 等价给出同伦相等，并且高阶括号相干由所有 $n$ 的 Segal 条件统一控制。

**证明状态.** $n=3$ 情形可按 BB.5 的 fiber 可收缩性逐项展开。所有高阶相干作为 complete Segal / simplicial type theory 的标准外部输入使用。

## BB.3 等价边与 Rezk 完备性

**定义 BB.7（可逆边）.** 在 Segal object $X$ 中，边 $f:X_1(x,y)$ 是 equivalence edge，若存在边 $g:X_1(y,x)$ 和二单纯形数据证明
$$
g\circ f\simeq\mathsf{id}_x,\qquad f\circ g\simeq\mathsf{id}_y.
$$
记该类型为 $\mathsf{isEqEdge}(f)$，并记
$$
\mathsf{Eq}_X(x,y)\coloneqq
\sum_{f:X_1(x,y)}\mathsf{isEqEdge}(f).
$$

**命题 BB.8（等价边性质是命题，证明说明）。** 若 $X$ 是 Segal object 且 Hom 空间满足相应 Rezk 层级条件，则 $\mathsf{isEqEdge}(f)$ 是命题。

**证明说明.** 逆边与左右逆二单纯形的选择由 Segal fiber 和等价边的左右单位相干控制；两组逆数据相等化为 Hom fiber 中的路径。完整证明需要 Rezk type 的标准相干引理。

**定义 BB.9（completeness / Rezk condition）。** Segal object $X$ 是 Rezk object，若退化边映射
$$
\mathsf{idtoeq}_X:(x=y)\to\mathsf{Eq}_X(x,y)
$$
是等价。等价地，$X_0$ 到“等价边对象”的退化映射满足 complete Segal completeness 条件。

**定理 BB.10（对象路径等于等价，书内证明核到接口）。** 若 $X$ 是 Rezk object，则对任意 $x,y:X_0$，
$$
(x=y)\simeq\mathsf{Eq}_X(x,y).
$$

**证明.** 这正是定义 BB.9 的分量形式。正向把对象路径 transport 为恒等边；反向由 completeness 的逆映射给出。两个方向互逆由 $\mathsf{idtoeq}_X$ 是等价的 fiber 可收缩定义给出。$\square$

## BB.4 Rezk type 与单值一范畴的关系

**定义 BB.11（1-truncated Rezk object）。** Rezk object $X$ 称为 1-truncated，若每个 Hom 类型 $X_1(x,y)$ 是集合，且所有高阶 simplex 数据由一范畴复合和集合性唯一决定。

**命题 BB.12（单值范畴给出 1-truncated Rezk object，证明架构）。** 每个单值范畴 $\mathcal C$ 给出一个 1-truncated Rezk object $N(\mathcal C)$。

**证明架构.** 取 $N(\mathcal C)_0=\mathcal C_0$，$N(\mathcal C)_1(x,y)=\mathcal C(x,y)$，$n$-simplex 为可复合 $n$ 串态射。Segal map 按定义为等价；Rezk completeness 化为第十三章的
$$
(x=y)\simeq(x\cong y).
$$
该等价由 $\mathcal C$ 单值性给出。$\square$

**命题 BB.13（1-truncated Rezk object 给出单值范畴，证明架构）。** 若 $X$ 是 1-truncated Rezk object，则其对象、Hom、恒等、复合和 Segal 相干给出单值范畴。

**证明架构.** 由 $X_0$ 取对象，由 $X_1(x,y)$ 取 Hom。BB.5 给出复合，退化边给出恒等，BB.6 给出结合律和单位律；Hom 集合性使这些律为命题。单值性由 BB.10 和一范畴同构等价于 equivalence edge 的引理推出。

## BB.5 函子、自然变换与高阶 Yoneda

**定义 BB.14（Rezk functor）。** Rezk object 间的函子是 simplicial map
$$
F:X\to Y
$$
即与所有面、退化和 simplex 结构相容的映射族。

**定义 BB.15（mapping Rezk object，外部输入）。** 在合适的 simplicial type theory 中，两个 Rezk object $X,Y$ 之间存在 mapping Rezk object
$$
\mathsf{Fun}(X,Y),
$$
其对象为 Rezk functor，边为自然变换，高边为高阶自然相干。

**定理 BB.16（高阶 Yoneda，外部输入 / 证明架构）。** 对 Rezk object $X$，存在 presheaf Rezk object $\mathsf{PSh}(X)$ 和 Yoneda 嵌入
$$
y:X\to\mathsf{PSh}(X)
$$
并且 $y$ fully faithful。

**证明架构.** 令 $y(x)$ 为映射对象 $\mathsf{Map}_X(-,x)$。对任意 presheaf $P$，evaluation at identity 给出
$$
\mathsf{Nat}(y(x),P)\to P(x),
$$
反向由沿态射作用给出。证明与附录 Q 的一范畴 Yoneda 相同，但自然性和相干提升为所有 simplex 维度的条件。

## BB.6 与 directed/simplicial type theory 的接口

**事实 BB.17（synthetic $\infty$-categories）。** Riehl-Shulman 的 synthetic $\infty$-category type theory 和后续 simplicial type theory 工作把 Rezk/Segal 结构转化为对象语言中的 directed hom、extension type、horn filler 和 completeness 原则。

**使用边界 BB.18.** 本书中出现“$\infty$-category”时必须区分：

1.  第十三章的普通单值范畴；
2.  本附录的 Rezk/complete Segal object；
3.  附录 AN、AS、AX 的 directed/simplicial type theory；
4.  元语言中的 quasicategory、complete Segal space 或 model category。

这些对象之间存在比较定理，但比较定理不是定义相等。

## BB.7 对象语言边界

Segal 条件、Rezk completeness 与高阶 Yoneda 分别控制复合、对象路径和可表性，但本附录没有固定一套能够内部形成全部 semisimplicial 数据的基础 HoTT 语法。Mapping Rezk object 与高阶 Yoneda 只能在附录 AN、AS、AX 所列的扩展语言或精确外部来源中使用；它们不是第十三章一范畴结果的自动高维推广。
