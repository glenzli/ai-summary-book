# 序章：范围、严格性标准和资料源

## 本章目标

本章说明本书的数学对象、约定、资料源和严格性标准。读者在进入第一章前，应当知道本书中“operad”默认是什么意思，哪些内容属于基础理论，哪些内容属于外部输入，哪些内容只是研究边界。

## 依赖前置知识

需要熟悉集合、函数、有限群、商集、范畴、函子、自然变换和幺半范畴的基本语言。模型范畴、同伦代数和 infinity-范畴知识不预设，会在后续章节逐步引入。

## 0.1 本书研究的对象

**约定 0.1.** 本书固定 Grothendieck universes
$$
\mathcal U\in\mathcal V\in\mathcal W.
$$
若不特别说明，“集合”指 $\mathcal U$-小集合，“有限集”指 $\mathcal U$-小有限集。

**约定 0.2.** 本书中单独出现的“operad”默认指含 arity $0$ 的单色对称 operad。更具体地说，它是有限集上的对称序列范畴在代入乘积下的幺半对象。若某处使用非对称 operad、reduced operad、colored operad、topological operad、dg-operad 或 infinity-operad，会在该处显式说明。

这个默认约定有三个后果。

1. arity $0$ 的元素被允许，因此 operad 的代数可以带零元运算，即常数或单位。
2. 对称群作用是结构的一部分，但基础章节优先用有限集重标号来表达，从而避免左右作用约定造成的公式歧义。
3. operad 不是某个具体代数结构，而是编码一类多元运算及其代入规则的对象。

## 0.2 严格性标准

**约定 0.3.** 本书采用如下证明标准。

- 一个定义必须指定其所在范畴、结构映射和公理。
- 一个例子必须验证定义中的结构映射存在，并说明公理为什么成立。
- 一个命题、引理或定理必须给出证明；若证明依赖大型外部结果，则标注“外部输入定理”。
- 一个同伦或 infinity 语境中的结论必须说明模型结构、弱等价或采用的 infinity-模型。

**例 0.4.** “$\operatorname{Com}$ 的代数是交换幺半群”不能只作为直觉陈述。严格写法必须说明：对任一集合 $X$，一个 operad morphism
$$
\operatorname{Com}\to\operatorname{End}_X
$$
等价于在 $X$ 上给出一个二元乘法和一个零元运算，并由 operad 公理推出结合律、交换律和单位律；反向也要由交换幺半群结构构造所有有限 arity 运算。

## 0.3 为什么从有限集开始

operad 的 arity 写法通常给出集合族
$$
\mathcal O(n),\qquad n\ge 0,
$$
并配有 $\Sigma_n$ 作用和代入映射
$$
\mathcal O(n)\times \mathcal O(k_1)\times\cdots\times \mathcal O(k_n)
\longrightarrow
\mathcal O(k_1+\cdots+k_n).
$$
这里 $k_i\ge0$；$k_i=0$ 正是把 nullary operation 代入第 $i$ 个输入槽。
这种写法适合计算，但对称群作用的左右约定和块置换公式容易遮蔽概念。

本书第一章采用有限集口径：令 $\mathbf B_{\mathcal U}$ 为有限集和双射构成的群胚，一个对称序列是函子
$$
X:\mathbf B_{\mathcal U}\to \mathbf{Set}_{\mathcal U}.
$$
允许 arity $0$ 时，代入乘积由任意有限集映射及其全部纤维定义；非满射的空纤维记录 nullary operations。只有内层 arity $0$ 为空时，才可缩成非空分块公式。这个口径直接表达“先在每个纤维内做运算，再把纤维结果作为外层输入”，且重标号自然由双射函子性处理。

## 0.4 资料源和研究边界

基础 operad 理论主要依赖 May、Boardman-Vogt、Markl-Shnider-Stasheff、Loday-Vallette 和 Fresse。dendroidal set 与 infinity-operad 部分主要依赖 Moerdijk-Weiss、Cisinski-Moerdijk、Lurie 以及模型比较文献。

截至 2026-06-30，operad theory 的近期研究仍在活跃发展，尤其在以下方向：

- infinity-operad 的同调和 Koszul 对偶；
- operadic categories 与 higher nerve；
- dendroidal Rezk nerve 和 operadic localization；
- Fukaya categories 与高阶 operadic 结构；
- Boardman-Vogt tensor product、wreath product 和 operadic Grothendieck construction 的关系。

这些方向会进入本书后部的研究边界章节。除非完成独立核验，本书不会把近期预印本的新结论纳入基础定理链。

## 本章小结

本书把 operad 作为严格数学对象处理：先定义对称序列和代入乘积，再把 operad 定义为幺半对象。基础部分不依赖同伦语言；同伦和 infinity 内容会在模型结构明确之后进入。近期研究会被记录和定位，但不会削弱基础正文的可验证性。

## 练习

**练习 0.1.** 说明若 operad 定义中禁止 arity $0$，则“交换幺半群”例子会变成哪一种代数结构。

**练习 0.2.** 查阅任意两本 operad 教材，记录它们对 $\Sigma_n$ 作用采用左作用还是右作用，并说明 arity 公式因此如何变化。

**练习 0.3.** 给出一个数学陈述，其中“同构”和“弱等价”不能互换使用。要求说明所在范畴或模型范畴。
