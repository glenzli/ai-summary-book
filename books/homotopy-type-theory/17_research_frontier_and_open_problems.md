# 第十七章：研究边界中的语言与定理

HoTT 的研究边界很少是一条孤立的未解公式。更常见的情形是：为了表达 directed morphism、无限相干、几何模态或可计算的 HIT，需要扩张对象语言；扩张后又要重新证明替换、可靠性、canonicity，并与已有模型比较。若省略其中任何一层，“某原则应当成立”就可能同时混合语法猜想、模型存在问题和内部数学定理。

本章用四个案例学习怎样辨认这种边界。弱单值性说明名称相近的原则可以被模型严格分离；半单纯形对象展示相干数据如何随维数增长；synthetic $\infty$-category 说明 directed hom 不能由可逆路径代替；strict Rezk completion 则展示一个模型构造怎样只推出 homotopy canonicity，而不是更强的 judgmental canonicity。这里讨论的是数学接口和量词方向，不是书稿维护任务。

## 17.1 研究问题至少有三个层次

设 $T$ 是一个已经给出语法和推导规则的类型论，$T^+$ 是加入新形成子的扩张。关于 $T^+$ 的问题至少分成三类。

**定义 17.1（构造与可靠性问题）.** 构造问题要求给出 $T^+$ 的形成、引入、消去、计算和替换规则；可靠性问题要求构造模型并证明每条规则在模型中有效。一个只有生成元列表、没有消去和替换定理的 HIT 或 QIIT 签名尚未形成完整对象理论。

**定义 17.2（计算问题）.** 计算问题固定 $T^+$ 的 judgmental equality，询问 normalization、canonicity、类型检查或 judgmental equality 可判定性。对象语言中存在路径
$n=_{\mathbb N}\overline k$
只给 homotopy canonicity；要得到
$n\equiv\overline k$
还需要关于归约的更强元定理。

**定义 17.3（比较问题）.** 比较问题要求给出两个理论或模型之间的翻译，并证明可靠、完备、保守或 Quillen/范畴等价中的某一种。两个系统都使用“path”“univalence”或“Rezk”一词，并不提供这种翻译。

第四类问题才是在固定语言内部证明一个具体数学结论，例如计算某个同伦群。区分这四类问题可以防止模型论结果被当作对象语言构造，也能防止内部定理被误称为某种 normalization。

## 17.2 弱单值性为何不能代替 universe univalence

第六章的 universe univalence 比较
$$
(A=_{\mathcal U}B)
\quad\text{与}\quad
(A\simeq B),
$$
其中右侧使用 fiber 可收缩意义下的等价。另一个可能的原则只把 universe path 与 wild category 中具有左右逆的函数比较；后者的逆律是函数路径，但不预先假设函数外延性。

**定义 17.4（categorical univalence）.** 令
$A\cong_{\mathrm w}B$
表示由函数、逆函数和两条函数路径组成的 wild-category 同构数据。若规范映射
$$
(A=_{\mathcal U}B)\longrightarrow(A\cong_{\mathrm w}B)
$$
对所有 $A,B:\mathcal U$ 都是等价，则称该 universe categorically univalent。

**外部输入定理 17.5（模型分离）.** 存在 Martin--Löf type theory 模型，其中某 universe categorically univalent，但函数外延性失败。

**来源与边界.** Cavallo--Höfer, *Univalence without function extensionality*, arXiv:2605.00812v1，Definitions 1.1--1.4、Theorem 1.6/5.6；完整模型假设见附录 AO.1。本书只采用不蕴含结论，不把 polynomial model 的 extensive coproduct 或 strict $\eta$ 假设加入基础语言。

与此相对，第六章定理 6.11 采用 HoTT Book Theorems 4.9.4--4.9.5：同一个 universe 的普通 univalence 推出相应的依赖函数外延性。两条结果合在一起给出严格区分：
$$
\text{universe univalence}
\Longrightarrow
\text{function extensionality},
$$
而 categorical univalence 没有这个蕴含。研究边界因此由原则的精确右端类型决定，不能靠“都是单值性”消除。

## 17.3 半单纯形数据与无限相干

低维半单纯形对象可以直接看见相干为何增长。先给顶点类型
$$
A_0:\mathcal U,
$$
再给有向边族
$$
A_1:A_0\to A_0\to\mathcal U.
$$
二维数据可写成依赖族
$$
A_2:
\prod_{x_0,x_1,x_2:A_0}
A_1(x_0,x_1)\to A_1(x_1,x_2)
\to A_1(x_0,x_2)\to\mathcal U,
$$
其项表示具有指定三条边的填充三角形。三维数据必须同时看到四个二维面，并表达它们共享边时的相容；继续升维会出现所有 face maps 与 simplicial identities。

对任意固定维数 $n$，这些数据可以写成有限的依赖记录。难点是把所有 $n$ 统一成一个内部对象，同时让 face identities 具有所需的严格度。若用恒等类型记录每条相容，下一维又要比较相容路径，并继续产生更高相干；若用外部 judgmental equality 强制相等，则已经离开第一章的基础语言。

**研究接口 17.6（两条常见路线）.**

1. Two-level type theory 增加 strict equality 层，用严格图表组织外部半单纯形索引，再要求内部 fibers fibrant；附录 BG 固定两层之间不得混同的桥接规则。
2. HIIT/QIIT 路线尝试让类型、面数据与相等同时归纳生成；每个具体签名都必须证明 strict positivity、替换稳定性、依赖消去和相应模型中的初始性，若还声称可计算性，则必须另证 normalization 或 canonicity。

这里不声称“半单纯形类型在所有 HoTT 中都是未解的”，也不声称任一 2LTT 或 QIIT 结果自动覆盖另一套语法。准确边界是：本书的基础理论没有给出上述无限对象；附录 BC、BG 只记录各自扩展语言的接口与外部元理论。

## 17.4 Directed hom 与 Rezk type

恒等路径总是可逆：若 $p:a=b$，第二章构造了 $p^{-1}:b=a$。因此它不能直接表示一般范畴中的有向态射。最简单的反例是偏序 $0<1$：存在从 $0$ 到 $1$ 的箭头，却没有反向箭头；若把箭头定义成 identity path，路径逆会错误地产生反向箭头。

Simplicial type theory 通过 directed interval 和 extension types 定义
$$
\mathsf{hom}_A(a,b),
$$
它与 $a=_Ab$ 是不同形成子。对可复合的
$f:\mathsf{hom}_A(a,b)$、$g:\mathsf{hom}_A(b,c)$，Segal 条件要求合成及其二维填充的候选类型可收缩；可收缩性同时给出存在与所有高阶唯一性。Rezk 条件进一步要求对象 identity 与内部 categorical isomorphism 之间的规范映射为等价。

**外部输入接口 17.7（synthetic $\infty$-categories）.** Riehl--Shulman 的三层 simplicial type theory 定义 Segal types、Rezk types 与 covariant families，并在该语言中证明 dependent Yoneda lemma；其 motivating semantics 使用 bisimplicial sets 的 Reedy 模型结构。

**来源与未重证边界.** Emily Riehl, Michael Shulman, *A type theory for synthetic $\infty$-categories*, Higher Structures 1 (2017), 147--224, DOI `10.21136/HS.2017.06`, arXiv:1705.07442v5。该来源加入 shapes、topes 和 extension types；本书不把这些判断解释为第十四章的一范畴 Rezk completion，也不从普通 identity type 推出 directed hom。

一个自然的比较问题是：给定该语言的模型，内部 Rezk type 如何对应外部 complete Segal space？回答需要从语法到 bisimplicial 语义的解释及适当的完备性或不变性结论，不能只比较两个定义的名称。

## 17.5 从 strict Rezk completion 到 homotopy canonicity

第十四章的 Rezk 完备化作用于一范畴对象；模型论中还可以对 HoTT 模型作 strict Rezk completion，使对象语言 identity terms 与 ambient cubical paths 更紧密地比较。二者有结构类比，但所在层次不同。

**外部输入定理 17.8（闭布尔项的 homotopy canonicity）.** 在 Bocquet 所固定的 HoTT 语法中，每个闭项 $b:\mathbf 2$ 都满足
$$
\mathsf{Id}_{\mathbf 2}(b,\mathsf{true})
+
\mathsf{Id}_{\mathbf 2}(b,\mathsf{false}).
$$

**来源与边界.** Rafaël Bocquet, *Strict Rezk completions of models of HoTT and homotopy canonicity*, arXiv:2311.05849v2，Definitions 5.1--5.2、Theorems 5.18、6.1；模型的 global、algebraically cofibrant 与 components fibrant 假设见附录 AO.2。结论给出到规范布尔值的 identity term，不断言
$b\equiv\mathsf{true}$ 或 $b\equiv\mathsf{false}$，也不覆盖任意扩展语法。

这个案例把三个层次分得很清楚：strict Rezk completion 是模型构造，gluing/sconing 给出元理论证明，最终结论是对象语言中的路径析取。若要升级为 judgmental canonicity，还需 normalization 或计算语义；若要覆盖新增 HIT、resizing 或 choice，又必须重新检查模型假设。

## 17.6 几何模态改变可表达的对象

Cohesive HoTT 和 synthetic differential geometry 采用另一种扩张方式：加入 shape、discrete、codiscrete 等模态，有些系统还区分 crisp 变量或加入无穷小对象。此时“连续映射”“无穷小邻近”“de Rham 形状”成为对象语言结构，而不是对普通类型的外部形容词。

例如，一个无穷小对象
$$
D\coloneqq\{\varepsilon:\mathbb R\mid\varepsilon^2=0\}
$$
只有在所选环对象允许 nilpotent 元素、子类型和相应几何公理时才具有预期内容；在普通实数域的集合模型中它只剩 $0$。因此从构造性 Cauchy 实数章节不能推出 SDG 的微分规则。附录 AT、BD 记录模态和几何模型接口，任何内部微分定理都必须列出所用模态、变量纪律与 microlinearity 假设。

## 17.7 把开放方向写成可判定的责任

上述案例给出三种可实际检验的研究陈述。第一种是构造定理：“对签名 $\Sigma$ 构造模型，并证明所有消去和替换规则可靠。”第二种是计算定理：“对固定语法证明 normalization，并推出某类闭项的 judgmental canonicity。”第三种是比较定理：“构造翻译 $T\to T^+$，并证明它对旧语言判断保守，或证明相应模型范畴之间的指定等价。”每一种都有明确输入和结论，失败时也能指出缺少的是 positivity、coherence、reification 还是收敛。

HoTT 的研究前沿由这些精确桥梁推进，而不是由越来越长的主题名录推进。弱单值性、directed hom、无限相干、几何模态和 strict Rezk completion 都展示了同一原则：先固定对象语言，再区分内部定理与外部元定理，最后才比较不同系统。这样，尚未完成的工作可以被写成数学命题，而不会回流成基础章节中的隐式规则。

## 练习

**练习 17.1.** 证明若把偏序的有向箭头解释为 identity path，则任一箭头都会产生反向箭头；指出矛盾使用了第二章的哪一构造。

**练习 17.2.** 展开 $A_2$ 的一个项所需的顶点与边数据，并说明定义三维 tetrahedron family 时哪些二维面必须共享边。

**练习 17.3.** 比较定理 17.5 与第六章定理 6.11 的右端类型，解释模型分离为何不与普通 univalence 推出函数外延性矛盾。

**练习 17.4.** 把定理 17.8 分别改写成 homotopy canonicity 与 judgmental canonicity 的形式，说明来源只证明哪一个。

**练习 17.5.** 为“某 QIIT 扩展对自然数等式保守”写出完整量词，并列出一个仅有模型存在性证明仍缺少的反向步骤。
