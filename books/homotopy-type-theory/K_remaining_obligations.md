# 附录 K：剩余证明义务登记

## 目标

本附录登记当前版本仍未完全书内展开的证明说明、外部输入和研究边界。后续每次推进时，应优先从本表中选择条目，把它们降级为书内证明、精确外部输入定理或形式化引用。

## K.1 基础层

**K.1.1 准逆相干化公式。**  
位置：附录 G.4。  
当前状态：已补为书内证明。  
已补充：定理 G.4 按本书路径复合方向写出
$$
\epsilon'_y
\coloneqq
\epsilon_{f(g(y))}^{-1}\cdot
\bigl(\mathsf{ap}_f(\eta_{g(y)})\cdot\epsilon_y\bigr)
$$
并给出三角相干证明。  
剩余义务：机器形式化时需按具体库的路径复合方向核对符号。

**K.1.2 单值性推出函数外延性。**  
位置：第六章 6.11。  
当前状态：精确外部输入 / 机器形式化入口。  
已补充：附录 T 给出 UniMath `UniMath/Foundations/UnivalenceAxiom.v` 中从 `univalenceStatement` 到 `isweqtoforallpathsStatement` 的形式化链条，并登记 Coq-HoTT `Univalence_implies_Funext` 入口。第六章 6.11 已改为引用附录 T。  
剩余义务：若要求完全书内证明，需要把附录 T.3 的四步路线展开成本书路径代数；若要求机器化，需要选择 UniMath 或 Coq-HoTT 的具体口径并固定依赖分析。

**K.1.3 同伦层级向上闭包。**  
位置：第四章 4.13。  
当前状态：书内证明核。  
已补充：附录 AB 证明可收缩类型的路径空间可收缩，并由自然数归纳推出 $\mathsf{isOfHLevel}_n(A)\to\mathsf{isOfHLevel}_{n+1}(A)$。  
剩余义务：机器化时需对齐具体库中 h-level 编号和本书编号。

## K.2 HIT 与截断

**K.2.1 一般 HIT 元理论。**  
位置：第九章、第十六章。  
当前状态：已精确登记输入规则；一般元理论仍外部。  
已补充：附录 L 列出本书使用的命题截断、$n$-截断、集合商、圆、悬挂和 pushout 的形成、构造、消去、计算规则；附录 Z 区分对象语言、元语言和实现语言，并登记 HIT 语义、cubical 计算规则与 canonicity 的边界。  
剩余义务：一般 HIT 语法、语义、normalization 和 canonicity 的证明仍作为第十六章外部元理论，不在本书对象语言中展开。

**K.2.2 一般 $n$-截断存在性。**  
位置：第八章 8.5。  
当前状态：已列为附录 L.5-L.7 输入规则。  
已补充：附录 S.4.2 给出 Cubical Agda 截断库路径，附录 Z 说明公理化 HoTT 与 Cubical 实现口径的差异。  
剩余义务：若机器化，需要固定 Agda/Cubical 版本和具体导入；若采用公理化 HoTT，附录 L 已把它作为输入 HIT。

**K.2.3 集合商的完整递归/归纳规则。**  
位置：第八章 8.7。  
当前状态：已列为附录 L.9-L.13 输入规则。  
剩余义务：当前正文只使用非依赖递归和集合性；完整依赖消去在需要时展开。

## K.3 圆的基本群

**K.3.1 整数类型与后继等价。**  
位置：第十一章。  
当前状态：书内证明核。  
已补充：附录 AE 证明自然数 no-confusion、自然数集合性和和类型集合性；附录 M 定义归纳整数 $\mathbb Z_{\mathsf{ind}}$、商整数 $\mathbb Z_{\mathsf{q}}$、successor/predecessor，并证明 successor 是自等价；同时给出 loop 幂和加法的基础计算规则。附录 W 进一步证明左右 successor/predecessor 平移、左右单位律、结合律、交换律和逆元律，从而补全 M.14 的整数加法交换群结构。  
剩余义务：机器形式化时需选择 Coq-HoTT、Cubical Agda 或 UniMath 的整数定义，并逐项翻译 W.1-W.13 的计算方向。

**K.3.2 code 覆盖族。**  
位置：第十一章。  
当前状态：书内证明核。  
已补充：第十一章构造 11.7 和附录 N.2-N.3 用 $\mathsf{ua}(\mathsf{succEquiv}_{\mathbb Z})$ 和圆递归定义 $\mathsf{code}:\mathbb S^1\to\mathcal U$，并展开沿 $\mathsf{loop}$ 和 $\mathsf{loop}^{-1}$ 的 transport 计算。  
剩余义务：若采用 propositional HIT computation，需要在机器化版本中显式插入基点计算路径 $c_0$。

**K.3.3 encode/decode 互逆。**  
位置：第十一章。  
当前状态：书内证明核。  
已补充：附录 N.4-N.11 定义 encode/decode，证明 decode-after-encode、encode-after-decode，并推出 $(\mathsf{base}=\mathsf{base})\simeq\mathbb Z$；附录 V 证明 loop 幂保持加法，并把该等价提升为 $\pi_1(\mathbb S^1,\mathsf{base})\cong\mathbb Z$ 的群同构。  
剩余义务：若采用 propositional HIT computation，机器化版本需显式插入基点计算路径；整数加法群律已由附录 W 补为书内证明核。

## K.4 单值范畴论

**K.4.1 预范畴与单值范畴的完整路径计算。**  
位置：第十三章。  
当前状态：书内证明核。  
已补充：附录 P 展开预范畴记录、同构类型、$\mathsf{idtoiso}$、$\mathsf{isUnivalentCat}$、集合范畴单值性和命题性公理代数结构范畴单值性；附录 X 展开一般函子范畴、自然同构与函子范畴单值性；附录 AG 补全结构 transport 与命题性公理代数结构范畴单值性证明。  
剩余义务：范畴等价提升为范畴同构依赖 Rezk 完备化泛性质；该泛性质的证明架构已在附录 AA 给出，剩余为 AA.8-AA.10 的机器级 transport 细节。

**K.4.2 Yoneda 引理。**  
位置：第十四章。  
当前状态：书内证明核。  
已补充：附录 Q 定义集合值预层、可表预层、自然变换、Yoneda evaluation 和 extension，并证明二者互逆；附录 U 进一步定义预层范畴、证明自然变换 Hom 集合性、给出自然变换复合和恒等，并把 Yoneda Hom 等价提升为 $y:\mathcal C\to\mathsf{PSh}(\mathcal C)$ fully faithful；附录 X 给出一般函子范畴和自然同构理论。  
剩余义务：Yoneda 本身的函子级 fully faithful 缺口已补齐；若机器化，需把 U 的手写预层范畴与 X 的一般函子范畴实例作定义翻译。

**K.4.3 Rezk 完备化。**  
位置：第十四章。  
当前状态：证明架构 / 外部机器化义务。  
已补充：附录 R 用 Yoneda 嵌入的本质像定义 $\widehat{\mathcal C}$，给出 Rezk 嵌入 fully faithful、essentially surjective 和单值性的证明路线；附录 AH 补全 full subcategory 保持单值性和本质像嵌入的 fully faithful/essentially surjective 证明；附录 X 证明函子范畴与预层范畴单值性；附录 AA 把泛性质 R.11 降为 weak equivalence 限制函子等价的证明架构。  
剩余义务：AA.8-AA.10 中代表元相容、对象路径 transport、自然性和函子律的全部逐行机器化仍未展开。

**K.4.4 极限与伴随基础。**  
位置：第十四章 14.6、14.8。  
当前状态：书内证明核。  
已补充：附录 AF 证明终对象之间唯一同构、单值范畴中终对象唯一到路径，并证明伴随的 Hom 等价形式与单位/余单位三角恒等式形式互相构造。  
剩余义务：若机器化，需要把自然性证明和三角恒等式的方向与目标库的复合记号对齐。

## K.5 形式化库索引

**K.5.1 Coq-HoTT 对照。**  
当前状态：已补充版本化入口索引。  
已补充：附录 S.2 固定 Coq-HoTT commit `a030184c0bfc9d61f3bcd33c67660b800e106427`，并为路径代数、等价、单值性、截断、商、整数、圆的 encode-decode、$\Omega(\mathbb S^1)\simeq\mathbb Z$ 和范畴论入口列出模块路径与 identifier。  
剩余义务：若把本书机器化到 Coq-HoTT，需要补完整导入图、Coq 版本、HoTT 编译选项、universe 约束和每条定理的脚本级依赖。

**K.5.2 UniMath 对照。**  
当前状态：已补充版本化入口索引。  
已补充：附录 S.3 固定 UniMath commit `9ed7661d3ad33c74e35824efccf861b4fdc17323`，并为 h-level 命题性、预范畴、单值范畴、同构、fully faithful、Yoneda、SIP 和代数结构范畴列出模块路径与 identifier。  
剩余义务：需要逐项比较 UniMath 的 `UU`、`isweq`、`hSet`、`category`、`z_iso` 与本书定义，并为移植结果写出翻译引理。

**K.5.3 Cubical Agda 对照。**  
当前状态：已补充版本化入口索引。  
已补充：附录 S.4 固定 Cubical Agda commit `92166033326aa59800a580b428125f3c654b5e45`，并为 cubical path、函数外延性、Glue/univalence、h-level、HIT、截断、商、单值集合范畴、预层范畴、Rezk 完备化、群结构等同和 Eilenberg-Mac Lane cohomology 列出模块路径与入口；附录 Z 给出这些实现入口与本书对象语言的边界。  
剩余义务：需要补 Agda 版本、cubical 选项、构建命令，并区分稳定模块、`ZCohomology` 旧接口和 `Experiments` 目录中的实验入口。

## K.6 合成同伦论与上同调

**K.6.1 Eilenberg-Mac Lane 型与上同调。**  
位置：第十二章。  
当前状态：证明核 / 形式化入口。  
已补充：附录 Y 给出 EM 型塔输入、非约化和约化上同调定义、阿贝尔群结构、反变函子性、悬挂同构、球面计算、cup product 和 Eilenberg-Steenrod 性质的精确状态；附录 AC 补全 Eckmann-Hilton 和高阶同伦群交换性；第十二章已改为引用 Y、AC。  
剩余义务：完整书内构造 EM 型塔、谱层级结构、长正合列和 graded commutativity 的所有高阶相干仍属于高级形式化工程；当前版本使用 Cubical Agda 入口作为可审计外部来源。

## K.7 基础 HIT 例子

**K.7.1 $\mathsf{susp}(\mathbf 2)\simeq\mathbb S^1$。**  
位置：第十章 10.5。  
当前状态：书内证明核。  
已补充：附录 AD 构造 $\mathsf{susp}(\mathbf 2)\to\mathbb S^1$ 和反向函数，证明两个复合与恒等同伦，并由准逆推出等价。  
剩余义务：机器化时若采用 propositional HIT computation，需要显式插入圆和悬挂路径构造子的计算路径。

**K.7.2 HIT 构造的等价不变性。**  
位置：第十章 10.12。  
当前状态：书内证明核 / 一般情形研究边界。  
已补充：附录 L 给出 pushout、悬挂等 HIT 输入规则；附录 AD 处理了核心低维例子；附录 AI 证明 pushout 的等价不变性，并推出 cofiber、wedge 等由 pushout 生成的基础构造的等价不变性。  
剩余义务：任意高阶图形的一般同伦余极限函子性仍属于一般 HIT 元理论，需形式化引用或单独展开。

**K.6.2 稳定同伦论与谱序列。**  
位置：第十二、十七章。  
当前状态：研究边界。  
已补充：第十二章和附录 Y 明确区分 EM 上同调证明核与更高稳定同伦工具。  
剩余义务：若要达到稳定同伦论教材级覆盖，需要单独展开谱、谱范畴、exact couples、谱序列收敛和典型计算；本书当前不把这些作为已证明基础定理。
