# 附录 K：关键依赖与不可逆边界

HoTT 的结论依赖于对象语言中已经加入的规则。低层路径代数可以被高层单值性章节使用，反向依赖却会使基础定理偷偷获得更强公理。本附录按逻辑层级列出全书的主要依赖，并指出外部输入与扩展语言停止在何处；它不是编辑进度或验收清单。

## K.1 基础内部链

**K.1.1 判断与类型形成。** 第一章只使用语境、替换、judgmental equality、非累积宇宙、$\Pi$ 与 $\Sigma$。这一层不能形成内部相等类型，也不能使用函数外延性。

**K.1.2 恒等类型与路径代数。** 第二章加入 $\mathsf{Id}$、$\mathsf{refl}$ 与 $J$，由此构造 transport、$\mathsf{ap}$、$\mathsf{apd}$、逆和复合。附录 A、D 展开 $\Sigma$ 路径与 transport。这里的群胚律是路径，不是 judgmental equality。

**K.1.3 归纳类型、层级与等价。** 第三至五章加入基础归纳类型，以
$$
\mathsf{isContr}(A),\quad
\mathsf{isProp}(A),\quad
\mathsf{isSet}(A)
$$
递归组织路径空间，并把等价定义为所有 fibers 可收缩。附录 E、G、AB、O 只依赖这一层及以下规则；涉及函数路径的命题性结论会显式要求函数外延性。

**K.1.4 外延性与单值性。** 第六、七章加入函数外延性和各 universe 的 univalence。它们允许把逐点路径变成函数路径，把类型等价变成 universe path，但不扩张 judgmental equality，也不提供 resizing。第一至五章的证明不得逆向使用这一层。

**K.1.5 截断与 HIT。** 第八至十章采用附录 L 的命题截断、集合截断、集合商、圆、悬挂与 pushout 规则包。每个规则包分别固定 universe、消去目标和计算强度；一般 HIT schema 不在基础语言中。圆的 encode--decode 还使用单值性把整数后继等价变成 universe loop。

**K.1.6 单值一范畴。** 第十三、十四章使用集合值 Hom、函数外延性、命题截断和单值性。Yoneda、函子范畴、极限与伴随的证明核位于 P、Q、U、X、AF；Rezk 完备化对象和 weak-equivalence 性质位于 R。Rezk 泛性质本身采用附录 AA.8 的精确外部输入。

## K.2 明确的外部输入

**K.2.1 单值性推出函数外延性。** 附录 T 采用 HoTT Book Theorems 4.9.4--4.9.5，只覆盖基底与 fibers 位于同一单值 universe 的实例。非累积宇宙间没有隐式 lift。

**K.2.2 Rezk 限制泛性质。** 附录 AA 采用 Ahrens--Kapulkin--Shulman 2015, Theorem 8.4。代表元相容、对象路径对 Hom 的 transport、扩张函子律均留在外部来源证明中。

**K.2.3 高级不稳定同伦论。** Blakers--Massey 与 Freudenthal 采用 AL、AU、AY 中列出的精确外部版本，Hopf fiber sequence 另按 AL 的输入使用。Path-code 是来源证明机制，不是调用 Blakers--Massey 时额外要求读者提供的假设；实际调用只列定理中的连通性条件，但不得改变其编号约定或据此声称已内部构造一般 pushout 路径空间。

**K.2.4 稳定同伦与上同调。** EM 型塔、谱模型、Steenrod operations、Adams resolution 和具体谱序列收敛分别是 Y、AZ、BN、AV 的高级输入。第十二章只在列出这些输入后作条件化推导。

**K.2.5 模型元定理。** Simplicial univalent universe、CCHM Glue univalence、cubical canonicity/normalization 与 strict Rezk completion 都按第十五至十七章所列的具体语法和元理论假设引用。一个模型的结论不能改变另一语法的 judgmental equality。

## K.3 扩展对象语言

Directed/simplicial type theory 新增 directed interval、shapes 或 extension types；two-level type theory 新增 strict equality 层；cohesive HoTT 新增模态与可能的 crisp 变量；HIIT/QIIT 新增同时生成类型、项和相等的签名。这些扩展分别在 AS/AX、BG、AT/BD、BC 中说明。

它们与基础 HoTT 的接口必须由翻译或模型解释给出。特别地：

1. directed hom 不因名称相近而成为 identity path；
2. strict equality 不能未经桥接变成 fibrant path；
3. cohesive 模态不由构造性实数或普通 truncation 推出；
4. 某个列举型 HIT 的计算语义不推出任意 QIIT 的 canonicity。

## K.4 分析与集合层接口

集合商、代数局部化、有限集、基数和序数依赖第八章的截断与商规则，具体 well-defined 证明位于 BH、BI。Cauchy/Dedekind 实数与构造性分析位于 AK、AR、AW、BA、BO；它们额外使用 HIIT 输入、locatedness、完备性或有理误差预算。

附录 BO 的 Banach 定理明确需要 $\|X\|$。其 complete 假设只给 mere limit；先证明极限总类型是命题后，才能消去截断。这个例子概括了全书的依赖纪律：每次使用更强消去原则前，先证明目标具有允许的同伦层级。
