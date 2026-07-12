# 第十四章：模型范畴中的 operad

本章进入同伦论口径。第一至第十三章中 operad 多在集合、模或链复形中定义；本章的目标是说明：何时可以把“逐 arity 弱等价”提升为 operad 的同伦理论，何时可以把 operad 代数范畴也赋予模型结构，以及何时一个 cofibrant resolution 真正给出可替换的同伦代数理论。

本章不预设一个未展开的“足够良好”总假设包。每个外部定理分别列出所需的单位、区间、monoid axiom、对称 h-monoidality、symmetric flatness、cofibrancy 或 admissibility 条件；这些条件只服务于对应结论。完整证明涉及小对象论证、树形 filtrations 和等变 cofibration 技术，故大型定理标为外部输入。

## 14.1 对称幺半模型范畴

**定义 14.1.** 一个模型范畴是范畴 $\mathcal M$ 连同三类态射：
$$
\mathsf W,\qquad \mathsf{Cof},\qquad \mathsf{Fib},
$$
分别称为弱等价、cofibration 和 fibration，满足 Quillen 模型范畴公理：

1. $\mathcal M$ 有所有小极限和小余极限。
2. $\mathsf W$ 满足 two-out-of-three：若 $f,g,gf$ 中两个在 $\mathsf W$，则第三个也在 $\mathsf W$。
3. 三类态射都对 retract 封闭。
4. 若 $i$ 是 cofibration 且 $p$ 是 fibration，并且二者之一为弱等价，则任意交换方块
   $$
   \begin{array}{ccc}
   A & \longrightarrow & X\\
   {\scriptstyle i}\downarrow & & \downarrow{\scriptstyle p}\\
   B & \longrightarrow & Y
   \end{array}
   $$
   有 lift $B\to X$。
5. 任意态射可函子地分解为
   $$
   X\overset{i}{\longrightarrow} Z\overset{p}{\longrightarrow}Y
   $$
   其中 $i$ 是 cofibration，$p$ 是 trivial fibration；也可分解为 trivial cofibration 后接 fibration。

**定义 14.2.** 对称幺半模型范畴是闭对称幺半范畴
$$
(\mathcal M,\otimes,\mathbb 1,\underline{\operatorname{Hom}})
$$
连同模型结构，使得：

1. $\otimes$ 是双变量左伴随；
2. pushout-product axiom 成立：若 $i:A\to B$ 与 $j:C\to D$ 是 cofibrations，则
   $$
   i\square j:(B\otimes C)\coprod_{A\otimes C}(A\otimes D)\to B\otimes D
   $$
   是 cofibration；若 $i$ 或 $j$ 是 trivial cofibration，则 $i\square j$ 是 trivial cofibration；
3. unit axiom 成立：若 $q:Q\mathbb 1\to\mathbb 1$ 是单位对象的 cofibrant replacement，则对每个 cofibrant 对象 $X$，态射
   $$
   q\otimes X:Q\mathbb 1\otimes X\to\mathbb 1\otimes X
   $$
   是弱等价。

**定义 14.3.** $\mathcal M$ 称为 cofibrantly generated，若存在集合 $I$ 和 $J$，使得：

1. $I$ 生成 cofibrations；
2. $J$ 生成 trivial cofibrations；
3. 小对象论证可用于 $I$ 与 $J$。

这里“生成”意为：$I$-injective 正好是 trivial fibrations，$J$-injective 正好是 fibrations。

**定义 14.4.** 设 $\mathcal M$ 是对称幺半模型范畴。若对任意 trivial cofibration $f$，由所有态射
$$
f\otimes X,\qquad X\in\mathcal M
$$
经过 pushout、transfinite composition 和 retract 得到的态射全为弱等价，则称 $\mathcal M$ 满足 monoid axiom。

**说明 14.5.** Monoid axiom 控制“自由 monoid 或自由 algebra 的胞腔附加”是否保持弱等价。Operad 代数的情形比 monoid 更复杂，因为自由 $\mathcal O$-代数包含对称群 coinvariants；这要求额外的 $\Sigma_n$-等变控制。

## 14.2 对称序列的 projective 模型结构

设 $\mathcal M$ 是 cofibrantly generated 模型范畴，并且所有小极限和小余极限存在。令
$$
\operatorname{SymSeq}(\mathcal M)=\operatorname{Fun}(\mathbf B_{\mathcal U},\mathcal M).
$$

**定义 14.6.** 对称序列态射 $f:X\to Y$ 称为 projective weak equivalence，若对每个有限集 $S$，
$$
f(S):X(S)\to Y(S)
$$
是 $\mathcal M$ 中的弱等价。

称 $f$ 为 projective fibration，若每个 $f(S)$ 是 $\mathcal M$ 中的 fibration。

**命题 14.7（作用范畴的积模型）.** 假设每个 $\mathcal M^{\Sigma_n}$ 都有由底层 $\mathcal M$ 创建 weak equivalences 与 fibrations 的 projective 模型结构。则 $\operatorname{SymSeq}(\mathcal M)$ 有 projective 模型结构，weak equivalences 与 fibrations 逐 arity 检测。

**证明.** 命题 A.7 给出范畴等价
$$
\operatorname{SymSeq}(\mathcal M)
\simeq
\prod_{n\ge0}\mathcal M^{\Sigma_n}.
$$
右侧的积模型结构逐分量定义三类态射并逐分量完成 lifting 与 factorization；模型范畴公理因而逐分量成立。把该结构沿范畴等价搬回左侧，所得 weak equivalences 与 fibrations 正是定义 14.6 的逐 arity 类。$\square$

**说明 14.8.** 这个模型结构不是 operad 的模型结构；它只是底层对称序列的模型结构。Operad 是 $\operatorname{SymSeq}(\mathcal M)$ 中代入乘积 $\circ$ 下的 monoid。把模型结构从对称序列转移到 monoids 需要额外假设。

**定义 14.9.** 对称序列 $X$ 称为 $\Sigma$-cofibrant，若对每个 $n\ge0$，对象 $X(n)$ 作为 $\mathcal M^{\Sigma_n}$ 中的对象是 cofibrant。

**命题 14.10.** 选择骨架
$$
\mathbf B_{\mathcal U}\simeq\coprod_{n\ge0}B\Sigma_n,
$$
并假设每个 $\mathcal M^{\Sigma_n}$ 都有逐底层态射检测 weak equivalence 与 fibration 的 projective 模型结构。则有模型范畴等价
$$
\operatorname{SymSeq}(\mathcal M)
\simeq
\prod_{n\ge0}\mathcal M^{\Sigma_n}.
$$
因此 $X$ projectively cofibrant 当且仅当每个 $X(n)$ 在 $\mathcal M^{\Sigma_n}$ 中 cofibrant；特别地，projectively cofibrant 蕴含 $\Sigma$-cofibrant。

若此外遗忘函子 $\mathcal M^{\Sigma_n}\to\mathcal M$ 保持 cofibrations，则 $X(n)$ 的底层对象在 $\mathcal M$ 中 cofibrant。在由自由 $\Sigma_n$-对象 $\Sigma_n\cdot i$ 生成的标准 projective 结构中，这个保持性质成立。

**证明.** 命题 A.7 把对称序列函子范畴识别为各作用范畴的积；定义 14.6 的 weak equivalences 和 fibrations 恰为积模型结构中的逐分量类。积模型结构的 cofibration 也逐分量检测，故
$$
\varnothing\to X
$$
是 cofibration 当且仅当每个 $\varnothing\to X(n)$ 是 $\mathcal M^{\Sigma_n}$ 中的 cofibration。这证明第一段。第二段在遗忘函子保持 cofibration 时直接应用于 $\varnothing\to X(n)$。标准 projective 结构的生成 cofibrations 为 $\Sigma_n\cdot i$；遗忘后是有限个 $i$ 的 coproduct。Cofibrations 对 coproduct、pushout、transfinite composition 和 retract 封闭，所以遗忘函子保持所有由这些生成元生成的 cofibrations。$\square$

## 14.3 Operad 的 transferred 模型结构

令
$$
U:\operatorname{Op}(\mathcal M)\to \operatorname{SymSeq}(\mathcal M)
$$
为遗忘函子。若自由 operad 函子存在，记为
$$
\mathbb F:\operatorname{SymSeq}(\mathcal M)\rightleftarrows\operatorname{Op}(\mathcal M):U.
$$

**定义 14.11.** $\operatorname{Op}(\mathcal M)$ 上的 transferred 模型结构指如下模型结构：

1. operad morphism $f:\mathcal P\to\mathcal Q$ 是弱等价，当且仅当 $U(f)$ 是 projective weak equivalence；
2. $f$ 是 fibration，当且仅当 $U(f)$ 是 projective fibration；
3. cofibration 由左提升性质确定。

若该模型结构存在，本书只说“$\operatorname{Op}(\mathcal M)$ 的 transferred 模型结构存在”。术语 admissible 保留给定义 14.12 和定义 14.19 的固定 operad 代数范畴，以免把两个转移问题混为一谈。

**定义 14.12.** 一类 colored operads 称为 admissible，若对该类中每个 operad $\mathcal O$，其代数范畴 $\operatorname{Alg}_{\mathcal O}(\mathcal M)$ 存在从 $\mathcal M^C$ 转移来的模型结构，其中 $C$ 为颜色集。

注意定义 14.11 是 operad 自身的模型结构，定义 14.12 是固定 operad 的代数范畴模型结构。两者相关但不等同。

**外部输入定理 14.13（operad 转移；BM-1 的本书版本）.** 设 $\mathcal M$ 满足 Berger--Moerdijk, arXiv:math/0206094v3, Theorem 3.1 的假设包：$\mathcal M$ 是 cofibrantly generated symmetric monoidal model category，单位 cofibrant，生成映射满足来源中的小性，并且给定 symmetric monoidal fibrant replacement 与来源要求的 commutative Hopf interval。则 $\operatorname{Op}(\mathcal M)$ 存在 transferred 模型结构，weak equivalences 与 fibrations 由底层 collections 逐 arity 创建。

本书不重证自由 operad 的等变树形胞腔论证。若不用上述 BM-1 假设包，必须改引一个明确的现代转移定理并逐条登记其假设；monoid axiom 本身不替代 Hopf interval、等变 cofibration 或树形 filtration 条件。

**说明 14.14.** 自由 operad $\mathbb F(X)$ 由 $X$-装饰树构造。沿生成 trivial cofibration 附加 generator 会产生所有带标记顶点的树形项。外部证明必须控制这些项经过 $\otimes$、有限群 coinvariants、pushout 和 transfinite composition 后仍为 weak equivalences。这解释了 14.13 的附加假设，但不构成该外部定理的书内证明。

**命题 14.15.** 若 transferred 模型结构存在，则遗忘函子
$$
U:\operatorname{Op}(\mathcal M)\to\operatorname{SymSeq}(\mathcal M)
$$
是右 Quillen 函子，自由 operad 函子 $\mathbb F$ 是左 Quillen 函子。

**证明.** transferred 模型结构按定义使得 $U$ 保 fibration 和 trivial fibration：一个 operad morphism 是 fibration 或 trivial fibration 当且仅当其底层对称序列有相应性质。右 Quillen 函子的定义正是保 fibration 与 trivial fibration。因此 $U$ 是右 Quillen，其左伴随 $\mathbb F$ 是左 Quillen。$\square$

**定义 14.16.** Operad $\mathcal P$ 称为 cofibrant operad，若初 operad 到 $\mathcal P$ 的态射是 $\operatorname{Op}(\mathcal M)$ 中的 cofibration。

称 $\mathcal P$ 为 entrywise cofibrant，若每个 $\mathcal P(n)$ 是 $\mathcal M$ 中 cofibrant。称 $\mathcal P$ 为 $\Sigma$-cofibrant，若其底层对称序列是 $\Sigma$-cofibrant。

**命题 14.17.** 在 transferred 模型结构存在且遗忘函子保持 cofibrant 对象到 projective cofibrant 对称序列的情形下，cofibrant operad 是 $\Sigma$-cofibrant。

**证明.** 若 $\mathcal P$ cofibrant，则初 operad $\mathcal I\to\mathcal P$ 为 cofibration。若 $U$ 保持该 cofibration 到 projective cofibration，则 $U\mathcal P$ 是 projective cofibrant 对称序列。由命题 14.10，$U\mathcal P$ 为 $\Sigma$-cofibrant。$\square$

**警告 14.18.** 命题 14.17 的保持条件不是形式结论。许多常用范畴中它成立或可在修正假设下成立，但在一般对称幺半模型范畴中，cofibrant operad 与底层 $\Sigma$-cofibrant 对称序列之间没有无条件等价。

## 14.4 固定 operad 的代数范畴

设 $\mathcal O$ 是 $C$-colored operad in $\mathcal M$。其代数范畴记为
$$
\operatorname{Alg}_{\mathcal O}(\mathcal M).
$$
遗忘函子
$$
U_{\mathcal O}:\operatorname{Alg}_{\mathcal O}(\mathcal M)\to \mathcal M^C
$$
通常有左伴随自由代数函子 $F_{\mathcal O}$。

**定义 14.19.** $\mathcal O$ 称为 admissible，若 $\operatorname{Alg}_{\mathcal O}(\mathcal M)$ 存在 transferred 模型结构，使得代数态射 $f:A\to B$ 是弱等价或 fibration 当且仅当每个颜色 $c\in C$ 上
$$
f_c:A_c\to B_c
$$
是 $\mathcal M$ 中的弱等价或 fibration。

**命题 14.20.** 若 $\mathcal O$ admissible，则
$$
F_{\mathcal O}:\mathcal M^C\rightleftarrows \operatorname{Alg}_{\mathcal O}(\mathcal M):U_{\mathcal O}
$$
是 Quillen adjunction。

**证明.** transferred 模型结构按定义由 $U_{\mathcal O}$ 检测 fibration 和 weak equivalence。因此 $U_{\mathcal O}$ 保 fibration 和 trivial fibration。故 $U_{\mathcal O}$ 是右 Quillen 函子，左伴随 $F_{\mathcal O}$ 是左 Quillen 函子。$\square$

**外部输入定理 14.21（colored admissibility；PSAR-2）.** 固定 $\mathcal U$-小颜色集 $C$。设 $\mathcal M$ 满足 Pavlov--Scholbach, arXiv:1410.5675v4, Definition 2.1 与 Theorem 5.11 的组合假设：$\mathcal M$ 为来源意义下的 combinatorial 或 admissibly generated、tractable symmetric monoidal model category，并且 symmetric h-monoidal。则每个 $C$-colored symmetric operad $\mathcal O$ admissible；即 $\operatorname{Alg}_{\mathcal O}(\mathcal M)$ 上存在 transferred 模型结构，weak equivalences 与 fibrations 在 $\mathcal M^C$ 中逐颜色创建。

Berger--Moerdijk 的较早固定-operad版本使用 BM-2，并要求该定理中的 operad 类型、cofibrancy 和底范畴假设。PSAR-2 不要求逐个 operad 先证明 $\Sigma$-cofibrant，但其 symmetric h-monoidality、tractability 和生成性假设不能省略。Symmetric flatness 主要进入 rectification，而不是 14.21 的存在性结论。

**例 14.22.** 在特征 $0$ 的域 $k$ 上，链复形范畴 $\mathbf{Ch}_k$ 中有限群表示范畴半单。因此许多对称群 coinvariants 与同调相容性问题大幅简化。由此 dg-operads 和其代数的模型结构较一般底环更稳定。

**例 14.23.** 在一般交换环 $R$ 上，$\mathbf{Ch}_R$ 中 coinvariants 不必保持 quasi-isomorphism。因此 Com-operad 的严格代数，即 commutative dg algebras，未必具有与特征 $0$ 情形同样良好的 transferred 模型结构。此处必须区分：

1. associative dg algebras；
2. commutative dg algebras；
3. $E_\infty$-algebras；
4. divided power 或 derived commutative 变体。

这些对象在一般底环上不能通过一句“取 cofibrant replacement”互相替代。

**例 14.23.1（对称幂风险的显式来源）.** 命题 X.15 给出 $k=\mathbb F_p$ 上 acyclic chain complex $C$，但 $\operatorname{Sym}^p(C)$ 具有非零同调类；推论 X.16 说明自由 commutative dg algebra functor 因此不能在正特征中无条件保持 quasi-isomorphisms。

**结论.** 例 14.23.1 只证明一个必要风险，不等于完整的非 rectification theorem。完整结论需要附录 D 的模型结构输入表、例 R.19 和 P0 引用定位批次 1 中 BM-4--BM-5 的外部输入；现代 symmetric flatness/rectification 版本按 P0 引用定位批次 9 中 PSAR-1--PSAR-6 与 PSP-1--PSP-2 使用。该例不承载这些外部定理，只标出其假设不可随意删除。

## 14.5 Weak equivalence of operads 与 rectification

设
$$
\varphi:\mathcal O\to\mathcal P
$$
是 $C$-colored operad morphism。存在 restriction functor
$$
\varphi^\*:\operatorname{Alg}_{\mathcal P}(\mathcal M)\to\operatorname{Alg}_{\mathcal O}(\mathcal M).
$$
若相应左伴随存在，记为
$$
\varphi_!:\operatorname{Alg}_{\mathcal O}(\mathcal M)\rightleftarrows\operatorname{Alg}_{\mathcal P}(\mathcal M):\varphi^\*.
$$

**定义 14.24.** 若 $\varphi_!\dashv\varphi^\*$ 是 Quillen equivalence，则称 $\varphi$ induces rectification。直观地说，$\mathcal O$-同伦代数可在同伦范畴中由严格 $\mathcal P$-代数替代。

**命题 14.25.** 若 $\mathcal O$ 与 $\mathcal P$ admissible，且 $\varphi^\*$ 保 fibration 与 trivial fibration，则
$$
\varphi_!\dashv\varphi^\*
$$
是 Quillen adjunction。

**证明.** 由 admissibility，两个代数范畴的 fibration 和 weak equivalence 都在底层 $\mathcal M^C$ 中检测。Restriction functor $\varphi^\*$ 不改变底层 $C$-indexed 对象，只改变结构映射。因此它保 fibration 和 weak equivalence，特别保 fibration 与 trivial fibration。故它是右 Quillen 函子。$\square$

**外部输入定理 14.26（rectification criterion；PSAR-4）.** 固定颜色集 $C$。设 $\mathcal M$ 是 tractable、symmetric h-monoidal 的 symmetric monoidal model category，且
$$
\varphi:\mathcal O\to\mathcal P
$$
是 admissible $C$-colored symmetric operads 之间的 morphism。Pavlov--Scholbach, arXiv:1410.5675v4, Theorem 7.5 给出的 free-cofibrant-algebra comparison 条件成立时，Quillen adjunction
$$
\varphi_!:\operatorname{Alg}_{\mathcal O}(\mathcal M)\rightleftarrows\operatorname{Alg}_{\mathcal P}(\mathcal M):\varphi^\*
$$
是 Quillen equivalence；$\varphi$ 对来源意义下的 symmetric flatness 是该比较条件的充分条件。因而正文若只知道 $\varphi$ entrywise weak equivalence，还必须另外验证 symmetric flatness，或直接验证 Theorem 7.5 的 free-cofibrant-algebra 条件。

**说明 14.27.** Rectification 不是自动的。若 $\mathcal O\to\mathcal P$ 是逐 arity 弱等价，但 $\mathcal O$ 不够 cofibrant，或 $\mathcal M$ 中对称幂不保持弱等价，则代数范畴可能不 Quillen equivalent。

**推论 14.28.** 若 $A_\infty\to\operatorname{Ass}$ 是 dg-operads 的 cofibrant resolution，两端 operads admissible，并且该态射满足外部输入定理 14.26 的 free-cofibrant-algebra comparison 条件（例如已验证相应 symmetric flatness），则 $A_\infty$-algebras 与 dg associative algebras 的模型范畴 Quillen equivalent。

**证明.** Admissibility 先给出 extension--restriction Quillen adjunction。其余假设正是外部输入定理 14.26 的充分输入，所以该 adjunction 是 Quillen equivalence。仅由“cofibrant resolution”得到的 entrywise weak equivalence 不足以完成此步。$\square$

**警告 14.29.** 对 $E_\infty\to\operatorname{Com}$ 不能无条件推出同样结论。在特征 $0$ 链复形中，只有选定具体 $E_\infty$-operad、验证两端 admissible 并套用 BM-4/HIN-2/PSAR-4 的相应版本后，才可声明 rectification。在正特征或一般底环上，命题 X.15 直接表明自由严格交换代数函子不保持某些 acyclic complexes；本书据此拒绝无假设 rectification，但不把这个局部计算夸大为所有模型中的完整非等价定理。

**说明 14.29.1.** 后续凡使用 transferred operad model structure、operad algebra admissibility 或 rectification，必须逐项核对定义 G.3--定义 G.6 和外部输入定理 G.11--外部输入定理 G.13 的检查表。特别是：

1. operads 自身有模型结构；
2. 固定 operad 的代数范畴有 transferred 模型结构；
3. operad weak equivalence 诱导代数范畴 Quillen equivalence；

这三件事应分别证明或分别引用外部输入定理，不能互相替代。

## 14.6 Boardman-Vogt resolution

**定义 14.30.** 一个 operad $\mathcal O$ 的 cofibrant resolution 是 cofibrant operad $Q\mathcal O$ 与弱等价
$$
Q\mathcal O\longrightarrow \mathcal O.
$$

若 $Q\mathcal O$ 由树、边长参数和顶点装饰构造，并通过收缩零长度内边编码复合，则称其为 Boardman-Vogt 型 resolution，记作 $W\mathcal O$。

**外部边界 14.31（Boardman-Vogt resolution）.** 在具体来源给定的区间对象、well-pointedness、$\Sigma$-cofibrancy 及幺半模型范畴假设下，经典 Boardman--Vogt 定理可给出 functorial map
$$
W\mathcal O\to\mathcal O
$$
并在该来源的模型结构中判定其 cofibrancy 与 weak-equivalence 性质。本书目前没有登记覆盖这一整套 $W$-construction 的精确 locator，故不把本段作为可直接调用的外部定理。后文命题 14.33 只作条件推理：必须另行给出所用 $W\mathcal O\to\mathcal O$ 确为 cofibrant resolution 的定理，才能应用 rectification。

**说明 14.32.** 拓扑 operad 的 $W$-construction 可理解为：一个点由一棵树给出，顶点标记为 $\mathcal O$ 中的运算，内部边标记为区间参数。边长为 $0$ 时收缩该边并用 $\mathcal O$ 的复合替代两个相邻顶点。边长为 $1$ 的边记录未被严格复合的同伦层级。

**命题 14.33.** 若 $W\mathcal O\to\mathcal O$ 是 cofibrant resolution，并且 rectification criterion 对该弱等价适用，则 $W\mathcal O$-algebras 与 $\mathcal O$-algebras 给出 Quillen equivalent 同伦理论。

**证明.** 这是外部输入定理 14.26 应用于 $\varphi:W\mathcal O\to\mathcal O$ 的直接结果。$\square$

## 14.7 Derived mapping spaces of operads

设 $\operatorname{Op}(\mathcal M)$ 有模型结构，且 $\mathcal M$ 为 simplicial model category 或有合适的 framings。

**定义 14.34.** Operads $\mathcal P,\mathcal Q$ 的 derived mapping space 定义为
$$
\mathbf R\operatorname{Map}_{\operatorname{Op}(\mathcal M)}(\mathcal P,\mathcal Q)
=\operatorname{Map}(Q\mathcal P,R\mathcal Q),
$$
其中 $Q\mathcal P\to\mathcal P$ 是 cofibrant replacement，$\mathcal Q\to R\mathcal Q$ 是 fibrant replacement。

**命题 14.35.** Derived mapping space 的弱同伦型不依赖于 replacement 的选择。

**证明.** 模型范畴中的 mapping space 在 cofibrant-fibrant 变量上是同伦不变量。若 $Q\mathcal P$ 与 $Q'\mathcal P$ 是两个 cofibrant replacements，则二者在 slice of weak equivalences over $\mathcal P$ 中由 zigzag of weak equivalences 相连。对 fibrant $R\mathcal Q$ 取 mapping space，得到 weak equivalences of simplicial sets。同理可替换 fibrant replacement。故所得弱同伦型只由 $\mathcal P,\mathcal Q$ 决定。$\square$

**说明 14.36.** Derived mapping spaces 是研究 operad 形式性和 automorphism 的核心工具。例如 $E_n$-operads 的 derived automorphism 与 Grothendieck-Teichmuller 型对象相关；这些结果属于更深的同伦 operad 理论，后续章节只在资料边界中使用。

## 14.8 本章小结

模型范畴中的 operad 理论有三个层次：

1. 对称序列的 projective 模型结构；
2. operads 自身的 transferred 模型结构；
3. 固定 operad 的代数范畴模型结构。

第一个层次通常最容易建立，第二和第三个层次需要额外的幺半、等变和小性条件。Weak equivalence of operads 只有在 admissibility 与 rectification 条件满足时才可替换代数的同伦理论。Cofibrant resolution，尤其 Boardman-Vogt resolution，是把严格代数结构替换为同伦相干结构的标准机制。

定义 G.3--定义 G.6 和外部输入定理 G.11--外部输入定理 G.13 给出本章所有模型结构假设的检查表。P0 引用定位批次 1 已定位 Berger--Moerdijk 的 BM-1--BM-5 和 Cisinski--Moerdijk 的 CM-1--CM-4；P0 引用定位批次 5 已定位 Hinich 的 HIN-1--HIN-2 和 Fresse 的 FRE-1--FRE-6；P0 引用定位批次 9 已定位 Pavlov--Scholbach colored/all-small/symmetric-flatness 版本的 PSAR/PSP 条目。最终出版只需把本章所用假设逐条对齐到相应 locator。

## 练习

**练习 14.1.** 证明 transferred 模型结构若存在，则弱等价满足 two-out-of-three。

**练习 14.2.** 设 $\mathcal M$ 是模型范畴。证明 $\mathcal M^C$ 的 projective 模型结构中，弱等价和 fibration 逐颜色检测。

**练习 14.3.** 对 $\mathcal O=\operatorname{Ass}$，写出自由 $\mathcal O$-代数函子，并解释为何其构造涉及张量幂但不涉及非平凡对称群 coinvariants。

**练习 14.4.** 对 $\mathcal O=\operatorname{Com}$，写出自由 commutative algebra
$$
\operatorname{Sym}(X)=\coprod_{n\ge0}(X^{\otimes n})_{\Sigma_n},
$$
并说明正特征中 coinvariants 可能破坏 quasi-isomorphism 的原因。

**练习 14.5.** 设 $\varphi:\mathcal O\to\mathcal P$ 为 operad weak equivalence。解释为什么“逐 arity 弱等价”本身不足以推出 $\operatorname{Alg}_{\mathcal O}$ 与 $\operatorname{Alg}_{\mathcal P}$ 的 Quillen equivalence。
