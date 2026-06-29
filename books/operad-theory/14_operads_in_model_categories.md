# 第十四章：模型范畴中的 operad

本章进入同伦论口径。第一至第十三章中 operad 多在集合、模或链复形中定义；本章的目标是说明：何时可以把“逐 arity 弱等价”提升为 operad 的同伦理论，何时可以把 operad 代数范畴也赋予模型结构，以及何时一个 cofibrant resolution 真正给出可替换的同伦代数理论。

本章只在一类足够良好的对称幺半模型范畴中陈述主定理。完整证明涉及小对象论证、monoid axiom、pushout-product axiom、树形 filtrations 和等变 cofibration 技术，故大型定理标为外部输入。

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
   \begin{CD}
   A @>>> X\\
   @V i VV @VV p V\\
   B @>>> Y
   \end{CD}
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

**外部输入定理 14.7.** 若 $\mathcal M$ 是 cofibrantly generated 模型范畴，则 $\operatorname{SymSeq}(\mathcal M)$ 存在 projective 模型结构，其弱等价和 fibration 逐 arity 定义。

**说明 14.8.** 这个模型结构不是 operad 的模型结构；它只是底层对称序列的模型结构。Operad 是 $\operatorname{SymSeq}(\mathcal M)$ 中代入乘积 $\circ$ 下的 monoid。把模型结构从对称序列转移到 monoids 需要额外假设。

**定义 14.9.** 对称序列 $X$ 称为 $\Sigma$-cofibrant，若对每个 $n\ge0$，对象 $X(n)$ 作为 $\mathcal M^{\Sigma_n}$ 中的对象是 cofibrant。

**命题 14.10.** 若 $\mathcal M$ 的终对象和初对象存在，则 projective cofibrant 对称序列逐 arity 是 cofibrant；若 $\mathcal M^{\Sigma_n}$ 的 projective 模型结构存在，则 projective cofibrant 对称序列是 $\Sigma$-cofibrant。

**证明.** Projective 模型结构中 evaluation 函子
$$
\operatorname{ev}_{[n]}:\operatorname{SymSeq}(\mathcal M)\to\mathcal M^{\Sigma_n}
$$
是右 Quillen 函子。它的左伴随把带 $\Sigma_n$-作用的对象放在 arity $n$ 并在其他 arity 取初对象。右 Quillen 函子保 fibrations 和 trivial fibrations，因此左伴随保 cofibrations 和 trivial cofibrations。由 cofibrant 对象定义，$\varnothing\to X$ 为 cofibration；对其应用 evaluation，得到 $\varnothing\to X(n)$ 是 $\mathcal M^{\Sigma_n}$ 中的 cofibration。故 $X(n)$ cofibrant。$\square$

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

若该模型结构存在，则称 $\mathcal M$ 中的 operads admissible。

**定义 14.12.** 一类 colored operads 称为 admissible，若对该类中每个 operad $\mathcal O$，其代数范畴 $\operatorname{Alg}_{\mathcal O}(\mathcal M)$ 存在从 $\mathcal M^C$ 转移来的模型结构，其中 $C$ 为颜色集。

注意定义 14.11 是 operad 自身的模型结构，定义 14.12 是固定 operad 的代数范畴模型结构。两者相关但不等同。

**外部输入定理 14.13（Berger-Moerdijk 型转移定理）.** 设 $\mathcal M$ 是 cofibrantly generated 对称幺半模型范畴，并满足适当的小性、monoid axiom、单位和等变 cofibration 条件。则 $\operatorname{Op}(\mathcal M)$ 存在 transferred 模型结构，弱等价与 fibration 逐 arity 检测。

**说明 14.14.** “适当条件”不能删除。原因是自由 operad $\mathbb F(X)$ 由 $X$-装饰树构造。若沿一个生成 trivial cofibration 附加 generator，则在自由 operad 中会产生所有树形复合。要证明所得 operad morphism 仍为弱等价，需要证明这些树形胞腔附加被 $\otimes$、coinvariants、pushout 和 transfinite composition 保持。一般对称幺半模型范畴未必满足这些性质。

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

**外部输入定理 14.21（代数范畴的 admissibility）.** 设 $\mathcal M$ 是足够良好的 cofibrantly generated 对称幺半模型范畴。若 $\mathcal O$ 满足相应的 $\Sigma$-cofibrancy 或更一般的等变平坦性条件，则 $\mathcal O$ admissible。

更强的现代形式给出条件，使得所有 small colored symmetric operads 都 admissible。具体条件包括对称版本的 h-monoidality、flatness、smallness 和 tractability；本书只把它们作为外部输入，不在此章展开证明。

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

**外部输入定理 14.26（rectification criterion）.** 在适当的对称幺半模型范畴中，若
$$
\varphi:\mathcal O\to\mathcal P
$$
是 admissible operads 之间的 entrywise weak equivalence，并且源 operad 满足足够的 cofibrancy 或 flatness 条件，则 Quillen adjunction
$$
\varphi_!:\operatorname{Alg}_{\mathcal O}(\mathcal M)\rightleftarrows\operatorname{Alg}_{\mathcal P}(\mathcal M):\varphi^\*
$$
是 Quillen equivalence。

**说明 14.27.** Rectification 不是自动的。若 $\mathcal O\to\mathcal P$ 是逐 arity 弱等价，但 $\mathcal O$ 不够 cofibrant，或 $\mathcal M$ 中对称幂不保持弱等价，则代数范畴可能不 Quillen equivalent。

**推论 14.28.** 若 $A_\infty\to\operatorname{Ass}$ 是 dg-operads 的 cofibrant resolution，且所在模型范畴满足 rectification criterion，则 $A_\infty$-algebras 的同伦理论与 dg associative algebras 的同伦理论 Quillen equivalent。

**证明.** 由 cofibrant resolution，$A_\infty\to\operatorname{Ass}$ 是 entrywise weak equivalence。由外部输入定理 14.26，诱导的 extension-restriction adjunction 是 Quillen equivalence。$\square$

**警告 14.29.** 对 $E_\infty\to\operatorname{Com}$ 不能无条件推出同样结论。在特征 $0$ 链复形中通常可 rectification；在正特征或一般底环上，严格 commutative dg algebra 与 $E_\infty$-algebra 的同伦理论通常不同。

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

**外部输入定理 14.31（Boardman-Vogt resolution）.** 在满足适当区间对象、幺半和模型范畴条件的 $\mathcal M$ 中，存在 functorial resolution
$$
W\mathcal O\to\mathcal O
$$
使得 $W\mathcal O$ 在合适意义下 cofibrant，并且该态射为 weak equivalence。

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
