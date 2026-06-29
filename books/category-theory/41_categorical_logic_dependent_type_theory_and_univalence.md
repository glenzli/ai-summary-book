# 第四十一章：范畴逻辑、依赖类型论与 Univalence

## 本章目标

本章把范畴论作为逻辑语义的语言来组织。普通极限、指数对象、slice 范畴、fibration 和 $\infty$-topos 分别解释命题逻辑、一阶逻辑、依赖类型论和同伦类型论。核心原则是：语法中的上下文、类型、项和替换，分别对应范畴中的对象、slice 对象、section 和 pullback。

## 依赖前置知识

需要有限极限、Cartesian closed categories、slice categories、fibrations、topos、$\infty$-topos、Cartesian fibration、可表现 $\infty$-范畴和基本同伦论。

## 41.1 子对象纤维化与谓词

**定义 41.1.** 设 $\mathcal C$ 有有限极限。对象 $X$ 上的子对象偏序 $\operatorname{Sub}_{\mathcal C}(X)$ 是所有单态 $U\hookrightarrow X$ 的同构类，按因子化排序。

**定义 41.2.** 子对象赋值

$$
\operatorname{Sub}_{\mathcal C}:\mathcal C^{op}\to\mathbf{Pos}
$$

把 $f:X\to Y$ 送到 pullback 函子 $f^*:\operatorname{Sub}(Y)\to\operatorname{Sub}(X)$。它称为 $\mathcal C$ 的谓词纤维化。

**命题 41.3.** 若 $\mathcal C$ 有有限极限，则 $\operatorname{Sub}_{\mathcal C}$ 是反变函子。

**证明.** 恒等态射的 pullback 同构于原单态，所以 $(\operatorname{id}_X)^*=\operatorname{id}_{\operatorname{Sub}(X)}$。若 $X\xrightarrow fY\xrightarrow gZ$，则沿 $gf$ 拉回子对象 $U\hookrightarrow Z$ 得到的方块与先沿 $g$ 再沿 $f$ 拉回得到的迭代 pullback 由 pullback 的粘合性质同构。因此 $(gf)^*=f^*g^*$。在子对象同构类上这给出严格函子性。$\square$

**定义 41.4.** 一个 regular category 是有有限极限、每个态射有 image factorization，且 regular epimorphisms 在 pullback 下稳定的范畴。

**命题 41.5.** 在 regular category 中，任意 $f:X\to Y$ 的拉回函子

$$
f^*:\operatorname{Sub}(Y)\to\operatorname{Sub}(X)
$$

有左伴随 $\exists_f$，由 image of composite 给出。

**证明.** 对子对象 $m:U\hookrightarrow X$，定义 $\exists_f(U)$ 为复合 $U\hookrightarrow X\xrightarrow fY$ 的 image。若 $V\hookrightarrow Y$ 为子对象，则

$$
\exists_f(U)\le V
$$

当且仅当 $U\to Y$ 经 $V$ 因子化。由 pullback 的泛性质，这等价于 $U\hookrightarrow X$ 经 $f^*V\hookrightarrow X$ 因子化，即

$$
U\le f^*V.
$$

这正是偏序伴随 $\exists_f\dashv f^*$。$\square$

## 41.2 Cartesian closed 语义与直觉命题逻辑

**定义 41.6.** Cartesian closed category 是有有限积且对每个 $A$，函子 $-\times A$ 有右伴随 $(-)^A$ 的范畴。

**定义 41.7.** Heyting category 是有有限极限的范畴 $\mathcal C$，使每个 $\operatorname{Sub}(X)$ 为 Heyting algebra，并且 pullback 保持有限 meets 与 Heyting implication 的相应结构。

**命题 41.8.** 在 Heyting category 中，$X$ 上谓词的合取由 pullback 给出，蕴含 $P\Rightarrow Q$ 是满足

$$
R\le(P\Rightarrow Q)\quad\Longleftrightarrow\quad R\wedge P\le Q
$$

的右伴随。

**证明.** 子对象偏序中的 meet 由两个单态 $P,Q\hookrightarrow X$ 在 $X$ 上的 pullback 给出。Heyting implication 按定义是 $(-)\wedge P$ 的右伴随，因此满足所列自然双条件。$\square$

**外部输入定理 41.9.** Elementary topos 的每个 slice $\mathcal E/X$ 是 Cartesian closed，且每个 $\operatorname{Sub}_{\mathcal E}(X)$ 是 Heyting algebra。因此 elementary topos 给出高阶直觉逻辑的语义。

## 41.3 局部 Cartesian closed 范畴与依赖类型

**定义 41.10.** 范畴 $\mathcal C$ 称为 locally Cartesian closed category，若 $\mathcal C$ 有有限极限，且每个 slice $\mathcal C/X$ 都是 Cartesian closed。

**定义 41.11.** 给定 $f:X\to Y$，替换函子为 pullback

$$
f^*:\mathcal C/Y\to\mathcal C/X.
$$

其左伴随若存在，记作 $\Sigma_f$；其右伴随若存在，记作 $\Pi_f$。在类型论语义中，$\Sigma_f$ 解释依赖和类型，$\Pi_f$ 解释依赖函数类型。

**命题 41.12.** 对任意有 pullback 的范畴，$f^*:\mathcal C/Y\to\mathcal C/X$ 有左伴随 $\Sigma_f$，其中 $\Sigma_f(U\to X)$ 为复合 $U\to X\xrightarrow fY$。

**证明.** 对 $p:U\to X$ 与 $q:V\to Y$，slice 中态射 $\Sigma_f(p)\to q$ 是交换三角

$$
U\to V\to Y
$$

且复合等于 $fp$。另一方面，态射 $p\to f^*q$ 是从 $U$ 到 pullback $X\times_YV$ 的态射，使到 $X$ 的分量为 $p$。由 pullback 泛性质，这等价于给出 $U\to V$ 且 $q(U\to V)=fp$。于是

$$
\operatorname{Hom}_{\mathcal C/Y}(\Sigma_f p,q)\cong
\operatorname{Hom}_{\mathcal C/X}(p,f^*q),
$$

自然于 $p,q$。$\square$

**命题 41.13.** 若 $\mathcal C$ locally Cartesian closed，则每个 $f^*$ 有右伴随 $\Pi_f$。

**证明.** 这是 locally Cartesian closed 的等价定义之一：每个 slice Cartesian closed 等价于每个 pullback functor between slices 有 dependent product 右伴随。采用该定义时结论直接成立。采用 slice 指数对象定义时，$\Pi_f$ 由相应 slice 中的内部 Hom 构造，并由指数对象的伴随性给出 $f^*\dashv\Pi_f$。$\square$

## 41.4 Comprehension categories

**定义 41.14.** 一个 comprehension category 由 fibration $p:\mathcal T\to\mathcal C$ 和函子

$$
\chi:\mathcal T\to\mathcal C^{\to}
$$

组成，满足 $\chi(A)$ 的余定义域为 $p(A)$，并且 Cartesian lift 与 pullback 方块相容。对象 $\Gamma\in\mathcal C$ 解释上下文；纤维 $\mathcal T_\Gamma$ 的对象解释 $\Gamma$ 中的类型；$\chi(A):\Gamma.A\to\Gamma$ 解释上下文扩张。

**定义 41.15.** 给定 $A\in\mathcal T_\Gamma$，$A$ 的项是 $\mathcal C$ 中 section

$$
s:\Gamma\to\Gamma.A
$$

满足 $\chi(A)s=\operatorname{id}_\Gamma$。

**命题 41.16.** comprehension category 中替换沿 $f:\Delta\to\Gamma$ 由 Cartesian lift 给出，并与上下文扩张的 pullback 相容。

**证明.** Fibration 的 Cartesian lift 给出 $f^*A\in\mathcal T_\Delta$ 和 Cartesian 态射 $f^*A\to A$。Comprehension 函子 $\chi$ 按定义把该 Cartesian 态射送到 pullback 方块

$$
\begin{array}{ccc}
\Delta.f^*A&\to&\Gamma.A\\
\downarrow&&\downarrow\\
\Delta&\xrightarrow f&\Gamma .
\end{array}
$$

因此替换后的上下文扩张正是原扩张沿 $f$ 的拉回。$\square$

## 41.5 恒等类型与路径对象

**定义 41.17.** 在有有限极限的范畴中，显示映射 $p:E\to B$ 的相对恒等类型可由对角线

$$
E\to E\times_BE
$$

的一个因子化

$$
E\to \operatorname{Id}_p(E)\to E\times_BE
$$

表示，其中第二个态射再次属于指定显示映射类。

**外部输入定理 41.18.** 带合适 weak factorization system 的 comprehension category 可解释 Martin-Lof identity types；若还满足稳定性和 Beck-Chevalley 条件，则解释替换下稳定的恒等类型。

**命题 41.19.** 在 groupoid 范畴语义中，对象的恒等类型由同构集给出。

**证明.** 设 $G$ 为 groupoid，两个项 $x,y\in G$ 的相等证据不应只是命题性真值，而应记录从 $x$ 到 $y$ 的可逆箭头。Groupoid 中所有箭头可逆，所以路径对象可取箭头对象 $G_1$，源靶映射给出 $G_1\to G_0\times G_0$，恒等箭头给出对角线的因子化 $G_0\to G_1\to G_0\times G_0$。纤维正是 $\operatorname{Iso}_G(x,y)=\operatorname{Hom}_G(x,y)$。$\square$

## 41.6 Univalence 与 $\infty$-topos 语义

**定义 41.20.** 在同伦语义中，universe 是一个 fibration $p:\mathcal U_\bullet\to\mathcal U$，其纤维分类一类小类型。它称为 univalent，若对任意 $A,B:\Gamma\to\mathcal U$，恒等类型

$$
\operatorname{Id}_{\mathcal U}(A,B)
$$

与等价类型 $\operatorname{Equiv}_\Gamma(A,B)$ 等价。

**外部输入定理 41.21.** Kan simplicial sets、合适模型范畴和足够好的 $\infty$-topos 中存在 univalent universes，因而给出 homotopy type theory 的模型。

**命题 41.22.** Univalence 蕴含等价对象可按相等对象替换。

**证明.** 设 $e:A\simeq B$ 为同一 universe 中两个类型的等价。Univalence 给出等价

$$
\operatorname{Id}_{\mathcal U}(A,B)\simeq \operatorname{Equiv}(A,B),
$$

故 $e$ 对应一个路径 $p:A=B$。类型论的消去规则允许沿等式路径运输任意依赖于 universe 元素的构造，因此依赖于 $A$ 的结构可沿 $p$ 运输到依赖于 $B$ 的结构。$\square$

## 41.7 几何逻辑与逆像函子

**定义 41.23.** 几何公式由有限合取、任意析取和存在量词生成。几何理论由几何 sequents 组成。

**命题 41.24.** 几何态射 $f:\mathcal E\to\mathcal F$ 的逆像函子 $f^*$ 保持几何逻辑的解释。

**证明.** 几何态射的逆像 $f^*$ 按定义保持有限极限，且作为左伴随保持所有小余极限。有限合取由有限极限解释，任意析取由相应子对象并或余极限解释，存在量词由 image 或相应左伴随解释。由于 $f^*$ 保持这些结构，并与 pullback 相容，几何公式和几何 sequent 的解释在 $f^*$ 下保持。$\square$

## 41.8 本章小结

范畴逻辑把语法中的结构逐层翻译为范畴结构：子对象解释谓词，regular image 解释存在量词，Heyting 结构解释直觉逻辑，locally Cartesian closed 结构解释依赖和类型与依赖函数类型，comprehension fibration 解释上下文与类型族，路径对象解释恒等类型，univalent universe 把等价与相等联系起来。由此，范畴论不仅描述数学对象，也提供形式语言和证明系统的语义基础。

## 练习

**练习 41.1.** 定义子对象偏序 $\operatorname{Sub}_{\mathcal C}(X)$。

**练习 41.2.** 证明子对象赋值关于态射反变。

**练习 41.3.** 定义 regular category。

**练习 41.4.** 在 regular category 中构造 $\exists_f\dashv f^*$。

**练习 41.5.** 定义 Cartesian closed category。

**练习 41.6.** 说明 Heyting implication 的伴随刻画。

**练习 41.7.** 定义 locally Cartesian closed category。

**练习 41.8.** 证明 $\Sigma_f\dashv f^*$。

**练习 41.9.** 解释 $\Pi_f$ 如何对应依赖函数类型。

**练习 41.10.** 定义 comprehension category。

**练习 41.11.** 说明项为何是上下文扩张的 section。

**练习 41.12.** 用 groupoid 解释恒等类型。

**练习 41.13.** 定义 univalent universe。

**练习 41.14.** 证明几何态射逆像保持几何逻辑解释。
