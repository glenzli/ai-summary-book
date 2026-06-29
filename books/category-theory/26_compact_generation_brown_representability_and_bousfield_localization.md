# 第二十六章：紧生成、Brown 表示性与 Bousfield 局部化

## 本章目标

本章补充稳定 presentable $\infty$-范畴中的紧生成理论。紧生成把“大”稳定范畴压缩为一小族紧对象控制；Brown 表示性给出伴随和表示对象的存在；Bousfield 局部化和 Verdier quotient 则描述同伦论中“强制某些态射成为等价”的过程。

## 依赖前置知识

需要可表现 $\infty$-范畴、稳定 $\infty$-范畴、映射谱、t-结构、Bousfield localization 和普通三角范畴的基本语言。

## 26.1 紧对象与紧生成

**定义 26.1.** 设 $C$ 为有小余积的 $\infty$-范畴。对象 $K\in C$ 称为 compact，若映射空间函子

$$
\operatorname{Map}_C(K,-):C\to\mathcal S
$$

保持滤过余极限。若 $C$ 稳定，也常要求映射谱函子

$$
\operatorname{Map}^{\operatorname{Sp}}_C(K,-)
$$

保持滤过余极限；在稳定 presentable 语境下这与上式相容。

**定义 26.2.** 稳定 presentable $\infty$-范畴 $C$ 称为 compactly generated，若存在一小集 compact objects $\mathcal G$，使得 $X\simeq0$ 当且仅当对所有 $G\in\mathcal G$ 和所有 $n\in\mathbb Z$，

$$
\pi_0\operatorname{Map}_C(\Sigma^nG,X)=0.
$$

等价地，$\mathcal G$ 生成的最小稳定、闭合于小余积的全子范畴是 $C$。

**例子 26.3.** 谱范畴 $\mathbf{Sp}$ 由 sphere spectrum $\mathbb S$ 紧生成。环谱 $R$ 的模范畴 $\operatorname{Mod}_R$ 由自由模 $R$ 紧生成。环 $A$ 的导出 $\infty$-范畴 $D(A)$ 由 $A$ 作为复形集中在 $0$ 次的对象紧生成。

**命题 26.4.** 若 $C$ 由 compact objects 集合 $\mathcal G$ 生成，且 $F:C\to D$ 是保持小余积的正合函子，则 $F$ 保守当且仅当 $F$ 在 $\mathcal G$ 生成的检测族上检测零对象。

**证明.** 若 $F$ 保守，则显然检测零对象。反过来设 $F(X)\simeq0$。对所有 $G\in\mathcal G$ 和 $n\in\mathbb Z$，若 $F$ 在生成检测族上反映零，则由 $F(X)=0$ 推出所有

$$
\pi_0\operatorname{Map}_C(\Sigma^nG,X)
$$

为零。由于 $\mathcal G$ 生成 $C$，定义 26.2 给出 $X\simeq0$。对一般态射 $u:X\to Y$，若 $F(u)$ 为等价，则 $F(\operatorname{fib}u)\simeq0$；由刚证的零对象反映性，$\operatorname{fib}u\simeq0$，故 $u$ 为等价。$\square$

## 26.2 Localizing subcategories 与 Verdier quotient

**定义 26.5.** 稳定 $\infty$-范畴 $C$ 的全子范畴 $L\subseteq C$ 称为 localizing subcategory，若它稳定，且对 $C$ 中所有小余积封闭。

**定义 26.6.** 若 $L\subseteq C$ 是稳定全子范畴，Verdier quotient 或稳定商

$$
C/L
$$

是带正合函子 $q:C\to C/L$ 的稳定 $\infty$-范畴，满足 $q$ 把 $L$ 中对象送为零，并且对任意稳定 $\infty$-范畴 $D$，预复合 $q$ 给出全忠实嵌入

$$
\operatorname{Fun}^{ex}(C/L,D)\hookrightarrow\operatorname{Fun}^{ex}(C,D)
$$

其像为那些把 $L$ 送为零的正合函子。

**外部输入定理 26.7.** 若 $C$ 是稳定 presentable $\infty$-范畴，$L\subseteq C$ 是由一小集对象生成的 localizing subcategory，则 Verdier quotient $C/L$ 存在且 presentable，商函子 $C\to C/L$ 是 accessible exact localization。

**例子 26.8.** 在导出范畴中，把 acyclic complexes 作为 localizing subcategory 商掉，可从同伦范畴层面得到 derived category 的 Verdier quotient。稳定 $\infty$-范畴口径下，这一过程由 localization of chain complexes at quasi-isomorphisms 提升为保留映射空间的稳定局部化。

## 26.3 Brown 表示性

**定义 26.9.** 设 $C$ 是稳定 $\infty$-范畴。函子

$$
H:C^{op}\to\mathcal S
$$

称为 cohomological，若它把余纤维序列送到纤维序列，并把小余积送到小积。

**外部输入定理 26.10（Brown 表示性）.** 若 $C$ 是 compactly generated stable presentable $\infty$-category，则满足适当集合值或 space 值条件的 cohomological functor

$$
H:C^{op}\to\mathcal S
$$

可由某个对象 $X\in C$ 表示：

$$
H(-)\simeq\operatorname{Map}_C(-,X).
$$

等价形式：保持小余积的正合函子 $F:C\to D$ 在广泛条件下有右伴随。

**命题 26.11.** 若 $F:C\to D$ 是 presentable stable $\infty$-categories 之间保持小余积的正合函子，则 $F$ 有右伴随。

**证明.** 因 $C,D$ presentable，第二十三章的 presentable $\infty$-范畴伴随函子定理说明保持所有小余极限的函子是左伴随。正合函子保持有限余极限，题设保持小余积；在稳定 presentable 范畴中，小余极限由小余积和有限余极限生成。因此 $F$ 保持小余极限，故有右伴随。$\square$

**注 26.12.** Brown 表示性在三角范畴层面常用于证明右伴随存在；presentable $\infty$-范畴层面则由伴随函子定理给出更结构化版本。二者在稳定同伦论中相互解释。

## 26.4 Bousfield localization

**定义 26.13.** 设 $C$ 为稳定 presentable $\infty$-范畴。Bousfield localization 是 exact accessible localization

$$
L:C\to C.
$$

对象 $X$ 称为 $L$-acyclic，若 $LX\simeq0$；称为 $L$-local，若单位 $X\to LX$ 是等价。

**命题 26.14.** $L$-acyclic objects 构成 localizing subcategory，$L$-local objects 构成稳定 presentable 反射子范畴。

**证明.** $L$ 正合，故若 $X\to Y\to Z$ 是余纤维序列，应用 $L$ 后仍为余纤维序列。若其中两个对象被 $L$ 送为零，则第三个也被送为零，因此 acyclic objects 稳定。$L$ 保持小余积，所以 acyclic objects 对小余积封闭，故为 localizing subcategory。局部对象是局部化的本质像，由第二十三章 accessible localization 定理，它们构成 presentable 反射子范畴；正合性保证其稳定。$\square$

**命题 26.15.** 对任意 $X\in C$，存在余纤维序列

$$
A_X\to X\to LX
$$

其中 $A_X$ 为 $L$-acyclic，$LX$ 为 $L$-local。

**证明.** 取单位 $\eta_X:X\to LX$，令 $A_X=\operatorname{fib}(\eta_X)$。应用 $L$ 得到纤维序列

$$
L A_X\to LX\to L^2X.
$$

局部化幂等性给出 $LX\to L^2X$ 为等价，因此 $L A_X\simeq0$。而 $LX$ 按定义是局部对象。稳定范畴中纤维序列也可写为余纤维序列，得到结论。$\square$

**例子 26.16.** 在谱范畴中，给定同调理论 $E$，$E$-localization 把诱导 $E_*$-同构的态射变为等价。$E$-acyclic spectra 是 $E\wedge X\simeq0$ 的谱。常见例子包括有理化、$p$-localization 和 Morava $K$-theory localization。

## 26.5 Smashing localization

**定义 26.17.** 谱范畴或闭幺半稳定 presentable $\infty$-范畴中的 Bousfield localization $L$ 称为 smashing，若存在对象 $E$，使得

$$
LX\simeq E\otimes X
$$

对所有 $X$ 自然成立。谱范畴中写为 $LX\simeq E\wedge X$。

**命题 26.18.** 若 $L$ 是 smashing localization，则 $L$ 保持所有小余极限。

**证明.** 函子 $E\otimes-$ 是左伴随，因此保持所有小余极限。若 $L\simeq E\otimes-$，则 $L$ 也保持所有小余极限。$\square$

**例子 26.19.** 有理化 $X\mapsto H\mathbb Q\wedge X$ 是 smashing localization。某些 chromatic localization 是 smashing，某些则不是；这正是稳定同伦论中 telescope conjecture 等问题的背景。

## 26.6 紧对象与局部化的相互作用

**外部输入定理 26.20.** 设 $C$ 为 compactly generated stable presentable $\infty$-category，$L\subseteq C$ 为由 compact objects 集合生成的 localizing subcategory。则 quotient $C/L$ 仍 compactly generated，并且其 compact objects 由 $C^\omega$ 中 compact objects 的相应 Verdier quotient 幂等完备化给出：

$$
(C/L)^\omega\simeq\operatorname{Kar}(C^\omega/L^\omega).
$$

这是 Neeman-Thomason 型定理的 $\infty$-范畴版本。

## 26.7 局部等价的余纤维判别

**定义 26.21.** 对 exact localization $L:C\to C$，态射 $f:X\to Y$ 称为 $L$-equivalence，若 $Lf$ 是等价。

**命题 26.22.** 在稳定 presentable $\infty$-范畴中，态射 $f:X\to Y$ 是 $L$-equivalence，当且仅当

$$
\operatorname{cofib}(f)
$$

是 $L$-acyclic。

**证明.** 对余纤维序列

$$
X\to Y\to\operatorname{cofib}(f)
$$

应用正合函子 $L$，得到余纤维序列

$$
LX\to LY\to L\operatorname{cofib}(f).
$$

在稳定范畴中，$LX\to LY$ 是等价，当且仅当其余纤维为零。因此 $Lf$ 是等价，当且仅当 $L\operatorname{cofib}(f)\simeq0$，也就是 $\operatorname{cofib}(f)$ 为 $L$-acyclic。$\square$

**推论 26.23.** 若 $A_X\to X\to LX$ 是命题 26.15 的局部化余纤维序列，则 $X\to LX$ 是 $L$-equivalence，且 $A_X$ 是所有阻碍 $X$ 局部性的 acyclic 部分。

**证明.** 由余纤维序列 $A_X\to X\xrightarrow{\eta_X}LX$，态射 $\eta_X$ 的余纤维为 $\Sigma A_X$。因为 $LA_X\simeq0$ 且 $L$ 正合，$L\Sigma A_X\simeq0$。命题 26.22 应用于 $\eta_X$ 给出它是 $L$-equivalence；纤维 $A_X$ 正是被局部化杀掉的部分。$\square$

**例子 26.24.** 若 $L$ 是有理化 $H\mathbb Q\wedge-$，则 $f$ 是有理等价，当且仅当 $\operatorname{cofib}(f)$ 的有理化为零。这个表述只使用稳定范畴的余纤维演算，而不依赖具体谱同伦群计算。

## 26.8 本章小结

紧生成把稳定 presentable $\infty$-范畴的大小控制在小的 compact objects 上。Brown 表示性和 presentable 伴随函子定理保证许多自然函子有伴随。Bousfield localization、Verdier quotient 和 smashing localization 则描述稳定同伦论和导出范畴中最常见的“局部化计算”。

## 练习

**练习 26.1.** 写出 compact object 的定义。

**练习 26.2.** 说明 $\mathbf{Sp}$ 为什么由 $\mathbb S$ 生成。

**练习 26.3.** 在 $D(R)$ 中解释为什么 $R$ 是生成元。

**练习 26.4.** 定义 localizing subcategory。

**练习 26.5.** 写出 Verdier quotient 的泛性质。

**练习 26.6.** 比较三角范畴中的 Verdier quotient 与稳定 $\infty$-范畴商。

**练习 26.7.** 陈述 Brown 表示性的一个形式。

**练习 26.8.** 证明保持小余积的正合函子在 presentable stable 语境下有右伴随。

**练习 26.9.** 对 Bousfield localization $L$，定义 acyclic 与 local objects。

**练习 26.10.** 证明 acyclic objects 对小余积封闭。

**练习 26.11.** 构造 $A_X\to X\to LX$ 的余纤维序列。

**练习 26.12.** 定义 smashing localization。

**练习 26.13.** 说明为什么 smashing localization 保持小余极限。

**练习 26.14.** 解释 Neeman-Thomason 型定理中为什么需要幂等完备化。

**练习 26.15.** 证明命题 26.22 的反向。

**练习 26.16.** 对任意 $X$，说明 $A_X\to X$ 的余纤维为什么是局部对象。

**练习 26.17.** 在有理化例子中，把 $L$-acyclic 条件写成 $H\mathbb Q\wedge X\simeq0$。
