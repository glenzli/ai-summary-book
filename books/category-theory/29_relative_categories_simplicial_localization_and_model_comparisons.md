# 第二十九章：相对范畴、单纯局部化与模型比较

## 本章目标

本章补齐从模型范畴到 $\infty$-范畴的模型无关路径。相对范畴 $(\mathcal C,W)$ 只指定哪些态射应成为等价；Dwyer-Kan 单纯局部化把它提升为带映射空间的单纯范畴；complete Segal spaces 和 quasi-categories 则给出等价的高阶范畴模型。本章的核心目标是说明：不同模型不是不同数学对象，而是同一同伦范畴论的不同坐标。

## 依赖前置知识

需要模型范畴、单纯集、Kan 复形、quasi-category、映射空间、同伦范畴、presentable $\infty$-范畴和局部化的泛性质。

## 29.1 相对范畴与 $\infty$-局部化

**定义 29.1.** 一个 relative category 是一对 $(\mathcal C,W)$，其中 $\mathcal C$ 是普通范畴，$W\subseteq\mathcal C$ 是含所有对象和恒等态射的宽子范畴。$W$ 中态射称为 weak equivalences。

**定义 29.2.** $(\mathcal C,W)$ 的 $\infty$-categorical localization 是 $\infty$-范畴 $\mathcal C[W^{-1}]$ 连同函子

$$
\ell:N(\mathcal C)\to\mathcal C[W^{-1}]
$$

使得 $\ell(W)$ 中态射成为等价，并且对任意 $\infty$-范畴 $D$，预复合诱导全忠实嵌入

$$
\operatorname{Fun}(\mathcal C[W^{-1}],D)\hookrightarrow\operatorname{Fun}(N\mathcal C,D)
$$

其本质像为那些把 $W$ 送为等价的函子。

**命题 29.3.** 若 $\mathcal C[W^{-1}]$ 存在，则它在等价意义下唯一。

**证明.** 若 $L$ 和 $L'$ 都满足定义 29.2，则取 $D=L'$，由 $L$ 的泛性质得到函子 $L\to L'$；取 $D=L$，由 $L'$ 的泛性质得到函子 $L'\to L$。两个复合在预复合到 $N\mathcal C$ 后都等价于恒等函子。因泛性质给出的映射空间全忠实，两个复合等价于恒等。因此 $L\simeq L'$。$\square$

**定义 29.4.** $W$ 称为 saturated，若态射 $f$ 属于 $W$ 当且仅当它在 ordinary localization $\mathcal C[W^{-1}]_{\operatorname{ord}}$ 中成为同构。

**命题 29.5.** 在 $\infty$-categorical localization 中成为等价的原态射构成 saturated weak equivalences。

**证明.** 设 $W'$ 为 $\mathcal C$ 中在 $\mathcal C[W^{-1}]$ 中成为等价的态射类。它含 $W$，并对 $2$-out-of-$3$、retract 和同构闭合，因为等价态射在任意 $\infty$-范畴中满足这些性质。

把普通范畴 $A$ 看作 nerve $N(A)$。定义 29.2 对所有 $N(A)$ 应用后说明，函子

$$
h(\mathcal C[W^{-1}])\to A
$$

等价于把 $W$ 送到同构的函子 $\mathcal C\to A$。因此 $h(\mathcal C[W^{-1}])$ 满足 ordinary localization 的泛性质。若 $f$ 在 ordinary localization 中成为同构，则它在 $h(\mathcal C[W^{-1}])$ 中成为同构；而 $\infty$-范畴中态射成为等价当且仅当其在同伦范畴中成为同构，故 $f\in W'$。反向显然。因此 $W'$ saturated。$\square$

## 29.2 单纯范畴与 Dwyer-Kan 等价

**定义 29.6.** 一个 simplicial category $\mathcal A$ 是 $\mathbf{sSet}$-富范畴。其对象为集合，Hom 对象为单纯集

$$
\operatorname{Map}_{\mathcal A}(x,y),
$$

复合为单纯集映射

$$
\operatorname{Map}_{\mathcal A}(y,z)\times\operatorname{Map}_{\mathcal A}(x,y)\to\operatorname{Map}_{\mathcal A}(x,z).
$$

**定义 29.7.** 若 $\mathcal A$ 是 simplicial category，其同伦范畴 $\pi_0\mathcal A$ 有同一对象，并定义

$$
\pi_0\mathcal A(x,y)=\pi_0\operatorname{Map}_{\mathcal A}(x,y).
$$

**定义 29.8.** simplicial functor $F:\mathcal A\to\mathcal B$ 称为 Dwyer-Kan equivalence，若：

1. 对所有 $x,y$，映射单纯集
   $$
   \operatorname{Map}_{\mathcal A}(x,y)\to\operatorname{Map}_{\mathcal B}(Fx,Fy)
   $$
   是弱同伦等价；
2. $\pi_0F:\pi_0\mathcal A\to\pi_0\mathcal B$ 本质满。

**命题 29.9.** Dwyer-Kan equivalence 诱导同伦范畴等价。

**证明.** 第一条件给出 $\pi_0$ 上 Hom 集同构，所以 $\pi_0F$ 完全忠实。第二条件给出本质满。因此 $\pi_0F$ 是范畴等价。$\square$

**外部输入定理 29.10（Dwyer-Kan 单纯局部化）.** 对任意 relative category $(\mathcal C,W)$，存在 simplicial category $L(\mathcal C,W)$，称为 simplicial localization，使得其 coherent nerve 或等价模型实现 $\mathcal C[W^{-1}]$ 的 $\infty$-categorical localization。Hammock localization 给出显式模型。

## 29.3 模型范畴产生的 $\infty$-范畴

**定义 29.11.** 若 $\mathcal M$ 是模型范畴，记 $W_{\mathcal M}$ 为弱等价子范畴。其 underlying $\infty$-category 定义为

$$
\mathcal M_\infty=\mathcal M[W_{\mathcal M}^{-1}].
$$

**外部输入定理 29.12.** 若 $\mathcal M$ 是 simplicial model category，且 $x$ cofibrant、$y$ fibrant，则 $\mathcal M_\infty$ 中的映射空间等价于导出单纯映射空间

$$
\operatorname{Map}_{\mathcal M_\infty}(x,y)\simeq\operatorname{Map}_{\mathcal M}(x,y).
$$

一般模型范畴可用 cosimplicial/simplicial resolutions 或 hammock localization 计算导出映射空间。

**命题 29.13.** Quillen 等价诱导 underlying $\infty$-categories 的等价。

**证明.** 设 $F:\mathcal M\rightleftarrows\mathcal N:G$ 是 Quillen 等价。其总左导出和总右导出给出同伦范畴等价。更强地，Quillen 等价诱导 hammock localizations 之间的 Dwyer-Kan equivalence：映射空间由 cofibrant-fibrant replacements 计算，而 Quillen 等价保持并反映这些导出映射空间的弱等价。由定理 29.10，coherent nerve 后得到 $\mathcal M_\infty\simeq\mathcal N_\infty$。$\square$

**注 29.14.** 命题 29.13 的关键不是同伦范畴等价本身，而是映射空间也被正确比较。因此普通三角范畴等价通常不足以判定稳定 $\infty$-范畴等价。

## 29.4 Coherent nerve 与 simplicial categories

**外部输入定理 29.15.** 存在 coherent nerve 函子

$$
N_{\operatorname{hc}}:\mathbf{sCat}\to\mathbf{sSet}
$$

它把映射对象为 Kan 复形的 simplicial category 送到 quasi-category。其左伴随可看作把单纯集自由生成同伦相干单纯范畴。

**命题 29.16.** 若 $\mathcal A$ 的 Hom 单纯集全为离散单纯集，则 $N_{\operatorname{hc}}(\mathcal A)$ 与普通 nerve $N(\pi_0\mathcal A)$ 等价。

**证明.** Hom 单纯集离散时，$\mathcal A$ 没有非平凡高维映射同伦。Coherent nerve 的 $n$-单纯形是从同伦相干 $[n]$ 形状到 $\mathcal A$ 的 simplicial functor；所有高维相干数据因 Hom 离散而唯一退化。因此只剩普通可复合箭头串，即普通 nerve 的 $n$-单纯形。逐维识别给出等价。$\square$

**外部输入定理 29.17（Bergner-Joyal 比较）.** Bergner 模型结构下的 simplicial categories 与 Joyal 模型结构下的 simplicial sets 由 Quillen 等价相连。弱等价分别是 Dwyer-Kan equivalences 与 categorical equivalences。

## 29.5 Complete Segal spaces

**定义 29.18.** 一个 simplicial space 是函子

$$
X:\Delta^{op}\to\mathcal S.
$$

它满足 Segal 条件，若对 $n\ge2$，自然映射

$$
X_n\to X_1\times_{X_0}\cdots\times_{X_0}X_1
$$

为等价。

**定义 29.19.** Segal space $X$ 的等价空间 $X_{\operatorname{eq}}\subseteq X_1$ 由那些在同伦范畴中成为同构的 $1$-单纯形组成。若退化映射

$$
X_0\to X_{\operatorname{eq}}
$$

为等价，则称 $X$ complete。Complete Segal space 简写为 CSS。

**命题 29.20.** Segal 条件使 $X_0$ 成为对象空间，$X_1$ 成为态射空间，并用 $X_2$ 编码复合。

**证明.** 当 $n=2$ 时，Segal 等价

$$
X_2\simeq X_1\times_{X_0}X_1
$$

说明一个 $2$-单纯形等价于一对可复合 $1$-单纯形加上复合数据。更高 $n$ 的 Segal 等价说明 $n$-单纯形由 $n$ 条可复合边控制。由于这些是空间而非集合，复合不是严格函数，而是在可缩选择空间中给出的同伦相干复合。$\square$

**外部输入定理 29.21（Rezk nerve 与 CSS 模型）.** Relative categories 有 Rezk nerve，取值于 complete Segal spaces；Rezk CSS 模型结构、Joyal quasi-category 模型结构、Bergner simplicial category 模型结构和相对范畴的合适模型结构之间存在 Quillen 等价链。

## 29.6 模型选择原则

**命题 29.22.** 若一个构造只依赖 $\infty$-categorical localization 的泛性质，则它在 relative categories、simplicial categories、quasi-categories 和 complete Segal spaces 的模型之间不变。

**证明.** 这些模型之间的 Quillen 等价保持并反映相应的弱等价对象。由定理 29.17 和 29.21，它们表示同一 homotopy theory of homotopy theories。若构造由 $\mathcal C[W^{-1}]$ 的泛性质刻画，则任意模型替换给出的对象满足同一泛性质；由命题 29.3 的唯一性，所得结果等价。$\square$

**例子 29.23.** Presentable $\infty$-categories 可用 combinatorial model categories 的左 Bousfield localizations 建模，也可直接用 accessible localizations of presheaf $\infty$-categories 建模。具体模型不同，左伴随、局部对象、映射空间和 Bousfield localization 的 $\infty$-范畴意义相同。

## 29.7 局部化的形式后果

**命题 29.24（相对函子的导出）.** 设

$$
F:(\mathcal C,W)\to(\mathcal D,V)
$$

是 relative functor，即 $F(W)\subseteq V$。若两个 $\infty$-categorical localizations 存在，则存在本质唯一的函子

$$
\overline F:\mathcal C[W^{-1}]\to\mathcal D[V^{-1}]
$$

使得图

$$
N\mathcal C\to\mathcal C[W^{-1}]\xrightarrow{\overline F}\mathcal D[V^{-1}]
$$

与

$$
N\mathcal C\xrightarrow{NF}N\mathcal D\to\mathcal D[V^{-1}]
$$

等价交换。该构造与复合相容，唯一性理解为选择空间可缩。

**证明.** 复合

$$
N\mathcal C\xrightarrow{NF}N\mathcal D\to\mathcal D[V^{-1}]
$$

把 $W$ 中态射送为等价，因为 $F(W)\subseteq V$。由 $\mathcal C[W^{-1}]$ 的泛性质，它唯一因子化经 $\mathcal C[W^{-1}]$，得到 $\overline F$。若有两个这样的因子化，则它们预复合到 $N\mathcal C$ 后等价；定义 29.2 中的全忠实性说明二者本身等价。对复合 $G\circ F$，直接导出与先导出 $F$ 再导出 $G$ 的两个函子预复合到 $N\mathcal C$ 后相同，故仍由全忠实性得到相容。$\square$

**命题 29.25（只倒置同构时无变化）.** 若 $W$ 正是 $\mathcal C$ 中所有同构，则

$$
N\mathcal C\to\mathcal C[W^{-1}]
$$

是 $\infty$-范畴等价。

**证明.** 对任意 $\infty$-范畴 $D$，任意函子 $N\mathcal C\to D$ 自动把 $\mathcal C$ 中同构送为 $D$ 中等价。因此定义 29.2 中本质像条件没有限制，预复合给出

$$
\operatorname{Fun}(\mathcal C[W^{-1}],D)\simeq\operatorname{Fun}(N\mathcal C,D)
$$

对所有 $D$ 成立。由 $\infty$-范畴的 Yoneda 判别，$\mathcal C[W^{-1}]\simeq N\mathcal C$。$\square$

**命题 29.26.** 在映射空间为 Kan 复形的 simplicial categories 之间，Dwyer-Kan equivalences 满足 $2$-out-of-$3$。

**证明.** 设 $\mathcal A\xrightarrow{F}\mathcal B\xrightarrow{G}\mathcal C$ 为 simplicial functors。若 $F$ 和 $G$ 都是 Dwyer-Kan equivalences，则 $GF$ 显然也是，因为映射空间弱等价和同伦范畴等价都对复合封闭。

若 $F$ 和 $GF$ 是 Dwyer-Kan equivalences，则 $\pi_0F$ 和 $\pi_0(GF)$ 是范畴等价，故 $\pi_0G$ 也是范畴等价。需证 $G$ 在所有映射空间上为弱等价。任取 $b,b'\in\mathcal B$。由 $\pi_0F$ 本质满，存在 $x,x'\in\mathcal A$ 以及 $\pi_0\mathcal B$ 中的同构 $Fx\simeq b$、$Fx'\simeq b'$。在映射空间为 Kan 复形的 simplicial category 中，与等价对象前后复合诱导映射空间弱等价，因此有弱等价

$$
\operatorname{Map}_{\mathcal B}(b,b')\simeq
\operatorname{Map}_{\mathcal B}(Fx,Fx')
$$

和

$$
\operatorname{Map}_{\mathcal C}(Gb,Gb')\simeq
\operatorname{Map}_{\mathcal C}(GFx,GFx').
$$

而

$$
\operatorname{Map}_{\mathcal A}(x,x')\to
\operatorname{Map}_{\mathcal B}(Fx,Fx')\to
\operatorname{Map}_{\mathcal C}(GFx,GFx')
$$

中第一箭头和复合箭头均为弱等价，故第二箭头为弱等价。于是 $G$ 在任意映射空间上为弱等价。

若 $G$ 和 $GF$ 是 Dwyer-Kan equivalences，则 $\pi_0G$ 和 $\pi_0(GF)$ 是范畴等价，故 $\pi_0F$ 是范畴等价。对任意 $x,y\in\mathcal A$，映射空间三角

$$
\operatorname{Map}_{\mathcal A}(x,y)\to
\operatorname{Map}_{\mathcal B}(Fx,Fy)\to
\operatorname{Map}_{\mathcal C}(GFx,GFy)
$$

中第二箭头和复合箭头均为弱等价，故第一箭头为弱等价。于是 $F$ 也是 Dwyer-Kan equivalence。三种情形合并即得 $2$-out-of-$3$。$\square$

## 29.8 本章小结

相对范畴提供最小输入：一个范畴和一类将被倒置的态射。Dwyer-Kan 局部化把它变成映射空间丰富的 simplicial category。Coherent nerve、quasi-category、complete Segal space 和 simplicial category 模型通过 Quillen 等价比较。模型范畴只是产生 $\infty$-范畴的一种方式；真正不变量是由泛性质确定的 $\infty$-categorical localization。

## 练习

**练习 29.1.** 定义 relative category。

**练习 29.2.** 写出 $\infty$-categorical localization 的泛性质。

**练习 29.3.** 证明满足泛性质的 localization 唯一。

**练习 29.4.** 定义 saturated weak equivalences。

**练习 29.5.** 定义 simplicial category。

**练习 29.6.** 定义 Dwyer-Kan equivalence。

**练习 29.7.** 证明 Dwyer-Kan equivalence 诱导同伦范畴等价。

**练习 29.8.** 对模型范畴 $\mathcal M$ 定义 underlying $\infty$-category。

**练习 29.9.** 解释为什么 Quillen 等价应比较映射空间而不只是同伦范畴。

**练习 29.10.** 陈述 coherent nerve 的作用。

**练习 29.11.** 说明 Hom 离散的 simplicial category 如何退化为普通 nerve。

**练习 29.12.** 定义 simplicial space 的 Segal 条件。

**练习 29.13.** 定义 complete Segal space。

**练习 29.14.** 解释 Rezk nerve 的用途。

**练习 29.15.** 说明为什么由 localization 泛性质刻画的构造与模型选择无关。

**练习 29.16.** 证明 relative functor 诱导 localizations 之间的本质唯一函子。

**练习 29.17.** 证明若只倒置普通同构，则 $\infty$-localization 等价于普通 nerve。

**练习 29.18.** 证明 Dwyer-Kan equivalences 满足 $2$-out-of-$3$。
