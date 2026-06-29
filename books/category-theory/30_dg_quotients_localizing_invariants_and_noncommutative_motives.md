# 第三十章：dg 商、局部化不变量与非交换 motives

## 本章目标

本章继续第二十七章的导出 Morita 理论，讨论 dg quotient、稳定范畴的 exact sequence、localizing invariants 和 noncommutative motives。核心思想是：许多同调型不变量并不区分 Morita 等价的 dg 范畴，并且把 Verdier quotient 或 dg quotient 送到纤维序列。非交换 motives 是把所有这类不变量统一表示的范畴论对象。

## 依赖前置知识

需要 dg 范畴、stable $\infty$-categories、Verdier quotient、compact generation、Morita equivalence、Hochschild chains、Bousfield localization 和稳定 presentable $\infty$-范畴的基本语言。

## 30.1 稳定范畴的 exact sequence

**定义 30.1.** 设 $A\to B\to C$ 是小的幂等完备稳定 $\infty$-范畴之间的正合函子列。称它为 exact sequence，若：

1. $A\to B$ 全忠实；
2. 复合 $A\to C$ 为零；
3. 诱导函子
   $$
   \operatorname{Kar}(B/A)\to C
   $$
   是稳定 $\infty$-范畴等价。

这里 $B/A$ 是稳定 Verdier quotient，$\operatorname{Kar}$ 表示幂等完备化。

**命题 30.2.** 若 $A\subseteq B$ 是稳定全子范畴，则

$$
A\to B\to\operatorname{Kar}(B/A)
$$

是 exact sequence。

**证明.** 嵌入 $A\to B$ 全忠实。商函子 $B\to B/A$ 按 Verdier quotient 的定义把 $A$ 中对象送为零。定义 30.1 的第三条在此成为恒等识别

$$
\operatorname{Kar}(B/A)\to\operatorname{Kar}(B/A),
$$

所以是等价。$\square$

**定义 30.3.** 小稳定 $\infty$-范畴 $C$ 称为 flasque，若存在正合函子 $T:C\to C$ 使得

$$
\operatorname{id}_C\oplus T\simeq T.
$$

这表示 $C$ 中存在 Eilenberg swindle 型无限吸收。

**命题 30.4.** 若 $E$ 是把有限直和送为直和的稳定范畴值不变量，且 $C$ flasque，则 $E(C)\simeq0$。

**证明.** 由 $\operatorname{id}_C\oplus T\simeq T$，应用 $E$ 得

$$
E(\operatorname{id}_C)+E(T)=E(T)
$$

作为 $\operatorname{End}_{\mathcal D}(E(C))$ 中的等式。这里加号来自稳定范畴中的加性结构，$E(T)$ 表示自函子 $T$ 诱导的自映射。两边相消得 $\operatorname{id}_{E(C)}=0$。稳定范畴中对象恒等态射为零当且仅当对象为零，故 $E(C)\simeq0$。$\square$

## 30.2 dg quotient 与 Drinfeld quotient

**定义 30.5.** 设 $\mathcal A\subseteq\mathcal B$ 是全 dg 子范畴。一个 dg quotient $\mathcal B/\mathcal A$ 是 dg 范畴连同 dg 函子

$$
q:\mathcal B\to\mathcal B/\mathcal A
$$

使 $\mathcal A$ 中对象在导出意义下变为零，并满足相应的 dg Morita 泛性质：任意把 $\mathcal A$ 送为零的 dg 函子从 $\mathcal B$ 唯一因子化经 $\mathcal B/\mathcal A$，唯一性在合适映射空间中理解。

**外部输入定理 30.6（Drinfeld quotient）.** 对全 dg 子范畴 $\mathcal A\subseteq\mathcal B$，存在 dg quotient $\mathcal B/\mathcal A$ 的显式模型：对每个 $a\in\mathcal A$ 添入次数 $-1$ 的闭包收缩元 $\varepsilon_a:a\to a$，满足 $d\varepsilon_a=\operatorname{id}_a$，并按 dg 范畴关系自由扩张。该构造给出同伦意义下的 quotient。

**命题 30.7.** 若 $\mathcal A\subseteq\mathcal B$ 是全 dg 子范畴，则 $H^0(\mathcal B/\mathcal A)$ 中 $\mathcal A$ 的对象成为零对象。

**证明.** 在 Drinfeld quotient 模型中，每个 $a\in\mathcal A$ 有次数 $-1$ 元 $\varepsilon_a:a\to a$ 满足 $d\varepsilon_a=\operatorname{id}_a$。因此 $\operatorname{id}_a$ 在 Hom 复形中为边界，其 $H^0$ 类为零。普通范畴中对象的恒等态射为零意味着该对象是零对象：对任意 $x$，任意态射 $x\to a$ 等于 $\operatorname{id}_a$ 后复合，故为零；任意态射 $a\to x$ 同理。$\square$

**外部输入定理 30.8.** 在合适预三角和幂等完备假设下，

$$
N_{\operatorname{dg}}(\mathcal B/\mathcal A)
$$

给出稳定 $\infty$-范畴 Verdier quotient 的 dg enhancement。

## 30.3 Localizing invariants

**定义 30.9.** 设 $\operatorname{Cat}^{\operatorname{perf}}_\infty$ 为小幂等完备稳定 $\infty$-范畴和正合函子组成的 $\infty$-范畴。一个 additive invariant 是保持滤过余极限且把 split-exact sequences 送为直和分解的函子

$$
E:\operatorname{Cat}^{\operatorname{perf}}_\infty\to\mathcal D,
$$

其中 $\mathcal D$ 是稳定 presentable $\infty$-范畴。

**定义 30.10.** 一个 localizing invariant 是保持滤过余极限且把 exact sequences

$$
A\to B\to C
$$

送为纤维序列

$$
E(A)\to E(B)\to E(C)
$$

的函子。

**命题 30.11.** 每个 localizing invariant 都是 additive invariant。

**证明.** Split-exact sequence 是 exact sequence 的特殊情形，并且在 $B$ 中由一个全忠实子范畴和其补子范畴给出直和分解。Localizing invariant 把它送为纤维序列

$$
E(A)\to E(B)\to E(C).
$$

由于序列 split，该纤维序列也 split，因此 $E(B)\simeq E(A)\oplus E(C)$。故 $E$ additive。$\square$

**例子 30.12.** 非连通代数 $K$-理论是 additive invariant；非连通 $K$-理论、Hochschild homology、cyclic homology、topological Hochschild homology 和许多 topological cyclic homology 版本是 localizing invariant，在适当基和完备性假设下成立。

**外部输入定理 30.13.** 非连通代数 $K$-理论把 exact sequence of small stable idempotent-complete $\infty$-categories 送到谱的纤维序列：

$$
K(A)\to K(B)\to K(C).
$$

## 30.4 Noncommutative motives 的普遍性质

**外部输入定理 30.14（普遍 additive motive）.** 存在稳定 presentable $\infty$-范畴 $\operatorname{Mot}_{\operatorname{add}}$ 和函子

$$
U_{\operatorname{add}}:\operatorname{Cat}^{\operatorname{perf}}_\infty\to\operatorname{Mot}_{\operatorname{add}}
$$

使得对任意稳定 presentable $\infty$-范畴 $\mathcal D$，预复合诱导等价

$$
\operatorname{Fun}^L(\operatorname{Mot}_{\operatorname{add}},\mathcal D)
\simeq
\operatorname{Fun}_{\operatorname{add}}(\operatorname{Cat}^{\operatorname{perf}}_\infty,\mathcal D).
$$

**外部输入定理 30.15（普遍 localizing motive）.** 存在稳定 presentable $\infty$-范畴 $\operatorname{Mot}_{\operatorname{loc}}$ 和函子

$$
U_{\operatorname{loc}}:\operatorname{Cat}^{\operatorname{perf}}_\infty\to\operatorname{Mot}_{\operatorname{loc}}
$$

使得左伴随 $\operatorname{Mot}_{\operatorname{loc}}\to\mathcal D$ 等价于取值于 $\mathcal D$ 的 localizing invariants。

**命题 30.16.** 若两个小稳定幂等完备 $\infty$-范畴 $A,B$ 在 $\operatorname{Mot}_{\operatorname{loc}}$ 中同构，则任意 localizing invariant $E$ 满足 $E(A)\simeq E(B)$。

**证明.** 由定理 30.15，$E$ 唯一因子化为

$$
\operatorname{Cat}^{\operatorname{perf}}_\infty\xrightarrow{U_{\operatorname{loc}}}
\operatorname{Mot}_{\operatorname{loc}}\xrightarrow{\overline E}\mathcal D
$$

其中 $\overline E$ 保持小余极限。若 $U_{\operatorname{loc}}(A)\simeq U_{\operatorname{loc}}(B)$，应用函子 $\overline E$ 得 $E(A)\simeq E(B)$。$\square$

## 30.5 Trace、Hochschild 同调与局部化

**命题 30.17.** Hochschild chains 对 Morita equivalence 不变。

**证明.** 第二十七章把 Hochschild chains 写作恒等 bimodule 的导出 trace：

$$
HH(\mathcal A)=\mathcal A\otimes^{\mathbb L}_{\mathcal A^{op}\otimes\mathcal A}\mathcal A.
$$

Morita equivalence 识别导出模范畴及其双模复合结构，因而识别恒等 bimodule 的 trace。严格证明使用导出 Morita 定理和 trace 在 Morita $(\infty,2)$-范畴中的函子性。$\square$

**外部输入定理 30.18.** Hochschild homology 和 topological Hochschild homology 是 localizing invariants；因此 exact sequence $A\to B\to C$ 给出纤维序列

$$
HH(A)\to HH(B)\to HH(C)
$$

和相应的 $THH$ 纤维序列。

**例子 30.19.** 若 $R$ 是环，$\operatorname{Perf}(R)$ 的 localizing invariants 只依赖 $R$ 的 derived Morita class。矩阵环 $M_n(R)$ 与 $R$ Morita 等价，所以

$$
K(M_n(R))\simeq K(R),\qquad HH(M_n(R))\simeq HH(R).
$$

## 30.6 形式后果与零判别

**命题 30.20.** 设 $E$ 是 localizing invariant，且

$$
A\to B\to C
$$

是 exact sequence。若 $E(A)\simeq0$，则 $E(B)\to E(C)$ 为等价；若 $E(C)\simeq0$，则 $E(A)\to E(B)$ 为等价。

**证明.** Localizing invariant 把 exact sequence 送为纤维序列

$$
E(A)\to E(B)\to E(C).
$$

在稳定范畴中，纤维为零的态射是等价，所以 $E(A)\simeq0$ 蕴含 $E(B)\simeq E(C)$。若 $E(C)\simeq0$，则 $E(B)\to0$ 的纤维为 $E(B)$，而该纤维又等价于 $E(A)$，故 $E(A)\to E(B)$ 是等价。$\square$

**命题 30.21.** 若 $E$ 是 additive invariant，则

$$
E(A\oplus B)\simeq E(A)\oplus E(B)
$$

对小幂等完备稳定 $\infty$-范畴 $A,B$ 成立。

**证明.** 有 split-exact sequence

$$
A\to A\oplus B\to B,
$$

其中第一箭头为第一因子嵌入，第二箭头为投影，并由第二因子嵌入分裂。Additive invariant 按定义把 split-exact sequences 送为直和分解，所以得到所需等价。$\square$

**命题 30.22.** 若 dg 范畴 $\mathcal A,\mathcal B$ derived Morita equivalent，则任意通过 perfect module 范畴定义的 additive 或 localizing invariant 在二者上取相同值。

**证明.** Derived Morita equivalence 的含义是

$$
\operatorname{Perf}(\mathcal A)\simeq\operatorname{Perf}(\mathcal B)
$$

作为小幂等完备稳定 $\infty$-范畴等价。Additive 和 localizing invariants 是定义在 $\operatorname{Cat}^{\operatorname{perf}}_\infty$ 上的函子，因此保持等价对象，给出

$$
E(\operatorname{Perf}(\mathcal A))\simeq E(\operatorname{Perf}(\mathcal B)).
$$

这正是通过 perfect modules 解释的 dg 不变量相等。$\square$

## 30.7 本章小结

dg quotient 和稳定 Verdier quotient 是同一个局部化思想在 dg 增强和稳定 $\infty$-范畴中的表现。Localizing invariants 把 exact sequences 送到纤维序列；noncommutative motives 用普遍性质统一所有 additive 或 localizing invariants。代数 $K$-理论、Hochschild 型不变量和 trace 结构是这一理论的基本例子。

## 练习

**练习 30.1.** 定义小幂等完备稳定 $\infty$-范畴的 exact sequence。

**练习 30.2.** 证明 $A\subseteq B$ 给出 $A\to B\to\operatorname{Kar}(B/A)$ 的 exact sequence。

**练习 30.3.** 定义 flasque 稳定范畴。

**练习 30.4.** 解释 Eilenberg swindle 为什么使加性不变量消失。

**练习 30.5.** 定义 dg quotient 的泛性质。

**练习 30.6.** 在 Drinfeld quotient 中证明 $\mathcal A$ 的对象在 $H^0$ 中为零。

**练习 30.7.** 定义 additive invariant。

**练习 30.8.** 定义 localizing invariant。

**练习 30.9.** 证明 localizing invariant 是 additive invariant。

**练习 30.10.** 写出 $K$-理论的局部化纤维序列。

**练习 30.11.** 陈述 $\operatorname{Mot}_{\operatorname{add}}$ 的普遍性质。

**练习 30.12.** 陈述 $\operatorname{Mot}_{\operatorname{loc}}$ 的普遍性质。

**练习 30.13.** 证明 motives 中等价推出所有 localizing invariants 等价。

**练习 30.14.** 写出 Hochschild chains 的 trace 公式。

**练习 30.15.** 说明为什么 $K(M_n(R))\simeq K(R)$ 与 $HH(M_n(R))\simeq HH(R)$。

**练习 30.16.** 设 $E$ 为 localizing invariant。证明 exact sequence $A\to B\to C$ 中若 $E(A)=0$，则 $E(B)\simeq E(C)$。

**练习 30.17.** 证明 additive invariant 把 $A\oplus B$ 送到 $E(A)\oplus E(B)$。

**练习 30.18.** 证明 derived Morita equivalent 的 dg 范畴有相同的 additive 和 localizing invariants。
