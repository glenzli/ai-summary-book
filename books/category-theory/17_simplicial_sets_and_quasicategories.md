# 第十七章：单纯集与 quasi-category

## 本章目标

本章为 $\infty$-范畴部分建立入口语言：单纯形范畴 $\Delta$、单纯集、标准单纯形、面、退化、角和 quasi-category。后续章节将在此基础上定义 $\infty$-范畴中的等价、映射空间、极限、伴随和 Cartesian fibration。

## 依赖前置知识

需要熟悉第一章的范畴、函子和反范畴。若读者熟悉拓扑中的单纯复形会有帮助，但本章不依赖几何直觉。

## 17.1 单纯形范畴

**定义 17.1.** 单纯形范畴（simplex category）$\Delta$ 定义如下：

- 对象是有限非空全序集
  $$
  [n]=\{0<1<\cdots<n\},\qquad n\geq 0.
  $$
- 态射 $[m]\to[n]$ 是保序映射，即函数 $\alpha:\{0,\dots,m\}\to\{0,\dots,n\}$ 满足
  $$
  i\leq j\Rightarrow \alpha(i)\leq\alpha(j).
  $$
- 复合是函数复合，恒等态射是恒等函数。

**命题 17.2.** $\Delta$ 是范畴。

**证明.** 保序映射的复合仍保序，恒等函数保序。函数复合满足结合律和单位律，因此 $\Delta$ 是范畴。$\square$

**定义 17.3.** 对 $0\leq i\leq n$，第 $i$ 个上同调面映射

$$
\delta^i:[n-1]\to[n]
$$

是唯一的严格递增且漏掉 $i$ 的保序单射。对 $0\leq i\leq n$，第 $i$ 个上同调退化映射

$$
\sigma^i:[n+1]\to[n]
$$

是唯一的保序满射，使得 $i$ 有两个原像。

这些映射生成 $\Delta$ 中的所有态射，并满足单纯恒等式；完整恒等式将在附录 B 统一列出和证明。

## 17.2 单纯集

**定义 17.4.** 一个单纯集（simplicial set）是函子

$$
X:\Delta^{\operatorname{op}}\to\mathbf{Set}_{\mathcal U}.
$$

其 $n$-单纯形集合记为

$$
X_n=X([n]).
$$

若 $\alpha:[m]\to[n]$ 是 $\Delta$ 中的态射，则由反变性得到函数

$$
X(\alpha):X_n\to X_m.
$$

**定义 17.5.** 单纯集之间的态射是自然变换。所有 $\mathcal U$-小单纯集构成范畴

$$
\mathbf{sSet}_{\mathcal U}=\operatorname{Fun}(\Delta^{\operatorname{op}},\mathbf{Set}_{\mathcal U}).
$$

**定义 17.6.** 标准 $n$-单纯形 $\Delta^n$ 是可表单纯集

$$
\Delta^n=\Delta(-,[n]).
$$

也就是说，

$$
(\Delta^n)_m=\Delta([m],[n]).
$$

**命题 17.7.** 对任意单纯集 $X$，存在自然双射

$$
\mathbf{sSet}(\Delta^n,X)\cong X_n.
$$

**证明.** 这是 Yoneda 引理在 $\Delta$ 上的直接应用：$\Delta^n$ 是由对象 $[n]$ 表示的预层 $\Delta(-,[n])$，因此从 $\Delta^n$ 到 $X$ 的自然变换与 $X([n])$ 的元素自然对应。$\square$

## 17.3 面、退化与边界

**定义 17.8.** 对单纯集 $X$，由 $\delta^i:[n-1]\to[n]$ 诱导的映射

$$
d_i=X(\delta^i):X_n\to X_{n-1}
$$

称为第 $i$ 个面映射。由 $\sigma^i:[n+1]\to[n]$ 诱导的映射

$$
s_i=X(\sigma^i):X_n\to X_{n+1}
$$

称为第 $i$ 个退化映射。

**定义 17.9.** 标准 $n$-单纯形的边界 $\partial\Delta^n\subseteq\Delta^n$ 是由所有面

$$
\delta^i:\Delta^{n-1}\to\Delta^n,\qquad 0\leq i\leq n
$$

生成的子单纯集。

**定义 17.10.** 对 $0\leq i\leq n$，第 $i$ 个角（horn）

$$
\Lambda_i^n\subseteq\Delta^n
$$

是由除第 $i$ 个面以外的所有面生成的子单纯集：

$$
\Lambda_i^n=\bigcup_{j\neq i}d_j\Delta^{n-1}\subseteq\Delta^n.
$$

当 $0<i<n$ 时，$\Lambda_i^n$ 称为内角（inner horn）；当 $i=0$ 或 $i=n$ 时，称为外角（outer horn）。

## 17.4 quasi-category 的定义

**定义 17.11.** 一个单纯集 $X$ 称为 quasi-category，若对所有 $n\geq 2$ 和所有 $0<i<n$，任意单纯集态射

$$
\Lambda_i^n\to X
$$

都可以扩张为态射

$$
\Delta^n\to X.
$$

也就是说，任意内角嵌入

$$
\Lambda_i^n\hookrightarrow\Delta^n
$$

对于 $X$ 都具有右提升性质：

$$
\begin{matrix}
\Lambda_i^n&\longrightarrow&X\\
\downarrow&&\downarrow\\
\Delta^n&\longrightarrow&*
\end{matrix}
$$

其中虚线填充 $\Delta^n\to X$ 要求存在，但不要求唯一。

**约定 17.12.** 本书后文若不特别说明，$\infty$-范畴指 quasi-category。此约定不是说其他模型不重要，而是为了固定证明语言；与 simplicial categories、complete Segal spaces 和 relative categories 的比较将在后续章节处理。

## 17.5 普通范畴的 nerve

**定义 17.13.** 设 $\mathcal C$ 是普通范畴。其 nerve 是单纯集

$$
N(\mathcal C):\Delta^{\operatorname{op}}\to\mathbf{Set},
$$

定义为

$$
N(\mathcal C)_n=\operatorname{Fun}([n],\mathcal C),
$$

其中 $[n]$ 被视为由全序集给出的薄范畴。若 $\alpha:[m]\to[n]$ 是 $\Delta$ 中态射，则

$$
N(\mathcal C)(\alpha):N(\mathcal C)_n\to N(\mathcal C)_m
$$

由预复合给出：

$$
F:[n]\to\mathcal C
\quad\longmapsto\quad
F\circ\alpha:[m]\to\mathcal C.
$$

**例子 17.14.** 一个 $0$-单纯形是 $\mathcal C$ 的对象。一个 $1$-单纯形是 $\mathcal C$ 中的一个态射。一个 $2$-单纯形是图形

$$
X_0\xrightarrow{f}X_1\xrightarrow{g}X_2
$$

连同复合边 $g\circ f:X_0\to X_2$；在 nerve 中，这条复合边不是额外选择，而由函子 $[2]\to\mathcal C$ 的函子性决定。

**定理 17.15.** 对任意普通范畴 $\mathcal C$，$N(\mathcal C)$ 是 quasi-category。更强地，$N(\mathcal C)$ 对所有内角都有唯一填充。

**证明.** 一个 $\Delta^n\to N(\mathcal C)$ 等价于一个函子 $[n]\to\mathcal C$，也等价于给出对象

$$
X_0,\dots,X_n
$$

和所有 $i\leq j$ 的态射 $X_i\to X_j$，并满足由 $[n]$ 中唯一复合关系强制的相容性。一个内角 $\Lambda_i^n\to N(\mathcal C)$ 已经包含所有相邻边 $X_{k-1}\to X_k$，并包含足够的二维面来强制所有长边必须是这些相邻边的复合。由于普通范畴中的复合严格结合，缺失的第 $i$ 个面被唯一确定。附录 B 的定理 B.4 展开了这一论证，并证明唯一填充。故 $N(\mathcal C)$ 满足所有内角填充条件，是 quasi-category。$\square$

## 17.6 三个基本计算

**计算 17.A.** 标准 $2$-单纯形 $\Delta^2$ 的 $0$-单纯形是保序映射 $[0]\to[2]$，即三个顶点 $0,1,2$。其非退化 $1$-单纯形是三个严格递增映射 $[1]\to[2]$：

$$
0\to1,\qquad 0\to2,\qquad 1\to2.
$$

退化 $1$-单纯形来自常值映射 $[1]\to[2]$，对应三个恒等边。边界 $\partial\Delta^2$ 包含全部三个非退化边，而内角 $\Lambda_1^2$ 只包含

$$
0\to1,\qquad 1\to2
$$

和三个顶点；它缺少长边 $0\to2$。因此一个映射 $\Lambda_1^2\to X$ 正是 $X$ 中两条可复合边的数据。

**计算 17.B.** 令 $[1]$ 表示有两个对象 $0<1$ 和唯一非恒等态射 $0\to1$ 的范畴。则

$$
N([1])\cong\Delta^1.
$$

事实上，

$$
N([1])_n=\operatorname{Fun}([n],[1])
$$

就是保序映射 $[n]\to[1]$ 的集合，而这正是

$$
(\Delta^1)_n=\Delta([n],[1]).
$$

对 $\Delta$ 中态射的作用在两边都是预复合，所以这些逐级等式组成单纯集同构。

**计算 17.C.** 对普通范畴 $\mathcal C$ 和态射

$$
X\xrightarrow{f}Y\xrightarrow{g}Z,
$$

由 $f,g$ 给出的内角 $\Lambda_1^2\to N(\mathcal C)$ 的唯一填充对应复合 $g\circ f$。映射 $\Lambda_1^2\to N(\mathcal C)$ 给出顶点 $X,Y,Z$ 和边 $f,g$。填充 $\Delta^2\to N(\mathcal C)$ 等价于函子 $[2]\to\mathcal C$，因此还必须给出边 $X\to Z$，并满足它等于 $g\circ f$。普通范畴中复合已唯一给定，所以填充存在且唯一。

## 17.7 为什么内角填充表达“可复合”

在普通范畴中，两个可复合态射

$$
X\xrightarrow{f}Y\xrightarrow{g}Z
$$

有唯一复合 $g\circ f$。在 quasi-category 中，内角

$$
\Lambda_1^2\to X
$$

可以理解为给出两条可复合的 $1$-单纯形。填充

$$
\Delta^2\to X
$$

给出一条候选复合边以及一个二维单纯形，表示这条边确实是 $f$ 与 $g$ 的一个复合。

关键差别是：填充只要求存在，不要求唯一。多个填充之间再由更高维单纯形组织，其“唯一性”应理解为同伦意义下的可缩选择空间，而不是集合论意义下的唯一元素。后续章节会把这句话转化为映射空间和同伦范畴的严格定义。

## 17.8 Kan 复形与 Joyal 模型结构

**定义 17.16.** 单纯集 $X$ 称为 Kan 复形（Kan complex），若对所有 $n\geq 1$ 和所有 $0\leq i\leq n$，任意 horn

$$
\Lambda_i^n\to X
$$

都可扩张为

$$
\Delta^n\to X.
$$

**命题 17.17.** 每个 Kan 复形都是 quasi-category。

**证明.** quasi-category 只要求对 $0<i<n$ 的内 horn 有填充。Kan 复形要求对所有 horn 有填充，特别包含所有内 horn。$\square$

**命题 17.18.** 若 $X$ 是 Kan 复形，则 $hX$ 中每个态射都是同构。

**证明.** 设 $f:x\to y$ 是 $X$ 的一条边。外 horn 填充给出 $g:y\to x$ 以及二维单纯形，表达 $g f$ 与 $\operatorname{id}_x$ 同伦；另一个外 horn 填充给出二维单纯形，表达 $f g$ 与 $\operatorname{id}_y$ 同伦。因此 $f$ 在同伦范畴 $hX$ 中有逆。$\square$

**外部输入定理 17.19（Joyal 模型结构）.** $\mathbf{sSet}$ 上存在模型结构，其 cofibration 为单射，fibrant objects 为 quasi-categories，weak equivalences 为 categorical equivalences。该模型结构称为 Joyal 模型结构。其证明和 categorical equivalence 的若干等价刻画见 Joyal、Lurie HTT、Cisinski 和 Kerodon。

**注 17.20.** Kan-Quillen 模型结构把 Kan 复形作为 fibrant objects，用来建模 spaces 或 $\infty$-群胚；Joyal 模型结构把 quasi-categories 作为 fibrant objects，用来建模 $\infty$-范畴。二者的差别正是“所有态射可逆”与“一般态射不必可逆”的差别。

## 17.9 本章小结

单纯集是 $\Delta$ 上的预层。标准单纯形由 Yoneda 表示，角是标准单纯形的特定子单纯集。quasi-category 是满足所有内角填充条件的单纯集。普通范畴通过 nerve 嵌入 quasi-category 世界；区别在于普通范畴的内角填充唯一，而一般 quasi-category 只要求存在。Kan 复形是所有 horn 可填的 quasi-category，因而建模 $\infty$-群胚；Joyal 模型结构则把 quasi-category 组织成同伦理论。

## 练习

**练习 17.1.** 写出 $\Delta([1],[2])$ 的全部元素，并说明它们对应 $\Delta^2$ 的哪些 $1$-单纯形。

**练习 17.2.** 验证 $\delta^i:[n-1]\to[n]$ 和 $\sigma^i:[n+1]\to[n]$ 是保序映射。

**练习 17.3.** 使用 Yoneda 引理证明命题 17.7，并明确指出自然性变量。

**练习 17.4.** 描述 $\Lambda_1^2$ 的 $0$-单纯形和非退化 $1$-单纯形。解释为什么它对应两条可复合边。

**练习 17.5.** 对普通范畴 $\mathcal C$，写出 $N(\mathcal C)_3$ 的数据，并说明其中哪些边由相邻三条边的复合决定。

**练习 17.6.** 说明为什么 quasi-category 定义只要求内角填充，而 Kan 复形要求所有角填充。此题只要求查阅定义并比较，不要求证明模型范畴结论。

**练习 17.7.** 证明若普通范畴 $\mathcal C$ 中存在非可逆态射，则 $N(\mathcal C)$ 不是 Kan 复形。

**练习 17.8.** 比较 Kan-Quillen 模型结构和 Joyal 模型结构的 fibrant objects。

**练习 17.9.** 解释为什么命题 17.18 支持“Kan 复形是 $\infty$-群胚”的说法。

**练习 17.10.** 证明 $N([n])\cong\Delta^n$。

**练习 17.11.** 对普通范畴 $\mathcal C$，证明由三条可复合边 $X_0\to X_1\to X_2\to X_3$ 给出的 $\Lambda_2^3\to N(\mathcal C)$ 有唯一填充，并写出缺失面。
