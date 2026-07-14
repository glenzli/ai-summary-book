# 第二十一章：Log schemes、perfect schemes 与 universal homeomorphisms

普通 motivic homotopy 用闭开局部化处理边界，并把 Frobenius 保留为一个通常不可逆的
态射。Log geometry 与 perfect geometry 分别改变这两个选择：前者把 divisor 的
边界数据编码进对象，后者只考虑 Frobenius 已可逆的几何。Universal homeomorphism
invariance 则精确描述了普通 `\mathbf{SH}` 在反演指数特征后忘掉多少纯不可分信息。

这三套观念不能混成一个定义。Log motivic homotopy 有自己的 site 与 interval；
perfect motivic homotopy 是正特征中的另一种模型；universal homeomorphism theorem
则是普通 `\mathbf{SH}` 的局部化结论。本章分别建立三者的定义域，再说明它们相交的
位置。

## 21.1 对数结构与边界

**定义 21.1.** 概形 `X` 上的 log structure 是交换幺半群层 `M_X`、态射
`\alpha:M_X\to\mathcal O_X`（右端按乘法看待），以及条件

$$
\alpha^{-1}(\mathcal O_X^\times)\xrightarrow{\ \simeq\ }
\mathcal O_X^\times.
$$

若特征幺半群 `\overline M_X=M_X/\mathcal O_X^\times` etale 局部来自 finitely
generated integral saturated monoid，则称 `(X,M_X)` 为 fine and saturated，简称
fs log scheme。Trivial log structure 是 `M_X=\mathcal O_X^\times`。

**例子 21.2（divisorial log structure）.** 设 `j:U=X\setminus D\hookrightarrow X`
是稠密开浸入。令

$$
M_{(X,D)}=\mathcal O_X\cap j_*\mathcal O_U^\times
$$

为在 `U` 上可逆的正则函数层，交取在 `j_*\mathcal O_U` 中。若 `D` 是 normal
crossings divisor，则它给出 fs log structure。局部方程 `x_1\cdots x_r=0` 对应图表
`\mathbb N^r\to\mathcal O_X`，第 `a` 个基向量送到 `x_a`；因而交叉分支数也被记录。

**外部输入定理 21.3（Park）.** Fs log schemes 上存在
`\mathbb A^1`-local stable motivic homotopy categories。它们满足 localization；
在 trivial log structures 上与 Morel--Voevodsky 的构造等价，并由此在 strict
morphisms of fs log schemes 上获得 Grothendieck 六操作形式主义。这里的
`\mathbb A^1` 与局部化均按 Park 的 log-site 构造理解。来源为 Doosung Park,
*A1-homotopy theory of log schemes*, arXiv:2205.14750 的主定理。

**命题 21.4.** Trivial log structure 的比较是外部输入定理的一部分，不能只由
`M_X=\mathcal O_X^\times` 形式推出。

**证明.** 对象层的 triviality 只说明没有额外特征幺半群；要得到范畴等价，还需比较
log smooth site、覆盖、`\mathbb A^1`-局部化及稳定化。定理 21.3 同时控制这些步骤，
所以比较依赖该定理。`\square`

## 21.2 Universal homeomorphisms

**定义 21.5.** 态射 `f:T\to S` 称为 universal homeomorphism，若任意基变换后都在
底层拓扑空间上诱导同胚。等价地，`f` integral、surjective 且 universally injective
（即 radicial）。

**外部输入定理 21.6（Elmanto--Khan）.** 设 `P` 是一组素数，并假设每个
`q\notin P` 在 `\mathcal O_S` 中可逆。对任意 universal homeomorphism
`f:T\to S`，inverse image 诱导等价

$$
f^*:\mathbf{SH}(S)[P^{-1}]\xrightarrow{\ \simeq\ }
\mathbf{SH}(T)[P^{-1}].
$$

这是 *Perfection in motivic homotopy theory*, Theorem 2.1.1。若 `S` 的指数特征为
`p`，可取 `P=\{p\}`，得到 `\mathbf{SH}[1/p]` 中的拓扑不变性。

**例子 21.7（Frobenius）.** 对每个 `\mathbb F_p`-scheme `X`，absolute Frobenius

$$
F_X:X\longrightarrow X
$$

在底层空间上为恒等，并在结构层上取 `p` 次幂。它对**任意** `X` 都是 universal
homeomorphism，不需要 Noetherian 或有限型假设；若 `X` perfect，它才进一步成为
同构。定理 21.6 因此给出

$$
F_X^*:\mathbf{SH}(X)[1/p]\xrightarrow{\ \simeq\ }
\mathbf{SH}(X)[1/p].
$$

**例子 21.8（nilpotent thickening）.** 若 `I\subset A` 是 nilpotent ideal，则
`\operatorname{Spec}(A/I)\hookrightarrow\operatorname{Spec}(A)` 是 universal
homeomorphism。这个特例比定理 21.6 更强：stable motivic homotopy 的闭开局部化
已经给出 nil-invariance，故该拉回在整系数下就是等价。一般纯不可分 universal
homeomorphism 则不能据此获得整系数等价。

**命题 21.9.** “nil-invariance”不蕴含“所有 universal homeomorphisms 下的 integral
invariance”。

**证明.** 前者处理由 nilpotent ideal 给出的闭浸入；后者还包括非平凡纯不可分域扩张
与 Frobenius。若域 `k` 的 Frobenius 在 integral `\mathbf{SH}(k)` 上诱导等价，则它在
`K_1(k)=k^\times` 上诱导的 `x\mapsto x^p` 也必须为同构；对非 perfect `k` 这不成立。
这正说明定理 21.6 中反演 `p` 不能一般删去。`\square`

## 21.3 Perfectization

**定义 21.10.** `\mathbb F_p`-scheme `X` 称为 perfect，若 `F_X` 是同构。其
perfectization 定义为概形极限

$$
X^{\mathrm{perf}}
=\varprojlim(\cdots\xrightarrow{F_X}X\xrightarrow{F_X}X).
$$

在仿射情形 `X=\operatorname{Spec}A`，这等于

$$
\operatorname{Spec}\!\left(\varinjlim
(A\xrightarrow{F_A}A\xrightarrow{F_A}\cdots)\right).
$$

因此必须区分 scheme 方向的 inverse limit 与 ring 方向的 direct limit。

**命题 21.11.** 若 `X` perfect，则 canonical map
`X^{\mathrm{perf}}\to X` 是同构；对一般 `X`，它是 universal homeomorphism。

**证明.** 第一项由系统中的所有 Frobenius 均为同构立即得到。一般情形可仿射局部
验证：`A\to A^{\mathrm{perf}}` integral、radicial，且在素谱上为同胚；这些性质可粘合。
`\square`

**推论 21.12.** 对任意 `\mathbb F_p`-scheme `X`，有

$$
\mathbf{SH}(X)[1/p]\simeq
\mathbf{SH}(X^{\mathrm{perf}})[1/p].
$$

**证明.** 对 canonical universal homeomorphism
`X^{\mathrm{perf}}\to X` 应用定理 21.6；这也是 Elmanto--Khan Corollary 2.1.7。
`\square`

**研究边界 21.13.** Dahlhausen--Hekking--Wolters 的 2025 预印本
*Motivic homotopy theory for perfect schemes* 直接在 perfect schemes 上构造
motivic category，验证 coefficient-system 与六操作公理，并证明其中乘 `p` 已可逆；
其模型识别 universal-homeomorphism localization 与 `\mathbf{SH}[1/p]`。截至本书
核查日期，此结果按预印本边界引用，不替代定理 21.6 的已发表结论。

## 21.4 三种理论如何相接

Log structure 保存边界的幺半群数据；perfectization 忘掉 Frobenius 方向；
universal-homeomorphism localization 则说明普通 motivic theory 在反演 `p` 后也看不见
这类方向。它们之间有比较，却没有无条件等同。

例如，带 normal crossings divisor `D` 的 `(X,M_{(X,D)})` 与开补
`U=X\setminus D` 拥有同一几何背景，但前者还记住各边界分支及其交叉。另一方面，
`X^{\mathrm{perf}}\to X` 在 `\mathbf{SH}[1/p]` 中成为等价，却不会使这些 log
边界数据自动消失。任何横跨三种理论的定理，都必须分别指定 log structure、特征、
系数局部化和允许的态射类。

## 21.5 边界、Frobenius 与系数

本章的关键区别可以概括为：nilpotent thickening 在 stable motivic homotopy 中整系数
不可见；一般 universal homeomorphism 只在适当素数局部化后不可见；perfect theory
则从定义开始就令 Frobenius 可逆。Log theory处理的是另一方向，它增加而不是删除
边界信息。这些差异决定了哪些比较可以成为定理，哪些只能保留为研究中的模型对应。

## 练习

**练习 21.1.** 验证 trivial log structure 满足定义 21.1。

**练习 21.2.** 对 normal crossings 方程 `xy=0` 写出 divisorial log structure 的局部图表。

**练习 21.3.** 证明 `\mathbb F_p`-scheme 的 absolute Frobenius 是 universal
homeomorphism。

**练习 21.4.** 说明定理 21.6 在指数特征 `p` 情形为什么要求反演 `p`。

**练习 21.5.** 证明 nilpotent thickening 诱导底层拓扑空间的同胚。

**练习 21.6.** 在仿射情形核对 perfectization 的两种极限公式。

**练习 21.7.** 证明 perfect `X` 满足 `X^{\mathrm{perf}}\simeq X`。

**练习 21.8.** 比较 divisorial log structure 与闭开 localization 各自保存的信息。

**练习 21.9.** 用 `K_1(k)` 解释非 perfect 域上 integral Frobenius invariance 为何失败。

**练习 21.10.** 说明研究边界 21.13 为什么不能反向推出普通
`\mathbf{SH}(X)` 已经是 perfect motivic category。
