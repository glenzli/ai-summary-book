# 第四十二章：因子化同调、$E_n$-代数与非阿贝尔 Poincare 对偶

一个 $E_n$-代数描述在小 $n$-圆盘上可进行的局部运算；因子化同调通过对称幺半左 Kan 延拓，把这些局部数据沿任意 $n$-流形装配。Excision 把沿 collar-gluing 的几何分解送到相对张量积，因此 $S^1$ 上的因子化同调恢复 Hochschild homology，而 grouplike $E_n$-space 的情形给出非阿贝尔 Poincare 对偶。本章从 Disk$_n$-algebra 的普适定义推导这些例子和局部--整体公式。

本章依赖 $\infty$-operad、presentable 对称幺半 $\infty$-范畴、Kan 延拓和 Morita theory。流形需带 framing 或相应 tangential structure；非阿贝尔对偶的 grouplike 与连通性假设会明确保留。

## 42.1 小圆盘范畴与 $E_n$-代数

**定义 42.1.** $\operatorname{Disk}^{fr}_n$ 表示由有限个带标准 framing 的 $\mathbb R^n$ 的不交并组成的对称幺半 $\infty$-范畴，态射为保持 framing 的光滑开嵌入空间，幺半结构为不交并。下文简写为 $\operatorname{Disk}_n$；若改用其他 tangential structure，必须同时替换圆盘和流形范畴。

**定义 42.2.** 设 $C^\otimes$ 为对称幺半 $\infty$-范畴。$C$ 中的 $E_n$-代数是对称幺半函子

$$
A:\operatorname{Disk}_n\to C.
$$

记其 $\infty$-范畴为 $\operatorname{Alg}_{E_n}(C)$。

**命题 42.3.** 若 $A$ 是 $E_n$-代数，则 $A$ 在单个圆盘 $\mathbb R^n$ 上的值决定其在 $\operatorname{Disk}_n$ 对象上的值。

**证明.** $\operatorname{Disk}_n$ 的对象是有限不交并 $\bigsqcup_{i\in I}\mathbb R^n$。因为 $A$ 是对称幺半函子，

$$
A\left(\bigsqcup_{i\in I}\mathbb R^n\right)\simeq\bigotimes_{i\in I}A(\mathbb R^n).
$$

因此对象层面的值由 $A(\mathbb R^n)$ 和张量单位决定。运算空间中的嵌入给出 $E_n$-乘法及其高阶相干性。$\square$

## 42.2 因子化同调的 Kan 延拓定义

**定义 42.4.** 记 $\operatorname{Mfld}^{fr}_n$ 为 framed $n$-维光滑流形与保持 framing 的开嵌入组成的对称幺半 $\infty$-范畴，幺半结构为不交并，并简写为 $\operatorname{Mfld}_n$。包含函子记作

$$
i:\operatorname{Disk}_n\hookrightarrow\operatorname{Mfld}_n.
$$

**定义 42.5.** 设 $C^\otimes$ presentable，且张量积分别保持小余极限。$E_n$-代数 $A:\operatorname{Disk}_n\to C$ 的因子化同调是 $A$ 沿 $i$ 的对称幺半左 Kan 延拓：

$$
\int_M A=(\operatorname{Lan}_iA)(M).
$$

点态地，

$$
\int_MA\simeq \operatorname*{colim}_{(U\hookrightarrow M)\in\operatorname{Disk}_{n/M}}A(U).
$$

**命题 42.6.** 对 $M=\mathbb R^n$，有自然等价

$$
\int_{\mathbb R^n}A\simeq A(\mathbb R^n).
$$

**证明.** 在 overcategory $\operatorname{Disk}_{n/\mathbb R^n}$ 中，恒等嵌入 $\mathbb R^n\hookrightarrow\mathbb R^n$ 是终对象：任意圆盘并 $U\hookrightarrow\mathbb R^n$ 到它有唯一的 overcategory 态射，即该嵌入本身。终对象上的余极限等于该对象处的值，所以点态公式给出

$$
\int_{\mathbb R^n}A\simeq A(\mathbb R^n).
$$

$\square$

## 42.3 不交并与张量结构

**命题 42.7.** 因子化同调作为对称幺半左 Kan 延拓满足

$$
\int_{M\sqcup N}A\simeq\left(\int_MA\right)\otimes\left(\int_NA\right).
$$

**证明.** $\operatorname{Lan}_iA$ 是对称幺半函子，因为它按定义为对称幺半左 Kan 延拓。因此它把 $\operatorname{Mfld}_n$ 的幺半积，即不交并，送到 $C$ 中的张量积。代入 $M,N$ 即得公式。$\square$

**例子 42.8.** 若 $M=\varnothing$，则

$$
\int_{\varnothing}A\simeq\mathbb 1_C.
$$

这是因为对称幺半函子保持幺半单位。

## 42.4 Excision

**定义 42.9.** 设 $M$ 由沿柱状开集 $N\times\mathbb R$ 的 collar-gluing 分解

$$
M=M_-\cup_{N\times\mathbb R}M_+.
$$

这种分解称为因子化同调的 excision 分解。

**外部输入定理 42.10（因子化同调 excision）.** 对 $E_n$-代数 $A$，上述分解诱导自然等价

$$
\int_MA\simeq
\left(\int_{M_-}A\right)\otimes_{\int_{N\times\mathbb R}A}
\left(\int_{M_+}A\right).
$$

**命题 42.11.** Excision 蕴含因子化同调由圆盘值和流形分解递归控制。

**证明.** 因子化同调在圆盘上由命题 42.6 计算，在不交并上由命题 42.7 计算。若一个流形可由圆盘、柱状边界和有限次 collar-gluing 得到，则每次 gluing 都由定理 42.10 把整体值表达为两个较小部分在公共边界值上的相对张量积。因此递归地由局部圆盘值和 gluing 数据确定。$\square$

## 42.5 圆周与 Hochschild homology

**外部输入定理 42.12.** 设 $C$ 是 presentable 对称幺半稳定 $\infty$-范畴，张量积分别保持小余极限；若 $A\in\operatorname{Alg}_{E_1}(C)$，则

$$
\int_{S^1}A\simeq HH(A).
$$

更一般地，圆周因子化同调是 $A$ 作为 $A$-$A$ 双模的 trace。

**命题 42.13.** 若 $A$ 是合适的 $E_1$-代数，则 $HH(A)$ 带有由 $S^1$ 的旋转诱导的圆作用。

**证明.** 定理 42.12 把 $HH(A)$ 识别为 $\int_{S^1}A$。圆周的自同胚，特别是旋转，作用在 overcategory $\operatorname{Disk}_{1/S^1}$ 上，从而作用在点态余极限

$$
\operatorname*{colim}_{\operatorname{Disk}_{1/S^1}}A(U).
$$

这些作用随旋转参数连续相干，给出 $S^1$-作用。$\square$

## 42.6 非阿贝尔 Poincare 对偶

**定义 42.14.** $E_n$-空间 $A$ 称为 grouplike，若 $\pi_0(A)$ 在诱导的 $E_n$-乘法下为群。

**外部输入定理 42.15（非阿贝尔 Poincare 对偶）.** 若 $A$ 是 grouplike $E_n$-space，且 $M$ 是 framed $n$-manifold（允许非紧，但采用 compact support），则

$$
\int_MA\simeq \operatorname{Map}_c(M,B^nA),
$$

其中 $B^nA$ 为 $n$-重 delooping，右侧为带紧支撑条件的映射空间。

**命题 42.16.** 当 $M=\mathbb R^n$ 时，非阿贝尔 Poincare 对偶与命题 42.6 相容。

**证明.** 命题 42.6 给出 $\int_{\mathbb R^n}A\simeq A$。另一方面，紧支撑映射 $\mathbb R^n\to B^nA$ 等价于基点映射 $S^n\to B^nA$，即

$$
\Omega^nB^nA\simeq A,
$$

其中最后等价由 $A$ grouplike 且可 $n$-重 delooping 得到。$\square$

## 42.7 因子化代数与局部常值性

**定义 42.17.** 流形 $M$ 上的 prefactorization algebra 是把 $M$ 中有限个两两不交开集到一个开集的包含

$$
U_1,\dots,U_k\subset V
$$

送到结构映射

$$
F(U_1)\otimes\cdots\otimes F(U_k)\to F(V)
$$

并满足对称性、单位和复合相干性的资料。

**定义 42.18.** Prefactorization algebra 称为 factorization algebra，若它还满足 Weiss cover 的 cosheaf 型 descent。

**外部输入定理 42.19.** $\mathbb R^n$ 上局部常值 factorization algebras 与 $E_n$-代数等价。

**命题 42.20.** 因子化同调可视为局部常值因子化代数在 $M$ 上的余全局截面。

**证明.** $E_n$-代数 $A$ 经定理 42.19 对应到局部常值 factorization algebra。余全局截面按 Weiss cover 和圆盘覆盖取同伦余极限。因子化同调的点态公式正是对所有嵌入 $U\hookrightarrow M$ 的圆盘开集取余极限，因此二者由同一个局部到整体泛性质刻画。$\square$

## 42.8 把局部代数沿流形积分

因子化同调以 $E_n$-代数为局部系数，把流形分解转化为相对张量积。圆盘计算、对称幺半性和 excision 是它的三个基本公理。圆周上的情形恢复 Hochschild homology；grouplike $E_n$-空间的情形给出非阿贝尔 Poincare 对偶；因子化代数则给出同一结构的 cosheaf 表述。

## 练习

**练习 42.1.** 定义 $\operatorname{Disk}_n$。

**练习 42.2.** 定义 $E_n$-代数为对称幺半函子。

**练习 42.3.** 证明 $E_n$-代数在有限不交并圆盘上的值由 $A(\mathbb R^n)$ 决定。

**练习 42.4.** 写出因子化同调的左 Kan 延拓定义。

**练习 42.5.** 证明 $\int_{\mathbb R^n}A\simeq A(\mathbb R^n)$。

**练习 42.6.** 证明 $\int_{M\sqcup N}A\simeq\int_MA\otimes\int_NA$。

**练习 42.7.** 陈述因子化同调 excision。

**练习 42.8.** 说明 excision 如何递归计算流形上的因子化同调。

**练习 42.9.** 陈述 $\int_{S^1}A$ 与 Hochschild homology 的关系。

**练习 42.10.** 说明 $HH(A)$ 的圆作用来源。

**练习 42.11.** 定义 grouplike $E_n$-空间。

**练习 42.12.** 陈述非阿贝尔 Poincare 对偶。

**练习 42.13.** 验证非阿贝尔 Poincare 对偶在 $\mathbb R^n$ 上与局部计算相容。

**练习 42.14.** 定义 factorization algebra，并说明其与 $E_n$-代数的关系。
