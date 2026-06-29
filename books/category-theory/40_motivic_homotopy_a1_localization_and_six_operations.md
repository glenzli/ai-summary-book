# 第四十章：Motivic homotopy、$\mathbb A^1$-局部化与六操作

## 本章目标

本章介绍 Morel-Voevodsky motivic homotopy theory 的范畴论骨架。Motivic homotopy 把光滑概形上的 presheaves of spaces 先 sheaf 化，再强制仿射直线 $\mathbb A^1$ 成为同伦等价，从而得到 motivic spaces；进一步稳定化得到 stable motivic homotopy category $\mathbf{SH}(S)$。这是把同伦论、代数几何和六操作形式主义统一的核心平台。

## 依赖前置知识

需要站点、sheaf、$\infty$-topos、Bousfield localization、presentable $\infty$-categories、spectra、稳定化、六操作、compact generation 和 derived/spectral 几何基础。

## 40.1 光滑站点与 motivic spaces

**定义 40.1.** 设 $S$ 为基概形。记 $\operatorname{Sm}_S$ 为 $S$ 上光滑有限型概形范畴。预层 $\infty$-范畴为

$$
\mathcal P(\operatorname{Sm}_S)=\operatorname{Fun}(\operatorname{Sm}_S^{op},\mathcal S).
$$

**定义 40.2.** Nisnevich topology 是 $\operatorname{Sm}_S$ 上的 Grothendieck topology，由 étale morphisms 组成的覆盖族满足点提升条件。Nisnevich sheaves of spaces 组成 $\infty$-topos

$$
\operatorname{Shv}_{Nis}(\operatorname{Sm}_S).
$$

**定义 40.3.** Motivic spaces $\mathbf H(S)$ 定义为 $\operatorname{Shv}_{Nis}(\operatorname{Sm}_S)$ 关于投影

$$
X\times\mathbb A^1\to X
$$

的 accessible localization。局部对象称为 $\mathbb A^1$-invariant Nisnevich sheaves。

**命题 40.4.** $\mathbf H(S)$ 是 presentable $\infty$-category。

**证明.** 预层 $\infty$-范畴 presentable。Nisnevich sheaf 化是 accessible left exact localization，故 sheaf $\infty$-topos presentable。再对一小集合形如 $X\times\mathbb A^1\to X$ 的态射作 accessible localization，仍得到 presentable $\infty$-category。$\square$

## 40.2 $\mathbb A^1$-不变性与局部对象

**定义 40.5.** Nisnevich sheaf $F$ 称为 $\mathbb A^1$-invariant，若对所有 $X\in\operatorname{Sm}_S$，映射

$$
F(X)\to F(X\times\mathbb A^1)
$$

为等价。

**命题 40.6.** $\mathbf H(S)$ 的对象可识别为 $\mathbb A^1$-invariant Nisnevich sheaves 的反射子范畴。

**证明.** $\mathbf H(S)$ 按定义是把所有 $X\times\mathbb A^1\to X$ 局部化后的范畴。Accessible localization 的局部对象定义为对这些态射取映射空间后为等价的对象，即对所有 $X$ 有

$$
\operatorname{Map}(X,F)\simeq\operatorname{Map}(X\times\mathbb A^1,F).
$$

由 Yoneda，这正是 $F(X)\simeq F(X\times\mathbb A^1)$。$\square$

**例子 40.7.** 代数 $K$-理论 presheaf 在合适正则性假设下满足 Nisnevich descent 和 $\mathbb A^1$-invariance，因此给出 motivic space 或 motivic spectrum 的例子。

## 40.3 稳定 motivic homotopy category

**定义 40.8.** Tate sphere 定义为

$$
T=\mathbb A^1/(\mathbb A^1\setminus0)
$$

或等价地 $T\simeq S^1\wedge\mathbb G_m$。Stable motivic homotopy category 定义为 $T$-spectra：

$$
\mathbf{SH}(S)=\operatorname{Sp}_T(\mathbf H_*(S)).
$$

**外部输入定理 40.9.** $\mathbf{SH}(S)$ 是稳定 presentable 对称幺半 $\infty$-范畴。其同伦范畴恢复 Morel-Voevodsky stable motivic homotopy category。

**外部输入定理 40.10（复 realization）.** 若 $S=\operatorname{Spec}\mathbb C$，复点构造与 Nisnevich descent、$\mathbb A^1$-局部化和 $T$-稳定化相容，并给出对称幺半函子

$$
\mathbf{SH}(S)\to\mathbf{Sp}
$$

把 motivic suspension coordinates 送到拓扑 suspension coordinates 的相应组合。

## 40.4 Motivic 六操作

**外部输入定理 40.11.** 对足够一般的基概形，$\mathbf{SH}(-)$ 形成六操作形式主义。对 $f:X\to Y$，存在

$$
f^*,\quad f_*,\quad f_!,\quad f^!,\quad -\otimes-,\quad \underline{\operatorname{Hom}}
$$

并满足基变换、投影公式、proper compatibility、purity 和 localization triangles。

**命题 40.12.** 若 $j:U\hookrightarrow X$ 为开嵌入，$i:Z\hookrightarrow X$ 为闭补，则 motivic localization triangle 形如

$$
j_!j^*E\to E\to i_*i^*E.
$$

**证明.** 这是六操作形式主义中 recollement/localization axiom 的 motivic 特例。开嵌入的 extension by zero 和闭嵌入的 direct image 给出局部-闭补分解；稳定范畴中该分解写成余纤维序列。$\square$

## 40.5 Purity 与 Thom spaces

**外部输入定理 40.13（Homotopy purity）.** 若 $i:Z\hookrightarrow X$ 是光滑概形之间的闭嵌入，法丛为 $N_{Z/X}$，则在 motivic homotopy 中有等价

$$
X/(X\setminus Z)\simeq \operatorname{Th}(N_{Z/X}).
$$

**定义 40.14.** 向量丛 $V\to X$ 的 Thom space 定义为

$$
\operatorname{Th}(V)=V/(V\setminus X)
$$

其中 $X$ 通过零截面嵌入 $V$。

**命题 40.15.** 对零向量丛 $0_X$，$\operatorname{Th}(0_X)\simeq X_+$。

**证明.** 零向量丛的总空间就是 $X$，零截面为恒等嵌入。补 $0_X\setminus X$ 为空，因此商

$$
X/\varnothing
$$

就是给 $X$ 加基点的对象 $X_+$。$\square$

## 40.6 Motives 与 Eilenberg-Mac Lane 谱

**外部输入定理 40.16.** 存在 motivic Eilenberg-Mac Lane spectrum $H\mathbb Z$。其模范畴与 Voevodsky motives 或相应 derived category of motives 在合适假设下密切相关：

$$
\operatorname{Mod}_{H\mathbb Z}(\mathbf{SH}(S)).
$$

**注 40.17.** 这把 stable motivic homotopy theory 和 triangulated categories of motives 联系起来。$\mathbf{SH}(S)$ 是更大的稳定同伦范畴，motives 可视为其中由 motivic cohomology 控制的线性化部分。

## 40.7 Compact generation 与 realization

**外部输入定理 40.18.** 在常见有限性假设下，$\mathbf{SH}(S)$ compactly generated，其紧生成子由 smooth schemes 的悬挂谱及 Tate twists 给出。

**命题 40.19（紧生成子检测）.** 设 $C$ 是 compactly generated stable $\infty$-category，$\mathcal G\subseteq C^\omega$ 是一组紧生成子。则对象 $X\in C$ 为零当且仅当

$$
\operatorname{Map}_C(\Sigma^mG,X)\simeq *
$$

对所有 $G\in\mathcal G$ 与 $m\in\mathbb Z$ 成立。因此态射 $u:X\to Y$ 是等价，当且仅当

$$
\operatorname{Map}_C(\Sigma^mG,u)
$$

对所有 $G,m$ 为等价。

**证明.** 令

$$
\mathcal L_X=\{A\in C\mid \operatorname{Map}_C(A,X)\simeq *\}.
$$

对固定的 $X$，$\mathcal L_X$ 对平移、余纤维和小余极限封闭，因为

$$
\operatorname{Map}_C(\operatorname*{colim} A_i,X)\simeq
\lim\operatorname{Map}_C(A_i,X).
$$

所以 $\mathcal L_X$ 是 localizing subcategory。若 $\operatorname{Map}_C(\Sigma^mG,X)\simeq *$ 对所有 $G,m$ 成立，则 $\mathcal L_X$ 含所有生成子及其平移，因而由生成性知 $\mathcal L_X=C$。特别地 $X\in\mathcal L_X$，于是 $\operatorname{Map}_C(X,X)$ 可缩，恒等态射同伦于零，故 $X\simeq0$。反向显然。对态射 $u$，令 $F=\operatorname{fib}(u)$；$u$ 为等价当且仅当 $F\simeq0$，再应用对象判别即可。$\square$

**注 40.20.** 某个 realization functor 是否保守并不能仅由 compact generation 推出；它要求其核中没有非零对象，是额外的几何或同伦论信息。

## 40.8 本章小结

Motivic homotopy theory 从光滑概形上的 space-valued sheaves 出发，先施加 Nisnevich descent，再施加 $\mathbb A^1$-invariance。稳定化后得到 $\mathbf{SH}(S)$，它既像谱范畴，又保留代数几何的 Tate 方向和六操作。Purity、Thom spaces、motivic cohomology、realization functors 和 compact generation 共同构成现代 motivic homotopy 的范畴论基础。

## 练习

**练习 40.1.** 定义 $\operatorname{Sm}_S$ 和 $\mathcal P(\operatorname{Sm}_S)$。

**练习 40.2.** 定义 Nisnevich sheaves of spaces。

**练习 40.3.** 定义 motivic spaces $\mathbf H(S)$。

**练习 40.4.** 证明 $\mathbf H(S)$ presentable。

**练习 40.5.** 定义 $\mathbb A^1$-invariant sheaf。

**练习 40.6.** 证明 motivic spaces 可识别为 $\mathbb A^1$-invariant Nisnevich sheaves。

**练习 40.7.** 定义 Tate sphere $T$。

**练习 40.8.** 定义 $\mathbf{SH}(S)$。

**练习 40.9.** 说明复 realization 如何由泛性质诱导。

**练习 40.10.** 陈述 motivic 六操作。

**练习 40.11.** 写出 motivic localization triangle。

**练习 40.12.** 陈述 homotopy purity。

**练习 40.13.** 证明 $\operatorname{Th}(0_X)\simeq X_+$。

**练习 40.14.** 说明 $\mathbf{SH}(S)$ 的 compact generation 如何用于检测等价。
