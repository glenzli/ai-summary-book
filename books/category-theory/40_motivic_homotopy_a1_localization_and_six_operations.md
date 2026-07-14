# 第四十章：Motivic homotopy、$\mathbb A^1$-局部化与六操作

代数簇上的同伦论必须同时尊重 Grothendieck 拓扑和代数同伦 $X\times\mathbb A^1\to X$。Morel--Voevodsky 构造先在 $\mathrm{Sm}_S$ 上取空间值 sheaves，再作 $\mathbb A^1$-局部化，最后相对于 $\mathbb P^1$ 或 Tate sphere 稳定化得到 $\mathbf{SH}(S)$。这个顺序决定了 motivic sphere 的双重分次，也为基变换、纯性与六操作提供环境。本章只建立其范畴论骨架，并把深层几何定理标为外部输入。

背景包括站点、$\infty$-topos、可达局部化、稳定化与六操作。本章固定采用 Nisnevich 口径，并逐项声明基概形条件以及 $S^1$ 与 $\mathbb G_m$ 稳定化的关系；不把不同 site 或 effective/stable 范畴的结论直接互换。

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

**外部输入定理 40.11.** 在 Noetherian、有限 Krull 维基概形组成的范畴上，$\mathbf{SH}(-)$ 形成六操作形式主义。任意态射 $f:X\to Y$ 有 $f^*\dashv f_*$；若 $f$ separated 且 finite type，则另有 $f_!\dashv f^!$。连同

$$
f^*,\quad f_*,\quad f_!,\quad f^!,\quad -\otimes-,\quad \underline{\operatorname{Hom}}
$$

这些操作满足适当的基变换、投影公式与 localization triangles；若 $f$ proper，则 $f_!\simeq f_*$。Purity 另要求光滑态射或 regular closed immersion 等几何条件。

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

**外部输入定理 40.16.** 设 $k$ 为 perfect field，$e$ 为其 characteristic exponent。存在 motivic Eilenberg--Mac Lane spectrum $H\mathbb Z$，并有对称幺半等价

$$
\operatorname{Mod}_{H\mathbb Z[1/e]}(\mathbf{SH}(k))
\simeq
\mathbf{DM}(k,\mathbb Z[1/e]).
$$

特征 $0$ 时 $e=1$，无需反演。对一般基概形，模范畴与 motives 的比较需要另行选择 transfers、系数与基底假设。

**注 40.17.** 这把 stable motivic homotopy theory 和 triangulated categories of motives 联系起来。$\mathbf{SH}(S)$ 是更大的稳定同伦范畴，motives 可视为其中由 motivic cohomology 控制的线性化部分。

## 40.7 Compact generation 与 realization

**外部输入定理 40.18.** 若 $S$ Noetherian 且有限 Krull 维，则 $\mathbf{SH}(S)$ compactly generated；一组紧生成子可取

$$
\Sigma^{p,q}\Sigma^\infty_T X_+,
$$

其中 $X\in\operatorname{Sm}_S$、$p,q\in\mathbb Z$。更一般基底上的紧性需单独验证。

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

所以 $\mathcal L_X$ 是 localizing subcategory。若 $\operatorname{Map}_C(\Sigma^mG,X)\simeq *$ 对所有 $G,m$ 成立，则 $\mathcal L_X$ 含所有生成子及其平移，因而由生成性知 $\mathcal L_X=C$。特别地 $X\in\mathcal L_X$，于是 $\operatorname{Map}_C(X,X)$ 可缩，恒等态射同伦于零，故 $X\simeq0$。反向由零对象的映射空间可缩性给出。对态射 $u$，令 $F=\operatorname{fib}(u)$；$u$ 为等价当且仅当 $F\simeq0$，再应用对象判别即可。$\square$

**注 40.20.** 某个 realization functor 是否保守并不能仅由 compact generation 推出；它要求其核中没有非零对象，是额外的几何或同伦论信息。

## 40.8 从代数同伦到 motivic spectra

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
