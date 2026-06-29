# 第三十四章：导出代数几何、cotangent complex 与 spectral stacks

## 本章目标

本章给出 derived algebraic geometry 与 spectral algebraic geometry 的范畴论入口。导出几何把交换环替换为 simplicial commutative rings、commutative dg algebras 或 $E_\infty$-rings；几何对象变为满足下降的 $\infty$-functors。Cotangent complex 控制变形理论；QCoh 和 IndCoh 形成稳定 presentable $\infty$-范畴值 sheaf；formal moduli problems 把局部变形与 dg Lie algebras 或谱 Lie algebras 联系起来。

## 依赖前置知识

需要可表现 $\infty$-范畴、$\infty$-topos、稳定 $\infty$-范畴、六操作、dg 范畴、spectra、$E_\infty$-rings、Cartesian fibration 和 sheaf descent。

## 34.1 派生环与仿射对象

**定义 34.1.** 派生交换环可以用下列等价或相近模型之一表示：

1. simplicial commutative rings；
2. 非正 cohomological commutative dg algebras，在特征 $0$ 语境中；
3. connective $E_\infty$-rings。

本章用 $\operatorname{CAlg}^{cn}$ 表示 connective $E_\infty$-rings 的 $\infty$-范畴。

**定义 34.2.** 派生仿射概形定义为

$$
\operatorname{Spec}A
$$

其中 $A\in\operatorname{CAlg}^{cn}$。仿射对象组成的 $\infty$-范畴为

$$
\operatorname{dAff}=(\operatorname{CAlg}^{cn})^{op}.
$$

**命题 34.3.** 普通交换环 $R$ 给出离散 $E_\infty$-ring，因此普通仿射概形嵌入派生仿射概形。

**证明.** 把 $R$ 看成只在 $\pi_0$ 非零的 Eilenberg-Mac Lane $E_\infty$-ring $HR$。映射空间从离散环到离散环时退化为普通环同态集合；因此 $R\mapsto HR$ 全忠实地把普通交换环嵌入 connective $E_\infty$-rings 的离散部分。取反范畴得到普通仿射概形到派生仿射概形的嵌入。$\square$

## 34.2 Derived stacks 与下降

**定义 34.4.** 一个 prestack 是函子

$$
F:\operatorname{dAff}^{op}\to\mathcal S.
$$

给定 Grothendieck topology $\tau$，若 $F$ 对 $\tau$-覆盖满足 hyperdescent，则称 $F$ 为 derived stack。

**定义 34.5.** 派生仿射 $\operatorname{Spec}A$ 的 functor of points 为

$$
\operatorname{Map}_{\operatorname{CAlg}^{cn}}(A,-):\operatorname{CAlg}^{cn}\to\mathcal S.
$$

**命题 34.6.** 派生仿射对象满足 Yoneda 型完全忠实性：

$$
\operatorname{Map}_{\operatorname{dAff}}(\operatorname{Spec}B,\operatorname{Spec}A)
\simeq
\operatorname{Map}_{\operatorname{CAlg}^{cn}}(A,B).
$$

**证明.** 这是反范畴定义的直接结果。$\operatorname{dAff}=(\operatorname{CAlg}^{cn})^{op}$，所以左边按定义等于右边。作为 prestack 的完全忠实性进一步由 $\infty$-Yoneda 引理给出。$\square$

**外部输入定理 34.7.** 在合适拓扑下，derived Artin stacks、derived Deligne-Mumford stacks 或 spectral stacks 可由 representability 条件刻画，包括对角线可表示性、光滑或 étale atlas、截断条件和 deformation-theoretic 条件。

## 34.3 QCoh 与 perfect complexes

**定义 34.8.** 对派生仿射 $\operatorname{Spec}A$，定义

$$
\operatorname{QCoh}(\operatorname{Spec}A)=\operatorname{Mod}_A,
$$

即 $A$-module spectra 的稳定 presentable $\infty$-范畴。

**定义 34.9.** 对 derived stack $X$，$\operatorname{QCoh}(X)$ 定义为沿仿射对象映射到 $X$ 的 $\operatorname{QCoh}$ 的极限：

$$
\operatorname{QCoh}(X)=\lim_{\operatorname{Spec}A\to X}\operatorname{Mod}_A.
$$

**命题 34.10.** 若 $X=\operatorname{Spec}A$，定义 34.9 恢复 $\operatorname{Mod}_A$。

**证明.** 在 overcategory $(\operatorname{dAff}_{/X})$ 中，恒等映射 $\operatorname{Spec}A\to\operatorname{Spec}A$ 是终对象。极限在有终对象的图形上等于终对象处的值，因此

$$
\operatorname{QCoh}(X)\simeq\operatorname{Mod}_A.
$$

$\square$

**定义 34.11.** Perfect complexes 是 $\operatorname{QCoh}(X)$ 中局部为 compact $A$-modules 的对象。记作

$$
\operatorname{Perf}(X)\subseteq\operatorname{QCoh}(X).
$$

**外部输入定理 34.12.** 在 quasi-compact quasi-separated 等合理假设下，$\operatorname{QCoh}(X)$ 是 compactly generated stable presentable $\infty$-category，且 compact objects 与 perfect complexes 一致或在精确假设下相容。

## 34.4 Cotangent complex

**定义 34.13.** 对 $A\in\operatorname{CAlg}^{cn}$，$A$-module $L_A$ 称为 cotangent complex，若对任意 $A$-module $M$，有自然等价

$$
\operatorname{Map}_{\operatorname{Mod}_A}(L_A,M)
\simeq
\operatorname{Der}(A,M),
$$

其中右边为从 $A$ 到 square-zero extension $A\oplus M$ 的导子空间。

**定义 34.14.** 对映射 $A\to B$，relative cotangent complex $L_{B/A}$ 表示 $A$-线性导子：

$$
\operatorname{Map}_{\operatorname{Mod}_B}(L_{B/A},M)
\simeq
\operatorname{Der}_A(B,M).
$$

**命题 34.15（transitivity triangle）.** 对 $A\to B\to C$，存在自然余纤维序列

$$
C\otimes_B L_{B/A}\to L_{C/A}\to L_{C/B}.
$$

**证明.** 对任意 $C$-module $M$，映射出该余纤维序列应给出纤维序列

$$
\operatorname{Map}(L_{C/B},M)\to
\operatorname{Map}(L_{C/A},M)\to
\operatorname{Map}(C\otimes_BL_{B/A},M).
$$

由 cotangent complex 的表示性，这分别是

$$
\operatorname{Der}_B(C,M)\to\operatorname{Der}_A(C,M)\to\operatorname{Der}_A(B,M)
$$

的限制序列。一个 $A$-导子在 $C$ 上为 $B$-线性，当且仅当它在 $B$ 上为零，因此该序列是纤维序列。由稳定范畴 Yoneda，得到余纤维序列。$\square$

## 34.5 Formal moduli problems

**定义 34.16.** 设 $k$ 为域。一个 formal moduli problem 是定义在 Artinian augmented derived $k$-algebras 上的函子

$$
F:\operatorname{Art}_k\to\mathcal S
$$

满足 $F(k)\simeq *$，并把小拉回方块送为拉回方块。

**外部输入定理 34.17（Lurie-Pridham）.** 在特征 $0$ 下，formal moduli problems 的 $\infty$-范畴等价于 dg Lie algebras 的合适 $\infty$-范畴。谱版本中由 spectral Lie algebras 或相应 Koszul dual objects 控制。

**命题 34.18.** 若 $X$ 是 derived stack，点 $x:\operatorname{Spec}k\to X$ 的切复形可由 cotangent complex 对偶给出：

$$
T_xX\simeq\operatorname{Map}_k(x^*L_X,k).
$$

**证明.** 点处一阶变形由 square-zero extension $k\oplus M$ 上的 lift 控制。Cotangent complex 的表示性给出变形空间

$$
\operatorname{Map}_{\operatorname{Mod}_k}(x^*L_X,M).
$$

取 $M=k$ 或让 $M$ 变量化，得到切对象为 $x^*L_X$ 的线性对偶。$\square$

## 34.6 IndCoh 与奇异支撑入口

**定义 34.19.** 对足够好的 derived stack $X$，$\operatorname{IndCoh}(X)$ 是 coherent sheaves 范畴的 Ind 完备化型增强。它与 $\operatorname{QCoh}(X)$ 在光滑情形接近，但在奇异或非光滑情形更适合表达 Grothendieck duality。

**外部输入定理 34.20.** 对 quasi-smooth derived schemes 或 stacks，可定义 singularity space $\operatorname{Sing}(X)$ 和对象的 singular support。带有指定奇异支撑条件的 IndCoh 子范畴

$$
\operatorname{IndCoh}_{\mathcal N}(X)
$$

在几何表示论和 Langlands 型理论中发挥核心作用。

**注 34.21.** $\operatorname{QCoh}$ 更像函数；$\operatorname{IndCoh}$ 更像分布。二者通过 dualizing sheaf、! pullback 和 Grothendieck duality 相连。

## 34.7 仿射计算与一阶形式后果

**命题 34.22（仿射拉回）.** 给定 connective $E_\infty$-rings 的图形

$$
B\to A,\qquad B\to C,
$$

在 $\operatorname{dAff}$ 中有自然等价

$$
\operatorname{Spec}A\times_{\operatorname{Spec}B}\operatorname{Spec}C
\simeq
\operatorname{Spec}(A\otimes_B C).
$$

**证明.** 对任意测试对象 $\operatorname{Spec}T$，由反范畴定义与 $\operatorname{CAlg}^{cn}$ 中张量积的推出泛性质，

$$
\begin{aligned}
\operatorname{Map}_{\operatorname{dAff}}(\operatorname{Spec}T,\operatorname{Spec}(A\otimes_BC))
&\simeq\operatorname{Map}_{\operatorname{CAlg}^{cn}}(A\otimes_BC,T)\\
&\simeq
\operatorname{Map}_{\operatorname{CAlg}^{cn}}(A,T)
\times_{\operatorname{Map}_{\operatorname{CAlg}^{cn}}(B,T)}
\operatorname{Map}_{\operatorname{CAlg}^{cn}}(C,T).
\end{aligned}
$$

右端正是从 $\operatorname{Spec}T$ 到拉回

$$
\operatorname{Spec}A\times_{\operatorname{Spec}B}\operatorname{Spec}C
$$

的映射空间。由 $\infty$-Yoneda 引理得到结论。$\square$

**命题 34.23.** 对映射 $A\to B$，$L_{B/A}\simeq0$ 当且仅当对所有 $B$-module $M$，导子空间 $\operatorname{Der}_A(B,M)$ 可缩。

**证明.** 由定义 34.14，

$$
\operatorname{Der}_A(B,M)\simeq
\operatorname{Map}_{\operatorname{Mod}_B}(L_{B/A},M).
$$

若 $L_{B/A}\simeq0$，则右侧为从零对象到 $M$ 的映射空间，故可缩。反过来，若这些映射空间对所有 $M$ 可缩，则 $L_{B/A}$ 与零对象表示同一函子；由稳定 $\infty$-范畴的 Yoneda 判别，$L_{B/A}\simeq0$。$\square$

**命题 34.24（切映射的形式来源）.** 设 $f:X\to Y$ 为具有 cotangent complexes 的 derived stacks 间态射，$x:\operatorname{Spec}k\to X$ 为点，$y=f\circ x$。Cotangent complexes 的函子性给出映射

$$
x^*f^*L_Y\simeq y^*L_Y\to x^*L_X,
$$

取线性对偶得到切复形映射

$$
T_xX\to T_yY.
$$

**证明.** $Y$ 上的一阶变形沿 $f$ 拉回为 $X$ 上的一阶变形。按 cotangent complex 的表示性，这一拉回自然变换由 $x^*f^*L_Y\to x^*L_X$ 表示。对 $k$-module 取映射到 $k$ 的内部 Hom，即得到对偶方向的切复形映射。$\square$

## 34.8 本章小结

Derived algebraic geometry 把仿射概形替换为 connective $E_\infty$-rings 的反范畴，把几何对象看作满足下降的 functor of points。$\operatorname{QCoh}$ 是稳定 presentable $\infty$-范畴值 sheaf；cotangent complex 用表示性刻画导子并控制变形；formal moduli problems 把局部变形理论与 Lie 型代数对象联系起来；IndCoh 和 singular support 则为奇异几何和表示论提供更精细的范畴工具。

## 练习

**练习 34.1.** 列出派生交换环的三个常用模型。

**练习 34.2.** 定义 derived affine scheme。

**练习 34.3.** 证明普通仿射概形嵌入派生仿射概形。

**练习 34.4.** 定义 prestack 与 derived stack。

**练习 34.5.** 证明派生仿射对象的映射空间公式。

**练习 34.6.** 定义 $\operatorname{QCoh}(\operatorname{Spec}A)$。

**练习 34.7.** 说明 $\operatorname{QCoh}(X)$ 如何由仿射图形极限定义。

**练习 34.8.** 证明 $X=\operatorname{Spec}A$ 时 $\operatorname{QCoh}(X)\simeq\operatorname{Mod}_A$。

**练习 34.9.** 定义 cotangent complex 的表示性。

**练习 34.10.** 证明 transitivity triangle。

**练习 34.11.** 定义 formal moduli problem。

**练习 34.12.** 陈述 Lurie-Pridham 定理。

**练习 34.13.** 说明点处切复形如何由 cotangent complex 对偶给出。

**练习 34.14.** 比较 $\operatorname{QCoh}$ 与 $\operatorname{IndCoh}$ 的用途。

**练习 34.15.** 证明派生仿射对象的拉回公式
$$
\operatorname{Spec}A\times_{\operatorname{Spec}B}\operatorname{Spec}C\simeq\operatorname{Spec}(A\otimes_BC).
$$

**练习 34.16.** 证明 $L_{B/A}\simeq0$ 等价于所有 $A$-线性导子空间 $\operatorname{Der}_A(B,M)$ 可缩。

**练习 34.17.** 说明态射 $X\to Y$ 如何诱导点处切复形的映射。
