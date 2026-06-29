# 第三十二章：Chromatic homotopy、Bousfield lattice 与 telescope conjecture

## 本章目标

本章扩展第二十六章的 Bousfield localization 到稳定同伦论的 chromatic 分层。Chromatic homotopy theory 用 Morava $K$-theories、Johnson-Wilson theories 和局部化塔把谱按高度分解；Bousfield lattice 记录同调理论对谱的“可见性”；telescope conjecture 则询问有限型局部化与显式 telescope 局部化是否一致。

## 依赖前置知识

需要谱、环谱、smash product、Bousfield localization、compact generation、localizing subcategories、smashing localization、Morita 型不变量和稳定 $\infty$-范畴的基本语言。

## 32.1 Bousfield 类与格结构

**定义 32.1.** 对谱 $E$，其 Bousfield class 定义为

$$
\langle E\rangle=\{X\in\mathbf{Sp}\mid E\wedge X\simeq0\}.
$$

若 $\langle E\rangle=\langle F\rangle$，则称 $E$ 与 $F$ Bousfield equivalent。

**定义 32.2.** Bousfield classes 上定义偏序

$$
\langle E\rangle\le \langle F\rangle
$$

若 $F$-acyclic 的谱必为 $E$-acyclic，即

$$
F\wedge X\simeq0\Rightarrow E\wedge X\simeq0.
$$

直觉上，$\langle E\rangle\le\langle F\rangle$ 表示 $E$ 看见的信息不多于 $F$。

**命题 32.3.** 若存在谱 $G$ 使 $E\simeq F\wedge G$，则 $\langle E\rangle\le\langle F\rangle$。

**证明.** 若 $F\wedge X\simeq0$，则

$$
E\wedge X\simeq F\wedge G\wedge X\simeq G\wedge F\wedge X\simeq0.
$$

故每个 $F$-acyclic 都是 $E$-acyclic，得到 $\langle E\rangle\le\langle F\rangle$。$\square$

**命题 32.4.** 对任意族 $\{E_i\}$，楔和 $\bigvee_iE_i$ 的 Bousfield class 满足

$$
\left\langle\bigvee_iE_i\right\rangle
$$

是这些 $\langle E_i\rangle$ 的 join。

**证明.** 对任意 $X$，

$$
\left(\bigvee_iE_i\right)\wedge X\simeq\bigvee_i(E_i\wedge X).
$$

该谱为零当且仅当所有 $E_i\wedge X$ 为零。因此 $\bigvee_iE_i$-acyclics 是所有 $E_i$-acyclics 的交。按定义 32.2，这给出最小的共同上界，即 join。$\square$

## 32.2 Morava $K$-theory 与高度

**外部输入定理 32.5.** 固定素数 $p$。存在 Morava $K$-theories $K(n)$，$n\ge0$，其系数环为

$$
K(n)_*\cong\mathbb F_p[v_n^{\pm1}],\qquad |v_n|=2(p^n-1),
$$

并且 $K(n)$ 是 graded field spectrum：$K(n)_*X$ 是 $K(n)_*$-向量空间。

**定义 32.6.** 谱 $X$ 的 chromatic height 信息由哪些 $K(n)\wedge X$ 非零记录。若 $K(n)\wedge X\simeq0$ 对所有 $n>m$，则称 $X$ 在高度 $>m$ 上不可见。

**外部输入定理 32.7（厚子范畴定理）.** 在 $p$-local finite spectra 的稳定同伦范畴中，thick subcategories 由 chromatic type 分类。具体地，有限谱 $F$ 的 type 至少为 $n$ 当且仅当

$$
K(0)_*F=\cdots=K(n-1)_*F=0.
$$

有限谱的厚子范畴形成按 type 线性排列的链。

**命题 32.8.** 若有限 $p$-local 谱 $F$ 的 type 为 $n$，则 $K(m)\wedge F\simeq0$ 对 $m<n$，且 $K(n)\wedge F\not\simeq0$。

**证明.** 这是 type 定义的直接翻译。$K(m)_*F=0$ 等价于 $K(m)\wedge F\simeq0$，因为 $K(m)$-模谱由其同伦群检测零对象。Type 为 $n$ 表示 $n$ 是第一个非零 Morava $K$-theory 高度。$\square$

## 32.3 有限局部化与 telescope

**定义 32.9.** 设 $F$ 是 type $n$ 有限谱。一个 $v_n$-self map 是映射

$$
v:\Sigma^dF\to F
$$

使得 $K(n)_*(v)$ 为同构，且 $K(m)_*(v)$ 对 $m\ne n$ 为幂零或零的相应形式。其 telescope 定义为

$$
T(F,v)=\operatorname{colim}(F\xrightarrow{\Sigma^{-d}v}\Sigma^{-d}F\to\Sigma^{-2d}F\to\cdots).
$$

**外部输入定理 32.10（周期性定理）.** 对任意 type $n$ 有限 $p$-local 谱，存在 $v_n$-self map，且不同选择的 telescope 在 Bousfield class 上只依赖 $n$。记该类为 $\langle T(n)\rangle$。

**定义 32.11.** 设 $L_n^f$ 为有限局部化，即以 type $n+1$ 有限谱的局部化补定义的 smashing localization。设 $L_{T(n)}$ 为关于 telescope $T(n)$ 的 Bousfield localization。

**外部输入猜想 32.12（Telescope conjecture）.** 对每个 $n$ 和素数 $p$，自然比较

$$
L_n^f\to L_{T(0)\vee\cdots\vee T(n)}
$$

或等价形式中的相应局部化应为等价。该猜想在低高度成立，在一般高度是 chromatic homotopy theory 的核心问题之一。

**命题 32.13.** 若 telescope conjecture 在高度 $n$ 成立，则对应的有限局部化由显式周期谱 $T(0),\dots,T(n)$ 检测。

**证明.** 猜想给出 $L_n^f$ 与 $T(0)\vee\cdots\vee T(n)$-localization 的等价。Bousfield localization 由其 acyclic objects 决定；后者由

$$
(T(0)\vee\cdots\vee T(n))\wedge X\simeq0
$$

检测，也即所有 $T(i)\wedge X$ 为零检测。因此有限局部化可由这些 telescope 谱显式检测。$\square$

## 32.4 Chromatic fracture squares

**外部输入定理 32.14.** 对适当谱 $X$，存在 chromatic fracture square，把 $L_nX$ 由 $L_{n-1}X$ 与 $K(n)$-local 信息粘合：

$$
\begin{array}{c}
L_nX\to L_{K(n)}X\\
\downarrow\quad\downarrow\\
L_{n-1}X\to L_{n-1}L_{K(n)}X.
\end{array}
$$

该方块为 homotopy pullback。

**命题 32.15.** 若 chromatic fracture square 是拉回方块，则 $L_nX\simeq0$ 当且仅当 $L_{n-1}X\simeq0$ 且 $L_{K(n)}X\simeq0$。

**证明.** 拉回方块中，若左上角为零，则两个投影对象为零。反过来，若 $L_{n-1}X\simeq0$ 且 $L_{K(n)}X\simeq0$，则右下角 $L_{n-1}L_{K(n)}X$ 也为零，拉回为 $0\times_0 0\simeq0$，故 $L_nX\simeq0$。$\square$

## 32.5 与范畴论结构的关系

**命题 32.16.** Chromatic localization 是第二十六章 Bousfield localization 的特例。

**证明.** 每个同调理论或谱 $E$ 给出 $E$-acyclic objects

$$
\{X\mid E\wedge X\simeq0\}.
$$

若相应 accessible localization 存在，则局部对象和局部化函子正是第二十六章定义的 Bousfield localization。取 $E=K(n)$、$E(n)$、$T(n)$ 或它们的楔和，即得到 chromatic localization。$\square$

**命题 32.17.** Smashing chromatic localization 与 compact objects 的行为受 Neeman-Thomason 型定理约束。

**证明.** Smashing localization $L$ 保持小余极限，并由 acyclic localizing subcategory $A$ 控制。若 $A$ 由 compact objects 生成，则第二十六章的 Neeman-Thomason 型定理给出 quotient compact objects 的描述：

$$
(\mathbf{Sp}/A)^\omega\simeq\operatorname{Kar}(\mathbf{Sp}^\omega/A^\omega).
$$

因此有限谱层面的 thick subcategories 与大稳定范畴的 Bousfield localization 相互约束。$\square$

## 32.6 Bousfield 类决定的局部化

**定义 32.18.** 态射 $f:X\to Y$ 称为 $E$-equivalence，若其余纤维满足

$$
E\wedge\operatorname{cofib}(f)\simeq0.
$$

换言之，$f$ 的误差对象为 $E$-acyclic。

**命题 32.19.** 若 $\langle E\rangle=\langle F\rangle$，则 $E$-equivalences 与 $F$-equivalences 相同。特别地，若相应 Bousfield localizations 存在，则它们有相同的局部等价类和相同的 acyclic objects，因而表示同一个局部化问题。

**证明.** 对任意态射 $f$，

$$
f\text{ 是 }E\text{-equivalence}
\iff
\operatorname{cofib}(f)\in\langle E\rangle.
$$

若 $\langle E\rangle=\langle F\rangle$，右侧条件等价于 $\operatorname{cofib}(f)\in\langle F\rangle$，即 $f$ 是 $F$-equivalence。Bousfield localization 的核和局部等价类由这些 acyclic cofibers 决定，因此两者表示同一局部化问题。$\square$

**命题 32.20.** 若 $\langle E\rangle\le\langle F\rangle$，则每个 $F$-equivalence 都是 $E$-equivalence。

**证明.** 设 $f:X\to Y$ 是 $F$-equivalence，则

$$
F\wedge\operatorname{cofib}(f)\simeq0.
$$

由 $\langle E\rangle\le\langle F\rangle$ 的定义，每个 $F$-acyclic 对象都是 $E$-acyclic，故

$$
E\wedge\operatorname{cofib}(f)\simeq0.
$$

因此 $f$ 是 $E$-equivalence。$\square$

## 32.7 本章小结

Chromatic homotopy theory 把谱按 Morava $K(n)$ 的高度分层。Bousfield lattice 记录不同同调理论的检测能力；厚子范畴定理说明有限 $p$-local spectra 的 thick subcategories 由高度分类；telescope conjecture 询问有限局部化能否由显式周期 telescope 谱给出；fracture square 则把高度 $n$ 信息由低高度和 $K(n)$-local 部分粘合。所有这些都是稳定 presentable $\infty$-范畴中 Bousfield localization、compact generation 和 recollement 思想的深层例子。

## 练习

**练习 32.1.** 定义谱 $E$ 的 Bousfield class。

**练习 32.2.** 证明若 $E\simeq F\wedge G$，则 $\langle E\rangle\le\langle F\rangle$。

**练习 32.3.** 证明楔和给出 Bousfield classes 的 join。

**练习 32.4.** 写出 Morava $K(n)$ 的系数环。

**练习 32.5.** 定义有限 $p$-local 谱的 chromatic type。

**练习 32.6.** 陈述厚子范畴定理。

**练习 32.7.** 定义 $v_n$-self map 和 telescope。

**练习 32.8.** 陈述 telescope conjecture。

**练习 32.9.** 解释 telescope conjecture 成立时有限局部化如何由 $T(i)$ 检测。

**练习 32.10.** 写出 chromatic fracture square。

**练习 32.11.** 用 fracture square 证明零对象检测。

**练习 32.12.** 说明 chromatic localization 是 Bousfield localization 的特例。

**练习 32.13.** 解释 smashing localization 与 compact objects 商定理的关系。

**练习 32.14.** 定义 $E$-equivalence，并证明 Bousfield equivalent 的谱给出相同的局部等价类。

**练习 32.15.** 证明若 $\langle E\rangle\le\langle F\rangle$，则每个 $F$-equivalence 都是 $E$-equivalence。
