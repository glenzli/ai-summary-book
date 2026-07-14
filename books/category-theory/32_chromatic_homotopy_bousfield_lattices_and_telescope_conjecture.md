# 第三十二章：Chromatic homotopy、Bousfield lattice 与 telescope conjecture

一个谱可能被某种同调理论完全看不见；Bousfield class 只记录这种消失信息，并按可见性组成格。固定素数后，Morava $K(n)$ 把稳定同伦范畴按形式群高度分层，Johnson--Wilson 理论和 chromatic tower 则逐级重建有限谱。历史上的 telescope conjecture 比较由有限谱产生的 telescopic 局部化与 Johnson--Wilson 局部化；它在高度 $0,1$ 成立，而在每个素数的所有高度 $n\ge2$ 均已知不成立。本章把这些问题写成稳定 presentable 范畴中的 localization 陈述。

读者需要谱、smash product、Bousfield localization、紧生成和 smashing localization。Nilpotence、periodicity 与 chromatic convergence 作为精确标记的外部输入；telescope conjecture 会作为历史命题连同其反例定理陈述。

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

**外部输入定理 32.5.** 固定素数 $p$。零高度 Morava 理论为有理同调理论

$$
K(0)=H\mathbb Q,\qquad K(0)_*=\mathbb Q.
$$

对 $n\ge1$，存在 Morava $K$-theories $K(n)$，其系数环为

$$
K(n)_*\cong\mathbb F_p[v_n^{\pm1}],\qquad |v_n|=2(p^n-1),
$$

并且 $K(n)$ 是 graded field spectrum：$K(n)_*X$ 是 $K(n)_*$-向量空间。

**定义 32.6.** 谱 $X$ 的 chromatic support 定义为

$$
\operatorname{supp}_{chr}(X)=\{n\ge0\mid K(n)\wedge X\not\simeq0\}.
$$

若该集合有最大元，可把它称为 $X$ 的最大 chromatic height；一般谱不必有单一有限高度。

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

使得 $K(n)_*(v)$ 为同构，且对每个 $m\ne n$，$K(m)_*(v)$ 为幂零自同态。其 telescope 定义为

$$
T(F,v)=\operatorname{colim}(F\xrightarrow{\Sigma^{-d}v}\Sigma^{-d}F\to\Sigma^{-2d}F\to\cdots).
$$

**外部输入定理 32.10（周期性定理）.** 对任意 type $n$ 有限 $p$-local 谱，存在 $v_n$-self map，且不同选择的 telescope 在 Bousfield class 上只依赖 $n$。记该类为 $\langle T(n)\rangle$。

**定义 32.11.** 令

$$
L_n=L_{E(n)}
$$

为 Johnson--Wilson $E(n)$-局部化。令 $L_n^f$ 为远离 type $n+1$ 有限谱厚子范畴的有限局部化；周期性定理给出

$$
L_n^f\simeq L_{T(0)\vee\cdots\vee T(n)}.
$$

因此 $L_n^f$ 是 telescopic 局部化，而 $L_n$ 是 chromatic 局部化。

**历史猜想 32.12（Telescope conjecture）.** 高度 $n$ 的 telescope conjecture 断言自然比较

$$
L_n^fX\longrightarrow L_nX
$$

对所有 $p$-局部谱 $X$ 都是等价。等价地，它断言

$$
\langle T(n)\rangle=\langle K(n)\rangle.
$$

**外部输入定理 32.13（当前状态）.** Telescope conjecture 在高度 $0$ 与 $1$ 成立。Burklund--Hahn--Levy--Schlank 证明：对每个素数 $p$ 及每个 $n\ge2$，存在 $p$-局部谱 $X$ 使

$$
L_n^fX\not\simeq L_nX;
$$

因而历史猜想在所有这些高度均不成立。无条件成立的是定义 32.11 的 telescopic 检测公式；错误的是把它进一步等同于 $E(n)$-局部化。

## 32.4 Chromatic fracture squares

**外部输入定理 32.14.** 对每个 $p$-局部谱 $X$ 与 $n\ge1$，存在 chromatic fracture square，把 $L_nX$ 由 $L_{n-1}X$ 与 $K(n)$-local 信息粘合：

$$
\begin{array}{c}
L_nX\to L_{K(n)}X\\
\downarrow\quad\downarrow\\
L_{n-1}X\to L_{n-1}L_{K(n)}X.
\end{array}
$$

该方块为 homotopy pullback。

**命题 32.15.** 若 chromatic fracture square 是拉回方块，则 $L_nX\simeq0$ 当且仅当 $L_{n-1}X\simeq0$ 且 $L_{K(n)}X\simeq0$。

**证明.** 局部化之间有自然等价

$$
L_{n-1}L_n\simeq L_{n-1},\qquad
L_{K(n)}L_n\simeq L_{K(n)}.
$$

因此 $L_nX\simeq0$ 蕴含另外两个局部化均为零。反过来，若 $L_{n-1}X\simeq0$ 且 $L_{K(n)}X\simeq0$，则右下角也为零；定理 32.14 的拉回为 $0\times_0 0\simeq0$，故 $L_nX\simeq0$。注意这里只由“拉回方块”本身不能推出正向结论，正向还使用了上述局部化复合公式。$\square$

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

## 32.7 按高度观察稳定同伦

Chromatic homotopy theory 把谱按 Morava $K(n)$ 的高度分层。Bousfield lattice 记录不同同调理论的检测能力；厚子范畴定理说明有限 $p$-local spectra 的 thick subcategories 由高度分类；telescopic 局部化由显式周期谱给出，而 telescope conjecture 曾断言它与 $E(n)$-局部化相同，现知该比较在所有高度 $n\ge2$ 失败；fracture square 则把高度 $n$ 信息由低高度和 $K(n)$-local 部分粘合。

## 练习

**练习 32.1.** 定义谱 $E$ 的 Bousfield class。

**练习 32.2.** 证明若 $E\simeq F\wedge G$，则 $\langle E\rangle\le\langle F\rangle$。

**练习 32.3.** 证明楔和给出 Bousfield classes 的 join。

**练习 32.4.** 写出 Morava $K(n)$ 的系数环。

**练习 32.5.** 定义有限 $p$-local 谱的 chromatic type。

**练习 32.6.** 陈述厚子范畴定理。

**练习 32.7.** 定义 $v_n$-self map 和 telescope。

**练习 32.8.** 陈述历史上的 telescope conjecture，并说明其当前已知状态。

**练习 32.9.** 区分无条件等价 $L_n^f\simeq L_{T(0)\vee\cdots\vee T(n)}$ 与历史猜想 $L_n^f\simeq L_n$。

**练习 32.10.** 写出 chromatic fracture square。

**练习 32.11.** 用 fracture square 证明零对象检测。

**练习 32.12.** 说明 chromatic localization 是 Bousfield localization 的特例。

**练习 32.13.** 解释 smashing localization 与 compact objects 商定理的关系。

**练习 32.14.** 定义 $E$-equivalence，并证明 Bousfield equivalent 的谱给出相同的局部等价类。

**练习 32.15.** 证明若 $\langle E\rangle\le\langle F\rangle$，则每个 $F$-equivalence 都是 $E$-equivalence。
