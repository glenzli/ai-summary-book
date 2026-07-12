# 第五章：Chromatic localization、fracture 与 convergence

## 本章目标

本章把 $K(0),K(1),\ldots,K(n)$ 组装成 chromatic tower。核心对象是 $L_n=L_{E(n)}$、monochromatic layer $M_n$、chromatic fracture square 和 chromatic convergence theorem。

## 依赖前置知识

需要第一章的 Bousfield localization、有限谱 dualizability，第二章的
$E(n)$ 和第三、四章的 $K(n)$ 与 finite type。以下四项明确作为外部
输入：$E(n)$ 的 Bousfield 类分解、Hopkins--Ravenel smash product
theorem、chromatic fracture square 和 chromatic convergence theorem。
正文只内部证明这些输入的形式后果。

## 5.1 $E(n)$-localization

**定义 5.1.** Johnson-Wilson theory $E(n)$ 的 Bousfield localization 记为
$$
L_n=L_{E(n)}.
$$
约定 $L_{-1}X=0$。

**外部输入定理 5.2（chromatic Bousfield 类，CHT-P0-08）.** 对每个
$n\ge0$，在 $\mathbf{Sp}_{(p)}$ 中有
$$
\langle E(n)\rangle=\langle K(0)\vee K(1)\vee\cdots\vee K(n)\rangle.
$$
此外，不同高度的 Morava K-theories 正交：
$$
K(i)\otimes K(j)\simeq0\qquad(i\ne j).
$$
第一式可逐级定位到 Lurie, *Chromatic Homotopy Theory*, Lecture 23,
Proposition 2（$E(n)$ 与 $E(n-1)\vee K(n)$ 的 Bousfield 等价）；该讲
Lemma 6 给出 fracture 所需的 $E(n-1)\otimes K(n)\simeq0$。Hovey 的
综述性原始论证见 Corollary 1.12。本书不从系数环重证这些谱层结论。
因此“$L_n$ 保留高度不超过 $n$”只是第一式的解释，不是额外定理。

**外部输入定理 5.2A（Hopkins--Ravenel smash product theorem，
CHT-P0-08A）.** 对每个 $n\ge0$ 和每个 $X\in\mathbf{Sp}_{(p)}$，自然映射
$$
L_n\mathbb S_{(p)}\otimes X\longrightarrow L_nX
$$
是等价。也就是说，$L_n$ 是 smashing localization。来源定位为
Ravenel, *Nilpotence and Periodicity in Stable Homotopy Theory*, Theorem
7.5.6；证明在该书 Chapter 8。这个深定理不能由定义 5.1 或 Landweber
exactness 形式推出。

**命题 5.3（chromatic localization 的嵌套方向）.** 若 $0\le m\le n$，则
$$
\mathcal A_{E(n)}\subseteq\mathcal A_{E(m)}.
$$
因而每个 $L_m$-local 谱都是 $L_n$-local，并有自然等价
$$
L_mL_n\simeq L_m,
\qquad
L_nL_m\simeq L_m.
$$
反向断言“$L_n$-local 推出 $L_m$-local”一般不成立。

**证明.** 若 $A$ 为 $E(n)$-acyclic，则由外部输入 5.2，
$$
K(i)\otimes A\simeq0\qquad(0\le i\le n).
$$
特别地，上式对 $0\le i\le m$ 成立；再用外部输入 5.2 得
$E(m)\otimes A\simeq0$。这证明 acyclic 类包含。其余结论逐项应用命题
1.14D。证毕。

**反例 5.3A（local 方向不可逆）.** 对 $n\ge1$，谱 $K(n)$ 是
$L_n$-local，但不是 $L_{n-1}$-local。

**证明.** $K(n)$ 是自身的 module。若 $A$ 为 $K(n)$-acyclic，则谱化的
自由--遗忘伴随给出 function spectra 的等价
$$
F(A,K(n))\simeq
F_{\operatorname{Mod}_{K(n)}}(K(n)\otimes A,K(n))\simeq0,
$$
故 $K(n)$ 为 $K(n)$-local。每个 $E(n)$-acyclic 谱都是 $K(n)$-acyclic，
所以 $K(n)$ 也 $L_n$-local。另一方面，外部输入 5.2 的正交性给出
$E(n-1)\otimes K(n)\simeq0$。若非零谱 $K(n)$ 同时
$L_{n-1}$-local，它会同时 acyclic 与 local，命题 1.14B 将迫使它为
零，矛盾。证毕。

## 5.2 Monochromatic layer

**定义 5.4.** 第 $n$ 个 monochromatic layer 定义为 fiber
$$
M_nX=\operatorname{fib}(L_nX\to L_{n-1}X).
$$

**命题 5.5.** 对任意 $X$，有 fiber/cofiber 序列
$$
M_nX\to L_nX\to L_{n-1}X.
$$

**证明.** 这是定义 5.4 在稳定 infinity-范畴中的直接展开。稳定范畴中 fiber 序列同时是 cofiber 序列。证毕。

**命题 5.6.** $M_nX$ 是 $L_n$-local。

**证明.** $L_nX$ 是 $L_n$-local。由命题 5.3，$L_{n-1}X$ 也是
$L_n$-local。命题 1.14B 说明 $L_n$-local 对象构成稳定全子范畴，
因此两者之 fiber $M_nX$ 仍 $L_n$-local。证毕。

**警告 5.7.** 对一般 $X$，$M_nX$ 不等同于 $L_{K(n)}X$。前者是
$E(n)$ tower 的 fiber，后者是单一高度 $n$ 的局部化。命题 5.14A
会证明它们对 type 恰为 $n$ 的有限谱相同；该特殊结论依赖有限
dualizability 与 smash product theorem，不能反推一般等同。

## 5.3 Chromatic tower

**定义 5.8.** 谱 $X$ 的 chromatic tower 是反向系统
$$
\cdots\to L_nX\to L_{n-1}X\to\cdots\to L_1X\to L_0X.
$$

**定义 5.8A.** 第 $n$ 个 chromatic acyclization 记为
$$
C_nX=\operatorname{fib}(X\longrightarrow L_nX).
$$
由 tower 的自然性，$\{C_nX\}_{n\ge0}$ 也是反向系统。不要把谱
$C_nX$ 与 thick 子范畴 $\mathcal C_n$ 混淆。

**命题 5.8B（极限误差的形式分解）.** 对任意 $X$，有自然 fiber 序列
$$
\operatorname*{holim}_{n}C_nX\longrightarrow X
\longrightarrow\operatorname*{holim}_{n}L_nX.
$$

**证明.** 把 $X$ 看成常值 tower。每层都有 fiber 序列
$C_nX\to X\to L_nX$；infinity-范畴中的 limits 彼此交换，而稳定范畴
中的 fiber 是有限 limit。逐层 fiber 后再取 homotopy limit，得到所示
fiber 序列。证毕。

**外部基础输入 5.8C（Milnor exact sequence）.** 对任意可数谱 tower
$\{Y_n\}$ 和任意整数 $t$，有自然短正合列
$$
0\longrightarrow\lim_n^1\pi_{t+1}Y_n
\longrightarrow\pi_t\operatorname*{holim}_nY_n
\longrightarrow\lim_n\pi_tY_n\longrightarrow0.
$$
因此要由同伦群 tower 证明 $\operatorname*{holim}_nC_nX\simeq0$，必须
同时控制 $\lim$ 与 $\lim^1$；只证明逐项在越来越多同调理论下消失并
不足够。Ravenel 书 Appendix A.5 的 homotopy-limit 讨论给出本书采用的
版本。

**外部输入定理 5.9（Hopkins--Ravenel chromatic convergence，
CHT-P0-09）.** 若 $X\in\mathbf{Sp}_{(p)}^\omega$，则自然映射
$$
X\longrightarrow \operatorname*{holim}_{n\ge0} L_nX
$$
是等价。

**证明责任与来源边界.** 这是深外部定理，不在本书重证。精确定位为
Ravenel, *Nilpotence and Periodicity in Stable Homotopy Theory*, Theorem
7.5.7；证明在 Section 8.6。该证明研究 $C_nX$，用 BP-based Adams
filtration 证明 transition maps 在固定 stem 中最终具有任意高 filtration，
并同时消去命题 5.8C 中的 $\lim$ 与 $\lim^1$。这段路线只说明来源证明
如何闭合，不算书内证明。

定理原文对 $p$-local finite CW-spectrum 陈述；本书的
$\mathbf{Sp}_{(p)}^\omega$ 是其等价与 retract 闭包。事实上，函子
$$
X\longmapsto\operatorname{fib}
\left(X\to\operatorname*{holim}_nL_nX\right)
$$
是 exact，故其零对象组成的 chromatic-complete 子范畴对有限
fiber/cofiber 与 retract 封闭，所以两种表述一致。对一般谱，tower
可能不收敛；后续调用必须检查有限性或另给 chromatic-completeness
定理。

**例 5.10.** 对有限 $p$-局部球谱 $\mathbb S_{(p)}$，chromatic convergence 给出
$$
\mathbb S_{(p)}\simeq\operatorname*{holim}_{n\ge0} L_n\mathbb S_{(p)}.
$$
这不是稳定 stems 计算的结束，而是把计算拆成每个高度局部问题。

## 5.4 Chromatic fracture square

**外部输入定理 5.11（chromatic fracture square，CHT-P0-10）.** 对每个
$n\ge1$ 和每个 $X\in\mathbf{Sp}_{(p)}$，下列自然交换方块是 pullback：
$$
\begin{array}{ccc}
L_nX & \longrightarrow & L_{K(n)}X\\
\downarrow & & \downarrow\\
L_{n-1}X & \longrightarrow & L_{n-1}L_{K(n)}X
\end{array}
$$
由外部输入 5.2，每个 $K(n)$-local 谱都是 $L_n$-local；所以上边映射
由 $X\to L_{K(n)}X$ 和 $L_n$ 的泛性质诱导，下边映射是对它施加
$L_{n-1}$。该表述没有有限性、connectivity 或 convergence 假设。

**来源与证明边界.** Lurie, Lecture 23, Proposition 5 对任意谱构造该
pullback，并把证明归约到同讲 Proposition 2 的 Bousfield 类分解和
Theorem 4 的 smash product theorem。本书把这三个深输入作为整体引用；
不能把 pullback 性说成定义，也不能用“适当谱”隐藏量词。

**解释 5.12.** fracture square 表明 $L_nX$ 可由低高度部分 $L_{n-1}X$、纯高度 $n$ 部分 $L_{K(n)}X$ 和两者的重叠 $L_{n-1}L_{K(n)}X$ 粘合得到。

**命题 5.13.** 若 fracture square 对 $X$ 成立，且右下角 $L_{n-1}L_{K(n)}X\simeq0$，则
$$
L_nX\simeq L_{n-1}X\times L_{K(n)}X.
$$

**证明.** 在任意 infinity-范畴中，若一个 pullback square 的右下角是终对象，则左上角是左下角和右上角在终对象上的乘积。稳定谱范畴的终对象是零谱。证毕。

**警告 5.14.** 右下角通常不为零。chromatic splitting conjecture 正是关于这个重叠项和相关分裂行为的深层问题，不能在基础章节中省略。

**命题 5.14A（有限 type $n$ 谱的纯高度特例）.** 设 $n\ge1$，且
$F\in\mathbf{Sp}_{(p)}^\omega$ 的 type 为 $n$。则
$$
L_{n-1}F\simeq0,
\qquad
L_{n-1}L_{K(n)}F\simeq0,
$$
并且有自然等价
$$
M_nF\simeq L_nF\simeq L_{K(n)}F.
$$

**证明.** 由 type 定义，$K(i)_*F=0$ 对所有 $i<n$。这等价于
$K(i)\otimes F\simeq0$；外部输入 5.2 因而给出
$E(n-1)\otimes F\simeq0$。命题 1.14B 得 $L_{n-1}F\simeq0$。

有限谱 $F$ dualizable，所以命题 1.14C 给出
$$
L_{K(n)}F\simeq F\otimes L_{K(n)}\mathbb S_{(p)}.
$$
再用外部输入 5.2A 的 smashing 性，
$$
\begin{aligned}
L_{n-1}L_{K(n)}F
&\simeq L_{n-1}\mathbb S_{(p)}\otimes F
   \otimes L_{K(n)}\mathbb S_{(p)}\\
&\simeq L_{n-1}F\otimes L_{K(n)}\mathbb S_{(p)}\simeq0.
\end{aligned}
$$
把两个消失结论代入外部输入 5.11 的 pullback square，得到
$L_nF\simeq L_{K(n)}F$。最后，定义 5.4 的 fiber 序列变为
$M_nF\to L_nF\to0$，故 $M_nF\simeq L_nF$。证毕。

**边界 5.14B.** 命题 5.14A 没有使用 telescope conjecture，也没有
断言 $v_n^{-1}F\simeq L_{K(n)}F$。前者比较的是同一有限 type $n$ 谱的
$E(n)$-与 $K(n)$-局部化；后者比较 telescope 与 $K(n)$-局部化，是另一
个且一般失败的问题。

## 5.5 有限 localization 与 chromatic localization

**定义 5.15.** $L_n^f$ 表示 finite localization，其 acyclic 类为
$$
\operatorname{Loc}^{\otimes}(\mathcal C_{n+1}),
$$
即由所有 type 至少 $n+1$ 的有限谱生成的 localizing tensor ideal。
由 thick subcategory、periodicity 和 class-invariance 定理包，可用一个
type $n+1$ 有限谱生成同一 acyclic 类，并可建立与
$T(0)\vee\cdots\vee T(n)$ 的 telescopic 模型比较；这些是外部结论，
不作为定义的形式推论。

**警告 5.16.** 有自然比较 $L_n^fX\to L_nX$。断言该比较对所有 $X$
为等价正是高度 $n$ telescope conjecture 的一种形式。2023 年后的标准
口径是：不得把 $L_n^f\simeq L_n$ 作为无条件事实。这个失败不与命题
5.14A 冲突，后者只处理输入对象本身有限且 type 恰为 $n$ 的情形。

## 5.6 Chromatic tower 的极限问题

**定义 5.17.** 谱 $X$ 称为 chromatically complete，若自然映射
$$
X\to\operatorname*{holim}_nL_nX
$$
是等价。

**推论 5.18（由外部输入 5.9）.** 有限 $p$-局部谱是 chromatically complete。

**证明.** 定义 5.17 的条件正是外部输入定理 5.9 的结论。证毕。

**警告 5.19.** 若 $X=\operatorname*{colim}_iX_i$ 是有限谱的 filtered colimit，即使每个 $X_i$ chromatically complete，也不能自动推出 $X$ chromatically complete，因为 inverse limit $\operatorname*{holim}_n$ 与 filtered colimit 的交换需要额外条件。

**误差边界 5.19A.** 第 $N$ 阶 chromatic 近似 $X\to L_NX$ 的精确
余项是 $C_NX$。对有限 $X$，定理 5.9 与命题 5.8B 只给出
$$
\operatorname*{holim}_{N}C_NX\simeq0.
$$
它不声称某个有限 $N$ 已有 $C_NX\simeq0$，也不给 stem-uniform 的
数值收敛速率。若计算只截到高度 $N$，必须把 $C_NX$ 或相应
$\lim^1$ 风险保留为未计算误差，而不能用“高色层较小”替代界。

## 5.7 Monochromatic layers 的局部性

**命题 5.20.** $M_nX$ 是 $L_{n-1}$-acyclic。

**证明.** 由定义有 fiber 序列
$$
M_nX\to L_nX\to L_{n-1}X.
$$
命题 1.14B 说明 $L_{n-1}$ exact，所以施加它得到 fiber 序列。命题
5.3 给出自然等价 $L_{n-1}L_n\simeq L_{n-1}$ 和
$L_{n-1}^2\simeq L_{n-1}$；在这些等价下，中间到右端的映射是等价。
因此
$$
L_{n-1}M_nX\simeq0.
$$
本命题的 chromatic 输入只有外部定理 5.2；从 acyclic 类包含到局部化
相容性及 fiber 消失的步骤已在命题 1.14D 与上述计算中内部证明。证毕。

**解释 5.21.** 因此 $M_nX$ 应理解为“高度恰为 $n$ 的层”的候选。
对一般 $X$ 它仍不是 $L_{K(n)}X$；命题 5.14A 则精确给出有限 type
$n$ 输入时二者相同的范围。

## 本章小结

Chromatic tower 通过 $E(n)$-localization 组织高度 $\le n$ 的信息，
monochromatic layer 取相邻层的 fiber，fracture square 对每个谱把高度
$n$ 与低高度沿重叠项粘合。有限 type $n$ 谱的重叠项因 dualizability
而消失，但一般谱没有这个简化。有限谱的 tower 由外部 chromatic
convergence theorem 收敛；其证明同时控制 $\lim$ 与 $\lim^1$，不提供
有限高度误差率。finite/telescopic localization 与 chromatic
localization 仍必须严格区分。

## 练习

**练习 5.1.** 按定义证明 $M_0X\simeq L_0X$。

**练习 5.2.** 若给定定理 5.11 的 pullback square，写出 $L_nX$ 到
$L_{n-1}X\times L_{K(n)}X$ 的自然映射，并指出它何时为等价。

**练习 5.3.** 说明为什么 chromatic convergence theorem 不能直接用于任意 filtered colimit 的有限谱。

**练习 5.4.** 设 $F$ 为 type $n$ 有限谱。逐项标出命题 5.14A 的证明
中哪些步骤是书内范畴论，哪些步骤分别调用 CHT-P0-08、
CHT-P0-08A 与 CHT-P0-10。
