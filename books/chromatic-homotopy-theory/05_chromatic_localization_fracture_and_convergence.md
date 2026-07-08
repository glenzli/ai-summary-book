# 第五章：Chromatic localization、fracture 与 convergence

## 本章目标

本章把 $K(0),K(1),\ldots,K(n)$ 组装成 chromatic tower。核心对象是 $L_n=L_{E(n)}$、monochromatic layer $M_n$、chromatic fracture square 和 chromatic convergence theorem。

## 依赖前置知识

需要第一章的 Bousfield localization、第二章的 $E(n)$ 和第三章的 $K(n)$。$E(n)$ 与 Morava K-theories 的 Bousfield 类关系、chromatic fracture square 和 convergence theorem 作为外部输入或标准定理包处理。

## 5.1 $E(n)$-localization

**定义 5.1.** Johnson-Wilson theory $E(n)$ 的 Bousfield localization 记为
$$
L_n=L_{E(n)}.
$$
约定 $L_{-1}X=0$。

**外部输入 5.2.** 在 $p$-局部稳定同伦论中，
$$
\langle E(n)\rangle=\langle K(0)\vee K(1)\vee\cdots\vee K(n)\rangle.
$$
因此 $L_n$ 可理解为保留高度不超过 $n$ 的信息。

**命题 5.3.** 若 $X$ 是 $L_n$-local，则 $X$ 也是 $L_m$-local 对所有 $m\le n$ 的说法不能无条件按形式推出；正确使用方式应通过 Bousfield acyclic 类包含关系检查。

**证明.** localization 的 local object 由 acyclic 类定义。若知道
$$
E(n)\otimes A\simeq0\Rightarrow E(m)\otimes A\simeq0,
$$
则 $E(m)$-acyclic 类包含 $E(n)$-acyclic 类，local 条件方向随之改变。由于文献中 Bousfield 偏序方向不同，本书要求每次直接写 acyclic implication，而不是只写 $\langle E(m)\rangle\le \langle E(n)\rangle$。证毕。

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

**证明.** $L_n$-local 对象构成稳定全子范畴。$L_nX$ 是 $L_n$-local；$L_{n-1}X$ 也是 $L_n$-local 这一点需要外部输入 5.2 给出的 acyclic 类包含关系。稳定全子范畴对 fiber 封闭，因此 $M_nX$ 是 $L_n$-local。证毕。

**警告 5.7.** $M_nX$ 不等同于 $L_{K(n)}X$。前者是 $E(n)$ tower 的 fiber，后者是单一高度 $n$ 的局部化。两者通过 fracture square 和 local duality 发生关系，但不能按定义混用。

## 5.3 Chromatic tower

**定义 5.8.** 谱 $X$ 的 chromatic tower 是反向系统
$$
\cdots\to L_nX\to L_{n-1}X\to\cdots\to L_1X\to L_0X.
$$

**外部输入定理 5.9 (chromatic convergence).** 若 $X$ 是有限 $p$-局部谱，则自然映射
$$
X\longrightarrow \operatorname*{holim}_{n} L_nX
$$
是等价。

**使用说明.** 定理 5.9 是有限谱定理。对一般谱，chromatic tower 可能不收敛到 $X$。后续任何使用 convergence 的地方必须检查有限性或额外完备性假设。

**例 5.10.** 对有限 $p$-局部球谱 $\mathbb S_{(p)}$，chromatic convergence 给出
$$
\mathbb S_{(p)}\simeq\operatorname*{holim}_n L_n\mathbb S.
$$
这不是稳定 stems 计算的结束，而是把计算拆成每个高度局部问题。

## 5.4 Chromatic fracture square

**外部输入定理 5.11 (chromatic fracture square).** 对适当谱 $X$，存在 pullback square
$$
\begin{array}{ccc}
L_nX & \longrightarrow & L_{K(n)}X\\
\downarrow & & \downarrow\\
L_{n-1}X & \longrightarrow & L_{n-1}L_{K(n)}X
\end{array}
$$
精确适用范围和模型假设必须按资料源定位。

**解释 5.12.** fracture square 表明 $L_nX$ 可由低高度部分 $L_{n-1}X$、纯高度 $n$ 部分 $L_{K(n)}X$ 和两者的重叠 $L_{n-1}L_{K(n)}X$ 粘合得到。

**命题 5.13.** 若 fracture square 对 $X$ 成立，且右下角 $L_{n-1}L_{K(n)}X\simeq0$，则
$$
L_nX\simeq L_{n-1}X\times L_{K(n)}X.
$$

**证明.** 在任意 infinity-范畴中，若一个 pullback square 的右下角是终对象，则左上角是左下角和右上角在终对象上的乘积。稳定谱范畴的终对象是零谱。证毕。

**警告 5.14.** 右下角通常不为零。chromatic splitting conjecture 正是关于这个重叠项和相关分裂行为的深层问题，不能在基础章节中省略。

## 5.5 有限 localization 与 chromatic localization

**定义 5.15.** $L_n^f$ 表示 finite localization，其 acyclics 由 type $n+1$ 有限谱生成。等价地，在存在相应 telescope 模型时，它与 $T(0)\vee\cdots\vee T(n)$ 的局部化相关。

**警告 5.16.** $L_n^f$ 与 $L_n$ 的比较是 telescope conjecture 的核心。2023 年后的标准口径是：不得把 $L_n^f=L_n$ 作为无条件事实。

## 5.6 Chromatic tower 的极限问题

**定义 5.17.** 谱 $X$ 称为 chromatically complete，若自然映射
$$
X\to\operatorname*{holim}_nL_nX
$$
是等价。

**命题 5.18.** 有限 $p$-局部谱是 chromatically complete。

**证明.** 这是 chromatic convergence theorem 的陈述，作为外部输入使用。证毕。

**警告 5.19.** 若 $X=\operatorname*{colim}_iX_i$ 是有限谱的 filtered colimit，即使每个 $X_i$ chromatically complete，也不能自动推出 $X$ chromatically complete，因为 inverse limit $\operatorname*{holim}_n$ 与 filtered colimit 的交换需要额外条件。

## 5.7 Monochromatic layers 的局部性

**命题 5.20.** $M_nX$ 是 $L_{n-1}$-acyclic。

**证明草图.** 由定义有 fiber 序列
$$
M_nX\to L_nX\to L_{n-1}X.
$$
对该序列施加 $L_{n-1}$。若使用 $L_{n-1}L_n\simeq L_{n-1}$ 这一 chromatic localization 的标准相容性，则
$$
L_{n-1}M_nX\simeq\operatorname{fib}(L_{n-1}X\to L_{n-1}X)\simeq0.
$$
相容性 $L_{n-1}L_n\simeq L_{n-1}$ 依赖 $E(n)$ 与 $E(n-1)$ 的 Bousfield 类关系，作为外部输入。证毕。

**解释 5.21.** 因此 $M_nX$ 应理解为“高度恰为 $n$ 的层”的候选。但它仍不是 $L_{K(n)}X$，二者通过 local duality 和 fracture 技术联系。

## 本章小结

Chromatic tower 通过 $E(n)$-localization 保留高度 $\le n$ 的信息，monochromatic layer 取相邻层的 fiber，fracture square 把高度 $n$ 和低高度粘合。有限谱有 chromatic convergence，但一般谱没有自动收敛。finite/telescopic localization 与 chromatic localization 的区别是现代前沿的关键。

## 练习

**练习 5.1.** 按定义证明 $M_0X\simeq L_0X$。

**练习 5.2.** 若给定 pullback square 如定理 5.11，写出 $L_nX$ 到 $L_{n-1}X\times L_{K(n)}X$ 的自然映射。

**练习 5.3.** 说明为什么 chromatic convergence theorem 不能直接用于任意 filtered colimit 的有限谱。
