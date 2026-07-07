# 附录 D：六操作相干图、mate calculus 与 Beck-Chevalley 记号

## 本附录目标

本附录补足第四章和第八章使用的 mate calculus。六操作形式主义中大量自然变换不是手工猜出的，而是由伴随单位、余单位和方块相干通过 mate 操作系统地产生。

## 依赖前置知识

需要 adjunctions、natural transformations、2-categories、Cartesian squares、base change、projection formula 和 stable categories。

## D.1 Adjunctions

**定义 D.1.** 伴随 `L\dashv R` 由单位和余单位

$$
\eta:\operatorname{id}\to RL,\qquad
\epsilon:LR\to\operatorname{id}
$$

组成，并满足三角恒等式。

**命题 D.2.** 给出自然变换 `LA\to B` 等价于给出 `A\to RB`。

**证明.** 从 `\alpha:LA\to B` 得到

$$
A\xrightarrow{\eta_A}RLA\xrightarrow{R\alpha}RB.
$$

从 `\beta:A\to RB` 得到

$$
LA\xrightarrow{L\beta}LRB\xrightarrow{\epsilon_B}B.
$$

三角恒等式保证两个构造互逆。`\square`

## D.2 Mates of natural transformations

**定义 D.3.** 设 `L\dashv R` 和 `L'\dashv R'`。自然变换

$$
\alpha:L'F\to GL
$$

的 right mate 是自然变换

$$
F R\to R'G
$$

由单位和余单位合成得到。

**命题 D.4.** Mate 操作保持可逆性：若 `\alpha` 为自然等价，则其 mate 也是自然等价。

**证明.** Mate construction 在相应 functor categories 之间给出等价，因为它由伴随给出的 mapping-space 等价逐对象构成。等价函子保持等价态射。`\square`

## D.3 Ordinary base change

**定义 D.5.** 对 Cartesian 方块

$$
\begin{array}{c}
X'\overset{g'}\longrightarrow X\\
\downarrow f'\qquad\downarrow f\\
Y'\overset{g}\longrightarrow Y
\end{array}
$$

pullback 伪函子给出等价

$$
g'^*f^*\simeq f'^*g^*.
$$

其 mate 是 ordinary base-change map

$$
g^*f_*\to f'_*g'^*.
$$

**命题 D.6.** 定义 D.5 与第四章命题 4.10 的单位-余单位公式一致。

**证明.** 对 pullback square 的相干等价取关于 `f^*\dashv f_*` 和 `f'^*\dashv f'_*` 的 right mate。按定义 D.3 展开，正是先插入 `\operatorname{id}\to f'_*f'^*`，再用 `f'^*g^*\simeq g'^*f^*`，最后用 `f^*f_*\to\operatorname{id}` 的合成。`\square`

## D.4 Extraordinary base change

**定义 D.7.** 若有 `f_!\dashv f^!` 和 `f'_!\dashv f'^!`，则 extraordinary base-change map

$$
g^*f_!\to f'_!g'^*
$$

可与右伴随方向的自然变换

$$
g'^*f^!\to f'^!g^*
$$

互为 mate。

**命题 D.8.** Extraordinary base change 的等价性不是 mate calculus 的形式结论。

**证明.** Mate calculus 只说明两个方向的自然变换等价地携带同一信息；它不说明该自然变换可逆。可逆性是几何定理，例如 proper base change 或 smooth base change。`\square`

## D.5 Projection formula as mate

**定义 D.9.** Projection formula map

$$
f_!(A\otimes f^*B)\to f_!A\otimes B
$$

可由 `f_!` 的 `\mathcal D(Y)`-module functor structure 或由 closed monoidal adjunction 的 mate 构造。

**命题 D.10.** 若 `f_!` 是 `\mathcal D(Y)`-linear，则 projection formula map 为等价。

**证明.** `\mathcal D(Y)`-linear 的定义正是给出自然等价

$$
f_!(A\otimes f^*B)\simeq f_!A\otimes B
$$

并满足结合和单位相干。`\square`

## D.6 Coherence and pasting

**定理 D.11（Pasting of Beck-Chevalley）.** 若两个相邻 Cartesian 方块的 base-change maps 为等价，则外矩形的 base-change map 为等价，并且该等价等于两个小方块等价的合成。

**证明.** Pullback 伪函子的方块相干对横向或纵向复合满足 pasting law。Mate 操作与复合相容；因此外矩形的 mate 等于两个小方块 mate 的合成。若两个小方块 mate 均为等价，其合成也是等价。`\square`

## D.7 本附录小结

Mate calculus 是六操作的语法。它构造 base-change maps、比较 `!` 与 `*` 方向的变换，并保证相干图在复合时行为正确。几何内容在于这些 mate 何时为等价，而不是 mate 是否能写出。

## 练习

**练习 D.1.** 写出伴随转置的两个方向并验证三角恒等式。

**练习 D.2.** 展开 ordinary base-change map 的单位-余单位公式。

**练习 D.3.** 说明 mate of equivalence 为 equivalence。

**练习 D.4.** 证明 Beck-Chevalley pasting。

**练习 D.5.** 举例说明“自然变换存在”和“自然变换为等价”的区别。
