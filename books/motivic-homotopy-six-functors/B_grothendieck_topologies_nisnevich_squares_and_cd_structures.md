# 附录 B：Grothendieck topologies、points、Nisnevich squares 与 cd-structures

## 本附录目标

本附录补足第一章关于 Nisnevich topology 的细节。正式教材中，Nisnevich topology 不应只作为“介于 Zariski 和 etale 之间”的口号出现；必须给出覆盖、点提升、elementary distinguished squares 和 descent 检测的精确关系。

## 依赖前置知识

需要 schemes、etale morphisms、residue fields、Grothendieck topologies、sieves、Cech nerves、points of topoi 和 elementary distinguished squares。

## B.1 Grothendieck topology

**定义 B.1.** 设 `C` 有拉回。Grothendieck topology 指定每个对象 `X` 的一类 covering sieves，满足：

1. maximal sieve 覆盖；
2. 覆盖筛沿任意态射拉回仍覆盖；
3. 若 `R` 覆盖 `X`，且 `S` 是 `X` 上筛，满足对每个 `U\to X` 属于 `R`，拉回 `S|_U` 覆盖 `U`，则 `S` 覆盖 `X`。

**定义 B.2.** 一个 covering family `\{U_i\to X\}` 生成覆盖筛：由所有通过某个 `U_i` 因子化的态射组成。

**命题 B.3.** 用 covering families 给出 topology，需要验证同构覆盖、拉回稳定和复合稳定。

**证明.** 由 covering families 生成的筛满足定义 B.1。拉回稳定确保公理 2；复合稳定确保公理 3；同构覆盖确保 maximal sieve 覆盖。`\square`

## B.2 Nisnevich covering families

**定义 B.4.** `\operatorname{Sm}_S` 中的族 `\{U_i\to X\}` 是 Nisnevich covering family，若每个 `U_i\to X` etale，且对每个点 `x\in X`，存在 `i` 和 `u\in U_i` 映到 `x`，使

$$
\kappa(x)\simeq\kappa(u).
$$

**命题 B.5.** Nisnevich covering families 生成 Grothendieck topology。

**证明.** 同构族满足定义。Etale morphisms 对 base change 封闭，剩余域同构的点提升也可沿 base change 拉回，因此覆盖拉回稳定。若 `V_{ij}\to U_i` 覆盖每个 `U_i`，对 `x\in X` 先由 `U_i` 覆盖选 `u` 使 `\kappa(x)\simeq\kappa(u)`，再由 `V_{ij}` 覆盖选 `v` 使 `\kappa(u)\simeq\kappa(v)`，合成得到 `\kappa(x)\simeq\kappa(v)`。`\square`

**命题 B.6.** 每个 Zariski covering family 都是 Nisnevich covering family。

**证明.** 开嵌入是 etale。若 `\{U_i\}` 是 Zariski 覆盖，对每个点 `x\in X`，存在 `i` 使 `x\in U_i`。取同一个点视为 `U_i` 中的点，剩余域不变。`\square`

**命题 B.7.** 每个 Nisnevich covering family 都是 etale covering family，但反向不成立。

**证明.** 定义要求每个映射 etale，故为 etale covering family。反向失败：有限可分域扩张 `L/k` 给出 `\operatorname{Spec}L\to\operatorname{Spec}k` 的 etale 覆盖；若 `L\ne k`，则闭点剩余域不是同构，因此不是 Nisnevich 覆盖。`\square`

## B.3 Elementary distinguished squares

**定义 B.8.** Elementary distinguished square 是 Cartesian 方块

$$
\begin{array}{c}
V'\longrightarrow V\\
\downarrow\qquad\downarrow p\\
U\overset{j}\longrightarrow X
\end{array}
$$

其中 `j` 开嵌入，`p` etale，且诱导闭补映射

$$
V\setminus V'\longrightarrow X\setminus U
$$

为同构。

**命题 B.9.** 对 elementary distinguished square，`U\amalg V\to X` 是 Nisnevich covering family。

**证明.** 两个映射都是 etale：`j` 是开嵌入，`p` 按定义 etale。对 `x\in X`，若 `x\in U`，由 `U` 中同一点提升，剩余域不变。若 `x\notin U`，则 `x\in X\setminus U`。闭补同构给出唯一 `v\in V\setminus V'` 映到 `x`，且剩余域同构。`\square`

**命题 B.10.** 若 `F` 是 Nisnevich sheaf，则 elementary distinguished square 被 `F` 送为拉回方块。

**证明.** 由命题 B.9，`U\amalg V\to X` 是 Nisnevich 覆盖。对该覆盖应用 Cech descent。由于 `U\times_XV\simeq V'`，覆盖的非退化交叠由 `V'` 控制；sheaf 条件给出

$$
F(X)\simeq F(U)\times_{F(V')}F(V).
$$

这就是拉回方块条件。`\square`

## B.4 cd-structure 生成定理

**外部输入定理 B.11.** 在 Noetherian 有限维基上，Nisnevich topology 可由空覆盖和 elementary distinguished squares 生成；满足适当 boundedness/regularity 条件的 presheaf 若把这些 squares 送为 homotopy pullback，则满足 Nisnevich descent。

**依赖源.** Voevodsky cd-structures，Morel-Voevodsky，后续 motivic homotopy 教材。

**注 B.12.** 第一章只使用了“sheaf 推出 square descent”的方向。反方向需要 cd-structure 理论，是外部输入或附录级定理。

## B.5 Points

**定义 B.13.** 概形 `X` 的 henselian local schemes `\operatorname{Spec}\mathcal O^h_{X,x}` 给出 Nisnevich topology 的 points。

**外部输入定理 B.14.** Nisnevich sheaf 的等价可在 henselian local points 上检测：若 `F\to G` 在所有 henselian local schemes 上诱导等价，则它是 Nisnevich local equivalence。

**命题 B.15.** `\mathbb A^1`-局部性不能只在 ordinary closed points 上检测。

**证明.** Nisnevich topology 的点包含 henselian local information，而 ordinary closed points 只记录剩余域值。Descent 和局部同伦等价依赖邻域粘合与 henselian lifting；只看闭点会忽略局部环信息。因此 closed points 不足以检测 Nisnevich local equivalence。`\square`

## B.6 本附录小结

Nisnevich topology 的核心是 etale 覆盖加剩余域同构提升。Elementary distinguished squares 是它的计算模型，henselian local points 是其检测模型。正式使用 Nisnevich descent 时必须说明使用的是覆盖族、覆盖筛、cd-structure 还是 points。

## 练习

**练习 B.1.** 证明 Zariski 覆盖是 Nisnevich 覆盖。

**练习 B.2.** 给出 etale 覆盖不是 Nisnevich 覆盖的域扩张例子。

**练习 B.3.** 验证 elementary distinguished square 中 `U\times_XV\simeq V'`。

**练习 B.4.** 从 Cech descent 推出命题 B.10。

**练习 B.5.** 解释 henselian local points 在 Nisnevich topology 中的角色。
