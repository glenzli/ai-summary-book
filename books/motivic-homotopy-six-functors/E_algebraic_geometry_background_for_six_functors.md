# 附录 E：代数几何最小背景：smooth、etale、proper、closed/open immersion

## 本附录目标

本附录列出本书反复使用的代数几何概念和稳定性性质。它不是完整代数几何教材，而是为了让六操作、purity、Gysin maps 和 localization 中的几何假设可检查。

## 依赖前置知识

需要 schemes、morphisms of schemes、fiber products、local rings、Kahler differentials、quasi-compactness、separatedness 和 finite type。

## E.1 Open and closed immersions

**定义 E.1.** 态射 `j:U\to X` 是开嵌入，若它把 `U` 同构到 `X` 的开子概形。

**定义 E.2.** 态射 `i:Z\to X` 是闭嵌入，若它把 `Z` 同构到由 quasi-coherent ideal sheaf 定义的闭子概形。

**命题 E.3.** 开嵌入和闭嵌入都对 base change 稳定。

**证明.** 开子集和闭子集的逆像分别为开子集和闭子集。概形结构由结构层限制或 ideal sheaf 拉回给出，因此纤维积仍是相应嵌入。`\square`

## E.2 Etale morphisms

**定义 E.4.** 态射 `f:X\to Y` 是 etale，若它 locally of finite presentation、flat，且相对微分 `\Omega_{X/Y}=0`。

**命题 E.5.** Etale morphisms 对复合和 base change 封闭。

**证明.** Locally finite presentation、flatness 和相对微分消失都对复合和 base change 具有标准稳定性。微分部分由 exact sequence of differentials 和 base-change formula for differentials 给出。`\square`

**命题 E.6.** Etale morphism 的相对切丛为零。

**证明.** 对 smooth morphism，相对切丛是 `\Omega_{X/Y}` 的对偶。Etale 时 `\Omega_{X/Y}=0`，故对偶为零向量丛。`\square`

## E.3 Smooth morphisms

**定义 E.7.** 态射 `f:X\to Y` 是 smooth of relative dimension `d`，若它 locally of finite presentation、flat，且几何纤维光滑维数 `d`。等价地，局部上 `\Omega_{X/Y}` 为秩 `d` 的向量丛并满足形式光滑性。

**定义 E.8.** Smooth morphism 的相对切丛为

$$
T_f=(\Omega_{X/Y})^\vee.
$$

**命题 E.9.** Smooth morphisms 对复合和 base change 封闭。

**证明.** Smoothness 可由 locally finite presentation 加形式光滑性刻画；二者对复合和 base change 稳定。相对维数在复合中相加。`\square`

## E.4 Proper morphisms

**定义 E.10.** 态射 `f:X\to Y` 是 proper，若它 separated、of finite type 且 universally closed。

**命题 E.11.** Proper morphisms 对复合和 base change 封闭。

**证明.** Separated、finite type 和 universally closed 分别对复合和 base change 稳定。三者合取即 proper。`\square`

**命题 E.12.** 闭嵌入是 proper。

**证明.** 闭嵌入是 finite type、separated 且 universally closed；闭子集在任意 base change 后仍闭。故 proper。`\square`

## E.5 Regular immersions and lci morphisms

**定义 E.13.** 闭嵌入 `i:Z\hookrightarrow X` 是 regular immersion，若局部由 regular sequence 定义。

**定义 E.14.** 态射 `f:X\to Y` 是 local complete intersection morphism，若局部可分解为 regular immersion 后接 smooth morphism。

**定义 E.15.** Regular immersion 的法丛定义为

$$
N_{Z/X}=(I/I^2)^\vee
$$

其中 `I` 是定义 `Z` 的 ideal sheaf。

**命题 E.16.** 若 `i` 是 regular immersion，则 `I/I^2` 是局部自由 sheaf。

**证明.** Regular immersion 局部由 regular sequence `f_1,\ldots,f_r` 定义。其 conormal sheaf `I/I^2` 局部由 `f_i` 的像自由生成，故局部自由。`\square`

## E.6 本附录小结

六操作中的主要几何类具有良好的复合和 base-change 稳定性。Smoothness 提供切丛和 purity twist；properness 提供 `f_!\simeq f_*`；open/closed immersions 提供 localization；regular/lci morphisms 提供法丛、cotangent complex 和 Gysin maps。

## 练习

**练习 E.1.** 证明开嵌入对 base change 稳定。

**练习 E.2.** 证明闭嵌入是 proper。

**练习 E.3.** 说明 etale morphism 的相对切丛为何为零。

**练习 E.4.** 给出 smooth morphism 对复合封闭的证明。

**练习 E.5.** 对 regular immersion 写出 conormal sheaf 和 normal bundle。
