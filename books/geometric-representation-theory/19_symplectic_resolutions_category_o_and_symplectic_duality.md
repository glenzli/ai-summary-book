# 第十九章：Symplectic resolutions、category O 与 symplectic duality

## 本章目标

本章介绍 conical symplectic resolutions、quantizations、category $\mathcal O$ 和 symplectic duality。

## 依赖前置知识

需要第五章的 Springer resolution、第十章的 quantization 接口，以及第十七章的 quiver varieties。

## 19.1 Conical symplectic resolution

**定义 19.1.** 一个 conical symplectic resolution 是 proper birational morphism
$$
\pi:X\to X_0
$$
其中 $X$ 是光滑代数 symplectic variety，$X_0$ 是 affine Poisson variety，并有 $\mathbb C^\times$-作用收缩 $X_0$ 到有限型核心，同时 symplectic form 具有正权。

**例 19.2.** Springer resolution
$$
T^\ast\mathcal B\to\mathcal N
$$
是 conical symplectic resolution 的原型。Nakajima quiver varieties 和 hypertoric varieties 也提供大量例子。

**定义 19.3.** $X$ 的 quantization 是 sheaf 或 filtered algebra $\mathcal A_\hbar$，使
$$
\operatorname{gr}\mathcal A_\hbar\simeq\mathcal O_X
$$
且 commutator 的 first order term 给出 Poisson bracket。

**例 19.4.** $T^\ast\mathbb A^1$ 的坐标环为 $E[x,\xi]$，Poisson bracket 由
$$
\{ \xi,x\}=1
$$
确定。Weyl algebra
$$
A_1=E\langle x,\partial\rangle/(\partial x-x\partial-1)
$$
带 order filtration，满足
$$
\operatorname{gr}A_1\simeq E[x,\xi].
$$
因此 $A_1$ 是 $T^\ast\mathbb A^1$ 的量子化。若引入参数 $\hbar$，关系写作
$$
\partial x-x\partial=\hbar,
$$
则 commutator 除以 $\hbar$ 的 associated graded 给出 Poisson bracket。

**命题 19.5.** $T^\ast\mathcal B\to\mathcal N$ 中的 cotangent fiber scaling 使 symplectic form 具有正权。

**证明.** cotangent bundle 上的 Liouville 1-form $\theta$ 在 fiber scaling $t\cdot(x,\xi)=(x,t\xi)$ 下满足 $t^\ast\theta=t\theta$。symplectic form 为 $\omega=d\theta$，故 $t^\ast\omega=t\omega$。因此 $\omega$ 权为 $1$。moment map $T^\ast\mathcal B\to\mathfrak g^\ast$ 对 fiber 坐标线性，也与该 scaling 相容。$\square$

## 19.2 Category $\mathcal O$ for symplectic resolutions

**定义 19.6.** 给定 Hamiltonian torus action 和 quantization $A$，symplectic resolution 的 category $\mathcal O$ 通常由满足如下条件的 $A$-modules 构成：有限生成、某个正向子代数 locally finite、并满足中心或 period 条件。

**定义 19.7.** 若 $A$ 有由 cocharacter $\nu:\mathbb C^\times\to T$ 诱导的 grading
$$
A=\bigoplus_{m\in\mathbb Z}A_m,
$$
则正向子代数可形式写为
$$
A_{>0}=\bigoplus_{m>0}A_m.
$$
一个 $A$-module $M$ 属于 $\mathcal O_\nu(A)$ 的基本条件是 $M$ 有限生成且 $A_{>0}$ 在 $M$ 上 locally finite 或 locally nilpotent，具体取决于文献 convention。

**命题 19.8.** BGG category $\mathcal O$ 是上述定义的原型。

**证明.** 对 $U(\mathfrak g)$，取 triangular decomposition
$$
\mathfrak g=\mathfrak n^-\oplus\mathfrak t\oplus\mathfrak n.
$$
在 BGG category $\mathcal O$ 中，要求 $U(\mathfrak n)$ locally finite，正是“正向子代数 locally finite”的条件；有限生成对应 $U(\mathfrak g)$-module 有限生成；权分解对应 torus action 的半单性。symplectic resolution category $\mathcal O$ 把这三个条件从 $U(\mathfrak g)$ 抽象到 filtered/quantized algebra。$\square$

**外部输入定理 19.9.** Braden-Licata-Proudfoot-Webster 和 Losev 建立 quantized conical symplectic resolutions 的 category $\mathcal O$ 理论，包括 highest weight 结构、twisting/shuffling functors 和 derived equivalences 的一系列结果。

## 19.3 Symplectic duality

**定义 19.10.** Symplectic duality 是两个 conical symplectic resolutions $X$ 和 $X^!$ 之间的一组结构对应，包括：

1. category $\mathcal O(X)$ 与 $\mathcal O(X^!)$ 的 Koszul duality；
2. Hamiltonian torus 与 Namikawa Weyl group 数据交换；
3. fixed points、cores、strata 和 chambers 的对应；
4. twisting functors 与 shuffling functors 的交换。

**外部输入定理 19.11.** BLPW 的 symplectic duality framework 在许多例子中成立或预测成立，包括 hypertoric varieties、部分 quiver varieties、Springer-type examples 和 3d mirror symmetry 相关空间。

**边界说明 19.12.** Symplectic duality 目前不是所有 conical symplectic resolutions 上的已证明全称定理。本书只在具体例子和文献假设下使用。

**表 19.13.** 常见对应模式如下。

| $X$ 侧 | $X^!$ 侧 | 对应内容 |
| --- | --- | --- |
| twisting functors | shuffling functors | chambers 与 cocharacters 交换 |
| simples in $\mathcal O(X)$ | standards/costandards under Koszul duality | highest weight 结构 |
| Hamiltonian torus | Namikawa/Weyl 参数 | 形变与分辨率参数交换 |
| core components | fixed points/strata | 组合数据反转 |

该表只是框架说明，不是无条件定理。

## 本章小结

本章定义 conical symplectic resolution、quantization、广义 category $\mathcal O$ 和 symplectic duality，补充了 Weyl algebra 量子化例子、cotangent scaling 检查和 BGG category $\mathcal O$ 的原型说明。BLPW-Losev 理论是外部输入。

## 练习

**练习 19.1.** 说明 $T^\ast(G/B)\to\mathcal N$ 中 $\mathbb C^\times$ 如何缩放 cotangent fibers。

**练习 19.2.** 给出 Weyl algebra 作为 $T^\ast\mathbb A^1$ 的量子化例子。

**练习 19.3.** 比较 BGG category $\mathcal O$ 和 symplectic resolution category $\mathcal O$ 的定义要素。

**练习 19.4.** 对 $T^\ast\mathbb P^1$ 写出 fiber scaling 对 Liouville 1-form 的作用。

**练习 19.5.** 解释为什么表 19.13 不能作为 symplectic duality 的定义。
