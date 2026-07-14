# 第十九章：Symplectic resolutions、category O 与 symplectic duality

第五章的 Springer map 与第十七章的 quiver variety map 都从光滑 cotangent 型空间落到奇异仿射簇；它们共有的结构是 conical symplectic resolution。锥作用控制正权参数，symplectic form 给 Poisson bracket，filtered quantization 再把函数乘法变成非交换代数。选定 Hamiltonian cocharacter 后，量子代数的正权部分定义一种 category $\mathcal O$，其形式与 BGG 的正幂零方向惊人地相似。这里用 $T^*\mathbb P^1\to\mathcal N(\mathfrak{sl}_2)$ 验证 proper、birational 与 fiber scaling，并在 Weyl algebra 点模上逐项检查正权算子的局部幂零性；symplectic duality 则只在有明确双方与 Koszul-duality 定理的例子中使用。

**约定 19.0.** 本章低秩几何与 Weyl algebra 计算均在 $\mathbb C$ 上进行；例 19.4 与例 19.8.1 中沿用的系数字母 $E$ 在本章取为 $E=\mathbb C$。

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

**命题 19.5.1（$T^*\mathbb P^1$ 的 conical resolution）.** 令 $W=\mathbb C^2$，把 $\mathbb P^1$ 看成 $W$ 中直线的空间。存在同构
$$
T^*\mathbb P^1\simeq
\{(A,\ell)\in\mathcal N(\mathfrak{sl}(W))\times\mathbb P(W)
\mid \operatorname{im}A\subset\ell\subset\ker A\}.
$$
投影
$$
\pi:T^*\mathbb P^1\longrightarrow\mathcal N(\mathfrak{sl}_2),
\qquad(A,\ell)\longmapsto A
$$
是 conical symplectic resolution；非零 nilpotent orbit 上的 fiber 是一点，零元 fiber 是 $\mathbb P^1$。

**证明.** 在直线 $\ell\subset W$ 处，
$$
T_\ell\mathbb P(W)\simeq\operatorname{Hom}(\ell,W/\ell),
$$
故 trace pairing 把 cotangent space 识别为 $\operatorname{Hom}(W/\ell,\ell)$。这样的映射延拓为 $A:W\to W$ 后恰满足
$$
\operatorname{im}A\subset\ell\subset\ker A,
$$
从而 $A^2=0$ 且 $\operatorname{tr}A=0$。这给出所述 incidence 同构。Incidence variety 是 $\mathcal N\times\mathbb P^1$ 的闭子簇，而到 $\mathcal N$ 的投影由 projective 第二因子推出 proper。对非零 nilpotent $A$，二维性给出
$$
\operatorname{im}A=\ker A,
$$
所以 $\ell$ 唯一；$\pi$ 因而在稠密 regular orbit 上为同构并且 birational。$A=0$ 时任意 $\ell$ 都可取，fiber 为 $\mathbb P^1$。源空间是光滑 cotangent bundle，带 canonical symplectic form；fiber scaling $(A,\ell)\mapsto(tA,\ell)$ 在目标上是 $A\mapsto tA$，收缩到零并由命题 19.5 使 symplectic form 具有权 $1$。故 $\pi$ 是 conical symplectic resolution。$\square$

这正是例 17.7.1 的 Hamiltonian reduction 所得到的映射。Quiver 坐标 $(i,j)$ 中的 invariant $ji$ 与 incidence 描述中的 $A$ 相同，说明“取 affine GIT quotient”在最低秩时就是把零截面上方的 $\mathbb P^1$ 压到锥顶。

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

**例 19.8.1（Weyl algebra 的正权点模）.** 在例 19.4 的 Weyl algebra 中，令 $\mathbb C^\times$ 按
$$
t\cdot x=tx,\qquad t\cdot\partial=t^{-1}\partial
$$
作用，并取 left module
$$
M=A_1/A_1x.
$$
由 PBW normal form，$M$ 有基 $\{\partial^n\overline1\}_{n\ge0}$，且
$$
x\partial^n\overline1=-n\partial^{n-1}\overline1.
$$
把 $M$ 识别为 $E[u]$，其中 $\partial$ 作用为乘 $u$，$x$ 作用为 $-d/du$。权为 $m>0$ 的齐次算子把多项式次数降低 $m$，所以 $A_{>0}$ 在每个向量上 locally nilpotent。因而 $M$ 满足定义 19.7 的正向有限性条件，是这一量子化模型中 category $\mathcal O$ 的基本对象。若反转 cocharacter，则正、负方向互换，同一个结论不再以 $M$ 的这组基呈现。

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

$T^*\mathbb P^1$ 的 incidence 模型把 cotangent covector、nilpotent endomorphism 与 exceptional fiber 放在同一公式中，fiber scaling 则同时给出锥作用和 symplectic form 的正权。Weyl algebra 点模说明 category $\mathcal O$ 的“正向局部有限”可以落实为多项式次数下降。一般 highest-weight 与 Koszul-duality 结论依赖 BLPW--Losev 理论；symplectic duality 也只有在明确例子中才是定理。下一章从另一类 convolution algebra 构造 affine Poisson variety，即 BFN Coulomb branch。

## 练习

**练习 19.1.** 说明 $T^\ast(G/B)\to\mathcal N$ 中 $\mathbb C^\times$ 如何缩放 cotangent fibers。

**练习 19.2.** 给出 Weyl algebra 作为 $T^\ast\mathbb A^1$ 的量子化例子。

**练习 19.3.** 比较 BGG category $\mathcal O$ 和 symplectic resolution category $\mathcal O$ 的定义要素。

**练习 19.4.** 对 $T^\ast\mathbb P^1$ 写出 fiber scaling 对 Liouville 1-form 的作用。

**练习 19.5.** 解释为什么表 19.13 不能作为 symplectic duality 的定义。

**练习 19.6.** 在 $M=A_1/A_1x$ 中证明 $x^r\partial^n\overline1=0$ 当 $r>n$，并说明这如何推出 $A_{>0}$ 的局部幂零性。
