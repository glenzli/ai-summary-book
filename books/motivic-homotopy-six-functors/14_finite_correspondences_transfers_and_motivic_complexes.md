# 第十四章：Finite correspondences、presheaves with transfers 与 motivic complexes

## 本章目标

本章介绍 Voevodsky motives 的 transfer 入口。有限对应（finite correspondences）把普通态射扩展为代数循环式的多值态射，presheaves with transfers 则是能沿这些对应反变的预层。该结构是 motivic complexes、Voevodsky motives 和 `H\mathbb Z` 的重要历史来源。

## 依赖前置知识

需要光滑概形、有限态射、代数循环、Chow 群、加性范畴、presheaves、Nisnevich sheaves、motivic cohomology 和 derived categories。

## 14.1 Finite correspondences

**约定 14.1.** 本章默认 `k` 为 perfect field，`\operatorname{Sm}_k` 为 `k` 上光滑有限型概形。更一般基上的 relative cycles 需要 Suslin-Voevodsky 的额外理论。

**定义 14.2.** 对 `X,Y\in\operatorname{Sm}_k`，从 `X` 到 `Y` 的 finite correspondence 是 `X\times_kY` 上的整闭子概形 `Z` 的有限整数线性组合，其中每个 `Z` 在 `X` 的某个连通分支上有限且满。对应群记为

$$
\operatorname{Cor}_k(X,Y).
$$

**外部输入定理 14.3.** Finite correspondences 可自然复合，形成加性范畴 `\operatorname{Cor}_k`，其对象为 `\operatorname{Sm}_k`，Hom 群为 `\operatorname{Cor}_k(X,Y)`。

**依赖源.** Suslin-Voevodsky、Voevodsky motives、Mazza-Voevodsky-Weibel。复合使用 proper pushforward、pullback 和交理论，非纯形式集合论构造。

**定义 14.4.** 图函子

$$
\Gamma:\operatorname{Sm}_k\longrightarrow\operatorname{Cor}_k
$$

在对象上为恒等，在态射 `f:X\to Y` 上取图 `\Gamma_f\subset X\times Y`。

**命题 14.5.** 对态射 `X\xrightarrow{f}Y\xrightarrow{g}Z`，有

$$
\Gamma_g\circ\Gamma_f=\Gamma_{gf}
$$

作为 finite correspondences。

**证明.** 图的复合由 `X\times Y\times Z` 中的交-推公式给出。`\Gamma_f` 与 `\Gamma_g` 的纤维积由三元组 `(x,y,z)` 满足 `y=f(x)` 且 `z=g(y)` 描述，因此等同于 `(x,gf(x))` 的图。推到 `X\times Z` 后得到 `\Gamma_{gf}`。`\square`

## 14.2 Presheaves with transfers

**定义 14.6.** 一个 presheaf with transfers 是加性反变函子

$$
F:\operatorname{Cor}_k^{op}\to\operatorname{Ab}.
$$

其范畴记为 `\operatorname{PST}(k)`。

**定义 14.7.** 对 `Y\in\operatorname{Sm}_k`，表示 presheaf with transfers 定义为

$$
\mathbb Z_{tr}(Y)(X)=\operatorname{Cor}_k(X,Y).
$$

**命题 14.8.** `\mathbb Z_{tr}(Y)` 限制到 `\operatorname{Sm}_k` 上的普通 presheaf，含有由普通态射到 `Y` 生成的子 presheaf。

**证明.** 通过图函子 `\Gamma`，每个态射 `X\to Y` 给出 correspondence `\Gamma_f\in\operatorname{Cor}_k(X,Y)`。因此 ordinary representable presheaf `h_Y(X)=\operatorname{Hom}_{\operatorname{Sm}_k}(X,Y)` 映入 `\mathbb Z_{tr}(Y)(X)`。自然性来自命题 14.5。`\square`

**定义 14.9.** 若 presheaf with transfers 的底层 presheaf 是 Nisnevich sheaf，则称为 Nisnevich sheaf with transfers。其范畴记为 `\operatorname{Shv}_{Nis}^{tr}(k)`。

**外部输入定理 14.10.** 在 perfect field 上，Nisnevich sheafification 与 transfers 相容：presheaf with transfers 的 Nisnevich sheafification 仍带 transfers。

**注 14.11.** 定理 14.10 是使用 Nisnevich topology 的关键原因之一。普通 sheafification 不自动保留额外 correspondence action。

## 14.3 复合的交理论细节

**定义 14.12.** 设 `\alpha\in\operatorname{Cor}_k(X,Y)`，`\beta\in\operatorname{Cor}_k(Y,Z)`。其复合由下列操作定义：

1. 在 `X\times Y\times Z` 中拉回 `\alpha` 与 `\beta`；
2. 取适当交积；
3. 沿投影 `X\times Y\times Z\to X\times Z` proper pushforward。

**外部输入定理 14.13.** 上述复合良定义、结合，并保持 finite-over-source 条件。

**命题 14.14.** 单位 correspondence 是对角线 `\Delta_X\subset X\times X`。

**证明.** 对任意 `\alpha:X\rightsquigarrow Y`，与 `\Delta_X` 或 `\Delta_Y` 复合时，交-推公式退化为沿恒等图拉回和推前，因此不改变 `\alpha`。这与普通关系复合中对角线为单位相同，严格良定义依赖定理 14.13 的交理论。`\square`

## 14.4 Motivic complexes

**定义 14.15.** Motivic complex `\mathbb Z(q)` 是从带 transfers 的循环复形构造出的复形；它代表 motivic cohomology 的 weight `q` 部分。

**外部输入定理 14.16.** 对光滑 `X/k`，有自然同构

$$
H^{p,q}(X,\mathbb Z)\simeq
\mathbb H^p_{Nis}(X,\mathbb Z(q)).
$$

**依赖源.** Voevodsky、Suslin-Voevodsky、Bloch higher Chow groups、MVW。

**例子 14.17.** 标准归一化下，`\mathbb Z(0)` 为常值 sheaf `\mathbb Z`，而 `\mathbb Z(1)` 与 `\mathbb G_m[-1]` 比较。该比较是 motivic complexes 的基本计算外部输入。

**定义 14.18.** Suslin complex `C_*(F)` 对 presheaf with transfers `F` 定义为

$$
C_n(F)(X)=F(X\times\Delta^n),
$$

其中 `\Delta^n` 是代数 `n`-simplex。面和退化映射由 cosimplicial simplex 诱导。

**命题 14.19.** Suslin complex 强制 `\mathbb A^1`-同伦不变性的链级影子。

**证明.** `C_*(F)` 把 `F` 沿所有代数 simplex 取值组织成单纯形链复形。由于 `\Delta^1` 是代数区间，两个由 `0,1:\operatorname{Spec}k\to\Delta^1` 给出的限制在链同伦意义下相同。因此 `C_*(F)` 是把 naive presheaf 向 `\mathbb A^1`-同伦不变对象推进的链级构造。`\square`

## 14.5 Transfers 与 HZ

**外部输入定理 14.20.** `H\mathbb Z` 可由带 transfers 的 motivic complexes 构造，并与第九章中作为 motivic Eilenberg-Mac Lane spectrum 的 `H\mathbb Z` 相容。

**命题 14.21.** 若定理 14.20 适用，则 finite correspondences 诱导 motivic cohomology 的 transfers。

**证明.** Motivic cohomology 由 `\mathbb Z(q)` 的 Nisnevich hypercohomology 表示。`\mathbb Z(q)` 是带 transfers 的复形，因此 correspondence `\alpha\in\operatorname{Cor}_k(X,Y)` 诱导复形映射，从而诱导 hypercohomology 群上的映射。与 `H\mathbb Z` 表示性的相容由定理 14.15 保证。`\square`

## 14.6 边界

**命题 14.22.** Finite correspondences 的 transfers 不等同于第十七章的 norm maps。

**证明.** Finite correspondences 给出加性转移，Hom 群本身是自由阿贝尔群式的线性组合；norm maps 是 multiplicative transfers，要求对称幺半结构和乘法相干。加性 correspondence action 不包含 norm 的乘法分配律，因此二者不能识别。`\square`

## 14.7 本章小结

Finite correspondences 把普通几何态射扩展为循环式对应，presheaves with transfers 则允许 cohomology theory 沿这些对应反变。Motivic complexes 和 `H\mathbb Z` 的传统构造依赖 transfers；这条路线与 `\mathbf{SH}` 中的 ring spectrum 表示性相容，但比较定理是外部输入。

## 练习

**练习 14.1.** 定义 finite correspondence 并解释“finite and surjective over a component of `X`”的作用。

**练习 14.2.** 证明图 correspondence 与复合相容。

**练习 14.3.** 定义 presheaf with transfers。

**练习 14.4.** 说明为什么 sheafification 与 transfers 相容是非平凡定理。

**练习 14.5.** 比较 additive transfers 与 multiplicative norms。

**练习 14.6.** 写出 finite correspondences 复合的三步交-推公式。

**练习 14.7.** 定义 Suslin complex 并说明 `\Delta^1` 的作用。
