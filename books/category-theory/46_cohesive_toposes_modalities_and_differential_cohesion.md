# 第四十六章：Cohesive Topos、模态与微分凝聚

普通 topos 区分变化集合，却未必同时记住一个空间的同伦形状、离散点和余离散包络。Cohesive topos 通过一串伴随 $\Pi\dashv\mathrm{Disc}\dashv\Gamma\dashv\mathrm{Codisc}$ 组织这些操作，并由相应幂等模态在内部语言中表达。微分凝聚再加入 infinitesimal shape 等结构，使形式邻域与 de Rham 信息进入同一语义框架。本章关注这些伴随和模态的精确公理，而不是把“凝聚性”当作一般 topos 的自动性质。

所需背景是几何态射、$\infty$-topos、左正合局部化和 reflective subcategories。具体 smooth/derived models 与 differential cohesion 作为外部例子；每个模态是否左正合、是否有 fully faithful 伴随都会分别声明。

## 46.1 Cohesive 几何态射串

**定义 46.1.** 本章所称 cohesive $\infty$-topos 是一个 essential 几何态射

$$
p:\mathcal H\to\mathcal S
$$

其逆像 $p^*:\mathcal S\to\mathcal H$ 与右伴随 $p^!:\mathcal S\to\mathcal H$ 都全忠实，并存在伴随串

$$
p_!=\Pi\dashv p^*=\operatorname{Disc}
\dashv p_*=\Gamma\dashv p^!=\operatorname{Codisc}.
$$

还要求 $\Pi$ 保持有限积。这里 $\Gamma$ 为全局截面，$\operatorname{Disc}$ 与 $\operatorname{Codisc}$ 分别给出离散和余离散对象，$\Pi$ 为 shape。不同文献对 cohesion 还会加入连续性或局部连通性公理；后文只使用本定义列出的资料。

**外部输入定理 46.2.** 光滑空间、拓扑空间或高阶 stacks 的合适 $\infty$-topos 在适当假设下给出 cohesive $\infty$-topos。

**命题 46.3.** 若 $\operatorname{Disc}\dashv\Gamma\dashv\operatorname{Codisc}$，则 $\operatorname{Disc}$ 全忠实当且仅当伴随单位 $\operatorname{id}\to\Gamma\operatorname{Disc}$ 为等价。

**证明.** 任意伴随 $L\dashv R$ 中，$L$ 全忠实当且仅当单位 $\operatorname{id}\to RL$ 为同构或等价。取 $L=\operatorname{Disc}$、$R=\Gamma$ 即得结论。$\square$

## 46.2 Shape、flat 与 sharp 模态

**定义 46.4.** 在 cohesive 语境中常定义三个模态：

$$
\int=\operatorname{Disc}\Pi,\qquad
\flat=\operatorname{Disc}\Gamma,\qquad
\sharp=\operatorname{Codisc}\Gamma.
$$

$\int$ 称为 shape modality，$\flat$ 称为 flat modality，$\sharp$ 称为 sharp modality。

**命题 46.5.** 若 $\operatorname{Disc}$ 全忠实，则 $\flat$ 是幂等模态。

**证明.** $\flat=\operatorname{Disc}\Gamma$。计算

$$
\flat\flat
=\operatorname{Disc}\Gamma\operatorname{Disc}\Gamma
\simeq \operatorname{Disc}(\Gamma\operatorname{Disc})\Gamma
\simeq \operatorname{Disc}\Gamma
=\flat,
$$

其中中间等价使用 $\operatorname{Disc}$ 全忠实时 $\Gamma\operatorname{Disc}\simeq\operatorname{id}$。$\square$

## 46.3 Modalities 与 left exact localization

**定义 46.6.** 本章采用较窄约定：$\infty$-topos $\mathcal H$ 上的模态是 left exact reflective localization

$$
L:\mathcal H\to\mathcal H_L
$$

或等价地是保持有限极限的幂等 monad。

**命题 46.7.** Left exact localization 保持 pullback。

**证明.** Left exact 的定义就是保持有限极限。Pullback 是有限极限的一种，故 $L$ 保持 pullback 方块为 pullback 方块。$\square$

**命题 46.8.** 若 $L$ 是 left exact localization，且 $Y\to W\leftarrow Z$ 的三个对象都 $L$-局部，则 pullback $Y\times_WZ$ 仍 $L$-局部。

**证明.** 设 $Y,Z$ 为局部对象，且 $X=Y\times_WZ$。应用 $L$ 得到

$$
LX\simeq L(Y\times_WZ)\simeq LY\times_{LW}LZ.
$$

题设保证右侧为 $Y\times_WZ=X$，故单位 $X\to LX$ 为等价。若 $W$ 不局部，该结论一般不能由 left exactness 单独推出。$\square$

**命题 46.9.** 若 $L$ 是 left exact localization，则 $L$-局部对象构成的全子范畴对有限极限封闭。

**证明.** 设 $D:K\to\mathcal H$ 是有限图形，且每个 $D(k)$ 都是 $L$-局部对象。令 $X=\lim_KD$。由 left exactness，

$$
LX\simeq L(\lim_KD)\simeq \lim_KLD.
$$

因为每个 $D(k)$ 局部，$LD(k)\simeq D(k)$，故

$$
LX\simeq\lim_KD=X.
$$

于是局部化单位 $X\to LX$ 为等价，$X$ 仍为局部对象。$\square$

## 46.4 Modal type theory

**定义 46.10.** Modal type theory 是在依赖类型论中加入模态算子 $\bigcirc$ 及其单位

$$
\eta_A:A\to\bigcirc A
$$

并使 $\bigcirc$ 满足反射性、幂等性和与替换相容的规则。

**命题 46.11.** Left exact modality 保持恒等类型。

**证明.** 恒等类型语义由对角线 $A\to A\times A$ 的路径对象或相应 pullback 结构解释。若模态 $L$ left exact，则它保持有限极限，特别保持对角线、pullback 和路径对象构造中的有限极限部分。因此 $L(\operatorname{Id}_A(x,y))$ 与 $\operatorname{Id}_{LA}(Lx,Ly)$ 相容。完整的类型论消去规则还需外部模型条件，但范畴层面的有限极限保持性正是关键。$\square$

## 46.5 Differential cohesion

**定义 46.12.** Differential cohesive $\infty$-topos 是 cohesive $\infty$-topos，另配 infinitesimal shape 或 de Rham 模态

$$
\Im:\mathcal H\to\mathcal H
$$

用于把对象的无穷小邻域信息压缩为 de Rham 型对象，并满足与 shape、flat、sharp 模态的指定相容公理。仅给出一个幂等端函子并不足以定义 differential cohesion。

**外部输入定理 46.13.** 光滑高阶 stacks 的合适 $\infty$-topos 支持 de Rham stack、infinitesimal shape 和 differential cohomology 的 cohesive 语义。

**定义 46.14.** 对定义在 connective commutative rings 上的 prestack $X$，其 de Rham prestack 定义为

$$
X_{\mathrm{dR}}(R)=X(R_{\mathrm{red}}),
\qquad
R_{\mathrm{red}}=\pi_0(R)/\sqrt{0}.
$$

商映射 $R\to R_{\mathrm{red}}$ 诱导自然变换 $X\to X_{\mathrm{dR}}$。在所用几何语境中再对该 prestack 作相应 sheafification。

**命题 46.15.** 若 $X$ nil-invariant，即对每个 connective ring $R$，映射 $X(R)\to X(R_{\mathrm{red}})$ 都是等价，则自然映射 $X\to X_{\mathrm{dR}}$ 为等价。

**证明.** 按定义 46.14，自然变换在测试环 $R$ 上正是 $X(R)\to X(R_{\mathrm{red}})$。Nil-invariance 说明它逐点为等价，因而是 prestacks 的等价；sheafification 后仍为等价。$\square$

## 46.6 Cohomology 的模态解释

**定义 46.16.** 在 cohesive $\infty$-topos 中，系数对象 $A$ 的 cohomology 可写为映射空间

$$
H^n(X;A)=\pi_0\operatorname{Map}_{\mathcal H}(X,B^nA)
$$

在适当截断和群对象假设下成立。

**命题 46.17.** Shape modality 使 cohesive cohomology 退化为同伦型上的 cohomology。

**证明.** 若 $B^nA$ 是离散或来自 spaces 的系数对象，则由 $\Pi\dashv\operatorname{Disc}$ 有

$$
\operatorname{Map}_{\mathcal H}(X,\operatorname{Disc}B^nA)
\simeq
\operatorname{Map}_{\mathcal S}(\Pi X,B^nA).
$$

取 $\pi_0$ 后得到 $X$ 的 cohesive cohomology 等于其 shape $\Pi X$ 上的普通同伦 cohomology。$\square$

## 46.7 形状、离散化与模态

Cohesive topos 把空间对象的形状、离散化、余离散化和无穷小结构统一到伴随串和模态中。Left exact localization 保证模态与依赖类型论中的替换和恒等类型相容。Differential cohesion 进一步把 de Rham 和微分上同调结构纳入同一范畴逻辑框架。

## 练习

**练习 46.1.** 写出 cohesive topos 的典型伴随串。

**练习 46.2.** 解释 $\Pi,\operatorname{Disc},\Gamma,\operatorname{Codisc}$ 的含义。

**练习 46.3.** 证明左伴随全忠实当且仅当伴随单位为等价。

**练习 46.4.** 定义 shape、flat、sharp 模态。

**练习 46.5.** 证明 $\flat$ 幂等。

**练习 46.6.** 定义 left exact modality。

**练习 46.7.** 证明 left exact localization 保持 pullback。

**练习 46.8.** 说明 left exact modality 与恒等类型的关系。

**练习 46.9.** 定义 modal type theory。

**练习 46.10.** 定义 differential cohesive $\infty$-topos。

**练习 46.11.** 定义 de Rham shape。

**练习 46.12.** 说明局部对象条件如何推出 $X\simeq X_{\mathrm{dR}}$。

**练习 46.13.** 写出 cohomology 的映射空间表达。

**练习 46.14.** 证明 shape modality 给出同伦型上的 cohomology。

**练习 46.15.** 证明 left exact localization 的局部对象对有限极限封闭。
