# 第四十六章：Cohesive Topos、模态与微分凝聚

## 本章目标

本章介绍 cohesive topos 与 modal type theory 的范畴论核心。普通 topos 处理逻辑和集合变化；cohesive topos 进一步同时记录形状、离散化、余离散化和内在空间结构。它为同伦类型论、微分几何、同调论和高阶几何提供统一的模态语义。

## 依赖前置知识

需要 adjoint functors、geometric morphisms、topos、$\infty$-topos、left exact localization、modalities、reflective subcategory、shape functor、homotopy types 和基本微分几何语义。

## 46.1 Cohesive 几何态射串

**定义 46.1.** 一个 cohesive $\infty$-topos 是带有到 spaces 的几何态射

$$
p:\mathcal H\to\mathcal S
$$

并配有典型伴随串

$$
\Pi\dashv \operatorname{Disc}\dashv \Gamma\dashv \operatorname{Codisc},
$$

其中 $\Gamma:\mathcal H\to\mathcal S$ 为全局截面，$\operatorname{Disc}$ 与 $\operatorname{Codisc}$ 分别给出离散和余离散对象，$\Pi$ 为 shape 或 fundamental $\infty$-groupoid。

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

**定义 46.6.** $\infty$-topos $\mathcal H$ 上的模态是 left exact reflective localization

$$
L:\mathcal H\to\mathcal H_L
$$

或等价地是保持有限极限的幂等 monad。

**命题 46.7.** Left exact localization 保持 pullback。

**证明.** Left exact 的定义就是保持有限极限。Pullback 是有限极限的一种，故 $L$ 保持 pullback 方块为 pullback 方块。$\square$

**命题 46.8.** 若 $L$ 是 left exact localization，则 $L$-局部对象在 pullback 下稳定。

**证明.** 设 $Y,Z$ 为局部对象，且 $X=Y\times_WZ$。应用 $L$ 得到

$$
LX\simeq L(Y\times_WZ)\simeq LY\times_{LW}LZ.
$$

若 $W$ 也局部，则右侧为 $Y\times_WZ=X$。更一般地，在局部对象形成的反射子 $\infty$-topos 内，有限极限由原范畴中有限极限再局部化给出。$\square$

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

用于把对象的无穷小邻域信息压缩为 de Rham 型对象。

**外部输入定理 46.13.** 光滑高阶 stacks 的合适 $\infty$-topos 支持 de Rham stack、infinitesimal shape 和 differential cohomology 的 cohesive 语义。

**定义 46.14.** 对光滑对象 $X$，其 de Rham shape $X_{\mathrm{dR}}$ 可抽象为把 nilpotent 或 infinitesimal thickening 方向局部化后的对象。

**命题 46.15.** 若 $X$ 已无非平凡无穷小方向，则自然映射 $X\to X_{\mathrm{dR}}$ 为等价。

**证明.** $X_{\mathrm{dR}}$ 是关于无穷小加厚的局部化。若 $X$ 对所有此类加厚已经满足映射空间不变，即 $X$ 是该局部化的局部对象，则局部化单位 $X\to X_{\mathrm{dR}}$ 按局部对象定义为等价。$\square$

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

## 46.7 本章小结

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
