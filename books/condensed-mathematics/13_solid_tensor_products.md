# 第十三章：固体张量积

## 本章目标

本章定义固体阿贝尔群上的完备张量积，并说明它为何比普通凝聚张量积更适合处理完备拓扑代数对象。

## 依赖前置知识

需要第九章的张量积、第十一章的派生张量积、第十二章的固化函子。

## 13.1 完备张量积

设 $M,N\in\mathbf{Solid}$。

**定义 13.1.** $M$ 与 $N$ 的固体张量积定义为

$$
M\otimes^\square N
=
(M\otimes N)^\square.
$$

其中右侧先在 $\mathbf{CondAb}$ 中取张量积，再固化。

若考虑派生范畴，则定义派生固体张量积为

$$
M\otimes_{\mathbb Z}^{L,\square}N
=
(M\otimes_{\mathbb Z}^{L}N)^{L\square},
$$

其中 $(-)^{L\square}$ 是固化函子的左导出函子。

**注 13.2.** 这个定义表达了“先张量，再完备化”的思想。普通张量积可能离开固体范畴，因此需要再固化。

## 13.2 对称幺半结构

**定理 13.3（Scholze）.** $\mathbf{Solid}$ 上的 $\otimes^\square$ 使其成为对称幺半范畴。派生范畴 $D(\mathbf{Solid})$ 上有相应的派生对称幺半结构。

**证明说明.** 关键是固化函子与张量积相容，即自然映射

$$
(M\otimes N)^\square
\to
(M^\square\otimes N^\square)^\square
$$

为同构。Scholze 通过把问题化到自由凝聚对象 $\mathbb Z[\underline S]$ 和 $\mathbb Z[\underline T]$，再使用固体对象的定义与派生判别证明该结论。

## 13.3 自由固体对象上的计算

Scholze 讲义给出如下自然同构：若 $S,T$ 是 profinite 集合，则

$$
\mathbb Z^\square[S]\otimes^{L,\square}\mathbb Z^\square[T]
\simeq
\mathbb Z^\square[S\times T].
$$

当 $S,T$ 极不连通时，这个同构尤其像“测度的外积”。

**定理 13.4.** 对集合 $I,J$，有

$$
\left(\prod_I\underline{\mathbb Z}\right)
\otimes^{L,\square}
\left(\prod_J\underline{\mathbb Z}\right)
\simeq
\prod_{I\times J}\underline{\mathbb Z}.
$$

**证明说明.** 由第十二章，$\prod_I\mathbb Z$ 可由某个 $\mathbb Z^\square[S]$ 型对象控制。Scholze 讲义第六讲证明了该公式；它是固体张量积行为良好的核心例子。

## 13.4 固体环与固体模

**定义 13.5.** 固体环是 $\mathbf{Solid}$ 中的交换环对象，即固体阿贝尔群 $R$，配备乘法和单位

$$
R\otimes^\square R\to R,
\qquad
\mathbb Z^\square\to R.
$$

这里 $\mathbb Z^\square$ 表示 $\underline{\mathbb Z}$ 的固化；它是固体张量结构的单位对象。

**定义 13.6.** 若 $R$ 是固体环，固体 $R$-模是 $\mathbf{Solid}$ 中的 $R$-模对象。

这给出范畴

$$
\mathbf{SolidMod}_R.
$$

当 $R=\underline{\mathbb Z}^\square$ 时，$\mathbf{SolidMod}_R$ 就是 $\mathbf{Solid}$。

## 13.5 固体化与完成

若 $M$ 是凝聚阿贝尔群，$M^\square$ 可看作 $M$ 的非阿基米德型完备化。这个说法需要谨慎：它不是普通拓扑群的 Hausdorff 完备化，也不是 Banach 空间完备化。

例如，在 Scholze 的讲义中，这种固体完备化对实数方向并不合适；处理实分析需要 analytic rings 和 liquid/analytic 结构。

因此，本书采用如下原则：

- 对非阿基米德或代数型完备对象，solid 结构是自然的。
- 对实数和泛函分析对象，solid 只是入口，后续需要 analytic ring。

## 13.6 本章小结

本章建立了 solid 代数的张量结构：

1. 固体张量积定义为普通张量积后的固化。
2. 自由固体对象满足良好的乘法公式。
3. 固体环和固体模是 solid 范畴中的环对象和模对象。
4. solid 完备化不是所有拓扑完备化的统一替代品，analytic rings 会进一步修正这个框架。

## 练习

**练习 13.1.** 说明为什么 $M\otimes N$ 对固体 $M,N$ 不必先验固体，因此定义中需要固化。

**练习 13.2.** 假设 $\mathbb Z^\square[S]\otimes^{L,\square}\mathbb Z^\square[T]\simeq\mathbb Z^\square[S\times T]$，证明有限离散 $S,T$ 情形与普通自由阿贝尔群张量积一致。

**练习 13.3.** 写出固体环定义中的结合律、交换律和单位律。
