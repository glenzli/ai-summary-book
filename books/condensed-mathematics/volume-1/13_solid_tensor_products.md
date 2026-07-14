# 第十三章：固体张量积

两个 solid 对象在 $\mathbf{CondAb}$ 中取普通张量后，不必仍满足测度延拓条件；换言之，
solid 子范畴对旧张量并非先验闭合。若希望在其中谈环、模和派生代数，乘法必须先使用
第九、十一章的普通或派生张量，再施加第十二章的固化反射。这个“张量后局部化”的
次序由泛性质决定，不能用逐点无限乘积的张量公式替代。

我们据此定义 $M\otimes^\square N$ 与
$M\otimes^{L,\square}_{\mathbb Z}N$，并精确标出对称幺半结构和自由 solid 对象乘法
公式所依赖的 Scholze 输入定理。有限离散空间提供可直接核对的模型，固体环与固体模
则展示该结构的代数用途；实分析为何仍超出 solid 完成化，将成为第十四章引入 analytic
ring 的动机。

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

**定理 13.4（Scholze）.** 在本卷固定的宇宙中，对任意集合 $I,J$，有

$$
\left(\prod_I\underline{\mathbb Z}\right)
\otimes^{L,\square}
\left(\prod_J\underline{\mathbb Z}\right)
\simeq
\prod_{I\times J}\underline{\mathbb Z}.
$$

**证明说明.** Scholze 的乘积公式直接给出这一结论。其证明不是把
$\prod_I\underline{\mathbb Z}$ 与某个自由 solid 对象含混地等同，而是先把所需乘积
写成适当自由 solid 对象的 retract，再利用
$\mathbb Z^\square[S]\otimes^{L,\square}\mathbb Z^\square[T]
\simeq\mathbb Z^\square[S\times T]$，最后令 retract 在张量下分裂。固定宇宙（或等价地
固定基数界）保证这里的集合、profinite 测试对象和生成族都落在同一大小约定中。

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

## 13.6 张量后再固化

公式 $M\otimes^\square N=(M\otimes N)^\square$ 保证乘积重新落回 solid 范畴；在
派生层，输入定理进一步给出自由对象的乘法
$\mathbb Z^\square[S]\otimes^{L,\square}\mathbb Z^\square[T]
\simeq\mathbb Z^\square[S\times T]$。因此 solid 环和模可以作为该幺半范畴中的
内部代数对象处理。这个反射只编码特定整值测度条件，并不等同于 Hausdorff 或 Banach
完备化；要改变允许的测度，就必须扩大到下一章的 analytic ring 数据。

## 练习

**练习 13.1.** 说明为什么 $M\otimes N$ 对固体 $M,N$ 不必先验固体，因此定义中需要固化。

**练习 13.2.** 假设 $\mathbb Z^\square[S]\otimes^{L,\square}\mathbb Z^\square[T]\simeq\mathbb Z^\square[S\times T]$，证明有限离散 $S,T$ 情形与普通自由阿贝尔群张量积一致。

**练习 13.3.** 写出固体环定义中的结合律、交换律和单位律。
