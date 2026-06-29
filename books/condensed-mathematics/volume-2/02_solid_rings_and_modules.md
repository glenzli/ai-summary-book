# 第二章：solid 环与 solid 模

## 本章目标

第一章把 solid 对象组织成派生范畴。本章讨论其中的代数对象：solid 环、solid 模以及派生 solid 张量积。

## 依赖

需要第二卷第一章和第一卷第九至十三章。

## 2.1 solid 张量积

设 $M,N\in D_{\square}(\mathbb Z)$。

**定义 2.1.** 派生 solid 张量积定义为

$$
M\otimes_{\mathbb Z}^{L,\square}N
=
L^\square(M\otimes_{\mathbb Z}^{L}N).
$$

这里右侧先在 $D(\mathbf{CondAb})$ 中取普通派生张量积，再做派生 solidification。

**输入定理 2.2（Scholze）.** $D_{\square}(\mathbb Z)$ 在 $\otimes_{\mathbb Z}^{L,\square}$ 下成为闭对称幺半范畴，单位对象是 $\mathbb Z^\square$。

**证明说明.** 关键是 solidification 与张量积相容，并且 localization 由张量稳定的一族态射生成。本卷后续会多次使用该定理。

## 2.2 自由 solid 对象的乘法公式

**输入定理 2.3（Scholze）.** 若 $S,T$ 是 profinite 集合，则

$$
\mathbb Z^\square[S]\otimes_{\mathbb Z}^{L,\square}
\mathbb Z^\square[T]
\simeq
\mathbb Z^\square[S\times T].
$$

特别地，若

$$
\mathbb Z^\square[S]\cong\prod_I\underline{\mathbb Z},
\qquad
\mathbb Z^\square[T]\cong\prod_J\underline{\mathbb Z},
$$

则

$$
\left(\prod_I\underline{\mathbb Z}\right)
\otimes_{\mathbb Z}^{L,\square}
\left(\prod_J\underline{\mathbb Z}\right)
\simeq
\prod_{I\times J}\underline{\mathbb Z}.
$$

**使用说明.** 这个公式是 solid 张量积不同于普通张量积的核心。普通阿贝尔群中，任意乘积与张量积通常不满足这样的乘积公式；solidification 修正了这一点。

## 2.3 solid 环

**定义 2.4.** solid 环是 $D_{\square}(\mathbb Z)$ 中的交换代数对象。若只讨论心脏中的对象，也可说它是 $\mathbf{Solid}$ 中的交换环对象。

显式地，一个 ordinary solid 环 $R$ 包含：

1. solid 阿贝尔群 $R$。
2. 乘法
   $$
   R\otimes^\square R\to R.
   $$
3. 单位
   $$
   \mathbb Z^\square\to R.
   $$
4. 结合律、交换律和单位律。

## 2.4 solid 模

**定义 2.5.** 若 $R$ 是 solid 环，则 solid $R$-模是 $\mathbf{Solid}$ 中的 $R$-模对象。其范畴记为

$$
R\text{-}\mathbf{Mod}_{\square}.
$$

派生版本记为

$$
D_{\square}(R).
$$

**输入定理 2.6（Scholze）.** 对 solid 环 $R$，$D_{\square}(R)$ 是可展示稳定范畴；若 $R$ 是交换 solid 环，则 $D_{\square}(R)$ 继承闭对称幺半结构，并带有相对派生 solid 张量积

$$
-\otimes_R^{L,\square}-.
$$

这里的可展示性、稳定性和闭对称幺半结构依赖第二卷输入定理 B.2-B.3；本章只使用这些列出的范畴性质。

## 2.5 例子

**例 2.7.** $\mathbb Z^\square$ 是初始 solid 环。$\mathbb Z^\square$-模就是 solid 阿贝尔群。

**例 2.8.** 对 $p$-进整数，$\mathbb Z_p^\square$ 是 solid 环。它的模范畴适合表达 $p$-进完备代数对象。

**例 2.9.** 若 $A$ 是有限生成 $\mathbb Z$-代数，则第一卷中的

$$
A^\square[S]=\varprojlim_iA[S_i]
$$

给出 solid/analytic 方向的基本例子。严格地说，后续需要区分 solid 环 $A^\square$ 与解析环 $(A,A^+)^\square$。

## 2.6 与 ordinary completion 的区别

solidification 不是普通拓扑环的 Hausdorff completion。它是凝聚范畴中的左伴随或派生 localization。

因此，下列说法需要区分：

1. $p$-进完备化：由理想 $(p)$ 的逆极限控制。
2. solidification：由 $\mathbb Z[\underline S]\to\mathbb Z^\square[S]$ 的 localization 控制。
3. analytic localization：由一套允许测度对象 $\mathcal M[S]$ 控制。

这些过程在例子中会相互作用，但定义不同。

## 2.7 本章小结

solid 环和 solid 模把第一章的派生 solid 范畴变成代数环境。下一步要做的是把 solid 的“测度对象”推广为一般解析环的测度理论。

## 练习

**练习 2.1.** 证明有限离散 $S,T$ 情形下，定理 2.3 退化为普通自由阿贝尔群张量公式。

**练习 2.2.** 写出 ordinary solid 环定义中的结合律交换图。

**练习 2.3.** 解释为什么普通张量积 $\prod_I\mathbb Z\otimes\prod_J\mathbb Z$ 不应直接等同于 $\prod_{I\times J}\mathbb Z$。

**练习 2.4.** 比较 $p$-进完备化和 solidification 的泛性质。
