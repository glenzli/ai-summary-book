# 第四章：解析化与 Bousfield localization

## 本章目标

本章解释解析化函子如何作为 Bousfield localization 出现。核心思想是：解析化强制所有 Dirac-to-measure 映射

$$
A[\underline S]\to\mathcal M[S]
$$

在派生范畴中变成同构。

幺半 Bousfield localization、核为张量理想的判别和相对张量积下降公式见附录 K。

## 依赖

需要第三章的预解析环与解析复形。

## 4.1 局部对象与零化对象

固定解析环 $(A,\mathcal M)$。设

$$
K_S^{\mathcal M}
=
\operatorname{Cone}(A[\underline S]\to\mathcal M[S]).
$$

**定义 4.1.** 对象 $C\in D(A)$ 称为 $\mathcal M$-局部对象，如果

$$
R\operatorname{Hom}_A(K_S^{\mathcal M},C)\simeq0
$$

对所有极不连通 $S$ 成立。

**定义 4.2.** 对象 $N\in D(A)$ 称为 $\mathcal M$-零化对象，如果对所有 $\mathcal M$-局部对象 $C$，

$$
R\operatorname{Hom}_A(N,C)\simeq0.
$$

## 4.2 解析化函子

**输入定理 4.3（Scholze）.** 对解析环 $(A,\mathcal M)$，包含函子

$$
D(A,\mathcal M)\hookrightarrow D(A)
$$

有左伴随

$$
L_{(A,\mathcal M)}:D(A)\to D(A,\mathcal M).
$$

该函子称为解析化。

**泛性质 4.4.** 对任意 $C\in D(A)$ 和任意解析对象 $N\in D(A,\mathcal M)$，有自然同构

$$
R\operatorname{Hom}_{D(A,\mathcal M)}(L_{(A,\mathcal M)}C,N)
\simeq
R\operatorname{Hom}_{D(A)}(C,N).
$$

**证明.** 这是左伴随的定义。证毕。

## 4.3 Bousfield localization 表述

**命题 4.5.** 若解析化函子存在，则对任意 $C$，cone

$$
\operatorname{Cone}(C\to L_{(A,\mathcal M)}C)
$$

是 $\mathcal M$-零化对象。

**证明.** 设 $N$ 是解析对象。由泛性质 4.4，

$$
R\operatorname{Hom}(L_{(A,\mathcal M)}C,N)
\to
R\operatorname{Hom}(C,N)
$$

是同构。把它放入 $R\operatorname{Hom}(-,N)$ 作用于三角

$$
C\to L_{(A,\mathcal M)}C\to Q\to
$$

所得的长三角，得到

$$
R\operatorname{Hom}(Q,N)\simeq0.
$$

证毕。

**命题 4.6.** 对所有极不连通 $S$，

$$
L_{(A,\mathcal M)}A[\underline S]\simeq L_{(A,\mathcal M)}\mathcal M[S].
$$

**证明.** 解析化强制 $K_S^{\mathcal M}$ 为零。由三角

$$
A[\underline S]\to\mathcal M[S]\to K_S^{\mathcal M}\to
$$

应用 $L_{(A,\mathcal M)}$，得到 $L_{(A,\mathcal M)}K_S^{\mathcal M}\simeq0$，因此前两项同构。证毕。

## 4.4 解析张量积

若 $A$ 交换，则 $D(A)$ 有派生张量积 $\otimes_A^L$。

**输入定理 4.7（Scholze）.** $D(A,\mathcal M)$ 在

$$
C\otimes_{(A,\mathcal M)}^L D
=
L_{(A,\mathcal M)}(C\otimes_A^L D)
$$

下成为闭对称幺半范畴。

这一定理要求 localization 与张量积相容。它是 analytic rings 的结构定理之一。

## 4.5 与 solidification 的比较

对 $(A,\mathcal M)=(\mathbb Z,\mathbb Z^\square)$，

$$
L_{(A,\mathcal M)}=L^\square.
$$

于是解析化就是派生 solidification，解析张量积就是派生 solid 张量积。

## 4.6 本章小结

解析化是通过杀掉

$$
\operatorname{Cone}(A[\underline S]\to\mathcal M[S])
$$

来实现的 localization。这个观点统一解释了 solidification、analytic modules 和后续 liquid 结构。

## 练习

**练习 4.1.** 证明命题 4.5。

**练习 4.2.** 在 solid 例子中写出 $K_S^{\mathcal M}$。

**练习 4.3.** 说明为什么解析张量积必须在普通张量积后再解析化。

**练习 4.4.** 比较 Bousfield localization 与阿贝尔范畴中的反射子范畴。
