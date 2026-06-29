# 第一章：solid 派生范畴

## 本章目标

本章把第一卷中的 solid 阿贝尔群推广到派生范畴。核心思想是：solid 条件不仅是对象层面的 Hom 判别，也可以作为 $D(\mathbf{CondAb})$ 中的 localization 条件。

## 依赖

需要第一卷第十二章、附录 F 和附录 G。

## 1.1 solid 复形

设 $C\in D(\mathbf{CondAb})$。

**定义 1.1.** 若对任意 profinite 集合 $S$，自然映射

$$
R\operatorname{Hom}(\mathbb Z^\square[S],C)
\longrightarrow
R\operatorname{Hom}(\mathbb Z[\underline S],C)
$$

是同构，则称 $C$ 为 solid 复形。

右侧可写为

$$
R\Gamma(S,C).
$$

因此 solid 复形的条件是：从 $S$ 到 $C$ 的派生截面已经自动对整值测度连续延拓。

## 1.2 solid 派生范畴

**定义 1.2.** solid 派生范畴记为

$$
D_{\square}(\mathbb Z)
\subset D(\mathbf{CondAb}),
$$

它是所有 solid 复形构成的全子范畴。

**命题 1.3.** 若 $C\simeq C'$ 在 $D(\mathbf{CondAb})$ 中同构，则 $C$ solid 当且仅当 $C'$ solid。

**证明.** 定义 1.1 只涉及导出范畴中的 $R\operatorname{Hom}$ 和同构条件。同构对象给出的两个 Hom 复形自然同构，因此 solid 条件保持。证毕。

## 1.3 localization 观点

对每个 profinite 集合 $S$，有自然态射

$$
\mathbb Z[\underline S]\to\mathbb Z^\square[S].
$$

记其 cone 为

$$
K_S=\operatorname{Cone}(\mathbb Z[\underline S]\to\mathbb Z^\square[S]).
$$

**命题 1.4.** 复形 $C$ solid 当且仅当对所有 profinite $S$，

$$
R\operatorname{Hom}(K_S,C)\simeq0.
$$

**证明.** 对三角

$$
\mathbb Z[\underline S]\to\mathbb Z^\square[S]\to K_S\to
$$

应用 $R\operatorname{Hom}(-,C)$，得到三角

$$
R\operatorname{Hom}(K_S,C)\to
R\operatorname{Hom}(\mathbb Z^\square[S],C)\to
R\operatorname{Hom}(\mathbb Z[\underline S],C)\to.
$$

中间箭头为同构当且仅当前一项为零。证毕。

这说明 solid 复形是对一族态射

$$
\mathbb Z[\underline S]\to\mathbb Z^\square[S]
$$

局部化后的局部对象。

## 1.4 solidification

**输入定理 1.5（Scholze）.** 包含函子

$$
D_{\square}(\mathbb Z)\hookrightarrow D(\mathbf{CondAb})
$$

有左伴随

$$
L^\square:D(\mathbf{CondAb})\to D_{\square}(\mathbb Z).
$$

并且 \(L^\square\) 是使所有映射

$$
\mathbb Z[\underline S]\to\mathbb Z^\square[S]
$$

变为等价的反射 Bousfield localization：对任意 solid 对象 \(C\)，自然映射

$$
R\operatorname{Hom}(L^\square M,C)\to R\operatorname{Hom}(M,C)
$$

为等价。

**定义 1.6.** $L^\square C$ 称为 $C$ 的派生 solidification。

若 $M$ 是凝聚阿贝尔群，第一卷的 $M^\square$ 是对象层面的 solidification；它与 \(H^0(L^\square M)\) 的相容性作为输入定理 D.1 的一部分使用。

## 1.5 solid 对象的生成元

第一卷附录 F 证明：对任意 profinite 集合 $S$，存在集合 $I$ 使

$$
\mathbb Z^\square[S]\cong\prod_I\underline{\mathbb Z}.
$$

因此 solid 理论可用乘积型对象

$$
P_I=\prod_I\underline{\mathbb Z}
$$

来测试。

**输入定理 1.7（Scholze）.** 对所有集合 $I$，对象 $P_I$ 构成 $D_{\square}(\mathbb Z)$ 的紧投射生成族的心脏层影子；在阿贝尔范畴 $\mathbf{Solid}$ 中，它们给出投射生成元。

**使用说明.** 本卷后续用该定理计算 solid 张量积和 solid 模。第一卷已经给出 $P_I$ 出现的原因；第二卷需要 Scholze 的结构定理来保证它们控制整个 solid 范畴。

## 1.6 截断与心脏

**输入定理 1.8（Scholze）.** $D_{\square}(\mathbb Z)$ 带有与 $D(\mathbf{CondAb})$ 相容的 $t$-结构，其心脏等价于 $\mathbf{Solid}$。

这表示：solid 复形的零次同调对象就是第一卷定义的 solid 阿贝尔群，而高次同调对象仍然在 solid 范畴中。

## 1.7 例子

**例 1.9.** $\mathbb Z^\square[S]$ 是 solid 复形。事实上它是 solid 阿贝尔群，置于次数 $0$ 后满足定义 1.1。

**例 1.10.** 若 $P_I=\prod_I\underline{\mathbb Z}$，则 $P_I$ 是 solid。由第一卷附录 F，它同构于某个 $\mathbb Z^\square[S]$ 型对象，或至少由这些对象生成。

**例 1.11.** 普通自由凝聚阿贝尔群 $\mathbb Z[\underline S]$ 对无限 profinite $S$ 一般不是 solid；它的 solidification 是 $\mathbb Z^\square[S]$。

## 1.8 本章小结

本章把 solid 条件写成 localization 条件：

$$
R\operatorname{Hom}(K_S,C)=0.
$$

这使 solidification 成为派生范畴中的左伴随，并为下一章的 solid 张量积和 solid 模奠定基础。

## 练习

**练习 1.1.** 证明命题 1.4 中的等价。

**练习 1.2.** 对有限离散 $S$，说明 $K_S\simeq0$。

**练习 1.3.** 若 $C$ solid，解释为什么 $C[n]$ 仍 solid。

**练习 1.4.** 设 $M$ 是凝聚阿贝尔群。写出 $M\to L^\square M$ 的伴随泛性质。
