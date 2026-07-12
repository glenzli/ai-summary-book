# 第一章：solid 派生范畴

## 本章目标

本章把第一卷中的 solid 阿贝尔群推广到派生范畴。核心思想是：solid 条件不仅是对象层面的 Hom 判别，也可以作为 $D(\mathbf{CondAb})$ 中的 localization 条件。

## 依赖

需要第一卷第十二章、附录 F 和附录 G。

## 1.0 范畴与 Hom 的类型约定

沿用第一卷附录 A 的固定 \(\kappa\)-层级，故
\(\mathbf{CondAb}=\mathbf{CondAb}_\kappa\)，所有 profinite 测试对象都属于
\(\mathbf{ProFin}_\kappa\)。\(D(\mathbf{CondAb})\) 表示**无界**导出范畴的
稳定 \(\infty\)-范畴增强；写 distinguished triangle 时指其同伦范畴。

对 \(M,C\in D(\mathbf{CondAb})\)，
\(R\operatorname{Hom}(M,C)\) 是 \(D(\mathbf{Ab}_{\mathcal U})\) 中的导出 Hom
复形，而 \(R\underline{\operatorname{Hom}}(M,C)\) 才表示内部凝聚 Hom。符号
“\(\simeq0\)”表示前者是零复形。cone 与 cofiber 采用上同调号差约定；本章不在
三角范畴中把未指定增强的 cone 当作函子。

## 1.1 solid 复形

设 $C\in D(\mathbf{CondAb})$。

**定义 1.1.** 若对任意 \(S\in\mathbf{ProFin}_\kappa\)，自然映射

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

**外部输入定理 1.5（solid 派生识别；Scholze）.** 包含函子

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

更精确地，自然函子

$$
D(\mathbf{Solid})\longrightarrow D(\mathbf{CondAb})
$$

全忠实，其本质像正是定义 1.1 的 solid 复形；\(L^\square\) 是心脏层
solidification \((-)^\square\) 的总左导出函子。

**来源与边界.** 这是 S26 Theorem 5.8(ii)；把普通 Hom 判别提升为内部
\(R\underline{\operatorname{Hom}}\) 判别还使用 S26 Corollary 6.1(iv)。本书不重证
Theorem 5.8 的分解与测度计算，后文只使用这里逐项列出的本质像、伴随和导出函子结论。

**定义 1.6.** $L^\square C$ 称为 $C$ 的派生 solidification。

若 $M$ 是凝聚阿贝尔群，第一卷的 $M^\square$ 是对象层面的 solidification；由
定理 1.5 的“总左导出”陈述，存在自然同构
\(H^0(L^\square(M[0]))\cong M^\square\)。除非另有幅度或正合性结论，本书不从
这个同构擅自推出任意左导出同调群消失。

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

**外部输入定理 1.7（solid 生成元；Scholze）.** 对每个满足
\(|I|<\kappa\) 的集合 \(I\)，对象

$$
P_I=\prod_I\underline{\mathbb Z}
$$

是 \(\mathbf{Solid}\) 中的紧投射对象；这些对象构成生成族，而且每个紧投射 solid
对象都同构于某个 \(P_I\)。相应地，\(D_\square(\mathbb Z)\) 的紧对象恰为可由
这些 \(P_I\) 表示的有界复形。

**来源与边界.** 心脏层结论来自 S26 Theorem 5.8(i) 与 Corollary 6.1(i)，导出紧对象
描述来自 Corollary 6.1(ii)。S26 在大凝聚范畴中允许任意小集合 \(I\)；本书固定
\(\kappa\)-层级的版本只量化 \(|I|<\kappa\)，跨层级结论须再使用附录 A.5。

**使用说明.** 本卷后续用该定理计算 solid 张量积和 solid 模。第一卷已经给出 $P_I$ 出现的原因；第二卷需要 Scholze 的结构定理来保证它们控制整个 solid 范畴。

## 1.6 截断与心脏

**外部输入定理 1.8（cohomology 判别；Scholze）.** 对
\(C\in D(\mathbf{CondAb})\)，以下条件等价：

1. \(C\) solid；
2. 对每个 \(n\in\mathbb Z\)，\(H^n(C)\) 是 solid 阿贝尔群。

因此标准 \(t\)-结构限制到 \(D_\square(\mathbb Z)\)，其心脏等价于
\(\mathbf{Solid}\)。

**来源与边界.** 这是 S26 Theorem 5.8(ii) 的 cohomology 判别，不是仅由定义 1.1
形式推出的结论。

这表示：solid 复形的零次同调对象就是第一卷定义的 solid 阿贝尔群，而高次同调对象仍然在 solid 范畴中。

## 1.7 例子

**例 1.9.** $\mathbb Z^\square[S]$ 是 solid 复形。事实上它是 solid 阿贝尔群，置于次数 $0$ 后满足定义 1.1。

**例 1.10.** 若 $P_I=\prod_I\underline{\mathbb Z}$，则 $P_I$ 是 solid。定理
1.7 还给出一个 \(\kappa\)-小 extremally disconnected 集合 \(S\)，使 \(P_I\) 是
某个 \(\mathbb Z^\square[S]\) 的 retract；这里不需要也不声称预先指定的 \(I\) 总有
一个典范的 \(S\)。

**例 1.11（无限测试对象上的严格差异）.** 令
\(S=\mathbb N\cup\{\infty\}\) 为离散 \(\mathbb N\) 的一点紧化。连续整值函数最终
常值，故

$$
C(S,\mathbb Z)
\cong
\mathbb Z\cdot1\oplus\bigoplus_{n\ge1}\mathbb Z\cdot e_n,
$$

其中 \(e_n\) 是 \(\{n\}\) 的示性函数。因此

$$
\mathbb Z^\square[S](*)
\cong
\operatorname{Hom}(C(S,\mathbb Z),\mathbb Z)
\cong
\mathbb Z\times\prod_{n\ge1}\mathbb Z.
$$

Dirac 映射 \(\mathbb Z[\underline S](*)\to\mathbb Z^\square[S](*)\) 的第二坐标只有
有限支撑，因而不包含 \((0,(1,1,\ldots))\)。所以该映射不是同构，普通自由凝聚群
\(\mathbb Z[\underline S]\) 不是 solid，而其 solidification 是
\(\mathbb Z^\square[S]\)。这也说明有限 \(S\) 的例子不能检测 solid 条件。

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
