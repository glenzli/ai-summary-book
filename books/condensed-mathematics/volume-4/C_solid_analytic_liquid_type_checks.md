# 附录 C：solid、analytic、liquid 的类型检查

## C.1 为什么需要类型检查

第四卷中的许多公式外观相似：

$$
M\otimes N,\qquad
M\otimes^LN,\qquad
M\otimes^{L,\square}N.
$$

它们属于不同范畴，含义不同。计算前必须回答：

1. 对象在哪个范畴中：$\mathbf{Ab}$、$\mathbf{CondAb}$、solid 模、analytic 模，还是 liquid 向量空间？
2. 张量积是否导出？
3. 是否需要 solidification、analytic localization 或 liquid localization？
4. 无限极限是否在该范畴中保持？

如果这些问题没有回答，公式即使形式上正确，也可能类型错误。

## C.2 solid 张量积的有限层检查

对有限集合 $S,T$，

$$
\mathbb Z^\square[S]\otimes^{L,\square}
\mathbb Z^\square[T]
\simeq
\mathbb Z[S\times T].
$$

类型检查：

1. $\mathbb Z^\square[S]=\mathbb Z[S]$，因为 $S$ 有限。
2. $\mathbb Z[S]$ 是有限自由对象，故无高阶导出 Tor。
3. 有限自由对象已经 solid，solidification 不改变结果。
4. 普通自由模公式给出 $\mathbb Z[S]\otimes\mathbb Z[T]\cong\mathbb Z[S\times T]$。

因此有限层公式是完全初等的。

## C.3 profinite 层的输入定理位置

对 profinite $S,T$，

$$
\mathbb Z^\square[S]\otimes^{L,\square}
\mathbb Z^\square[T]
\simeq
\mathbb Z^\square[S\times T]
$$

不是由 C.2 直接推出的普通代数命题。证明需要输入：

1. profinite 集合表示为有限集合逆极限；
2. solid 自由对象与这些逆极限相容；
3. solid 导出张量积与对应的完备极限相容；
4. 高阶导出项按 solid 理论消失或被正确控制。

教材中可把它作为 solid tensor 的基本定理使用。若要形式化，应先把它作为 theorem 参数，而不是从普通张量积库硬推。

## C.4 普通张量积反例的类型意义

反例

$$
\left(\prod_n\mathbb Z\right)\otimes\mathbb Q
\not\cong
\prod_n\mathbb Q
$$

说明 $\mathbf{Ab}$ 中的张量积不能满足 solid 计算需要的无限乘积行为。solidification 的目标之一正是让 profinite/测度型对象拥有正确张量行为。

这并不意味着普通张量积“错误”；它只是在错误范畴中回答了另一个问题。

## C.5 analytic ring 的有限和无限检查

对 analytic ring 的例子，有限层计算通常为

$$
A^\square[S]\cong A[S]
\quad(S\text{ finite}).
$$

无限 profinite 情形写作

$$
A^\square[S]=\varprojlim_iA[S_i],
\qquad S=\varprojlim_iS_i.
$$

类型检查重点：

1. $S_i$ 是否有限；
2. 极限在哪个范畴中取；
3. $A$ 是否带有额外解析结构；
4. 若来自 Huber pair，是否保留 $A^+$；
5. rational localization 是否在 analytic ring 范畴中进行。

## C.6 Huber pair 的检查表

给定离散 Huber pair $(A,A^+)$，写出 $(A,A^+)^\square$ 时应记录：

1. 底层环 $A$；
2. 有界子环 $A^+$；
3. 允许的 rational localization；
4. 解析模范畴；
5. 与普通 $A$-模范畴的比较函子。

若只写 $A^\square$，则默认没有记录 $A^+$ 的几何边界条件。第二卷中的相干性和 $f^!$ 问题通常依赖这些边界条件。

## C.7 liquid 不是 Banach 完备化

Banach 完备化处理 Cauchy 列：

$$
V\mapsto\widehat V.
$$

liquid localization 处理相对于测度测试对象的范畴性质。典型判别涉及

$$
\operatorname{Hom}(\mathcal M_{<p}[S],V).
$$

因此二者层级不同：

1. Banach 完备化是对象内部的拓扑完备化。
2. liquid 条件是对象在一个凝聚/解析范畴中对测试对象的响应。
3. liquid 张量积是范畴级张量结构，不是简单的范数张量。

## C.8 Dolbeault 复形的类型检查

第三卷中出现的 Dolbeault 复形进入第四卷的计算语境时，应检查：

1. 每项 $\mathcal A^{p,q}(X)$ 的拓扑向量空间结构；
2. $\bar\partial$ 的连续性；
3. 凝聚化是否逐项进行；
4. 进入 liquid/analytic 范畴时是否需要局部化；
5. 取同调时是在代数、拓扑、凝聚还是 liquid 导出范畴中。

只有这些层级明确后，Serre duality、Dolbeault resolution 和 GAGA 比较才能在凝聚语言中保持正确类型。
