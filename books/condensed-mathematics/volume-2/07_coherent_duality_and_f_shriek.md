# 第七章：$f_!$、投影公式与相干对偶

## 本章目标

本章解释凝聚数学如何处理非 proper 映射中的紧支撑推前。核心对象是

$$
f_!
$$

以及它的右伴随 $f^!$。

## 依赖

需要第二、三、六章。

## 7.1 问题背景

在经典代数几何中，proper 映射 $f:X\to Y$ 有良好的推前 $Rf_*T$。非 proper 映射的问题在于：普通推前不能自动记录边界行为。

凝聚数学的策略是把边界行为放入 solid/analytic 模范畴中，并构造紧支撑推前

$$
f_!.
$$

## 7.2 仿射有限型情形

设 $A$ 是有限生成 $\mathbb Z$-代数，并令

$$
f:\operatorname{Spec}A\to\operatorname{Spec}\mathbb Z.
$$

对应的 solid/analytic 环为

$$
A^\square,
\qquad
\mathbb Z^\square.
$$

**输入定理 7.1（Scholze）.** 存在函子

$$
f_!:D(A^\square)\to D(\mathbb Z^\square)
$$

满足：

1. 与直接和相容。
2. 保持紧对象。
3. 满足投影公式
   $$
   f_!\left((M\otimes_{\mathbb Z^\square}^LA^\square)
   \otimes_{A^\square}^LN\right)
   \simeq
   M\otimes_{\mathbb Z^\square}^Lf_!N.
   $$
4. 有右伴随
   $$
   f^!:D(\mathbb Z^\square)\to D(A^\square).
   $$

## 7.3 边界项

对 $A=\mathbb Z[T]$，映射

$$
\mathbb A^1_{\mathbb Z}\to\operatorname{Spec}\mathbb Z
$$

不是 proper。紧支撑推前应当记录无穷远边界。

Scholze 的构造通过比较 $A^\square$ 与相对解析环，并引入边界控制项。直观上，$f_!$ 不是简单地把 $A$-模忘成 $\mathbb Z$-模，而是把“在无穷远处的增长或支撑条件”纳入 solid/analytic 框架。

## 7.4 投影公式

投影公式说明 $f_!$ 与张量积相容。设

$$
M\in D(\mathbb Z^\square),
\qquad
N\in D(A^\square).
$$

则

$$
f_!(f^*M\otimes_{A^\square}^LN)
\simeq
M\otimes_{\mathbb Z^\square}^Lf_!N,
$$

其中

$$
f^*M=M\otimes_{\mathbb Z^\square}^LA^\square.
$$

这个公式是相干对偶的核心技术条件。

## 7.5 右伴随 $f^!$

若 $f_!$ 保持足够的 colimit，并满足紧性条件，则可用 Brown representability 或紧生成范畴理论得到右伴随

$$
f^!.
$$

**命题 7.2.** 假设 $f_!$ 是可展示稳定范畴之间的可访问余极限保持函子，则 $f_!$ 有右伴随。

**证明.** 这是 presentable adjoint functor theorem 的直接应用；见附录 F 推论 F.3。若只使用三角范畴语言，可用附录 F 定理 F.5 的 Brown representability 版本。证毕。

附录 F 还证明：一旦投影公式成立，右伴随 $f^!$ 与内部 Hom 满足

$$
f^!\mathcal Hom(M,Y)
\simeq
\mathcal Hom(f^*M,f^!Y).
$$

该公式的闭幺半背景、dualizable/perfect 假设和失败边界见附录 L。

## 7.6 相干对偶图景

在 proper 情形中，Grothendieck duality 研究

$$
Rf_*,\quad f^!,\quad \operatorname{Tr},\quad \text{projection formula}.
$$

在凝聚数学中，非 proper 情形使用

$$
f_!
$$

替代 proper pushforward，并把边界贡献放入 analytic/solid 模范畴。

## 7.7 本章小结

本章给出 $f_!$ 的第二卷入口。完整构造依赖 Scholze 讲义的相干对偶部分；本卷保留其形式性质，并在第八章解释它如何服务于复几何。

## 练习

**练习 7.1.** 写出 proper 情形中 $f_*$ 与非 proper 情形中 $f_!$ 的直观区别。

**练习 7.2.** 验证投影公式中 $f^*M=M\otimes_{\mathbb Z^\square}^LA^\square$ 的类型正确。

**练习 7.3.** 说明为什么 $A=\mathbb Z[T]$ 会出现无穷远边界。

**练习 7.4.** 查阅第一卷附录 G，解释为什么投影公式应当在派生范畴中陈述。
