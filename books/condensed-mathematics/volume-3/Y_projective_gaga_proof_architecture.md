# 附录 Y：Projective GAGA 的证明结构

## Y.0 目标

Serre GAGA 是卷三的核心输入之一。本附录不重证完整 GAGA，而是把 projective case 的证明拆成可复核的模块，说明每个模块的数学职责。

设 $X$ 是 proper projective $\mathbb C$-scheme，$X^{an}$ 为解析化。解析化函子记为

$$
(-)^{an}:\operatorname{Coh}(X)\to\operatorname{Coh}(X^{an}).
$$

目标是证明：

1. full faithfulness：
   $$
   \operatorname{Hom}_X(\mathcal F,\mathcal G)
   \cong
   \operatorname{Hom}_{X^{an}}(\mathcal F^{an},\mathcal G^{an});
   $$
2. essential surjectivity：每个相干解析层来自唯一代数相干层；
3. cohomology comparison：
   $$
   H^q(X,\mathcal F)\cong H^q(X^{an},\mathcal F^{an}).
   $$

## Y.1 Serre twisting 输入

令 $\mathcal O_X(1)$ 是 projective embedding 给出的 ample line bundle。

**输入定理 Y.1（Serre vanishing 与生成）.** 对每个代数相干层 $\mathcal F$，存在整数 $N$，使对所有 $m\ge N$：

1. $\mathcal F(m)$ 由全局截面生成；
2. $H^q(X,\mathcal F(m))=0$ 对 $q>0$ 成立。

解析侧也有对应输入：

**输入定理 Y.2（解析 Serre 定理）.** 对每个相干解析层 $\mathcal G$，存在 $N$，使 $\mathcal G(m)$ 由全局解析截面生成，且高阶解析上同调消失。

## Y.2 从线丛比较到全局截面比较

**输入定理 Y.3（射影空间结构层上同调比较）.** 对所有整数 $m$ 和所有 $q$，

$$
H^q(\mathbb P^n,\mathcal O(m))
\to
H^q((\mathbb P^n)^{an},\mathcal O(m)^{an})
$$

为同构。

附录 S 计算左侧，并给出解析 Čech 计算的同型模型；该输入要求代数和解析标准覆盖下的 Laurent 单项式描述相容。

**命题 Y.4（有限直和比较）.** 若 cohomology comparison 对 $\mathcal O_X(m)$ 成立，则对有限直和

$$
\mathcal O_X(m)^{\oplus r}
$$

也成立。

**证明.** 上同调与有限直和交换，解析化也与有限直和交换。证毕。

## Y.3 Cohomology comparison 的 devissage

**命题 Y.5.** 设短正合列

$$
0\to\mathcal F'\to\mathcal F\to\mathcal F''\to0
$$

中的任意两项已知 cohomology comparison，则第三项也成立。

**证明.** 代数和解析上同调各自给出长正合列，解析化给出两列之间的交换图。若两项比较为同构，则由 five lemma 得第三项比较为同构。证毕。

**命题 Y.6.** 若每个相干层都可由有限个 $\mathcal O_X(-m)$ 的有限 resolution 表示，且线丛 $\mathcal O_X(k)$ 的 cohomology comparison 已知，则所有代数相干层满足 cohomology comparison。

**证明.** 对 resolution 长度归纳。长度零是 Y.4。一般情形把 resolution 的第一步写成短正合列

$$
0\to\mathcal K\to\mathcal E\to\mathcal F\to0,
$$

其中 $\mathcal E$ 是有限个线丛直和，$\mathcal K$ 有更短 resolution。由归纳假设和 Y.5 得结论。证毕。

## Y.4 Full faithfulness

**命题 Y.7.** 若对所有相干层 $\mathcal H$ 有

$$
H^0(X,\mathcal H)\cong H^0(X^{an},\mathcal H^{an}),
$$

并且解析化与 sheaf Hom 相容：

$$
\mathcal Hom_X(\mathcal F,\mathcal G)^{an}
\cong
\mathcal Hom_{X^{an}}(\mathcal F^{an},\mathcal G^{an}),
$$

则解析化函子 full faithful。

**证明.** 有

$$
\operatorname{Hom}_X(\mathcal F,\mathcal G)
=
H^0(X,\mathcal Hom_X(\mathcal F,\mathcal G)).
$$

使用 $H^0$ 比较和 sheaf Hom 比较，得到

$$
H^0(X^{an},\mathcal Hom_{X^{an}}(\mathcal F^{an},\mathcal G^{an}))
=
\operatorname{Hom}_{X^{an}}(\mathcal F^{an},\mathcal G^{an}).
$$

证毕。

## Y.5 Essential surjectivity

**输入定理 Y.8（解析相干层的有限扭充分表示）.** 对 projective $X$，每个相干解析层 $\mathcal G$ 存在整数 $m_0,m_1$ 和解析有限 presentation

$$
\mathcal O_{X^{an}}(-m_1)^{\oplus r_1}
\to
\mathcal O_{X^{an}}(-m_0)^{\oplus r_0}
\to
\mathcal G
\to0.
$$

这是解析侧 Serre 生成定理和相干层有限表示性的结合。

**命题 Y.9（essential surjectivity 的归纳骨架）.** 若 Y.8 成立，且 full faithfulness 已知，则每个相干解析层来自代数相干层。

**证明.** 由 Y.8 取解析有限 presentation

$$
\mathcal O_{X^{an}}(-m_1)^{\oplus r_1}
\to
\mathcal O_{X^{an}}(-m_0)^{\oplus r_0}
\to
\mathcal G\to0.
$$

full faithfulness 说明第一箭头来自唯一代数态射

$$
\mathcal O_X(-m_1)^{\oplus r_1}
\to
\mathcal O_X(-m_0)^{\oplus r_0}.
$$

令其 cokernel 为 $\mathcal F$。解析化保持 cokernel，故 $\mathcal F^{an}\cong\mathcal G$。证毕。

## Y.6 condensed/analytic GAGA 的边界

condensed/analytic 版本的 GAGA 需要在上述 classical GAGA 外再检查：

1. 解析化后的函数空间拓扑与 analytic/liquid 对象相容。
2. $R\Gamma$ 比较不仅是向量空间 quasi-isomorphism，还在目标派生范畴中成立。
3. tensor、Hom、trace、dualizing object 与比较函子相容。
4. properness 保证全局函数和上同调没有无穷远逃逸。

附录 Q 已给出 non-proper 反例；本附录说明 proper projective 情形中 classical GAGA 的证明模块。

## 练习

1. 用 Y.5 写出短正合列中 five lemma 的具体交换图。
2. 证明 Y.7 中 $\mathcal Hom$ 相容足以推出 full faithfulness。
3. 在 $X=\mathbb P^1$ 上，用 $\mathcal O(d)$ 的上同调比较验证 Y.3。
4. 解释 properness 在 Y.6 的四条边界中分别控制什么问题。
