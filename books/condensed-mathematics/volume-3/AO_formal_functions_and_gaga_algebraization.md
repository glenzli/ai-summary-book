# 附录 AO：形式函数、形式 GAGA 与代数化

## AO.0 目标

附录 AI 给出 projective GAGA 的 graded module 证明骨架。本附录补充另一条标准路线：形式函数定理、Grothendieck existence 和解析形式邻域的同一性。

本附录的功能是把 GAGA 中的“代数化”拆成明确的三个输入：形式函数、形式相干层代数化、解析-形式比较。

## AO.1 形式完备化

设 \(A\) 是 Noetherian 环，\(I\subset A\) 为理想，\(X\to\operatorname{Spec}A\) 为 proper morphism。记

$$
A_n=A/I^{n+1},\qquad X_n=X\times_A A_n.
$$

对 \(X\) 上 coherent sheaf \(\mathcal F\)，记

$$
\mathcal F_n=\mathcal F\otimes_A A_n.
$$

形式完备化为

$$
\widehat X=\varprojlim X_n,\qquad
\widehat{\mathcal F}=\{\mathcal F_n\}_{n\ge0}.
$$

**定义 AO.1.** \(\operatorname{Coh}(\widehat X)\) 的对象是相容系统

$$
\{\mathcal G_n\in\operatorname{Coh}(X_n)\}_{n\ge0}
$$

以及同构

$$
\mathcal G_{n+1}\otimes_{A_{n+1}}A_n\cong\mathcal G_n.
$$

## AO.2 形式函数定理

**输入定理 AO.2（theorem on formal functions）.** 若 \(f:X\to\operatorname{Spec}A\) proper，\(\mathcal F\) coherent，则对每个 \(q\) 有自然同构

$$
\widehat{R^qf_\ast\mathcal F}
\cong
\varprojlim_n H^q(X_n,\mathcal F_n),
$$

其中左侧是 \(I\)-adic completion。

**推论 AO.3（cohomology 比较的完备形式）.** 若 \(A\) 是 \(I\)-adically separated，且 \(R^qf_\ast\mathcal F\) 有限生成，则 \(R^qf_\ast\mathcal F\) 由所有

$$
H^q(X_n,\mathcal F_n)
$$

的相容系统决定。

**证明.** 有限生成 \(A\)-模 \(M\) 满足 Krull intersection

$$
M\hookrightarrow\widehat M
$$

在 \(I\)-adically separated 条件下单射。AO.2 识别 \(\widehat M\) 与形式层上同调的逆极限。证毕。

## AO.3 Grothendieck existence

**输入定理 AO.4（Grothendieck existence theorem）.** 若 \(f:X\to\operatorname{Spec}A\) proper，\(A\) Noetherian 且 \(I\)-adically complete，则完备化函子

$$
\operatorname{Coh}(X)\to\operatorname{Coh}(\widehat X)
$$

是范畴等价。

**命题 AO.5（full faithfulness 的形式证明）.** 在 AO.4 的假设下，对 coherent sheaves \(\mathcal F,\mathcal G\)，自然映射

$$
\operatorname{Hom}_X(\mathcal F,\mathcal G)
\to
\varprojlim_n\operatorname{Hom}_{X_n}(\mathcal F_n,\mathcal G_n)
$$

为同构。

**证明.** 令

$$
\mathcal H=\mathcal Hom_X(\mathcal F,\mathcal G).
$$

由于 \(\mathcal F,\mathcal G\) coherent 且 \(X\) Noetherian，\(\mathcal H\) coherent。Hom 集等于 \(H^0(X,\mathcal H)\)。对 \(q=0\) 应用 AO.2，并用有限生成模的完备化，得到

$$
\widehat{H^0(X,\mathcal H)}
\cong
\varprojlim_nH^0(X_n,\mathcal H_n).
$$

AO.4 保证完备化函子 full faithful，因此该完备映射反映并给出全部 Hom。证毕。

**命题 AO.6（essential surjectivity 的形式内容）.** 给定相容系统 \(\{\mathcal G_n\}\in\operatorname{Coh}(\widehat X)\)，AO.4 给唯一的 \(\mathcal G\in\operatorname{Coh}(X)\)，使

$$
\mathcal G\otimes_A A_n\cong\mathcal G_n
$$

对所有 \(n\) 成立。

**证明.** 这正是 AO.4 的本质满性。唯一性来自 full faithfulness。证毕。

## AO.4 解析形式邻域与 GAGA

设 \(X\) 是 \(\mathbb C\) 上 proper scheme，\(X^{an}\) 为其解析化。对闭子空间 \(Z\subset X\)，记 \(\widehat X_Z\) 与 \(\widehat{X^{an}}_{Z^{an}}\) 为形式邻域。

**输入定理 AO.7（解析-代数形式邻域比较）.** 对 proper \(\mathbb C\)-scheme \(X\) 和 coherent algebraic sheaf \(\mathcal F\)，自然映射

$$
\widehat{\mathcal F}_Z
\to
\widehat{(\mathcal F^{an})}_{Z^{an}}
$$

与形式上同调、Hom 和 tensor 操作相容。若 \(Z=X\)，则给出全局 GAGA 比较的形式版本。

**输入定理 AO.8（解析相干层的形式代数化）.** 对 projective \(X/\mathbb C\)，每个 \(X^{an}\) 上 coherent analytic sheaf \(\mathcal G\) 的形式完备化来自唯一的 algebraic coherent sheaf。

AO.8 可由 Serre twisting 与解析有限生成证明，也可由 Chow 型代数化和 Grothendieck existence 证明。

**定理 AO.9（GAGA 的形式代数化路线）.** 接受 AO.2、AO.4、AO.7、AO.8 后，解析化函子

$$
\operatorname{Coh}(X)\to\operatorname{Coh}(X^{an})
$$

是等价。

**证明.** full faithfulness：对 \(\mathcal F,\mathcal G\in\operatorname{Coh}(X)\)，把 Hom 写为 \(H^0(X,\mathcal Hom(\mathcal F,\mathcal G))\)。AO.7 比较形式完备化，AO.2 比较 \(H^0\)，properness 给有限性和分离性，因此 Hom 映射为同构。

essential surjectivity：给 \(\mathcal G\in\operatorname{Coh}(X^{an})\)。AO.8 给 algebraic coherent sheaf \(\mathcal F\)，其形式完备化与 \(\mathcal G\) 的形式完备化相同。full faithfulness 应用于 \(\mathcal Hom\) 与 identity morphism 的形式提升，得到

$$
\mathcal F^{an}\cong\mathcal G.
$$

证毕。

## AO.5 Properness 的使用点

GAGA 的形式路线在四个位置使用 properness：

1. \(R^qf_\ast\mathcal F\) 有限生成；
2. theorem on formal functions 成立；
3. Grothendieck existence 成立；
4. 解析全局截面不在无穷远产生额外函数。

附录 Q 的 \(\mathbb A^1\) 反例说明第四点不能省略。

## 练习

1. 对 \(A=\mathbb C[[t]]\)、\(I=(t)\)，写出 \(A_n\) 和形式系统的相容条件。
2. 证明有限生成 \(A\)-模的 \(I\)-adic completion 与 \(\varprojlim M/I^{n+1}M\) 同构。
3. 解释 AO.5 中为什么可把 Hom 写为 \(H^0\mathcal Hom\)。
4. 比较附录 AI 的 graded module 路线与本附录的形式 GAGA 路线。
