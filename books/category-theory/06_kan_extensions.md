# 第六章：Kan 延拓

## 本章目标

本章定义左、右 Kan 延拓，给出通过预复合函子的伴随刻画，并在余完备/完备条件下写出点态公式。Kan 延拓统一了伴随、余极限、密度和“沿函子延拓”的语言。

## 依赖前置知识

需要函子范畴、伴随函子、逗号范畴的基本语言和极限/余极限。

## 6.1 逗号范畴

**定义 6.1.** 设 $K:\mathcal C\to\mathcal D$ 为函子，$d\in\mathcal D$。逗号范畴 $K/d$ 定义如下：

- 对象是二元组 $(c,\alpha:Kc\to d)$。
- 从 $(c,\alpha)$ 到 $(c',\alpha')$ 的态射是 $u:c\to c'$，使得
  $$
  \alpha'\circ K(u)=\alpha.
  $$

对偶地，$d/K$ 的对象是 $(c,\beta:d\to Kc)$。

## 6.2 Kan 延拓的泛性质

**定义 6.2.** 设 $K:\mathcal C\to\mathcal D$ 与 $F:\mathcal C\to\mathcal E$。$F$ 沿 $K$ 的左 Kan 延拓（left Kan extension）是函子

$$
\operatorname{Lan}_K F:\mathcal D\to\mathcal E
$$

和自然变换

$$
\eta:F\Rightarrow(\operatorname{Lan}_K F)\circ K
$$

使得对任意 $H:\mathcal D\to\mathcal E$，预复合给出自然双射

$$
\operatorname{Nat}(\operatorname{Lan}_K F,H)
\cong
\operatorname{Nat}(F,H K).
$$

**定义 6.3.** 右 Kan 延拓（right Kan extension）$\operatorname{Ran}_K F$ 是函子 $\mathcal D\to\mathcal E$ 和自然变换

$$
\epsilon:(\operatorname{Ran}_K F)K\Rightarrow F
$$

使得对任意 $H:\mathcal D\to\mathcal E$ 有自然双射

$$
\operatorname{Nat}(H,\operatorname{Ran}_K F)
\cong
\operatorname{Nat}(HK,F).
$$

**命题 6.4.** 若存在，则 $\operatorname{Lan}_K$ 是预复合函子

$$
K^*:\operatorname{Fun}(\mathcal D,\mathcal E)\to\operatorname{Fun}(\mathcal C,\mathcal E)
$$

的左伴随；$\operatorname{Ran}_K$ 是 $K^*$ 的右伴随。

**证明.** 这只是定义 6.2 和 6.3 的重写。左 Kan 延拓的泛双射正是

$$
\operatorname{Fun}(\mathcal D,\mathcal E)(\operatorname{Lan}_K F,H)
\cong
\operatorname{Fun}(\mathcal C,\mathcal E)(F,K^*H).
$$

右 Kan 延拓同理。$\square$

## 6.3 点态公式

**定理 6.5.** 若 $\mathcal E$ 有所有形状 $K/d$ 的余极限，则左 Kan 延拓逐点由公式

$$
(\operatorname{Lan}_K F)(d)\cong
\operatorname{colim}_{(c,Kc\to d)\in K/d}F(c)
$$

给出。

**证明.** 定义 $L:\mathcal D\to\mathcal E$ 如下。对 $d\in\mathcal D$，令

$$
L(d)=\operatorname{colim}_{(c,\alpha:Kc\to d)\in K/d}F(c).
$$

若 $v:d\to d'$，则后复合给出函子

$$
K/d\to K/d',\qquad (c,\alpha)\mapsto(c,v\alpha).
$$

由余极限泛性质，图形 $F:K/d\to\mathcal E$ 的余锥到 $L(d')$ 诱导唯一态射 $L(v):L(d)\to L(d')$。恒等和复合由余极限诱导态射的唯一性验证，因此 $L$ 是函子。

对每个 $c\in\mathcal C$，对象 $(c,\operatorname{id}_{Kc})\in K/Kc$ 的余极限结构映射给出

$$
\eta_c:F(c)\to L(Kc).
$$

这些态射对 $c$ 自然：若 $u:c\to c'$，则在 $K/Kc'$ 中有态射

$$
(c,Ku)\to(c',\operatorname{id}_{Kc'}),
$$

而余锥相容性正给出 $L(Ku)\eta_c=\eta_{c'}F(u)$。故 $\eta:F\Rightarrow LK$。

现在给定 $H:\mathcal D\to\mathcal E$。若 $\theta:L\Rightarrow H$，复合

$$
F\xrightarrow{\eta}LK\xrightarrow{\theta K}HK
$$

给出自然变换 $F\Rightarrow HK$。反过来，设 $\beta:F\Rightarrow HK$。对每个 $d$ 和每个 $(c,\alpha:Kc\to d)$，定义

$$
F(c)\xrightarrow{\beta_c}H(Kc)\xrightarrow{H(\alpha)}H(d).
$$

这些态射对 $K/d$ 中态射相容：若 $u:(c,\alpha)\to(c',\alpha')$，即 $\alpha'K(u)=\alpha$，则

$$
H(\alpha')H(Ku)\beta_c
=H(\alpha'K u)\beta_c
=H(\alpha)\beta_c
=H(\alpha')\beta_{c'}F(u),
$$

其中最后一步用 $\beta$ 自然性。由余极限泛性质，得到唯一

$$
\bar\beta_d:L(d)\to H(d).
$$

这些 $\bar\beta_d$ 对 $d$ 自然：对 $v:d\to d'$，两条 $L(d)\to H(d')$ 的复合在每个结构映射 $F(c)\to L(d)$ 上都等于

$$
H(v\alpha)\beta_c,
$$

故相等。于是 $\bar\beta:L\Rightarrow H$。两个构造互逆，且对 $H$ 自然，所以 $L$ 满足左 Kan 延拓的泛性质。$\square$

**定理 6.6.** 若 $\mathcal E$ 有所有形状 $d/K$ 的极限，则右 Kan 延拓逐点由公式

$$
(\operatorname{Ran}_K F)(d)\cong
\lim_{(c,d\to Kc)\in d/K}F(c)
$$

给出。

**证明.** 对定理 6.5 对偶化。$\square$

## 6.4 特殊情形

**例子 6.7.** 若 $K:\mathcal C\to *$ 是到终范畴的唯一函子，则 $\operatorname{Lan}_K F(*)$ 是 $\operatorname{colim}_{\mathcal C}F$，而 $\operatorname{Ran}_K F(*)$ 是 $\lim_{\mathcal C}F$。所以极限和余极限是 Kan 延拓的特例。

**命题 6.8.** 若 $K:\mathcal C\to\mathcal D$ 完全忠实，且 $\operatorname{Lan}_K F$ 存在，则单位

$$
F\to(\operatorname{Lan}_K F)K
$$

在点态公式给出的条件下是同构。

**证明.** 对 $c\in\mathcal C$，逗号范畴 $K/Kc$ 有终对象 $(c,\operatorname{id}_{Kc})$。事实上，若 $(c',\alpha:Kc'\to Kc)$ 是对象，由 $K$ 完全忠实，存在唯一 $u:c'\to c$ 使 $K(u)=\alpha$，这正是到 $(c,\operatorname{id}_{Kc})$ 的唯一态射。因此余极限

$$
(\operatorname{Lan}_K F)(Kc)
\cong\operatorname{colim}_{K/Kc}F
$$

由终对象处的值给出，故同构于 $F(c)$。在该同构下，单位分量 $F(c)\to(\operatorname{Lan}_K F)(Kc)$ 正是终对象对应的余极限结构映射，因此是同构。$\square$

## 6.5 本章小结

Kan 延拓是沿函子改变定义域的泛构造。它统一了极限、余极限、伴随和稠密嵌入。点态公式说明左 Kan 延拓由逗号范畴上的余极限给出，右 Kan 延拓由逗号范畴上的极限给出。

## 练习

**练习 6.1.** 验证 $K/d$ 的复合和恒等态射确实给出范畴。

**练习 6.2.** 对包含函子 $i:\mathcal A\hookrightarrow\mathcal C$，解释 $\operatorname{Lan}_i F$ 如何看作从子范畴延拓函子。

**练习 6.3.** 证明若 $K=\operatorname{id}_{\mathcal C}$，则 $\operatorname{Lan}_K F\cong F$ 且 $\operatorname{Ran}_K F\cong F$。

**练习 6.4.** 完成定理 6.5 的自然性检查。

**练习 6.5.** 用 Kan 延拓语言重述预层密度定理。
