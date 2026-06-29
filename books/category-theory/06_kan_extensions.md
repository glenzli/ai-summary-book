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

**证明.** 定义 $R:\mathcal D\to\mathcal E$ 如下。对 $d\in\mathcal D$，令

$$
R(d)=\lim_{(c,\beta:d\to Kc)\in d/K}F(c),
$$

并记极限投影为

$$
p_{(c,\beta)}:R(d)\to F(c).
$$

若 $v:d\to d'$ 是 $\mathcal D$ 中态射，则预复合给出函子

$$
d'/K\to d/K,\qquad (c,\beta:d'\to Kc)\mapsto(c,\beta v:d\to Kc).
$$

于是族

$$
p_{(c,\beta v)}:R(d)\to F(c)
$$

是到图形 $d'/K\to\mathcal E$ 的锥。由 $R(d')$ 的极限泛性质，存在唯一态射

$$
R(v):R(d)\to R(d')
$$

满足

$$
p_{(c,\beta)}R(v)=p_{(c,\beta v)}
$$

对所有 $(c,\beta)\in d'/K$ 成立。恒等与复合由这些投影共同检测，所以 $R$ 是函子。

对每个 $c\in\mathcal C$，令

$$
\epsilon_c:R(Kc)\to F(c)
$$

为对象 $(c,\operatorname{id}_{Kc})\in Kc/K$ 对应的投影。若 $u:c\to c'$，则在 $Kc/K$ 中有态射

$$
(c,\operatorname{id}_{Kc})\to(c',K u).
$$

极限锥相容性给出

$$
p_{(c',Ku)}=F(u)p_{(c,\operatorname{id})}.
$$

另一方面，由 $R(Ku)$ 的定义，

$$
p_{(c',\operatorname{id})}R(Ku)=p_{(c',Ku)}.
$$

因此

$$
\epsilon_{c'}R(Ku)=F(u)\epsilon_c,
$$

故 $\epsilon:RK\Rightarrow F$ 是自然变换。

现在设 $H:\mathcal D\to\mathcal E$，并给定自然变换 $\alpha:HK\Rightarrow F$。对每个 $d$ 和每个 $(c,\beta:d\to Kc)\in d/K$，取复合

$$
H(d)\xrightarrow{H(\beta)}H(Kc)\xrightarrow{\alpha_c}F(c).
$$

这些态射形成从 $H(d)$ 到图形 $d/K\to\mathcal E$ 的锥：若

$$
u:(c,\beta)\to(c',\beta')
$$

即 $K(u)\beta=\beta'$，则

$$
F(u)\alpha_cH(\beta)
=\alpha_{c'}H(Ku)H(\beta)
=\alpha_{c'}H(\beta'),
$$

其中第一步用 $\alpha$ 的自然性。由 $R(d)$ 的极限泛性质，存在唯一态射

$$
\bar\alpha_d:H(d)\to R(d)
$$

使

$$
p_{(c,\beta)}\bar\alpha_d=\alpha_cH(\beta).
$$

若 $v:d\to d'$，则对任意 $(c,\beta:d'\to Kc)$，

$$
p_{(c,\beta)}R(v)\bar\alpha_d
=p_{(c,\beta v)}\bar\alpha_d
=\alpha_cH(\beta v)
=p_{(c,\beta)}\bar\alpha_{d'}H(v).
$$

由极限投影共同检测，$R(v)\bar\alpha_d=\bar\alpha_{d'}H(v)$。故 $\bar\alpha:H\Rightarrow R$ 自然。

把 $\bar\alpha$ 与 $\epsilon$ 复合得到

$$
HK\xrightarrow{\bar\alpha K}RK\xrightarrow{\epsilon}F.
$$

在 $c$ 处分量为

$$
p_{(c,\operatorname{id})}\bar\alpha_{Kc}
=\alpha_c,
$$

故恢复 $\alpha$。反过来，从 $\theta:H\Rightarrow R$ 出发得到 $\epsilon\theta K:HK\Rightarrow F$；再按上述构造恢复的自然变换与 $\theta$ 在每个投影 $p_{(c,\beta)}$ 下相同，因此相等。于是

$$
\operatorname{Nat}(H,R)\cong\operatorname{Nat}(HK,F)
$$

自然成立，$R$ 满足右 Kan 延拓泛性质。$\square$

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

**命题 6.9.** 若 $K:\mathcal C\to\mathcal D$ 完全忠实，且 $\operatorname{Ran}_K F$ 存在，则余单位

$$
(\operatorname{Ran}_K F)K\to F
$$

在点态公式给出的条件下是同构。

**证明.** 对 $c\in\mathcal C$，逗号范畴 $Kc/K$ 有始对象 $(c,\operatorname{id}_{Kc})$。事实上，若 $(c',\beta:Kc\to Kc')$ 是对象，由 $K$ 完全忠实，存在唯一 $u:c\to c'$ 使 $K(u)=\beta$，这正是从 $(c,\operatorname{id}_{Kc})$ 到 $(c',\beta)$ 的唯一态射。因此极限

$$
(\operatorname{Ran}_K F)(Kc)
\cong\lim_{Kc/K}F
$$

由始对象处的值给出，故同构于 $F(c)$。在该同构下，余单位分量正是始对象对应的极限投影，因此是同构。$\square$

## 6.5 点态公式的稳定性

**命题 6.10.** 设 $\operatorname{Lan}_K F$ 由定理 6.5 的点态公式给出。若函子 $H:\mathcal E\to\mathcal E'$ 保持所有形状 $K/d$ 的余极限，则有自然同构

$$
H(\operatorname{Lan}_K F)\cong \operatorname{Lan}_K(HF).
$$

**证明.** 对每个 $d\in\mathcal D$，

$$
H((\operatorname{Lan}_K F)(d))
\cong
H\left(\operatorname{colim}_{K/d}F\right)
\cong
\operatorname{colim}_{K/d}HF.
$$

右边正是 $\operatorname{Lan}_K(HF)$ 的点态公式。对 $d$ 的自然性来自定理 6.5 中由余极限唯一性定义的函子结构；$H$ 保持这些余极限使相同结构映射在 $H$ 后仍满足同一泛性质。$\square$

**命题 6.11（共尾缩小点态公式）.** 设对每个 $d\in\mathcal D$ 给定共尾函子

$$
V_d:\mathcal I_d\to K/d.
$$

若所需余极限存在，则

$$
(\operatorname{Lan}_K F)(d)
\cong
\operatorname{colim}_{i\in\mathcal I_d}F(\pi V_d(i)),
$$

其中 $\pi:K/d\to\mathcal C$ 是遗忘函子。右 Kan 延拓有始函子的对偶版本。

**证明.** 由定理 6.5，

$$
(\operatorname{Lan}_K F)(d)
\cong \operatorname{colim}_{K/d}F\pi.
$$

再由定理 3.16 的共尾性，

$$
\operatorname{colim}_{K/d}F\pi
\cong
\operatorname{colim}_{\mathcal I_d}F\pi V_d.
$$

右 Kan 延拓的陈述把余极限、共尾函子替换为极限、始函子。$\square$

## 6.6 例子与存在性边界

**命题 6.12.** 若 $K:\mathcal C\to\mathcal D$ 有右伴随 $R:\mathcal D\to\mathcal C$，且 $\operatorname{Lan}_K F$ 按点态公式存在，则有自然同构

$$
(\operatorname{Lan}_K F)(d)\cong F(Rd).
$$

对偶地，若 $K$ 有左伴随 $L:\mathcal D\to\mathcal C$，且 $\operatorname{Ran}_K F$ 按点态公式存在，则

$$
(\operatorname{Ran}_K F)(d)\cong F(Ld).
$$

**证明.** 设 $K\dashv R$，余单位为 $\varepsilon:KR\Rightarrow\operatorname{id}_{\mathcal D}$。对每个 $d$，对象

$$
(Rd,\varepsilon_d:KRd\to d)
$$

是 $K/d$ 的终对象。若 $(c,\alpha:Kc\to d)$ 是 $K/d$ 的对象，则伴随给出唯一 $\bar\alpha:c\to Rd$，满足 $\varepsilon_dK(\bar\alpha)=\alpha$，这正是到 $(Rd,\varepsilon_d)$ 的唯一态射。故

$$
(\operatorname{Lan}_K F)(d)\cong\operatorname{colim}_{K/d}F\cong F(Rd).
$$

右 Kan 延拓情形对偶：若 $L\dashv K$，则 $(Ld,\eta_d:d\to KLd)$ 是 $d/K$ 的始对象，极限由始对象处的值给出。$\square$

**例子 6.13（离散子范畴延拓）.** 设 $i:\{0,1\}\hookrightarrow [1]$ 包含两个对象到箭头范畴 $0\to1$，并给出 $F(0)=A,F(1)=B$。在有二元余积的范畴 $\mathcal E$ 中，点态公式给

$$
(\operatorname{Lan}_iF)(0)\cong A,\qquad
(\operatorname{Lan}_iF)(1)\cong A\sqcup B.
$$

原因是 $i/0$ 只有对象 $0\to0$，而 $i/1$ 有两个对象 $0\to1$ 和 $1\to1$，且无非恒等态射。因此左 Kan 延拓把“只在端点给值”的资料自由扩展为箭头

$$
A\to A\sqcup B.
$$

**例子 6.14（存在性边界）.** 令 $K:\varnothing\to *$ 为唯一函子，令 $F:\varnothing\to\mathcal E$ 为空图形。若 $\operatorname{Lan}_K F$ 存在，则其在 $*$ 处应为

$$
\operatorname{colim}_{\varnothing}F,
$$

即 $\mathcal E$ 的始对象。因此当 $\mathcal E$ 没有始对象时，该左 Kan 延拓不存在。点态公式不是纯形式记号；它要求目标范畴有相应余极限。

## 6.7 本章小结

Kan 延拓是沿函子改变定义域的泛构造。它统一了极限、余极限、伴随和稠密嵌入。点态公式说明左 Kan 延拓由逗号范畴上的余极限给出，右 Kan 延拓由逗号范畴上的极限给出。完全忠实函子上的 Kan 延拓在原范畴上恢复原函子；保余极限函子与共尾缩小可直接作用于点态公式。

## 练习

**练习 6.1.** 验证 $K/d$ 的复合和恒等态射确实给出范畴。

**练习 6.2.** 对包含函子 $i:\mathcal A\hookrightarrow\mathcal C$，解释 $\operatorname{Lan}_i F$ 如何看作从子范畴延拓函子。

**练习 6.3.** 证明若 $K=\operatorname{id}_{\mathcal C}$，则 $\operatorname{Lan}_K F\cong F$ 且 $\operatorname{Ran}_K F\cong F$。

**练习 6.4.** 完成定理 6.5 的自然性检查。

**练习 6.5.** 用 Kan 延拓语言重述预层密度定理。

**练习 6.6.** 检查定理 6.6 中 $R(v)$ 的定义确实给出函子 $R:\mathcal D\to\mathcal E$。

**练习 6.7.** 证明命题 6.9 中 $Kc/K$ 的始对象刻画。

**练习 6.8.** 对偶化命题 6.10，写出右 Kan 延拓在保极限函子下的稳定性。

**练习 6.9.** 令 $K:\mathcal C\to *$ 为唯一函子。用点态公式重新证明 Kan 延拓恢复极限和余极限。

**练习 6.10.** 设 $V:\mathcal I\to K/d$ 共尾。把命题 6.11 的同构写成余锥泛性质的双射，而不是只引用定理 3.16。

**练习 6.11.** 证明命题 6.12 中 $(Rd,\varepsilon_d)$ 是 $K/d$ 的终对象。

**练习 6.12.** 对例子 6.13，计算右 Kan 延拓 $\operatorname{Ran}_iF$ 在 $0$ 和 $1$ 处的值，假设 $\mathcal E$ 有二元积。

**练习 6.13.** 给出一个右 Kan 延拓不存在的例子，使用没有终对象的目标范畴。
