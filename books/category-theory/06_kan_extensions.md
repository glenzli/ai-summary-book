# 第六章：Kan 延拓

给定只在子范畴或索引范畴上定义的函子，怎样以最普适的方式把它延拓到更大范畴？左、右 Kan 延拓分别把这个问题化为预复合函子的左、右伴随，并在逐点处由逗号范畴上的余极限或极限计算。这个构造同时包含普通延拓、伴随、密度公式和逐点余极限，因此是前几章第一次真正汇流。本章先给出全局泛性质，再证明点态公式与逐次延拓定律。

大小条件直接决定这些公式是否有定义：$\mathcal C$ 固定为 $\mathcal U$-小，$\mathcal D,\mathcal E$ 在 $\mathcal V$ 层小且局部 $\mathcal U$-小，所以 $K/d$ 与 $d/K$ 是 $\mathcal U$-小。若源范畴只在 $\mathcal V$ 层小，Kan 泛性质的自然变换集合和双射也相应在 $\mathbf{Set}_{\mathcal V}$ 中读取。

## 6.1 逗号范畴

**定义 6.1.** 设 $K:\mathcal C\to\mathcal D$ 为函子，$d\in\mathcal D$。逗号范畴 $K/d$ 定义如下：

- 对象是二元组 $(c,\alpha:Kc\to d)$。
- 从 $(c,\alpha)$ 到 $(c',\alpha')$ 的态射是 $u:c\to c'$，使得
  $$
  \alpha'\circ K(u)=\alpha.
  $$

忘掉结构箭头给出投影函子

$$
\pi_d:K/d\to\mathcal C,\qquad(c,\alpha)\mapsto c.
$$

对偶地，$d/K$ 的对象是 $(c,\beta:d\to Kc)$；从
$(c,\beta)$ 到 $(c',\beta')$ 的态射是 $u:c\to c'$，满足

$$
K(u)\circ\beta=\beta'.
$$

相应投影记为

$$
\rho_d:d/K\to\mathcal C.
$$

## 6.2 Kan 延拓的泛性质

**定义 6.2.** 设 $K:\mathcal C\to\mathcal D$ 与 $F:\mathcal C\to\mathcal E$。$F$ 沿 $K$ 的左 Kan 延拓（left Kan extension）是函子

$$
\operatorname{Lan}_K F:\mathcal D\to\mathcal E
$$

和自然变换

$$
\eta:F\Rightarrow(\operatorname{Lan}_K F)\circ K
$$

使得对任意 $H:\mathcal D\to\mathcal E$，映射

$$
\Lambda_H:
\operatorname{Nat}(\operatorname{Lan}_K F,H)
\longrightarrow
\operatorname{Nat}(F,H K),
\qquad
\theta\longmapsto(\theta K)\circ\eta
$$

是双射，并且对 $H$ 自然。特别地，$\eta$ 是
$\operatorname{id}_{\operatorname{Lan}_K F}$ 在该双射下的像。

**定义 6.3.** 右 Kan 延拓（right Kan extension）
$\operatorname{Ran}_K F$ 是函子 $\mathcal D\to\mathcal E$ 和自然变换

$$
\varepsilon:(\operatorname{Ran}_K F)K\Rightarrow F
$$

使得对任意 $H:\mathcal D\to\mathcal E$，映射

$$
\mathrm{P}_H:
\operatorname{Nat}(H,\operatorname{Ran}_K F)
\longrightarrow
\operatorname{Nat}(HK,F),
\qquad
\theta\longmapsto\varepsilon\circ(\theta K)
$$

是双射，并且对 $H$ 自然。

**命题 6.4（结构化唯一性与函子性）.**

1. 固定 $F$ 后，任意两个左 Kan 延拓之间存在唯一与结构映射
   $\eta$ 相容的自然同构；任意两个右 Kan 延拓之间存在唯一与
   $\varepsilon$ 相容的自然同构。
2. 若为每个 $F:\mathcal C\to\mathcal E$ 选择一个左、右 Kan 延拓，则
   各自的泛性质唯一决定它们在自然变换上的作用，使
   $\operatorname{Lan}_K$ 成为预复合函子

$$
K^*:\operatorname{Fun}(\mathcal D,\mathcal E)\to\operatorname{Fun}(\mathcal C,\mathcal E)
$$

   的左伴随，并使 $\operatorname{Ran}_K$ 成为 $K^*$ 的右伴随。

**证明.** 先证左延拓唯一性。设 $(L,\eta)$ 与 $(L',\eta')$ 都是
$F$ 的左 Kan 延拓。把 $H=L'$ 代入 $(L,\eta)$ 的泛性质，存在唯一
$\alpha:L\Rightarrow L'$ 满足

$$
(\alpha K)\eta=\eta'.
$$

交换二者得到唯一 $\beta:L'\Rightarrow L$ 满足
$(\beta K)\eta'=\eta$。于是

$$
((\beta\alpha)K)\eta
=(\beta K)(\alpha K)\eta
=(\beta K)\eta'
=\eta.
$$

$\operatorname{id}_L$ 也满足同一等式，故唯一性给出
$\beta\alpha=\operatorname{id}_L$；交换二者得
$\alpha\beta=\operatorname{id}_{L'}$。所以 $\alpha$ 是唯一相容自然同构。

对右延拓，若 $(R,\varepsilon)$ 与 $(R',\varepsilon')$ 都表示同一右
Kan 泛性质，则存在唯一 $\alpha:R\Rightarrow R'$ 满足

$$
\varepsilon'\circ(\alpha K)=\varepsilon.
$$

同理存在唯一 $\beta:R'\Rightarrow R$ 满足
$\varepsilon(\beta K)=\varepsilon'$。于是

$$
\varepsilon((\beta\alpha)K)
=\varepsilon(\beta K)(\alpha K)
=\varepsilon'(\alpha K)
=\varepsilon.
$$

右 Kan 泛性质的单射性给出
$\beta\alpha=\operatorname{id}_R$。交换二者，

$$
\varepsilon'((\alpha\beta)K)
=\varepsilon(\beta K)
=\varepsilon',
$$

故 $\alpha\beta=\operatorname{id}_{R'}$。这也证明了相容态射的唯一性。

现在选择每个 $F$ 的左延拓 $(L_F,\eta_F)$。对自然变换
$\sigma:F\Rightarrow F'$，定义

$$
\operatorname{Lan}_K(\sigma):L_F\Rightarrow L_{F'}
$$

为唯一满足

$$
\bigl(\operatorname{Lan}_K(\sigma)K\bigr)\eta_F
=\eta_{F'}\sigma
$$

的自然变换。$\sigma=\operatorname{id}_F$ 时恒等变换满足该式，故由唯一性
$\operatorname{Lan}_K(\operatorname{id}_F)=\operatorname{id}_{L_F}$。
若 $F\xRightarrow{\sigma}F'\xRightarrow{\tau}F''$，则
$\operatorname{Lan}_K(\tau)\operatorname{Lan}_K(\sigma)$ 与
$\operatorname{Lan}_K(\tau\sigma)$ 预合 $\eta_F$ 后都等于
$\eta_{F''}\tau\sigma$，故二者相等。因此 $\operatorname{Lan}_K$ 是函子。

定义 6.2 的双射已经对 $H$ 自然。它也对 $F$ 自然：若
$\theta:L_{F'}\Rightarrow H$，则

$$
\begin{aligned}
\Lambda_H(\theta\operatorname{Lan}_K(\sigma))
&=(\theta K)(\operatorname{Lan}_K(\sigma)K)\eta_F\\
&=(\theta K)\eta_{F'}\sigma
=\Lambda_H(\theta)\sigma.
\end{aligned}
$$

故这些双射正是伴随
$\operatorname{Lan}_K\dashv K^*$ 的 Hom 双射。

对选择的右延拓 $(R_F,\varepsilon_F)$，定义
$\operatorname{Ran}_K(\sigma):R_F\Rightarrow R_{F'}$ 为唯一满足

$$
\varepsilon_{F'}\bigl(\operatorname{Ran}_K(\sigma)K\bigr)
=\sigma\varepsilon_F
$$

的自然变换。具体地，$\operatorname{id}_{R_F}$ 满足
$\sigma=\operatorname{id}_F$ 时的定义式，故
$\operatorname{Ran}_K(\operatorname{id}_F)=\operatorname{id}_{R_F}$。若
$F\xRightarrow{\sigma}F'\xRightarrow{\tau}F''$，则

$$
\begin{aligned}
\varepsilon_{F''}
\bigl((\operatorname{Ran}_K(\tau)\operatorname{Ran}_K(\sigma))K\bigr)
&=\tau\varepsilon_{F'}
  (\operatorname{Ran}_K(\sigma)K)\\
&=\tau\sigma\varepsilon_F,
\end{aligned}
$$

所以唯一性给出
$\operatorname{Ran}_K(\tau)\operatorname{Ran}_K(\sigma)
=\operatorname{Ran}_K(\tau\sigma)$。因此 $\operatorname{Ran}_K$ 是函子。此外，

$$
\mathrm{P}_H(\operatorname{Ran}_K(\sigma)\theta)
=\varepsilon_{F'}(\operatorname{Ran}_K(\sigma)K)(\theta K)
=\sigma\varepsilon_F(\theta K)
=\sigma\mathrm{P}_H(\theta)
$$

给出对 $F$ 的自然性。因此 $K^*\dashv\operatorname{Ran}_K$。$\square$

## 6.3 点态公式

**定理 6.5（左 Kan 延拓的点态公式）.** 若对每个
$d\in\mathcal D$，$\mathcal U$-小图形

$$
F\pi_d:K/d\to\mathcal E
$$

的余极限存在，则左 Kan 延拓存在，并逐点由公式

$$
(\operatorname{Lan}_K F)(d)\cong
\operatorname{colim}_{K/d}(F\pi_d)
$$

给出。要求 $\mathcal E$ 有所有形状 $K/d$ 的余极限是一个常用充分条件，但定理实际只需要上述特定图形的余极限。

**证明.** 定义 $L:\mathcal D\to\mathcal E$ 如下。对 $d\in\mathcal D$，令

$$
L(d)=\operatorname{colim}_{(c,\alpha:Kc\to d)\in K/d}F(c).
$$

记余极限结构映射为

$$
\iota^d_{(c,\alpha)}:F(c)\to L(d).
$$

若 $v:d\to d'$，则后复合给出函子

$$
v_*:K/d\to K/d',\qquad (c,\alpha)\mapsto(c,v\alpha).
$$

族 $\iota^{d'}_{(c,v\alpha)}:F(c)\to L(d')$ 是图形
$F\pi_d$ 的余锥，故余极限泛性质诱导唯一态射
$L(v):L(d)\to L(d')$ 满足

$$
L(v)\iota^d_{(c,\alpha)}
=\iota^{d'}_{(c,v\alpha)}.
$$

当 $v=\operatorname{id}_d$ 时，$L(v)$ 与
$\operatorname{id}_{L(d)}$ 在每个 $\iota^d_{(c,\alpha)}$ 上复合相同，故相等。若
$d\xrightarrow{v}d'\xrightarrow{w}d''$，则
$L(w)L(v)$ 与 $L(wv)$ 在每个结构映射上都给出
$\iota^{d''}_{(c,wv\alpha)}$，故也相等。因此 $L$ 是函子。

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

故相等。于是 $\bar\beta:L\Rightarrow H$。

还需验证两个方向互逆。若从 $\beta:F\Rightarrow HK$ 构造
$\bar\beta$，则在 $c\in\mathcal C$ 上，

$$
\bigl((\bar\beta K)\eta\bigr)_c
=\bar\beta_{Kc}\iota^{Kc}_{(c,\operatorname{id}_{Kc})}
=H(\operatorname{id}_{Kc})\beta_c
=\beta_c.
$$

反过来，从 $\theta:L\Rightarrow H$ 得到
$\beta=(\theta K)\eta$，再构造 $\bar\beta$。对每个
$(c,\alpha:Kc\to d)$，有

$$
\begin{aligned}
\bar\beta_d\iota^d_{(c,\alpha)}
&=H(\alpha)\beta_c\\
&=H(\alpha)\theta_{Kc}\eta_c\\
&=\theta_dL(\alpha)\eta_c\\
&=\theta_d\iota^d_{(c,\alpha)}.
\end{aligned}
$$

第三行用 $\theta$ 对 $\alpha$ 的自然性，第四行用 $L(\alpha)$ 的定义。
余极限结构映射联合检测从 $L(d)$ 出发的态射，所以
$\bar\beta_d=\theta_d$。两种构造互逆。若
$\gamma:H\Rightarrow H'$，上述分量公式表明由 $\gamma K\circ\beta$
构造的态射是 $\gamma\circ\bar\beta$，故双射对 $H$ 自然。于是
$(L,\eta)$ 满足定义 6.2。$\square$

**定理 6.6（右 Kan 延拓的点态公式）.** 若对每个
$d\in\mathcal D$，$\mathcal U$-小图形

$$
F\rho_d:d/K\to\mathcal E
$$

的极限存在，则右 Kan 延拓存在，并逐点由公式

$$
(\operatorname{Ran}_K F)(d)\cong
\lim_{d/K}(F\rho_d)
$$

给出。要求 $\mathcal E$ 有所有形状 $d/K$ 的极限只是一个常用充分条件。

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
v^*:d'/K\to d/K,\qquad
(c,\beta:d'\to Kc)\mapsto(c,\beta v:d\to Kc).
$$

于是族

$$
p_{(c,\beta v)}:R(d)\to F(c)
$$

是到图形 $F\rho_{d'}:d'/K\to\mathcal E$ 的锥。由 $R(d')$ 的极限泛性质，存在唯一态射

$$
R(v):R(d)\to R(d')
$$

满足

$$
p_{(c,\beta)}R(v)=p_{(c,\beta v)}
$$

对所有 $(c,\beta)\in d'/K$ 成立。当 $v=\operatorname{id}_d$ 时，
$R(v)$ 与 $\operatorname{id}_{R(d)}$ 在所有极限投影下相同，故相等。若
$d\xrightarrow{v}d'\xrightarrow{w}d''$，则
$R(w)R(v)$ 与 $R(wv)$ 经任意
$p_{(c,\gamma:d''\to Kc)}$ 后都等于
$p_{(c,\gamma wv)}$，故也相等。所以 $R$ 是函子。

对每个 $c\in\mathcal C$，令

$$
\varepsilon_c:R(Kc)\to F(c)
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
\varepsilon_{c'}R(Ku)=F(u)\varepsilon_c,
$$

故 $\varepsilon:RK\Rightarrow F$ 是自然变换。

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

把 $\bar\alpha$ 与 $\varepsilon$ 复合得到

$$
HK\xrightarrow{\bar\alpha K}RK\xrightarrow{\varepsilon}F.
$$

在 $c$ 处分量为

$$
p_{(c,\operatorname{id})}\bar\alpha_{Kc}
=\alpha_c,
$$

故恢复 $\alpha$。反过来，从 $\theta:H\Rightarrow R$ 出发得到
$\varepsilon(\theta K):HK\Rightarrow F$；再按上述构造恢复的自然变换
$\bar\alpha$ 满足

$$
p_{(c,\beta)}\bar\alpha_d
=\varepsilon_c\theta_{Kc}H(\beta)
=p_{(c,\operatorname{id})}\theta_{Kc}H(\beta)
=p_{(c,\beta)}\theta_d.
$$

最后一步依次使用 $\theta$ 对 $\beta:d\to Kc$ 的自然性
$\theta_{Kc}H(\beta)=R(\beta)\theta_d$，以及 $R(\beta)$ 的定义式
$p_{(c,\operatorname{id})}R(\beta)=p_{(c,\beta)}$。极限投影联合检测态射，
故 $\bar\alpha_d=\theta_d$。两种构造互逆。若
$\gamma:H'\Rightarrow H$，则预复合 $\bar\alpha\gamma$ 对应于
$\alpha(\gamma K)$，所以该双射对 $H$ 自然。于是

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

## 6.7 沿函子的普适延拓

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
