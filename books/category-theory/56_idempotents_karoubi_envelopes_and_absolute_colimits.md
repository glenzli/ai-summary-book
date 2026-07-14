# 第五十六章：幂等分裂、Karoubi 包络与绝对余极限

幂等态射 $e:X\to X$ 在集合和模范畴中总能表示为某个 retract，但一般范畴未必包含它的像。Karoubi 包络以普适方式补入全部幂等分裂，得到 Cauchy 完备范畴。另一方面，被所有函子保持的绝对余极限恰刻画这种无需额外结构即可识别的 retract 型构造；在富范畴中，它们推广为绝对加权余极限，并与具有伴随的 profunctors 相连。

本章使用 retract、余等化子、profunctor、富范畴和加权余极限。我们会证明 Karoubi 包络的普适性质，并区分 ordinary Cauchy completeness、富 Cauchy completeness 与“具有所有小余极限”，后三者并不等价。

## 56.1 幂等与分裂

**定义 56.1.** 范畴 $\mathcal C$ 中态射 $e:X\to X$ 称为幂等，若

$$
e^2=e.
$$

**定义 56.2.** 幂等 $e:X\to X$ 称为分裂，若存在对象 $Y$ 和态射

$$
r:X\to Y,\qquad s:Y\to X
$$

使 $rs=\operatorname{id}_Y$ 且 $sr=e$。

**命题 56.3.** 若幂等分裂，则分裂对象在唯一同构意义下唯一。

**证明.** 设 $e=sr=s'r'$ 是两个分裂，且 $rs=\operatorname{id}_Y$、$r's'=\operatorname{id}_{Y'}$。定义

$$
u= r's:Y\to Y',\qquad v=rs':Y'\to Y.
$$

则

$$
vu=rs'r's= r e s = r s r s = \operatorname{id}_Y,
$$

同理 $uv=\operatorname{id}_{Y'}$。故 $Y\cong Y'$。$\square$

## 56.2 Karoubi 包络

**定义 56.4.** 范畴 $\mathcal C$ 的 Karoubi 包络 $\operatorname{Kar}(\mathcal C)$ 的对象为对 $(X,e)$，其中 $e:X\to X$ 为幂等。态射

$$
f:(X,e)\to(Y,d)
$$

是 $\mathcal C$ 中态射 $f:X\to Y$，满足

$$
f= d f e.
$$

对象 $(X,e)$ 的恒等态射是 $e$，而不是 $\operatorname{id}_X$；确有 $e=eee$，并且对任意满足 $f=dfe$ 的态射，$df=f=fe$。

**命题 56.5.** $\operatorname{Kar}(\mathcal C)$ 中每个幂等都分裂。

**证明.** 设 $p:(X,e)\to(X,e)$ 为 $\operatorname{Kar}(\mathcal C)$ 中幂等。则在 $\mathcal C$ 中 $p^2=p$ 且 $p= epe$。对象 $(X,p)$ 存在于 $\operatorname{Kar}(\mathcal C)$。态射

$$
r:(X,e)\to(X,p),\quad r=p,\qquad
s:(X,p)\to(X,e),\quad s=p
$$

满足 $rs=p$ 在 $(X,p)$ 上为恒等，且 $sr=p$ 为原幂等。因此 $p$ 分裂。$\square$

**命题 56.6.** 嵌入 $i:\mathcal C\to\operatorname{Kar}(\mathcal C)$ 把 $X$ 送到 $(X,\operatorname{id}_X)$，且全忠实。

**证明.** 从 $(X,\operatorname{id})$ 到 $(Y,\operatorname{id})$ 的态射是 $f:X\to Y$ 满足

$$
f=\operatorname{id}_Y f\operatorname{id}_X,
$$

这是空条件。因此 Hom 集完全相同，$i$ 全忠实。$\square$

## 56.3 幂等完备范畴

**定义 56.7.** 范畴 $\mathcal C$ 称为幂等完备，若其中每个幂等都分裂。

**命题 56.8.** 若 $\mathcal C$ 幂等完备，则 $i:\mathcal C\to\operatorname{Kar}(\mathcal C)$ 为等价。

**证明.** 命题 56.6 给全忠实。任意对象 $(X,e)$ 中 $e$ 在 $\mathcal C$ 中分裂为 $X\xrightarrow rY\xrightarrow sX$。在 $\operatorname{Kar}(\mathcal C)$ 中，$(X,e)$ 与 $(Y,\operatorname{id}_Y)$ 由 $r,s$ 给出同构。因此 $i$ 本质满，故为等价。$\square$

## 56.4 绝对余极限

**定义 56.9.** 范畴 $\mathcal C$ 中图形 $D:J\to\mathcal C$ 的余极限称为绝对余极限，若对任意函子 $F:\mathcal C\to\mathcal D$，$F$ 都保持该余极限。

**命题 56.10.** 分裂 coequalizer 是绝对余极限。

**证明.** 分裂 coequalizer 的数据可写为

$$
A\mathrel{\substack{\xrightarrow{f}\\[-2pt]\xrightarrow[g]{}}}B
\xrightarrow{q}Q,
\qquad
s:Q\to B,\quad t:B\to A,
$$

满足

$$
qf=qg,\qquad qs=\operatorname{id}_Q,\qquad
ft=\operatorname{id}_B,\qquad gt=sq.
$$

这些等式直接验证 $q$ 的 coequalizer 泛性质。任意函子保持复合、恒等与等式，因此把这组 splitting data 送到同样的分裂 coequalizer 数据；故任意函子保持它。$\square$

## 56.5 Cauchy 完备性

**定义 56.11.** 普通范畴 $\mathcal C$ 称为 Cauchy complete，若所有绝对余极限存在。对普通小范畴，这等价于幂等完备。

**外部输入定理 56.12.** 在 enriched category theory 中，Cauchy completion 等价于加入所有绝对加权余极限；普通范畴情形退化为 Karoubi 包络。

**命题 56.13.** 若 $\mathcal D$ 幂等完备，则预合成 $i:\mathcal C\to\operatorname{Kar}(\mathcal C)$ 给出范畴等价

$$
\operatorname{Fun}(\operatorname{Kar}(\mathcal C),\mathcal D)
\xrightarrow{\ \simeq\ }
\operatorname{Fun}(\mathcal C,\mathcal D).
$$

因此 Karoubi 包络是幂等完备化。

**证明.** 命题 56.5 说明 $\operatorname{Kar}(\mathcal C)$ 幂等完备，命题 56.6 给出全忠实嵌入。若 $F:\mathcal C\to\mathcal D$，则对每个 $(X,e)$，幂等 $F(e)$ 在 $\mathcal D$ 中可分裂；令扩张在 $(X,e)$ 上取该分裂对象，并用 $f=dfe$ 诱导分裂对象间态射。分裂对象及其诱导态射的选择空间可缩，故得到本质唯一的扩张。对自然变换作同一限制与扩张，说明预合成函子全忠实且本质满，因而为等价。$\square$

**命题 56.14.** Karoubi 包络是幂等的：

$$
\operatorname{Kar}(\operatorname{Kar}(\mathcal C))\simeq\operatorname{Kar}(\mathcal C).
$$

**证明.** 由命题 56.5，$\operatorname{Kar}(\mathcal C)$ 已幂等完备。再由命题 56.8，把幂等完备范畴嵌入其 Karoubi 包络是等价。取该范畴为 $\operatorname{Kar}(\mathcal C)$ 即得结论。$\square$

## 56.6 分裂幂等与绝对余极限

幂等分裂是范畴中 retract 存在性的最小完备性要求。Karoubi 包络自由加入所有幂等的分裂。绝对余极限是不依赖目标函子的余极限，普通范畴中与幂等分裂紧密相关；富范畴中则导向 Cauchy completion 和 Morita 理论。

## 练习

**练习 56.1.** 定义幂等态射。

**练习 56.2.** 定义幂等分裂。

**练习 56.3.** 证明分裂对象唯一到唯一同构。

**练习 56.4.** 定义 Karoubi 包络。

**练习 56.5.** 证明 $\operatorname{Kar}(\mathcal C)$ 中幂等分裂。

**练习 56.6.** 证明 $\mathcal C\to\operatorname{Kar}(\mathcal C)$ 全忠实。

**练习 56.7.** 定义幂等完备范畴。

**练习 56.8.** 证明若 $\mathcal C$ 幂等完备，则 $\mathcal C\simeq\operatorname{Kar}(\mathcal C)$。

**练习 56.9.** 定义绝对余极限。

**练习 56.10.** 说明分裂 coequalizer 为何是绝对余极限。

**练习 56.11.** 定义 Cauchy complete。

**练习 56.12.** 说明 Karoubi 包络的泛性质。

**练习 56.13.** 证明 $\operatorname{Kar}(\operatorname{Kar}(\mathcal C))\simeq\operatorname{Kar}(\mathcal C)$。
