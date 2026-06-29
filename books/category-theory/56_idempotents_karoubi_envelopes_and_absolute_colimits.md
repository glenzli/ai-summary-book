# 第五十六章：幂等分裂、Karoubi 包络与绝对余极限

## 本章目标

本章补充范畴论内部的完备化主题：幂等分裂、Karoubi 包络、Cauchy 完备性和绝对余极限。一个余极限若被所有函子保持，称为绝对余极限。绝对余极限与分裂幂等、可分裂 coequalizer 和 Cauchy 完备化紧密相连，是 Morita 理论、profunctor 和 enriched category theory 中的基本工具。

## 依赖前置知识

需要幂等态射、retract、余等化子、伴随、profunctor、Cauchy completion、富范畴和加权余极限。

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

**证明.** 分裂 coequalizer 的数据由有限个态射和等式给出：$q:B\to Q$ coequalizes $f,g:A\rightrightarrows B$，并有 section $s:Q\to B$ 及 splitting data，使 coequalizer 泛性质可由这些等式直接验证。任意函子保持复合和等式，因此把这组分裂数据送到同样的分裂 coequalizer 数据。故任意函子保持它。$\square$

## 56.5 Cauchy 完备性

**定义 56.11.** 普通范畴 $\mathcal C$ 称为 Cauchy complete，若所有绝对余极限存在。对普通小范畴，这等价于幂等完备。

**外部输入定理 56.12.** 在 enriched category theory 中，Cauchy completion 等价于加入所有绝对加权余极限；普通范畴情形退化为 Karoubi 包络。

**命题 56.13.** 普通范畴中，Karoubi 包络是幂等完备化。

**证明.** 命题 56.5 说明 $\operatorname{Kar}(\mathcal C)$ 幂等完备，命题 56.6 给出全忠实嵌入。若 $F:\mathcal C\to\mathcal D$ 且 $\mathcal D$ 幂等完备，则对每个 $(X,e)$，$F(e)$ 在 $\mathcal D$ 中分裂，取其分裂对象作为扩张值，可把 $F$ 延拓到 $\operatorname{Kar}(\mathcal C)$。分裂对象唯一到唯一同构保证延拓在等价意义下唯一。$\square$

## 56.6 本章小结

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
