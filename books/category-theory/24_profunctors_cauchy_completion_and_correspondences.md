# 第二十四章：Profunctor、Cauchy 完备化与 Correspondence

## 本章目标

本章补充一个贯穿普通范畴论、富范畴、双范畴和高阶 Morita 理论的主题：profunctor。普通函子只给出方向明确的映射，而 profunctor 允许把“从 $\mathcal C$ 到 $\mathcal D$ 的广义态射”表示为双变量函子。其复合由 coend 给出，是第十一章 coend 和第二十二章 Morita 复合的普通范畴论原型。

## 依赖前置知识

需要 Yoneda 引理、coend、双范畴、幺半/富范畴和 correspondence 的基本语言。

## 24.1 Profunctor

**定义 24.1.** 设 $\mathcal C,\mathcal D$ 为小范畴。从 $\mathcal C$ 到 $\mathcal D$ 的 profunctor 是函子

$$
P:\mathcal C^{op}\times\mathcal D\to\mathbf{Set}.
$$

记作

$$
P:\mathcal C\nrightarrow\mathcal D.
$$

对 $c\in\mathcal C,d\in\mathcal D$，集合 $P(c,d)$ 可理解为从 $c$ 到 $d$ 的广义态射集合。

**例子 24.2.** 任意函子 $F:\mathcal C\to\mathcal D$ 给出两个 profunctors：

$$
F_*:\mathcal C\nrightarrow\mathcal D,\qquad
F_*(c,d)=\mathcal D(Fc,d),
$$

以及

$$
F^*:\mathcal D\nrightarrow\mathcal C,\qquad
F^*(d,c)=\mathcal D(d,Fc).
$$

它们分别称为 $F$ 的 companion 和 conjoint。

**定义 24.3.** 恒等 profunctor 为 Hom 双函子

$$
\operatorname{id}_{\mathcal C}(c,c')=\mathcal C(c,c').
$$

## 24.2 Coend 复合

**定义 24.4.** 若

$$
P:\mathcal C\nrightarrow\mathcal D,\qquad
Q:\mathcal D\nrightarrow\mathcal E,
$$

则其复合 $Q\circ P:\mathcal C\nrightarrow\mathcal E$ 定义为 coend

$$
(Q\circ P)(c,e)=
\int^{d\in\mathcal D}P(c,d)\times Q(d,e).
$$

元素可记为等价类 $[p,q]$，其中 $p\in P(c,d)$、$q\in Q(d,e)$，并按 $\mathcal D$ 中态射的作用施加平衡关系。

**命题 24.5.** 恒等 profunctor 对 coend 复合起单位作用：

$$
P\circ\operatorname{id}_{\mathcal C}\cong P,\qquad
\operatorname{id}_{\mathcal D}\circ P\cong P.
$$

**证明.** 第一式在 $(c,d)$ 处为

$$
\int^{c'\in\mathcal C}\mathcal C(c,c')\times P(c',d).
$$

由 co-Yoneda 公式，该 coend 自然同构于 $P(c,d)$。第二式同理：

$$
\int^{d'\in\mathcal D}P(c,d')\times\mathcal D(d',d)\cong P(c,d).
$$

自然性来自 co-Yoneda 同构的自然性。$\square$

**外部输入定理 24.6.** 小范畴、profunctors 和 profunctor 之间的自然变换构成双范畴 $\mathbf{Prof}$。其水平复合由定义 24.4 的 coend 给出，结合律由 coend 的 Fubini 定理和笛卡尔积的结合性给出。

## 24.3 函子作为可表示 profunctor

**命题 24.7.** 对函子 $F:\mathcal C\to\mathcal D$，profunctor $F_*$ 与 $F^*$ 之间在 $\mathbf{Prof}$ 中形成伴随

$$
F_*\dashv F^*.
$$

**证明.** 单位是 profunctor 变换

$$
\mathcal C(c,c')\to
(F^*\circ F_*)(c,c')
=
\int^{d\in\mathcal D}\mathcal D(Fc,d)\times\mathcal D(d,Fc').
$$

由 co-Yoneda，右侧同构于 $\mathcal D(Fc,Fc')$；单位即 $F$ 在 Hom 上的映射。余单位为

$$
(F_*\circ F^*)(d,d')
=
\int^{c\in\mathcal C}\mathcal D(d,Fc)\times\mathcal D(Fc,d')
\to
\mathcal D(d,d'),
$$

把 $d\to Fc$ 与 $Fc\to d'$ 复合。三角恒等式化为普通范畴中复合和恒等态射的单位律，并由 co-Yoneda 的单位性保证。$\square$

**推论 24.8.** 若 $F$ 完全忠实，则单位 $\operatorname{id}_{\mathcal C}\to F^*F_*$ 是同构。

**证明.** 在 $(c,c')$ 处分量是

$$
\mathcal C(c,c')\to\mathcal D(Fc,Fc'),
$$

这正是 $F$ 的 Hom 映射。完全忠实意味着它是双射。$\square$

## 24.4 Cauchy 完备化与幂等分裂

**定义 24.9.** 范畴 $\mathcal C$ 称为幂等完备或 Cauchy complete，若每个幂等态射

$$
e:X\to X,\qquad e^2=e
$$

都分裂，即存在 $r:X\to Y$ 与 $s:Y\to X$，使得

$$
sr=e,\qquad rs=\operatorname{id}_Y.
$$

**定义 24.10.** 小范畴 $\mathcal C$ 的 Cauchy 完备化 $\operatorname{Kar}(\mathcal C)$ 的对象为对 $(X,e)$，其中 $e:X\to X$ 是幂等。态射

$$
(X,e)\to(Y,f)
$$

为 $\mathcal C$ 中态射 $u:X\to Y$，满足

$$
u=fu=ue.
$$

复合由 $\mathcal C$ 中复合给出。

**命题 24.11.** $\operatorname{Kar}(\mathcal C)$ 是幂等完备的，并且嵌入

$$
\mathcal C\to\operatorname{Kar}(\mathcal C),\qquad X\mapsto(X,\operatorname{id}_X)
$$

对任意幂等完备范畴 $\mathcal D$ 满足预复合等价

$$
\operatorname{Fun}(\operatorname{Kar}(\mathcal C),\mathcal D)
\simeq
\operatorname{Fun}(\mathcal C,\mathcal D)
$$

在保持幂等分裂数据的意义下成立。

**证明.** 在 $\operatorname{Kar}(\mathcal C)$ 中，对象 $(X,e)$ 的幂等 $e$ 已由

$$
(X,\operatorname{id})\xrightarrow{e}(X,e)
\xrightarrow{e}(X,\operatorname{id})
$$

分裂。一般幂等态射的分裂可通过同样的 Karoubi envelope 构造完成。若 $\mathcal D$ 幂等完备，任意函子 $F:\mathcal C\to\mathcal D$ 把幂等 $e$ 送到 $\mathcal D$ 中可分裂幂等 $F(e)$，于是可把 $(X,e)$ 送到 $F(e)$ 的像对象。态射条件 $u=fu=ue$ 保证该赋值在分裂像之间良定义。反向限制到 $\mathcal C$ 显然。两者互逆到自然同构。$\square$

**注 24.12.** Cauchy completion 也可由可表预层在小余极限中的绝对闭包刻画：它加入所有由幂等分裂产生的 retract。该观点连接到富范畴中的 Cauchy modules 和 Morita 理论。

## 24.5 Profunctor 与加权余极限

**定义 24.13.** 设 $W:\mathcal J^{op}\to\mathbf{Set}$ 为权重，$D:\mathcal J\to\mathcal C$ 为图形。若存在对象 $W\star D\in\mathcal C$，满足对任意 $X\in\mathcal C$ 有自然同构

$$
\mathcal C(W\star D,X)
\cong
\operatorname{Nat}(W,\mathcal C(D-,X)),
$$

则称 $W\star D$ 为加权余极限。

**命题 24.14.** 在 $\mathbf{Set}$ 中，加权余极限可由 coend 表示：

$$
W\star D\cong\int^{j\in\mathcal J}W(j)\times D(j).
$$

**证明.** 对任意集合 $X$，

$$
\mathbf{Set}\left(\int^{j}W(j)\times D(j),X\right)
\cong
\int_j\mathbf{Set}(W(j)\times D(j),X).
$$

由笛卡尔闭结构，

$$
\mathbf{Set}(W(j)\times D(j),X)
\cong
\mathbf{Set}(W(j),\mathbf{Set}(D(j),X)).
$$

end 的自然性条件正是从 $W$ 到 $\mathbf{Set}(D-,X)$ 的自然变换条件，所以得到

$$
\mathbf{Set}\left(\int^{j}W(j)\times D(j),X\right)
\cong
\operatorname{Nat}(W,\mathbf{Set}(D-,X)).
$$

由表示性得到结论。$\square$

## 24.6 $\infty$-correspondences

**定义 24.15.** $\infty$-范畴中的 correspondence 可粗略看作 span

$$
C\xleftarrow{p}M\xrightarrow{q}D
$$

或等价地看作函子

$$
C^{op}\times D\to\mathcal S.
$$

后者是第二十三章预层 $\infty$-范畴语言下的 space 值 profunctor。

**外部输入定理 24.16.** 在合适模型中，小 $\infty$-范畴、space 值 profunctors

$$
C^{op}\times D\to\mathcal S
$$

和自然变换构成 $(\infty,2)$-范畴 $\operatorname{Corr}$。复合由同伦 coend 给出：

$$
(Q\circ P)(c,e)\simeq
\int^{d\in D}P(c,d)\times Q(d,e).
$$

该结构是 ordinary $\mathbf{Prof}$ 的高阶版本，也是第二十二章 Morita $\infty$-范畴中“以 bimodule 为态射、相对张量积为复合”的抽象原型。

## 24.7 本章小结

Profunctor 把函子、关系、双模和 correspondence 统一为“广义态射”。Coend 复合给出 $\mathbf{Prof}$ 的双范畴结构；Cauchy completion 说明幂等分裂是 Morita 不变量的基本有限性修正；$\infty$-correspondence 则把这些思想提升到高阶范畴和高阶代数。

## 练习

**练习 24.1.** 给定函子 $F:\mathcal C\to\mathcal D$，写出 $F_*$ 与 $F^*$ 的定义。

**练习 24.2.** 对 profunctors $P,Q$，写出 $(Q\circ P)(c,e)$ 的 coend 公式。

**练习 24.3.** 用 co-Yoneda 证明 $\operatorname{id}_{\mathcal D}\circ P\cong P$。

**练习 24.4.** 说明命题 24.7 的余单位如何由 $\mathcal D$ 中复合给出。

**练习 24.5.** 在一个范畴中写出幂等态射分裂的定义。

**练习 24.6.** 验证 $\operatorname{Kar}(\mathcal C)$ 中态射条件 $u=fu=ue$ 对复合封闭。

**练习 24.7.** 若 $\mathcal C$ 已幂等完备，说明 $\mathcal C\to\operatorname{Kar}(\mathcal C)$ 为什么是等价。

**练习 24.8.** 在 $\mathbf{Set}$ 中，把普通余极限写成加权余极限。

**练习 24.9.** 比较 profunctor $C^{op}\times D\to\mathbf{Set}$ 与 $\infty$-correspondence $C^{op}\times D\to\mathcal S$。

**练习 24.10.** 解释为什么 Morita 理论中的双模复合可看作 profunctor coend 复合的高阶代数版本。
