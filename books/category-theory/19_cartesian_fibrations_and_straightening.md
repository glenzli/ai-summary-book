# 第十九章：Cartesian fibration 与 straightening

## 本章目标

本章介绍 Cartesian fibration、coCartesian fibration 和 straightening/unstraightening 定理。这是 $\infty$-范畴中处理变动范畴族和高阶函子的核心工具。

## 依赖前置知识

需要 quasi-category、slice、映射空间和普通 Grothendieck 构造的直觉。

## 19.1 Cartesian 边

**定义 19.1.** 设 $p:X\to S$ 是 quasi-category 之间的内纤维。边 $e:x\to y$ 称为 $p$-Cartesian，若对任意 $z\in X$，诱导方块

$$
\operatorname{Map}_X(z,x)\to\operatorname{Map}_X(z,y)
\times_{\operatorname{Map}_S(pz,py)}
\operatorname{Map}_S(pz,px)
$$

是同伦拉回，等价地满足相应 horn lifting 条件。

**定义 19.2.** 内纤维 $p:X\to S$ 称为 Cartesian fibration，若对每条边 $\alpha:s\to t$ 和每个 $y\in X_t$，存在 $p$-Cartesian 边 $x\to y$ 覆盖 $\alpha$。

coCartesian fibration 对偶定义。

**定义 19.3.** marked simplicial set 是二元组 $(X,E)$，其中 $X$ 是单纯集，$E\subseteq X_1$ 包含所有退化边。若 $C$ 是 quasi-category，则 $C^\natural$ 表示标记所有等价边，$C^\sharp$ 表示标记所有边，$C^\flat$ 表示只标记退化边。

**外部输入定理 19.4.** 对固定 quasi-category $S$，存在 marked simplicial sets over $S$ 的 Cartesian model structure。其 fibrant objects 是 Cartesian fibrations $X\to S$，并且 marked edges 正是 Cartesian edges。

## 19.2 普通范畴中的类比

**例子 19.5.** 普通范畴中的 Grothendieck fibration $p:E\to B$ 的 nerve 在适当条件下给出 Cartesian fibration

$$
N(E)\to N(B).
$$

Cartesian 边对应普通 fibration 中的 Cartesian lift。

**外部输入命题 19.6.** 若普通 Grothendieck fibration $p:E\to B$ 的 Cartesian arrows 被标记，则

$$
N(E)^\natural\to N(B)
$$

是 marked simplicial set over $N(B)$ 的 Cartesian model structure 中的 fibrant object。

## 19.3 普通 Grothendieck construction

**定义 19.A.** 设 $F:B^{\operatorname{op}}\to\mathbf{Cat}$ 是函子。其 Grothendieck construction 记为

$$
\int_BF.
$$

对象是对 $(b,x)$，其中 $b\in B$ 且 $x\in F(b)$。从 $(b,x)$ 到 $(c,y)$ 的态射是对

$$
(\alpha:b\to c,\ \varphi:x\to F(\alpha)(y)),
$$

其中 $\varphi$ 是 $F(b)$ 中的态射。复合定义如下：若

$$
(b,x)\xrightarrow{(\alpha,\varphi)}(c,y)
\xrightarrow{(\beta,\psi)}(d,z),
$$

则复合为

$$
(\beta\alpha,\ F(\alpha)(\psi)\circ\varphi).
$$

**命题 19.B.** 投影

$$
\pi:\int_BF\to B,\qquad (b,x)\mapsto b
$$

是普通 Grothendieck fibration。

**证明.** 先验证 $\int_BF$ 是范畴。恒等态射为

$$
(\operatorname{id}_b,\operatorname{id}_x):(b,x)\to(b,x).
$$

复合的结合律来自 $B$ 中复合的结合律、$F$ 的函子性以及各纤维范畴 $F(b)$ 中复合的结合律。

现在给定 $\alpha:b\to c$ 和对象 $(c,y)$，考虑态射

$$
(\alpha,\operatorname{id}_{F(\alpha)y}):(b,F(\alpha)y)\to(c,y).
$$

它覆盖 $\alpha$。设有态射

$$
(\gamma,\theta):(a,z)\to(c,y)
$$

并给定分解 $\gamma=\alpha\beta$，其中 $\beta:a\to b$。由于 $F$ 反变，

$$
F(\gamma)y=F(\beta)F(\alpha)y.
$$

于是 $\theta:z\to F(\gamma)y$ 也可看作

$$
\theta:z\to F(\beta)(F(\alpha)y),
$$

从而给出唯一态射

$$
(\beta,\theta):(a,z)\to(b,F(\alpha)y)
$$

覆盖 $\beta$。按复合公式，它与 $(\alpha,\operatorname{id})$ 的复合正是 $(\gamma,\theta)$。唯一性由 Grothendieck construction 中态射的第二分量被该等式强制确定。故该态射 Cartesian，且每个 $\alpha$ 和 $(c,y)$ 都有 Cartesian lift。$\square$

## 19.4 straightening/unstraightening

**外部输入定理 19.7（straightening/unstraightening）.** 对任意 $\infty$-范畴 $S$ 和合适目标 $\infty$-范畴 $\mathcal C$，存在等价

$$
\operatorname{Fun}(S^{\operatorname{op}},\mathcal C)
\simeq
\operatorname{CartFib}_{/S}(\mathcal C)
$$

在 $\mathcal C=\mathcal{Cat}_\infty$ 时，右边是 $S$ 上的 Cartesian fibrations。对偶地，

$$
\operatorname{Fun}(S,\mathcal{Cat}_\infty)
\simeq
\operatorname{coCartFib}_{/S}.
$$

这是 Lurie HTT 的核心定理之一。

**注 19.8.** 在普通范畴中，伪函子 $B^{\operatorname{op}}\to\mathbf{Cat}$ 与 Grothendieck fibrations over $B$ 等价。定理 19.7 是其 $\infty$-范畴版本。

**例子 19.C（基为 $[1]$ 的低维 straightening）.** 令 $B=[1]$，即有唯一非恒等箭头 $0\to1$ 的范畴。给定反变函子

$$
F:[1]^{\operatorname{op}}\to\mathbf{Cat},
$$

等价于给出两个范畴

$$
\mathcal A=F(0),\qquad \mathcal B=F(1)
$$

和一个函子

$$
u=F(0\to1):\mathcal B\to\mathcal A.
$$

Grothendieck construction $\int_{[1]}F$ 的对象为 $(0,a)$ 与 $(1,b)$，其中 $a\in\mathcal A$，$b\in\mathcal B$。同一纤维内的态射就是 $\mathcal A$ 或 $\mathcal B$ 中的态射；跨纤维态射只能从 $0$ 到 $1$：

$$
(0,a)\to(1,b)
$$

由 $\mathcal A$ 中的态射

$$
a\to u(b)
$$

给出。覆盖 $0\to1$ 的 Cartesian lift 以 $b\in\mathcal B$ 为终点时是

$$
(0,u b)\xrightarrow{(0\to1,\operatorname{id}_{u b})}(1,b).
$$

因此，在基为 $[1]$ 的情形，Cartesian fibration 精确记录了一个“拉回”或“限制”函子 $u:\mathcal B\to\mathcal A$。这就是 straightening/unstraightening 在最低维基范畴上的可见模型。

**定义 19.D.** 设 $p:X\to S$ 是 Cartesian fibration，$\alpha:s\to t$ 是 $S$ 中边。对 $y\in X_t$ 选择覆盖 $\alpha$ 的 Cartesian 边

$$
\alpha^*y\to y.
$$

该选择在可缩空间中唯一，并诱导传输函子

$$
\alpha^*:X_t\to X_s.
$$

它称为沿 $\alpha$ 的 Cartesian pullback 或 restriction。

**命题 19.E.** 若 $\alpha:s\to t$ 与 $\beta:t\to u$ 可复合，则存在自然等价

$$
(\beta\alpha)^*\simeq\alpha^*\beta^*.
$$

**证明.** 对 $z\in X_u$，取 Cartesian lift

$$
\beta^*z\to z
$$

覆盖 $\beta$，再取

$$
\alpha^*\beta^*z\to\beta^*z
$$

覆盖 $\alpha$。两条 Cartesian 边的复合覆盖 $\beta\alpha$。Cartesian 边定义中的映射空间同伦拉回方块在复合下仍为同伦拉回，因此该复合仍是 Cartesian。另一方面，$(\beta\alpha)^*z\to z$ 也是覆盖 $\beta\alpha$ 的 Cartesian lift。Cartesian lift 的选择空间可缩，故二者自然等价。态射上的相容性同样由 Cartesian 泛性质给出。$\square$

**例子 19.F（基为 $[2]$ 的相干性）.** 若

$$
F:[2]^{op}\to\mathbf{Cat}
$$

给出三个范畴 $\mathcal C_0,\mathcal C_1,\mathcal C_2$ 和限制函子

$$
u_{01}:\mathcal C_1\to\mathcal C_0,\qquad
u_{12}:\mathcal C_2\to\mathcal C_1,\qquad
u_{02}:\mathcal C_2\to\mathcal C_0,
$$

则函子性给出

$$
u_{02}=u_{01}u_{12}.
$$

在 $\infty$-范畴情形中，该等式被提升为指定等价和更高相干。命题 19.E 是这种相干性在一条 $2$-单纯形上的表现。

## 19.5 左右 Kan 延拓的高阶形式

**定义 19.9.** 在 $\infty$-范畴中，沿 $K:C\to D$ 的左 Kan 延拓仍可定义为预复合函子

$$
K^*:\operatorname{Fun}(D,E)\to\operatorname{Fun}(C,E)
$$

的左伴随在对象上的值。若点态存在，则公式为

$$
(\operatorname{Lan}_K F)(d)\simeq
\operatorname{colim}_{C\times_D D_{/d}}F.
$$

**外部输入定理 19.10.** presentable $\infty$-categories 中的 Kan 延拓在广泛条件下存在，并可由相应 slice $\infty$-范畴上的极限/余极限逐点计算。

## 19.6 Cartesian sections 与极限

**定义 19.G.** 设 $p:X\to S$ 是 Cartesian fibration。一个 section 是满足 $p\sigma=\operatorname{id}_S$ 的函子

$$
\sigma:S\to X.
$$

若 $\sigma$ 把 $S$ 中每条边送到 $p$-Cartesian 边，则称 $\sigma$ 为 Cartesian section。所有 Cartesian sections 组成的 $\infty$-范畴记为

$$
\operatorname{Sect}^{Cart}_S(X).
$$

**外部输入定理 19.H.** 若 $p:X\to S$ 在 straightening 下对应函子

$$
F:S^{op}\to\mathcal{Cat}_\infty,
$$

则存在自然等价

$$
\operatorname{Sect}^{Cart}_S(X)\simeq\lim_{s\in S^{op}}F(s).
$$

因此，Cartesian section 是“对每个 $s$ 选 $F(s)$ 中对象，并且对每条边给出与 restriction 相容的等价”的同伦相干系统。

**外部输入命题 19.I.** 在 straightening/unstraightening 等价下，$S$ 上 Cartesian fibrations 之间保持 Cartesian edges 的函子对应自然变换

$$
F\to G
$$

其中 $F,G:S^{op}\to\mathcal{Cat}_\infty$ 是对应的 straightened functors。

## 19.7 低维边界与纤维内态射

**命题 19.11（基为点）.** 映射 $p:X\to\Delta^0$ 是 Cartesian fibration，当且仅当 $X$ 是 quasi-category。此时每个对象上方的退化边给出所需的 Cartesian lift。

**证明.** 到 $\Delta^0$ 的映射唯一。若 $p$ 是 Cartesian fibration，则按定义它是内纤维，故 $X$ 是 quasi-category。反过来，若 $X$ 是 quasi-category，则 $X\to\Delta^0$ 是内纤维。基底只有恒等边；任意对象 $y\in X$ 上方的 Cartesian lift 可取退化边 $\operatorname{id}_y:y\to y$。对恒等基底和退化边，Cartesian 条件化为由恒等态射诱导的映射空间同伦拉回，显然成立。因此 Cartesian lift 存在，$p$ 是 Cartesian fibration。注意这里不需要断言所有非退化边都是 Cartesian；在点基情形，该判别与边是否为等价边相关。$\square$

**命题 19.12.** 在普通 Grothendieck construction $\int_BF$ 中，覆盖 $\operatorname{id}_b$ 的态射

$$
(b,x)\to(b,y)
$$

正是纤维范畴 $F(b)$ 中的态射 $x\to y$。

**证明.** 按定义 19.A，一个态射由

$$
(\alpha:b\to c,\ \varphi:x\to F(\alpha)(y))
$$

给出。若它覆盖 $\operatorname{id}_b$，则 $c=b$ 且 $\alpha=\operatorname{id}_b$。由函子性 $F(\operatorname{id}_b)=\operatorname{id}_{F(b)}$，所以第二分量就是 $F(b)$ 中的态射

$$
\varphi:x\to y.
$$

反向任意 $\varphi:x\to y$ 给出 $(\operatorname{id}_b,\varphi)$。$\square$

**例子 19.13（常值普通族）.** 若 $F:B^{op}\to\mathbf{Cat}$ 为常值函子，取值为范畴 $\mathcal A$ 且所有限制函子为恒等，则 $\int_BF$ 同构于 $B\times\mathcal A$。投影到 $B$ 的 Cartesian lift 沿 $\alpha:b\to c$、以 $(c,a)$ 为终点时为

$$
(b,a)\xrightarrow{(\alpha,\operatorname{id}_a)}(c,a).
$$

因此常值族的 Cartesian section 正是选择一个对象 $a\in\mathcal A$，并在基范畴方向保持该对象不变。

## 19.8 本章小结

Cartesian fibration 是“随基点变化的 $\infty$-范畴”的几何化表示。基为点时它退化为一个普通 quasi-category；普通 Grothendieck construction 则说明纤维内态射和跨纤维限制态射如何同时编码。Straightening/unstraightening 说明这种几何对象等价于从基 $\infty$-范畴到 $\mathcal{Cat}_\infty$ 的函子。Cartesian sections 进一步把相容选择解释为一个高阶极限。

## 练习

**练习 19.1.** 回顾普通 Grothendieck fibration 的 Cartesian lift 定义。

**练习 19.2.** 解释为什么 coCartesian fibration 对应协变函子 $S\to\mathcal{Cat}_\infty$。

**练习 19.3.** 对普通函子 $F:B^{\operatorname{op}}\to\mathbf{Cat}$，写出其 Grothendieck construction 的对象和态射。

**练习 19.4.** 比较第六章 Kan 延拓点态公式与定义 19.9。

**练习 19.5.** 查阅 HTT 中 Cartesian fibration 的 horn lifting 定义，并与映射空间定义比较。

**练习 19.6.** 比较 $C^\natural$、$C^\sharp$ 与 $C^\flat$。

**练习 19.7.** 说明 marked simplicial sets 为什么适合记录 Cartesian edges。

**练习 19.8.** 在普通 Grothendieck fibration 中解释 Cartesian arrow 的唯一分解性质如何对应映射空间同伦拉回条件。

**练习 19.9.** 验证定义 19.A 中的恒等态射确实是复合单位。

**练习 19.10.** 在命题 19.B 中，若 $B=[1]$ 且 $F(0)=\mathcal A,F(1)=\mathcal B,F(0\to1)=u:\mathcal B\to\mathcal A$，写出 $\int_BF$ 的对象、纤维和跨纤维态射。

**练习 19.11.** 在例子 19.C 中证明 $(0,u b)\to(1,b)$ 满足普通 Cartesian arrow 的唯一分解性质。

**练习 19.12.** 对 Cartesian fibration $p:X\to S$，解释为什么 $\alpha^*y$ 只在等价意义下唯一。

**练习 19.13.** 在普通 Grothendieck fibration 中证明 Cartesian arrows 的复合仍 Cartesian。

**练习 19.14.** 对 $F:[2]^{op}\to\mathbf{Cat}$，写出 $u_{02}=u_{01}u_{12}$ 对对象 $x\in\mathcal C_2$ 的含义。

**练习 19.15.** 若 $S=[1]$，用定义 19.G 描述 Cartesian section 的数据。

**练习 19.16.** 用外部输入定理 19.H 解释 descent data 为什么可看成某个 Cartesian fibration 的 Cartesian sections。

**练习 19.17.** 证明命题 19.12 的反向：纤维范畴中的态射给出覆盖恒等态射的 Grothendieck construction 态射。

**练习 19.18.** 对常值函子 $F:B^{op}\to\mathbf{Cat}$，构造 $\int_BF\cong B\times\mathcal A$ 的对象和态射映射。

**练习 19.19.** 当 $S=\Delta^0$ 时，用定理 19.H 解释 Cartesian sections 与总空间本身的关系。
