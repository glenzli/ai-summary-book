# 第二十一章：高阶 topos

## 本章目标

本章定义 $\infty$-topos，说明它如何推广 Grothendieck topos，并列出高阶 Giraud 定理和 sheaf $\infty$-范畴的基本性质。

## 依赖前置知识

需要 Grothendieck topos、presentable $\infty$-category、极限、余极限和 left exact localization。

## 21.1 spaces

**定义 21.1.** spaces 的 $\infty$-范畴记作 $\mathcal S$，可由 Kan 复形的同伦理论或 anima 模型给出。集合范畴嵌入为离散 spaces。

**注 21.2.** 高阶 topos 中的 sheaf 通常取值于 $\mathcal S$，而不是 $\mathbf{Set}$。这保留了高阶同伦信息。

## 21.2 $\infty$-sheaves

**定义 21.3.** 设 $(\mathcal C,J)$ 为小站点。一个 space 值预层

$$
F:\mathcal C^{\operatorname{op}}\to\mathcal S
$$

称为 $\infty$-sheaf，若对每个覆盖超对象或覆盖筛，$F$ 把它送为极限图形。对覆盖族形式，条件可写作

$$
F(U)\simeq
\lim\left(
\prod_i F(U_i)
\rightrightarrows
\prod_{i,j}F(U_i\times_U U_j)
\triplearrows\cdots
\right)
$$

即 Čech nerve 的下降条件。

**定义 21.4.** 记

$$
\operatorname{Sh}_\infty(\mathcal C,J)
$$

为 $\infty$-sheaf 构成的 $\infty$-范畴。

**命题 21.5.** 在覆盖族有有限交叉的普通站点上，集合值预层 $F:\mathcal C^{\operatorname{op}}\to\mathbf{Set}$ 是 ordinary sheaf，当且仅当把 $F$ 视为离散 space 值预层后满足 Čech 形式的 $\infty$-sheaf 条件。

**证明.** 将集合看成离散 space。对覆盖族 $\{U_i\to U\}$，ordinary sheaf 条件说下列图是等化子：

$$
F(U)\longrightarrow \prod_iF(U_i)
\rightrightarrows
\prod_{i,j}F(U_i\times_UU_j).
$$

Čech nerve 的更高层由三重、四重交叉组成。因为所有 $F(V)$ 都是离散 space，Čech 全化的点正是满足两两相容条件的截面族；更高层条件不产生额外同伦数据，只表达两两相容的传递一致性。因此 ordinary sheaf 等化子与离散 space 值 Čech 极限给出同一个集合。

反过来，若离散 space 值预层满足 Čech 极限条件，取 $\pi_0$ 后得到上面的 ordinary sheaf 等化子。故二者等价。$\square$

**定义 21.A.** 设站点有足够有限极限。对象 $U$ 的超覆盖（hypercover）是增广单纯对象

$$
U_\bullet\to U
$$

使得 $U_0\to U$ 是覆盖，并且对每个 $n\ge1$，自然映射

$$
U_n\to M_n(U_\bullet)
$$

是覆盖，其中

$$
M_n(U_\bullet)=(\operatorname{cosk}_{n-1}U_\bullet)_n
$$

是由低于 $n$ 维的数据构成的第 $n$ 个 matching object。

**定义 21.B.** space 值预层 $F$ 满足超下降（hyperdescent），若对每个超覆盖 $U_\bullet\to U$，自然映射

$$
F(U)\longrightarrow \lim_{[n]\in\Delta}F(U_n)
$$

是等价。Čech nerve 是由一个覆盖族产生的特殊超覆盖；因此超下降强于只检查 Čech nerve 的下降条件。

**外部输入定理 21.C.** 在常用的 $\infty$-topos 构造中，sheaf 条件可用合适的覆盖筛、Čech descent 或 hyperdescent 表述；这些表述的等价需要站点和拓扑满足标准小性与局部化假设。完整陈述见 Lurie HTT。

## 21.3 $\infty$-topos

**定义 21.6.** $\infty$-topos 是与某个 $\operatorname{Sh}_\infty(\mathcal C,J)$ 等价的 $\infty$-范畴，或等价地，是 spaces 预层 $\infty$-范畴的 left exact accessible localization。

**命题 21.7.** $\mathcal S$ 是 $\infty$-topos。

**证明.** 取终小范畴 $*$ 上的平凡拓扑。则

$$
\operatorname{Fun}(*^{\operatorname{op}},\mathcal S)\simeq\mathcal S
$$

且 sheaf 条件为空条件。$\square$

## 21.4 高阶 Giraud 定理

**外部输入定理 21.8（高阶 Giraud 定理）.** presentable $\infty$-category $X$ 是 $\infty$-topos，当且仅当它满足高阶 Giraud 公理：小余极限普遍、群胚对象有效、余极限与有限极限满足相容性等。

完整陈述见 Lurie HTT。该定理是普通 Giraud 定理的 $\infty$-范畴升级。

**定义 21.D.** 设 $\mathcal X$ 是有有限极限的 $\infty$-范畴。态射 $f:U\to X$ 称为 effective epimorphism，若其 Čech nerve

$$
U_\bullet=U\times_X\cdots\times_XU
$$

的几何实现存在，且自然态射

$$
\left|U_\bullet\right|\to X
$$

是等价。

**命题 21.E.** 在 ordinary topos 的 nerve 中，effective epimorphism 恢复通常意义下的满射型有效覆盖。

**证明.** ordinary topos 中，态射 $U\to X$ 的 Čech nerve 是内部等价关系

$$
U\times_XU\rightrightarrows U.
$$

其几何实现退化为该等价关系的商。若 $U\to X$ 是 effective epimorphism，则这个商为 $X$，也就是说 $X$ 是由 $U$ 按同纤维关系粘合得到。反过来，ordinary topos 中的有效满射按定义正是其核偶的商，因此 Čech nerve 的几何实现恢复目标。$\square$

**定义 21.F.** $\infty$-范畴 $\mathcal X$ 中的 groupoid object 是函子

$$
G_\bullet:\Delta^{op}\to\mathcal X
$$

满足 Segal 条件和所有箭头可逆的条件。直观上，它是 $\mathcal X$ 内部的同伦等价关系。

**外部输入定理 21.G.** 在 $\infty$-topos 中，groupoid objects 有效：任意 groupoid object 都等价于某个态射 $U\to X$ 的 Čech nerve。该结论是高阶 Giraud 定理中“等价关系有效”的同伦升级。

## 21.5 截断、Postnikov 塔与普通 topos

**定义 21.9.** 在 $\infty$-topos $X$ 中，对象称为 $0$-截断，若其任意映射对象是离散 space。$0$-截断对象组成普通范畴 $X_{\le0}$。

**外部输入命题 21.10.** 若 $X$ 是 $\infty$-topos，则 $X_{\le0}$ 是 Grothendieck topos。对 $X=\operatorname{Sh}_\infty(\mathcal C,J)$，$0$-截断部分恢复集合值 sheaf topos。

**定义 21.H.** 对整数 $n\ge -2$，对象 $A\in\mathcal X$ 称为 $n$-截断，若对任意 $T\in\mathcal X$，映射空间

$$
\operatorname{Map}_{\mathcal X}(T,A)
$$

是 $n$-截断 space。这里 $(-2)$-截断表示要么空要么可缩，$(-1)$-截断对象也称为 subterminal 或 proposition-valued object。

**外部输入定理 21.I.** 在 $\infty$-topos $\mathcal X$ 中，包含

$$
\mathcal X_{\le n}\hookrightarrow\mathcal X
$$

有左伴随

$$
\tau_{\le n}:\mathcal X\to\mathcal X_{\le n},
$$

称为 $n$-截断函子。它保持有限极限，并且这些截断组成 Postnikov tower

$$
\cdots\to\tau_{\le n}X\to\tau_{\le n-1}X\to\cdots\to\tau_{\le0}X.
$$

**例子 21.J.** 在 $\mathcal S$ 中，$0$-截断对象是集合，$1$-截断对象是群胚的同伦类型，$n$-截断对象是高于 $n$ 的同伦群消失的 homotopy types。截断函子 $\tau_{\le n}$ 是通常 Postnikov 截断。

**定义 21.K.** $\infty$-topos $\mathcal X$ 称为 hypercomplete，若每个对象 $X$ 都可由其 Postnikov tower 恢复，即自然映射

$$
X\to\lim_n\tau_{\le n}X
$$

在适当意义下为等价。一般 $\infty$-topos 的 hypercompletion 是一个 left exact localization

$$
\mathcal X\to\widehat{\mathcal X}.
$$

**外部输入定理 21.L.** Hypercompletion 可由局部化所有 $\infty$-connective morphisms 得到，并且是 left exact accessible localization。对满足足够好局部同伦条件的站点，hypercomplete sheaves 与满足 hyperdescent 的 sheaves 一致。

## 21.6 几何态射与点

**定义 21.M.** $\infty$-topoi 之间的几何态射

$$
f:\mathcal X\to\mathcal Y
$$

由伴随

$$
f^*:\mathcal Y\rightleftarrows\mathcal X:f_*
$$

给出，其中 $f^*$ 是左伴随并保持有限极限。若 $\mathcal Y=\mathcal S$，则几何态射

$$
\mathcal S\to\mathcal X
$$

称为 $\mathcal X$ 的一个点。

**例子 21.N.** 若 $X$ 是拓扑空间，则 space 值 sheaf 构成 $\infty$-topos

$$
\operatorname{Sh}_\infty(X).
$$

点 $x\in X$ 给出几何态射

$$
\mathcal S\to\operatorname{Sh}_\infty(X),
$$

其 inverse image 为取 stalk：

$$
F\mapsto F_x.
$$

stalk 保持有限极限，因为它由邻域系统上的滤过余极限计算，而 spaces 中滤过余极限与有限极限在此标准情形下相容。

**外部输入命题 21.O.** 连续映射 $f:X\to Y$ 诱导几何态射

$$
\operatorname{Sh}_\infty(X)\to\operatorname{Sh}_\infty(Y),
$$

其 inverse image 是 sheafified pullback，direct image 是沿开集反像限制后的截面函子。这是 ordinary sheaf 几何态射的 space 值版本。

## 21.7 本章小结

$\infty$-topos 是 homotopy type 取值的 sheaf 理论。它保留 ordinary topos 的逻辑和几何结构，同时允许对象有高阶同伦。Čech descent 被替换为同伦极限条件，高阶 Giraud 定理给出内在刻画。截断和 Postnikov 塔连接高阶对象与普通 topos；hypercompletion 则控制是否所有对象都能由截断层恢复。

## 练习

**练习 21.1.** 解释集合值 sheaf 条件与 space 值 sheaf 条件的差别。

**练习 21.2.** 对平凡站点 $*$，计算 $\operatorname{Sh}_\infty(*,\mathrm{triv})$。

**练习 21.3.** 写出 Čech nerve 的前三层。

**练习 21.4.** 查阅 left exact localization 的定义，并说明为什么 sheaf 化应保持有限极限。

**练习 21.5.** 比较普通 Giraud 定理和高阶 Giraud 定理的公理类型。

**练习 21.6.** 证明离散 space 值 $\infty$-sheaf 的 $0$-截断部分满足 ordinary sheaf 条件。

**练习 21.7.** 说明 $\mathcal S$ 中的 $0$-截断对象为什么等价于集合。

**练习 21.8.** 说明 Čech nerve 为什么是超覆盖的一个特例。

**练习 21.9.** 比较 Čech descent 与 hyperdescent：后者多检查了什么数据？

**练习 21.10.** 写出 effective epimorphism 的 Čech nerve，并解释其几何实现为何应恢复目标。

**练习 21.11.** 在集合范畴中，说明 effective epimorphism 与满射的关系。

**练习 21.12.** 解释 groupoid object 为什么是“内部同伦等价关系”。

**练习 21.13.** 在 $\mathcal S$ 中给出 $0$-截断和 $1$-截断对象的例子。

**练习 21.14.** 说明 Postnikov tower 在 $\infty$-topos 中的作用。

**练习 21.15.** 比较 hyperdescent 与 hypercompletion 的关系。

**练习 21.16.** 写出 $\infty$-topoi 几何态射中 $f^*$ 与 $f_*$ 的伴随方向。

**练习 21.17.** 对拓扑空间的点 $x\in X$，解释 stalk 为什么可看作几何态射的 inverse image。

**练习 21.18.** 比较 ordinary topos 的几何态射和 $\infty$-topos 的几何态射：有限极限保持条件是否相同？
