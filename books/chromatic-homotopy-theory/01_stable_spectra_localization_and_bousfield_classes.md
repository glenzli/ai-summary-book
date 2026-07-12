# 第一章：稳定谱、Bousfield 类与局部化

## 本章目标

本章建立 chromatic theory 的范畴论底座：谱、有限谱、同调理论、Bousfield acyclic、local object 和 localization。后续所有 $K(n)$、$E(n)$、$T(n)$ 和 chromatic tower 都是本章结构的特例。

## 依赖前置知识

需要稳定 infinity-范畴的 fiber/cofiber 序列、mapping spectrum、张量积和 exact functor。若只熟悉三角范畴，可把 fiber/cofiber 序列读作 distinguished triangles，但要注意本书证明使用稳定 infinity-范畴中的泛性质。

## 1.1 谱与有限谱

**定义 1.1.** 谱的稳定 infinity-范畴记为 $\mathbf{Sp}$。它带有对称幺半结构
$$
\otimes:\mathbf{Sp}\times\mathbf{Sp}\to\mathbf{Sp},
$$
单位为球谱 $\mathbb S$，内部 Hom 记为 $F(X,Y)$。

**定义 1.2.** 一个 $p$-局部谱是 $\mathbf{Sp}$ 中使 $\pi_*X$ 为 $\mathbb Z_{(p)}$-模的谱。$p$-局部谱的全子范畴记为 $\mathbf{Sp}_{(p)}$。

**定义 1.3.** 有限 $p$-局部谱是 $\mathbf{Sp}_{(p)}$ 中由 $\mathbb S_{(p)}$ 经有限次悬挂、脱悬挂、cofiber 和 retract 生成的对象。有限谱范畴记作 $\mathbf{Sp}^{\omega}_{(p)}$。

**命题 1.4.** 每个有限 $p$-局部谱在 $\mathbf{Sp}_{(p)}$ 中 compact。

**证明.** $\mathbb S_{(p)}$ compact，因为
$$
\operatorname{Map}(\mathbb S_{(p)},\operatorname*{colim}_i X_i)\simeq \Omega^\infty \operatorname*{colim}_i X_i
\simeq \operatorname*{colim}_i\Omega^\infty X_i
$$
在 filtered colimit 下保持同伦群。compact 对象类对有限 colimit、悬挂、脱悬挂和 retract 封闭。因此由这些操作生成的有限谱 compact。证毕。

**警告 1.5.** compact 谱不应在未说明范畴时与有限 CW-spectrum 混用。在 $\mathbf{Sp}$、$\mathbf{Sp}_{(p)}$、$K(n)$-local category 或 module category 中，compact 的含义可能不同。

## 1.2 同调理论和 Bousfield acyclic

**定义 1.6.** 给定谱 $E$，它定义同调理论
$$
E_*X=\pi_*(E\otimes X).
$$
谱 $X$ 称为 $E$-acyclic，若
$$
E\otimes X\simeq 0.
$$

**定义 1.7.** 两个谱 $E,F$ 称为 Bousfield 等价，若对任意谱 $X$，
$$
E\otimes X\simeq 0\quad\Longleftrightarrow\quad F\otimes X\simeq 0.
$$
其等价类记作 $\langle E\rangle$。

**例 1.8.** $\langle 0\rangle$ 的 acyclic 对象是所有谱。$\langle \mathbb S\rangle$ 的 acyclic 对象只有零谱，因为 $\mathbb S\otimes X\simeq X$。

**命题 1.9.** 若 $E\simeq E'$，则 $E$ 与 $E'$ Bousfield 等价。

**证明.** 对任意 $X$，等价 $E\simeq E'$ 张量 $X$ 后给出 $E\otimes X\simeq E'\otimes X$。因此一个为零当且仅当另一个为零。证毕。

**命题 1.10.** 若 $X\to Y\to Z$ 是 fiber/cofiber 序列，且 $X,Y$ 都 $E$-acyclic，则 $Z$ 也是 $E$-acyclic。

**证明.** 在稳定范畴中，exact functor $E\otimes -$ 保持 fiber/cofiber 序列，得到
$$
E\otimes X\to E\otimes Y\to E\otimes Z.
$$
前两项为零，故第三项为零。证毕。

## 1.3 Local objects 和 localization

**定义 1.11.** 谱 $Y$ 称为 $E$-local，若对任意 $E$-acyclic 谱 $A$，
$$
F(A,Y)\simeq 0.
$$

**定义 1.12.** 一个 $E$-localization of $X$ 是映射 $\eta_X:X\to L_EX$，满足：

1. $L_EX$ 是 $E$-local；
2. fiber $C_EX=\operatorname{fib}(X\to L_EX)$ 是 $E$-acyclic。

**命题 1.13.** 若 $X\to L_EX$ 满足定义 1.12，则对任意 $E$-local 谱 $Y$，诱导映射
$$
F(L_EX,Y)\to F(X,Y)
$$
是等价。

**证明.** 由 fiber 序列
$$
C_EX\to X\to L_EX
$$
对 $Y$ 取 $F(-,Y)$，得到 fiber 序列
$$
F(L_EX,Y)\to F(X,Y)\to F(C_EX,Y).
$$
因为 $C_EX$ 是 $E$-acyclic 且 $Y$ 是 $E$-local，第三项为零。因此第一项到第二项为等价。证毕。

**推论 1.14.** $E$-localization 若存在，则在 contractible choice 意义下唯一。

**证明.** 若 $X\to L_EX$ 与 $X\to L'_EX$ 都是 localization，则由命题 1.13 分别取 $Y=L'_EX$ 和 $Y=L_EX$，得到唯一的互逆映射 $L_EX\to L'_EX$ 与 $L'_EX\to L_EX$，并且它们与 $X$ 下方结构相容。稳定 infinity-范畴中的 mapping space 版本给出 contractible choice。证毕。

**定义 1.14A.** 谱映射 $f:X\to Y$ 称为 $E$-等价，若
$$
E\otimes f:E\otimes X\longrightarrow E\otimes Y
$$
是等价。由于 $E\otimes-$ 是 exact functor，这等价于
$\operatorname{fib}(f)$（也等价于 $\operatorname{cofib}(f)$）为
$E$-acyclic。

**命题 1.14B（局部化的 exact 性与消失判据）.** 假设 $L_E$ 存在。则
$E$-local 谱构成稳定全子范畴，$L_E$ 是 exact 且幂等的反射函子；并且
对任意谱 $X$，
$$
L_EX\simeq0\quad\Longleftrightarrow\quad E\otimes X\simeq0.
$$

**证明.** 若 $Y_1\to Y_2\to Y_3$ 是 fiber/cofiber 序列，则对任意
$E$-acyclic 谱 $A$，函子 $F(A,-)$ 把它送到 fiber/cofiber 序列。因此
三个 $Y_i$ 中任意两个 $E$-local 时第三个也 $E$-local；对任意
$E$-acyclic 谱 $A$ 都有 $F(A,0)\simeq0$，故零谱也 $E$-local，所以
local 对象构成稳定全子范畴。反射函子 $L_E$ 作为左
伴随保持有限 colimit，而稳定范畴中的有限 colimit 与有限 limit 相互
决定，故 $L_E$ exact。目标 $L_EX$ 已经 local；对它再次局部化并使用
推论 1.14，得到 $L_E^2X\simeq L_EX$。

若 $X$ 为 $E$-acyclic，则 localization map 是 $E$-等价，所以 $L_EX$
也 $E$-acyclic。一个同时 acyclic 与 local 的谱 $Y$ 满足
$F(Y,Y)\simeq0$；其恒等映射为零，故 $Y\simeq0$。反之，若
$L_EX\simeq0$，则 $X\simeq\operatorname{fib}(X\to L_EX)$ 按定义
$E$-acyclic。证毕。

**命题 1.14C（有限谱与局部化的 base change）.** 设
$P\in\mathbf{Sp}_{(p)}^\omega$，$E,X\in\mathbf{Sp}_{(p)}$，且
$L_E:\mathbf{Sp}_{(p)}\to\mathbf{Sp}_{(p)}$ 存在。则 $P$ dualizable，
并有自然等价
$$
L_E(P\otimes X)\simeq P\otimes L_EX.
$$
这里不要求 $L_E$ smashing；有限性假设施加在 $P$ 上。

**证明.** 球谱 dualizable，dualizable 对象对有限 cofiber、悬挂、
脱悬挂和 retract 封闭，故有限谱 $P$ dualizable。记
$DP=F(P,\mathbb S_{(p)})$。映射
$$
P\otimes X\longrightarrow P\otimes L_EX
$$
的 fiber 是 $P\otimes C_EX$；它仍 $E$-acyclic，因为
$$
E\otimes P\otimes C_EX\simeq P\otimes(E\otimes C_EX)\simeq0.
$$
另一方面，对任意 $E$-acyclic 谱 $A$，dualizability 给出
$$
F(A,P\otimes L_EX)\simeq F(DP\otimes A,L_EX).
$$
$DP\otimes A$ 仍 $E$-acyclic，所以右端为零。因此
$P\otimes L_EX$ 是 $E$-local，所示映射满足定义 1.12，结论由推论
1.14 得到。证毕。

**命题 1.14D（嵌套 acyclic 类的局部化）.** 设 $L_E,L_F$ 均存在，且
$$
\mathcal A_E\subseteq\mathcal A_F,
\qquad
\mathcal A_E=\{A\mid E\otimes A\simeq0\}.
$$
则每个 $F$-local 谱都是 $E$-local，并有自然等价
$$
L_EL_F\simeq L_F,
\qquad
L_FL_E\simeq L_F.
$$

**证明.** $F$-local 对象对较大的 acyclic 类 $\mathcal A_F$ 正交，因而
也对 $\mathcal A_E$ 正交；这证明第一项及 $L_EL_F\simeq L_F$。映射
$X\to L_EX$ 的 fiber 属于 $\mathcal A_E\subseteq\mathcal A_F$，所以它
也是 $F$-等价。施加 $L_F$ 后得到
$L_FX\simeq L_FL_EX$。两项等价都与 localization unit 自然相容。证毕。

## 1.4 smashing 与非 smashing

**定义 1.15.** localization $L$ 称为 smashing，若对任意谱 $X$，自然映射
$$
L\mathbb S\otimes X\to LX
$$
是等价。

**命题 1.16.** 若 $L$ 是 smashing localization，则 $L$ 保持任意 colimit。

**证明.** $L(X)\simeq L\mathbb S\otimes X$。张量固定对象 $L\mathbb S$ 是左伴随，因此保持 colimit。证毕。

**警告 1.17.** $K(n)$-localization 一般不是 smashing。$E(n)$-localization 和 finite localization 的 smashing 性需要分开讨论。把所有 chromatic localization 都当作 smashing 是常见错误。

## 1.5 Bousfield 类的有限生成测试

**定义 1.18.** 若一族谱 $\{E_i\}_{i\in I}$ 给定，记
$$
\bigvee_iE_i
$$
为其 wedge。一个谱 $X$ 对 $\bigvee_iE_i$ acyclic，当且仅当对所有 $i$，$E_i\otimes X\simeq 0$。

**证明.** 因为张量积保持 colimit，
$$
\left(\bigvee_iE_i\right)\otimes X\simeq \bigvee_i(E_i\otimes X).
$$
若右端为零，则每个 summand 通过包含和投影是零。反之若每个 $E_i\otimes X$ 为零，则 wedge 为零。证毕。

**例 1.19.** $E(n)$ 的 Bousfield 类常按
$$
\langle E(n)\rangle=\langle K(0)\vee K(1)\vee\cdots\vee K(n)\rangle
$$
处理。这个等式是 chromatic theory 的外部输入/标准定理，不能只由系数环形式推出。

## 1.6 Localizing subcategory 生成的 acyclics

**定义 1.20.** 给定一族谱 $\mathcal S$，记
$$
\operatorname{Loc}^{\otimes}(\mathcal S)
$$
为包含 $\mathcal S$ 且对悬挂、cofiber、小 colimit 和任意谱张量封闭的最小全子范畴。若一个 localization 的 acyclics 等于 $\operatorname{Loc}^{\otimes}(\mathcal S)$，则称该 localization 由 $\mathcal S$ 生成。

**命题 1.21.** 若 $E=\bigvee_{S\in\mathcal S}S$，则 $E$-acyclic 谱构成包含所有 $E$-acyclic 生成操作的 localizing tensor ideal。

**证明.** 由命题 H.2，$E$-acyclics 对悬挂、cofiber 和小 colimit 封闭。若 $X$ 为 $E$-acyclic，则
$$
E\otimes(X\otimes Y)\simeq(E\otimes X)\otimes Y\simeq0
$$
对任意谱 $Y$ 成立，所以对张量封闭。证毕。

**警告 1.22.** “由 $\mathcal S$ 生成的 localization”有两个层次：一是 acyclic localizing subcategory 由 $\mathcal S$ 生成；二是实际 localization functor 存在并有小性控制。presentable stable infinity-category 中的可访问 localization 提供存在性，但具体引用仍应定位。

## 1.7 计算 local object 的基本方法

**命题 1.23.** 谱 $Y$ 为 $E$-local，当且仅当对所有生成 acyclics 的集合 $\mathcal S$ 及其 localizing tensor ideal 中对象 $A$，有 $F(A,Y)\simeq0$。若已知 $E$-acyclics 由 $\mathcal S$ 生成，则只需检查 $F(-,Y)$ 把生成操作送到 limits 中的零对象。

**证明.** “仅当”是定义。反向中，$F(-,Y)$ 把 colimit 变成 limit，把 cofiber 变成 fiber，并与悬挂相容。若它在生成元上为零，则在由生成元经这些操作得到的对象上仍为零。证毕。

**例 1.24.** 对 finite localization $L_n^f$，acyclics 由 type $n+1$ 有限谱生成。判断 $Y$ 为 $L_n^f$-local 时，不能只检查 $F(F_{n+1},Y)$ 对某一个 type $n+1$ 谱为零，除非已经引用 thick subcategory theorem 和 telescope 选择无关等外部输入。

## 本章小结

Bousfield localization 把“用某个同调理论看不见的谱”系统地变成 acyclic 子类，并通过 local object 给出反射。chromatic homotopy theory 的色层来自把 $E$ 取为 $K(n)$、$E(n)$、$T(n)$ 及其组合。基础范畴事实可在本书内部证明，但 chromatic 特定等式和 finite localization 需要外部定理。

## 练习

**练习 1.1.** 证明 $E$-acyclic 谱构成稳定子范畴，并对任意 colimit 封闭。

**练习 1.2.** 若 $E\to F$ 是谱映射，且对任意 $X$ 有 $E\otimes X\simeq 0\Rightarrow F\otimes X\simeq 0$，说明这不是由映射 $E\to F$ 形式自动推出的。给出需要额外假设的地方。

**练习 1.3.** 证明若 $L$ 是 smashing localization，则 $LX\simeq 0$ 当且仅当 $L\mathbb S\otimes X\simeq 0$。
