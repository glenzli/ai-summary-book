# 第十一章：end、coend 与 Fubini 定理

## 本章目标

本章定义 end 和 coend，给出它们作为等化子和余等化子的计算公式，并说明它们如何表达自然变换、迹状构造和 co-Yoneda 引理。

## 依赖前置知识

需要函子范畴、极限、余极限、Yoneda 引理和幺半范畴。

## 11.1 Dinatural 变换

**定义 11.1.** 设 $H:\mathcal C^{\operatorname{op}}\times\mathcal C\to\mathcal D$。从对象 $E\in\mathcal D$ 到 $H$ 的 dinatural 变换是一族态射

$$
\omega_C:E\to H(C,C)
$$

使得对任意 $f:C\to C'$，下列两个态射相等：

$$
E\xrightarrow{\omega_C}H(C,C)\xrightarrow{H(C,f)}H(C,C')
$$

和

$$
E\xrightarrow{\omega_{C'}}H(C',C')\xrightarrow{H(f,C')}H(C,C').
$$

## 11.2 end

**定义 11.2.** $H$ 的 end 是对象

$$
\int_C H(C,C)
$$

连同 dinatural 变换

$$
\pi_C:\int_C H(C,C)\to H(C,C)
$$

其对任意 dinatural 变换 $E\to H$ 是终的。

**命题 11.3.** 若 $\mathcal D$ 有相应积和等化子，则

$$
\int_C H(C,C)
$$

可由等化子计算：

$$
\int_C H(C,C)\to
\prod_C H(C,C)
\rightrightarrows
\prod_{f:C\to C'}H(C,C').
$$

两箭头分别由 $H(C,f)$ 与 $H(f,C')$ 给出。

**证明.** 一个态射 $E\to\prod_C H(C,C)$ 等价于族 $\omega_C:E\to H(C,C)$。它落入等化子当且仅当对每个 $f:C\to C'$，两条到 $H(C,C')$ 的复合相等，这正是 dinatural 条件。由等化子的泛性质得到 end 的终性。$\square$

## 11.3 coend

**定义 11.4.** $H$ 的 coend 是对象

$$
\int^C H(C,C)
$$

连同 dinatural 余变换

$$
\iota_C:H(C,C)\to\int^C H(C,C)
$$

其对任意 dinatural 余变换 $H\to E$ 是始的。

**命题 11.5.** 若 $\mathcal D$ 有相应余积和余等化子，则 coend 可由余等化子计算：

$$
\coprod_{f:C\to C'}H(C',C)
\rightrightarrows
\coprod_C H(C,C)
\to
\int^C H(C,C).
$$

两箭头分别由 $H(f,C)$ 与 $H(C',f)$ 给出。

**证明.** 对命题 11.3 对偶化。$\square$

## 11.4 自然变换的 end 公式

**命题 11.6.** 若 $F,G:\mathcal C\to\mathcal D$ 且 $\mathcal D$ 局部小，则

$$
\operatorname{Nat}(F,G)\cong
\int_{C\in\mathcal C}\mathcal D(F C,G C).
$$

**证明.** 右边的 end 是集合

$$
\prod_C\mathcal D(F C,G C)
$$

中满足对每个 $f:C\to C'$，

$$
G(f)\circ\alpha_C=\alpha_{C'}\circ F(f)
$$

的族。该条件正是自然变换条件。$\square$

## 11.5 co-Yoneda

**定理 11.7（co-Yoneda）.** 对预层 $P:\mathcal C^{\operatorname{op}}\to\mathbf{Set}$，存在自然同构

$$
\int^{C\in\mathcal C}P(C)\times\mathcal C(-,C)\cong P.
$$

**证明.** 在对象 $A$ 处，左边为

$$
\int^C P(C)\times\mathcal C(A,C).
$$

由 coend 的余等化子描述，它是所有三元组 $(C,x,f:A\to C)$ 的集合按关系

$$
(C,P(u)(y),f)\sim(D,y,u\circ f)
$$

商掉。映射到 $P(A)$ 定义为

$$
(C,x,f)\longmapsto P(f)(x).
$$

若 $(C,P(u)(y),f)\sim(D,y,u f)$，则

$$
P(f)(P(u)(y))=P(u f)(y),
$$

因为 $P$ 反变。所以映射与 coend 关系相容。

设 $\Psi_A$ 为上述映射，$\Gamma_A:P(A)\to\int^C P(C)\times\mathcal C(A,C)$ 为

$$
a\mapsto[A,a,\operatorname{id}_A].
$$

则

$$
\Psi_A\Gamma_A(a)=P(\operatorname{id}_A)(a)=a.
$$

另一方面，对任意代表 $(C,x,f:A\to C)$，在 coend 关系中取 $u=f$、$D=C$、$y=x$，得到

$$
[C,x,f]=[A,P(f)(x),\operatorname{id}_A]=\Gamma_A(\Psi_A[C,x,f]).
$$

故 $\Gamma_A\Psi_A=\operatorname{id}$。因此 $\Psi_A$ 是双射。

若 $v:A'\to A$，左边预层的限制把 $[C,x,f]$ 送到 $[C,x,fv]$，而右边 $P(v)$ 把 $P(f)(x)$ 送到 $P(v)P(f)(x)=P(fv)(x)$。故双射对 $A$ 自然。$\square$

## 11.6 集合值 coend 的商公式与 Yoneda 技巧

**命题 11.8（集合值 coend 的商公式）.** 设 $H:\mathcal C^{\operatorname{op}}\times\mathcal C\to\mathbf{Set}$。若相应 coend 存在，则

$$
\int^C H(C,C)
$$

可描述为集合

$$
\coprod_C H(C,C)
$$

关于下列关系生成的等价关系之商：对每个 $f:C\to C'$ 与每个 $x\in H(C',C)$，

$$
H(f,C)(x)\sim H(C',f)(x).
$$

**证明.** 命题 11.5 给出余等化子

$$
\coprod_{f:C\to C'}H(C',C)
\rightrightarrows
\coprod_C H(C,C)
\to
\int^C H(C,C).
$$

在 $\mathbf{Set}$ 中，余等化子正是把目标集合按两箭头像相等所生成的等价关系商掉。代入两箭头的定义即得。$\square$

**命题 11.9（coend 形式的 Yoneda）.** 对协变函子 $F:\mathcal C\to\mathbf{Set}$ 和对象 $A\in\mathcal C$，有自然同构

$$
\int^{C\in\mathcal C}\mathcal C(C,A)\times F(C)\cong F(A).
$$

对反变函子 $P:\mathcal C^{\operatorname{op}}\to\mathbf{Set}$，有自然同构

$$
\int^{C\in\mathcal C}P(C)\times\mathcal C(A,C)\cong P(A).
$$

**证明.** 第一式与定理 11.7 对偶；也可直接把代表 $(C,f:C\to A,x\in F(C))$ 送到 $F(f)(x)$。第二式正是定理 11.7 在对象 $A$ 处的公式。$\square$

**命题 11.10（end 形式的 Yoneda）.** 对协变函子 $F:\mathcal C\to\mathbf{Set}$ 有自然同构

$$
\int_{C\in\mathcal C}\mathbf{Set}(\mathcal C(A,C),F(C))\cong F(A).
$$

对反变函子 $P:\mathcal C^{\operatorname{op}}\to\mathbf{Set}$ 有自然同构

$$
\int_{C\in\mathcal C}\mathbf{Set}(\mathcal C(C,A),P(C))\cong P(A).
$$

**证明.** 第一式左边由命题 11.6 识别为

$$
\operatorname{Nat}(\mathcal C(A,-),F),
$$

再由协变 Yoneda 引理同构于 $F(A)$。第二式同理，是反变 Yoneda 引理。$\square$

**例子 11.11（幺半群范畴的 coend 关系）.** 设 $M$ 为幺半群，视为一对象范畴。函子 $H:M^{\operatorname{op}}\times M\to\mathbf{Set}$ 等价于集合 $S=H(*,*)$ 上相容的左右 $M$-作用。其 coend 是 $S$ 对关系

$$
x\cdot m\sim m\cdot x
$$

生成的商集合。这里左右作用分别来自 $H$ 的两个变量；该例把 coend 理解为“把左右作用强制相等”的商。

**例子 11.12（存在性边界）.** coend 的定义不是存在性定理。令 $\mathcal C$ 为无限离散小范畴，令 $H(C,C)=\{0,1\}$，并把目标范畴取为有限集合范畴 $\mathbf{FinSet}$。若 end 存在，它应为无限积 $\prod_C\{0,1\}$，不是有限集合；若 coend 存在，它应为无限余积 $\coprod_C\{0,1\}$，也不是有限集合。因此在 $\mathbf{FinSet}$ 中这些 end/coend 不存在。

## 11.7 Fubini 定理

**外部输入定理 11.13（Fubini for ends/coends）.** 在存在性条件满足时，迭代 end/coend 可交换：

$$
\int_x\int_y H(x,y,x,y)
\cong
\int_{x,y}H(x,y,x,y),
$$

coend 版本同理。混合 end/coend 需要额外条件，不能无条件交换。

## 11.8 本章小结

end 是“满足自然性条件的积”，coend 是“按自然性关系商掉的余积”。在集合值情形，coend 可明确写为商集合；自然变换集合、co-Yoneda、Day 卷积和富范畴中的 Hom 对象都可用 end/coend 统一表达。Fubini 定理允许在存在性条件下重排同类 end 或 coend，但混合 end/coend 交换必须另证。

## 练习

**练习 11.1.** 展开命题 11.5 的两个余等化子箭头。

**练习 11.2.** 对 $\mathcal C$ 为离散小范畴的情形，计算 end 和 coend。

**练习 11.3.** 用命题 11.6 重新证明自然变换的纵向复合逐点定义。

**练习 11.4.** 完成 co-Yoneda 定理中逆映射良定义的证明。

**练习 11.5.** 使用 co-Yoneda 证明预层密度定理。

**练习 11.6.** 证明命题 11.8 中的商关系确实给出 coend 的泛性质。

**练习 11.7.** 设 $\mathcal C$ 为无限离散范畴。计算常值二元集合图形在 $\mathbf{Set}$ 中的 end 和 coend，并说明为什么它们不属于 $\mathbf{FinSet}$。

**练习 11.8.** 用命题 11.6 和 Yoneda 引理证明命题 11.10 的反变版本。

**练习 11.9.** 对协变函子 $F:\mathcal C\to\mathbf{Set}$，直接构造同构 $\int^C\mathcal C(C,A)\times F(C)\cong F(A)$ 的逆映射。

**练习 11.10.** 对一对象幺半群范畴，写出命题 11.8 中两个余等化子箭头对应的左右作用。

**练习 11.11.** 当 $\mathcal C,\mathcal D$ 均为有限离散范畴时，直接验证同类 coend 的 Fubini 同构。
