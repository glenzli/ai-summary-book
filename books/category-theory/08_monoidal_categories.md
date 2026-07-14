# 第八章：幺半范畴与相干性

许多范畴带有一种“张量”运算，但它通常只在典范同构意义下满足结合律，并非逐字相等。幺半范畴把结合子、左右单位子及其五边形、三角形相干条件作为结构数据；Mac Lane 相干性定理保证由这些约束生成的合法图形自动交换。这样既能严格处理向量空间张量积，也能在任意幺半范畴中定义代数对象、余代数对象和保持张量的函子。

本章沿用范畴、函子、自然同构和积的基本语言。重点不在背诵相干图，而在区分 strict、strong、lax 与 oplax 结构，并理解代数对象的乘法为何必须与背景结合子一起写出。

## 8.1 幺半范畴

**定义 8.1.** 一个幺半范畴（monoidal category）是六元组

$$
(\mathcal C,\otimes,\mathbb 1,\alpha,\lambda,\rho)
$$

其中 $\mathcal C$ 是范畴，$\otimes:\mathcal C\times\mathcal C\to\mathcal C$ 是函子，$\mathbb 1\in\mathcal C$ 是单位对象，

$$
\alpha_{X,Y,Z}:(X\otimes Y)\otimes Z\xrightarrow{\cong}X\otimes(Y\otimes Z)
$$

是结合约束，

$$
\lambda_X:\mathbb 1\otimes X\xrightarrow{\cong}X,\qquad
\rho_X:X\otimes\mathbb 1\xrightarrow{\cong}X
$$

是左右单位约束。它们满足五边形公理和三角公理。

**定义 8.2.** 五边形公理要求从 $(((W\otimes X)\otimes Y)\otimes Z)$ 到 $W\otimes(X\otimes(Y\otimes Z))$ 的两条由 $\alpha$ 组成的典范路径相等。三角公理要求

$$
(X\otimes\mathbb 1)\otimes Y
\xrightarrow{\alpha_{X,\mathbb 1,Y}}
X\otimes(\mathbb 1\otimes Y)
\xrightarrow{\operatorname{id}_X\otimes\lambda_Y}
X\otimes Y
$$

等于

$$
(X\otimes\mathbb 1)\otimes Y
\xrightarrow{\rho_X\otimes\operatorname{id}_Y}
X\otimes Y.
$$

**例子 8.3.** $(\mathbf{Set},\times,*)$ 是幺半范畴，其中 $*$ 为单点集合。$(\mathbf{Vect}_k,\otimes_k,k)$ 是幺半范畴。任意有有限积的范畴 $(\mathcal C,\times,1)$ 由积给出笛卡尔幺半结构。

## 8.2 严格幺半范畴与相干性

**定义 8.4.** 若 $\alpha,\lambda,\rho$ 都是恒等自然变换，则幺半范畴称为严格幺半范畴。

**外部输入定理 8.5（Mac Lane 相干性定理）.** 任意幺半范畴幺半等价于某个严格幺半范畴。并且在任意幺半范畴中，由 $\alpha,\lambda,\rho$ 组成且源、目标相同的典范图都交换。

本书使用该定理来省略括号不影响结果的证明，但不会把非严格等式误写成定义上的严格相等。来源见 Mac Lane 和 Kelly。

## 8.3 幺半函子

**定义 8.6.** 设 $\mathcal C,\mathcal D$ 为幺半范畴。一个强幺半函子（strong monoidal functor）由函子 $F:\mathcal C\to\mathcal D$、同构

$$
\phi_{X,Y}:F X\otimes F Y\xrightarrow{\cong}F(X\otimes Y),
\qquad
\phi_0:\mathbb 1_{\mathcal D}\xrightarrow{\cong}F(\mathbb 1_{\mathcal C})
$$

组成，并满足与结合约束和单位约束相容的图交换。

若 $\phi_{X,Y},\phi_0$ 不要求为同构，则称为 lax monoidal functor。

**定义 8.7.** 两个强幺半函子 $F,G:\mathcal C\to\mathcal D$ 之间的幺半自然变换是自然变换 $\theta:F\Rightarrow G$，使得

$$
\theta_{X\otimes Y}\circ\phi^F_{X,Y}
=
\phi^G_{X,Y}\circ(\theta_X\otimes\theta_Y)
$$

且 $\theta_{\mathbb 1}\circ\phi^F_0=\phi^G_0$。

## 8.4 辫子、对称性与代数对象

**定义 8.8.** 辫幺半范畴（braided monoidal category）是幺半范畴 $\mathcal C$ 连同自然同构

$$
\beta_{X,Y}:X\otimes Y\xrightarrow{\cong}Y\otimes X
$$

使得两个六边形公理成立。第一个要求从 $X\otimes(Y\otimes Z)$ 到 $(Y\otimes Z)\otimes X$ 的下列复合等于 $\beta_{X,Y\otimes Z}$：

$$
\alpha_{Y,Z,X}^{-1}
\circ(\operatorname{id}_Y\otimes\beta_{X,Z})
\circ\alpha_{Y,X,Z}
\circ(\beta_{X,Y}\otimes\operatorname{id}_Z)
\circ\alpha_{X,Y,Z}^{-1}.
$$

第二个要求从 $(X\otimes Y)\otimes Z$ 到 $Z\otimes(X\otimes Y)$ 的下列复合等于 $\beta_{X\otimes Y,Z}$：

$$
\alpha_{Z,X,Y}
\circ(\beta_{X,Z}\otimes\operatorname{id}_Y)
\circ\alpha_{X,Z,Y}^{-1}
\circ(\operatorname{id}_X\otimes\beta_{Y,Z})
\circ\alpha_{X,Y,Z}.
$$

若还满足

$$
\beta_{Y,X}\circ\beta_{X,Y}=\operatorname{id}_{X\otimes Y},
$$

则称为对称幺半范畴（symmetric monoidal category）。

**定义 8.9.** 幺半范畴 $\mathcal C$ 中的代数对象（algebra object）是对象 $A$ 与态射

$$
m:A\otimes A\to A,\qquad u:\mathbb 1\to A
$$

使得结合律和单位律成立：

$$
m\circ(m\otimes\operatorname{id}_A)
=
m\circ(\operatorname{id}_A\otimes m)\circ\alpha_{A,A,A},
$$

以及

$$
m\circ(u\otimes\operatorname{id}_A)=\lambda_A,\qquad
m\circ(\operatorname{id}_A\otimes u)=\rho_A
$$

在省略必要约束同构后成立。

**定义 8.10.** 若 $\mathcal C$ 是辫幺半范畴，代数对象 $A$ 称为交换的，若

$$
m\circ\beta_{A,A}=m.
$$

**例子 8.11.** $\mathbf{Vect}_k$ 中的代数对象正是通常的结合含幺 $k$-代数；交换代数对象正是交换 $k$-代数。

## 8.5 函子性、单子与边界例子

**命题 8.12.** 松幺半函子 $F:\mathcal C\to\mathcal D$ 把 $\mathcal C$ 中的代数对象送到 $\mathcal D$ 中的代数对象。特别地，强幺半函子也有此性质。

**证明.** 设 $A$ 的乘法和单位为

$$
m:A\otimes A\to A,\qquad u:\mathbb1_{\mathcal C}\to A.
$$

令 $FA$ 的乘法为

$$
FA\otimes FA\xrightarrow{\phi_{A,A}}F(A\otimes A)\xrightarrow{F m}FA,
$$

单位为

$$
\mathbb1_{\mathcal D}\xrightarrow{\phi_0}F(\mathbb1_{\mathcal C})
\xrightarrow{F u}FA.
$$

结合律的两边都是从 $(FA\otimes FA)\otimes FA$ 到 $FA$ 的态射。把松幺半函子的结合相干图与 $A$ 的结合律图外贴，得到这两条复合相等。单位律同理，由 $\phi_0$ 与左右单位约束的相干图以及 $A$ 的单位律得到。$\square$

**例子 8.13（笛卡尔幺半结构中的代数对象）.** 在 $(\mathbf{Set},\times,*)$ 中，代数对象就是通常的幺半群。乘法 $m:A\times A\to A$ 给出二元运算，单位 $*:1\to A$ 给出单位元。结合律和单位律正是幺半群公理。交换代数对象就是交换幺半群。

**例子 8.14（余积幺半结构的边界）.** 在 $(\mathbf{Set},\sqcup,\varnothing)$ 中，每个集合 $A$ 都有典范代数对象结构：

$$
A\sqcup A\xrightarrow{\nabla}A,\qquad \varnothing\to A,
$$

其中 $\nabla$ 在两个副本上都是恒等映射。单位律强制乘法在左右两个副本上均为恒等，因此这是唯一的代数对象结构。这个例子说明：幺半范畴中的“代数对象”依赖所选张量；它不必对应集合上的二元运算。

**命题 8.15（单子作为代数对象）.** 设 $\mathcal C$ 为范畴。若函子范畴 $\operatorname{Fun}(\mathcal C,\mathcal C)$ 存在，则它在函子复合 $\circ$ 与单位函子 $\operatorname{id}_{\mathcal C}$ 下成为严格幺半范畴。该幺半范畴中的代数对象正是 $\mathcal C$ 上的单子。

**证明.** 函子复合严格结合，单位函子严格为左右单位。因此

$$
(\operatorname{Fun}(\mathcal C,\mathcal C),\circ,\operatorname{id}_{\mathcal C})
$$

是严格幺半范畴。一个代数对象是自函子 $T$ 与自然变换

$$
\mu:T\circ T\Rightarrow T,\qquad
\eta:\operatorname{id}_{\mathcal C}\Rightarrow T
$$

满足结合律和单位律。逐项展开这些图，正是定义 7.1 的单子公理。$\square$

**例子 8.16（非辫性的来源）.** 并非每个幺半范畴都有辫子。考虑 $\operatorname{Fun}(\mathbf{Set},\mathbf{Set})$ 在复合下的幺半结构。若它有辫子，则任意两个自函子 $F,G$ 都有自然同构 $F\circ G\cong G\circ F$。取常值二元集函子 $K_2$ 与平方函子 $S(X)=X\times X$，则

$$
K_2\circ S\cong K_2,\qquad
S\circ K_2\cong K_4,
$$

二者在每个集合上分别取二元集和四元集，不可能自然同构。故该幺半范畴不是辫幺半范畴。

## 8.6 相干性如何控制张量

幺半范畴把“可张量的对象”抽象化。结合律和单位律不要求严格相等，而由相干同构控制。Mac Lane 相干性定理保证常见括号省略是安全的。辫子和对称性控制交换变量的合法性。代数对象把幺半群、代数和单子统一为幺半范畴内部的结构，但其含义依赖所选张量。

## 练习

**练习 8.1.** 写出 $(\mathbf{Set},\sqcup,\varnothing)$ 的幺半结构，并判断它是否为笛卡尔幺半结构。

**练习 8.2.** 在单对象范畴中解释幺半结构等价于什么额外代数结构。

**练习 8.3.** 证明强幺半函子把代数对象送到代数对象。

**练习 8.4.** 对偶定义余代数对象。

**练习 8.5.** 写出辫幺半范畴和对称幺半范畴的六边形公理。

**练习 8.6.** 证明 $(\operatorname{Fun}(\mathcal C,\mathcal C),\circ,\operatorname{id}_{\mathcal C})$ 中的余代数对象正是 $\mathcal C$ 上的余单子。

**练习 8.7.** 证明 $(\mathbf{Set},\times,*)$ 中的代数对象与幺半群范畴等价。

**练习 8.8.** 证明 $(\mathbf{Set},\sqcup,\varnothing)$ 中每个对象有且仅有一个代数对象结构，并判断它是否交换。

**练习 8.9.** 设 $F:\mathcal C\to\mathcal D$ 为松幺半函子。逐图验证命题 8.12 中 $FA$ 的结合律。

**练习 8.10.** 用常值二元集函子和平方函子证明 $\operatorname{Fun}(\mathbf{Set},\mathbf{Set})$ 在复合下不是辫幺半范畴。
