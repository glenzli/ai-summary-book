# 第八章：幺半范畴与相干性

## 本章目标

本章定义幺半范畴、强幺半函子、幺半自然变换、代数对象和余代数对象，并说明 Mac Lane 相干性定理的作用。

## 依赖前置知识

需要范畴、函子、自然同构和积的基本语言。

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

## 8.4 代数对象

**定义 8.8.** 幺半范畴 $\mathcal C$ 中的代数对象（algebra object）是对象 $A$ 与态射

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

**定义 8.9.** 若 $\mathcal C$ 是辫幺半范畴，并有辫子 $\beta_{X,Y}:X\otimes Y\to Y\otimes X$，代数对象 $A$ 称为交换的，若

$$
m\circ\beta_{A,A}=m.
$$

**例子 8.10.** $\mathbf{Vect}_k$ 中的代数对象正是通常的结合含幺 $k$-代数；交换代数对象正是交换 $k$-代数。

## 8.5 本章小结

幺半范畴把“可张量的对象”抽象化。结合律和单位律不要求严格相等，而由相干同构控制。Mac Lane 相干性定理保证常见括号省略是安全的。代数对象则把环、代数和幺半对象统一为幺半范畴内部的结构。

## 练习

**练习 8.1.** 写出 $(\mathbf{Set},\sqcup,\varnothing)$ 的幺半结构，并判断它是否为笛卡尔幺半结构。

**练习 8.2.** 在单对象范畴中解释幺半结构等价于什么额外代数结构。

**练习 8.3.** 证明强幺半函子把代数对象送到代数对象。

**练习 8.4.** 对偶定义余代数对象。

**练习 8.5.** 写出辫幺半范畴和对称幺半范畴的六边形公理。
