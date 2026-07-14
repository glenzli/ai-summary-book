# 第五章：可表函子、密度与生成元

Yoneda 引理说明可表预层完全记住原范畴的态射；密度定理进一步说明，一般预层可以由这些可表对象按其元素范畴取余极限重建。于是“对象由哪些基本对象生成”从类比变成一个可证明的余极限公式，也自然引出生成元、投射对象和用小对象族检测同构的条件。本章从 co-Yoneda 形式推导密度，再比较生成性、强生成性和投射性的不同量词。

为使元素范畴和密度余极限保持在同一层，本章固定 $\mathcal C$ 为 $\mathcal U$-小范畴，预层取值于 $\mathbf{Set}_{\mathcal U}$。所用的 Yoneda、极限、余极限和伴随均沿用前四章约定。

## 5.1 元素范畴

**定义 5.1.** 设
$P:\mathcal C^{\operatorname{op}}\to\mathbf{Set}_{\mathcal U}$ 为预层。元素范畴（category of elements）$\int_{\mathcal C}P$ 定义如下：

- 对象是二元组 $(C,x)$，其中 $C\in\mathcal C$ 且 $x\in P(C)$。
- 从 $(C,x)$ 到 $(D,y)$ 的态射是态射 $f:C\to D$，满足
  $$
  P(f)(y)=x.
  $$

复合由 $\mathcal C$ 中复合给出。

其对象集是 $\mathcal U$-小余并
$\coprod_{C\in\operatorname{Ob}(\mathcal C)}P(C)$，态射集是
$\operatorname{Mor}(\mathcal C)$ 与两端元素数据的一个子集；故
$\int_{\mathcal C}P$ 为 $\mathcal U$-小范畴。

**命题 5.2.** $\int_{\mathcal C}P$ 是范畴。

**证明.** 若 $f:(C,x)\to(D,y)$ 与 $g:(D,y)\to(E,z)$，则 $P(f)(y)=x$ 且 $P(g)(z)=y$。由反变函子性，

$$
P(g\circ f)(z)=P(f)(P(g)(z))=P(f)(y)=x.
$$

故 $g\circ f$ 是 $(C,x)\to(E,z)$ 的态射。恒等态射条件由 $P(\operatorname{id}_C)=\operatorname{id}_{P(C)}$ 得到。$\square$

## 5.2 预层是可表预层的余极限

**定理 5.3（预层密度定理）.** 对任意预层
$P:\mathcal C^{\operatorname{op}}\to\mathbf{Set}_{\mathcal U}$，存在预层范畴中的自然同构

$$
P\cong\operatorname{colim}_{(C,x)\in\int_{\mathcal C}P} yC
$$

其中 $yC=\mathcal C(-,C)$。

**证明.** 元素 $x\in P(C)$ 由 Yoneda 引理对应自然变换

$$
yC\to P.
$$

这些自然变换组成从图形 $(C,x)\mapsto yC$ 到 $P$ 的余锥。它诱导态射

$$
\operatorname{colim}_{(C,x)}yC\to P.
$$

逐对象在 $A\in\mathcal C$ 处计算。由于预层范畴的余极限逐点计算，左边在 $A$ 处是集合

$$
\operatorname{colim}_{(C,x)}\mathcal C(A,C).
$$

记这个映射为 $\Theta_A$。

先证满射。给定 $a\in P(A)$，三元组 $(A,a,\operatorname{id}_A)$ 给出左边元素，且

$$
\Theta_A(A,a,\operatorname{id}_A)=P(\operatorname{id}_A)(a)=a.
$$

再证单射。设

$$
\Theta_A(C,x,f)=\Theta_A(D,y,g),
$$

即

$$
P(f)(x)=P(g)(y)=a\in P(A).
$$

在元素范畴中，$(A,a)$ 是对象，且 $f:A\to C$ 给出态射

$$
(A,a)\to(C,x)
$$

因为 $P(f)(x)=a$；同理 $g:A\to D$ 给出态射 $(A,a)\to(D,y)$。在余极限集合

$$
\operatorname{colim}_{(E,z)}\mathcal C(A,E)
$$

中，态射 $(A,a)\to(C,x)$ 把 $\operatorname{id}_A\in\mathcal C(A,A)$ 与 $f\in\mathcal C(A,C)$ 识别；态射 $(A,a)\to(D,y)$ 把 $\operatorname{id}_A$ 与 $g$ 识别。因此 $(C,x,f)$ 与 $(D,y,g)$ 代表同一等价类。故 $\Theta_A$ 为单射。

对态射 $u:A'\to A$，左边的限制把 $(C,x,f)$ 送到 $(C,x,f u)$；右边限制为 $P(u)$。于是

$$
P(u)\Theta_A(C,x,f)=P(u)P(f)(x)=P(fu)(x)=\Theta_{A'}(C,x,fu),
$$

所以 $\Theta$ 自然。故诱导态射是预层同构。$\square$

**注 5.4.** 该定理说明 Yoneda 嵌入 $y:\mathcal C\to\widehat{\mathcal C}$ 是稠密的：预层范畴中的每个对象都是可表预层的典范余极限。

## 5.3 生成元与检测态射

**定义 5.5.** 设 $\mathcal C$ 局部小。对象族 $\mathcal G\subseteq\operatorname{Ob}(\mathcal C)$ 称为生成族（generating family），若对任意不同态射 $f,g:X\rightrightarrows Y$，存在 $G\in\mathcal G$ 和态射 $u:G\to X$，使得

$$
f u\neq g u.
$$

单个对象 $G$ 若组成生成族，则称为生成元（generator）。

**命题 5.6.** 在预层范畴 $\widehat{\mathcal C}$ 中，可表预层族 $\{yC\}_{C\in\mathcal C}$ 是生成族。

**证明.** 设 $\alpha,\beta:P\rightrightarrows Q$ 是不同自然变换。则存在 $C\in\mathcal C$ 和 $x\in P(C)$ 使得

$$
\alpha_C(x)\neq\beta_C(x).
$$

由 Yoneda 引理，$x$ 对应自然变换 $\bar x:yC\to P$。于是

$$
(\alpha\bar x)_C(\operatorname{id}_C)=\alpha_C(x)\neq\beta_C(x)=(\beta\bar x)_C(\operatorname{id}_C),
$$

所以 $\alpha\bar x\neq\beta\bar x$。$\square$

## 5.4 投射对象

**定义 5.7.** 在有满射概念的范畴中，若对象 $P$ 满足：对任意满射 $e:X\to Y$ 和任意态射 $f:P\to Y$，存在提升 $\tilde f:P\to X$ 使 $e\tilde f=f$，则称 $P$ 为投射对象（projective object）。

在一般范畴中，该定义需相对于指定的正合结构或满射类解释。本书在阿贝尔范畴章节会给出更精确版本。

**命题 5.8.** 在预层范畴 $\widehat{\mathcal C}$ 中，每个可表预层 $yC$ 相对于逐点满射是投射的。

**证明.** 设 $\alpha:P\to Q$ 是逐点满射，且给定态射 $f:yC\to Q$。由 Yoneda 引理，$f$ 对应元素 $q\in Q(C)$。由于 $\alpha_C:P(C)\to Q(C)$ 满，存在 $p\in P(C)$ 使 $\alpha_C(p)=q$。元素 $p$ 对应 $\tilde f:yC\to P$，并且 $\alpha\tilde f=f$，因为二者在 Yoneda 对应下都给出 $q$。$\square$

## 5.5 生成与稠密的边界条件

**命题 5.9.** 对象族 $\mathcal G$ 是 $\mathcal C$ 的生成族，当且仅当函子

$$
\prod_{G\in\mathcal G}\mathcal C(G,-):\mathcal C\to\prod_{G\in\mathcal G}\mathbf{Set}
$$

忠实。

**证明.** 该乘积函子把态射 $f:X\to Y$ 送到族

$$
(fu:G\to Y)_{G\in\mathcal G,\ u:G\to X}.
$$

它忠实当且仅当任意平行态射 $f,g:X\rightrightarrows Y$ 若对所有 $G$ 和所有 $u:G\to X$ 都有 $fu=gu$，则 $f=g$。这正是生成族定义的否定形式。$\square$

**例子 5.10.** 在 $\mathbf{Set}$ 中，单点集 $1$ 是生成元，因为函数 $f,g:X\rightrightarrows Y$ 若不同，则存在 $x\in X$ 使 $f(x)\ne g(x)$；该元素等价于态射 $1\to X$。在 $\mathbf{Grp}$ 中，整数群 $\mathbb Z$ 是生成元，因为群同态 $\mathbb Z\to G$ 等价于选择 $G$ 的一个元素。

**例子 5.11（非生成元）.** $\mathbf{Grp}$ 中平凡群 $1$ 不是生成元。对任意群 $G$，从 $1$ 到 $G$ 只有一个群同态。因此若 $f,g:G\rightrightarrows H$ 是不同群同态，预复合所有 $1\to G$ 的态射仍无法区分它们。

**例子 5.12（稠密不等于本质满）.** Yoneda 嵌入

$$
y:\mathcal C\to\widehat{\mathcal C}
$$

是稠密的，即每个预层是可表预层的典范余极限。但除非每个预层都可表，$y$ 不是本质满。例如当 $\mathcal C=*$ 时，$\widehat{\mathcal C}\simeq\mathbf{Set}$，Yoneda 嵌入只命中单点集；它是稠密的，因为每个集合是若干单点集的余积，但它不是本质满。

## 5.6 可表对象怎样生成预层

可表预层不只是例子，而是预层范畴的基本构件。任意预层都是可表预层按其元素范畴组织的余极限；可表预层族检测自然变换，并且相对于逐点满射投射。后续 Kan 延拓、sheaf 化和可表现范畴都会反复使用这一思想。

## 练习

**练习 5.1.** 对给定预层 $P$，写出 $\int_{\mathcal C}P\to\mathcal C$ 的投影函子。

**练习 5.2.** 计算可表预层 $yA$ 的元素范畴，并说明它与 slice 范畴 $\mathcal C/A$ 的关系。

**练习 5.3.** 完成定理 5.3 中逐对象单射性的证明。

**练习 5.4.** 证明 $\mathbf{Set}$ 中单点集合是生成元。

**练习 5.5.** 说明为什么 $\mathbf{Grp}$ 中整数群 $\mathbb Z$ 是生成元。

**练习 5.6.** 证明命题 5.9 在单个生成元 $G$ 的情形下等价于 $\mathcal C(G,-)$ 忠实。

**练习 5.7.** 证明 $\mathbf{Ab}$ 中 $\mathbb Z$ 是投射生成元。

**练习 5.8.** 对 $\mathcal C=*$，把预层密度定理具体写成“每个集合是单点集的余积”。
