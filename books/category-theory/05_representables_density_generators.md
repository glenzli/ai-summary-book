# 第五章：可表函子、密度与生成元

## 本章目标

本章把 Yoneda 引理发展为“可表对象生成预层范畴”的密度思想，并引入生成元、投射对象和小对象检测态射的语言。

## 依赖前置知识

需要 Yoneda 引理、极限、余极限和伴随。

## 5.1 元素范畴

**定义 5.1.** 设 $P:\mathcal C^{\operatorname{op}}\to\mathbf{Set}$ 为预层。元素范畴（category of elements）$\int_{\mathcal C}P$ 定义如下：

- 对象是二元组 $(C,x)$，其中 $C\in\mathcal C$ 且 $x\in P(C)$。
- 从 $(C,x)$ 到 $(D,y)$ 的态射是态射 $f:C\to D$，满足
  $$
  P(f)(y)=x.
  $$

复合由 $\mathcal C$ 中复合给出。

**命题 5.2.** $\int_{\mathcal C}P$ 是范畴。

**证明.** 若 $f:(C,x)\to(D,y)$ 与 $g:(D,y)\to(E,z)$，则 $P(f)(y)=x$ 且 $P(g)(z)=y$。由反变函子性，

$$
P(g\circ f)(z)=P(f)(P(g)(z))=P(f)(y)=x.
$$

故 $g\circ f$ 是 $(C,x)\to(E,z)$ 的态射。恒等态射条件由 $P(\operatorname{id}_C)=\operatorname{id}_{P(C)}$ 得到。$\square$

## 5.2 预层是可表预层的余极限

**定理 5.3（预层密度定理）.** 对任意预层 $P:\mathcal C^{\operatorname{op}}\to\mathbf{Set}$，存在自然同构

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

## 5.5 本章小结

可表预层不只是例子，而是预层范畴的基本构件。任意预层都是可表预层按其元素范畴组织的余极限；可表预层族检测自然变换，并且相对于逐点满射投射。后续 Kan 延拓、sheaf 化和可表现范畴都会反复使用这一思想。

## 练习

**练习 5.1.** 对给定预层 $P$，写出 $\int_{\mathcal C}P\to\mathcal C$ 的投影函子。

**练习 5.2.** 计算可表预层 $yA$ 的元素范畴，并说明它与 slice 范畴 $\mathcal C/A$ 的关系。

**练习 5.3.** 完成定理 5.3 中逐对象单射性的证明。

**练习 5.4.** 证明 $\mathbf{Set}$ 中单点集合是生成元。

**练习 5.5.** 说明为什么 $\mathbf{Grp}$ 中整数群 $\mathbb Z$ 是生成元。
