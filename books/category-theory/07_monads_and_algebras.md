# 第七章：单子、余单子与代数

## 本章目标

本章定义单子、单子代数、Kleisli 范畴和 Eilenberg-Moore 范畴，并说明伴随如何产生单子。Beck 单子性定理作为外部输入定理记录。

## 依赖前置知识

需要伴随函子、自然变换和函子复合。

## 7.1 单子的定义

**定义 7.1.** 范畴 $\mathcal C$ 上的单子（monad）是三元组 $(T,\eta,\mu)$，其中

$$
T:\mathcal C\to\mathcal C
$$

是函子，

$$
\eta:\operatorname{id}_{\mathcal C}\Rightarrow T,\qquad
\mu:T^2\Rightarrow T
$$

是自然变换，满足单位律和结合律：

$$
\mu\circ T\eta=\operatorname{id}_T,\qquad
\mu\circ \eta T=\operatorname{id}_T,
$$

以及

$$
\mu\circ T\mu=\mu\circ\mu T:T^3\Rightarrow T.
$$

**例子 7.2.** 在 $\mathbf{Set}$ 上，自由幺半群函子 $T(S)$ 取 $S$ 上有限字。单位把元素送到长度一的字，乘法 $\mu:T^2(S)\to T(S)$ 把“字的字”拼接成一个字。单位律和结合律分别来自空层拼接无效和拼接结合律。

## 7.2 伴随产生单子

**命题 7.3.** 若 $F:\mathcal C\rightleftarrows\mathcal D:G$ 且 $F\dashv G$，单位为 $\eta$、余单位为 $\varepsilon$，则

$$
T=GF:\mathcal C\to\mathcal C
$$

带有单位 $\eta:\operatorname{id}_{\mathcal C}\to GF$ 和乘法

$$
\mu=G\varepsilon F:GFGF\to GF
$$

构成单子。

**证明.** 单位律为

$$
G\varepsilon F\circ GF\eta=\operatorname{id}_{GF},
\qquad
G\varepsilon F\circ\eta GF=\operatorname{id}_{GF},
$$

它们分别是伴随三角恒等式在 $F$ 或 $G$ 后的函子像。结合律要求

$$
G\varepsilon F\circ GF(G\varepsilon F)
=
G\varepsilon F\circ G\varepsilon FGF.
$$

这由 $\varepsilon:FG\Rightarrow\operatorname{id}_{\mathcal D}$ 的自然性应用于态射 $\varepsilon_{F X}:FGF X\to F X$ 得到。$\square$

## 7.3 单子代数

**定义 7.4.** 设 $(T,\eta,\mu)$ 是 $\mathcal C$ 上的单子。一个 $T$-代数是对象 $A\in\mathcal C$ 和态射

$$
a:T A\to A
$$

满足

$$
a\circ\eta_A=\operatorname{id}_A,
\qquad
a\circ T(a)=a\circ\mu_A.
$$

若 $(A,a)$ 与 $(B,b)$ 是 $T$-代数，代数同态 $f:(A,a)\to(B,b)$ 是态射 $f:A\to B$，满足

$$
f\circ a=b\circ T(f).
$$

**命题 7.5.** $T$-代数和代数同态构成范畴，记作 $\mathcal C^T$。

**证明.** 恒等态射满足代数同态条件，因为

$$
\operatorname{id}_A\circ a=a=a\circ T(\operatorname{id}_A).
$$

若 $f:(A,a)\to(B,b)$ 和 $g:(B,b)\to(C,c)$ 是代数同态，则

$$
(g f)\circ a
=g\circ(f\circ a)
=g\circ b\circ T(f)
=c\circ T(g)\circ T(f)
=c\circ T(g f).
$$

故复合仍为代数同态。结合律和单位律来自 $\mathcal C$。$\square$

## 7.4 Kleisli 范畴

**定义 7.6.** 单子 $(T,\eta,\mu)$ 的 Kleisli 范畴 $\mathcal C_T$ 定义为：

- 对象与 $\mathcal C$ 相同。
- Hom 集为
  $$
  \mathcal C_T(X,Y)=\mathcal C(X,T Y).
  $$
- 态射 $f:X\to T Y$ 与 $g:Y\to T Z$ 的 Kleisli 复合为
  $$
  X\xrightarrow{f}T Y\xrightarrow{Tg}T^2 Z\xrightarrow{\mu_Z}T Z.
  $$
- $X$ 的恒等态射为 $\eta_X:X\to T X$。

**命题 7.7.** $\mathcal C_T$ 是范畴。

**证明.** 设 $f:X\to TY$ 为 Kleisli 态射。右单位复合为

$$
\mu_Y\circ T(\eta_Y)\circ f=f
$$

由单子单位律 $\mu\circ T\eta=\operatorname{id}_T$。左单位复合为

$$
\mu_Y\circ T(f)\circ\eta_X.
$$

由 $\eta$ 的自然性，$T(f)\eta_X=\eta_{TY}f$，再由另一单位律 $\mu_Y\eta_{TY}=\operatorname{id}_{TY}$，得到左单位复合等于 $f$。

现在设

$$
f:X\to TY,\qquad g:Y\to TZ,\qquad h:Z\to TW.
$$

先复合 $f$ 与 $g$，再与 $h$，得到

$$
\mu_W\circ T(h)\circ \mu_Z\circ T(g)\circ f.
$$

由 $\mu$ 的自然性应用于 $h:Z\to TW$，

$$
T(h)\circ \mu_Z=\mu_{TW}\circ T^2(h).
$$

所以该复合等于

$$
\mu_W\circ\mu_{TW}\circ T^2(h)\circ T(g)\circ f.
$$

另一种括号先复合 $g$ 与 $h$，再与 $f$，得到

$$
\mu_W\circ T(\mu_W\circ T(h)\circ g)\circ f
=\mu_W\circ T\mu_W\circ T^2(h)\circ T(g)\circ f.
$$

单子结合律在对象 $W$ 处给出

$$
\mu_W\circ\mu_{TW}=\mu_W\circ T\mu_W.
$$

故两种复合相等。$\mathcal C_T$ 的结合律和单位律成立。$\square$

## 7.5 单子性

**定义 7.8.** 对伴随 $F\dashv G$ 产生的单子 $T=GF$，比较函子

$$
K:\mathcal D\to\mathcal C^T
$$

把 $Y\in\mathcal D$ 送到 $T$-代数

$$
(G Y,\, G\varepsilon_Y:GFGY\to GY).
$$

若 $K$ 是范畴等价，则称右伴随 $G$ 是单子的（monadic）。

**外部输入定理 7.9（Beck 单子性定理）.** 设 $G:\mathcal D\to\mathcal C$ 有左伴随。则在适当完备性条件下，$G$ 单子当且仅当 $G$ 保守并且保持且反映某类 $G$-split coequalizers。

本书在本章不证明该定理；后续讨论可表现范畴和代数理论时会使用其精确版本。来源见 `SOURCES.md` 中 Mac Lane、Borceux 与 Riehl 的单子章节。

## 7.6 本章小结

单子把“自由-遗忘”伴随中的代数结构压缩到一个自函子 $T$ 及其单位、乘法中。Eilenberg-Moore 范畴记录真实代数，Kleisli 范畴记录带效应的态射。Beck 定理说明何时一个范畴可以完全由某个单子的代数恢复。

## 练习

**练习 7.1.** 验证自由幺半群单子的单位律和结合律。

**练习 7.2.** 证明自由阿贝尔群伴随产生的单子，其代数范畴等价于 $\mathbf{Ab}$。

**练习 7.3.** 写出 powerset 单子 $\mathcal P$ 在 $\mathbf{Set}$ 上的单位和乘法。

**练习 7.4.** 完成命题 7.7 的结合律证明。

**练习 7.5.** 对偶定义余单子（comonad）及其余代数。
