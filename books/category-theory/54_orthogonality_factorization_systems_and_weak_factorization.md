# 第五十四章：正交性、因子化系统与弱因子化系统

## 本章目标

本章系统介绍范畴论内部的正交性与因子化系统。因子化系统把每个态射分解为两类态射的复合，并用提升唯一性刻画这两类。弱因子化系统放松唯一性，成为模型范畴、小对象论证和同伦代数的基础。本章只处理范畴论形式主义，不展开具体模型结构的外部构造。

## 依赖前置知识

需要态射类、交换方块、提升性质、极限、余极限、反射子范畴、局部化和模型范畴基础定义。

## 54.1 正交性

**定义 54.1.** 设 $\mathcal C$ 为范畴，$f:A\to B$ 与 $g:X\to Y$ 为态射。称 $f$ 左正交于 $g$，记作 $f\perp g$，若对每个交换方块

$$
\begin{array}{ccc}
A&\to&X\\
\downarrow f&&\downarrow g\\
B&\to&Y
\end{array}
$$

存在唯一对角填充 $B\to X$ 使两个三角交换。

**定义 54.2.** 对态射类 $\mathcal S$，定义

$$
{}^\perp\mathcal S=\{f\mid f\perp s,\ \forall s\in\mathcal S\},
\qquad
\mathcal S^\perp=\{g\mid s\perp g,\ \forall s\in\mathcal S\}.
$$

**命题 54.3.** 若 $\mathcal S\subseteq\mathcal T$，则

$$
{}^\perp\mathcal T\subseteq{}^\perp\mathcal S,\qquad
\mathcal T^\perp\subseteq\mathcal S^\perp.
$$

**证明.** 若 $f\in{}^\perp\mathcal T$，则 $f$ 正交于 $\mathcal T$ 中所有态射。由于 $\mathcal S\subseteq\mathcal T$，它特别正交于 $\mathcal S$ 中所有态射，故 $f\in{}^\perp\mathcal S$。右正交的证明相同。$\square$

## 54.2 正交因子化系统

**定义 54.4.** 范畴 $\mathcal C$ 上的正交因子化系统是两类态射 $(\mathcal E,\mathcal M)$，满足：

1. 每个态射 $f$ 可分解为
   $$
   f=m e,\qquad e\in\mathcal E,\ m\in\mathcal M.
   $$
2. $\mathcal E={}^\perp\mathcal M$ 且 $\mathcal M=\mathcal E^\perp$。

**命题 54.5.** 正交因子化在唯一同构意义下唯一。

**证明.** 设 $f=me=m'e'$ 为两个分解，其中 $e,e'\in\mathcal E$，$m,m'\in\mathcal M$。考虑交换方块

$$
\begin{array}{ccc}
\operatorname{dom}f&\xrightarrow{e'}&\operatorname{dom}m'\\
\downarrow e&&\downarrow m'\\
\operatorname{dom}m&\xrightarrow{m}&\operatorname{cod}f .
\end{array}
$$

由 $e\perp m'$，存在唯一 $u:\operatorname{dom}m\to\operatorname{dom}m'$，使 $ue=e'$ 且 $m'u=m$。对称地由 $e'\perp m$ 得 $v:\operatorname{dom}m'\to\operatorname{dom}m$。复合 $vu$ 与 $\operatorname{id}$ 都填充同一正交方块，唯一性给出 $vu=\operatorname{id}$；同理 $uv=\operatorname{id}$。$\square$

## 54.3 例子：epi-mono 分解

**命题 54.6.** 在 $\mathbf{Set}$ 中，surjections 与 injections 构成正交因子化系统。

**证明.** 任意函数 $f:X\to Y$ 分解为

$$
X\twoheadrightarrow \operatorname{im}(f)\hookrightarrow Y.
$$

设 $e:A\twoheadrightarrow B$ 为满射，$m:X\hookrightarrow Y$ 为单射。给交换方块，欲定义填充 $h:B\to X$。对 $b\in B$，取 $a\in A$ 使 $e(a)=b$，令 $h(b)$ 为上边映射 $A\to X$ 在 $a$ 的值。若 $a,a'$ 有同一像，则它们在 $Y$ 中像相同；因 $m$ 单射，二者在 $X$ 中像相同，故 $h$ 良定义。满射性保证存在，单射性保证唯一。$\square$

## 54.4 反射局部化与正交对象

**定义 54.7.** 对态射类 $\mathcal S$，对象 $X$ 称为 $\mathcal S$-局部，若对每个 $s:A\to B$，诱导映射

$$
\mathcal C(B,X)\to\mathcal C(A,X)
$$

为双射。

**命题 54.8.** $X$ 为 $\mathcal S$-局部当且仅当每个 $s\in\mathcal S$ 正交于终态射 $X\to1$，假设 $\mathcal C$ 有终对象。

**证明.** 给出方块

$$
\begin{array}{ccc}
A&\to&X\\
\downarrow s&&\downarrow\\
B&\to&1
\end{array}
$$

等价于给映射 $A\to X$。对角填充 $B\to X$ 正是把该映射沿 $s$ 延拓。存在唯一填充等价于 $\mathcal C(B,X)\to\mathcal C(A,X)$ 为双射。$\square$

## 54.5 弱因子化系统

**定义 54.9.** 弱因子化系统是两类态射 $(\mathcal L,\mathcal R)$，满足：

1. 每个态射可分解为 $r l$，其中 $l\in\mathcal L$、$r\in\mathcal R$。
2. $\mathcal L$ 由对 $\mathcal R$ 有左提升性质的态射组成。
3. $\mathcal R$ 由对 $\mathcal L$ 有右提升性质的态射组成。

提升只要求存在，不要求唯一。

**命题 54.10.** 每个正交因子化系统给出弱因子化系统。

**证明.** 正交因子化系统已经给出分解。若 $e\in\mathcal E$、$m\in\mathcal M$，则每个交换方块有唯一填充，特别有填充。因此 $\mathcal E$ 对 $\mathcal M$ 有左提升性质，$\mathcal M$ 对 $\mathcal E$ 有右提升性质。唯一性被遗忘后得到弱因子化系统。$\square$

## 54.6 闭包性质

**命题 54.11.** 在弱因子化系统 $(\mathcal L,\mathcal R)$ 中，$\mathcal L$ 与 $\mathcal R$ 都对 retracts 封闭。

**证明.** 设 $f$ 是 $g\in\mathcal L$ 的 retract。对任意 $r\in\mathcal R$ 和以 $f$ 为左边的交换方块，利用 retract 数据把它扩张为以 $g$ 为左边的交换方块。因 $g$ 对 $r$ 有左提升性质，得到填充。再沿 retract 投回，得到原方块的填充。因此 $f\in{}^\square\mathcal R=\mathcal L$。$\mathcal R$ 的证明对偶。$\square$

**命题 54.12.** 正交因子化系统 $(\mathcal E,\mathcal M)$ 中，$\mathcal E$ 与 $\mathcal M$ 都包含所有同构，并且都对复合封闭。

**证明.** 同构与任意态射正交：给定交换方块，沿同构搬运即可得到唯一填充。因此所有同构属于 ${}^\perp\mathcal M=\mathcal E$ 且属于 $\mathcal E^\perp=\mathcal M$。

若 $e_1,e_2\in\mathcal E$，则对任意 $m\in\mathcal M$ 和以 $e_2e_1$ 为左边的方块，先用 $e_1\perp m$ 得到第一步唯一填充，再用 $e_2\perp m$ 得到第二步唯一填充；唯一性同样逐步推出。因此 $e_2e_1\perp m$，故 $e_2e_1\in\mathcal E$。$\mathcal M$ 对复合封闭的证明对偶。$\square$

## 54.7 本章小结

正交性把“唯一提升”编码为态射间关系；正交因子化系统给出唯一中间对象的分解理论；弱因子化系统放松唯一性，保留同伦论所需的提升结构。反射局部化可用正交对象刻画，模型范畴的 cofibration/fibration 结构则由弱因子化系统组织。

## 练习

**练习 54.1.** 定义 $f\perp g$。

**练习 54.2.** 证明 $\mathcal S\subseteq\mathcal T$ 时正交类反向包含。

**练习 54.3.** 定义正交因子化系统。

**练习 54.4.** 证明正交分解唯一到唯一同构。

**练习 54.5.** 证明 $\mathbf{Set}$ 中 surjection-injection 是正交因子化系统。

**练习 54.6.** 定义 $\mathcal S$-局部对象。

**练习 54.7.** 证明局部对象可由终态射正交刻画。

**练习 54.8.** 定义弱因子化系统。

**练习 54.9.** 说明正交因子化系统为何给出弱因子化系统。

**练习 54.10.** 证明弱因子化系统的两类态射对 retracts 封闭。

**练习 54.11.** 证明正交因子化系统中的两类态射包含同构并对复合封闭。
