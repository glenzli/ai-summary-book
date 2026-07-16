# $\sigma$-代数、测度与可测结构

区间长度可以加，有限集合大小可以加，概率也应该能加。但随机事件不只出现有限并交：极限事件、无穷次发生事件和路径条件都要求可列并交稳定的事件族。若事件族不对这些操作封闭，同一个随机过程的长期事件甚至可能没有概率。$\sigma$-代数正是为这种稳定性付出的最小结构成本。

## 1.1 事件族

**定义 1.1（$\sigma$-代数）。** 设 $\Omega$ 是集合。子集族 $\mathcal F\subseteq 2^\Omega$ 称为 $\sigma$-代数，若满足：

1. $\Omega\in\mathcal F$；
2. $A\in\mathcal F$ 蕴含 $A^c\in\mathcal F$；
3. 若 $A_1,A_2,\ldots\in\mathcal F$，则 $\bigcup_{n\ge 1}A_n\in\mathcal F$。

$(\Omega,\mathcal F)$ 称为可测空间，$\mathcal F$ 中元素称为事件或可测集。

**定义 1.2（生成 $\sigma$-代数）。** 若 $\mathcal A\subseteq 2^\Omega$，则 $\sigma(\mathcal A)$ 是包含 $\mathcal A$ 的所有 $\sigma$-代数的交。该交仍是 $\sigma$-代数，称为由 $\mathcal A$ 生成的 $\sigma$-代数。

**例 1.1（抛硬币两次）。** $\Omega=\{HH,HT,TH,TT\}$。若只观察第一次结果，则可观测事件族为

$$
\mathcal G=\{\varnothing,\Omega,\{HH,HT\},\{TH,TT\}\}.
$$

事件“第二次为 H”不在 $\mathcal G$ 中，因为它不能由第一次结果决定。

## 1.2 测度

**定义 1.3（测度与概率测度）。** 在可测空间 $(\Omega,\mathcal F)$ 上，函数 $\mu:\mathcal F\to[0,\infty]$ 称为测度，若 $\mu(\varnothing)=0$ 且对两两不交的 $A_n\in\mathcal F$ 有

$$
\mu\left(\bigcup_{n\ge1}A_n\right)=\sum_{n\ge1}\mu(A_n).
$$

若 $\mu(\Omega)=1$，则称 $\mu$ 为概率测度。

**定理 1.1（测度的单调性与连续性）。** 设 $\mu$ 是 $(\Omega,\mathcal F)$ 上的测度。

1. 若 $A\subseteq B$ 且 $A,B\in\mathcal F$，则 $\mu(A)\le \mu(B)$。
2. 若 $A_n\uparrow A$，即 $A_n\subseteq A_{n+1}$ 且 $A=\bigcup_nA_n$，则 $\mu(A_n)\uparrow\mu(A)$。
3. 若 $A_n\downarrow A$ 且 $\mu(A_1)<\infty$，则 $\mu(A_n)\downarrow\mu(A)$。

**证明.** 对 1，写 $B=A\cup(B\setminus A)$ 为不交并，由可加性得 $\mu(B)=\mu(A)+\mu(B\setminus A)\ge\mu(A)$。对 2，令 $B_1=A_1$，$B_n=A_n\setminus A_{n-1}$。则 $B_n$ 两两不交且 $A=\bigcup_nB_n$，$A_m=\bigcup_{n\le m}B_n$。因此 $\mu(A_m)=\sum_{n\le m}\mu(B_n)$ 单调上升到 $\sum_n\mu(B_n)=\mu(A)$。对 3，令 $C_n=A_1\setminus A_n$，则 $C_n\uparrow A_1\setminus A$。由 2 得 $\mu(C_n)\uparrow\mu(A_1\setminus A)$。因为 $\mu(A_1)<\infty$，有 $\mu(A_n)=\mu(A_1)-\mu(C_n)$，取极限得到结论。证毕。

## 1.3 Borel 结构与外部输入

**定义 1.4（Borel $\sigma$-代数）。** 若 $E$ 是拓扑空间，则由开集生成的 $\sigma$-代数称为 Borel $\sigma$-代数，记为 $\mathcal B(E)$。

实数轴上的随机变量通常取值于 $(\mathbb R,\mathcal B(\mathbb R))$。区间事件、阈值事件和极限事件都在此结构内。

**外部输入定理 1.2（Caratheodory 扩张，EI-1）。** 设 $\mathcal A$ 是集合 $\Omega$ 上的代数，$\mu_0:\mathcal A\to[0,\infty]$ 是预测度：对每个两两不交的 $(A_n)\subseteq\mathcal A$，只要 $\bigcup_nA_n\in\mathcal A$，就有

$$
\mu_0\left(\bigcup_nA_n\right)=\sum_n\mu_0(A_n).
$$

若存在 $E_n\in\mathcal A$ 使 $\Omega=\bigcup_nE_n$ 且 $\mu_0(E_n)<\infty$，则存在唯一测度 $\mu$ 在 $\sigma(\mathcal A)$ 上延拓 $\mu_0$。本书用该定理从区间或矩形代数上的预测度构造生成 $\sigma$-代数上的测度；无限过程的一致有限维分布拼接由 EI-5 而非 EI-1 承担。

外部输入 1.2 不在正文中证明。其作用是保证从区间长度或矩形体积这类“小事件族上的值”进入完整 $\sigma$-代数；来源和未重证边界见 [SOURCES.md](SOURCES.md)。

## 练习

**练习 1.1.** 证明任意 $\sigma$-代数对可列交封闭。

**练习 1.2.** 在有限集合 $\Omega=\{1,2,3\}$ 上列出由 $\{\{1\}\}$ 生成的 $\sigma$-代数。

**练习 1.3.** 给出一个递减事件列 $A_n$，说明定理 1.1 第 3 条中 $\mu(A_1)<\infty$ 条件不能直接删除。
