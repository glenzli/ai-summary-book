# 第十章：$A_\infty$、$L_\infty$ 与 $E_n$-operad

## 本章目标

本章把第九章的 bar-cobar 工具用于三类核心同伦代数结构：

1. $A_\infty$-operad：控制同伦结合代数。
2. $L_\infty$-operad：控制同伦 Lie 代数。
3. $E_n$-operad：控制 $n$ 重 loop space 型或局部 $n$ 维交换性的代数结构。

本章采用严格的 operad 定义作为主定义；手写高阶恒等式只作为展开说明。这样可以避免在符号约定尚未完全固定时把错误符号写进核心定义。

## 依赖前置知识

需要第六章的 $\operatorname{Ass}$、$\operatorname{Com}$、$\operatorname{Lie}$，第八章的 Koszul 对偶，以及第九章的 dg-operad、cooperad、bar-cobar 和 twisting morphism。

## 10.1 同伦 $\mathcal P$-代数的 operadic 定义

**定义 10.1.** 设 $\mathcal P$ 是 augmented dg-operad。一个同伦 $\mathcal P$-代数通常指某个 cofibrant resolution
$$
\mathcal P_\infty\xrightarrow{\sim}\mathcal P
$$
上的代数。更具体地，若 $\mathcal P$ 是 Koszul operad，则本书定义
$$
\mathcal P_\infty=\Omega\mathcal P^¡.
$$
于是一个 $\mathcal P_\infty$-代数是 dg-operad morphism
$$
\mathcal P_\infty\to\operatorname{End}_V
$$
其中 $V$ 是链复形。

**解释 10.2.** 这个定义把“严格代数公理”替换为“严格公理只在同伦意义下成立，并带有所有高阶相干修正”。这些高阶修正不是额外选择的松散数据，而是由单个 dg-operad morphism 统一编码。

**外部输入定理 10.3.** 若 $\mathcal P$ 是 Koszul operad，则自然映射
$$
\Omega\mathcal P^¡\to\mathcal P
$$
是 quasi-isomorphism。因此 $\mathcal P_\infty=\Omega\mathcal P^¡$ 是 $\mathcal P$ 的准自由 cofibrant resolution。该结论依赖第九章外部输入定理 9.23。

## 10.2 $A_\infty$-operad

**定义 10.4.** $A_\infty$-operad 定义为
$$
A_\infty=\Omega\operatorname{Ass}^¡.
$$
由于 $\operatorname{Ass}$ 是 Koszul 且自对偶，$\operatorname{Ass}^¡$ 可理解为结合 cooperad 的适当悬挂版本。

**定义 10.5.** 一个 $A_\infty$-代数是链复形 $A$ 连同 dg-operad morphism
$$
A_\infty\to\operatorname{End}_A.
$$
等价地，它由一族次数
$$
|m_n|=n-2
$$
的 $k$-线性映射
$$
m_n:A^{\otimes n}\to A,\qquad n\ge1,
$$
满足 Stasheff 恒等式组成。本章使用同调分次，因此 $m_1$ 的次数为 $-1$，可作为底层链复形微分。

**展开 10.6.** 在常用悬挂约定下，Stasheff 恒等式可写为
$$
\sum_{r+s+t=n}
(-1)^{r+st}
m_{r+1+t}
(\operatorname{id}^{\otimes r}\otimes m_s\otimes
\operatorname{id}^{\otimes t})=0,
$$
其中 $r,t\ge0$、$s\ge1$。不同文献的同调/上同调分次和 suspension 约定会改变符号；本书的核心定义是定义 10.4 和 10.5。

**命题 10.7.** 若 $A$ 是 $A_\infty$-代数，则：

1. $m_1^2=0$。
2. $m_1$ 对 $m_2$ 满足带符号 Leibniz 规则。
3. $m_2$ 的结合律缺陷由 $m_3$ 的边界控制。

**证明.** 取展开 10.6 中 $n=1$，唯一项为 $m_1m_1$，故 $m_1^2=0$。取 $n=2$，得到三项关系，正是 $m_1$ 与 $m_2$ 的带符号 Leibniz 规则。取 $n=3$，包含两项二元乘法的复合
$$
m_2(m_2\otimes\operatorname{id}),\qquad
m_2(\operatorname{id}\otimes m_2)
$$
以及含 $m_1$ 和 $m_3$ 的项；移项后说明 associator 是 $m_3$ 的边界。$\square$

**例 10.8.** 若 $m_n=0$ 对所有 $n\ge3$ 成立，则 $A_\infty$-代数退化为 dg associative algebra：$m_1$ 是微分，$m_2$ 是链映射并严格结合。

## 10.3 $L_\infty$-operad

**定义 10.9.** $L_\infty$-operad 定义为
$$
L_\infty=\Omega\operatorname{Com}^¡.
$$
因为 $\operatorname{Com}^!\cong\operatorname{Lie}$，这里的 cooperad $\operatorname{Com}^¡$ 是控制 Lie 型高阶括号的对偶 cooperad。

**定义 10.10.** 一个 $L_\infty$-代数是链复形 $V$ 连同 dg-operad morphism
$$
L_\infty\to\operatorname{End}_V.
$$
等价地，它给出一族带适当反对称性的多重括号
$$
\ell_n:V^{\otimes n}\to V,\qquad n\ge1,
$$
其次数在本书同调约定下为
$$
|\ell_n|=n-2,
$$
并满足高阶 Jacobi 恒等式。

**展开 10.11.** 高阶 Jacobi 恒等式可概念性写为
$$
\sum_{i+j=n+1}
\sum_{\sigma\in\operatorname{Sh}(i,n-i)}
\pm
\ell_j\big(\ell_i(x_{\sigma(1)},\ldots,x_{\sigma(i)}),
x_{\sigma(i+1)},\ldots,x_{\sigma(n)}\big)=0.
$$
这里 $\operatorname{Sh}(i,n-i)$ 是 shuffle 集，符号由 shuffle 的 Koszul 符号、反对称性和悬挂约定共同决定。

**命题 10.12.** 若 $V$ 是 $L_\infty$-代数，则 $\ell_1^2=0$，$\ell_1$ 对 $\ell_2$ 满足带符号导子关系，$\ell_2$ 的 Jacobi 恒等式缺陷由 $\ell_3$ 的边界控制。

**证明.** 这是展开 10.11 在 $n=1,2,3$ 的直接结果。$n=1$ 给出 $\ell_1^2=0$。$n=2$ 给出微分与二元括号的相容性。$n=3$ 中的二元括号嵌套项是 Jacobiator，含 $\ell_3$ 与 $\ell_1$ 的项说明该 Jacobiator 为边界。$\square$

**例 10.13.** 若 $\ell_n=0$ 对所有 $n\ge3$ 成立，则 $L_\infty$-代数退化为 dg Lie algebra。

## 10.4 $C_\infty$-代数

**定义 10.14.** $C_\infty$-operad 定义为
$$
C_\infty=\Omega\operatorname{Lie}^¡.
$$
它控制同伦交换代数，也称 homotopy commutative algebra。

**说明 10.15.** $C_\infty$-代数可以看作带有高阶同伦的交换 dg-algebra。它与 $A_\infty$-代数的关系不是简单地令所有 $m_n$ 对称，而是通过 Lie cooperad 的对偶结构和 shuffle 关系刻画。完整展开需要 operadic suspension 和 shuffle 子空间，本书将在 Gerstenhaber 与 Deligne 章节中继续使用。

## 10.5 Little cubes operad

**定义 10.16.** 对 $n\ge1$，little $n$-cubes operad $\mathcal C_n$ 是拓扑 operad。其 $r$ 元空间 $\mathcal C_n(r)$ 由 $r$ 个互不相交的小 $n$-立方体嵌入组成。一个小 $n$-立方体是映射
$$
[0,1]^n\to[0,1]^n
$$
形如
$$
(t_1,\ldots,t_n)\mapsto
(a_1t_1+b_1,\ldots,a_nt_n+b_n),
$$
其中 $a_i>0$，且像包含在 $[0,1]^n$ 中。$r$ 个小立方体要求其内部两两不交。

对称群 $\Sigma_r$ 通过重标号小立方体作用。Operad 代入把一个配置中的第 $i$ 个小立方体替换为另一个配置，并用仿射缩放把后者嵌入前者的位置。

**命题 10.17.** $\mathcal C_n$ 是拓扑 operad。

**证明.** 单位是恒等嵌入 $[0,1]^n\to[0,1]^n$。代入的结合律来自仿射映射复合的结合律：先把小立方体缩放进中间立方体，再缩放进外层立方体，与直接复合仿射缩放得到同一嵌入。对称群等变性来自配置重标号与仿射代入相容。连续性由仿射参数的连续变化给出。$\square$

**定义 10.18.** 一个拓扑 operad $\mathcal E$ 称为 $E_n$-operad，若它与 $\mathcal C_n$ 在拓扑 operad 的同伦理论中弱等价。一个 dg $E_n$-operad 通常指与链 operad
$$
C_\*(\mathcal C_n;k)
$$
弱等价的 dg-operad。

**警告 10.19.** $E_n$-operad 不是 Lurie-style infinity-operad。前者是一个拓扑或链 operad 的同伦类型；后者是 higher category theory 中编码多输入运算的 infinity-categorical 对象。本书后续会在第三部分单独处理 infinity-operad。

## 10.6 $E_1$、$E_\infty$ 与 Poisson 同调

**命题 10.20.** $\mathcal C_1$ 的连通分支 operad 与对称结合 operad $\operatorname{Ass}$ 同构；因此链级 $E_1$-代数是同伦结合代数的拓扑来源。

**证明.** $\mathcal C_1(r)$ 是 $r$ 个互不相交小区间在 $[0,1]$ 中的配置空间。每个连通分支由这些区间从左到右的顺序决定，并且每个分支可收缩。因而 $\pi_0\mathcal C_1(r)\cong\Sigma_r$，每个分支对应 $[r]$ 上的一个全序，这正是第一章定义的 $\operatorname{Ass}(r)$。代入与区间嵌入复合相容，所以得到 operad 同构 $\pi_0\mathcal C_1\cong\operatorname{Ass}$。$\square$

**定义 10.21.** 一个 $E_\infty$-operad 是与交换 operad $\operatorname{Com}$ 弱等价且各 arity 带自由 $\Sigma_r$-作用的 cofibrant 型拓扑或链 operad。直观上，$E_\infty$-代数是“同伦意义下交换”的代数。

**外部输入定理 10.22.** 对 $n\ge2$，little $n$-cubes operad 的同调 operad 满足
$$
H_\*(\mathcal C_n;k)\cong \operatorname{Pois}_n,
$$
其中 $\operatorname{Pois}_n$ 是 $n$-Poisson operad：它有交换乘法和次数 $n-1$ 的 Lie bracket，并满足相应带符号 Leibniz 关系。该定理通常归于 Cohen 的 iterated loop spaces 同调计算，后续使用时应引用 May、Cohen 或 Fresse。

**说明 10.23.** 当 $n=2$ 时，$\operatorname{Pois}_2$ 通常称为 Gerstenhaber operad。它控制 Hochschild cochains 上 Deligne 猜想中的同调结构。链级 $E_2$-结构比同调 Gerstenhaber 结构更强，因为它包含所有高阶同伦。

## 10.7 识别定理的边界

**外部输入定理 10.24.** May 的 recognition principle 说明，适当连通性和群完备条件下，带有 $\mathcal C_n$-代数结构的空间识别为 $n$ 重 loop space。精确陈述依赖 based spaces、group completion 和模型范畴语境，本书在拓扑 operad 章节再给出完整版本。

**外部输入定理 10.25.** Chains on little disks/cubes 给出 dg-operad，其代数在链复形中建模 $E_n$-algebras。若底域特征为 $0$，$E_n$-operad 的形式性在若干情形成立；特别 $E_2$ 的形式性与 Drinfeld associators 和 Grothendieck-Teichmüller 理论有关。该方向必须在后续章节引用 Tamarkin、Kontsevich、Fresse 等来源，不在本章证明。

## 本章小结

$A_\infty$、$L_\infty$ 和 $C_\infty$ 是通过 bar-cobar/Koszul 对偶得到的 dg-operad，分别控制同伦结合、同伦 Lie 和同伦交换代数。$E_n$-operad 来自 little cubes/disks 的拓扑 operad，控制 $n$ 维局部交换性；其同调是 Poisson 型 operad，但链级结构包含更高同伦信息。本章把这两条路线放入同一 operadic 框架，但保持模型层级的区分。

## 练习

**练习 10.1.** 用 Stasheff 恒等式写出 $n=1,2,3$ 的三条关系，并标明每一项的次数。

**练习 10.2.** 证明若 $A_\infty$-代数中 $m_n=0$ 对所有 $n\ge3$，则 $m_2$ 在链同伦意义外已经严格结合。

**练习 10.3.** 写出 $L_\infty$ 恒等式中 $n=3$ 时的 Jacobiator 项和含 $\ell_3$ 的边界项。

**练习 10.4.** 证明 $\mathcal C_1(r)$ 的每个连通分支可收缩。

**练习 10.5.** 解释为什么 $H_\*(\mathcal C_n)$ 只记录 $E_n$-operad 的同调层信息，而不能恢复完整链级 $E_n$-结构。
