# 附录 L：$\mathcal P_\infty$-代数、$A_\infty/L_\infty$ 与 $E_n$ 约定

## 本附录目标

第十章把 $A_\infty$、$L_\infty$、$C_\infty$ 和 $E_n$ 放入同一 operadic 框架。本附录补充三类约定：

1. $\mathcal P_\infty=\Omega\mathcal P^¡$ 与任意 cofibrant replacement 的区别。
2. $A_\infty$ 和 $L_\infty$ 的 suspended coalgebra/coderivation 口径。
3. $E_n$-operad、$n$-Poisson operad、形式性和 rectification 的边界。

## L.1 $\mathcal P_\infty$ 的含义

**定义 L.1.** 若 $\mathcal P$ 是 Koszul operad，本书约定
$$
\mathcal P_\infty=\Omega\mathcal P^¡.
$$
一个 $\mathcal P_\infty$-代数是 dg-operad morphism
$$
\Omega\mathcal P^¡\to\operatorname{End}_V.
$$

**警告 L.2.** $\mathcal P_\infty$ 在本书不是“任意与 $\mathcal P$ 弱等价的 operad”。若使用 Boardman-Vogt resolution $W\mathcal P$、任意 cofibrant replacement $Q\mathcal P$ 或 minimal operad model，必须写出不同记号，并说明它与 $\Omega\mathcal P^¡$ 的比较。

**外部输入定理 L.3.** 若 $\mathcal P$ 是 Koszul operad，则
$$
\Omega\mathcal P^¡\to\mathcal P
$$
是 quasi-isomorphism；在合适模型结构中，它给出 $\mathcal P$ 的 cofibrant resolution。该结论依赖附录 I 的 Koszul 判别。

## L.2 $A_\infty$ 的 suspended coalgebra 口径

**定义 L.4.** 对链复形 $A$，令
$$
T^c(sA)=\bigoplus_{n\ge1}(sA)^{\otimes n}
$$
为 reduced tensor coalgebra，余乘法由 deconcatenation 给出。一个 $A_\infty$-结构等价于次数 $-1$ 的 coderivation
$$
b:T^c(sA)\to T^c(sA)
$$
满足
$$
b^2=0.
$$
Coderivation $b$ 由其 Taylor 分量
$$
b_n:(sA)^{\otimes n}\to sA
$$
唯一决定。未悬挂运算定义为
$$
m_n=s^{-1}b_n s^{\otimes n}.
$$
在本书同调约定下，$|b_n|=-1$，故
$$
|m_n|=n-2.
$$

**命题 L.5.** Coderivation 方程 $b^2=0$ 等价于所有 Stasheff 恒等式。

**证明.** 由于 $b$ 是 tensor coalgebra 上的 coderivation，$b^2$ 仍是 coderivation。因此只需投影到 cogenerators $sA$。投影到 arity $n$ 的分量是所有把某个 $b_s$ 插入另一个 $b_{r+1+t}$ 的和：
$$
\sum_{r+s+t=n}
b_{r+1+t}(1^{\otimes r}\otimes b_s\otimes1^{\otimes t}).
$$
该和为零对所有 $n$ 成立，正是 suspended Stasheff 恒等式。转回 $A$ 上的 $m_n$ 后得到第十章展开公式中的符号版本。$\square$

**命题 L.6.** 若 $b_n=0$ 对所有 $n\ge3$，则 $A_\infty$-结构等价于 dg associative algebra。

**证明.** $b^2=0$ 的 arity $1$ 分量给 $b_1^2=0$，即 $m_1$ 是微分。Arity $2$ 分量给 $m_1$ 对 $m_2$ 的 Leibniz rule。Arity $3$ 分量只剩 $m_2$ 的两种复合，给出严格结合律。更高 arity 分量因 $b_n=0$ 自动为空。$\square$

## L.3 $L_\infty$ 的 cocommutative coalgebra 口径

**定义 L.7.** 对链复形 $V$，令
$$
S^c(sV)=\bigoplus_{n\ge1}\operatorname{Sym}^n(sV)
$$
为 reduced cofree cocommutative coalgebra。一个 $L_\infty$-结构等价于次数 $-1$ 的 coderivation
$$
q:S^c(sV)\to S^c(sV)
$$
满足
$$
q^2=0.
$$
其 Taylor 分量
$$
q_n:\operatorname{Sym}^n(sV)\to sV
$$
转回未悬挂 bracket
$$
\ell_n=s^{-1}q_n s^{\otimes n}
$$
并带 Koszul 反对称性。

**命题 L.8.** 方程 $q^2=0$ 等价于 $L_\infty$ 高阶 Jacobi 恒等式。

**证明.** 与命题 L.5 相同。Coderivation $q^2$ 由其投影到 $sV$ 的分量决定。Arity $n$ 分量是所有把 $q_i$ 插入 $q_j$ 的 shuffle 求和。由于 $S^c(sV)$ 是 cocommutative coalgebra，求和按 shuffle 而非平面插入槽组织。转回 $V$ 后得到带 Koszul 反对称符号的高阶 Jacobi 恒等式。$\square$

**命题 L.9.** 若 $\ell_n=0$ 对所有 $n\ge3$，则 $L_\infty$-结构等价于 dg Lie algebra。

**证明.** Arity $1$ 给微分平方为零；arity $2$ 给微分与 bracket 相容；arity $3$ 给 graded Jacobi 恒等式。反对称性来自 $q_2$ 在 symmetric coalgebra 上的定义经 suspension 转换后的 Koszul 反对称规则。$\square$

## L.4 $C_\infty$ 与 homotopy commutativity

**定义 L.10.** $C_\infty$-operad 是
$$
C_\infty=\Omega\operatorname{Lie}^¡.
$$
$C_\infty$-代数是 homotopy commutative algebra。

**说明 L.11.** $C_\infty$ 不是简单地“$A_\infty$ 加上 $m_2$ 交换”。它由 Lie cooperad 的对偶控制，其展开包含 shuffle relations。若把 $C_\infty$ 结构写成 $A_\infty$ 运算族，则必须附加所有 shuffle 消失条件；这些条件依赖 suspension convention。

## L.5 $E_n$-operad 的模型层级

**定义 L.12.** 拓扑 $E_n$-operad 是与 little $n$-cubes operad $\mathcal C_n$ 弱等价的拓扑 operad。dg $E_n$-operad 是与链 operad
$$
C_\*(\mathcal C_n;k)
$$
通过 quasi-isomorphism zigzag 相连的 dg-operad。

**警告 L.13.** $E_n$-operad 是拓扑或链 operad 的同伦类型；Lurie-style infinity-operad 是 over $N(\mathbf{Fin}_*)$ 的 higher categorical object。二者都描述多输入结构，但所在模型不同。

**外部输入定理 L.14（Cohen-May）.** 对 $n\ge2$，在合适系数和分次约定下，
$$
H_\*(\mathcal C_n;k)\cong\operatorname{Pois}_n.
$$
其中 $\operatorname{Pois}_n$ 有交换乘法和同调次数 $n-1$ 的 Lie bracket。

**说明 L.15.** $H_\*(\mathcal C_n)$ 只记录同调 operad。链级 $C_\*(\mathcal C_n;k)$ 还包含高阶同伦信息；不能从同调同构推出链级形式性。

## L.6 $E_1$、$E_\infty$ 与 rectification

**命题 L.16.** $\pi_0\mathcal C_1\cong\operatorname{Ass}$。

**证明.** $\mathcal C_1(r)$ 的点是 $r$ 个不交小区间。每个连通分支由小区间从左到右的顺序决定，且每个分支可收缩。故 $\pi_0\mathcal C_1(r)$ 是 $[r]$ 上全序集合，即 $\operatorname{Ass}(r)$。代入由区间仿射嵌入复合给出，与全序字代入一致。$\square$

**定义 L.17.** $E_\infty$-operad 是与 $\operatorname{Com}$ 弱等价且具有良好 cofibrancy/自由对称群作用性质的 operad 模型。具体定义依赖拓扑、simplicial、dg 或 spectra 语境。

**警告 L.18.** $E_\infty\to\operatorname{Com}$ 的弱等价不自动给出 $E_\infty$-algebras 与 strict commutative algebras 的等价同伦理论。Rectification 需要附录 G 的假设；在正特征或一般底环上的链复形中通常不能无条件使用。

## L.7 Additivity 与因子化同调边界

**外部输入定理 L.19（Dunn/Lurie additivity；DUNN-1）.** 对 $m,n\ge0$，Lurie *Higher Algebra* Theorem 5.1.2.2 断言典范 bifunctor
$$
E_m^\otimes\times E_n^\otimes\longrightarrow E_{m+n}^\otimes
$$
把目标展示为两个 infinity-operads 的 tensor product，故
$$
E_m^\otimes\otimes E_n^\otimes\simeq E_{m+n}^\otimes
$$
并由 tensor product 的泛性质得到 $E_m$-algebras in $E_n$-algebras 与 $E_{m+n}$-algebras 的比较。这里使用 infinity-operads 的 tensor product；strict topological operad 的 Boardman--Vogt tensor product 仍需另行给出 cofibrancy 与 comparison 定理。

**说明 L.20.** Additivity 是 factorization homology 和 higher algebra 中的重要桥梁，但不属于 little cubes operad 定义本身。正文引用时必须说明使用 Boardman-Vogt tensor product、Lurie tensor product of infinity-operads，还是其他模型。

## L.8 本附录小结

$A_\infty$ 和 $L_\infty$ 的最安全定义分别是 tensor coalgebra 与 cocommutative coalgebra 上的 square-zero coderivation。$\mathcal P_\infty$ 在本书中特指 Koszul dual cooperad 的 cobar resolution。$E_n$-operads 来自拓扑或链模型；其同调 Poisson 结构、形式性、additivity 和 rectification 都是额外定理，不应混入定义。
