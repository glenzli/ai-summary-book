# 第八章：二次 operad 与 Koszul 对偶

许多经典线性 operad 都由二元生成元和三输入关系给出：三输入关系对应恰有两个顶点的树，因而具有权重 $2$。一旦自由 operad 按顶点数分次，关系子模 $R\subset\mathbb F(E)^{(2)}$ 就能取正交补；但“存在二次对偶”与“对偶能计算原 operad 的同伦信息”是两件不同的事，后者正是 Koszul 性。本章在 reduced、augmented 和逐 arity 有限维假设下定义二次数据、Ginzburg--Kapranov 对偶与 Koszul complex，并用无幺 $\operatorname{Ass}$、$\operatorname{Com}$、$\operatorname{Lie}$ 检查关系。自由 operad、线性代入和有限维表示对偶是所需的前置工具。

## 8.1 Reduced 约定与权重

**约定 8.1.** 本章固定域 $k$。为避免对偶和符号问题，除非特别说明，所有 $\mathbb S$-模在每个 arity 都有限维。本章讨论的二次 operad 默认是 reduced operad：
$$
\mathcal P(0)=0,\qquad \mathcal P(1)=k\cdot\mathbf 1.
$$
因此本章的 $\operatorname{Ass}$、$\operatorname{Com}$、$\operatorname{Lie}$ 默认是不含 arity $0$ 单位运算的版本。含单位版本由后续的非齐次二次理论处理。

**定义 8.2.** 设 $E$ 是 reduced $\mathbb S$-模，即 $E(0)=E(1)=0$。自由 operad $\mathbb F(E)$ 按顶点数带有权重分次：
$$
\mathbb F(E)=I\oplus\bigoplus_{r\ge1}\mathbb F^{(r)}(E),
$$
其中 $\mathbb F^{(r)}(E)$ 由恰有 $r$ 个内部顶点的 $E$-装饰树张成。

**命题 8.3.** 设
$$
x\in\mathbb F^{(r)}(E)(n),
\qquad
y_t\in\mathbb F^{(s_t)}(E)(m_t)
\quad(1\le t\le n).
$$
则自由 operad 的代入满足
$$
\gamma\bigl(x;(y_t)_{1\le t\le n}\bigr)
\in
\mathbb F^{\left(r+\sum_{t=1}^n s_t\right)}(E)
\left(\sum_{t=1}^n m_t\right).
$$
特别地，若所有内层运算都具有同一权重 $s$，则输出权重为 $r+ns$，而不是一般地为 $r+s$。

**证明.** 取表示 $x$ 的外层装饰树 $T$，以及表示各 $y_t$ 的内层装饰树 $T_t$。Grafting 后的内部顶点集合为不交并
$$
V(T)\amalg\coprod_{t=1}^nV(T_t).
$$
第一项有 $r$ 个顶点，第 $t$ 个内层项有 $s_t$ 个顶点，所以总顶点数是 $r+\sum_t s_t$。叶集合是各 $T_t$ 叶集合的不交并，因而总 arity 是 $\sum_t m_t$。$\square$

例如，经由典范识别 $E(2)\cong\mathbb F^{(1)}(E)(2)$ 取二元生成元 $x,y_1,y_2$。同时把 $y_1,y_2$ 代入 $x$ 的两个输入槽得到权重 $1+1+1=3$ 的三顶点树；偏复合 $x\circ_1y_1$ 则在第二槽代入 operad 单位，故只有一个非单位内层树，权重为 $1+1=2$。因此 $r+s$ 只描述一次非单位插入，不能描述完整的 $\circ$-代入。

## 8.2 二次数据与二次 operad

**定义 8.4.** 一个二次数据（quadratic datum）是二元组 $(E,R)$，其中 $E$ 是 reduced $\mathbb S$-模，$R$ 是子 $\mathbb S$-模
$$
R\subseteq\mathbb F^{(2)}(E).
$$
由 $(E,R)$ 定义的二次 operad 是
$$
\mathcal P(E,R)=\mathbb F(E)/(R),
$$
其中 $(R)$ 表示由 $R$ 生成的 operadic ideal。

**定义 8.5.** Operadic ideal 是子 $\mathbb S$-模 $\mathcal I\subseteq\mathcal O$，满足任意把 $\mathcal I$ 中元素代入 $\mathcal O$ 中元素，或把 $\mathcal O$ 中元素代入 $\mathcal I$ 中元素，所得元素仍属于 $\mathcal I$。商 $\mathcal O/\mathcal I$ 逐 arity 取商，并继承唯一 operad 结构。

**命题 8.6.** 定义 8.4 中的 $\mathcal P(E,R)$ 满足泛性质：给定任意 reduced operad $\mathcal Q$，operad morphism
$$
\mathcal P(E,R)\to\mathcal Q
$$
等价于 $\mathbb S$-模态射 $f:E\to U\mathcal Q$，使得 $R$ 中每个二次关系在 $\mathcal Q$ 中取值为 $0$。

**证明.** 由自由 operad 泛性质，$f:E\to U\mathcal Q$ 唯一延拓为 $\widehat f:\mathbb F(E)\to\mathcal Q$。该延拓穿过商 $\mathbb F(E)/(R)$ 当且仅当由 $R$ 生成的 operadic ideal 被送到 $0$。由于 $\ker(\widehat f)$ 是 operadic ideal，后一条件等价于 $R\subseteq\ker(\widehat f)$。$\square$

## 8.3 三个二次例子

**例 8.7.** 非含单位结合 operad $\operatorname{Ass}$ 由一个二元生成元 $\mu$ 生成，关系为
$$
\mu\circ_1\mu-\mu\circ_2\mu=0.
$$
在对称 operad 口径中，$E(2)$ 是由 $\mu$ 的 $\Sigma_2$-轨道生成的右表示；关系的所有输入重标号也同时属于 $R$。

**例 8.8.** 非含单位交换 operad $\operatorname{Com}$ 由一个二元生成元 $\mu$ 生成，满足
$$
(12)\cdot\mu=\mu
$$
以及结合关系
$$
\mu\circ_1\mu-\mu\circ_2\mu=0.
$$
因为 $\mu$ 对输入对称，结合关系的不同标号版本等价于交换乘法的结合律。

**例 8.9.** Lie operad $\operatorname{Lie}$ 由一个二元生成元 $\lambda$ 生成，满足
$$
(12)\cdot\lambda=-\lambda
$$
和 Jacobi 关系
$$
\lambda\circ_1\lambda
+(\lambda\circ_1\lambda)\cdot(123)
+(\lambda\circ_1\lambda)\cdot(132)
=0,
$$
其中右侧两个项表示对三个输入作循环重标号。该关系在代数上对应
$$
[[x,y],z]+[[y,z],x]+[[z,x],y]=0.
$$

**警告 8.10.** 命题 6.15、命题 6.16 和命题 6.21 中含单位的 $\operatorname{Ass}$、$\operatorname{Com}$ 和 $\operatorname{Pois}$ 包含 arity $0$ 运算。它们不是本章 reduced 二次理论的对象。若要处理单位，通常使用 augmented operad、非齐次二次关系或 curved 版本。

## 8.4 二次对偶

**定义 8.11.** 对有限维 $\mathbb S$-模 $E$，定义其 operadic 对偶生成元
$$
E^\vee(n)=E(n)^*\otimes\operatorname{sgn}_n,
$$
其中 $\operatorname{sgn}_n$ 是 $\Sigma_n$ 的符号表示。符号扭曲是 Ginzburg-Kapranov 对偶约定的一部分。

自由 operad 的权重 $2$ 部分之间存在自然非退化配对
$$
\langle-,-\rangle:
\mathbb F^{(2)}(E^\vee)(n)\otimes
\mathbb F^{(2)}(E)(n)\to k.
$$
它由顶点装饰的线性对偶配对、树形匹配和符号表示扭曲共同定义。

**定义 8.12.** 设 $\mathcal P=\mathcal P(E,R)$ 是二次 operad。其 Ginzburg-Kapranov 二次对偶 operad 定义为
$$
\mathcal P^!=\mathcal P(E^\vee,R^\perp),
$$
其中
$$
R^\perp\subseteq\mathbb F^{(2)}(E^\vee)
$$
是 $R$ 在上述配对下的正交补。

**命题 8.13.** 在有限维假设下，有自然同构
$$
(\mathcal P^!)^!\cong\mathcal P.
$$

**证明.** 有限维性给出 $(E^\vee)^\vee\cong E$，符号表示满足 $\operatorname{sgn}_n\otimes\operatorname{sgn}_n\cong k$。非退化配对下的线性代数恒等式
$$
(R^\perp)^\perp=R
$$
给出关系空间恢复。自由 operad 的权重 $2$ 配对与 $\Sigma_n$-作用相容，因此得到二次 operad 的自然同构。$\square$

**外部输入定理 8.14.** 在本章约定下，
$$
\operatorname{Ass}^!\cong\operatorname{Ass},\qquad
\operatorname{Com}^!\cong\operatorname{Lie},\qquad
\operatorname{Lie}^!\cong\operatorname{Com}.
$$
这些同构依赖二次关系空间的显式正交补计算和符号扭曲约定。后续使用时以 Ginzburg-Kapranov 或 Loday-Vallette 为来源。

## 8.5 Koszul 性

**定义 8.15.** 设 $\mathcal P=\mathcal P(E,R)$ 是二次 operad。其二次对偶 cooperad 通常记为
$$
\mathcal P^¡.
$$
在本书的同调分次中，定义
$$
\mathcal P^¡=\mathcal C(sE,s^2R)
\subseteq\mathbb T^c(sE),
$$
即 cofree conilpotent cooperad 中由 cogenerators $sE$ 和 weight-$2$ corelations $s^2R$ 决定的最大子 cooperad。这里 $s$ 是定义 9.2 的链悬挂，不是定义 E.11 的 operadic suspension $\Lambda$；$s^2R$ 通过两顶点树上“每个顶点各悬挂一次”的典范识别嵌入，交换悬挂符号时使用 Koszul braiding。特别地，
$$
(\mathcal P^¡)^{(0)}=I,\qquad
(\mathcal P^¡)^{(1)}=sE,\qquad
(\mathcal P^¡)^{(2)}=s^2R
$$
作为 $\mathbb T^c(sE)$ 相应权重部分的子对象。它带有到 $\mathcal P$ 的典范 twisting morphism
$$
\kappa:\mathcal P^¡\to\mathcal P.
$$
该映射在 weight $1$ 上为 $sE\xrightarrow{s^{-1}}E$，在其他权重为零。这里 $\mathcal P^!$ 是定义 8.12 由 $E^\vee$ 与 $R^\perp$ 构造的正交对偶 operad，而 $\mathcal P^¡$ 是本定义的 conilpotent cooperad；只有在有限型假设下作双对偶或线性对偶比较时，才能把二者经 suspension/sign twist 联系起来，不能只靠更换感叹号互换。完整符号还使用定义 9.6 的 dg cooperad、定义 9.2 的链悬挂和定义 9.11 的 infinitesimal decomposition。

**定义 8.16.** 二次 operad $\mathcal P$ 称为 Koszul，若由 $\kappa:\mathcal P^¡\to\mathcal P$ 定义的 Koszul complex
$$
\mathcal P^¡\circ_\kappa \mathcal P
$$
在正权重上无同调，并且其到单位 operad $I$ 的增广是 quasi-isomorphism。

**警告 8.17.** Koszul 性不是“存在二次对偶”本身。每个有限型二次 operad 都有 $\mathcal P^!$，但不一定 Koszul。Koszul 性是关于一个链复形无同调的同调断言。

**外部输入定理 8.18.** $\operatorname{Ass}$、$\operatorname{Com}$ 和 $\operatorname{Lie}$ 是 Koszul operad。更一般的判别方法包括 distributive law、PBW 型基和 Gröbner basis 方法。Classical quadratic 来源已定位为 Ginzburg--Kapranov Corollary 4.2.7，即 GK-5；其他模型或符号版本仍可引用 Loday--Vallette 或 Fresse。

**外部输入定理 8.19.** 若 $\mathcal P$ 是有限型 Koszul operad，则其二次对偶 $\mathcal P^!$ 也是 Koszul，并且指数生成函数满足 Ginzburg-Kapranov 关系
$$
g_{\mathcal P}\big(-g_{\mathcal P^!}(-t)\big)=t
$$
或等价符号约定下的同一反函数关系。Classical 来源已定位为 Ginzburg--Kapranov Proposition 4.1.4，即 GK-2；使用该公式前必须明确生成函数和悬挂符号约定。

## 8.6 为什么 Koszul 对偶重要

Koszul operad 的意义在于：若 $\mathcal P$ Koszul，则其同伦版本通常可由对偶 cooperad 的 cobar 构造给出。典型结果包括：

- $\operatorname{Ass}$ 的 Koszul 性导向 $A_\infty$-operad；
- $\operatorname{Lie}$ 的 Koszul 性导向 $L_\infty$-operad；
- $\operatorname{Com}$ 的 Koszul 性导向 $C_\infty$ 或 homotopy commutative algebra；
- Poisson、Gerstenhaber 等 operad 的 Koszul 性控制相应同伦代数结构。

这些陈述需要 dg-operad、cooperad、bar-cobar 和模型范畴语言。下一章会建立这些工具。

## 8.7 正交关系何时控制同伦

二次数据 $(E,R)$ 只使用自由 operad 的权重 $1$ 与 $2$ 部分；对偶数据则把 $R$ 替换为带正确悬挂和符号的正交补。真正有计算力的是 Koszul complex 的无额外同调：只有在这一条件下，二次对偶 cooperad 才能提供原 operad 的有效分解。$\operatorname{Ass}$ 的自对偶以及 $\operatorname{Com}$ 与 $\operatorname{Lie}$ 的互换由外部经典定理保证，不能从符号形式直接推出。下一章将构造不依赖 Koszul 假设也存在的 bar 与 cobar；Koszul 性随后表现为这些普遍构造在特定二次数据上足够精确。

## 练习

**练习 8.1.** 证明命题 8.3 的多权重公式 $r+\sum_t s_t$，并在所有 $s_t=s$ 时推出 $r+ns$。解释为什么只写 $r+s$ 会漏掉其余非单位内层树的顶点。

**练习 8.2.** 写出 $\operatorname{Ass}$ 的二次关系在 arity $3$ 中的所有输入重标号版本。

**练习 8.3.** 在 $\operatorname{Com}$-代数中验证结合关系和输入对称关系共同推出所有三元乘积与加括号和输入顺序无关。

**练习 8.4.** 假设 $E(2)$ 一维且带平凡 $\Sigma_2$ 作用，计算 $E^\vee(2)$ 的 $\Sigma_2$ 作用。

**练习 8.5.** 解释为什么“$\mathcal P$ 有二次对偶”不推出“$\mathcal P$ Koszul”。
