# 第四十八章：高阶 Morita、Trace 与 $E_n$-Koszul 对偶

普通 Morita 范畴以代数为对象、双模为态射；高阶版本对 $E_n$-代数逐级迭代这一结构，使 1-态射成为 $E_{n-1}$ 型双模，直到高阶交织子。可对偶对象的 trace 在圆周方向产生 Hochschild 型不变量，因子化同调则把这种 trace 延伸到更一般流形。对增广 $E_n$-代数，迭代 bar construction 又导向 $E_n$-Koszul 对偶及余代数结构。

本章综合 $E_n$-代数、模 $\infty$-范畴、相对张量积、Morita theory、因子化同调与 bar/cobar。Cobordism hypothesis 与高阶 Morita 可对偶性作为外部输入；Koszul 对偶只在完备性、连通性或 nilpotence 条件下陈述。

## 48.1 高阶 Morita 范畴

**定义 48.1.** 对合适对称幺半 $\infty$-范畴 $C$，$n$-重 Morita $(\infty,n)$-范畴 $\operatorname{Alg}_n(C)$ 的对象为 $E_n$-代数，1-态射为相容的 $E_{n-1}$-双模，更高态射递归地由更低阶双模、双模间态射和同伦给出。

**外部输入定理 48.2.** 在 presentable 且张量保持小余极限的假设下，$\operatorname{Alg}_n(C)$ 可构造为 $(\infty,n)$-范畴，并带有由相对张量积给出的复合。

**命题 48.3.** 当 $n=1$ 时，$\operatorname{Alg}_1(C)$ 的 1-态射是普通 $A$-$B$ 双模，复合为相对张量积。

**证明.** $E_1$-代数即结合代数。Morita 1-态射从 $A$ 到 $B$ 是左 $A$、右 $B$ 作用相容的对象 $M$。若 $M:{}_AM_B$ 与 $N:{}_BN_C$ 可复合，则中间 $B$-作用需被平衡，复合为

$$
M\otimes_BN.
$$

这与通常 Morita bicategory 定义一致。$\square$

**命题 48.4.** 由可逆双模定义的 Morita 等价是代数对象上的等价关系。

**证明.** 自反性由恒等 $A$-$A$ 双模 $A$ 给出。若 $M:{}_AM_B$ 有逆双模 $N:{}_BN_A$，满足

$$
M\otimes_BN\simeq A,\qquad N\otimes_AM\simeq B,
$$

则同一数据也说明 $N$ 是从 $B$ 到 $A$ 的 Morita 逆，所以对称性成立。若 $A$ 经 $M$ 与 $B$ 等价，$B$ 经 $P$ 与 $C$ 等价，则复合双模 $M\otimes_BP$ 从 $A$ 到 $C$，其逆为相应逆双模的反向复合；相对张量积的结合律给出两侧评价同构。因此传递性成立。$\square$

## 48.2 可对偶性与光滑真性

**定义 48.5.** 设 $C=\operatorname{Mod}_k$。$E_1$-$k$-代数 $A$ 称为 proper，若其底层 $k$-module 是 perfect；称为 smooth，若 $A$ 作为 $A^e=A^{op}\otimes_kA$-module 是 perfect。

**外部输入定理 48.6.** 在小、幂等完备、$k$-线性稳定 $\infty$-范畴及完美双模组成的 Morita $(\infty,2)$-范畴中，对象 fully dualizable 当且仅当它 smooth 且 proper。对代数 $A$，这正是定义 48.5 的两个完美性条件。

**命题 48.7.** 矩阵代数 $M_n(k)$ 与 $k$ Morita 等价，因此二者有相同的 Morita 不变量。

**证明.** 令 $P=k^n$ 为 $k$-$M_n(k)$ 双模，$Q=(k^n)^*$ 为 $M_n(k)$-$k$ 双模。评价和余评价给出

$$
P\otimes_{M_n(k)}Q\simeq k,\qquad
Q\otimes_kP\simeq M_n(k).
$$

故 $P,Q$ 在 Morita bicategory 中互为逆。任何 Morita 不变量把等价对象送到等价值。$\square$

## 48.3 Trace 与 Hochschild homology

**定义 48.8.** 在 Morita $(\infty,2)$-范畴中，代数 $A$ 的 trace 是恒等 $A$-$A$ 双模 $A$ 的 trace，记作

$$
\operatorname{Tr}(\operatorname{id}_A).
$$

**外部输入定理 48.9.** 对合适 $E_1$-代数，

$$
\operatorname{Tr}(\operatorname{id}_A)\simeq HH(A)\simeq\int_{S^1}A.
$$

**命题 48.10.** Morita 等价的代数有等价的 Hochschild homology。

**证明.** Morita 等价是在 Morita $(\infty,2)$-范畴中的对象等价。Trace 是 Morita $(\infty,2)$-范畴中的等价不变量：等价对象的恒等 1-态射在共轭下对应，其 traces 等价。由定理 48.9，trace 即 Hochschild homology，故 $HH$ Morita 不变。$\square$

## 48.4 Higher traces 与环形因子化同调

**定义 48.11.** 对 $E_n$-代数 $A$，沿 $k$ 个彼此独立的可追踪方向作迭代 trace，对应的局部几何模型是

$$
\int_{(S^1)^k\times\mathbb R^{n-k}}A,\qquad 0\le k\le n.
$$

这里必须使用 $k$-维环面 $(S^1)^k$，而不是球面 $S^k$；后者的因子化同调是另一个有意义但一般不同的不变量。

**外部输入定理 48.12.** Higher Morita traces 与 factorization homology 相容：在相应可对偶性和切向结构假设下，可对偶 $E_n$-代数的 $k$-重迭代 trace 由 $(S^1)^k\times\mathbb R^{n-k}$ 上的因子化同调计算。

**命题 48.13.** 当 $k=1,n=1$ 时 higher trace 公式恢复 $HH(A)$。

**证明.** 此时 $S^k\times\mathbb R^{n-k}=S^1$。定义 48.11 给出 $\int_{S^1}A$。由第四十二章和定理 48.9，

$$
\int_{S^1}A\simeq HH(A).
$$

$\square$

## 48.5 增广代数与 Koszul 对偶

**定义 48.14.** 增广 $E_n$-代数是 $E_n$-代数 $A$ 配 $E_n$-代数态射

$$
\epsilon:A\to\mathbb 1.
$$

其 bar construction 记作 $\operatorname{Bar}^n(A)$。

**定义 48.15.** $E_n$-Koszul dual 可形式地写为

$$
A^! = \operatorname{End}_A(\mathbb 1)
$$

或等价地由 $n$-重 bar/cobar 构造在合适完备性条件下给出。

**外部输入定理 48.16.** 对满足连通性、完备性或 nilpotence 条件的增广 $E_n$-代数，$E_n$-Koszul duality 给出代数与余代数或对偶代数之间的反等价，并满足双重对偶恢复。

**命题 48.17.** 若 $A=\mathbb 1$ 带恒等增广，则 $A^!\simeq\mathbb 1$。

**证明.** 由定义

$$
A^!=\operatorname{End}_A(\mathbb 1).
$$

当 $A=\mathbb 1$ 时，$A$-module 范畴就是 $C$ 本身，单位对象作为自身模的 endomorphism object 为

$$
\operatorname{End}_{\mathbb 1}(\mathbb 1)\simeq\mathbb 1.
$$

$\square$

## 48.6 Koszul 对偶与因子化同调

**外部输入定理 48.18.** 在合适有边界或带框架流形上，增广 $E_n$-代数的因子化同调与 Koszul dual coalgebra 的 factorization cohomology 之间存在 Poincare/Koszul 型对偶。

**命题 48.19.** Koszul 对偶把“局部代数数据”转化为“余代数型全局函数数据”。

**证明.** $A^!=\operatorname{End}_A(\mathbb 1)$ 把 $A$-作用下的单位对象自同态作为对偶对象。Bar 构造把乘法的迭代合成转化为余乘法型结构；cobar 构造反向恢复乘法。因此在满足收敛条件时，局部乘法数据可由对偶余代数控制，因子化同调/上同调的对偶定理正是这种转换的全局形式。$\square$

## 48.7 双模、trace 与迭代对偶

高阶 Morita 理论把 $E_n$-代数、双模、相对张量积和 higher traces 放入统一的 $(\infty,n)$-范畴。Trace 恢复 Hochschild homology 并与圆周因子化同调一致。Koszul 对偶通过 bar/cobar 和 endomorphism of the unit 连接代数与余代数，是因子化同调对偶性和高阶代数有限性理论的核心工具。

## 练习

**练习 48.1.** 定义 $n$-重 Morita $(\infty,n)$-范畴的对象和 1-态射。

**练习 48.2.** 说明 $n=1$ 时恢复普通 Morita bicategory。

**练习 48.3.** 定义 proper 与 smooth $E_1$-代数。

**练习 48.4.** 说明 smooth/proper 与 fully dualizable 的关系。

**练习 48.5.** 证明 $M_n(k)$ 与 $k$ Morita 等价。

**练习 48.6.** 定义 Morita trace。

**练习 48.7.** 陈述 trace、$HH(A)$ 和 $\int_{S^1}A$ 的关系。

**练习 48.8.** 证明 Morita 等价代数有等价 Hochschild homology。

**练习 48.9.** 定义 higher Hochschild object 的因子化同调表达。

**练习 48.10.** 说明 $k=1,n=1$ 时恢复普通 $HH(A)$。

**练习 48.11.** 定义增广 $E_n$-代数。

**练习 48.12.** 定义 $E_n$-Koszul dual。

**练习 48.13.** 证明 $\mathbb 1^!\simeq\mathbb 1$。

**练习 48.14.** 解释 Koszul 对偶如何把代数数据转化为余代数数据。

**练习 48.15.** 证明由可逆双模定义的 Morita 等价满足自反、对称和传递。
