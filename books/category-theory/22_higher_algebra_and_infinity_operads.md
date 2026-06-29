# 第二十二章：高阶代数、$\infty$-operad 与幺半 $\infty$-范畴

## 本章目标

本章给出高阶代数的入口：$\infty$-operad、$E_n$-代数、幺半 $\infty$-范畴和代数对象。目标是明确后续学习 Lurie *Higher Algebra* 所需的范畴论语言。

## 依赖前置知识

需要 Cartesian fibration、coCartesian fibration、幺半范畴和 quasi-category。

## 22.1 从 operad 到 $\infty$-operad

**定义 22.1.** 普通有色 operad 记录多输入一输出运算及其复合。颜色集替代单一对象，运算形如

$$
(x_1,\dots,x_n)\to y.
$$

**定义 22.2.** $\infty$-operad 可建模为满足 Segal 条件的单纯集映射

$$
\mathcal O^\otimes\to N(\mathbf{Fin}_*)
$$

并带有 inert morphism 的 coCartesian lift 条件。该定义把多输入运算和高阶同伦相干性编码在一个 coCartesian fibration 型结构中。

**注 22.3.** 本书采用 Lurie 的 $\infty$-operad 口径。其他模型包括 dendroidal sets、simplicial operads 和 complete Segal operads。

## 22.2 $\mathbf{Fin}_*$ 中的 active 与 inert 态射

**定义 22.4.** 记 $\langle n\rangle=\{*,1,\dots,n\}$。态射 $f:\langle m\rangle\to\langle n\rangle$ 称为 inert，若对每个 $j\in\{1,\dots,n\}$，集合 $f^{-1}(j)$ 恰有一个元素。称 $f$ 为 active，若

$$
f^{-1}(*)=\{*\}.
$$

**命题 22.5.** $\mathbf{Fin}_*$ 中任意态射可分解为一个 inert 态射后接一个 active 态射。

**证明.** 设 $f:\langle m\rangle\to\langle n\rangle$。令

$$
T=\{i\in\{1,\dots,m\}\mid f(i)\ne *\}.
$$

取带基点集合 $\langle k\rangle$，其中非基点与 $T$ 一一对应。定义

$$
\rho:\langle m\rangle\to\langle k\rangle
$$

把 $T$ 中元素按该一一对应送到非基点，把 $*$ 和 $\{1,\dots,m\}\setminus T$ 送到 $*$。于是 $\rho$ 是 inert：$\langle k\rangle$ 的每个非基点有唯一原像。再定义

$$
\alpha:\langle k\rangle\to\langle n\rangle
$$

把对应于 $i\in T$ 的非基点送到 $f(i)$，把 $*$ 送到 $*$。由于 $i\in T$ 时 $f(i)\ne *$，故 $\alpha^{-1}(*)=\{*\}$，所以 $\alpha$ 是 active。逐点检查基点与非基点可得 $f=\alpha\rho$。$\square$

**注 22.6.** 在 $\infty$-operad 中，inert 态射控制投影到各输入槽，active 态射控制真正的多输入运算。Segal 条件正是说总纤维由这些输入槽恢复。

## 22.3 Segal 条件与多重映射空间

对 $1\le i\le n$，定义 inert 态射

$$
\rho_i:\langle n\rangle\to\langle1\rangle
$$

为 $\rho_i(i)=1$，并把所有其他非基点和基点送到 $*$。

**定义 22.A.** 一个 $\infty$-operad

$$
p:\mathcal O^\otimes\to N(\mathbf{Fin}_*)
$$

的 Segal 条件要求：对每个 $n\ge0$，由 inert 态射 $\rho_i$ 的 $p$-coCartesian lift 诱导的函子

$$
\mathcal O^\otimes_{\langle n\rangle}
\longrightarrow
\prod_{i=1}^n\mathcal O^\otimes_{\langle1\rangle}
$$

是 $\infty$-范畴等价。这里 $n=0$ 时右侧为空积，即终 $\infty$-范畴。

因此，位于 $\langle n\rangle$ 上的对象可理解为 $n$ 个颜色的有序列表

$$
(X_1,\dots,X_n).
$$

**定义 22.B.** 设 $X_1,\dots,X_n,Y\in\mathcal O^\otimes_{\langle1\rangle}$。取一个位于 $\langle n\rangle$ 上并投影到 $X_1,\dots,X_n$ 的对象 $X$。令

$$
\mu:\langle n\rangle\to\langle1\rangle
$$

为把所有非基点送到 $1$ 的 active 态射。定义从 $(X_1,\dots,X_n)$ 到 $Y$ 的多重映射空间为映射空间

$$
\operatorname{Mul}_{\mathcal O}(X_1,\dots,X_n;Y)
$$

中那些覆盖 $\mu$ 的分量，即

$$
\operatorname{fib}_{\mu}\Bigl(
\operatorname{Map}_{\mathcal O^\otimes}(X,Y)
\to
\operatorname{Map}_{N(\mathbf{Fin}_*)}(\langle n\rangle,\langle1\rangle)
\Bigr).
$$

由 Segal 条件，$X$ 的选择在可缩选择空间中唯一，因此该定义在等价意义下只依赖 $X_1,\dots,X_n,Y$。

**例子 22.C.** 当 $n=1$ 时，$\operatorname{Mul}_{\mathcal O}(X;Y)$ 是 unary operation 的空间。当 $n=0$ 时，$\operatorname{Mul}_{\mathcal O}(;Y)$ 是常元或单位型运算的空间。当 $\mathcal O^\otimes=C^\otimes$ 来自幺半 $\infty$-范畴时，active 态射 $\mu:\langle n\rangle\to\langle1\rangle$ 编码把 $n$ 个输入张量后映到一个输出的操作。

## 22.4 幺半 $\infty$-范畴

**定义 22.7.** 幺半 $\infty$-范畴是 $\infty$-operad

$$
C^\otimes\to N(\mathbf{Fin}_*)
$$

满足其纤维 $C^\otimes_{\langle n\rangle}$ 等价于 $C^n$。这里 $C=C^\otimes_{\langle1\rangle}$ 是底层 $\infty$-范畴。

**例子 22.8.** spaces、谱、链复形的导出 $\infty$-范畴、以及 presentable $\infty$-categories 在适当张量积下都有幺半 $\infty$-范畴结构。

## 22.5 代数对象

**定义 22.9.** 设 $\mathcal O^\otimes$ 为 $\infty$-operad，$C^\otimes$ 为幺半 $\infty$-范畴。$C$ 中的 $\mathcal O$-代数是 $\infty$-operad 映射

$$
\mathcal O^\otimes\to C^\otimes.
$$

所有 $\mathcal O$-代数组成 $\infty$-范畴

$$
\operatorname{Alg}_{\mathcal O}(C).
$$

**例子 22.10.** 当 $\mathcal O=E_1$ 时，$\operatorname{Alg}_{E_1}(C)$ 是结合代数对象的 $\infty$-范畴。当 $\mathcal O=E_\infty$ 时，得到交换代数对象的 $\infty$-范畴。

## 22.6 $E_n$-代数

**定义 22.11.** $E_n$-operad 由 $\mathbb R^n$ 中小圆盘配置的同伦类型给出。$E_1$ 控制结合代数，$E_\infty$ 控制同伦交换代数，有限 $n$ 控制介于二者之间的高阶交换性。

**外部输入定理 22.12（Dunn additivity）.** 在适当模型中有等价

$$
E_m\otimes E_n\simeq E_{m+n}.
$$

该定理解释迭代 $E_m$ 与 $E_n$ 代数结构如何合成为 $E_{m+n}$ 结构。

## 22.7 高阶代数中的伴随和模

**外部输入定理 22.13.** 若 $C$ 是 presentable 幺半 $\infty$-范畴且张量积分别保持余极限，则 $\operatorname{Alg}_{\mathcal O}(C)$ 在广泛条件下仍是 presentable，并且自由-遗忘伴随存在。

**定义 22.14.** 对 $E_1$-代数 $A$，可定义左 $A$-模 $\infty$-范畴 $\operatorname{LMod}_A(C)$、右模和双模。若 $C=\mathbf{Sp}$，这给出环谱及其模谱的同伦代数。

**定义 22.D.** 设 $C^\otimes$ 为幺半 $\infty$-范畴，$A\in\operatorname{Alg}_{E_1}(C)$。左 $A$-模是对象 $M\in C$ 连同作用

$$
A\otimes M\to M
$$

以及由 $E_1$-代数结构要求的全部同伦相干结合律和单位律。所有左模组成 $\infty$-范畴

$$
\operatorname{LMod}_A(C).
$$

右模 $\operatorname{RMod}_A(C)$ 对偶定义。若 $A,B$ 是 $E_1$-代数，则 $(A,B)$-双模是同时带左 $A$-作用和右 $B$-作用且二者相干交换的对象，记为

$$
{}_{A}\operatorname{BMod}_{B}(C).
$$

**定义 22.E.** 若 $M\in\operatorname{RMod}_A(C)$ 且 $N\in\operatorname{LMod}_A(C)$，其 bar 构造是单纯对象

$$
\operatorname{Bar}_\bullet(M,A,N)
$$

其中

$$
\operatorname{Bar}_n(M,A,N)=M\otimes A^{\otimes n}\otimes N.
$$

面映射由 $M$ 的右作用、$N$ 的左作用和 $A$ 的乘法给出；退化映射由 $A$ 的单位给出。若几何实现存在，定义相对张量积

$$
M\otimes_A N=\left|\operatorname{Bar}_\bullet(M,A,N)\right|.
$$

**外部输入定理 22.F.** 若 $C$ 是 presentable 幺半 $\infty$-范畴且张量积分别保持余极限，则上述 bar 构造的几何实现存在，并给出相对张量积 $M\otimes_A N$。它满足预期的双线性泛性质：从 $M\otimes_A N$ 到对象 $P$ 的映射空间等价于从 $M,N$ 到 $P$ 的 $A$-平衡双线性映射空间。

**例子 22.G.** 若 $C=\mathbf{Sp}$ 且 $A=R$ 为环谱，则 $\operatorname{LMod}_R(\mathbf{Sp})$ 是左 $R$-模谱的稳定 $\infty$-范畴。若 $M$ 为右 $R$-模谱、$N$ 为左 $R$-模谱，则

$$
M\otimes_RN
$$

由双边 bar 构造给出，是导出张量积的谱级提升。

**定义 22.H.** 在合适的幺半 $\infty$-范畴 $C$ 中，Morita $\infty$-范畴 $\operatorname{Mor}_1(C)$ 的对象是 $E_1$-代数；从 $A$ 到 $B$ 的态射对象由 $(A,B)$-双模给出；复合由相对张量积给出：

$$
{}_{A}M_{B}\circ {}_{B}N_{C}
=
{}_{A}(M\otimes_BN)_{C}.
$$

高阶结合律由 bar 构造和相对张量积的泛性质提供。

**定义 22.I.** $E_1$-代数 $A$ 的中心定义为双模范畴中的 endomorphism object

$$
Z(A)=\operatorname{End}_{{}_{A}\operatorname{BMod}_{A}(C)}(A).
$$

在谱或 spaces 等合适环境中，$Z(A)$ 带有 $E_2$-代数结构。更一般地，$E_n$-代数的中心带有 $E_{n+1}$ 型结构。

**外部输入定理 22.J（高阶 Deligne 型定理）.** 对合适的 $E_n$-代数 $A$，其 Hochschild cochains 或中心对象自然带有 $E_{n+1}$-代数结构。$n=1$ 时，这说明 $E_1$-代数的中心是 $E_2$-代数。

**定义 22.K.** 设 $A$ 是 $E_n$-代数，$M$ 是 framed $n$-流形。因子化同调 $\int_M A$ 是把 $A$ 沿 $M$ 中小 $n$-圆盘配置作局部到整体粘合得到的对象；形式上可写为余极限

$$
\int_M A\simeq \operatorname*{colim}_{(U\hookrightarrow M)\in\operatorname{Disk}_n/M} A(U),
$$

其中 $A(U)$ 由 $U$ 的圆盘分量数和 $E_n$-代数结构确定。

**例子 22.L.** 若 $A$ 是 $E_1$-代数，则圆周上的因子化同调恢复 Hochschild homology：

$$
\int_{S^1}A\simeq HH(A).
$$

这可看作把 bar/cyclic bar 构造几何化。

**外部输入定理 22.M.** 因子化同调满足 excision：若流形 $M$ 沿带领的公共部分分解，则 $\int_MA$ 可由相应两部分的因子化同调经相对张量积粘合得到。该定理是因子化同调的核心计算原则。

**定义 22.N.** 在 Morita $\infty$-范畴中，$E_1$-代数 $A$ 与 $B$ 称为 Morita 等价，若存在双模

$$
{}_{A}M_{B},\qquad {}_{B}N_{A}
$$

以及等价

$$
M\otimes_BN\simeq A,\qquad
N\otimes_AM\simeq B
$$

分别作为 $(A,A)$-双模和 $(B,B)$-双模成立。等价地，$M$ 在 Morita $\infty$-范畴中是可逆 1-态射。

**定义 22.O.** 对称幺半 $(\infty,n)$-范畴中的对象称为 fully dualizable，若它有对偶，且其评价/余评价 1-态射、以及继续出现的高阶 adjoints 一直到 $n$ 层都存在。这个条件是高维拓扑场论中“可赋给一点的对象”的有限性条件。

**外部输入定理 22.P（cobordism hypothesis）.** framed fully extended $n$-dimensional topological field theories with values in a symmetric monoidal $(\infty,n)$-category $\mathcal C$ are classified by fully dualizable objects of $\mathcal C$。更精确地，赋值到点的操作给出 $\infty$-群胚等价

$$
\operatorname{Fun}^{\otimes}(\operatorname{Bord}^{fr}_n,\mathcal C)
\simeq
(\mathcal C^{fd})^{\simeq}.
$$

该定理把 Morita 理论、可对偶性和因子化同调连接到拓扑场论。

**例子 22.Q.** 在二维情形中，有限半单代数或更一般的适当可对偶代数对象给出 fully extended TFT 的点值候选。Morita 等价的代数给出等价的场论数据，因为真正不变量是 Morita $\infty$-范畴中的对象。

## 22.8 Morita 等价、单位双模与具体计算

**定义 22.R.** 对 $E_1$-代数 $A$，其单位双模是 $A$ 本身，左、右 $A$-作用均由乘法

$$
A\otimes A\to A
$$

给出。它记为 ${}_AA_A$，是 Morita $\infty$-范畴中对象 $A$ 的恒等 1-态射。

**命题 22.S（单位律）.** 若 ${}_AM_B$ 是 $(A,B)$-双模，则存在自然等价

$$
A\otimes_A M\simeq M,\qquad M\otimes_BB\simeq M
$$

作为 $(A,B)$-双模成立。

**证明.** 第一式由 bar 构造

$$
\operatorname{Bar}_n(A,A,M)=A\otimes A^{\otimes n}\otimes M
$$

的几何实现给出。增广映射把

$$
a_0\otimes a_1\otimes\cdots\otimes a_n\otimes m
$$

送到 $((a_0a_1)\cdots a_n)m$。由 $A$ 的单位可在单纯方向加入 extra degeneracy；单纯恒等式正是单位律和结合律。因此该增广是几何实现后的等价。第二式同理，使用 $B$ 的右单位和右作用。$\square$

**命题 22.T.** 若 ${}_AM_B$ 与 ${}_BN_A$ 给出 Morita 等价，即

$$
M\otimes_BN\simeq A,\qquad N\otimes_AM\simeq B,
$$

则张量函子

$$
-\otimes_AM:\operatorname{RMod}_A(C)\to\operatorname{RMod}_B(C)
$$

和

$$
-\otimes_BN:\operatorname{RMod}_B(C)\to\operatorname{RMod}_A(C)
$$

互为等价。

**证明.** 对右 $A$-模 $P$，相对张量积的结合等价给出

$$
(P\otimes_AM)\otimes_BN
\simeq
P\otimes_A(M\otimes_BN)
\simeq
P\otimes_AA
\simeq
P,
$$

最后一步是命题 22.S。对右 $B$-模 $Q$ 同理：

$$
(Q\otimes_BN)\otimes_AM
\simeq
Q\otimes_B(N\otimes_AM)
\simeq
Q\otimes_BB
\simeq
Q.
$$

这些等价在模态射中自然，故两个函子互为逆等价。$\square$

**例子 22.U（矩阵代数的 Morita 等价）.** 设 $k$ 为域，$V$ 为非零有限维 $k$-向量空间，$A=\operatorname{End}_k(V)$。取

$$
M={}_AV_k,\qquad N={}_kV^*_A.
$$

则 $A$ 与 $k$ Morita 等价。

**证明.** 第一个复合为

$$
M\otimes_kN=V\otimes_kV^*\to\operatorname{End}_k(V)=A,
$$

其中 $v\otimes\varphi$ 送到秩一算子 $w\mapsto v\varphi(w)$。有限维线性代数说明该映射是向量空间同构，且左右 $A$-作用正是算子的左右复合，因此它是 $(A,A)$-双模同构。

第二个复合为

$$
N\otimes_AM=V^*\otimes_{\operatorname{End}_k(V)}V\to k,\qquad
\varphi\otimes v\mapsto\varphi(v).
$$

该映射与 $A$-平衡关系相容。取基 $e_1,\dots,e_n$ 与对偶基 $e^1,\dots,e^n$，任意 $\varphi\otimes v$ 在平衡张量积中化为标量倍的 $e^1\otimes e_1$，而评价映射把它送到同一标量；故为同构。于是满足定义 22.N 的两个互逆条件。$\square$

**外部输入定理 22.V（smooth/proper 判别）.** 在小的幂等完备稳定 $k$-线性 $\infty$-范畴组成的 Morita $(\infty,2)$-范畴中，fully dualizable objects 可由 smooth 且 proper 的对象刻画。对 dg 范畴模型，这意味着 Hom 复形在 $k$ 上 perfect，并且对角双模作为自张量范畴上的模是 perfect。该判别是现代 Morita 理论中连接有限性、双模对偶和 extended TFT 的核心外部输入。

## 22.9 低维拓扑场论的普通影子

**定义 22.W.** 域 $k$ 上的交换 Frobenius 代数是有限维交换 $k$-代数 $A$，连同线性泛函

$$
\varepsilon:A\to k
$$

使配对

$$
\langle a,b\rangle=\varepsilon(ab)
$$

非退化。

**命题 22.X.** 交换 Frobenius 代数给出普通二维 oriented TFT 的代数数据：圆周取值为 $A$，pair-of-pants 曲面给出乘法 $\mu:A\otimes A\to A$ 和其伴随余乘法 $\Delta:A\to A\otimes A$，圆盘给出单位 $\eta:k\to A$ 与余单位 $\varepsilon:A\to k$。

**证明.** 非退化配对给出同构 $A\cong A^*$。令 $\Delta$ 为乘法 $\mu$ 关于该配对的伴随，即

$$
\langle \mu(a,b),c\rangle
=
\langle a\otimes b,\Delta(c)\rangle
$$

对所有 $a,b,c\in A$ 成立。曲面沿公共边界粘合时，代数上对应用配对收缩相邻的 $A$ 与 $A^*$。pair-of-pants 的两种分解给出结合律

$$
\mu(\mu\otimes1)=\mu(1\otimes\mu),
$$

圆柱分解给出单位律，反向 pair-of-pants 的分解给出余结合律。乘法与余乘法的混合分解正是 Frobenius 恒等式

$$
(\mu\otimes1)(1\otimes\Delta)=\Delta\mu=(1\otimes\mu)(\Delta\otimes1).
$$

因此这些线性映射满足二维 bordism 范畴的生成关系，给出普通二维 oriented TFT 的代数数据。完整分类定理还需证明这些生成关系给出全部曲面关系；这属于低维 cobordism category 的外部拓扑输入。$\square$

## 22.10 普通幺半范畴的低维恢复

**命题 22.Y.** 设 $\mathcal C$ 是普通幺半范畴，并把它视为带离散映射空间的幺半 $\infty$-范畴 $N(\mathcal C)^\otimes$。则 $N(\mathcal C)^\otimes$ 中的 $E_1$-代数对象正是 $\mathcal C$ 中的普通代数对象。

**证明.** $E_1$-代数给出对象 $A$、乘法

$$
\mu:A\otimes A\to A
$$

和单位

$$
\eta:\mathbb 1\to A,
$$

并由 $E_1$-operad 的二维和三维相干单纯形给出结合律与单位律。由于 $N(\mathcal C)^\otimes$ 的映射空间离散，所有同伦相干等式只能是普通等式。因此这些数据正是普通幺半范畴中的 monoid object。反过来，普通代数对象的结合律和单位律给出相应 operad 映射。$\square$

**命题 22.Z.** 强幺半 $\infty$-函子把 $\mathcal O$-代数送到 $\mathcal O$-代数。

**证明.** 设 $F:C^\otimes\to D^\otimes$ 是保持 inert/coCartesian 结构的幺半 $\infty$-函子。若

$$
A:\mathcal O^\otimes\to C^\otimes
$$

是 $\mathcal O$-代数，则复合

$$
\mathcal O^\otimes\xrightarrow{A}C^\otimes\xrightarrow{F}D^\otimes
$$

仍是 $\infty$-operad 映射，因而定义 $D$ 中的 $\mathcal O$-代数。相干性不需重新证明，因为它已经包含在 operad 映射的复合中。$\square$

**例子 22.AA.** 对强对称幺半函子 $\Sigma^\infty_+:\mathcal S\to\mathbf{Sp}$，$E_\infty$-space 被送到 $E_\infty$-ring spectrum。这是“代数结构可沿幺半函子传递”的高阶版本；具体构造依赖该函子的对称幺半性。

## 22.11 本章小结

高阶代数把普通代数对象放入 $\infty$-范畴和 operad 控制的相干体系中。$E_n$-代数记录不同层级的同伦交换性，幺半 $\infty$-范畴为谱、导出几何和拓扑场论提供统一语言。Morita 理论把代数按模范畴和双模组织起来，fully dualizable 条件则是 extended TFT 中的有限性核心。

## 练习

**练习 22.1.** 比较普通幺半范畴中的代数对象与定义 22.9。

**练习 22.2.** 解释为什么 $E_\infty$-代数不是严格交换代数，而是同伦相干交换代数。

**练习 22.3.** 查阅 $\mathbf{Fin}_*$ 的对象和 inert morphism 定义。

**练习 22.4.** 给出环谱可以视为 $E_1$-代数的直观解释。

**练习 22.5.** 阅读 Higher Algebra 中关于 presentable monoidal $\infty$-categories 的假设，列出张量积保持余极限的用途。

**练习 22.6.** 对映射 $f:\langle3\rangle\to\langle2\rangle$，其中 $f(1)=1,f(2)=*,f(3)=1$，写出命题 22.5 的 inert-active 分解。

**练习 22.7.** 说明 inert 态射为什么对应“选取输入槽”，而 active 态射对应“真正的运算”。

**练习 22.8.** 写出 $\rho_2:\langle3\rangle\to\langle1\rangle$，并验证它是 inert。

**练习 22.9.** 解释为什么 Segal 条件允许把 $\mathcal O^\otimes_{\langle n\rangle}$ 的对象看成 $n$ 个颜色的列表。

**练习 22.10.** 在普通环和模的情形下，把定义 22.D 的左模作用退化为通常的映射 $A\times M\to M$。

**练习 22.11.** 写出 $\operatorname{Bar}_0(M,A,N)$、$\operatorname{Bar}_1(M,A,N)$ 和 $\operatorname{Bar}_2(M,A,N)$。

**练习 22.12.** 解释为什么 $M\otimes_A N$ 应该是“平衡”张量积。

**练习 22.13.** 在 Morita $\infty$-范畴中，说明两个双模的复合为什么应由相对张量积给出。

**练习 22.14.** 普通代数中 $Z(A)$ 是哪些元素组成的集合？比较定义 22.I。

**练习 22.15.** 说明 $\int_{S^1}A\simeq HH(A)$ 与 cyclic bar construction 的关系。

**练习 22.16.** 展开 Morita 等价定义，说明为什么 $M\otimes_BN\simeq A$ 和 $N\otimes_AM\simeq B$ 是“互逆”的条件。

**练习 22.17.** 说明 fully dualizable 比 dualizable 强在哪里。

**练习 22.18.** 用 cobordism hypothesis 解释为什么 fully extended TFT 由点的取值决定。

**练习 22.19.** 写出单位双模 ${}_AA_A$ 的左右作用，并说明它为什么是 $(A,A)$-双模。

**练习 22.20.** 在普通环情形中，证明 $A\otimes_AM\cong M$。

**练习 22.21.** 设 $M,N$ 给出 Morita 等价。验证命题 22.T 中两个张量函子的复合在对象上同构于恒等。

**练习 22.22.** 对 $A=M_n(k)$，把 $k^n\otimes_k(k^n)^*$ 到 $A$ 的同构写成矩阵单位。

**练习 22.23.** 解释 proper 条件为什么可理解为 Hom 对象的有限性条件。

**练习 22.24.** 给出交换 Frobenius 代数中 $\Delta$ 由 $\mu$ 和配对唯一决定的理由。

**练习 22.25.** 在二维 TFT 的 Frobenius 代数数据中，说明圆柱对应恒等算子。

**练习 22.26.** 比较普通二维 TFT 中“圆周取值”与 fully extended TFT 中“点取值”的差异。

**练习 22.27.** 在普通幺半范畴中写出代数对象的结合律和单位律图。

**练习 22.28.** 解释为什么映射空间离散时，“同伦相干结合律”退化为普通结合律。

**练习 22.29.** 设 $F:C^\otimes\to D^\otimes$ 为强幺半函子，说明它如何把普通代数对象 $A$ 的乘法送到 $FA$ 的乘法。
