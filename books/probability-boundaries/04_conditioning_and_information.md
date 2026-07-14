# 第四章：条件期望、Bayes 公式与信息

设四个等可能样本点上的收益依次为 $0,2,4,6$。不知道样本点时，平均收益为 $3$；
若只获知样本点属于前两个还是后两个，合理的更新分别是 $1$ 与 $5$。这个更新不是
把某个事件代入一个比值，而是把原随机变量压缩成只依赖可见信息的新随机变量，同时
在每个可见事件上保留原来的积分。

条件期望正是这种“保留可见事件上的平均值”的对象。有限划分给出直观计算；一般
$\sigma$-代数上的存在性需要 Radon--Nikodym 定理。建立该对象以后，Bayes 公式、
条件概率和条件独立都可以在同一套信息语言中处理。

## 4.1 子 $\sigma$-代数作为信息

设 $(\Omega,\mathcal F,\mathbb P)$ 是概率空间。子 $\sigma$-代数 $\mathcal G\subseteq\mathcal F$ 表示一组可区分的事件。若观察随机元素 $Y$，则其产生的信息是

$$
\sigma(Y)=\{Y^{-1}(B):B\text{ 在值域中可测}\}.
$$

在有限样本空间中，子 $\sigma$-代数等价于一个划分：同一原子中的样本点无法被该信息区分。一般空间中，“信息”仍由可测事件表达，而不需要赋予观察者心理状态。

## 4.2 条件期望

**外部输入 4.1（Radon--Nikodym 定理，有限全变差符号测度版本）.** 设 $(\Omega,\mathcal G,\mu)$ 是 $\sigma$-有限测度空间，$\nu$ 是 $(\Omega,\mathcal G)$ 上满足 $|\nu|(\Omega)<\infty$ 的符号测度，并且 $\nu\ll\mu$。则存在 $Z\in L^1(\mu)$，使

$$
\nu(G)=\int_GZ\,d\mu,
\qquad G\in\mathcal G,
$$

且 $Z$ 在 $\mu$-几乎处处意义下唯一。来源见 [SOURCES.md](SOURCES.md)。本书只把它用于 $\mu=\mathbb P|_{\mathcal G}$，不重证该测度论定理。

**定义 4.2（条件期望）.** 设 $X\in L^1(\mathbb P)$，$\mathcal G\subseteq\mathcal F$ 为子 $\sigma$-代数。若随机变量 $Z$ 满足：

1. $Z$ 是 $\mathcal G$-可测的；
2. $Z\in L^1$；
3. 对每个 $G\in\mathcal G$，

$$
\int_GZ\,d\mathbb P=\int_GX\,d\mathbb P,
$$

则称 $Z$ 是 $X$ 关于 $\mathcal G$ 的条件期望的一个版本，记为 $\mathbb E[X\mid\mathcal G]$。

**例（有限划分上的条件平均）.** 令 $\Omega=\{1,2,3,4\}$ 等概率，
$X(1),\ldots,X(4)=0,2,4,6$，并令

$$
\mathcal G=\{\varnothing,\{1,2\},\{3,4\},\Omega\}.
$$

定义 $Z=1$ 于 $\{1,2\}$，$Z=5$ 于 $\{3,4\}$。它在 $\mathcal G$ 的两个原子上
为常数，因而 $\mathcal G$-可测。并且

$$
\int_{\{1,2\}}Z\,d\mathbb P=\frac12
=\int_{\{1,2\}}X\,d\mathbb P,
\qquad
\int_{\{3,4\}}Z\,d\mathbb P=\frac52
=\int_{\{3,4\}}X\,d\mathbb P.
$$

空集与全空间上的等式随之成立，所以 $Z$ 是 $\mathbb E[X\mid\mathcal G]$ 的一个
版本。这里的数 $1,5$ 是各信息原子内部按原概率重新归一化后的平均值。

有限例子可以逐原子写出 $Z$，一般空间却未必有可枚举的原子。下一定理把存在问题
转化为一个关于测度密度的问题。

**定理 4.3（条件期望的存在与几乎处处唯一性）.** 对每个 $X\in L^1$ 和子 $\sigma$-代数 $\mathcal G\subseteq\mathcal F$，条件期望存在，且任意两个版本几乎处处相等。

**证明.** 在 $\mathcal G$ 上定义集合函数

$$
\nu(G)=\int_GX\,d\mathbb P.
$$

写 $X=X^+-X^-$。由非负 Lebesgue 积分关于积分区域的可数可加性，$G\mapsto\int_GX^+d\mathbb P$ 与 $G\mapsto\int_GX^-d\mathbb P$ 都是有限正测度；所以 $\nu$ 是二者之差，因而是符号测度。它的全变差满足 $|\nu|(\Omega)\le\mathbb E|X|<\infty$。若 $\mathbb P(G)=0$，则 $\nu(G)=0$，故 $\nu\ll\mathbb P|_{\mathcal G}$。外部输入 4.1 给出 $\mathcal G$-可测的 $Z\in L^1(\mathbb P)$，并且对每个 $G\in\mathcal G$ 满足定义 4.2 的积分恒等式，故存在性成立。

若 $Z_1,Z_2$ 都是版本，令 $D=Z_1-Z_2$。对 $A=\{D>0\}\in\mathcal G$，有

$$
\int_AD\,d\mathbb P=0.
$$

若 $\mathbb P(A)>0$，因为 $A=\bigcup_{n\ge1}\{D\ge1/n\}$，必有某个 $n$ 使 $\mathbb P(D\ge1/n)>0$，从而 $\int_AD\,d\mathbb P\ge n^{-1}\mathbb P(D\ge1/n)>0$，矛盾。因此 $\mathbb P(D>0)=0$。对 $-D$ 重复论证，得到 $\mathbb P(D<0)=0$，故 $Z_1=Z_2$ 几乎处处。证毕。

存在性来自 Radon--Nikodym 密度，唯一性则只使用“在所有可见事件上积分相同”。
后者只能排除正概率集合上的差异，因此结论自然是几乎处处唯一，而不是逐点唯一。

条件期望是等价类，不是逐点唯一函数。若后文在某个零概率点讨论其值，必须额外指定版本。

## 4.3 基本运算

**定理 4.4（线性、保序、全期望与 $L^1$ 收缩）.** 设 $X,Y\in L^1(\mathbb P)$，$a,b\in\mathbb R$。则：

1. $\mathbb E[aX+bY\mid\mathcal G]=a\mathbb E[X\mid\mathcal G]+b\mathbb E[Y\mid\mathcal G]$ 几乎处处；
2. 若 $X\le Y$ 几乎处处，则 $\mathbb E[X\mid\mathcal G]\le\mathbb E[Y\mid\mathcal G]$ 几乎处处；
3. $\mathbb E[\mathbb E[X\mid\mathcal G]]=\mathbb E[X]$；
4. $|\mathbb E[X\mid\mathcal G]|\le\mathbb E[|X|\mid\mathcal G]$ 几乎处处，因而

$$
\|\mathbb E[X\mid\mathcal G]\|_1\le\|X\|_1.
$$

**证明.** 第一项右侧可积且 $\mathcal G$-可测；对任意 $G\in\mathcal G$，积分等于 $\int_G(aX+bY)d\mathbb P$，由唯一性成立。

对第二项，令 $U=\mathbb E[Y\mid\mathcal G]-\mathbb E[X\mid\mathcal G]$。若 $A=\{U<0\}\in\mathcal G$ 有正概率，则

$$
\int_AU\,d\mathbb P=\int_A(Y-X)\,d\mathbb P\ge0,
$$

但 $U<0$ 在 $A$ 上且可积；与定理 4.3 唯一性证明中的分层论证相同，若 $\mathbb P(A)>0$ 则积分严格小于零，矛盾。第三项在定义中取 $G=\Omega$。

最后，由 $-|X|\le X\le|X|$ 以及保序性，

$$
-\mathbb E[|X|\mid\mathcal G]
\le\mathbb E[X\mid\mathcal G]
\le\mathbb E[|X|\mid\mathcal G]
$$

几乎处处成立，故得到绝对值不等式。两边取期望并用第三项，得到 $L^1$ 收缩。证毕。

**定理 4.5（拉出已知因子）.** 若 $X\in L^1$，$Y$ 有界且 $\mathcal G$-可测，则

$$
\mathbb E[XY\mid\mathcal G]
=Y\mathbb E[X\mid\mathcal G]
\quad\text{几乎处处}.
$$

**证明.** 先令 $Y=\mathbf1_G$，$G\in\mathcal G$。右侧 $\mathcal G$-可测；对任意 $H\in\mathcal G$，

$$
\int_H\mathbf1_G\mathbb E[X\mid\mathcal G]d\mathbb P
=\int_{H\cap G}X\,d\mathbb P
=\int_H\mathbf1_GX\,d\mathbb P.
$$

所以由唯一性成立。有限线性组合给出简单 $Y$ 的情形。

对一般有界、$\mathcal G$-可测的实函数 $Y$，存在 $\mathcal G$-可测简单函数 $Y_n$，使 $\|Y_n-Y\|_\infty\to0$ 且 $|Y_n|\le\|Y\|_\infty+1$。由定理 4.4 的 $L^1$ 收缩性，

$$
\begin{aligned}
\|\mathbb E[XY_n\mid\mathcal G]-\mathbb E[XY\mid\mathcal G]\|_1
&\le \|X(Y_n-Y)\|_1\\
&\le \|Y_n-Y\|_\infty\|X\|_1\to0.
\end{aligned}
$$

另一方面，

$$
\|Y_n\mathbb E[X\mid\mathcal G]-Y\mathbb E[X\mid\mathcal G]\|_1
\le\|Y_n-Y\|_\infty\|\mathbb E[X\mid\mathcal G]\|_1\to0.
$$

对每个 $n$，两个待比较序列由简单函数情形几乎处处相等，因此它们在 $L^1$ 中的极限也相等。这两个极限分别是上式左右两侧，结论成立。证毕。

**定理 4.6（塔式法则）.** 若 $\mathcal H\subseteq\mathcal G\subseteq\mathcal F$ 且 $X\in L^1$，则

$$
\mathbb E[\mathbb E[X\mid\mathcal G]\mid\mathcal H]
=\mathbb E[X\mid\mathcal H]
\quad\text{几乎处处}.
$$

**证明.** 左侧 $\mathcal H$-可测。对任意 $H\in\mathcal H\subseteq\mathcal G$，

$$
\int_H\mathbb E[\mathbb E[X\mid\mathcal G]\mid\mathcal H]d\mathbb P
=\int_H\mathbb E[X\mid\mathcal G]d\mathbb P
=\int_HX\,d\mathbb P.
$$

因此左侧满足 $\mathbb E[X\mid\mathcal H]$ 的定义，由唯一性得结论。证毕。

## 4.4 条件概率与有限 Bayes 公式

把条件期望应用于事件的示性函数，就得到随信息变化的条件概率。只有当条件本身是
正概率事件时，它才退化为熟悉的概率比值。

**定义 4.7（条件概率）.** 对 $A\in\mathcal F$，定义

$$
\mathbb P(A\mid\mathcal G)
\coloneqq\mathbb E[\mathbf1_A\mid\mathcal G].
$$

若 $B\in\mathcal F$ 且 $\mathbb P(B)>0$，定义数值条件概率

$$
\mathbb P(A\mid B)=\frac{\mathbb P(A\cap B)}{\mathbb P(B)}.
$$

当 $\mathbb P(B)=0$ 时，该比值未定义。不能把任意指定的数称为由原概率测度唯一决定的 $\mathbb P(A\mid B)$。

**定理 4.8（有限全概率与 Bayes 公式）.** 设 $H_1,\ldots,H_n$ 是 $\Omega$ 的可测划分，且每个 $\mathbb P(H_i)>0$。对事件 $E$：

$$
\mathbb P(E)=\sum_{i=1}^n\mathbb P(E\mid H_i)\mathbb P(H_i).
$$

若另有 $\mathbb P(E)>0$，则

$$
\mathbb P(H_j\mid E)
=\frac{\mathbb P(E\mid H_j)\mathbb P(H_j)}
{\sum_{i=1}^n\mathbb P(E\mid H_i)\mathbb P(H_i)}.
$$

**证明.** 因 $E=\bigsqcup_i(E\cap H_i)$，可数可加性给出

$$
\mathbb P(E)=\sum_i\mathbb P(E\cap H_i)
=\sum_i\mathbb P(E\mid H_i)\mathbb P(H_i).
$$

再由 $\mathbb P(H_j\mid E)=\mathbb P(E\cap H_j)/\mathbb P(E)$，将分子写成 $\mathbb P(E\mid H_j)\mathbb P(H_j)$，分母代入全概率公式。证毕。

Bayes 公式是同一联合分布的代数重排。若先验、似然或观测机制没有指定，公式本身不会生成它们。

## 4.5 正则条件分布

称可测空间 $(S,\mathcal S)$ 为**标准 Borel 空间**，若存在某个 Polish 空间（即其拓扑可由某个完备可分度量诱导）的 Borel 子集 $B$，使 $(S,\mathcal S)$ 与 $(B,\mathcal B(B))$ 之间存在双向都可测的双射。有限或可数集合配备幂集、$\mathbb R^d$ 配备 Borel $\sigma$-代数，都是标准 Borel 空间；任意可测空间则未必如此。

**定义 4.9（正则条件分布）.** 设 $X:(\Omega,\mathcal F)\to(S,\mathcal S)$、$Y:(\Omega,\mathcal F)\to(T,\mathcal T)$ 是随机元素。从 $T$ 到 $S$ 的 Markov 核 $K$ 称为 $X$ 给定 $Y$ 的正则条件分布，若对每个 $A\in\mathcal S$ 与 $B\in\mathcal T$，

$$
\mathbb P(X\in A,Y\in B)
=\int_BK(y,A)\,\mathbb P_Y(dy).
$$

等价地，对每个固定 $A\in\mathcal S$，随机变量 $K(Y,A)$ 是 $\mathbb P(X\in A\mid\sigma(Y))$ 的一个版本。

**外部输入 4.10（标准 Borel 值域上的存在性与版本唯一性）.** 若被条件随机元素 $X$ 的值域 $(S,\mathcal S)$ 是标准 Borel 空间，而 $(T,\mathcal T)$ 是任意可测空间，则定义 4.9 的正则条件分布存在。若 $K,K'$ 都是版本，则存在 $N\in\mathcal T$，满足 $\mathbb P_Y(N)=0$，并且对每个 $y\notin N$，有 $K(y,\cdot)=K'(y,\cdot)$ 作为 $S$ 上的概率测度。这里的标准 Borel 条件落在要构造条件概率测度的值域 $S$ 上；它提供可数决定类，从而可把逐个 $A\in\mathcal S$ 的零测例外集合统一为一个例外集合。若 $S$ 是一般可测空间，存在性可能失败。该结果是 [SOURCES.md](SOURCES.md) 所列 Kallenberg 第三版定理 8.5 的直接应用。

这解释了为什么“给定连续变量 $Y=y$”不能直接用零概率事件比值定义，却可以在适当空间中由条件核给出几乎处处版本。

## 4.6 条件独立

**定义 4.11（给定信息的条件独立）.** 设 $\mathcal A,\mathcal B,\mathcal G\subseteq\mathcal F$ 是子 $\sigma$-代数。称 $\mathcal A$ 与 $\mathcal B$ 在给定 $\mathcal G$ 时条件独立，记为

$$
\mathcal A\perp\!\!\!\perp\mathcal B\mid\mathcal G,
$$

若对每个 $A\in\mathcal A$ 与 $B\in\mathcal B$，

$$
\mathbb E[\mathbf1_A\mathbf1_B\mid\mathcal G]
=\mathbb E[\mathbf1_A\mid\mathcal G]
 \mathbb E[\mathbf1_B\mid\mathcal G]
\quad\text{几乎处处}.
$$

随机元素 $X,Y$ 在给定 $Z$ 时条件独立，是指 $\sigma(X)$ 与 $\sigma(Y)$ 在给定 $\sigma(Z)$ 时条件独立，记作 $X\perp\!\!\!\perp Y\mid Z$。等式中的每个条件期望只有版本意义；定义要求对每一对固定事件等式几乎处处成立。

**命题 4.12（有限状态条件独立的逐层刻画）.** 设 $X,Y,Z$ 取值于有限集合。则 $X\perp\!\!\!\perp Y\mid Z$ 当且仅当对每个满足 $\mathbb P(Z=z)>0$ 的 $z$ 以及所有 $x,y$，

$$
\mathbb P(X=x,Y=y\mid Z=z)
=\mathbb P(X=x\mid Z=z)\mathbb P(Y=y\mid Z=z).
$$

**证明.** $\sigma(Z)$ 的正概率原子正是事件 $\{Z=z\}$。条件期望在该原子上等于相应条件概率。把定义 4.11 应用于单点事件 $\{X=x\}$、$\{Y=y\}$，得到必要性。反之，有限集合的任意事件是单点事件的不交并；对单点分解求和，得到任意 $A\in\sigma(X)$、$B\in\sigma(Y)$ 所需的乘积分解。零概率的 $Z$-原子上的版本值不受约束。证毕。

条件独立与无条件独立均不蕴含对方。若 $X,Y$ 是独立公平比特且 $Z=X\oplus Y$，则 $X\perp Y$，但给定 $Z$ 后二者受奇偶关系约束，并不条件独立。反之，若 $Z$ 是公平比特且 $X=Y=Z$，则给定 $Z$ 后 $X,Y$ 的条件分布均为同一个点质量，因而条件独立；但无条件下 $X=Y$，二者不独立。

## 4.7 条件化不是因果干预

$\mathbb P(Y\in A\mid X=x)$ 描述在联合分布中筛选到 $X=x$ 后的条件规律；$\mathbb P(Y\in A\mid\operatorname{do}(X=x))$ 描述改变生成机制后的规律。二者需要不同对象，第九章给出形式模型和反例。

条件期望把信息组织成可测结构，Bayes 公式和条件独立则是它在事件与联合结构上的
具体表现。下一章改变问题：不再固定一个随机变量，而研究一列随机变量何时以及以何种
意义逼近极限。

## 练习

**练习 4.1.** 在有限样本空间中，证明条件期望在每个 $\mathcal G$-原子上为 $X$ 的概率加权平均。

**练习 4.2.** 若 $X$ 本身 $\mathcal G$-可测且可积，证明 $\mathbb E[X\mid\mathcal G]=X$ 几乎处处。

**练习 4.3.** 若 $X$ 与 $\mathcal G$ 独立且可积，证明 $\mathbb E[X\mid\mathcal G]=\mathbb EX$ 几乎处处。

**练习 4.4.** 给出一个 $\mathbb P(B)=0$ 的事件，说明比值定义无法给出 $\mathbb P(A\mid B)$。

**练习 4.5.** 对医学筛查例子写出先验、灵敏度、特异度和阳性后的后验概率，明确每个条件概率的事件。

**练习 4.6.** 对正文中的两个有限比特例子，逐个条件值验证“无条件独立不蕴含条件独立”和“条件独立不蕴含无条件独立”。
