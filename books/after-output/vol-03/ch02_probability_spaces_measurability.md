# 第二章 概率空间与可测性

概率不是对任意陈述随意附加一个数，而是在给定事件代数上满足可列可加性的测度。样本空间决定可能结果，可测集合决定可判定事件，测度只在这些事件上赋值；三者缺一，条件概率、随机变量和期望都没有确定类型。

从审计角度看，可测性同时规定了记录系统能够观察什么。日志若只保留奇偶，就不能事后恢复骰子点数事件；观察接口造成的信息损失必须在概率计算之前声明，而不能由后续统计模型补写。

## 第一章：概率空间与可测性

掷出一枚六面骰子以后，可以记录点数，也可以只记录奇偶。前一种记录能区分六个
结果，后一种记录只能区分两个结果。两种描述面对的是同一次实验，却允许提出不同的
事件：只保存奇偶时，“点数不超过三”不再是由记录能够判断的事件。

概率模型因此不能只给一个样本点集合。它还必须说明哪些样本点集合可以作为事件，
以及这些事件怎样获得概率。以下构造只使用集合、函数、逆像以及可数并交；这些操作
之所以反复出现，是因为概率极限最终仍要落回事件。

### P1.1 可测空间


**定义 P1.1（$\sigma$-代数）.** 设 $\Omega$ 是集合。其子集族 $\mathcal F\subseteq 2^\Omega$ 称为 $\Omega$ 上的 $\sigma$-代数，若：

1. $\Omega\in\mathcal F$；
2. 若 $A\in\mathcal F$，则 $A^c\in\mathcal F$；
3. 若 $A_1,A_2,\ldots\in\mathcal F$，则 $\bigcup_{n\ge1}A_n\in\mathcal F$。

二元组 $(\Omega,\mathcal F)$ 称为可测空间，$\mathcal F$ 中的集合称为事件。

由 De Morgan 律，$\mathcal F$ 对可数交封闭；取空并可得 $\varnothing\in\mathcal F$。要求 $\sigma$-代数而不是任意子集族，是为了使极限事件仍可被赋予概率。例如

$$
\limsup_{n\to\infty}A_n=\bigcap_{m\ge1}\bigcup_{n\ge m}A_n
$$

表示“无穷多个 $A_n$ 发生”，它由可数并交构成。

**定义 P1.2（生成的 $\sigma$-代数）.** 对 $\mathcal A\subseteq2^\Omega$，定义

$$
\sigma(\mathcal A)
\coloneqq
\bigcap\{\mathcal G:\mathcal G\text{ 是 }\Omega\text{ 上的 }\sigma\text{-代数且 }\mathcal A\subseteq\mathcal G\}.
$$

该交集非空，因为 $2^\Omega$ 属于被交族；任意一族 $\sigma$-代数的交仍是 $\sigma$-代数。因此 $\sigma(\mathcal A)$ 是包含 $\mathcal A$ 的最小 $\sigma$-代数。

**定义 P1.3（Borel $\sigma$-代数）.** 若 $S$ 是拓扑空间，记

$$
\mathcal B(S)\coloneqq\sigma(\{U\subseteq S:U\text{ 开}\}).
$$

特别地，$\mathcal B(\mathbb R)$ 由开区间生成。并非所有实数子集都是 Borel 集；概率模型必须说明采用 Borel $\sigma$-代数、其完备化还是更大的事件族。

**例（只记录骰子的奇偶）.** 令 $\Omega=\{1,2,3,4,5,6\}$，
$A=\{1,3,5\}$。只记录奇偶所产生的事件族是

$$
\sigma(\{A\})=\{\varnothing,A,A^c,\Omega\}.
$$

它确实对补集与可数并封闭，而且是包含 $A$ 的最小此类集合族。相比之下，记录完整
点数时通常取 $2^\Omega$。这说明 $\sigma$-代数是观察分辨率的一部分。

### P1.2 概率测度


**定义 P1.4（概率测度）.** 在可测空间 $(\Omega,\mathcal F)$ 上，函数 $\mathbb P:\mathcal F\to[0,1]$ 称为概率测度，若：

1. $\mathbb P(\Omega)=1$；
2. 对任意两两不交的 $A_1,A_2,\ldots\in\mathcal F$，

$$
\mathbb P\left(\bigcup_{n\ge1}A_n\right)=\sum_{n\ge1}\mathbb P(A_n).
$$

三元组 $(\Omega,\mathcal F,\mathbb P)$ 称为概率空间。

定义只给出归一化与可数可加性。单调性、补集公式和极限连续性都应当从这两项推出；
下一命题完成这一步，也说明为什么无需把这些性质分别写入公理。

**命题 P1.5（概率测度的基本性质）.** 设 $(\Omega,\mathcal F,\mathbb P)$ 是概率空间。则：

1. $\mathbb P(\varnothing)=0$；
2. 若 $A\subseteq B$ 且 $A,B\in\mathcal F$，则 $\mathbb P(A)\le\mathbb P(B)$；
3. $\mathbb P(A^c)=1-\mathbb P(A)$；
4. 对任意 $A_n\in\mathcal F$，
   $\mathbb P(\bigcup_nA_n)\le\sum_n\mathbb P(A_n)$；
5. 若 $A_n\uparrow A$，则 $\mathbb P(A_n)\uparrow\mathbb P(A)$；若 $A_n\downarrow A$，则 $\mathbb P(A_n)\downarrow\mathbb P(A)$。

**证明.** 由 $\Omega=\Omega\sqcup\varnothing\sqcup\varnothing\sqcup\cdots$ 的可数可加性，$1=1+\sum_{n\ge1}\mathbb P(\varnothing)$，故第一项成立。若 $A\subseteq B$，则 $B=A\sqcup(B\setminus A)$，故第二项成立；取 $B=\Omega$ 得第三项。

对第四项令 $B_1=A_1$，$B_n=A_n\setminus\bigcup_{k<n}A_k$。则 $B_n$ 两两不交、$B_n\subseteq A_n$，且 $\bigcup_nB_n=\bigcup_nA_n$，由可数可加性和单调性得结论。

若 $A_n\uparrow A$，令 $C_1=A_1$，$C_n=A_n\setminus A_{n-1}$。则 $A=\bigsqcup_nC_n$ 且 $A_m=\bigsqcup_{n\le m}C_n$，故有限部分和收敛到总和。递减情形对补集使用递增结论与第三项。证毕。

证明的共同机制是把集合改写为不交并，再调用可数可加性。以后遇到极限事件时，
“先不交化、再取极限”会反复出现。

**定义 P1.6（零集与几乎处处）.** 若 $N\in\mathcal F$ 且 $\mathbb P(N)=0$，称 $N$ 为零集。性质 $Q(\omega)$ 几乎处处成立，是指存在零集 $N$，使得对每个 $\omega\in\Omega\setminus N$，$Q(\omega)$ 成立。

“概率为零”不等于“不可能”。在 $[0,1]$ 上的均匀分布中，每个单点概率为零，但某个点必然被取到。概率论中的不可能事件是空集；零概率事件只是在测度意义上可忽略。

**定义 P1.7（完备概率空间与完备化）.** 若对每个 $N\in\mathcal F$，$\mathbb P(N)=0$ 都蕴含 $N$ 的每个子集属于 $\mathcal F$，则概率空间称为完备。注意这里被加入的子集起初未必属于原来的 $\mathcal F$。

令 $\mathcal N$ 为所有可测零集的任意子集所成的族，并定义

$$
\overline{\mathcal F}
=\{A\cup N:A\in\mathcal F,\ N\in\mathcal N\}.
$$

则 $\overline{\mathcal F}$ 是包含 $\mathcal F$ 的 $\sigma$-代数。事实上，可数并由

$$
\bigcup_j(A_j\cup N_j)=\left(\bigcup_jA_j\right)\cup\left(\bigcup_jN_j\right)
$$

封闭；其中 $\bigcup_jN_j$ 包含于可测零集的可数并。对补集，若 $N\subseteq N_0\in\mathcal F$ 且 $\mathbb P(N_0)=0$，则

$$
(A\cup N)^c
=(A^c\setminus N_0)\cup\bigl(N_0\setminus(A\cup N)\bigr),
$$

仍具有“可测集并零集子集”的形式。

若 $E=A\cup N\in\overline{\mathcal F}$，定义 $\overline{\mathbb P}(E)=\mathbb P(A)$；该值与表示无关。事实上，若 $A\cup N=B\cup M$，并分别取可测零集 $N_0\supseteq N$、$M_0\supseteq M$，则

$$
A\mathbin\triangle B\subseteq N_0\cup M_0,
$$

从而 $\mathbb P(A)=\mathbb P(B)$。若 $E_j=A_j\cup N_j$ 两两不交，则 $A_j\subseteq E_j$ 也两两不交，并且

$$
\overline{\mathbb P}\left(\bigsqcup_jE_j\right)
=\mathbb P\left(\bigsqcup_jA_j\right)
=\sum_j\mathbb P(A_j)
=\sum_j\overline{\mathbb P}(E_j).
$$

故 $\overline{\mathbb P}$ 可数可加。还需核对完备性的量词覆盖完成后新出现的零测事件。设 $E=A\cup N\in\overline{\mathcal F}$ 且 $\overline{\mathbb P}(E)=\mathbb P(A)=0$，并取原 $\mathcal F$ 中的零集 $N_0\supseteq N$。则

$$
E\subseteq A\cup N_0,
\qquad
A\cup N_0\in\mathcal F,
\qquad
\mathbb P(A\cup N_0)=0.
$$

所以 $E$ 的任意子集 $H$ 也是某个原可测零集的子集，即 $H\in\mathcal N$；于是 $H=\varnothing\cup H\in\overline{\mathcal F}$，且 $\overline{\mathbb P}(H)=0$。这证明 $(\Omega,\overline{\mathcal F},\overline{\mathbb P})$ 完备。它称为原空间的完备化，并且不改变原事件的概率。

### P1.3 可测映射


事件与概率已经定义，但实验结果往往不是样本点本身，而是样本点经过测量得到的数值
或标签。要使“结果落在某集合中”仍是事件，结果映射必须与两端的事件族相容。

**定义 P1.8（可测映射）.** 设 $(S,\mathcal S)$ 与 $(T,\mathcal T)$ 是可测空间。映射 $f:S\to T$ 称为 $(\mathcal S,\mathcal T)$-可测，若对每个 $B\in\mathcal T$，有 $f^{-1}(B)\in\mathcal S$。

可测性的方向由逆像决定。概率空间上的结果映射只有在可测时，事件“结果落在 $B$ 中”才属于 $\mathcal F$，从而 $\mathbb P(f\in B)$ 才有定义。

**命题 P1.9（可测映射的复合）.** 若 $f:(S,\mathcal S)\to(T,\mathcal T)$ 与 $g:(T,\mathcal T)\to(U,\mathcal U)$ 可测，则 $g\circ f$ 可测。

**证明.** 对任意 $C\in\mathcal U$，由 $g$ 可测知 $g^{-1}(C)\in\mathcal T$；再由 $f$ 可测知
$f^{-1}(g^{-1}(C))=(g\circ f)^{-1}(C)\in\mathcal S$。证毕。

**定理 P1.10（连续映射是 Borel 可测的）.** 设 $S,T$ 是拓扑空间，$f:S\to T$ 连续。则

$$
f:(S,\mathcal B(S))\longrightarrow(T,\mathcal B(T))
$$

可测。

**证明.** 定义

$$
\mathcal G\coloneqq\{B\subseteq T:f^{-1}(B)\in\mathcal B(S)\}.
$$

逆像保持补集和可数并，因此 $\mathcal G$ 是 $T$ 上的 $\sigma$-代数。由连续性，任意开集 $U\subseteq T$ 的逆像 $f^{-1}(U)$ 在 $S$ 中开，故属于 $\mathcal B(S)$；所以所有开集属于 $\mathcal G$。由 Borel $\sigma$-代数的最小性，$\mathcal B(T)\subseteq\mathcal G$。这正是 $f$ 的 Borel 可测性。证毕。

连续性是可测性的充分条件而非必要条件。示性函数 $\mathbf1_A$ 在 $A$ 为 Borel 集时可测，即使它在边界点不连续。

### P1.4 有限模型与建模选择


若 $\Omega$ 有限或可数，常取 $\mathcal F=2^\Omega$。给出非负数 $p(\omega)$ 且 $\sum_{\omega\in\Omega}p(\omega)=1$，即可定义

$$
\mathbb P(A)=\sum_{\omega\in A}p(\omega).
$$

在有限模型中，所有函数都可测，许多技术困难消失。但样本空间仍是模型选择。例如同一次语言模型执行可以用“随机种子集合”“随机比特流集合”“完整硬件轨迹集合”作为不同样本空间；它们支持的概率陈述不同。

回到骰子例子，若六个点数等可能，则完整事件空间上的质量函数为 $p(i)=1/6$；
推到奇偶记录后，两个可见结果的概率均为 $1/2$。下一章把这种“由样本点得到结果、
再把概率推到结果空间”的操作定义为随机变量及其分布。

### 练习


**练习 P1.1.** 证明任意一族 $\sigma$-代数的交仍是 $\sigma$-代数。

**练习 P1.2.** 设 $\Omega=\{1,2,3,4\}$。写出由划分 $\{\{1,2\},\{3,4\}\}$ 生成的 $\sigma$-代数。

**练习 P1.3.** 证明若 $A_n\uparrow A$ 且每个 $\mathbb P(A_n)=0$，则 $\mathbb P(A)=0$。

**练习 P1.4.** 设 $A\in\mathcal B(\mathbb R)$。证明 $\mathbf1_A:\mathbb R\to\mathbb R$ 可测。

**练习 P1.5.** 给出两个不同的有限样本空间，它们都可以描述一次六面骰子实验，但其中一个保留的信息严格更多。

## 观察事件接口

### S8.1 概率空间与随机变量


概率论从三元组开始：

$$
(\Omega,\mathcal F,\mathbb P).
$$

随机变量不是“会变的数”，而是可测映射

$$
X:(\Omega,\mathcal F)\to(E,\mathcal E).
$$

分布是推前测度：

$$
\mathcal L(X)=X_\#\mathbb P.
$$

这套语言直接约束 AI 叙述：若说输出随机，必须说明 $\Omega$ 是随机 seed、采样流、训练数据抽样、用户分布、服务端隐藏状态，还是审计者的信息状态。
