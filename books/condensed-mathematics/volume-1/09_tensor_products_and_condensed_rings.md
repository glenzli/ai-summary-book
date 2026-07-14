# 第九章：张量积与凝聚环

对凝聚阿贝尔群 $A,B$，公式 $S\mapsto A(S)\otimes_{\mathbb Z}B(S)$ 很自然，
却通常只给出预层：张量积不自动保留覆盖等化子。若直接把它当作凝聚对象，得到的
“乘法”会在粘合处失效。正确构造必须先逐点张量，再 sheafification，并通过双线性
态射的泛性质证明所得对象不依赖这一具体表示。

第四章提供阿贝尔 sheaf 范畴，第七章的自由对象则给出可操作的生成元。借助普通阿贝尔
群张量积的泛性质，我们将验证 $\mathbf{CondAb}$ 的对称幺半结构、辨认单位对象，并
把凝聚环定义为其中的交换环对象。这样得到的乘法既保留拓扑环给出的例子，也能在
下一章内部定义模、相对张量积与自由模。

## 9.1 Sheaf 范畴中的张量积

设 $A,B\in\mathbf{CondAb}$。先在预层层面定义

$$
P_{A,B}(S)=A(S)\otimes_{\mathbb Z}B(S),
\qquad S\in\mathbf{CHaus}.
$$

这给出阿贝尔群值预层

$$
P_{A,B}:\mathbf{CHaus}^{\operatorname{op}}\to\mathbf{Ab}.
$$

它一般不一定是 sheaf。因此需要 sheafification。

**定义 9.1.** 凝聚阿贝尔群 $A$ 与 $B$ 的张量积定义为

$$
A\otimes B=a(P_{A,B}),
$$

其中 $a$ 是阿贝尔群值 sheafification。

**注 9.2.** 这个定义与一般 sheaf of abelian groups 的张量积一致。张量积不是简单逐点张量；逐点张量后还要 sheafification。

## 9.2 泛性质

**定义 9.3.** 设 $A,B,C\in\mathbf{CondAb}$。一个双加性态射

$$
A\times B\to C
$$

是指对每个 $S$ 给出双加性映射

$$
A(S)\times B(S)\to C(S),
$$

并且这些映射关于 $S$ 自然。

**命题 9.4.** 张量积 $A\otimes B$ 表示双加性态射。即有自然双射

$$
\operatorname{Hom}_{\mathbf{CondAb}}(A\otimes B,C)
\cong
\operatorname{Bilin}(A,B;C).
$$

**证明.** 由 sheafification 的泛性质，

$$
\operatorname{Hom}_{\mathbf{CondAb}}(a(P_{A,B}),C)
\cong
\operatorname{Hom}_{\operatorname{PSh}(\mathbf{CHaus};\mathbf{Ab})}(P_{A,B},C).
$$

右侧逐点使用普通阿贝尔群张量积泛性质，等价于给出自然的双加性映射

$$
A(S)\times B(S)\to C(S)
$$

对所有 $S$ 成立。证毕。

## 9.3 单位对象

**定义 9.5.** 记 $\mathbb Z$ 为离散拓扑阿贝尔群，$\underline{\mathbb Z}$ 为对应凝聚阿贝尔群。

**命题 9.6.** $\underline{\mathbb Z}$ 是 $\mathbf{CondAb}$ 中张量积的单位对象：

$$
\underline{\mathbb Z}\otimes A\simeq A.
$$

**证明.** 由命题 9.4，给出态射

$$
\underline{\mathbb Z}\otimes A\to C
$$

等价于给出双加性态射

$$
\underline{\mathbb Z}\times A\to C.
$$

由于 $\underline{\mathbb Z}$ 是整数对象，这等价于给出加性态射 $A\to C$。这些双射关于 $C$ 自然；由 Yoneda 引理，$\underline{\mathbb Z}\otimes A$ 与 $A$ 表示同一个 Hom 函子，因而存在唯一自然同构 $\underline{\mathbb Z}\otimes A\simeq A$。证毕。

**注 9.7.** 严格证明可在预层层面使用普通同构 $\mathbb Z\otimes M\simeq M$，再 sheafification。

## 9.4 对称幺半结构

**定理 9.8.** $\mathbf{CondAb}$ 配备张量积 $\otimes$ 和单位对象 $\underline{\mathbb Z}$，成为对称幺半范畴。

**证明说明.** 交换律、结合律和单位约束均来自预层层面的阿贝尔群张量积对应约束，再经 sheafification 得到。由于 sheafification 是左伴随并保持余极限，这些自然同构满足相应 coherence 条件。完整细节属于 sheaf of modules 的标准构造。证毕。

本书后续默认使用该对称幺半结构。

## 9.5 凝聚环

**定义 9.9.** 凝聚环（condensed ring）是 $\mathbf{CondAb}$ 中的交换环对象。也就是说，它是凝聚阿贝尔群 $R$，配备乘法和单位

$$
\mu:R\otimes R\to R,
\qquad
\eta:\underline{\mathbb Z}\to R,
$$

满足结合律、交换律和单位律。

等价地，凝聚环可以理解为 $\mathbf{CHaus}$ 上的环值 sheaf：

$$
R:\mathbf{CHaus}^{\operatorname{op}}\to\mathbf{Ring}
$$

其底层阿贝尔群值函子是凝聚阿贝尔群。

**注 9.10.** 这里 $\mathbf{Ring}$ 默认指交换含幺环范畴。若需要非交换环，应另作说明。

## 9.6 拓扑环给出的凝聚环

**定义 9.11.** 设 $R$ 是拓扑环。定义

$$
\underline R(S)=\operatorname{Cont}(S,R),
\qquad S\in\mathbf{CHaus}.
$$

点态加法、乘法和单位使 $\underline R(S)$ 成为环。

**命题 9.12.** 若 $R$ 是拓扑环，则 $\underline R$ 是凝聚环。

**证明.** 第三章已证明底层集合值函子是 sheaf，第四章已说明拓扑阿贝尔群给出凝聚阿贝尔群。由于 $R$ 的乘法

$$
R\times R\to R
$$

和单位映射 $*\to R$ 连续，它们逐点给出自然的乘法和单位。环公理在每个 $S$ 上逐点成立。因此 $\underline R$ 是凝聚环。证毕。

**例 9.13.** 离散交换环 $R$ 给出凝聚环 $\underline R$。对紧 Hausdorff 空间 $S$，

$$
\underline R(S)=\operatorname{Cont}(S,R_{\operatorname{disc}})
$$

是局部常值 $R$-值函数环。

## 9.7 张量积与自由对象

若 $S\in\mathbf{CHaus}$，自由凝聚阿贝尔群 $\mathbb Z[\underline S]$ 满足

$$
\operatorname{Hom}_{\mathbf{CondAb}}(\mathbb Z[\underline S],A)\cong A(S).
$$

张量积与自由对象有如下关系。

对凝聚环 $R$，定义

$$
R[\underline S]=R\otimes \mathbb Z[\underline S].
$$

它将成为自由 $R$-模。

## 9.8 粘合后的乘法

逐点张量经 sheafification 后表示凝聚双线性态射，因而给出真正的
$A\otimes B$；$\underline{\mathbb Z}$ 由相同泛性质成为单位。凝聚环随之成为
$\mathbf{CondAb}$ 中的交换环对象，拓扑环的连续运算则产生基本例子。最后的公式
$R[\underline S]=R\otimes\mathbb Z[\underline S]$ 已经预示了模论：下一章将证明
它表示 $S$ 上取值，并在 $S$ 极不连通时成为投射 $R$-模。

## 练习

**练习 9.1.** 证明命题 9.4 中的自然双射确实与态射复合相容。

**练习 9.2.** 设 $R$ 是离散环，$S$ 是有限离散空间。证明

$$
\underline R(S)\cong R^S.
$$

**练习 9.3.** 写出凝聚环定义中的结合律和单位律交换图。

**练习 9.4.** 说明为什么逐点张量 $S\mapsto A(S)\otimes B(S)$ 一般只能先得到预层，而不是自动得到 sheaf。
