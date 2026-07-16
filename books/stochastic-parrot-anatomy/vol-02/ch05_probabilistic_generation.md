# 第五章 概率核、实现映射与轨迹分布

第四章通过固定选择状态得到了一条唯一 token 轨迹。现在忘掉那串选择输入，只保留每个状态上的下一步条件分布。贯穿案例立刻出现两条可能路径：一条使用第一章的规范 token
$(101,102)$ 表示 `SP404`，另一条使用一个负载为 `SP404` 的合并 token
$201$；后续 token 相同。两条路径可以解码为同一个 $u_\star$，却不是同一 token 轨迹。

为了对“路径概率相同”“最终文本分布相同”或“固定 seed 后运行相同”作出可判真的陈述，需要给配置和事件配上可测结构。本章沿用第四章的操作语义，引入随机核、乘积 $\sigma$-代数和推前测度；一般测度扩张部分作为外部输入精确登记。

## 5.1 可测随机核

设 $(C,\mathcal C)$ 为配置可测空间，$(A,\mathcal A)$ 为事件标签可测空间。

**定义 5.1（带标签转移核）.** 从 $C$ 到 $A\times C$ 的随机核是函数

$$
K:C\times(\mathcal A\otimes\mathcal C)\to[0,1]
$$

满足：

1. 对每个 $c\in C$，$K(c,\cdot)$ 是
   $(A\times C,\mathcal A\otimes\mathcal C)$ 上的概率测度；
2. 对每个 $E\in\mathcal A\otimes\mathcal C$，映射
   $c\mapsto K(c,E)$ 是 $\mathcal C$-可测函数。

若只关心下一状态，取边缘核
$K_C(c,B)=K(c,A\times B)$。终止状态可以吸收化。具体地，设
$F_C\in\mathcal C$ 是可测终止集，并把标签空间扩为不交并
$\overline{A}=A\sqcup\{\mathtt{idle}\}$，记其由
$\mathcal A$ 的嵌入像与 $\{\mathtt{idle}\}$ 生成的 $\sigma$-代数为
$\overline{\mathcal A}$。令
$j:A\times C\to\overline{A}\times C$ 为自然嵌入。对
$E\in\overline{\mathcal A}\otimes\mathcal C$ 定义

$$
K^{\mathrm{abs}}(c,E)=
\begin{cases}
\delta_{(\mathtt{idle},c)}(E),&c\in F_C,\\
K(c,j^{-1}(E)),&c\notin F_C.
\end{cases}
$$

映射 $c\mapsto(\mathtt{idle},c)$、嵌入 $j$ 与 $F_C$ 均可测，故两个分支对 $c$ 可测，上式仍为从 $C$ 到
$\overline{A}\times C$ 的随机核。吸收化让所有轨迹具有统一长度，同时必须在观察时删除或另标填充自环，以免把它误算成终止后的新事件。

在可数离散特例中，$\mathcal C=2^C$、$\mathcal A=2^A$，核可写成质量函数
$K(c,a,c')\ge0$，且

$$
\sum_{a\in A}\sum_{c'\in C}K(c,a,c')=1.
$$

## 5.2 有限轨迹测度

给定初始概率测度 $\mu_0$ 于 $(C,\mathcal C)$。长度 $n$ 的路径空间为

$$
\Omega_n=C\times(A\times C)^n
$$

并带乘积 $\sigma$-代数
$\mathcal F_n=\mathcal C\otimes
(\mathcal A\otimes\mathcal C)^{\otimes n}$。

**外部输入 5.A（有限核迭代）.** 概率测度与随机核可逐次积分，唯一得到
$(\Omega_n,\mathcal F_n)$ 上概率测度 $\mathbb P_n$。对可测矩形
$B_0\times(E_1\times B_1)\times\cdots\times(E_n\times B_n)$，其值为

$$
\int_{B_0}\mu_0(dc_0)
\int_{E_1\times B_1}K(c_0,d(a_1,c_1))
\cdots
\int_{E_n\times B_n}K(c_{n-1},d(a_n,c_n)).
$$

该构造的存在与唯一性是随机核积分的标准结果；一般证明使用单调类论证。本书在可数离散情形内证明归一化，在一般情形引用该结果。

若 $C,A$ 可数离散，则单条路径
$\omega=(c_0,a_1,c_1,\ldots,a_n,c_n)$ 的质量为

$$
p_n(\omega)=\mu_0(c_0)
\prod_{t=1}^nK(c_{t-1},a_t,c_t).
$$

**定理 5.2（离散有限路径质量归一化）.** 若 $C,A$ 可数离散，$\mu_0$ 是概率质量函数且 $K$ 满足定义 5.1，则

$$
\sum_{\omega\in\Omega_n}p_n(\omega)=1.
$$

**证明.** 所有项非负，故 Tonelli 定理允许任意次序求和。先对
$(a_n,c_n)$ 求和，核归一性给出

$$
\sum_{a_n,c_n}K(c_{n-1},a_n,c_n)=1.
$$

依次消去第 $n,n-1,\ldots,1$ 步因子，最后剩
$\sum_{c_0}\mu_0(c_0)=1$。证毕。

## 5.3 可测观察与推前分布

固定观察可测空间 $(Y,\mathcal Y)$，并令

$$
O:(\Omega_n,\mathcal F_n)\to(Y,\mathcal Y)
$$

为可测函数。其输出分布是推前测度

$$
\mathbb P_Y=\mathbb P_n\circ O^{-1},
\qquad
\mathbb P_Y(B)=\mathbb P_n(O^{-1}(B)),
\quad B\in\mathcal Y.
$$

$O$ 的可测性确保 $O^{-1}(B)\in\mathcal F_n$，所以上式有定义。若路径空间本身取可数离散 $\sigma$-代数，例如 $C,A$ 均可数离散，则从 $\Omega_n$ 到任意可测空间的每个函数都可测。仅仅假设 $Y$ 可数离散并不够：在一般
$(\Omega_n,\mathcal F_n)$ 上，仍须检查每个点逆像是否属于
$\mathcal F_n$。对带 Borel $\sigma$-代数的文本评分、时间或嵌入空间，可测性同样不能省略。

token 序列、Unicode 文本、最终任务状态和审计标签是不同的观察函数。若解码是部分函数，可把观察值域扩为
$Y\sqcup\{\operatorname{DecodeError}\}$ 使其总化，或把分布限制到解码成功事件并明确条件化。

现在把章首的两条路径写完整。扩充玩具 tokenizer，使 $201$ 的负载为
`SP404`，并令

$$
v^{(a)}=(101,102,103,104,105,106,107),
\qquad
v^{(b)}=(201,103,104,105,106,107).
$$

两者都属于 admissible domain，且

$$
\operatorname{Dec}_{\Theta_\star}(v^{(a)})
=\operatorname{Dec}_{\Theta_\star}(v^{(b)})=u_\star.
$$

构造一个有限离散核：初始配置以概率 $0.6$ 进入路径 $a$ 的首个 token 状态，以概率 $0.4$ 进入路径 $b$ 的首个 token 状态；进入任一分支后，余下转移及 EOS 都以概率 $1$ 发生。由定理 5.2，两条完整路径的质量分别为 $0.6$ 与 $0.4$，总和为 $1$。若 $O_{\mathrm{tok}}$ 观察 token 序列，则推前分布有两个原子；若 $O_{\mathrm{text}}$ 解码为 Unicode，则

$$
\mathbb P\circ O_{\mathrm{text}}^{-1}=\delta_{u_\star}.
$$

这给出一个完整反例：文本几乎处处相同，不表示 token 路径几乎处处相同。

## 5.4 无限轨迹与可变终止时间

无限路径空间为

$$
\Omega_\infty=C\times(A\times C)^{\mathbb N}
$$

并带圆柱集合生成的乘积 $\sigma$-代数。

**外部输入 5.B（Ionescu--Tulcea 扩张）.** 给定初始概率测度以及一列从既往有限历史到下一坐标的可测随机核，存在唯一的无限乘积路径概率测度，其每个有限维边缘由相应核迭代给出。定义 5.1 的齐次 Markov 核是该定理的特例。

本书使用该定理构造 $\mathbb P_\infty$，不重证其测度扩张部分。只给有限维分布而不检查一致性，不能无条件宣称无限轨迹测度已经存在；由同一核递推得到的有限维分布满足所需一致性。

可变长度输出可用停止时间
$T:\Omega_\infty\to\mathbb N\cup\{\infty\}$ 建模；这里“停止时间”包含条件

$$
\{T\le n\}\in\mathcal F_n
\quad\text{对每个 }n\in\mathbb N.
$$

若定义在 $\{T<\infty\}$ 上的停止时观察对该事件的迹
$\sigma$-代数可测，则可把值域扩展一个
$\operatorname{Nontermination}$ 值得到总可测观察；也可在
$\mathbb P(T<\infty)>0$ 时条件化到终止事件。两种构造的样本空间与概率质量不同，不能混写。

## 5.5 随机核与实现映射

随机核描述条件分布，不指定如何消费随机输入。若下一步空间
$A\times C$ 是标准 Borel 空间，则可使用下列外部输入。

**外部输入 5.C（随机化引理）.** 存在
$(\mathcal C\otimes\mathcal B([0,1]))$-可测映射

$$
G:C\times[0,1]\to A\times C
$$

使对所有 $c\in C$ 与
$E\in\mathcal A\otimes\mathcal C$，

$$
\lambda\{u:G(c,u)\in E\}=K(c,E),
$$

其中 $\lambda$ 是 $[0,1]$ 上 Lebesgue 概率测度。对任意非标准 Borel 值域，本书不声称单一均匀变量实现必然存在。

称 $G$ 为核的**实现映射**。若
$U_1,U_2,\ldots$ 独立且均匀，递归式

$$
(A_{t+1},C_{t+1})=G(C_t,U_{t+1})
$$

实现核轨迹。固定完整初态、$G$ 与随机输入序列
$u_1,u_2,\ldots$ 后，该递归是确定的。不同实现映射可以实现同一核，却在相同 $u_t$ 上给出不同轨迹；这正是“同分布”不等于“同耦合实现”。

对上面的二分支核，可以取一个实现映射在 $U_1<0.6$ 时选择路径 $a$，否则选择路径 $b$，后续忽略随机输入并按对应分支确定前进。若本次 $U_1=0.73$，实现产生 $v^{(b)}$；固定 $0.73$ 后再无随机选择。换一个仍以长度 $0.6$ 的可测集合实现路径 $a$ 的映射，核分布不变，同一个数 $0.73$ 却可能落入另一分支。

## 5.6 seed 与伪随机实现

实际 seed 通常先经确定性 PRNG 映射

$$
R:\mathsf{Seed}\to[0,1]^{\mathbb N}
$$

或某个有限精度离散序列，再由实现映射消费。固定 seed 只在同时固定下列对象时确定选择输入：

- PRNG 算法、版本和种子派生；
- 浮点与边界比较规则；
- 每一步消费随机数的数量和顺序；
- 并发调度、工具响应及所有其他转移输入。

seed 本身既不是随机核，也不是抽象分布中的一次样本路径。只有给 seed 指定概率分布后，才可谈其经 $R$ 和 $G$ 推前所得的路径分布；该分布也未必等于理想核。

## 5.7 相同输出边缘不决定轨迹

**命题 5.3（输出边缘不足以确定轨迹律）.** 存在两个路径概率测度，其同一最终输出观察的推前测度相同，但中间状态边缘不同。

**证明.** 取离散路径空间。系统一从初态确定转到中间状态 $a$，系统二确定转到 $b\ne a$，随后都确定转到最终输出值 $0$ 的终态。以最终值为观察时，两者推前测度都是 $\delta_0$；以第一步状态为观察时，推前分别为 $\delta_a$ 与 $\delta_b$。证毕。

因此最终文本的经验频率不能唯一恢复内部状态、工具路径或随机数消费方式。

## 5.8 终止概率

令 $\mathcal F_n$ 为前 $n$ 步生成的历史 $\sigma$-代数，$T$ 为终止时间，
$A_n=\{T>n\}\in\mathcal F_n$。

**命题 5.4（统一条件终止下界）.** 若存在 $\varepsilon>0$，使对所有 $n\ge0$，

$$
\mathbb P(T=n+1\mid\mathcal F_n)\ge\varepsilon
\quad\text{几乎处处地在 }A_n\text{ 上成立},
$$

则 $\mathbb P(T=\infty)=0$，且
$\mathbb P(T>n)\le(1-\varepsilon)^n$。

**证明.** 在 $A_n$ 上，下一步仍未终止的条件概率至多
$1-\varepsilon$。由条件期望塔式性质，

$$
\begin{aligned}
\mathbb P(A_{n+1})
&=\mathbb E[
 \mathbf 1_{A_n}
 \mathbb P(A_{n+1}\mid\mathcal F_n)]\\
&\le(1-\varepsilon)\mathbb P(A_n).
\end{aligned}
$$

从 $\mathbb P(A_0)\le1$ 归纳得
$\mathbb P(A_n)\le(1-\varepsilon)^n$。事件 $A_n$ 递减且
$\{T=\infty\}=\bigcap_nA_n$，由概率测度从上连续性，

$$
\mathbb P(T=\infty)
=\lim_{n\to\infty}\mathbb P(A_n)=0.
$$

证毕。

“每一步终止概率都正”不足以得到统一下界；正概率可以衰减得足够快，使永不终止仍有正概率。

## 5.9 概率观察等价

给定两系统在每个上下文 $K_0\in\mathcal K$ 下的路径测度
$\mathbb P_{K_0}$、$\mathbb Q_{K_0}$ 和共同可测观察
$O_{K_0}$，定义概率观察等价为

$$
\forall K_0\in\mathcal K,\quad
\mathbb P_{K_0}\circ O_{K_0}^{-1}
=
\mathbb Q_{K_0}\circ O_{K_0}^{-1}.
$$

这比第三章只比较可能观察集合更强：相同支持集但概率质量不同的系统不满足该等价。

贯穿案例由此有了三种彼此兼容的描述：核给出两条 token 路径的条件分布；实现映射把均匀输入变成具体分支；固定 $U_1=0.73$ 后又回到第四章式的确定轨迹。无论走哪条分支，文本观察都是 $u_\star$，但这仍只涵盖模型内部的生成。句中的“已取消”和“已写入”来自外部世界；下一章把查询与文件写入接到生成器上，并指出为什么一个高概率文本不能代替工具提交证据。

## 练习

**练习 5.1.** 为两个内容 token、EOS 和吸收终态写出离散带标签转移核，并核对每行归一化。

**练习 5.2.** 证明吸收化保持核归一化，并说明静默自环为何必须从可见观察中删除。

**练习 5.3.** 构造相同 Bernoulli 边缘、不同联合耦合的两个二元输出对。

**练习 5.4.** 构造每一步条件 EOS 概率都正、但永不终止概率为正的过程，并严格证明无限乘积为正。

**练习 5.5.** 分别说明温度、实现映射、独立均匀随机输入和 seed 在本章模型中改变哪一个对象。
