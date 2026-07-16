# 附录 E 操作语义证明内核

一次可审计运行不只留下自然语言结论，还留下状态、事件、观测、提交前缀和外部副作用。本附录固定这些对象的最小数学内核：关系、偏函数、确定性转移、偏序调度、散列碰撞、概率与真值的分离，以及事实描述与规范责任的分离。

## E.1 关系与偏函数

**定义 E.1（关系、右唯一性与定义域）.** 给定集合 $X,Y$，从 $X$ 到 $Y$ 的关系是子集 $R\subseteq X\times Y$。记

$$
R(x,y)\quad\text{表示}\quad (x,y)\in R.
$$

关系 $R$ 的定义域为

$$
\operatorname{dom}(R)=\{x\in X:\exists y\in Y,\ R(x,y)\}.
$$

若对任意 $x\in X$ 与 $y_1,y_2\in Y$，

$$
R(x,y_1)\land R(x,y_2)\Longrightarrow y_1=y_2,
$$

则称 $R$ 右唯一。

**定义 E.2（偏函数）.** 从 $X$ 到 $Y$ 的偏函数写作 $f:X\rightharpoonup Y$，由一个子集 $\operatorname{dom}(f)\subseteq X$ 和一个普通函数

$$
f:\operatorname{dom}(f)\to Y
$$

组成。若 $x\notin\operatorname{dom}(f)$，则 $f(x)$ 未定义。

**命题 E.3（偏函数与右唯一关系等价）.** 给定集合 $X,Y$，偏函数 $X\rightharpoonup Y$ 与右唯一关系 $R\subseteq X\times Y$ 一一对应。

**证明.** 若 $f:X\rightharpoonup Y$ 是偏函数，定义其图像关系

$$
\Gamma_f=\{(x,y)\in X\times Y:x\in\operatorname{dom}(f),\ y=f(x)\}.
$$

若 $\Gamma_f(x,y_1)$ 且 $\Gamma_f(x,y_2)$，则 $y_1=f(x)=y_2$，故 $\Gamma_f$ 右唯一。

反过来，设 $R\subseteq X\times Y$ 右唯一。令 $D=\operatorname{dom}(R)$。对每个 $x\in D$，至少存在一个 $y$ 使 $R(x,y)$；右唯一性保证这样的 $y$ 至多一个。因此可定义函数 $f_R:D\to Y$，令 $f_R(x)$ 为唯一满足 $R(x,f_R(x))$ 的元素。于是 $f_R:X\rightharpoonup Y$ 是偏函数。

最后，$\Gamma_{f_R}=R$，且由 $\Gamma_f$ 反向构造得到的偏函数仍为 $f$。两种构造互为逆。$\square$

**定义 E.4（关系复合）.** 若 $R\subseteq X\times Y$，$Q\subseteq Y\times Z$，定义复合关系 $Q\circ R\subseteq X\times Z$ 为

$$
(Q\circ R)(x,z)\Longleftrightarrow \exists y\in Y,\ R(x,y)\land Q(y,z).
$$

**命题 E.5（关系复合结合律）.** 对关系

$$
R\subseteq W\times X,\quad Q\subseteq X\times Y,\quad P\subseteq Y\times Z,
$$

有

$$
P\circ(Q\circ R)=(P\circ Q)\circ R.
$$

**证明.** 对任意 $w\in W,z\in Z$，

$$
\begin{aligned}
(P\circ(Q\circ R))(w,z)
&\Longleftrightarrow \exists y\in Y,\ (Q\circ R)(w,y)\land P(y,z)\\
&\Longleftrightarrow \exists y\in Y,\exists x\in X,\ R(w,x)\land Q(x,y)\land P(y,z).
\end{aligned}
$$

另一方面，

$$
\begin{aligned}
((P\circ Q)\circ R)(w,z)
&\Longleftrightarrow \exists x\in X,\ R(w,x)\land (P\circ Q)(x,z)\\
&\Longleftrightarrow \exists x\in X,\exists y\in Y,\ R(w,x)\land Q(x,y)\land P(y,z).
\end{aligned}
$$

两个条件只差存在量词顺序，而有限或无限集合上的存在量词可交换，因此二者等价。$\square$

**推论 E.6（偏函数复合结合律）.** 把偏函数视为右唯一关系时，偏函数复合满足结合律。

**证明.** 偏函数的复合图像等于图像关系的复合。由命题 E.5 得结合律；右唯一性保证复合仍对应偏函数。$\square$

## E.2 确定性 LTS、轨迹唯一性与观察等价

**定义 E.7（带标签转移系统）.** 一个带标签转移系统是三元组

$$
\mathcal T=(S,E,\to),
$$

其中 $S$ 是状态集合，$E$ 是事件标签集合，$\to\subseteq S\times E\times S$ 是转移关系。写作

$$
s\xrightarrow e s'
$$

表示 $(s,e,s')\in\to$。

若对任意 $s\in S,e\in E$ 与 $s_1,s_2\in S$，

$$
s\xrightarrow e s_1\land s\xrightarrow e s_2\Longrightarrow s_1=s_2,
$$

则称 $\mathcal T$ 确定。等价地，每个事件标签给出一个偏函数

$$
\delta_e:S\rightharpoonup S.
$$

**定义 E.8（事件词与运行轨迹）.** 对事件词 $w=e_1\cdots e_n\in E^*$，从初态 $s_0$ 出发的运行轨迹是序列

$$
\tau=(s_0,e_1,s_1,\ldots,e_n,s_n)
$$

满足 $s_{i-1}\xrightarrow{e_i}s_i$，$1\le i\le n$。若这样的轨迹存在，称 $w$ 在 $s_0$ 可执行。

**命题 E.9（确定性 LTS 的轨迹唯一性）.** 若 $\mathcal T$ 确定，则对任意初态 $s_0$ 和事件词 $w\in E^*$，从 $s_0$ 执行 $w$ 的轨迹至多一条。特别地，若执行终态存在，则终态唯一。

**证明.** 对 $|w|$ 归纳。

当 $|w|=0$ 时，空词的唯一轨迹是 $(s_0)$。

设结论对长度 $n$ 的词成立。令 $w'=e_1\cdots e_n e_{n+1}$。若有两条从 $s_0$ 执行 $w'$ 的轨迹，则其前 $n$ 步都是从 $s_0$ 执行 $e_1\cdots e_n$ 的轨迹。由归纳假设，前缀轨迹相同，尤其第 $n$ 步状态相同，记为 $s_n$。最后一步分别满足

$$
s_n\xrightarrow{e_{n+1}}s_{n+1},\qquad s_n\xrightarrow{e_{n+1}}s'_{n+1}.
$$

由确定性，$s_{n+1}=s'_{n+1}$。故整条轨迹相同。$\square$

**定义 E.10（观察函数与观察词）.** 给定观察集合 $O$ 和函数 $\operatorname{obs}:S\to O$。若轨迹

$$
\tau=(s_0,e_1,s_1,\ldots,e_n,s_n)
$$

存在，其观察词为

$$
\operatorname{Obs}(\tau)=\operatorname{obs}(s_0)\operatorname{obs}(s_1)\cdots\operatorname{obs}(s_n)\in O^{n+1}.
$$

**定义 E.11（观察等价）.** 在同一确定性 LTS 中，状态 $s,t\in S$ 观察等价，记作 $s\equiv_{\operatorname{obs}}t$，若对每个事件词 $w\in E^*$：

1. $w$ 在 $s$ 可执行当且仅当在 $t$ 可执行；
2. 当二者可执行时，从 $s$ 与 $t$ 出发的唯一轨迹具有相同观察词。

**命题 E.12（观察等价是等价关系）.** 关系 $\equiv_{\operatorname{obs}}$ 是 $S$ 上的等价关系。

**证明.** 自反性直接成立。若 $s\equiv_{\operatorname{obs}}t$，定义中的两个条件关于 $s,t$ 对称，故 $t\equiv_{\operatorname{obs}}s$。若 $s\equiv_{\operatorname{obs}}t$ 且 $t\equiv_{\operatorname{obs}}u$，则对任意 $w$，$w$ 在 $s$ 可执行当且仅当在 $t$ 可执行，也当且仅当在 $u$ 可执行；若可执行，$s$ 与 $t$ 的观察词相同，$t$ 与 $u$ 的观察词相同，所以 $s$ 与 $u$ 的观察词相同。故传递性成立。$\square$

观察等价只比较可观察行为，不要求内部状态相等。两个缓存布局、线程局部变量或日志缓冲区不同的状态，可以在所有允许事件下产生同一观察词。

## E.3 生成前缀单调与步数界

**定义 E.13（前缀序）.** 对 token 词表 $\mathcal V$，若存在 $r\in\mathcal V^*$ 使 $y=xr$，则称 $x$ 是 $y$ 的前缀，记作 $x\preceq y$。

**定义 E.14（生成状态）.** 一个确定性生成器由状态集合 $S$、提交投影 $\operatorname{com}:S\to\mathcal V^*$、预算函数 $B:S\to\mathbb N$ 和偏转移 $\delta:S\rightharpoonup S$ 构成。称 $\delta$ 前缀单调，若每当 $\delta(s)$ 有定义时，

$$
\operatorname{com}(s)\preceq\operatorname{com}(\delta(s)).
$$

称 $\delta$ 严格耗费预算，若每当 $\delta(s)$ 有定义时，

$$
B(\delta(s))<B(s).
$$

称 $\delta$ 单步至多提交 $m$ 个 token，若每当 $\delta(s)$ 有定义时，存在 $r\in\mathcal V^*$ 使

$$
\operatorname{com}(\delta(s))=\operatorname{com}(s)r,\qquad |r|\le m.
$$

**命题 E.15（生成前缀单调）.** 若 $\delta$ 前缀单调，且

$$
s_0,s_1,\ldots,s_n
$$

满足 $s_i=\delta(s_{i-1})$，则

$$
\operatorname{com}(s_0)\preceq\operatorname{com}(s_1)\preceq\cdots\preceq\operatorname{com}(s_n).
$$

**证明.** 每一步由定义给出 $\operatorname{com}(s_{i-1})\preceq\operatorname{com}(s_i)$。前缀关系传递，故链式结论成立。$\square$

**命题 E.16（步数界与长度界）.** 若 $\delta$ 严格耗费预算，则从状态 $s_0$ 出发的任意有限或无限执行最多有 $B(s_0)$ 步。若此外 $\delta$ 单步至多提交 $m$ 个 token，则任意终止状态 $s_n$ 满足

$$
|\operatorname{com}(s_n)|\le |\operatorname{com}(s_0)|+mB(s_0).
$$

**证明.** 每执行一步，预算都是自然数中的严格下降。自然数不存在长度超过初始值 $B(s_0)$ 的严格下降链，因此执行步数不超过 $B(s_0)$。若每步至多新增 $m$ 个 token，则 $n$ 步至多新增 $mn\le mB(s_0)$ 个 token。$\square$

该命题只给结构性上界，不声称实际生成一定达到上界，也不声称 token 前缀携带真值或规范责任。

## E.4 幂等操作

**定义 E.17（幂等）.** 函数 $p:X\to X$ 称为幂等，若

$$
p\circ p=p.
$$

偏函数 $p:X\rightharpoonup X$ 称为幂等，若对每个 $x\in\operatorname{dom}(p)$，$p(x)$ 仍在 $\operatorname{dom}(p)$，且

$$
p(p(x))=p(x).
$$

**命题 E.18（重复执行不改变结果）.** 若 $p:X\to X$ 幂等，则对任意整数 $k\ge 1$，

$$
p^k=p.
$$

偏函数情形下，只要 $p^k(x)$ 有定义，也有 $p^k(x)=p(x)$。

**证明.** 全函数情形对 $k$ 归纳。$k=1$ 显然。若 $p^k=p$，则

$$
p^{k+1}=p\circ p^k=p\circ p=p.
$$

偏函数情形在定义域内同理，且定义 E.17 保证第二次应用后不离开定义域。$\square$

幂等性解释了为什么某些操作适合重试：把目标文件内容设置为给定字节串是幂等的；向日志追加一行通常不是幂等的，除非日志系统以事件标识去重。

## E.5 有限偏序与线性扩张

**定义 E.19（偏序与线性扩张）.** 偏序是集合 $P$ 上满足自反、反对称、传递的关系 $\le$。若 $x\le y$ 且 $x\ne y$，写作 $x<y$。

有限偏序 $(P,\le)$ 的线性扩张是一个全序 $\preccurlyeq$，满足

$$
x\le y\Longrightarrow x\preccurlyeq y.
$$

**引理 E.20（有限非空偏序有极小元）.** 若 $P$ 是有限非空偏序，则存在 $m\in P$，使得不存在 $x\in P$ 满足 $x<m$。

**证明.** 任取 $p_0\in P$。若 $p_0$ 不是极小元，则取 $p_1<p_0$。若 $p_1$ 不是极小元，则取 $p_2<p_1$。如此若永不停止，就得到无限严格下降链

$$
p_0>p_1>p_2>\cdots.
$$

但 $P$ 有限，某个元素必重复。若 $p_i=p_j$ 且 $i<j$，由传递性得 $p_j<p_i$，即 $p_i<p_i$，这与严格小于的定义矛盾。故过程必须停止，停止点即极小元。$\square$

**定理 E.21（有限偏序存在线性扩张）.** 每个有限偏序都存在线性扩张。

**证明.** 对 $|P|$ 归纳。若 $|P|=0$ 或 $|P|=1$，结论显然。

设所有大小小于 $n$ 的有限偏序都有线性扩张，令 $|P|=n$。由引理 E.20，取极小元 $m\in P$。在 $P\setminus\{m\}$ 上限制原偏序。由归纳假设，存在其线性扩张

$$
x_1\preccurlyeq x_2\preccurlyeq\cdots\preccurlyeq x_{n-1}.
$$

把 $m$ 放在最前，得到序列

$$
m\preccurlyeq x_1\preccurlyeq\cdots\preccurlyeq x_{n-1}.
$$

它是全序。若原偏序中 $a\le b$，分三种情况：若 $a=m$，则 $a$ 位于最前；若 $b=m$，由 $a\le m$ 与 $m$ 极小可得 $a=m$；若二者都不等于 $m$，则由归纳构造保持顺序。因此该全序扩张原偏序。$\square$

## E.6 可交换事件、调度不变性与确定性归并

**定义 E.22（事件语义与可交换性）.** 固定状态集合 $S$。每个事件 $e$ 的语义是偏函数

$$
\llbracket e\rrbracket:S\rightharpoonup S.
$$

事件 $e,d$ 在状态集合 $U\subseteq S$ 上可交换，若对每个 $s\in U$，两个复合偏函数

$$
\llbracket d\rrbracket\circ\llbracket e\rrbracket,
\qquad
\llbracket e\rrbracket\circ\llbracket d\rrbracket
$$

在 $s$ 处具有相同的定义性，并且在有定义时满足

$$
\llbracket d\rrbracket(\llbracket e\rrbracket(s))
=
\llbracket e\rrbracket(\llbracket d\rrbracket(s)).
$$

**命题 E.23（相邻可交换事件可交换）.** 若 $e,d$ 在包含相关中间状态的集合上可交换，则任意上下文事件序列 $\alpha,\beta$ 中，把相邻片段 $ed$ 换成 $de$ 不改变最终状态，只要两边执行均有定义。

**证明.** 设执行完前缀 $\alpha$ 后到达状态 $s$。由可交换性，

$$
\llbracket d\rrbracket(\llbracket e\rrbracket(s))
=
\llbracket e\rrbracket(\llbracket d\rrbracket(s)).
$$

因此交换相邻二事件后，后缀 $\beta$ 的起始状态相同。确定性偏函数复合保证后缀最终状态相同。$\square$

**引理 E.24（线性扩张的相邻交换连通性）.** 同一有限偏序的任意两个线性扩张，可以通过有限次交换相邻且不可比的元素相互得到。

**证明.** 对元素个数归纳。设两个线性扩张分别为

$$
L=(a_1,\ldots,a_n),\qquad M=(b_1,\ldots,b_n).
$$

若 $a_1=b_1$，删除该元素并对剩余偏序应用归纳假设。

若 $a_1\ne b_1$，在 $L$ 中找到 $b_1$ 的位置，记为 $a_j=b_1$。因为 $b_1$ 在 $M$ 中排第一，它在偏序中没有严格前驱；否则其前驱必须在 $M$ 中排在它之前。故 $a_{j-1}$ 与 $b_1$ 不可能满足 $a_{j-1}<b_1$。又因 $L$ 是线性扩张，若 $b_1<a_{j-1}$，则 $b_1$ 应排在 $a_{j-1}$ 之前，矛盾。因此 $a_{j-1}$ 与 $b_1$ 不可比，可以交换。反复把 $b_1$ 左移到第一位。随后删除第一元素并应用归纳假设。$\square$

**定理 E.25（可交换事件调度不变性）.** 令 $(P,\le)$ 是有限事件偏序。若每对不可比事件在所有可达中间状态上可交换，则从同一初态出发，任意线性扩张调度得到的最终状态相同。

**证明.** 由定理 E.21，线性扩张存在。由引理 E.24，任意两个线性扩张由相邻不可比元素交换连接。每次交换不改变最终状态，见命题 E.23。有限次交换后最终状态仍相同。$\square$

**推论 E.26（确定性归并）.** 两个或多个局部日志各自保持内部顺序。若跨日志的不可比事件在所有可达中间状态上可交换，则任何保持各局部顺序的归并都会产生同一最终状态。

**证明.** 局部顺序给出有限偏序；合法归并正是该偏序的线性扩张。应用定理 E.25。$\square$

确定性归并不是说字节级日志顺序唯一，而是说在声明的交换条件下，最终语义状态唯一。若两个事件写同一路径且写入内容不同，通常不可交换，不能套用该推论。

## E.7 有限散列非单射

**命题 E.27（有限散列碰撞）.** 若 $X$ 和 $D$ 是有限集合，$|X|>|D|$，则任意函数

$$
h:X\to D
$$

都不是单射。特别地，存在 $x_1\ne x_2$ 使 $h(x_1)=h(x_2)$。

**证明.** 若 $h$ 单射，则不同输入有不同输出，因此 $h(X)$ 至少含 $|X|$ 个元素。但 $h(X)\subseteq D$，所以 $|h(X)|\le |D|$，与 $|X|>|D|$ 矛盾。$\square$

**推论 E.28（定长摘要不能唯一命名任意长内容）.** 若摘要空间 $D$ 有限，而可输入内容集合 $X$ 的大小大于 $|D|$，则摘要函数 $h:X\to D$ 必有碰撞。

**证明.** 直接以摘要函数为命题 E.27 中的函数应用鸽巢原理。$\square$

该结论不否认密码散列在工程上的抗碰撞性；它只说明有限摘要不可能成为所有内容的数学单射。审计中使用摘要时，还需要算法、规范化、长度、上下文和碰撞风险口径。

## E.8 Token 概率不推出真值

**定义 E.29（句子真值模型）.** 令 $\mathsf{Sent}$ 是句子集合。一个真值解释是函数

$$
\nu:\mathsf{Sent}\to\{0,1\},
$$

其中 $1$ 表示真，$0$ 表示假。一个语言生成分布是概率测度

$$
P\in\mathcal P(\mathsf{Sent}).
$$

**命题 E.30（高概率不推出真）.** 对任意 $0<p<1$，存在 $(\mathsf{Sent},\nu,P)$ 和句子 $s\in\mathsf{Sent}$，使得

$$
P(s)>p,\qquad \nu(s)=0.
$$

**证明.** 取 $\mathsf{Sent}=\{s,t\}$。定义 $\nu(s)=0$、$\nu(t)=1$。令

$$
P(s)=\frac{1+p}{2},\qquad P(t)=\frac{1-p}{2}.
$$

则 $P(s)>p$，但 $s$ 为假。$\square$

**推论 E.31（token 概率不是事实核验证书）.** 若一个系统只给出句子或 token 序列的生成概率，而没有把句子连接到证据关系、测量协议或真值解释，则生成概率本身不能推出句子为真。

**证明.** 命题 E.30 给出一个满足全部概率信息、但高概率句子为假的反模型。既然系统没有额外关系排除该反模型，单由生成概率到真值的推理无效。$\square$

该推论不是说概率无用。概率可以描述模型在给定上下文和解码接口下的输出倾向，也可以进入校准、风险和排序协议；它不能单独承担事实证明。

## E.9 描述事实不唯一决定规范责任

**定义 E.32（事实结构与责任规则）.** 一个事实结构为

$$
D=(A,C,O,\operatorname{act},\operatorname{cause}),
$$

其中 $A$ 是行动者集合，$C$ 是条件或上下文集合，$O$ 是结果集合，

$$
\operatorname{act}\subseteq A\times C
$$

记录谁在什么条件下行动，

$$
\operatorname{cause}\subseteq (A\times C)\times O
$$

记录行动与结果之间的描述性因果或贡献关系。

责任规则是函数

$$
\rho:D\times O\to\mathcal P(A),
$$

把事实结构和结果映射为承担某种规范责任的行动者集合。

**命题 E.33（同一事实结构可支持不同责任分配）.** 存在同一个事实结构 $D$、同一个结果 $o\in O$ 和两个责任规则 $\rho_1,\rho_2$，使得

$$
\rho_1(D,o)\ne \rho_2(D,o).
$$

**证明.** 取

$$
A=\{\operatorname{user},\operatorname{operator}\},\quad
C=\{c\},\quad
O=\{o\}.
$$

令

$$
\operatorname{act}=\{(\operatorname{user},c),(\operatorname{operator},c)\},
$$

且两者都对结果有贡献：

$$
\operatorname{cause}=\{((\operatorname{user},c),o),((\operatorname{operator},c),o)\}.
$$

定义责任规则

$$
\rho_1(D,o)=\{\operatorname{user}\},
$$

表示按直接发起规则分配责任；再定义

$$
\rho_2(D,o)=\{\operatorname{operator}\},
$$

表示按系统部署与控制规则分配责任。二者面对完全相同的行动和因果事实，却给出不同责任集合。因此描述事实不唯一决定规范责任。$\square$

**结论 E.34（事实记录与责任判断分层）.** 审计记录可以证明某事件发生、某行动者参与、某工具返回、某文件被写入；但从这些事实到责任、许可、过错、赔偿或署名，需要额外规范规则。缺少规范规则时，责任判断不是事实结构的函数。

## E.10 字节、Unicode 标量序列与 tokenizer

本节把标准规定的事实与书内推论分开。标准规定哪些字节串合法、如何解码以及何为规范化；书内只在这些固定输入之上证明函数性质。

**外部规范输入 E.U1（固定版本的 UTF-8）.** 固定一个 Unicode 版本 $U$，并采用该版本引用的 UTF-8 一致性条款。记

$$
\mathbb B_U\subseteq\{0,\ldots,255\}^*
$$

为该条款接受的合法 UTF-8 字节串集合，记 $\mathbb S_U$ 为该版本 Unicode 标量值的有限序列集合。外部规范给出函数

$$
\operatorname{utf8enc}_U:\mathbb S_U\to\mathbb B_U,
\qquad
\operatorname{utf8dec}_U:\mathbb B_U\to\mathbb S_U,
$$

并规定

$$
\operatorname{utf8dec}_U\circ\operatorname{utf8enc}_U
=\operatorname{id}_{\mathbb S_U},
\qquad
\operatorname{utf8enc}_U\circ\operatorname{utf8dec}_U
=\operatorname{id}_{\mathbb B_U}.
$$

因此“合法字节域与标量序列双射”在本附录中是带版本的外部规范输入，不是由后续 tokenizer 定理证明的事实。非法字节串不在 $\operatorname{utf8dec}_U$ 的定义域中；替换错误、忽略错误或猜测编码是另一个显式接口，不能悄悄并入该双射。

**外部规范输入 E.U2（固定版本的 Unicode 规范化）.** 在同一版本 $U$ 下，外部规范给出总函数

$$
\operatorname{NFC}_U,\operatorname{NFD}_U:\mathbb S_U\to\mathbb S_U
$$

并规定其幂等性：

$$
\operatorname{NFC}_U(\operatorname{NFC}_U(x))=\operatorname{NFC}_U(x),
$$

$$
\operatorname{NFD}_U(\operatorname{NFD}_U(x))=\operatorname{NFD}_U(x).
$$

作为后文反例使用的外部规范见证，还固定

$$
u_c=(\texttt{U+00E9}),
\qquad
u_d=(\texttt{U+0065},\texttt{U+0301}),
$$

并由该版本的规范化表确认

$$
u_c\ne u_d,
\qquad
\operatorname{NFC}_U(u_c)=\operatorname{NFC}_U(u_d).
$$

这些等式及哪些序列被归到同一规范化类，均是外部规范事实。它们不推出 NFC 或 NFD 单射，也不允许省略版本、错误策略或规范化形式。

**定义 E.35（tokenizer 接口与规范像）.** 固定词表 $\mathcal V$。一个在文本域 $X\subseteq\mathbb S_U$ 上的 tokenizer 接口由总函数

$$
\operatorname{Enc}:X\to\mathcal V^*,
\qquad
\operatorname{Dec}:\mathcal V^*\to\mathbb S_U
$$

组成。称它满足文本 round-trip，若

$$
\forall x\in X,
\qquad
\operatorname{Dec}(\operatorname{Enc}(x))=x.
$$

定义编码器的规范像

$$
C=\operatorname{Enc}(X)\subseteq\mathcal V^*.
$$

该定义不假设每个 token 序列都是规范编码，也不假设先解码再编码会在整个 $\mathcal V^*$ 上保持原序列。

**定理 E.36（round-trip 的单射性与规范像双侧逆）.** 若定义 E.35 的 tokenizer 满足文本 round-trip，则：

1. $\operatorname{Enc}$ 在 $X$ 上单射；
2. $\operatorname{Dec}|_C:C\to X$ 良定义；
3. $\operatorname{Enc}:X\to C$ 与 $\operatorname{Dec}|_C:C\to X$ 互为双侧逆，即

$$
(\operatorname{Dec}|_C)\circ\operatorname{Enc}=\operatorname{id}_X,
\qquad
\operatorname{Enc}\circ(\operatorname{Dec}|_C)=\operatorname{id}_C.
$$

**证明.** 先证单射。任取 $x,y\in X$，设

$$
\operatorname{Enc}(x)=\operatorname{Enc}(y).
$$

对等式两边应用 $\operatorname{Dec}$，并使用 round-trip，得到

$$
x=\operatorname{Dec}(\operatorname{Enc}(x))
=\operatorname{Dec}(\operatorname{Enc}(y))=y.
$$

故 $\operatorname{Enc}$ 单射。

再证限制函数良定义。任取 $c\in C$。由 $C=\operatorname{Enc}(X)$，存在 $x\in X$ 使 $c=\operatorname{Enc}(x)$。于是

$$
\operatorname{Dec}(c)
=\operatorname{Dec}(\operatorname{Enc}(x))=x\in X,
$$

故 $\operatorname{Dec}|_C$ 的值确实落在 $X$。

第一条逆等式就是 round-trip。对第二条逆等式，任取 $c\in C$，选择 $x\in X$ 使 $c=\operatorname{Enc}(x)$，则

$$
\operatorname{Enc}((\operatorname{Dec}|_C)(c))
=\operatorname{Enc}(\operatorname{Dec}(\operatorname{Enc}(x)))
=\operatorname{Enc}(x)=c.
$$

故两函数在 $X$ 与规范像 $C$ 之间互为双侧逆。证明没有使用 $c\notin C$ 时的任何性质。$\square$

## E.11 强确定系统的唯一最大轨迹

**定义 E.37（强确定、正常终止与最大轨迹）.** 给定无标签转移系统 $(S,\to)$ 和正常终止状态集合 $F\subseteq S$。称系统强确定，若

$$
s\to t_1\land s\to t_2\Longrightarrow t_1=t_2.
$$

相对于初态 $s_0$，再要求：

1. **终止无后继：** 每个从 $s_0$ 可达的 $f\in F$ 都不存在 $t$ 使 $f\to t$；
2. **非终止进展：** 每个从 $s_0$ 可达的 $s\notin F$ 都存在 $t$ 使 $s\to t$。

轨迹是有限序列 $(s_0,\ldots,s_n)$ 或无限序列 $(s_0,s_1,\ldots)$，相邻状态满足转移关系。有限轨迹最大，当且仅当其末态没有后继；无限轨迹按定义最大。

**定理 E.38（唯一最大轨迹与正常终止）.** 若定义 E.37 的三个条件成立，则从 $s_0$ 出发恰有一条最大轨迹。该轨迹若有限，则其末态属于 $F$；它若到达 $F$，就在首次到达处终止，不能继续延长。

**证明.** 强确定性使转移关系在可达状态上对应一个偏函数 $\delta:S\rightharpoonup S$。从 $s_0$ 递归定义

$$
s_{n+1}=\delta(s_n)
$$

只要右边有定义。分两种情形。

若存在最小的 $n$ 使 $\delta(s_n)$ 未定义，则构造得到有限轨迹

$$
(s_0,\ldots,s_n).
$$

它因 $s_n$ 无后继而最大。由非终止进展，若 $s_n\notin F$ 就必须有后继，矛盾；故 $s_n\in F$。

若对每个 $n$，$\delta(s_n)$ 都有定义，则递归得到无限轨迹

$$
(s_0,s_1,\ldots),
$$

它按定义最大。

两种情形至少发生一种，且不能同时发生。下面证明唯一性。设 $\tau$ 是任意从 $s_0$ 出发的轨迹。对位置 $i$ 归纳：位置 $0$ 必为 $s_0$；若 $\tau$ 的第 $i$ 个状态等于 $s_i$ 且还有下一步，强确定性迫使下一状态等于唯一的 $\delta(s_i)=s_{i+1}$。因此任何轨迹都是上述递归轨迹的前缀。它若在仍有后继的位置有限停止，就不是最大；所以最大轨迹只能是构造出的整条有限或无限轨迹。

最后，若轨迹到达某个 $f\in F$，终止无后继条件说明下一步不存在，故该处必为有限最大轨迹的末端。$\square$

进展条件不可省略。若某个可达状态既不在 $F$ 中又没有后继，则仍可能有唯一最大轨迹，但它以“卡住”而不是正常终止结束。

## E.12 may、must、上下文与概率观察

**定义 E.39（四种观察口径）.** 令闭系统 $P$ 的最大运行集合为非空集合 $\operatorname{Run}(P)$，每条最大运行 $\pi$ 有终局观察

$$
\operatorname{out}_P(\pi)\in O.
$$

对观察事件 $A\subseteq O$，定义

$$
P\Downarrow_{\mathrm{may}}A
\Longleftrightarrow
\exists\pi\in\operatorname{Run}(P),
\ \operatorname{out}_P(\pi)\in A,
$$

$$
P\Downarrow_{\mathrm{must}}A
\Longleftrightarrow
\forall\pi\in\operatorname{Run}(P),
\ \operatorname{out}_P(\pi)\in A.
$$

may 与 must 观察等价分别定义为：对每个 $A\subseteq O$，两个系统的相应判断同真同假。

再令 $\mathcal C$ 是声明的观察上下文集合，$C[P]$ 表示把系统放入上下文。上下文 may 等价要求

$$
\forall C\in\mathcal C\ \forall A\subseteq O,
\qquad
C[P]\Downarrow_{\mathrm{may}}A
\Longleftrightarrow
C[Q]\Downarrow_{\mathrm{may}}A,
$$

上下文 must 等价把式中的 may 换成 must。上下文集合是定义的一部分；只比较空上下文不能推出对交互上下文等价。

若 $P,Q$ 是概率系统，且 $\mu_{C[P]}$ 是 $O$ 上的概率分布，则概率观察等价定义为

$$
\forall C\in\mathcal C\ \forall A\subseteq O,
\qquad
\mu_{C[P]}(A)=\mu_{C[Q]}(A).
$$

因此四种口径的核心量词依次是“存在运行”“所有运行”“所有声明上下文再量化运行”“所有声明上下文与事件上的概率相等”。

**例 E.40（may 不等于 must 的最小分支反例）.** 取 $O=\{\checkmark,\times\}$。系统 $P$ 从初态有两条最大运行，一条观察为 $\checkmark$，另一条观察为 $\times$。令 $A=\{\checkmark\}$。于是

$$
P\Downarrow_{\mathrm{may}}A,
\qquad
P\not\Downarrow_{\mathrm{must}}A.
$$

一个初态、两个终局分支已经足以分开存在量词与全称量词。若把发散记为独立观察 $\bot$，也可用“成功或发散”两个分支得到同样反例。

**例 E.41（闭观察等价不推出上下文等价）.** 令 $P_0,P_1$ 在空上下文中都只有观察 $\operatorname{idle}$，但各自保存一位内部状态 $0$ 与 $1$。声明一个探针上下文 $C_{?}$，它发送查询事件 `?` 并把返回位作为观察。则空上下文下二者的 may 与 must 判断完全相同，而

$$
\operatorname{out}(C_{?}[P_0])=0,
\qquad
\operatorname{out}(C_{?}[P_1])=1.
$$

取 $A=\{0\}$ 即可区分二者。因此上下文量词不是装饰；它规定观察者能够实施哪些实验。

**命题 E.42（同一可能集仍可有不同概率）.** 存在两个有限概率系统，它们在任何只对终局观察作确定性后处理的上下文中 may 等价且 must 等价，但不概率观察等价。

**证明.** 取 $O=\{a,b\}$。令 $P$ 输出 $a,b$ 的概率分别为 $1/3,2/3$，令 $Q$ 输出 $a,b$ 的概率分别为 $2/3,1/3$。二者的可能观察集都是 $\{a,b\}$。

任取确定性后处理上下文 $C$，它由函数 $g_C:O\to O_C$ 表示。$C[P]$ 与 $C[Q]$ 的可能观察集都等于

$$
g_C(\{a,b\}).
$$

对任意 $A\subseteq O_C$，may 判断等价于该公共可能集与 $A$ 相交；must 判断等价于该公共可能集包含于 $A$。故二者在所有这类上下文中 may 等价且 must 等价。

但在恒等上下文中取事件 $A=\{a\}$，有

$$
\mu_P(A)=\frac13\ne\frac23=\mu_Q(A).
$$

所以二者不概率观察等价。$\square$

该命题也说明：在“每条最大运行都有一个终局观察”的简化模型里，若对所有观察事件量化，则 may 等价与 must 等价都只识别可能观察集；二者的满足判断仍有不同量词。要表达公平性、拒绝发散或调度保证，必须把相应运行性质放进观察或测试定义。

## E.13 有限随机核与几乎处处终止

**定义 E.43（有限离散随机核与路径质量）.** 令 $S$ 是有限非空状态集合。有限离散随机核是函数

$$
K:S\times S\to[0,1]
$$

满足对每个 $s\in S$，

$$
\sum_{t\in S}K(s,t)=1.
$$

给定初始分布

$$
\mu_0:S\to[0,1],
\qquad
\sum_{s\in S}\mu_0(s)=1,
$$

长度 $n$ 的路径

$$
\pi=(s_0,\ldots,s_n)
$$

的质量定义为

$$
\operatorname{wt}(\pi)
=\mu_0(s_0)\prod_{i=0}^{n-1}K(s_i,s_{i+1}),
$$

空乘积取 $1$。

**定理 E.44（固定长度路径质量归一化）.** 对每个 $n\ge0$，全部长度 $n$ 路径的质量之和为 $1$：

$$
\sum_{(s_0,\ldots,s_n)\in S^{n+1}}
\operatorname{wt}(s_0,\ldots,s_n)=1.
$$

**证明.** 对 $n$ 归纳。$n=0$ 时，

$$
\sum_{s_0\in S}\operatorname{wt}(s_0)
=\sum_{s_0\in S}\mu_0(s_0)=1.
$$

设长度 $n$ 的路径质量和为 $1$。每条长度 $n+1$ 的路径唯一写成长度 $n$ 的前缀 $\pi=(s_0,\ldots,s_n)$ 再接一个 $t\in S$，且

$$
\operatorname{wt}(\pi,t)=\operatorname{wt}(\pi)K(s_n,t).
$$

由于 $S$ 有限，可以重排有限和：

$$
\begin{aligned}
\sum_{(s_0,\ldots,s_{n+1})}\operatorname{wt}(s_0,\ldots,s_{n+1})
&=\sum_{\pi\in S^{n+1}}\operatorname{wt}(\pi)
  \sum_{t\in S}K(s_n,t)\\
&=\sum_{\pi\in S^{n+1}}\operatorname{wt}(\pi)\\
&=1.
\end{aligned}
$$

归纳完成。$\square$

**定理 E.45（统一条件终止下界推出几乎处处有限终止）.** 令 $F\subseteq S$ 为终止状态集合，并令

$$
T=\inf\{n\ge0:X_n\in F\}
$$

为首次到达时间，约定从未到达时 $T=\infty$。假设存在固定 $\varepsilon\in(0,1]$，使每个可在终止前以正概率到达的 $s\notin F$ 都满足

$$
K(s,F):=\sum_{t\in F}K(s,t)\ge\varepsilon.
$$

则对每个 $n\ge0$，

$$
\Pr(T>n)\le(1-\varepsilon)^n,
$$

并且

$$
\Pr(T<\infty)=1.
$$

**证明.** 事件 $T>n$ 由前 $n+1$ 个状态均不在 $F$ 的长度 $n$ 路径组成。对任意这样的正质量路径 $\pi=(s_0,\ldots,s_n)$，末态 $s_n$ 在终止前可达，故

$$
\sum_{t\notin F}K(s_n,t)
=1-K(s_n,F)
\le1-\varepsilon.
$$

把所有仍未终止的长度 $n+1$ 路径按长度 $n$ 前缀分组，得到

$$
\begin{aligned}
\Pr(T>n+1)
&=\sum_{\substack{\pi=(s_0,\ldots,s_n)\\s_0,\ldots,s_n\notin F}}
  \operatorname{wt}(\pi)
  \sum_{t\notin F}K(s_n,t)\\
&\le(1-\varepsilon)
  \sum_{\substack{\pi=(s_0,\ldots,s_n)\\s_0,\ldots,s_n\notin F}}
  \operatorname{wt}(\pi)\\
&=(1-\varepsilon)\Pr(T>n).
\end{aligned}
$$

又 $\Pr(T>0)\le1$，归纳得

$$
\Pr(T>n)\le(1-\varepsilon)^n.
$$

由于 $0<\varepsilon\le1$，几何项趋于 $0$。另一方面，事件 $\{T\le n\}$ 随 $n$ 单调增加，且

$$
\{T<\infty\}=\bigcup_{n\ge0}\{T\le n\}.
$$

把首次终止事件写成互不相交并集

$$
\{T<\infty\}=\bigsqcup_{j\ge0}\{T=j\}.
$$

由概率的可列可加性，

$$
\Pr(T<\infty)
=\sum_{j\ge0}\Pr(T=j)
=\lim_{n\to\infty}\sum_{j=0}^{n}\Pr(T=j)
=\lim_{n\to\infty}\Pr(T\le n).
$$

而

$$
\Pr(T\le n)=1-\Pr(T>n)
\ge1-(1-\varepsilon)^n\longrightarrow1.
$$

概率不超过 $1$，故极限等于 $1$，即 $\Pr(T<\infty)=1$。$\square$

统一下界是关键。若第 $n$ 步的条件终止概率为 $\varepsilon_n>0$ 但其总和有限，则无限不终止概率可能为正；逐步“总有一点终止机会”不足以替代统一下界或其他可求和条件。

## E.14 固定边界下的工具乘积系统

**定义 E.46（工具乘积状态与固定组件）.** 令

$$
S=S_0\times S_1\times\cdots\times S_m\times Q
$$

为控制器、$m$ 个工具世界和调度器位置的乘积状态。固定：

1. 外部输入值 $x$，包括本次运行允许读取的工具响应与环境值；
2. 调度函数 $\sigma:Q\to\{0,1,\ldots,m\}$ 及位置更新函数 $\operatorname{next}:Q\to Q$；
3. 每个组件的偏函数实现 $f_i$。$f_i$ 接收整个当前乘积状态和固定输入 $x$，并在有定义时唯一给出组件 $i$ 的新状态、其他受该原子步骤影响的状态以及事件记录。

所有真实副作用、回执和可观察外部世界都必须包含在某个 $S_i$ 中；若把它们留在状态之外，所谓乘积系统并不封闭。

由这些固定项定义一步偏函数 $\Delta_x:S\rightharpoonup S$：在状态 $(s_0,\ldots,s_m,q)$ 选择唯一索引 $i=\sigma(q)$，应用 $f_i$，再把调度位置更新为 $\operatorname{next}(q)$。约定 $\Delta_x$ 未定义表示该封闭系统停止。

**定理 E.47（工具乘积系统的唯一最大轨迹）.** 在定义 E.46 的边界下，从任意初始乘积状态 $z_0$ 出发，$\Delta_x$ 诱导唯一最大轨迹；轨迹可以有限，也可以无限。

**证明.** 固定 $x$ 后，每个 $f_i$ 是偏函数；固定 $\sigma$ 后，每个调度位置只选择一个索引；固定 $\operatorname{next}$ 后，调度位置更新也唯一。因此，对任意乘积状态 $z$，若 $\Delta_x(z)$ 有定义，其值唯一，故 $\Delta_x$ 是偏函数。

从 $z_0$ 开始反复应用 $\Delta_x$。若存在第一个 $n$ 使 $\Delta_x(z_n)$ 未定义，就得到有限轨迹 $(z_0,\ldots,z_n)$，其末态无后继，因而最大。若每一步都有定义，就得到无限轨迹。任意另一条轨迹在第 $0$ 位等于 $z_0$；若前 $n$ 位相同，偏函数性迫使下一位也相同。归纳可知任意轨迹都是所构造轨迹的前缀，故最大轨迹唯一。$\square$

该定理不声称实际部署天然确定。未固定的网络响应、时钟、随机流、并发调度、人工审批或组件版本都会使定义 E.46 的前提失败；它们必须被固定、记录，或改用随机核和非确定语义。

## E.15 幂等键、成功重试与声明世界

**定义 E.48（声明世界等价与幂等键合同）.** 令服务状态为

$$
Z=W\times L,
$$

其中 $W$ 是包含全部声明可观察副作用的世界状态，$L$ 是持久幂等键表。固定总观察投影

$$
d:W\to D,
$$

并定义

$$
(w,L)\equiv_D(w',L')
\Longleftrightarrow d(w)=d(w').
$$

对作用域内的键 $k$ 和规范化请求 $p$，调用语义为偏函数 $T_{k,p}:Z\rightharpoonup Z$。称其满足成功重试合同，若以下条件全部成立：

1. **请求稳定：** 每次重试使用同一作用域、同一键 $k$ 和同一规范化请求 $p$；
2. **原子首次提交：** 第一次成功把 $p$ 的全部声明可观察副作用与条目 $L[k]=(p,r,\operatorname{committed})$ 原子地持久化，其中 $r$ 是逻辑结果；
3. **提交定义：** “成功”指服务端已完成上述持久提交，不是客户端收到确认；
4. **重复去重：** 若 $L[k]=(p,r,\operatorname{committed})$，再次调用不增加任何 $D$ 可观察副作用，保持原条目并返回同一逻辑结果 $r$；
5. **冲突拒绝：** 若 $L[k]$ 已绑定 $p'\ne p$，服务拒绝请求且不改变 $D$ 观察；
6. **覆盖与持久性：** 所有需要声明为“恰好一次”的副作用都在原子去重边界内；在重试窗口内条目不丢失、键不回收、不跨作用域复用；
7. **故障闭包：** 失败尝试要么在首次原子提交之前不改变 $D$，要么已经形成第 2 条所说的成功提交；不存在只写一部分声明世界却不写键表的中间结果。

**定理 E.49（成功后的任意有限重试保持声明世界等价类）.** 若定义 E.48 的合同成立，且某次调用首次成功后状态为 $z_1=(w_1,L_1)$，则对任意 $n\ge1$，在其后进行任意 $n-1$ 次同键同请求重试，只要每次调用返回或按故障闭包结束，所得状态 $z_n=(w_n,L_n)$ 都满足

$$
z_n\equiv_D z_1.
$$

**证明.** 对重试次数归纳。$n=1$ 时，$d(w_1)=d(w_1)$，结论成立。

设第 $n$ 次后的状态满足 $z_n\equiv_D z_1$。由于首次成功已经原子持久化

$$
L_1[k]=(p,r,\operatorname{committed}),
$$

覆盖与持久性保证该绑定在后续状态中仍存在。第 $n+1$ 次若使用同一 $p$，重复去重条件保证不增加或改变任何 $D$ 可观察副作用，所以

$$
d(w_{n+1})=d(w_n)=d(w_1).
$$

请求稳定条件保证这里不会进入冲突支路；冲突拒绝条款只保护误用，不能替代请求稳定。若该次尝试以故障结束，故障闭包排除部分声明副作用：它要么不改变 $D$，要么只能重新落入已提交且去重的情形。因此所有允许分支都满足 $z_{n+1}\equiv_D z_1$。归纳完成。$\square$

结论只关于声明的投影 $D$。若计费、通知、下游队列或审计追加被排除在去重边界之外，它们不受定理保护；若希望这些量也保持“恰好一次”，必须把它们纳入 $W$ 与 $d$，并重新验证合同。

## E.16 完整唯一序号与确定归并

**定义 E.50（完整唯一序号日志）.** 令 $E$ 是含 $N$ 个事件的有限集合。完整唯一序号是双射

$$
q:E\to\{1,\ldots,N\}.
$$

一个到达流是 $E$ 的任意排列；因此每个事件恰好到达一次，事件内容和序号在到达后不变。确定归并器收集全部 $N$ 个事件，验证双射条件，再按递增序号输出

$$
M_q=(e_{q^{-1}(1)},\ldots,e_{q^{-1}(N)}).
$$

**定理 E.51（确定归并与到达顺序无关）.** 对固定事件集合 $E$ 和完整唯一序号 $q$，任意两个到达流经定义 E.50 的归并器都输出同一序列 $M_q$。

**证明.** 因 $q$ 是双射，对每个 $j\in\{1,\ldots,N\}$，存在唯一事件 $e_j=q^{-1}(j)$。完整性保证任意到达流都包含每个 $e_j$；唯一性保证没有另一个事件与它争用序号 $j$。归并器的第 $j$ 个输出位置因此在任何到达流下都只能是 $e_j$。两个输出序列在每个位置相同，故整体相同。$\square$

若序号缺失，归并器无法区分“尚未到达”与“永远不存在”；若序号重复，它不能唯一选择同一位置的事件；若未等待完整集合，早期输出还可能受水位线策略影响。三者都不满足定理前提。

## E.17 provenance 图同构与投影身份

**定义 E.52（有根类型化 provenance 图及其同构）.** 有根类型化 provenance 图是有限结构

$$
G=(V,E,r,\tau_V,\tau_E,\lambda),
$$

其中 $V$ 是顶点集，$E\subseteq V\times V$ 是有向边集，$r\in V$ 是根制品，$\tau_V$ 与 $\tau_E$ 分别给顶点和边赋类型，$\lambda$ 给声明需要保留的属性赋值。

两个图 $G,G'$ 同构，记作 $G\cong G'$，若存在双射 $\phi:V\to V'$ 满足：

1. $\phi(r)=r'$；
2. 对每个顶点 $v$，$\tau_V(v)=\tau'_V(\phi(v))$，且所有声明保留的顶点属性相等；
3. $(u,v)\in E$ 当且仅当 $(\phi(u),\phi(v))\in E'$；对应边的类型及所有声明保留的边属性相等。

**定理 E.53（有根类型化 provenance 图同构是等价关系）.** 关系 $\cong$ 在上述图的集合上自反、对称且传递。

**证明.** 对任意图 $G$，恒等双射 $\operatorname{id}_V$ 固定根，保持每个顶点、边、类型和属性，故 $G\cong G$，自反性成立。

若 $\phi:V\to V'$ 见证 $G\cong G'$，则逆映射 $\phi^{-1}:V'\to V$ 也是双射。由 $\phi(r)=r'$ 得 $\phi^{-1}(r')=r$。$\phi$ 对顶点类型、边邻接、边类型和属性的保持都是双向“当且仅当”或相等式，反向应用即说明 $\phi^{-1}$ 保持这些结构，故 $G'\cong G$，对称性成立。

若 $\phi:G\to G'$ 与 $\psi:G'\to G''$ 都是同构，则复合 $\psi\circ\phi:V\to V''$ 是双射，并且

$$
(\psi\circ\phi)(r)=\psi(r')=r''.
$$

顶点类型和属性先经 $\phi$ 保持、再经 $\psi$ 保持，故经复合仍保持。对任意 $u,v$，

$$
(u,v)\in E
\Longleftrightarrow
(\phi(u),\phi(v))\in E'
\Longleftrightarrow
((\psi\circ\phi)(u),(\psi\circ\phi)(v))\in E''.
$$

边类型和属性同理。因此 $\psi\circ\phi$ 是同构，传递性成立。$\square$

**命题 E.54（相同内容不推出 provenance 同构）.** 存在根制品内容相同、但有根类型化 provenance 图不同构的两个记录。

**证明.** 令两个根制品 $r_1,r_2$ 的内容属性都等于同一字节串 $b$。图 $G_1$ 只有一个制品顶点 $r_1$，没有边。图 $G_2$ 有制品顶点 $r_2$ 和活动顶点 $a$，并有一条类型为 `wasGeneratedBy` 的边 $(r_2,a)$。两图的根内容相同。

但 $|V(G_1)|=1$，$|V(G_2)|=2$，不存在两顶点集之间的双射，因而不存在图同构。故相同内容不推出 provenance 同构。$\square$

**定理 E.55（总投影诱导身份等价关系）.** 给定集合 $R$ 和任意总函数

$$
p:R\to Y,
$$

定义

$$
r\equiv_p r'
\Longleftrightarrow p(r)=p(r').
$$

则 $\equiv_p$ 是 $R$ 上的等价关系。

**证明.** 对任意 $r\in R$，$p(r)=p(r)$，故自反。若 $p(r)=p(r')$，由等号对称性有 $p(r')=p(r)$，故对称。若 $p(r)=p(r')$ 且 $p(r')=p(r'')$，由等号传递性有 $p(r)=p(r'')$，故传递。$\square$

总性不可被默默省略。若投影只在部分记录上有定义，则上述公式首先只在其定义域上给出等价关系；若要覆盖全部记录，必须规定缺失值语义或拒绝不良构记录。

## E.18 记录良构接口与身份关系边界

**定义 E.56（参数化输出记录与良构性接口）.** 固定版本化外部接口参数

$$
\Sigma=(U,\operatorname{utf8enc}_U,\operatorname{utf8dec}_U,
\operatorname{NFC}_U,\operatorname{NFD}_U,
\operatorname{Enc},\operatorname{Dec},
\mathsf{DigestSpec},\mathsf{ProvSchema},\mathsf{ToolSchema}).
$$

其中 Unicode 与 UTF-8 条款来自外部规范输入 E.U1--E.U2；摘要格式、provenance 模式和工具协议若引用行业标准，也都作为带版本的外部规范输入。书内不从名称猜测这些规范的行为。

一个输出记录是元组

$$
R=(b,u,v,c,\tau,\mathbf a,G,\mathbf k,\mathbf q,n),
$$

其中：

- $b$ 是最终字节串，$u$ 是 Unicode 标量序列，$v$ 是 token 序列，$c$ 是已提交 token 前缀；
- $\tau$ 是运行轨迹，$\mathbf a$ 是制品表，$G$ 是有根类型化 provenance 图；
- $\mathbf k$ 是工具调用、幂等键、请求、回执和提交状态表；
- $\mathbf q$ 是带证据引用的主张表，$n$ 是规范元数据。

相对于 $\Sigma$ 的良构谓词 $\operatorname{WF}_\Sigma(R)$ 要求至少满足以下可检查接口：

1. **表示一致性：** $b\in\mathbb B_U$ 且 $\operatorname{utf8dec}_U(b)=u$；$v\in C=\operatorname{Enc}(X)$ 且 $\operatorname{Dec}(v)=u$；
2. **提交一致性：** $c\preceq v$，轨迹中的提交投影以前缀单调方式结束于 $c$；若记录声明“完整提交”，则还要求 $c=v$；
3. **轨迹闭合：** 初态、固定输入、组件版本、随机流或随机核、调度和终止状态均有类型正确的引用；若声明确定复现，则这些字段足以实例化定义 E.46 的偏函数系统；
4. **制品一致性：** 每个制品引用可解析，声明的长度与按 $\mathsf{DigestSpec}$ 计算的摘要匹配；这只验证记录一致性，不把有限摘要当作数学单射；
5. **provenance 覆盖：** $G$ 通过 $\mathsf{ProvSchema}$ 的类型检查，根指向最终制品，记录中声称参与生成的关键 entity、activity 和 agent 均有对应顶点与类型正确的边；
6. **工具闭合：** 每个工具调用按 $\mathsf{ToolSchema}$ 记录请求、权限、作用域、幂等键、返回或未知状态；凡声明幂等成功重试者，还须能核对定义 E.48 的合同条件；
7. **主张可追踪：** 每个事实主张都有状态、来源或验证结果字段；“未核验”“不适用”和“缺失”是不同状态；
8. **规范分层：** 责任、许可、署名等 $n$ 中判断显式引用其规则或授权依据，不由事实轨迹或 provenance 自动推出。

该接口定义何时一个记录内部闭合，不保证主张在现实中为真，也不保证外部来源可靠。后两者需要独立的证据核验与来源评估。

在 $\operatorname{WF}_\Sigma$ 的记录集合上，可以定义多个总投影，例如字节投影

$$
p_b(R)=b,
$$

规范化文本投影

$$
p_N(R)=\operatorname{NFC}_U(u),
$$

以及 provenance 同构类投影

$$
p_G(R)=[G]_{\cong}.
$$

由定理 E.55，每个投影都诱导一个合法的身份等价关系。

**定理 E.57（不存在由良构接口唯一决定的用途无关身份关系）.** 定义 E.56 的良构条件不能唯一决定一个适用于所有用途的输出身份关系。具体地，存在一组接口参数 $\Sigma$ 和三个相对于它良构的记录 $R_1,R_2,R_3$，使得

$$
R_1\equiv_{p_b}R_2,
\qquad
R_1\not\equiv_{p_G}R_2,
$$

并且

$$
R_1\equiv_{p_N}R_3,
\qquad
R_1\not\equiv_{p_b}R_3.
$$

因此字节身份、规范化文本身份与 provenance 身份均可满足等价关系公理，却彼此作出不同判定；选择哪一种必须由保存、检索、复现、归责等用途给出。

**证明.** 取外部规范输入 E.U2 中的见证

$$
u=u_c,
\qquad
u'=u_d.
$$

令文本域 $X=\{u,u'\}$，选择含两个不同 token 的词表，并把对应的两个单 token 词记为 $\alpha,\beta\in\mathcal V^*$，定义

$$
\operatorname{Enc}(u)=\alpha,
\qquad
\operatorname{Enc}(u')=\beta,
$$

并令 $\operatorname{Dec}(\alpha)=u$、$\operatorname{Dec}(\beta)=u'$；在其余 token 序列上任意定义 $\operatorname{Dec}$。该 tokenizer 满足 round-trip。再由 E.U1 定义

$$
b=\operatorname{utf8enc}_U(u),
\qquad
b'=\operatorname{utf8enc}_U(u').
$$

先构造 $R_1,R_2$。令二者的文本、字节、token 和完整提交字段都分别为 $u,b,\alpha,\alpha$。取类型检查器接受的制品表与轨迹，并让所有引用闭合；工具表和主张表可取空。令 $R_1$ 的 provenance 图只有根制品，令 $R_2$ 的图另含一个生成活动及 `wasGeneratedBy` 边，并让各自记录只声明其图中已有的活动。其余字段按定义 E.56 补齐。两记录因表示、提交和引用均一致而良构。它们的字节投影相同，故 $R_1\equiv_{p_b}R_2$；两图由命题 E.54 不同构，故 $R_1\not\equiv_{p_G}R_2$。

再令 $R_3$ 的文本、字节、token 和完整提交字段分别为 $u',b',\beta,\beta$，并用同样方式补齐其余字段。由 E.U1 的单射性，$u\ne u'$ 推出 $b\ne b'$。又由 E.U2 的外部见证，

$$
\operatorname{NFC}_U(u)=\operatorname{NFC}_U(u'),
$$

有 $R_1\equiv_{p_N}R_3$；因为 $b\ne b'$，有 $R_1\not\equiv_{p_b}R_3$。

定理 E.55 已证明 $\equiv_{p_b},\equiv_{p_N},\equiv_{p_G}$ 都是等价关系。本构造表明，良构接口与等价关系公理允许它们同时存在，却不迫使它们对记录对作出相同判断。因此这些条件不能逻辑地选出唯一的用途无关身份关系。$\square$

这里使用的不同 NFC 表示对已在 E.U2 中显式标为外部规范输入，不是书内推导。即便另一个受限文本域碰巧让 NFC 单射，$R_1,R_2$ 的内容身份与 provenance 身份反例仍足以否定用途无关的唯一选择。

## 练习

**练习 E.1.** 设 $R\subseteq X\times Y$ 与 $Q\subseteq Y\times Z$ 都右唯一。证明 $Q\circ R$ 右唯一，并写出它对应的偏函数定义域。

**练习 E.2.** 构造两个内部状态不同但观察等价的确定性 LTS 状态。要求给出状态集合、事件集合、转移和观察函数。

**练习 E.3.** 给定有限偏序 $a<b$、$a<c$，且 $b,c$ 不可比。列出全部线性扩张，并说明在 $b,c$ 对应事件可交换时最终状态为何不依赖调度。

**练习 E.4.** 构造一个三元素句子集合上的真值解释和概率分布，使概率最大的句子为假，概率第二大的句子为真。说明该例与事实核验协议之间的关系。

**练习 E.5.** 设 tokenizer 满足 $\operatorname{Dec}(\operatorname{Enc}(x))=x$。证明它在规范像上不能有两个不同 token 序列解码为同一文本；再构造规范像之外两个不同 token 序列解码为同一文本的接口，说明这不违反定理 E.36。指出构造中哪些 Unicode 性质必须作为固定版本的外部规范输入。

**练习 E.6.** 构造一个含终止状态的三状态随机核，使每个非终止状态的一步终止概率至少为 $1/4$。逐项列出长度 $2$ 路径及其质量，验证归一化，并用定理 E.45 给出 $\Pr(T>n)$ 的上界。随后把该核实现为固定输入与固定调度下的工具乘积系统，说明唯一最大样本轨迹还需要固定哪一条随机流。

**练习 E.7.** 为“首次写入成功但确认丢失，随后两次重试”的服务写出世界状态、键表和声明观察投影。逐条检查定义 E.48 的七项条件；再给两个带完整唯一序号的客户端日志到达顺序，证明服务端提交记录的确定归并结果相同。

**练习 E.8.** 构造三个良构输出记录，使其中两个字节相同但 provenance 图不同构，另两个 NFC 投影相同但字节不同。分别判断字节身份、规范化文本身份、token 规范像身份和 provenance 身份，并说明档案保存、语义检索与责任审计各自为何可能选择不同关系。
