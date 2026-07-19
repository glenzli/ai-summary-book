# 第四章 解码怎样诱导输出分布

基础模型给出逐步条件分布 $q_\theta$；用户最终看到的输出由温度、截断、历史惩罚、语法约束、搜索、重排序、过滤和停止规则共同决定。解码不是无关紧要的显示设置，而是把基础模型分布变成系统分布的算法。

## 4.1 解码器是一个条件随机核

把当前前缀记为 $h_t=(c,y_{<t})$，基础模型分布为 $q_t(\cdot\mid h_t)$。解码配置 $\delta$ 定义一个变换

$$
D_{\delta,t}:q_t\longmapsto
\widetilde q_{\delta,t}(\cdot\mid h_t).
$$

一般服务的停止并不是词表中的另一个 token。最大长度、stop string、grammar 接受和上下文上限都依赖额外状态，且可能在 token 选择后触发。为统一表示，令 $S_t\in\mathcal S_\delta$ 包含前缀、处理器状态、停止匹配器和长度计数，并加入吸收态 $\dagger$。解码初始化核 $\nu_\delta(ds_0\mid c)$ 规定初始处理器与随机源状态；确定初始化是它退化为 $\delta_{s_0(c)}$ 的特例。基础模型与解码协议共同定义转移核

$$
T_{\theta,\delta}(ds'\mid s,c),
$$

其中一次转移可选择 token、更新状态并继续，也可因 EOS 或非 token 规则进入 $\dagger$。给定 $\nu_\delta$ 与逐步转移核，Ionescu–Tulcea 定理在路径空间上唯一确定测度 $P_{\theta,\delta}(\cdot\mid c)$。令

$$
\tau=\inf\{t\ge 0:S_t=\dagger\}
$$

为停止时间，并记有限停止路径集合为 $\Omega_{\mathrm{fin}}=\{\omega:\tau(\omega)<\infty\}$。返回映射 $R_\delta:\Omega_{\mathrm{fin}}\to\mathcal Y$ 删除 EOS、处理 stop string 并反分词。对输出空间中的可测集合 $B$，固定协议诱导

$$
K_{\theta,\delta}(B\mid c)
=P_{\theta,\delta}
\bigl(\{\omega\in\Omega_{\mathrm{fin}}:
R_\delta(\omega)\in B\}\mid c\bigr).
$$

若 $P(\tau<\infty\mid c)=1$，它是有限输出上的概率分布；否则它是总质量小于 $1$ 的次概率测度，缺失质量是不终止事件。也可以把“不终止/超时”作为显式结果加入输出空间，使其重新成为概率测度。

只有在停止规则恰为额外 EOS token、没有返回后处理时，有限序列概率才简化为

$$
K_{\theta,\delta}(y_{1:m}\mid c)
=\left[
\prod_{t=1}^{m}
\widetilde q_{\delta,t}(y_t\mid c,y_{<t})
\right]
\widetilde q_{\delta,m+1}(\mathrm{EOS}\mid c,y_{1:m}).
$$

如果随后还有显示映射 $d$、过滤器 $F$ 和工具环境 $E$，用户输出分布还要继续对这些随机或确定变换取推前。基础模型概率、解码概率和用户可见频率因此不能互换。

## 4.2 Temperature 是指数族尺度变换

对固定 logits $z$ 和 $T>0$，

$$
q_T(i)
=\frac{e^{z_i/T}}{\sum_je^{z_j/T}}
=\frac{q_1(i)^{1/T}}
{\sum_jq_1(j)^{1/T}}.
$$

赔率满足

$$
\log\frac{q_T(i)}{q_T(j)}
=\frac{z_i-z_j}{T}.
$$

$T<1$ 放大赔率差，$T>1$ 缩小赔率差。若最大 logit 唯一，$T\to0^+$ 时分布趋向该 argmax；$T\to\infty$ 时在未屏蔽的有限词表上趋向均匀。

固定 $z$ 时，熵随 $T$ 单调不减。令 $\beta=1/T$、$Z(\beta)=\sum_ie^{\beta z_i}$，则

$$
H(q_T)=\log Z(\beta)-\beta\mathbb E_{q_T}[z],
$$

由于

$$
\frac{d}{d\beta}\log Z(\beta)=\mathbb E_{q_T}[z],
\qquad
\frac{d}{d\beta}\mathbb E_{q_T}[z]
=\operatorname{Var}_{q_T}(z),
$$

所以 $dH/d\beta=-\beta\operatorname{Var}_{q_T}(z)$。再结合
$d\beta/dT=-1/T^2$，得到

$$
\frac{dH(q_T)}{dT}
=\frac{\operatorname{Var}_{q_T}(z)}{T^3}\ge0.
$$

该结论只针对单步固定 logits。生成路径改变后，较高 temperature 不保证整段文本的事实正确率、语义多样性或任务效用单调变化。

## 4.3 Top-k 删除固定数量的支持点

设 $K_t$ 是当前分布概率最高的 $k$ 个 token 集合。top-k 分布为

$$
q^{(k)}_t(i)
=\frac{q_t(i)\mathbf 1\{i\in K_t\}}
{\sum_{j\in K_t}q_t(j)}.
$$

$k=1$ 时退化为对当前分布的单点采样，即 greedy；$k=|V|$ 时不变。固定 $k$ 不考虑原分布尖锐程度。边界存在并列 logits 时，集合 $K_t$ 还依赖 tie-breaking 与词表顺序。

被删除 token 在基础模型中可能有正概率，只是在解码分布中成为零概率。以后以该 token 为前缀的整棵序列子树也随之被删除。

## 4.4 Top-p 删除可变数量的尾部

将 token 按概率降序排列，令 $N_{p_0}$ 为累计质量至少达到 $p_0\in(0,1]$ 的最小前缀集合：

$$
q^{(p)}_t(i)
=\frac{q_t(i)\mathbf 1\{i\in N_{p_0}\}}
{\sum_{j\in N_{p_0}}q_t(j)}.
$$

分布尖锐时集合较小，分布平坦时集合较大。边界并列、最少保留 token 数和浮点累计顺序均属于协议的一部分。

temperature 与 top-p 一般不交换：先改变温度会改变达到累计阈值所需的集合，再截断；先按原概率截断后再调温度则固定了另一支持集。只记录最终 $T,p_0$ 而不记录处理顺序，不足以复现分布。

## 4.5 逐步截断不是完整序列的一次截断

top-k/top-p 在每个随机前缀上重新计算。最终分布是

$$
\widetilde q(y_{1:m}\mid c)
=\prod_t\widetilde q_t(y_t\mid c,y_{<t}),
$$

而不是先列出所有完整序列，再按其 $q_\theta(y\mid c)$ 做一次全局 top-k 或 top-p。

一个早期 token 被删除后，它的全部延续概率都归零；另一路径上相同的后续 token 仍可能被保留。逐步解码改变的是前缀树上的条件核。

这也解释了为何不能仅从基础模型的完整序列分数反推出实际采样频率：必须知道每个已访问前缀上的处理后分布。

## 4.6 历史惩罚与处理顺序

frequency、presence 或 repetition penalty 根据已生成前缀修改当前 logits。例如

$$
z'_{t,i}=z_{t,i}-\lambda n_i(y_{<t}),
$$

其中 $n_i$ 是 token $i$ 的出现计数。此时策略依赖整个运行历史与实现的计数规则。

处理顺序通常不交换。先减惩罚再除温度得到

$$
\frac{z_i-\lambda n_i}{T},
$$

而先除温度再减同样数值的惩罚得到

$$
\frac{z_i}{T}-\lambda n_i.
$$

除非 $T=1$ 或同步缩放 $\lambda$，二者不同。token 级惩罚也不识别语义重复：同义改写可能绕过惩罚，必要术语和标点却会被压低。

## 4.7 语法约束给出局部条件化策略

若自动机状态只允许 token 集 $A_t(h_t)$，且该集合在 $q$ 下具有正质量，运行时可设

$$
q_A(i\mid h_t)
=\frac{q(i\mid h_t)\mathbf 1\{i\in A_t(h_t)\}}
{\sum_{j\in A_t(h_t)}q(j\mid h_t)}.
$$

只要状态机和 token 接口正确，这可保证路径符合局部语法，例如生成可解析 JSON。它不保证字段事实正确，也不保证业务约束。

更细致地说，逐步屏蔽通常**不等于**把基础完整序列分布全局条件化到“最终字符串合法”。全局条件分布在首步还要考虑每个前缀未来合法完成的总概率；局部屏蔽只在当前可行动作间归一化。

例如首步 A、B 的基础概率各为 $1/2$，但 A 后只有 $0.1$ 的基础概率能合法完成，B 后则为 1。全局条件于最终合法时，B 的首步质量应为 $1/(1+0.1)$；若 A、B 当前都语法可行，局部 mask 仍给二者各 $1/2$。只有使用包含未来完成质量的精确条件算法，两者才一致。

## 4.8 Greedy 与 beam search 是决策规则

greedy 在每步选择局部最大概率 token：

$$
y_t=\arg\max_iq_t(i\mid y_{<t},c).
$$

它不保证得到全局最大概率完整序列，因为局部最优前缀可能只有低概率延续。beam search 保留有限数量高分前缀，近似搜索高分序列；beam 宽度、长度惩罚与终止规则都会改变结果。

二者不是从基础分布抽取的无偏样本。重复 beam search 的相同结果不能用来估计 $q_\theta$；更大的 beam 也不保证任务质量更高，因为序列似然与外部效用不是同一个目标。

当 argmax 并列或数值实现变化时，所谓 deterministic decoding 仍需规定 tie-breaking 与执行环境。

## 4.9 Best-of-N 与重排序诱导新的分布

设候选独立来自提议分布 $g(y\mid x)$，评分函数为 $s(x,y)$，系统选取最高分：

$$
\widehat Y=\arg\max_{1\le i\le N}s(x,Y_i),
\qquad Y_i\overset{\mathrm{iid}}{\sim}g.
$$

固定 $x$，令评分分布的左右累计质量为

$$
F_s(a)=P_{Y\sim g}(s(x,Y)\le a),
\qquad
F_s(a^-)=P_{Y\sim g}(s(x,Y)<a),
$$

并令 $m_s(a)=F_s(a)-F_s(a^-)$。若最高分并列时在并列候选索引间均匀选择，则对任意可测候选集合 $B$，

$$
P(\widehat Y\in B)
=\int_B w_N(s(x,y))g(dy\mid x),
$$

其中

$$
w_N(a)=
\begin{cases}
\displaystyle
\frac{F_s(a)^N-F_s(a^-)^N}{m_s(a)},
&m_s(a)>0,\\[8pt]
N F_s(a)^{N-1},&m_s(a)=0.
\end{cases}
$$

第一种情形来自枚举其余 $N-1$ 个候选中有多少个与当前分数并列，并在并列项间分摊选择概率；第二种情形是无原子极限。于是被选分布相对于 $g$ 的 Radon-Nikodym 导数是 $w_N(s(x,y))$。高评分区域随 $N$ 增大而被放大。

这说明 best-of-N 不只是“多试几次”：它改变了用户输出分布，并可能系统放大评分器漏洞。若候选相关、提前停止，或并列规则依候选内容而非只依索引，上式不再直接适用，必须按实际算法分析。

## 4.10 拒绝、过滤与有限重试

若从 $q$ 独立抽样直到事件 $A$ 成立，且允许无限重试并有 $q(A)>0$，接受样本服从

$$
q(y\mid A)
=\frac{q(y)\mathbf 1\{y\in A\}}{q(A)}.
$$

真实系统常只重试至多 $M$ 次。此时成功概率为

$$
1-(1-q(A))^M,
$$

还剩余失败、默认回答或人工升级的概率质量。若过滤器本身有随机误差，实际条件事件是“过滤器接受”，不是真实属性 $A$。

只观察通过过滤的文本会产生选择偏差：无法由可见样本频率恢复基础模型中被拒绝模式的质量，除非记录拒绝数和选择机制。

## 4.11 停止规则与返回映射

EOS、最大长度、stop string、超时与工具终止共同决定输出空间。stop string 可能跨 token 边界；服务也可能先生成 stop 内容再从返回文本中删除。因此要区分：

1. 模型实际生成的 token 路径；
2. 服务识别到的终止事件；
3. API 返回的字符串；
4. 客户端最终显示的内容。

若长度 $L$ 强制截断，返回前缀事件包含所有以该前缀开头而尚未终止的路径，不等于模型自然赋给一个完整 $L$-token 回答的概率。

## 4.12 Seed 只固定随机状态的一部分

seed 初始化伪随机数生成器。复现同一路径还需固定：

- PRNG 算法、状态和每步消费顺序；
- 原始 logits 与全部处理器；
- batch、并发调度和设备映射；
- 数值 kernel、精度与 tie-breaking；
- 模型、template、过滤器和工具版本。

相同 seed 而随机数消费顺序不同，可以产生不同输出；不同 seed 也可能偶然产生相同路径。seed 是实验条件，不是输出或分布的身份标识。

## 4.13 自回归与扩散的概率路径不同

自回归模型直接分解离散序列：

$$
q(y_{1:m}\mid c)=\prod_tq(y_t\mid y_{<t},c).
$$

扩散模型通常从噪声状态 $X_T$ 出发，经反向转移得到

$$
q_\theta(x_{0:T}\mid c)
=q(x_T)
\prod_{t=1}^{T}q_\theta(x_{t-1}\mid x_t,c).
$$

边缘输出分布 $q_\theta(x_0\mid c)$ 需要对中间轨迹积分或求和。随机性可以来自初始噪声与每步 SDE/反向核；概率流 ODE 在给定初始状态和求解器后可以是确定轨迹，但初始状态仍由分布产生。

步数、噪声日程、ODE/SDE 求解器、guidance 和数值误差都会改变诱导输出分布。语言模型的 top-p 是在离散词表删除概率质量，guidance 则改变连续状态的 score 或向量场；二者不能只因都叫“采样参数”而直接类比。

## 4.14 研究与部署应选不同协议

解码策略应由分析目标或任务损失决定：

- 研究基础模型概率：保留原始 logits，避免未记录的处理器与后过滤；
- 估计某部署行为频率：复现完整系统协议并重复采样；
- 精确抽取和代码：低随机性、语法约束与外部验证配合；
- 多样创作：允许高熵提议，再显式评估筛选偏差；
- 稳定能力评测：固定协议，同时报告运行间变异。

不存在跨任务最优的 temperature、top-p 或 beam width。对一个损失函数有利的分布变换，可能对另一个任务有害。

## 4.15 本章结论

解码器在每个前缀上定义新的条件随机核。温度缩放赔率，top-k/top-p 改变支持，历史惩罚和语法约束改变路径条件，搜索、best-of-N、过滤与停止又继续重加权或映射输出。局部约束通常不等于对完整基础分布做一次全局条件化。任何概率分析都必须先声明研究的是基础模型、解码器还是完整系统。
