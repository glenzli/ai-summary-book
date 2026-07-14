# 符号与术语

| 符号或术语 | 类型与含义 |
|---|---|
| $A^*$ | 集合 $A$ 上有限序列的集合 |
| $\mathbb B$ | 八位字节集合 $\{0,\ldots,255\}$ |
| $\mathbb U$ | Unicode 标量值集合；不含 UTF-16 surrogate code point |
| $\mathbb C_{16}$ | UTF-16 编码单元集合 |
| $\mathbb V$ | 固定 tokenizer 的有限 token ID 集合 |
| $\operatorname{UTF8}$ | 合法 UTF-8 字节串子集 $E_8(\mathbb U^*)\subset\mathbb B^*$ |
| $E_8,D_8$ | UTF-8 编码及其在合法域上的严格解码 |
| $\operatorname{grapheme}_{\nu,\gamma}$ | 固定 Unicode 版本 $\nu$ 与 UAX #29 profile $\gamma$ 后的扩展字素簇分段函数 |
| $\Theta$ | tokenizer 完整配置，包括词表、预处理、负载和特殊 token 规则 |
| $\operatorname{AdmIn}_\Theta$ | tokenizer 编码器的可接受输入域 |
| $\operatorname{AdmTok}_\Theta$ | tokenizer 解码器可接受的 token 序列域 |
| $\operatorname{Enc}_\Theta,\operatorname{Dec}_\Theta$ | tokenizer 编码与解码函数；可逆性需另证 |
| $f:X\rightharpoonup Y$ | 从潜在输入集 $X$ 到 $Y$ 的部分函数 |
| $\operatorname{Result}(Y,E)$ | 成功值与显式错误值的不交并 |
| $R\subseteq X\times Y$ | 一般二元关系；可非右唯一 |
| LTS | 带标签转移系统 $(S,A,\to,F)$ |
| strong determinism | 每个状态至多有一个“标签、下一状态”二元组 |
| maximal trace | 不能延长的有限轨迹或无限轨迹 |
| stuck state | 不在正常终止集内且无合法后继的状态 |
| divergence | 存在无限执行轨迹 |
| $\sim_\pi$ | 单条轨迹在观察函数 $\pi$ 的核等价 |
| $\operatorname{Obs}_\pi(s)$ | 从状态 $s$ 出发的最大轨迹可能观察值集合 |
| $\approx_\pi^{\mathrm{may}}$ | 可能观察集相等的状态等价 |
| $\approx_{\mathcal K,\pi}$ | 对所有允许上下文量化的观察等价 |
| $K(c,\cdot)$ | 从当前配置到标签和下一配置的随机核 |
| $\mathbb P_n,\mathbb P_\infty$ | 有限或无限轨迹空间上的概率测度 |
| pushforward | 可测观察函数把路径测度推到输出空间所得测度 |
| implementation map | 用显式随机输入实现随机核的可测映射 $G(c,u)$ |
| seed | PRNG 的输入；不是随机核，也不单独保证路径复现 |
| world state $W$ | 工具可读取或改变的外部状态 |
| commit | 外部副作用或流片段越过协议规定的提交边界 |
| unknown commit state | 调用方不能判断副作用是否已提交的状态 |
| idempotency key | 把重试绑定到一个逻辑操作的键；保证依赖服务端协议 |
| happens-before | Lamport 的严格因果顺序 $\prec$；正文以其自反闭包 $\preceq$ 作为通常偏序 |
| linear extension | 保持偏序的事件全序 |
| candidate / committed | 尚可撤回的候选片段 / 按协议不可再修改的已提交片段 |
| artifact | 可保存、复制、引用或签名的数据制品 |
| provenance | entity、activity、agent 及其关系组成的来源记录 |
| content identity | 原始或规范字节相等定义的等价关系 |
| digest identity | 哈希值相等；不无条件蕴含精确内容相同 |
| run identity | logical 或 attempt 运行标识相等 |
| provenance identity | 在固定标签签名下有根来源图同构 |
| denotation | 项在结构和变量赋值下的指称 |
| logical equivalence | 对指定模型类及全部赋值具有相同满足状态 |
| verification status | 指定证据协议输出的 Supported/Refuted/Unknown/OutOfScope |
| engineering agent | 相对于 $\delta:S\times O\rightharpoonup S\times A$ 或其关系/核版本定义的控制系统 |
| normative subject | 相对于理论或制度被赋予承诺或责任资格的主体 |
| $\mathsf{Factor}$ | 主体、活动、输入制品、软件组件与环境事件的不交并；因果关系还须声明因果模型 |
| authorship | 由明确规范体系解释的作者关系 |
| $\operatorname{Field}(X)$ | Value$(x)$ 或带 $\mathsf{AbsenceReason}$ 的 Absent 字段；不把缺失、空串和发散合并 |
| $\mathsf{OutputRec}_\Sigma$ | 第 11 章相对于事件 schema $\Sigma$ 的十二分量输出记录类型 |
| $\operatorname{WF}(\mathcal O)$ | 第 11 章输出记录的良构与跨层一致性谓词 |

正文中的箭头必须按函数、部分函数、LTS、因果边或随机核分别声明；同一箭头不跨类型复用。
