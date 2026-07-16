# 术语表

| 术语 | 本书口径 |
|---|---|
| run / 运行 | 一次带初态、外生事件、轨迹、终止状态与制品谱系的执行对象 |
| observation / 观察 | 从运行或制品到合同所比较对象的全函数投影 |
| reproduction contract / 复现合同 | 五元组 $C=(\mathcal D,N,P,M,A)$；规定 schema 域、规范化、前置条件、比较规则与合法输入上的三值判决算法，schema 失败另返结构错误 |
| exact equivalence / 精确等价 | 由全观察函数的核 $x\equiv_fy\iff f(x)=f(y)$ 诱导的等价关系 |
| tolerance acceptance / 容差接受 | 先以全定义资格谓词处理形状、单位与特殊值，再在合法数值域检查 $d(x,y)\le\varepsilon$；正容差下一般不传递 |
| bitwise identity / 字节身份 | 合同指定的规范有限字节串完全相等，不等同于摘要相同 |
| semantic value identity / 语义值身份 | 固定解析和解释函数后得到同一语义对象 |
| trace identity / 轨迹身份 | 固定轨迹投影后的有限/无限标签序列完全相等 |
| statistical compatibility / 统计相容 | 有限样本通过预定统计规则；不表示未知分布被证明相等 |
| scientific reproduction / 科学复制 | 新证据在目标量对齐、独立性和适用性前提下支持同一科学主张；不是天然等价关系 |
| sufficient state description / 观察充分状态描述 | 对允许的未来事件流，记录相同足以推出目标观察相同的状态映射 |
| deletion distinguishing witness / 删项区分见证 | 其余记录分量相同、被删分量不同且观察不同的状态对；只证明删项后的记录不充分 |
| exogenous event / 外生事件 | 能影响转移但不属于当前内部状态的输入，如随机字节、服务响应或调度选择 |
| deterministic transition / 确定转移 | 对给定状态和外生事件至多给出一个下一状态的部分函数 |
| PRNG state / 伪随机生成器状态 | 决定后续伪随机字流的当前内部状态，不等同于初始 seed |
| seed | 通过版本化 Seed 映射初始化一个生成器状态的输入；不是完整执行状态 |
| schedule / 调度 | 并发事件偏序的线性扩张 |
| happens-before | 由程序顺序、同步和消息因果等产生的严格偏序；具体语义按相应模型声明 |
| linear extension / 线性扩张 | 保持偏序约束的事件全序 |
| linearizability / 线性化 | 并发历史等价于保持线程顺序与实时先后的一份合法顺序历史的正确性条件 |
| storage conflict / 存储冲突 | 两访问重叠且至少一个为写；语言级 data race 后果由具体内存模型规定 |
| consistent cut / 一致割 | 对因果前驱向下闭合的事件子集，可表示一致分布式快照边界 |
| binary64 | IEEE 754 的 64 位二进制浮点格式；本书默认 `roundTiesToEven` 时会明说 |
| subnormal / 次正规数 | binary64 中形如 $k2^{-1074}$、$1\le k<2^{52}$ 的非零数 |
| $u$ / unit roundoff | 本书 binary64 默认舍入下取 $u=2^{-53}$ |
| $\alpha$（浮点章） | 最小正次正规数 $2^{-1074}$；与统计显著性水平不是同一类型，跨章使用时按上下文声明 |
| $\gamma_k$ | $ku/(1-ku)$，仅在 $ku<1$ 时使用，且定义 $\gamma_0=0$ |
| estimand / 目标量 | 由总体、处理/算法、结果、时间和汇总函数共同定义的统计参数 |
| experimental unit / 实验单位 | 设计中可视为独立重复的最小单位；训练运行与固定模型下的测试样本属于不同层级 |
| confidence set / 置信集 | 在指定抽样模型下具有频率覆盖保证的随机集合 |
| equivalence margin / 等效界 | 分析前由领域意义规定的可忽略差异边界 |
| TOST | two one-sided tests；两个单侧检验均拒绝才判等效 |
| FWER | 至少错误拒绝一个真零假设的概率 |
| FDR | 错误发现比例的期望；分母为零时须预定约定 |
| content identity / 内容身份 | 固定规范编码 $c$ 后满足 $c(a)=c(b)$ |
| digest identity / 摘要相同 | $H(c(a))=H(c(b))$；因有限摘要必有碰撞，不逻辑等同于内容身份 |
| provenance / 谱系 | Entity、Activity、Agent 及类型化使用、生成、责任关联边组成的来源陈述；派生 DAG 与责任关联是不同结构 |
| reproducible build / 可复现构建 | 对声明的源、指令和允许环境变化，独立构建产生合同指定的逐字节相同输出 |
| Repeatability / Reproducibility / Replicability | 首字母大写时只在已声明的 ACM/NASEM/VIM 来源口径中使用，不设跨机构唯一译法 |
| `INCONCLUSIVE` | 输入格式合法，但证据不足、制品不可访问或前置条件失败时的一种合同判决；不等同于 `FAIL`，schema 不合法则另属结构错误 |
| executable contract / 可执行合同 | 域检查、规范化、前置条件和判决均为全定义、确定、终止算法的合同 |
