# 符号与类型约定

本表只收录跨章复用的符号。局部变量仍在使用处定义；同一字母若在不同章节承担局部含义，以该章声明为准。

## 基本集合与映射

| 符号 | 类型或含义 |
| --- | --- |
| $\mathcal X,\mathcal Y$ | 输入空间与目标/输出空间 |
| $\mathcal V$ | 有限或可数 token 词表 |
| $\mathcal V^*$ | 有限 token 序列集合，含空序列 |
| $\rightharpoonup$ | 部分函数；某些输入上可以未定义 |
| $\mathcal P(A)$ | 集合 $A$ 上的概率测度集合；只在可测结构已声明时使用 |
| $\mathbf 1\{E\}$ | 事件 $E$ 的指示函数 |
| $x\equiv_R y$ | 相对于关系 $R$ 的等价或接受，不默认逐位相等 |
| $x\preceq y$ | 序列前缀关系；存在序列 $r$ 使 $y=xr$ |

## 模型、系统与运行

| 符号 | 类型或含义 |
| --- | --- |
| $a_\theta$ | 参数为 $\theta$ 的模型工件及其加载约定 |
| $c$ | 已组装上下文，不只指用户可见 prompt |
| $D$ | 解码、停止和后处理规则 |
| $\xi$ | 显式采样随机流或伪随机状态 |
| $r$ | 检索、工具、时钟、文件系统等外部返回 |
| $\eta$ | 硬件、内核、批处理、并发和数值环境 |
| $\mathcal B$ | 一次运行的边界条件元组 |
| $S,O,A$ | Agent 的状态、观察和行动集合 |
| $\delta:S\times O\rightharpoonup S\times A$ | 确定性部分转移；随机策略时改用核 |
| $\tau=(s_0,e_1,s_1,\ldots,e_n,s_n)$ | 有限执行轨迹 |
| $R$ | 类型化运行记录；不与实数集混用 |

## 概率与语言模型

| 符号 | 类型或含义 |
| --- | --- |
| $(\Omega,\mathcal F,P)$ | 概率空间 |
| $X:\Omega\to\mathcal X$ | 随机变量；默认要求可测 |
| $X_\#P$ | 测度 $P$ 在可测映射 $X$ 下的推前分布 |
| $P_\theta(v_t\mid c,v_{<t})$ | 位置 $t$ 的条件 token 分布 |
| $z\in\mathbb R^{|\mathcal V|}$ | logits 向量 |
| $T>0$ | softmax 温度；$T=0$ 不代入 softmax，而表示另行定义的极限/贪心规则 |
| $H(P)$ | 熵；离散情形使用自然对数，单位为 nat |
| $D_{\mathrm{KL}}(P\|Q)$ | KL 散度；若 $P\not\ll Q$，取 $+\infty$ |
| $P\ll Q$ | 测度 $P$ 关于 $Q$ 绝对连续：$Q(A)=0$ 蕴含 $P(A)=0$ |
| $\operatorname{TV}(P,Q)$ | 总变差；有限集合上采用 $\frac12\sum_x|P(x)-Q(x)|$ |
| $K(y\mid x)$ | 从输入空间到输出空间的 Markov 核；有限情形每一行非负且和为 $1$ |
| $\mathbb E[X\mid\mathcal G]$ | 相对于子 $\sigma$-代数 $\mathcal G$ 的条件期望，按 a.s. 等价类理解 |
| $\operatorname{ATE}=\mathbb E[Y(1)-Y(0)]$ | 平均处理效应；识别式另需交换性、正性与一致性 |
| $q(x)$ | 预测概率或分数，必须同时声明预测事件 |

## 评测、解释与审计

| 符号 | 类型或含义 |
| --- | --- |
| $\Pi$ | 评测协议，包含任务、提示、工具、评分和聚合规则 |
| $m(x,y)$ | 明确方向和量纲的指标或评分函数 |
| $\widehat\mu_n$ | $n$ 个已声明样本上的经验平均 |
| $E$ | 被解释项，例如 logit 差、行为事件或工具选择 |
| $I$ | 干预算子或干预方案 |
| $C$ | 主张集合；在上下文空间章节中不沿用此义 |
| $\operatorname{prov}(e)$ | entity $e$ 的 provenance 子图或记录引用 |
| $\operatorname{digest}(e)$ | 在声明算法和规范化下的内容摘要 |
| $\operatorname{IG}_i(x;x')$ | 从基线 $x'$ 到输入 $x$ 的第 $i$ 个积分梯度归因 |
| $\operatorname{VCdim}(\mathcal H)$ | 二分类假设类 $\mathcal H$ 的 VC 维 |
| $\operatorname{prox}_{\lambda\|\cdot\|_1}$ | 相对于欧氏二次项的 $L^1$ 近端映射 |
| `Pass/Fail/Inconclusive` | 判定三值；schema 非法时返回结构错误而非三值之一 |

## 数值与验证

| 符号 | 类型或含义 |
| --- | --- |
| $\operatorname{fl}(x\circ y)$ | 在声明格式、舍入模式与异常条件下，一次机器浮点运算的结果 |
| $u$ | unit roundoff；只在局部声明的浮点模型中使用 |
| $\gamma_k=ku/(1-ku)$ | $ku<1$ 时的最坏情形舍入误差因子界 |
| $\operatorname{RN}_{64}$ | 在声明 IEEE 754 binary64 与舍入模式下的正确舍入函数 |
| $\operatorname{WF}_\Sigma(R)$ | 记录 $R$ 相对于版本化接口参数 $\Sigma$ 的良构谓词 |
| $\Gamma\vdash C$ | 在所声明形式系统中，结论 $C$ 可由前提集 $\Gamma$ 推导 |
| $\mathcal M\models C$ | 结构或语义模型 $\mathcal M$ 满足命题 $C$ |

## 等式与近似

- `$=$` 只表示所声明类型中的相等；文本规范化相同、语义等价和统计相容使用各自关系。
- `$\approx$` 必须在局部声明误差度量和容差，不能作为“差不多相同”的默认符号。
- 概率式中的条件竖线表示条件信息，不表示因果干预；干预使用 $\operatorname{do}(X=x)$ 或明确干预算子。
- 经验均值、总体期望和单次观测分别写作 $\widehat\mu_n$、$\mathbb E[X]$ 和 $X(\omega)$，不互相替代。
