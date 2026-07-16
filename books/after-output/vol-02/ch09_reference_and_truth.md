# 第九章：表达式、指称、真值与核验状态

第八章已经保存了最终消息的字节和来源，却仍不能回答它说得对不对。字符串
`SP404` 只有在航班编号、运营日期和查询时点确定后才指向一条航班记录；“已写入”还要说明文件对象、提交时点和所要求的内容。贯穿案例的一句话实际上包含两个可以分别成立、失败或暂时无法核验的主张。

本章从这个证据缺口进入表达式、指称、真值和协议相对的核验状态。第一章提供 Unicode 文本；一阶逻辑与概率分布提供形式语言；Tarski 语义的递归定义作为外部输入。目标不是为自然语言建立唯一完整语义，而是把每一次事实判断所需的语境和桥接接口写出来。

## 9.1 从字符串到带类型表达式

固定字母表 $\Sigma$。解析器是部分函数

$$
\operatorname{parse}:\Sigma^*
\rightharpoonup\operatorname{Expr}.
$$

类型检查器在环境 $\Gamma$ 下也是部分函数

$$
\operatorname{type}_\Gamma:
\operatorname{Expr}\rightharpoonup\operatorname{Type}.
$$

字符串语法合法不保证表达式类型正确；表达式类型正确也不保证闭合。句子是没有自由变量且类型为命题的公式。

对自然语言 Unicode 序列，通常更诚实的接口是语境相关关系

$$
\mathcal P_c\subseteq
\mathbb U^*\times\operatorname{MeaningRep},
$$

因为歧义文本在同一粗语境下仍可能有多个候选分析。只有再给出消歧规则或概率模型，才可细化为函数或核。

对贯穿案例，固定运营日期 $d_\star$、查询时点 $t_q$、写入提交时点
$t_w$ 和目标文件实体 $f_\star$。一次明确的语义分析把 $u_\star$ 分为两个句子，并映到闭合公式

$$
\varphi_f:
\operatorname{Status}(\mathtt{SP404},d_\star,t_q)
=\operatorname{Cancelled},
$$

$$
\varphi_w:
\operatorname{Contains}(f_\star,t_w,
\operatorname{CancellationRecord}(\mathtt{SP404},d_\star)).
$$

若省略 $d_\star$，编号复用会使第一个指称欠定；若把 $f_\star$ 只写成路径
`trip.md`，第六章的 rename 与命名空间变化会使第二个指称欠定。补全这些参数以后，两个公式才有可由结构解释的真值条件。

## 9.2 一阶结构与满足

固定一阶签名 $\mathcal L$。一个 $\mathcal L$-结构
$\mathcal M$ 包含非空论域 $D^\mathcal M$，并为常量、函数符号和谓词符号给出相应类型的解释。变量赋值为
$\rho:\mathsf{Var}\to D^\mathcal M$。

项 $t$ 的指称由递归解释

$$
\llbracket t\rrbracket_{\mathcal M,\rho}
\in D^\mathcal M.
$$

公式的满足关系
$\mathcal M,\rho\models\varphi$ 也按逻辑联结词与量词递归定义。若 $\varphi$ 是句子，其真值不依赖 $\rho$，写作
$\mathcal M\models\varphi$。

这里“真”是相对于结构的形式语义概念。若 $\mathcal M$ 被用来表示现实，还需另给模型元素、测量和现实对象之间的校准；形式解释本身不证明该校准正确。

## 9.3 三类相同与完整量词

**语法相同**是表达式树相等，或在更粗规则下的 alpha-equivalence。

**在固定解释下指称相同**定义为

$$
t_1\equiv_{\mathcal M,\rho}^{\mathrm{den}}t_2
\Longleftrightarrow
\llbracket t_1\rrbracket_{\mathcal M,\rho}
=
\llbracket t_2\rrbracket_{\mathcal M,\rho}.
$$

**相对于模型类 $\mathcal K$ 逻辑等价**定义为

$$
\varphi\equiv_{\mathcal K}\psi
\Longleftrightarrow
\forall\mathcal M\in\mathcal K\;
\forall\rho:\mathsf{Var}\to D^\mathcal M,\quad
(\mathcal M,\rho\models\varphi
\Longleftrightarrow
\mathcal M,\rho\models\psi).
$$

若 $\varphi,\psi$ 为句子，赋值量词可省略；若 $\mathcal K$ 是全部
$\mathcal L$-结构，得到通常的逻辑等价。只在一个结构中同真不蕴含逻辑等价。

“晨星”和“昏星”在某现实解释下可同指，却语法不同；同一代词在不同语境可异指；$p\land q$ 与 $q\land p$ 在经典命题结构类中逻辑等价。

## 9.4 语境与索引词

自然语言语境至少可抽象为

$$
c=(\mathsf{speaker},\mathsf{addressee},
\mathsf{time},\mathsf{place},\mathsf{world},
\mathsf{discourse}).
$$

索引词“我”“这里”“今天”和时态解释依赖这些分量。文本相同而语境不同，可以表达不同命题。保存消息字符串却丢失时区、说话者或引用对象，会使未来解释关系不再右唯一。

上下文窗口中的 first-person persona 只是语境构造的一部分；它不自动确立第十章中的持续身份或法律主体。

## 9.5 模型概率不蕴含真值

固定非空有限闭合句子集 $\Phi$。语言模型原生地给 token 序列分配概率；若评测协议为每个句子固定一个或多个字符串实现，并在有限候选集上聚合、归一化，才得到本节使用的概率质量函数

$$
q:\Phi\to[0,1],
\qquad
\sum_{\chi\in\Phi}q(\chi)=1.
$$

因此 $q$ 是一个明确协议诱导的句子分布，不是把语义等价类无条件当作模型原生样本空间。固定结构 $\mathcal M$ 后，真值函数为
$v_\mathcal M:\Phi\to\{0,1\}$。

**命题 9.1（无桥接假设时句子概率与真值无蕴含）.** 假设存在
$\varphi,\psi\in\Phi$ 使
$v_\mathcal M(\varphi)=0$、$v_\mathcal M(\psi)=1$。则存在 $\Phi$ 上概率质量函数 $q$，使下列蕴含为假：

$$
q(\varphi)>q(\psi)
\Longrightarrow
v_\mathcal M(\varphi)\ge v_\mathcal M(\psi).
$$

特别地，概率质量公理本身不能把
$q(\varphi)=1$ 桥接为 $\mathcal M\models\varphi$。

**证明.** 定义 $q(\varphi)=1$，并对所有
$\chi\in\Phi\setminus\{\varphi\}$ 令 $q(\chi)=0$。有限和为 $1$，故 $q$ 是合法概率质量函数。尤其
$q(\varphi)=1>0=q(\psi)$，但
$v_\mathcal M(\varphi)=0<1=v_\mathcal M(\psi)$，所以显示的排序蕴含失败；同一个 $q$ 也给出概率一的假句。证毕。

若另有经验证的校准、可靠性或证据模型，可得到统计性结论；这些是额外经验前提，不是 token 概率定义的一部分。

## 9.6 真值与核验状态

真值条件与观察者当前证据不同。若固定核验协议 $\Pi$ 是确定且保证终止的，则可定义总函数

$$
\operatorname{Verify}_\Pi:
\operatorname{Claim}\to
\{\operatorname{Supported},
\operatorname{Refuted},
\operatorname{Unknown},
\operatorname{OutOfScope}\}.
$$

若协议可能挂起、请求人工分歧裁决或保留随机选择，则接口应分别写成部分函数、关系或随机核，而不能仍称总函数。上式函数值是**协议相对的核验状态**，不是无条件真值 oracle。协议 $\Pi$ 至少应记录：

1. 规范化后的主张及其原文跨度；
2. 时间、法域、对象范围和量词；
3. 允许的证据源与版本；
4. 证据到状态的推理规则；
5. 冲突、缺失和测量误差的处理；
6. 审阅者、自动工具和不确定性记录。

自动评分器输出只是协议中的证据或决定步骤；若它本身会错，需要单独校准。

固定贯穿案例的核验协议 $\Pi_\star$：$\varphi_f$ 由查询活动
$a_q$ 产生的带时点快照 $e_f$ 核验；$\varphi_w$ 由文件提交记录 $e_k$ 和提交后的对象读取核验。在第二次写入 attempt 返回后，协议给出

$$
\operatorname{Verify}_{\Pi_\star}(\varphi_f)
=\operatorname{Supported},
\qquad
\operatorname{Verify}_{\Pi_\star}(\varphi_w)
=\operatorname{Supported}.
$$

但在第一次确认丢失、重试尚未完成的时点，$\varphi_w$ 在虚构世界中已经为真，控制器的证据却只足以返回
$\operatorname{Unknown}$。真值不随控制器是否收到确认而改变，核验状态会随证据史改变。若随后读回的是另一个命名空间中的同名路径，协议也不能把它当作 $f_\star$ 的证据。

## 9.7 时间、变化与版本

“模型 X 最快”“现任负责人是 Y”都隐含评价时刻、比较类、工作负载与来源版本。把陈述形式化为谓词时，应显式写入参数，例如

$$
\operatorname{Throughput}(m,h,b,p,t),
$$

其中 $m$ 为模型，$h$ 为硬件，$b$ 为批大小，$p$ 为测量协议，$t$ 为时间。省略参数的自然语言可能仍可交流，但其核验对象尚未封闭。

## 9.8 引用、使用与元语言

对象语言句子 $\varphi$ 与谈论其字符串或语法树的元语言名称不同。“雪是白的”用于作出断言；“被引字符串‘雪是白的’由四个汉字组成”谈论表达式本身。在这里，不计外层引号，内层字符串恰含“雪”“是”“白”“的”四个汉字，也对应四个 Unicode 标量值。代码围栏、引号和反引号改变层级，不能把字符串长度、模型概率或字体属性误归给命题指称。

## 9.9 指称失败与语义欠定

解析成功后仍可能：

- 名称没有指称；
- 一个描述有多个候选对象；
- 比较词缺少比较类；
- 量词范围不明；
- 事实标准本身有争议。

这些情形不应强行二值化。可以让语义解释为部分函数、关系或超估值等更丰富结构；本书不选定完整自然语言真值理论，只要求在事实核验前暴露欠定参数。

## 9.10 引用证据的限度

一条来源可支持“某机构在某版本文档中作出某陈述”，但不必直接支持该陈述在现实中为真。二手摘要、网页快照、数据库测量和专家判断具有不同证据角色。provenance 证明证据从何而来，证据评价决定它对主张支持多强；二者不能合并。

贯穿案例到此已经从字节推进到两个闭合主张：$\varphi_f$ 的指称依赖航班日期与查询时点，$\varphi_w$ 的指称依赖文件实体与提交时点；二者分别由不同来源证据支持。第五章的文本概率即使等于一，也不能代替这些桥接条件。现在还剩最后一种常见跃迁：从“哪些活动和主体参与了生成”跳到“谁是作者、谁应获信用、谁应负责”。下一章保留描述事实，同时把决定这些分类的规范体系单独写出。

## 练习

**练习 9.1.** 给出字符串相同但指称不同的两个完整语境，并写出变化的语境分量。

**练习 9.2.** 给出字符串不同但在指定模型类中逻辑等价的两个公式，写全结构与赋值量词。

**练习 9.3.** 为“它现在很快”补全对象、说话时间、比较类、测量协议和阈值。

**练习 9.4.** 用一个有限句子集构造“高模型概率但为假”的反例，并说明加入什么桥接假设后才可能作统计推断。

**练习 9.5.** 为含三个事实主张的文本设计核验表，区分真值条件、证据、协议状态和审阅结论。


## 真值审计接口

### S5.5 真值和核验状态


最终文本可以表达多个主张：

1. SP404 已取消；
2. 系统已写入 `trip.md`。

二者需要不同证据。第一项需要航班状态来源；第二项需要文件系统或制品提交证据。核验状态可分为：

- Supported；
- Refuted；
- Unknown；
- OutOfScope。

概率分数、token logprob 或模型自述不能替代证据协议。

### S6.10 真值、核验与语境


文本表达主张时必须给出语境和指称。`SP404 已取消` 需要说明航班号、日期、查询时间和来源。核验状态可取：

| 状态 | 含义 |
| --- | --- |
| Supported | 指定证据支持该主张 |
| Refuted | 指定证据反驳该主张 |
| Unknown | 证据不足 |
| OutOfScope | 本审计不处理该主张 |

模型 logprob、自然语言自信和用户信任都不是核验状态。
