# 卷四 术语表

## 解释篇

| 术语 | 本书口径 |
|---|---|
| target system | 模型版本、输入/环境/输出可测空间、确定可测映射或随机核、可测指标和乘积空间目标分布组成的研究对象 |
| explanandum / 被解释项 | 目标系统中范围明确的待解释现象、规律或计算结构；它不是研究者向该现象提出的问题 |
| explanation question / 解释问题 | 针对被解释项提出的“为什么/如何”问题；应声明所寻求的功能、因果、机制或语义关系，以及评价候选回答的证据标准 |
| kernel-averaged metric | 随机系统中对输出核积分得到的标量 $\overline m_\theta(x,e)=\int m(y)K_\theta((x,e),\mathrm{d}y)$；不把概率测度直接代入 $m$ |
| evidence operator | 把候选假说映到指定协议可观察总体量的映射 $\mathcal O_P$ |
| identifiability | 全局识别指证据算子在假说类上单射；点识别指指定假说的纤维为单点；商空间识别只确定等价类 |
| observational equivalence | 两系统在指定测试集或观察协议上不可区分；不等于全域函数或机制相同 |
| attribution | 在固定目标标量、基线、路径或合作规则下，把差异分配给坐标/组件的规则 |
| saliency | 局部敏感度或显著性数值，不预设因果语义 |
| completeness | 路径归因总和等于端点输出差；是守恒恒等式，不是语义唯一性 |
| attention weight | 指定头、查询位置和掩码下值向量的归一化混合系数 |
| direct value readout | 固定注意力权重与线性读出时的项 $\alpha_j u^\top W_Ov_j$ |
| probe | 从内部表示预测外部目标属性的辅助模型，必须连同探针类、训练算法和分布说明；命题 5.3 专指全体仿射分数探针类 $\mathcal P_{\mathrm{aff}}$ |
| decodability | 某属性在指定探针类、分布、损失和阈值下可预测 |
| control task | 与真实任务共享接口、但按设计主要测量探针记忆或容量的辅助任务 |
| intervention | 用新方程替换计算图中一个或多个结构方程，再重算后代节点 |
| source/base | patching 中提供激活的运行/接收替换激活的运行；不必分别等于正确/错误输入 |
| activation patching | 用 source 运行的激活替换 base 运行的指定节点，并测量结果变化 |
| interchange intervention | 在低层模型中交换表示，以检验其是否具有对齐高层变量的干预行为 |
| causal tracing | 通过一组内部替换定位对指定输出指标有局部模型内因果作用的状态 |
| component granularity | 电路研究预先选择的节点或已类型化边消息；两种粒度不得在同一 $\Gamma$ 中含混混用 |
| $A_b(S)$ | 同时按冻结替换规则 $b$ 消融组件集 $S\subseteq\Gamma$ 并重算后代的算子；$A_b(C)$ 测候选集删除响应，$A_b(\Gamma\setminus C)$ 测只保留候选集时的充分性 |
| circuit | 在声明分布、指标、组件粒度、替换规则与干预族内满足保留充分性、删除响应和未见干预预测要求的组件假说 |
| feature | 内部激活方向、稀疏坐标或可操作属性的建模单位；不预设自然概念身份 |
| SAE | sparse autoencoder，按重构与稀疏代理目标学习字典的模型 |
| feature absorption | 语义上应触发某特征的样本由其他更专门特征吸收，导致该特征漏触发的经验现象 |
| faithfulness | 解释对已声明计算量或干预响应的保真关系；必须给判据和评估分布 |
| stability | 解释对基线、seed、输入、checkpoint 或等价参数化变化的保持程度 |
| plausibility | 解释对人类是否自然或有说服力；不等同于 faithfulness |
| metric emergence | 报告指标在有限观测尺度间出现跃升 |
| mechanism transition | 内部计算假说随尺度发生可识别的定性变化，证据责任强于指标跃升 |
| operational label | 由明确情境、指标、干预和判据支持的心理词简称，不自动成为内在属性 |
| visible CoT | 会进入后续自回归上下文的可见 token 序列；其因果作用、报告忠实性和隐藏计算对应关系是不同问题 |
| hallucination | 输出与选定事实标准、来源或任务约束不一致的工程类别；报告时仍需细分检索失败、引用伪造、实体混淆等机制 |
| optimization / strategy / intention | 训练算法使用的优化标量、从行为拟合的策略目标与涉及主体归属的意图；三者不得互换 |
| deception protocol | 同时检查错误陈述、真值区分能力以及对受众、监督或后果的策略性反事实依赖；通过协议不自动裁决主体本体论 |
| $F_\theta$ | 固定参数确定模型的可测输入输出映射 |
| $h_l(x)$ | 输入 $x$ 在层 $l$ 的内部表示 |
| $m(y)$ | 作用于输出 $y\in\mathcal Y$ 的可测标量指标，如 logit 差 |
| $I_i$ | 对第 $i$ 个变量的干预 |

“表示”“概念”“机制”“理解”“意图”和“欺骗”都不是无条件的本体标签。正文使用时会说明操作定义、证据层级和未支持的升级。

## 验证篇

| 术语 | 本书口径 |
|---|---|
| sentence / 句子 | 语言中的语法表达；可能歧义、含自由变量或依赖语境 |
| formula / 公式 | 由指定形式语言生成规则构造的合式表达式 |
| context / 语境 | 消解指称、时间、比较类、版本和术语含义所需信息 |
| proposition / 命题 | 在固定语境与解释下可评价的陈述内容 |
| interpretation / model / 解释或模型 | 为语言符号提供论域及指称，并据此定义满足关系的结构 |
| satisfaction / 满足 | 解释使公式为真的关系，写作 $\mathcal M\models A$ |
| premise / 前提 | 论证暂时接受为起点的主张 |
| conclusion / 结论 | 论证目标主张 |
| argument / 论证 | 前提、结论及声称支持结论的理由结构 |
| semantic consequence / 语义蕴涵 | 每个满足前提的目标模型都满足结论，写作 $\Gamma\models A$ |
| derivability / 可推导性 | 在指定系统 $D$ 中存在从前提到结论的有限推导，写作 $\Gamma\vdash_D A$ |
| validity / 有效性 | 演绎论证不存在前提全真而结论假的目标模型 |
| sound argument / 健全论证 | 相对于目标解释，论证有效且前提实际为真 |
| sound calculus / 可靠演算 | $\Gamma\vdash_D A$ 总能推出 $\Gamma\models A$ |
| complete calculus / 完备演算 | 目标语义中的 $\Gamma\models A$ 总能推出 $\Gamma\vdash_D A$ |
| derivation / 推导 | 由带未释放假设上下文的判断、公理叶与规则实例组成的有限语法对象 |
| proof / 证明 | 在明确前提与标准下关闭演绎责任的推导或可恢复论证 |
| proof obligation / 证明义务 | 为使目标主张进入合法终态而必须完成的合式性、演绎、计算、经验或来源子任务 |
| satisfiable / 可满足 | 存在一个模型使公式集全部为真 |
| countermodel / 反模型 | 使前提真而目标结论假的模型 |
| type / sort / 类型或论域 | 项可取值的对象类及允许的操作；类型正确不等于命题为真 |
| free variable / 自由变量 | 不在相应量词作用域内的变量 |
| quantifier scope / 量词作用域 | $\forall$ 或 $\exists$ 实际约束的子公式范围 |
| necessary condition / 必要条件 | $A$ 是 $B$ 的必要条件表示 $B\to A$ |
| sufficient condition / 充分条件 | $A$ 是 $B$ 的充分条件表示 $A\to B$ |
| witness / 见证 | 使存在命题成立的具体对象 |
| well-founded / 良基 | 不存在无限严格下降链；授权归纳与递归终止的一类结构 |
| induction / 归纳 | 由基例和对构造/后继的闭合建立整个良基生成类上的性质 |
| convergence mode / 收敛模式 | 点态、一致、几乎处处、依概率、$L^p$ 等互不自动等价的极限关系 |
| external input / 外部输入 | 正文使用但不在书内证明的精确结果，须登记版本、假设、用途与来源 |
| explanans / 解释项 | 用于回答为何/如何的规律、条件、结构或机制 |
| explanatory proof / 解释性证明 | 同时完成证明并暴露与指定解释问题相关结构的证明 |
| mathematical explanation / 数学解释 | 对数学对象的结构、依赖、统一或生成方式的说明；不都承担证明责任 |
| scientific explanation / 科学解释 | 使用规律、因果、机制、统一或模型回答经验现象为何/如何的说明 |
| structural causal model / 结构因果模型 | 由带状态空间的外生变量、内生变量、结构方程、父节点图及外生分布组成的因果模型 |
| causal intervention / 因果干预 | 干预在结构因果模型中的专门形式：替换目标结构方程并保持其余机制不变，写作 $\operatorname{do}(X=x)$ |
| identification / 识别 | 目标因果或统计量是否由可观测分布与公开假设唯一决定 |
| likelihood / 似然 | 固定数据后，密度 $p_\theta(d)$ 作为参数 $\theta$ 的函数 |
| p-value / p 值 | 在零假设下，观察到至少同样极端统计量的尾部概率口径；不是零假设为真的后验概率 |
| confidence set / 置信集 | 重复抽样覆盖率由程序保证的随机参数集合 |
| posterior / 后验 | 在先验与似然组成的联合概率模型内，对参数的条件分布 |
| evidence / 证据 | 在明确测量与推断框架中支持或削弱经验主张的观察 |
| object diagram / 对象图 | 节点和箭头解释为数学对象与关系的图；交换性仍须建立 |
| data graphic / 数据图 | 把观测或估计编码为位置、颜色、面积等视觉通道的图 |
| analogy / 类比 | 从源域到目标域的部分结构映射，只转移已证明保留的关系 |
| rhetoric / 修辞 | 面向受众组织表达与说服的选择；不增加逻辑证明力 |
| source support / 来源支持 | 来源中的结果与正文主张之间的转述、蕴涵或经验外推关系 |
| correctness relation / 正确性关系 | 任务预先定义的 $R\subseteq X\times Y$，用于判断输出是否正确 |
| certificate / 证书 | 可由检查器读取、用于证明输出满足某关系的形式对象 |
| verifier / 验证器 | 对预先声明的输入类型给出接受/拒绝的程序或形式过程；本书分别使用 $(x,y,\pi)$ 三元验证器与 $(\widehat x,\pi)$ 形式证明检查器 |
| verifier soundness / 验证器可靠性 | $V(x,y,\pi)=1$ 必然推出目标关系 $R(x,y)$ |
| process faithfulness / 过程忠实性 | 可见理由与产生答案的实际过程之间、按指定抽象与因果口径成立的对应 |

本书中“证明”只用于演绎责任已关闭的情形；统计支持、因果识别和过程忠实性均使用各自的对象与保证。
