# 术语表

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
| explanandum / 被解释项 | 解释任务中所问的现象、规律或结构 |
| explanans / 解释项 | 用于回答为何/如何的规律、条件、结构或机制 |
| explanatory proof / 解释性证明 | 同时完成证明并暴露与指定解释问题相关结构的证明 |
| mathematical explanation / 数学解释 | 对数学对象的结构、依赖、统一或生成方式的说明；不都承担证明责任 |
| scientific explanation / 科学解释 | 使用规律、因果、机制、统一或模型回答经验现象为何/如何的说明 |
| structural causal model / 结构因果模型 | 由带状态空间的外生变量、内生变量、结构方程、父节点图及外生分布组成的因果模型 |
| intervention / 干预 | 替换目标结构方程并保持其余机制不变的模型操作，写作 $\operatorname{do}(X=x)$ |
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
| plausibility / 可信感 | 文本对读者是否自然可信；不等于有效、真实或得到证据支持 |

本书中“证明”只用于演绎责任已关闭的情形；统计支持、因果识别和过程忠实性均使用各自的对象与保证。
