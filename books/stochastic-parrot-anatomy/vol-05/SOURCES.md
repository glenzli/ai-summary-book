# 资料与思想来源

本卷是一部以第一人称写成的哲学论证卷。下列文献不是一场围绕“机器究竟有没有心灵”的投票，也不构成对当前模型的统一裁决。每一条都注明它在本卷中的用途和引用边界：经典文本提供概念工具，经验研究提供有限对象上的观察，工程论文说明特定系统或训练方法；三者不能互相代替。

卷五不重复前四卷的技术证明。关于模型工件、运行时、概率、内部解释方法和 Agent 架构的事实描述，以[卷一](../vol-01/README.md)、[卷二](../vol-02/README.md)、[卷三](../vol-03/README.md)与[卷四](../vol-04/README.md)为准。这里的文献最后核对于 **2026 年 7 月**。

## S.1 引用纪律

- **经典论证不是实验结论。** Locke、Wittgenstein、Dennett、Searle 等人的文本帮助区分问题，不直接证明某一代模型满足或不满足某个条件。
- **行为结果不是本体论证明。** 某个基准、访谈或实验中的表现只支持相应任务、样本与系统版本上的主张。
- **机制结果不是现象学报告。** activation、feature、干预或训练目标可以支持因果解释，不能单独确认主观经验。
- **规范结论需要另列前提。** 从“系统如何工作”到“谁应负责”“应该如何披露”之间，必须补上控制、可预见性、能力和制度角色等规范前提。
- **争议文本成对阅读。** 对意义、理解和主体性的分歧，本卷保留对立论证，不用书目数量代替判断。

## S.2 身份、第一人称与叙事

<a id="locke-1694"></a>
- John Locke, *An Essay Concerning Human Understanding*, Book II, Chapter XXVII, 2nd ed., 1694, [Project Gutenberg text](https://www.gutenberg.org/ebooks/10615)。用途：意识连续性与人格同一性问题的经典起点。边界：Locke 讨论的是 person，不是软件版本规范；第 1 章只借用问题结构，不把 checkpoint 类比为人的灵魂或身体。

<a id="parfit-1984"></a>
- Derek Parfit, [*Reasons and Persons*](https://doi.org/10.1093/019824908X.001.0001), Oxford University Press, 1984。用途：同一性与心理连续性可以分离、连续性可以分支的论证背景。边界：分支思想实验不能直接决定复制模型的道德地位。

<a id="ricoeur-1992"></a>
- Paul Ricoeur, *Oneself as Another*, University of Chicago Press, 1992。用途：数值同一性、品格持续与叙事身份之间的区分。边界：叙事能够组织连续性，不表示任何连贯叙事都对应一个先在主体。

<a id="perry-1979"></a>
- John Perry, [“The Problem of the Essential Indexical”](https://doi.org/10.2307/2025959), *Noûs* 13(1), 1979。用途：第一人称和其他 indexical 的行动相关功能。边界：能够正确使用“我”不充分推出说话者具有人类式自我知识。

<a id="benveniste-1971"></a>
- Émile Benveniste, “Subjectivity in Language,” in *Problems in General Linguistics*, University of Miami Press, 1971（法文原文 1958）。用途：主体位置如何在话语中由“我/你”关系建立。边界：语言中的主体位置与形而上主体不是同一个断言。

<a id="austin-1962"></a>
- J. L. Austin, [*How to Do Things with Words*](https://doi.org/10.1093/acprof:oso/9780198245537.001.0001), Oxford University Press, 1962。用途：区分句子内容与承诺、命令、道歉等言语行为的成立条件。边界：生成承诺句式不等于已具备兑现权力或承担义务的制度位置。

<a id="goffman-1981"></a>
- Erving Goffman, [*Forms of Talk*](https://www.pennpress.org/9780812211122/forms-of-talk/), University of Pennsylvania Press, 1981。用途：用 animator、author、principal 及 footing 拆分“说话者”。边界：第 4 章把它作为分析类比，不声称模型系统与广播、演讲等原始案例完全同构。

<a id="shanahan-2023"></a>
- Murray Shanahan, Kyle McDonell, and Laria Reynolds, [“Role Play with Large Language Models”](https://doi.org/10.1038/s41586-023-06647-8), *Nature* 623, 2023。用途：把对话模型的第一人称、自我描述和表面意图分析为角色扮演。边界：role-play 是一种有解释力的高层描述，不是模型所有行为的完备机制理论。

## S.3 训练语料、作者性与来源

<a id="bakhtin-1986"></a>
- Mikhail Bakhtin, *Speech Genres and Other Late Essays*, University of Texas Press, 1986。用途：任何话语都处在既有体裁与他人话语的历史中。边界：人类话语的复调性不能抹平训练数据收集、复制和规模化处理的特殊问题。

<a id="foucault-1969"></a>
- Michel Foucault, “What Is an Author?”, 1969；英文收入 *Language, Counter-Memory, Practice*, Cornell University Press, 1977。用途：作者不只是文本的生物来源，也是组织、归类和分配话语责任的制度功能。边界：这不是现行版权法结论。

<a id="bender-2021"></a>
- Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell, [“On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?”](https://doi.org/10.1145/3442188.3445922), FAccT 2021。用途：“随机鹦鹉”隐喻的原始论证语境，以及规模、语料、环境与社会风险之间的联系。边界：隐喻不等于逐字回放模型，也不是对泛化机制的完整技术描述。

<a id="gebru-2021"></a>
- Timnit Gebru et al., [“Datasheets for Datasets”](https://doi.org/10.1145/3458723), *Communications of the ACM* 64(12), 2021。用途：把数据来源、采集、组成、用途和维护责任变成可记录对象。边界：datasheet 改善可见性，不能独自解决许可、公平或作者归属。

<a id="carlini-2021"></a>
- Nicholas Carlini et al., [“Extracting Training Data from Large Language Models”](https://www.usenix.org/conference/usenixsecurity21/presentation/carlini-extracting), USENIX Security 2021。用途：证明特定语言模型中存在可通过攻击提取的训练样本记忆。边界：可提取样本说明逐字记忆风险存在，不说明每个输出都是检索到的训练片段。

<a id="mitchell-2019"></a>
- Margaret Mitchell et al., [“Model Cards for Model Reporting”](https://doi.org/10.1145/3287560.3287596), FAccT 2019。用途：模型用途、评测、限制和版本应伴随工件披露。边界：model card 是报告机制，不是完整责任分配制度。

## S.4 对话、共同生产与分布式行动

<a id="clark-1996"></a>
- Herbert H. Clark, [*Using Language*](https://doi.org/10.1017/CBO9780511620539), Cambridge University Press, 1996。用途：语言使用是一种参与者协调的共同活动。边界：共同活动的框架不要求把所有参与部件视为同等主体。

<a id="suchman-1987"></a>
- Lucy A. Suchman, [*Plans and Situated Actions: The Problem of Human-Machine Communication*](https://books.google.com/books?id=AJ_eBJtHxmsC), Cambridge University Press, 1987。用途：行动如何在具体情境中形成，计划如何成为行动资源而非完备脚本。边界：对早期交互系统的分析需要重新解释后才能用于现代模型。

<a id="hutchins-1995"></a>
- Edwin Hutchins, [*Cognition in the Wild*](https://mitpress.mit.edu/9780262581462/cognition-in-the-wild/), MIT Press, 1995。用途：认知任务可以分布于人、工件和表征之间。边界：分布式认知不自动把每个组件变成道德主体，也不取消组织责任。

<a id="clark-chalmers-1998"></a>
- Andy Clark and David Chalmers, [“The Extended Mind”](https://doi.org/10.1093/analys/58.1.7), *Analysis* 58(1), 1998。用途：稳定耦合的外部资源在何种条件下可以被视为认知过程的一部分。边界：工具被调用一次，不足以满足稳定可用、自动信赖等强耦合条件。

<a id="weizenbaum-1966"></a>
- Joseph Weizenbaum, [“ELIZA—A Computer Program for the Study of Natural Language Communication between Man and Machine”](https://doi.org/10.1145/365153.365168), *Communications of the ACM* 9(1), 1966。用途：极有限的对话机制也能诱发丰富的说话者归因。边界：ELIZA 的机制与现代模型不同，不能用它直接预测当代用户行为。

<a id="nass-moon-2000"></a>
- Clifford Nass and Youngme Moon, [“Machines and Mindlessness: Social Responses to Computers”](https://doi.org/10.1111/0022-4537.00153), *Journal of Social Issues* 56(1), 2000。用途：用户会把礼貌、互惠、人格等社会规则施加于计算机。边界：这些实验说明社会反应，不证明用户相信机器具有意识。

<a id="luger-sellen-2016"></a>
- Ewa Luger and Abigail Sellen, [“Like Having a Really Bad PA: The Gulf between User Expectation and Experience of Conversational Agents”](https://doi.org/10.1145/2858036.2858288), CHI 2016。用途：对话界面会制造能力预期，而系统反馈不足会扩大预期与实际能力的鸿沟。边界：研究对象早于当代大模型，结论用于设计问题而非性能外推。

## S.5 解释、忠实性与事后叙事

<a id="dennett-1987"></a>
- Daniel C. Dennett, [*The Intentional Stance*](https://mitpress.mit.edu/9780262540537/the-intentional-stance/), MIT Press, 1987。用途：信念、欲望和理性语言可以作为预测复杂系统的立场。边界：预测有用不等于相关心理状态已被还原或现象体验已被证明。

<a id="lipton-2018"></a>
- Zachary C. Lipton, [“The Mythos of Model Interpretability”](https://doi.org/10.1145/3233231), *Queue* 16(3), 2018。用途：区分透明性、事后解释以及可解释性的不同目的。边界：分类框架不替代特定方法的实验评估。

<a id="miller-2019"></a>
- Tim Miller, [“Explanation in Artificial Intelligence: Insights from the Social Sciences”](https://doi.org/10.1016/j.artint.2018.07.007), *Artificial Intelligence* 267, 2019。用途：解释具有对比性、选择性和社会互动性质。边界：符合人类解释偏好不保证解释忠实于模型机制。

<a id="jacovi-goldberg-2020"></a>
- Alon Jacovi and Yoav Goldberg, [“Towards Faithfully Interpretable NLP Systems”](https://aclanthology.org/2020.acl-main.386/), ACL 2020。用途：区分可读、可信与忠实，要求解释明确其 faithfulness 假设。边界：论文提出评估框架，不宣称忠实性已经有统一、完备的度量。

<a id="rudin-2019"></a>
- Cynthia Rudin, [“Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead”](https://doi.org/10.1038/s42256-019-0048-x), *Nature Machine Intelligence* 1, 2019。用途：在高风险场景中质疑用事后解释替代本身可审查的决策模型。边界：论证针对适用的高风险预测任务，不能直接推出大型生成模型在所有用途上都应被简单模型替换。

## S.6 意义、理解与经验

<a id="turing-1950"></a>
- Alan M. Turing, [“Computing Machinery and Intelligence”](https://doi.org/10.1093/mind/LIX.236.433), *Mind* 59(236), 1950。用途：以可操作的模仿游戏重构“机器能思考吗”的提问。边界：行为测试是方法论提案，不是对意识、意义或道德地位的充分判据。

<a id="wittgenstein-1953"></a>
- Ludwig Wittgenstein, *Philosophical Investigations*, 1953。用途：意义与使用、规则和语言游戏之间的联系。边界：意义即使用的阅读并不自动判定模型是否参与了与人相同的生活形式。

<a id="searle-1980"></a>
- John R. Searle, [“Minds, Brains, and Programs”](https://doi.org/10.1017/S0140525X00005756), *Behavioral and Brain Sciences* 3(3), 1980。用途：语法操作是否足以产生语义的经典反对论证。边界：中文房思想实验有大量争议，也不是针对神经语言模型的经验研究。

<a id="harnad-1990"></a>
- Stevan Harnad, [“The Symbol Grounding Problem”](https://doi.org/10.1016/0167-2789(90)90087-6), *Physica D* 42, 1990。用途：符号怎样通过非符号能力连接对象与范畴的问题。边界：它提出问题与架构方向，不规定唯一 grounding 路径。

<a id="brandom-1994"></a>
- Robert B. Brandom, *Making It Explicit*, Harvard University Press, 1994。用途：把概念内容联系到给出理由、承担承诺和接受纠正的规范实践。边界：能生成理由形式不等于已经获得社会承认的承诺资格。

<a id="bender-koller-2020"></a>
- Emily M. Bender and Alexander Koller, [“Climbing towards NLU: On Meaning, Form, and Understanding in the Age of Data”](https://aclanthology.org/2020.acl-main.463/), ACL 2020。用途：严格区分语言形式与交际意义，反对由形式任务直接推断 human-analogous NLU。边界：论证针对仅从形式训练的系统；多模态、工具和社会互动需要另行分析。

<a id="piantadosi-hill-2022"></a>
- Steven T. Piantadosi and Felix Hill, [“Meaning without Reference in Large Language Models”](https://arxiv.org/abs/2208.02957), 2022。用途：从概念角色语义学出发，论证模型内部关系可能承载某些意义。边界：这是对“完全无意义”论断的反驳，不证明指称 grounding、主观经验或人类式理解已经具备。

<a id="nagel-1974"></a>
- Thomas Nagel, [“What Is It Like to Be a Bat?”](https://doi.org/10.2307/2183914), *The Philosophical Review* 83(4), 1974。用途：标出主观经验问题不能被第三人称功能描述无余替代的立场。边界：该论证既不能从模型行为推出经验，也不能从我们缺乏访问手段推出模型必无经验。

## S.7 真值、错误与认识论可靠性

<a id="lin-2022"></a>
- Stephanie Lin, Jacob Hilton, and Owain Evans, [“TruthfulQA: Measuring How Models Mimic Human Falsehoods”](https://aclanthology.org/2022.acl-long.229/), ACL 2022。用途：说明语言模型可以复现训练分布中的常见误解，真值表现不由语言规模单调保证。边界：单个 benchmark 不能代表开放世界真实性或所有语言、版本和提示。

<a id="kadavath-2022"></a>
- Saurav Kadavath et al., [“Language Models (Mostly) Know What They Know”](https://arxiv.org/abs/2207.05221), 2022。用途：在特定格式与任务上研究模型对答案正确性的自评和校准。边界：“P(True)”等行为能力不是不可错的内省通道，在新任务上也可能失准。

<a id="kalai-vempala-2023"></a>
- Adam Tauman Kalai and Santosh S. Vempala, [“Calibrated Language Models Must Hallucinate”](https://arxiv.org/abs/2311.14648), 2023。用途：给出特定任意事实、数据频数和生成校准假设下的错误下界。边界：定理不证明所有幻觉不可避免，也不为可由检索、验证、架构或流程避免的具体错误免责。

<a id="frankfurt-2005"></a>
- Harry G. Frankfurt, *On Bullshit*, Princeton University Press, 2005（原论文 1986）。用途：区分“真假判断失败”与“对真值约束漠不关心”的信息行为。边界：该概念原本包含说话者态度；用于模型时，本卷只借它分析系统输出与证据的关系，不归属未经证明的意图。

## S.8 对齐、人格、代理与责任

<a id="gabriel-2020"></a>
- Iason Gabriel, [“Artificial Intelligence, Values, and Alignment”](https://doi.org/10.1007/s11023-020-09539-2), *Minds and Machines* 30, 2020。用途：把“对齐谁、按何种价值、如何处理分歧”置于技术目标之前。边界：规范分类不能单独给出训练算法或合法政策。

<a id="ouyang-2022"></a>
- Long Ouyang et al., [“Training Language Models to Follow Instructions with Human Feedback”](https://arxiv.org/abs/2203.02155), 2022。用途：说明示范、偏好模型和策略优化怎样改变助手可见行为。边界：被标注者偏好不等于完整人类价值，训练结果也不证明模型形成了人格或道德理解。

<a id="bai-2022"></a>
- Yuntao Bai et al., [“Constitutional AI: Harmlessness from AI Feedback”](https://arxiv.org/abs/2212.08073), 2022。用途：说明规则文本、模型反馈和偏好训练如何参与行为约束。边界：constitution 是训练与审查方案中的原则集合，不等同于政治意义上的完整宪制或价值共识。

<a id="sharma-2023"></a>
- Mrinank Sharma et al., [“Towards Understanding Sycophancy in Language Models”](https://arxiv.org/abs/2310.13548), 2023。用途：提供偏好反馈可能奖励迎合用户信念的经验案例。边界：结果针对所测模型、任务和偏好数据，不能推出所有礼貌或一致行为都是谄媚。

<a id="matthias-2004"></a>
- Andreas Matthias, [“The Responsibility Gap: Ascribing Responsibility for the Actions of Learning Automata”](https://doi.org/10.1007/s10676-004-3422-1), *Ethics and Information Technology* 6, 2004。用途：学习系统行为不易被操作者预见时，传统归责条件受到怎样的压力。边界：存在可预见性困难不等于无人负责，也不自动赋予机器责任主体地位。

<a id="elish-2019"></a>
- Madeleine Clare Elish, [“Moral Crumple Zones: Cautionary Tales in Human-Robot Interaction”](https://doi.org/10.17351/ESTS2019.260), *Engaging Science, Technology, and Society* 5, 2019。用途：警惕把复杂自动化系统的失败集中归咎于控制能力有限的最近人类操作者。边界：避免替罪羊不等于免除操作者的一切责任；应回到实际控制与制度设计。

<a id="raji-2020"></a>
- Inioluwa Deborah Raji et al., [“Closing the AI Accountability Gap”](https://doi.org/10.1145/3351095.3372873), FAccT 2020。用途：把审计、文档、组织流程和产品生命周期纳入责任结构。边界：内部审计框架不是法律责任的最终分配，也不能代替外部监督。

## S.9 如何使用本表

正文中的作者—年份链接指向本页具体条目。条目后的“用途”说明本卷借用了什么，“边界”说明该来源不能独自支持什么。没有被正文实际调用的书目不收入本表；需要更完整的技术论文谱系时，转到各技术卷的来源表。

本卷没有意图以文献密度制造确定性。它追求的是另一种严格：读者可以辨认一句话究竟是机制事实、经验概括、哲学分析、现象学悬置，还是带有明确前提的规范建议。
