# 来源与外部输入

本文件登记正文直接依赖或用于定位方法边界的一手论文、规范和公认专著。正文中的定义、短命题和案例若由本书承担，必须在书内完成；大型数学理论、协议事实和经验研究则作为外部输入，不因被概述而变成书内证明。

卷二至卷四另有按局部 locator 编排的来源表；根表主要服务卷一、卷五、证明附录和跨卷共同输入。同一来源若同时出现在局部表与根表，只表示两个可反查入口，不产生第二份独立证据，也不要求顺序阅读本文件。

本书不依赖模型排行榜或持续变化的产品能力。2024 年以后仍会变化的产品与协议事实若未来进入正文，必须记录固定版本、核验日期和官方来源。

## 一、学习、架构与训练

- Frank Rosenblatt, [*The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain*](https://doi.org/10.1037/h0042519), Psychological Review 65(6), 1958。用于附录 C 的感知机历史定位；有限步收敛定理由附录在可分与间隔假设下完整证明。
- David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams, [*Learning representations by back-propagating errors*](https://doi.org/10.1038/323533a0), Nature 323, 1986。用于卷一第 1--2 章反向传播的历史与方法定位；链式法则本身由正文直接使用。
- Diederik P. Kingma, Jimmy Ba, [*Adam: A Method for Stochastic Optimization*](https://arxiv.org/abs/1412.6980), ICLR 2015。用于附录 C 对 Adam 更新式及其状态变量的定位；书中不把算法定义升级为无条件收敛保证。
- Ilya Loshchilov, Frank Hutter, [*Decoupled Weight Decay Regularization*](https://arxiv.org/abs/1711.05101), ICLR 2019。用于附录 C 区分 AdamW 的解耦权重衰减与把二次惩罚直接加入自适应目标。
- George Cybenko, [*Approximation by Superpositions of a Sigmoidal Function*](https://doi.org/10.1007/BF02551274), Mathematics of Control, Signals and Systems 2, 1989。作为通用逼近理论的来源入口；正文不调用其定理作证明前提。
- Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton, [*ImageNet Classification with Deep Convolutional Neural Networks*](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html), NeurIPS 2012。用于 CNN 技术史背景。
- Sepp Hochreiter, Jurgen Schmidhuber, [*Long Short-Term Memory*](https://doi.org/10.1162/neco.1997.9.8.1735), Neural Computation 9(8), 1997。用于门控序列模型来源。
- Ashish Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), 2017。卷一第 3 章 Transformer、缩放点积注意力和多头结构的主要来源。
- Jacob Devlin et al., [*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*](https://arxiv.org/abs/1810.04805), 2018。用于 encoder-only 预训练接口。
- Tom B. Brown et al., [*Language Models are Few-Shot Learners*](https://arxiv.org/abs/2005.14165), NeurIPS 2020。用于 decoder-only 基座模型与 in-context 接口的历史定位；不据此外推当前模型能力。
- Colin Raffel et al., [*Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*](https://arxiv.org/abs/1910.10683), JMLR 21, 2020。用于 text-to-text 和 encoder-decoder 接口。
- Long Ouyang et al., [*Training language models to follow instructions with human feedback*](https://arxiv.org/abs/2203.02155), NeurIPS 2022。用于 SFT、奖励模型与 RLHF 训练链。
- John Schulman et al., [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347), 2017。用于卷一第 5 章和附录 C 的 PPO clipped surrogate 定义；书中明确不把 clipping 解释为逐状态硬约束或一般单调改进保证。
- Rafael Rafailov et al., [*Direct Preference Optimization: Your Language Model is Secretly a Reward Model*](https://arxiv.org/abs/2305.18290), NeurIPS 2023。用于 DPO 方法定位；正文不声称所有偏好优化等价。
- Edward J. Hu et al., [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685), ICLR 2022。卷一第 5 章低秩适配来源。
- Tim Dettmers et al., [*QLoRA: Efficient Finetuning of Quantized LLMs*](https://arxiv.org/abs/2305.14314), NeurIPS 2023。用于量化底模上的参数高效微调来源。
- Patrick Lewis et al., [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://arxiv.org/abs/2005.11401), NeurIPS 2020。用于 RAG 的检索-生成接口。
- Woosuk Kwon et al., [*Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://arxiv.org/abs/2309.06180), SOSP 2023。用于服务端 KV cache 与批处理背景。
- Albert Gu, Tri Dao, [*Mamba: Linear-Time Sequence Modeling with Selective State Spaces*](https://arxiv.org/abs/2312.00752), 2023。用于注意力之外的选择性状态空间路线；不将单篇结果推广到全部长序列任务。

## 二、生成模型、多模态与世界模型

- Ian Goodfellow et al., [*Generative Adversarial Nets*](https://proceedings.neurips.cc/paper/2014/hash/5ca3e9b122f61f8f06494c97b1afccf3-Abstract.html), NeurIPS 2014。卷一第 6 章对抗生成目标来源。
- Diederik P. Kingma, Max Welling, [*Auto-Encoding Variational Bayes*](https://arxiv.org/abs/1312.6114), ICLR 2014。用于潜变量生成与证据下界背景。
- Jonathan Ho, Ajay Jain, Pieter Abbeel, [*Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239), NeurIPS 2020。卷一第 6 章离散扩散过程来源。
- Alexey Dosovitskiy et al., [*An Image is Worth 16x16 Words*](https://arxiv.org/abs/2010.11929), ICLR 2021。用于视觉 Transformer 的接口背景。
- Alec Radford et al., [*Learning Transferable Visual Models From Natural Language Supervision*](https://arxiv.org/abs/2103.00020), ICML 2021。卷一第 6 章跨模态对比学习来源。
- Aaron van den Oord, Yazhe Li, Oriol Vinyals, [*Representation Learning with Contrastive Predictive Coding*](https://arxiv.org/abs/1807.03748), 2018。用于附录 C 的 InfoNCE 方法与互信息下界入口；书内只证明有限批次交叉熵梯度，一般下界按声明的正负样本概率实验外引。
- Robin Rombach et al., [*High-Resolution Image Synthesis with Latent Diffusion Models*](https://arxiv.org/abs/2112.10752), CVPR 2022。用于潜空间扩散和制品生成接口。
- David Ha, Jurgen Schmidhuber, [*World Models*](https://arxiv.org/abs/1803.10122), 2018。用于潜在动力学与控制的世界模型路线。
- Danijar Hafner et al., [*Dream to Control: Learning Behaviors by Latent Imagination*](https://arxiv.org/abs/1912.01603), ICLR 2020。用于潜在轨迹规划和闭环评测背景。

## 三、输出对象、语义与 provenance

- Unicode Consortium, [*The Unicode Standard*](https://www.unicode.org/standard/standard.html)，以及 F. Yergeau, [RFC 3629: *UTF-8, a transformation format of ISO 10646*](https://www.rfc-editor.org/rfc/rfc3629)。附录 E 把合法 UTF-8 域与唯一编解码作为固定版本外部规范输入；非法字节错误策略须另行声明。
- Unicode Consortium, [UAX #15: *Unicode Normalization Forms*](https://unicode.org/reports/tr15/), revision 57；[UAX #29: *Unicode Text Segmentation*](https://unicode.org/reports/tr29/), revision 47。用于附录 E 的 NFC/NFD 幂等性、规范等价见证和字素簇接口；现实记录还须保存 Unicode 版本、profile 与勘误状态。
- W3C, [*PROV-O: The PROV Ontology*](https://www.w3.org/TR/prov-o/), W3C Recommendation, 2013。用于 entity、activity、agent 及来源关系；本书 schema 是教学压缩，不宣称覆盖 PROV-O 全部语义。
- Leslie Lamport, [*Time, Clocks, and the Ordering of Events in a Distributed System*](https://doi.org/10.1145/359545.359563), Communications of the ACM 21(7), 1978。卷二第 7 章 happens-before 偏序来源。
- Marta Kwiatkowska, Gethin Norman, David Parker, [*Stochastic Model Checking*](https://doi.org/10.1007/978-3-540-85114-1_39), SFM 2007。作为随机转移系统与性质检查的扩展入口；正文只使用有限状态/事件接口。

## 四、概率、校准、复现与评测

- Olav Kallenberg, *Foundations of Modern Probability*, 3rd ed., Springer, 2021。卷三第 2--4、7 章及附录 D--E 的测度、核、条件期望、Ionescu--Tulcea 与随机化引理入口；尤其使用 kernels and randomization 的 Lemma 4.22、conditional distributions 的 Theorem 8.5 和 Ionescu--Tulcea 的 Theorem 8.24。本书没有重证这些一般存在定理。
- Rick Durrett, [*Probability: Theory and Examples*, 5th ed.](https://services.math.duke.edu/~rtd/PTE/PTE5_011119.pdf), 2019, Chapters 2--3。附录 D 的 i.i.d. 可积强大数律与有限方差中心极限定理的精确外部入口；两者不提供任意有限样本保证。
- Thomas M. Cover, Joy A. Thomas, *Elements of Information Theory*, 2nd ed., Wiley, 2006。卷三第 6 章熵、KL 与交叉熵来源入口。
- Tilmann Gneiting, Adrian E. Raftery, [*Strictly Proper Scoring Rules, Prediction, and Estimation*](https://doi.org/10.1198/016214506000001437), JASA 102(477), 2007。用于适当评分规则与校准边界。
- Chuan Guo et al., [*On Calibration of Modern Neural Networks*](https://proceedings.mlr.press/v70/guo17a.html), ICML 2017。用于神经网络经验校准与温度缩放背景。
- Wassily Hoeffding, [*Probability Inequalities for Sums of Bounded Random Variables*](https://doi.org/10.1080/01621459.1963.10500830), JASA 58(301), 1963。命题 10.1 的外部输入；正文完整说明其有界独立同分布版本的调用。
- IEEE, [*IEEE Standard for Floating-Point Arithmetic*](https://standards.ieee.org/ieee/754/6210/), IEEE 754-2019。卷三第 11 章浮点与归约次序的规范来源；正文不重建全部舍入规范。
- Nicholas J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002。附录 F 的标准舍入模型、$\gamma_k$ 记号与求和误差分析入口；附录只在明确的无溢出、无下溢或相应模型条件下使用。
- National Academies of Sciences, Engineering, and Medicine, [*Reproducibility and Replicability in Science*](https://doi.org/10.17226/25303), 2019。用于计算复现、科学复制和术语边界。
- ACM, [*Artifact Review and Badging*](https://www.acm.org/publications/policies/artifact-review-and-badging-current)。用于 artifact 可获得、功能与结果复现的出版实践词典；动态网页在具体引用时记录访问日期。
- Joint Committee for Guides in Metrology, [*International Vocabulary of Metrology, 3rd ed., JCGM 200:2012*](https://jcgm.bipm.org/vim/en/index.html)。用于区分 measurement repeatability 与 measurement reproducibility；不与 ACM 或 NASEM 的计算制品术语无来源混用。
- Donald J. Schuirmann, [*A Comparison of the Two One-Sided Tests Procedure and the Power Approach for Assessing the Equivalence of Average Bioavailability*](https://pubmed.ncbi.nlm.nih.gov/3450848/), 1987。附录 F 的 TOST 方法入口；附录在已知尺度精确正态模型中自行证明大小与区间对应。
- Sture Holm, *A Simple Sequentially Rejective Multiple Test Procedure*, *Scandinavian Journal of Statistics* 6(2), 1979, DOI `10.2307/4615733`。附录 F 的 Holm step-down 来源；任意依赖下的有限 FWER 结论由附录完整证明。
- Percy Liang et al., [*Holistic Evaluation of Language Models*](https://arxiv.org/abs/2211.09110), TMLR 2023。用于多维语言模型评测背景；正文不继承其全部指标体系。
- Rishi Bommasani et al., [*On the Opportunities and Risks of Foundation Models*](https://arxiv.org/abs/2108.07258), 2021。用于基座模型、能力与社会技术系统边界的背景来源。

## 五、解释与机制证据

- Mukund Sundararajan, Ankur Taly, Qiqi Yan, [*Axiomatic Attribution for Deep Networks*](https://proceedings.mlr.press/v70/sundararajan17a.html), ICML 2017。卷四第 4 章积分梯度来源。
- Sarthak Jain, Byron C. Wallace, [*Attention is not Explanation*](https://aclanthology.org/N19-1357/), NAACL 2019。用于注意力权重与解释边界的经验来源；不外推为“注意力永远无用”。
- John Hewitt, Percy Liang, [*Designing and Interpreting Probes with Control Tasks*](https://aclanthology.org/D19-1275/), EMNLP-IJCNLP 2019。用于探针控制任务和选择性边界。
- David Bau et al., [*Understanding the Role of Individual Units in a Deep Neural Network*](https://doi.org/10.1073/pnas.1907375117), PNAS 117(48), 2020。用于单元解释与干预背景。
- Kevin Meng et al., [*Locating and Editing Factual Associations in GPT*](https://arxiv.org/abs/2202.05262), NeurIPS 2022。用于因果追踪与模型编辑实验来源；不将局部结果外推为完整事实存储理论。
- Nelson Elhage et al., [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html), 2021。用于 Transformer 电路分析的研究框架。
- Trenton Bricken et al., [*Towards Monosemanticity: Decomposing Language Models With Dictionary Learning*](https://transformer-circuits.pub/2023/monosemantic-features/index.html), 2023。用于稀疏字典学习和特征解释的经验入口。
- Finale Doshi-Velez, Been Kim, [*Towards A Rigorous Science of Interpretable Machine Learning*](https://arxiv.org/abs/1702.08608), 2017。用于解释评测层次与方法学背景。

## 六、证明、来源与 AI 生成理由

- Michael Sipser, *Introduction to the Theory of Computation*, 3rd ed., Cengage, 2012。用于附录 F 的可判定性与停机问题背景；程序等价不可判定由附录给出归约。
- Gerald B. Folland, *Real Analysis: Modern Techniques and Their Applications*, 2nd ed., Wiley, 1999, Theorems 2.24 and 3.8。附录 H 以 Theorem 2.24 作为支配收敛定理的精确外部输入，并另给删除共同支配条件后的尖峰反例；附录 D 以 Theorem 3.8 的 Lebesgue--Radon--Nikodym 定理说明概率密度存在所需的绝对连续条件。
- Open Logic Project, [*The Open Logic Text*](https://builds.openlogicproject.org/open-logic-complete.pdf), release `9620cc7` (2026-07-12)。附录 H 对指定经典命题自然演绎系统调用 Theorem 10.22、Theorem 13.6、Corollary 13.7 与 Proposition 10.16；不把一个演算的元定理推广到任意验证器或自然语言推理。
- Judea Pearl, *Causality: Models, Reasoning, and Inference*, 2nd ed., Cambridge University Press, 2009。用于卷三第 8 章、卷四第 20 章及附录 H 的结构因果模型、观察与干预语义入口；本书不重证一般识别演算。
- Miguel A. Hernan, James M. Robins, *Causal Inference: What If*, Chapman & Hall/CRC, 2020, Part I。用于附录 D 在有限随机分配模型中区分一致性、条件交换性与正性；有限 ATE 公式由附录书内证明。
- Ronald L. Wasserstein, Nicole A. Lazar, [*The ASA's Statement on p-Values: Context, Process, and Purpose*](https://doi.org/10.1080/00031305.2016.1154108), *The American Statistician* 70(2), 2016。用于附录 H 的 p 值解释边界；有限随机化 p 值的 super-uniform 性由附录直接证明。
- Alon Jacovi, Yoav Goldberg, [*Towards Faithfully Interpretable NLP Systems*](https://aclanthology.org/2020.acl-main.386/), ACL 2020。用于忠实性定义和评价边界。
- Miles Turpin et al., [*Language Models Don't Always Say What They Think*](https://arxiv.org/abs/2305.04388), NeurIPS 2023。用于可见 CoT 对提示因素披露不足的特定实验；不外推为所有模型和任务。
- Tamera Lanham et al., [*Measuring Faithfulness in Chain-of-Thought Reasoning*](https://arxiv.org/abs/2307.13702), 2023。用于 CoT 干预式忠实性测量背景。
- Qing Lyu et al., [*Faithful Chain-of-Thought Reasoning*](https://aclanthology.org/2023.ijcnlp-main.20/), IJCNLP-AACL 2023。用于自然语言到显式符号链再执行的系统级接口。
- Hunter Lightman et al., [*Let's Verify Step by Step*](https://proceedings.iclr.cc/paper_files/paper/2024/hash/aca97732e30bcf1303bc22ac3924fd16-Abstract-Conference.html), ICLR 2024。用于过程监督和学习型过程评分器的经验来源；评分器不被当作形式证明内核。
- Jin Peng Zhou et al., [*Don't Trust: Verify -- Grounding LLM Quantitative Reasoning with Autoformalization*](https://openreview.net/forum?id=V5tdi14ple), ICLR 2024。用于 autoformalization 与形式验证接口；自然语言目标对齐仍是独立义务。

## 七、Agent、权限与治理

- Shunyu Yao et al., [*ReAct: Synergizing Reasoning and Acting in Language Models*](https://arxiv.org/abs/2210.03629), ICLR 2023。卷五第 1 章观察-行动循环的历史来源；正文将其实现为类型化运行时而非心理模型。
- Jerome H. Saltzer, Michael D. Schroeder, [*The Protection of Information in Computer Systems*](https://doi.org/10.1109/PROC.1975.9939), Proceedings of the IEEE 63(9), 1975。用于最小权限、完全检查和保护设计原则。
- Kai Greshake et al., [*Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection*](https://arxiv.org/abs/2302.12173), 2023。用于间接 prompt injection 与工具数据污染的经验来源。
- NIST, [*Artificial Intelligence Risk Management Framework, AI RMF 1.0*](https://doi.org/10.6028/NIST.AI.100-1), 2023。用于风险治理的组织级背景；本书的责任链不是该框架的替代实现。
- Model Context Protocol, [*Specification 2025-11-25*](https://modelcontextprotocol.io/specification/2025-11-25)。用于卷五第 1 章工具、资源和上下文协议的固定版本例子；正文原则不依赖该协议为唯一实现。

## 八、不在书内重证的范围

本书不重证一般测度扩张、正则条件分布存在性、Hoeffding 不等式的一般形式、IEEE 754 全部规范、分布式线性化理论、因果识别演算、证明助手元理论或上述经验论文的全部实验。正文若调用这些材料，只能使用本文件登记的版本和范围；未经登记的“经典结果”或“研究表明”不得进入主张链。
