# 资料源与外部输入

本书优先引用一手论文、正式标准和研究生教材。正文中的外部输入必须在下表中能定位到定理主题与用途；书目只承担被明确列出的部分，不为正文中更强的说法背书。

## 概率、测度与收敛

- Olav Kallenberg, [*Foundations of Modern Probability*, 3rd ed.](https://link.springer.com/book/10.1007/978-3-030-61871-1), Springer, 2021：`Measure Extension and Decomposition`（pp. 33--54）、`Kernels, Disintegration, and Invariance`（pp. 55--77）、`Processes, Distributions, and Independence`（pp. 81--99，尤其引理 4.22 `kernels and randomization`）、`Random Sequences, Series, and Averages`（pp. 101--123）、`Gaussian and Poisson Convergence`（pp. 125--146）及 `Conditioning and Disintegration`（pp. 163--183，尤其定理 8.5 `conditional distributions, disintegration` 与定理 8.24 `extension by conditioning, Ionescu Tulcea`）。用于 Radon--Nikodym、乘积与核、随机化表示、独立性、极限定理、正则条件分布和路径测度扩张。
- Patrick Billingsley, *Probability and Measure*, 3rd ed., Wiley, 1995：测度与积分、独立性、条件期望及随机变量收敛各章。用于本书第一至第五章的标准测度论口径。
- Patrick Billingsley, [*Convergence of Probability Measures*, 2nd ed.](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316962), Wiley, 1999, Chapter 1 `Weak Convergence in Metric Spaces`。用于 Portmanteau 理论及依分布收敛。
- Rick Durrett, [*Probability: Theory and Examples*, 5th ed.](https://services.math.duke.edu/~rtd/PTE/PTE5_011119.pdf), 2019：Chapter 1 `Measure Theory`、Chapter 2 `Laws of Large Numbers`、Chapter 3 `Central Limit Theorems`。用于条件期望、Kolmogorov 强大数律与 Lindeberg--Levy 中心极限定理的版本核对。
- Wassily Hoeffding, [*Probability Inequalities for Sums of Bounded Random Variables*](https://doi.org/10.1080/01621459.1963.10500830), *Journal of the American Statistical Association* 58(301), 1963, 13--30。用于外部输入 5.14 的有界独立变量尾界。
- A. N. Kolmogorov, *Foundations of the Theory of Probability*, 2nd English ed., Chelsea, 1956。用于概率公理体系的历史定位，不承担本书现代测度论细节的唯一来源。

## 信息、评分与决策

- Thomas M. Cover and Joy A. Thomas, *Elements of Information Theory*, 2nd ed., Wiley, 2006, Chapter 2 `Entropy, Relative Entropy, and Mutual Information`。用于有限或可数离散情形的熵、相对熵与数据处理不等式。
- Imre Csiszár, *Information-Type Measures of Difference of Probability Distributions and Indirect Observations*, *Studia Scientiarum Mathematicarum Hungarica* 2 (1967), 299--318。用于一般概率测度的 $f$-散度及随机观测下的数据处理原则；本书的 KL 核数据处理是其特例。
- Tilmann Gneiting and Adrian E. Raftery, [*Strictly Proper Scoring Rules, Prediction, and Estimation*](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf), *JASA* 102 (2007), 359--378。用于适当与严格适当评分规则的量词和风险方向。
- Glenn W. Brier, *Verification of Forecasts Expressed in Terms of Probability*, *Monthly Weather Review* 78 (1950), 1--3。用于 Brier 分数的原始定义。
- A. Philip Dawid, *The Well-Calibrated Bayesian*, *JASA* 77 (1982), 605--610。用于概率校准的统计解释边界。

## 因果概率

- Judea Pearl, [*A Causal Calculus for Statistical Research*](https://proceedings.mlr.press/r0/pearl95a.html), PMLR R0, 1995, 430--449。用于普通条件与干预条件的区分以及 do-calculus 的一手定位。
- Judea Pearl, [*Causality*, 2nd ed.](https://bayes.cs.ucla.edu/BOOK-99/book-toc.html), Cambridge University Press, 2009：§§1.3--1.4（因果 Bayesian 网络与函数因果模型）及 §§3.2--3.4（Markovian 模型中的干预、截断分解与干预计算）。用于第九章的结构语义和外部识别边界。
- Miguel A. Hernán and James M. Robins, *Causal Inference: What If*, Chapman & Hall/CRC, 2020, Part I 的随机试验章节。用于一致性、交换性与正性在随机试验识别中的职责划分。
- Peter Spirtes, Clark Glymour, and Richard Scheines, *Causation, Prediction, and Search*, 2nd ed., MIT Press, 2000。用于图模型与因果发现范围边界；本书不调用其一般发现算法作为书内结论。

## 算法、实现与语言模型

- IEEE, [*IEEE Standard for Floating-Point Arithmetic 754-2019*](https://standards.ieee.org/ieee/754/6210/)。用于浮点格式与舍入语义；该标准不固定整个并行软件栈。
- PyTorch, [*Reproducibility Notes*](https://docs.pytorch.org/docs/stable/notes/randomness.html)。用于软件实现中随机源、内核和跨平台复现的工程边界，不作为数学定理来源。
- Donald E. Knuth, *The Art of Computer Programming, Volume 2: Seminumerical Algorithms*, 3rd ed., Addison-Wesley, 1997。用于伪随机生成器的经典实现接口。
- Claude E. Shannon, *A Mathematical Theory of Communication*, *Bell System Technical Journal* 27 (1948), 379--423, 623--656。用于序列概率与信息量的历史基础。
- Ilya Sutskever, Oriol Vinyals, and Quoc V. Le, [*Sequence to Sequence Learning with Neural Networks*](https://arxiv.org/abs/1409.3215), NeurIPS 2014。用于条件序列模型的标准分解。
- Ashish Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), NeurIPS 2017。用于 Transformer 前向计算的架构定位；第十章的概率结论不依赖注意力机制细节。
- Ari Holtzman et al., [*The Curious Case of Neural Text Degeneration*](https://arxiv.org/abs/1904.09751), ICLR 2020。用于 nucleus sampling 的原始算法定位。

## 外部输入与正文用途

| 外部输入 | 本书使用的精确版本 | 正文位置 | 主要来源 |
|---|---|---|---|
| Lebesgue 积分、单调/支配收敛与 Fatou 引理 | 非负扩展积分；可积函数的正负部分；共同 $L^p$ 支配下的收敛 | 2.6、5.10 | Kallenberg pp. 9--54；Billingsley *Probability and Measure* |
| Radon--Nikodym 定理 | $\sigma$-有限正测度与绝对连续、有限全变差的符号测度 | 2.5、4.1--4.3、6.3 | Kallenberg `Measure Extension and Decomposition` |
| 有限测度的 $\pi$-$\lambda$ 唯一性 | 包含全空间并生成目标 $\sigma$-代数的 $\pi$-系统；两个有限测度在其上相同 | 3.5--3.6 | Kallenberg `Sets and Functions, Measures and Integration`；Billingsley *Probability and Measure* |
| 乘积测度、Fubini--Tonelli 与核积 | 有限/可数概率乘积；非负或可积函数的迭代积分；$\mu\otimes K$ | 3.5、3.10、3.12 | Kallenberg `Kernels, Disintegration, and Invariance` |
| Ionescu--Tulcea 定理 | 可数可测空间与依历史 Markov 核的唯一路径测度 | 3.13、10.4 | Kallenberg 第三版定理 8.24 |
| 正则条件分布 | 被条件变量的值域为标准 Borel；条件变量值域可为任意可测空间；核在 $\mathcal L(Y)$-零集外作为测度同时唯一 | 4.10 | Kallenberg 第三版定理 8.5 |
| 标准 Borel 随机化引理 | 任意输入可测空间、标准 Borel 输出空间、单一 $U\sim\mathrm{Unif}[0,1]$ | 8.7 | Kallenberg 第三版引理 4.22 |
| Portmanteau 与 Vitali 收敛 | 依概率蕴含依分布；依概率加一致可积等价于 $L^1$ 收敛方向 | 5.7、5.10 | Billingsley；Kallenberg |
| 强大数律与中心极限定理 | i.i.d. 可积 SLLN；i.i.d. 非退化有限方差 Lindeberg--Levy CLT | 5.12、5.13 | Durrett Chapters 2--3 |
| Hoeffding 不等式 | 相互独立且各自几乎处处落在 $[a_i,b_i]$ 的有限和 | 5.14 | Hoeffding 1963 |
| 相对熵数据处理 | 同一 Markov 核对两个概率测度的共同推前，采用扩展 KL 定义 | 6.9 | Csiszár 1967；Cover--Thomas Chapter 2（离散特例） |
| 一般因果识别与 do-calculus | 结构模型给定、图假设给定时的干预演算；不把观察相关自动升级为干预效应 | 9.4--9.7 | Pearl 1995；Pearl *Causality* §§3.2--3.4 |

正文中的有限 Jensen、弱大数律、Gibbs 不等式、对数损失严格适当性、Brier 正交分解、有限 Bayes 行动、有限逆变换采样、有限机制核截断分解和有限长度自回归联合分布均在书内证明，不由本表代替证明。
