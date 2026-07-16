# 资料源与外部输入

本表只登记正文实际调用而未在书内重证的结果。每项同时给出本书采用的版本、用途、来源定位和未重证边界；历史上更强或更一般的版本不自动进入本书证明链。

## 固定版本书目

- Gerald B. Folland, [*Real Analysis: Modern Techniques and Their Applications*](https://www.wiley-vch.de/en/areas-interest/mathematics-statistics/real-analysis-978-0-471-31716-6), 2nd ed., Wiley, 1999。定位：Chapter 1 “Measures”；Chapter 2 “Integration”；Chapter 3 “Signed Measures and Differentiation”。
- Olav Kallenberg, [*Foundations of Modern Probability*](https://link.springer.com/book/10.1007/978-3-030-61871-1), 3rd ed., Springer, 2021。定位：Chapter 2 “Measure Extension and Decomposition”，Chapter 3 “Kernels, Disintegration, and Invariance”，Chapter 4 “Processes, Distributions, and Independence”，Chapter 8 “Conditioning and Disintegration”。
- Rick Durrett, [*Probability: Theory and Examples*](https://www.cambridge.org/core/books/probability/DD9A1907F810BB14CCFF022CDFC5677A), 5th ed., Cambridge University Press, 2019。定位：Chapter 1 “Measure Theory”，Chapter 2 “Laws of Large Numbers”，尤其 §2.4；Chapter 3 “Central Limit Theorems”。
- Thomas M. Cover and Joy A. Thomas, [*Elements of Information Theory*](https://onlinelibrary.wiley.com/doi/book/10.1002/047174882X), 2nd ed., Wiley, 2006。定位：Chapter 2 “Entropy, Relative Entropy, and Mutual Information”，Chapter 3 “Asymptotic Equipartition Property”，Chapter 4 “Entropy Rates of a Stochastic Process”，Chapter 5 “Data Compression”，Chapter 7 “Channel Capacity”。
- Robert M. Gray, [*Entropy and Information Theory*](https://link.springer.com/book/10.1007/978-1-4419-7970-4), 2nd ed., Springer, 2011。定位：Chapter 3 “Entropy”，Chapter 4 “The Entropy Ergodic Theorem”，Chapter 12 “Ergodic Theorems for Densities”，Chapters 13 and 15 “Source Coding Theorems” 与 “Coding for Noisy Channels”。
- George D. Birkhoff, [“Proof of the Ergodic Theorem”](https://pmc.ncbi.nlm.nih.gov/articles/PMC1076138/), *Proceedings of the National Academy of Sciences* 17 (1931), 656--660。作为 EI-9 的一手历史来源；正文使用的条件期望表述以现代概率论版本为准。
- Claude E. Shannon, [“A Mathematical Theory of Communication”](https://onlinelibrary.wiley.com/doi/abs/10.1002/j.1538-7305.1948.tb00917.x), *Bell System Technical Journal* 27 (1948), 379--423, 623--656。作为 EI-11、EI-12 的一手历史来源；正文精确有限字母表版本以 Cover--Thomas Chapters 5 and 7 为准。

## EI-1：Caratheodory 扩张

- **本书版本。** 设 $\mathcal A$ 是 $\Omega$ 上的代数，$\mu_0:\mathcal A\to[0,\infty]$ 是预测度，即 $\mu_0(\varnothing)=0$，且只要不交的 $(A_n)\subseteq\mathcal A$ 满足 $\bigcup_nA_n\in\mathcal A$，就有 $\mu_0(\bigcup_nA_n)=\sum_n\mu_0(A_n)$。若 $\mu_0$ 为 $\sigma$-有限，则存在唯一测度 $\mu$ 在 $\sigma(\mathcal A)$ 上延拓 $\mu_0$。
- **用途。** 第 1 章与附录 A 从代数上的长度或柱集合数据进入生成 $\sigma$-代数上的测度。
- **定位。** Folland 2nd ed. (1999), Chapter 1, outer-measure construction and the measure-extension theorem；Kallenberg 3rd ed. (2021), Chapter 2。
- **未重证边界。** 本书不重建外测度构造。过程有限维分布的拼接不由 EI-1 单独承担，而由 EI-5 承担。

## EI-2：Lebesgue 收敛定理

- **EI-2a，单调收敛。** 在任意测度空间上，若 $0\le f_n\uparrow f$ 几乎处处，则 $\int f_n\,d\mu\uparrow\int f\,d\mu$。
- **EI-2b，Fatou 引理。** 若 $f_n\ge0$ 可测，则 $\int\liminf_nf_n\,d\mu\le\liminf_n\int f_n\,d\mu$。
- **EI-2c，控制收敛。** 若 $f_n\to f$ 几乎处处且 $|f_n|\le g\in L^1(\mu)$，则 $f\in L^1$、$\int|f_n-f|\,d\mu\to0$，从而积分收敛。
- **用途。** 第 2、3、6 章的简单函数逼近、期望连续性与几乎处处收敛接口。
- **定位。** Folland 2nd ed. (1999), Chapter 2, sections on positive functions and convergence of integrals；Durrett 5th ed. (2019), Chapter 1, §§1.4--1.6。
- **未重证边界。** 三项定理不互相混用；无单调性、非负性或可积控制时，本书不交换极限与积分。

## EI-3：Radon--Nikodym

- **本书版本。** 设 $\mu$ 为 $\sigma$-有限正测度，$\nu$ 为对 $\mu$ 绝对连续的 $\sigma$-有限正测度。则存在可测 $f\ge0$，使 $\nu(A)=\int_Af\,d\mu$ 对所有可测 $A$ 成立；$f$ 在 $\mu$-几乎处处意义下唯一。有限有符号测度版本由正负变差分别应用该结论得到。
- **用途。** 第 5 章把 $A\mapsto\int_AX^\pm\,d\mathbb P$ 表示为 $\mathcal G$ 上密度，从而证明条件期望存在。
- **定位。** Folland 2nd ed. (1999), Chapter 3, Radon--Nikodym theorem；Kallenberg 3rd ed. (2021), Chapter 2。
- **未重证边界。** 本书证明条件期望的唯一性与演算，但不重证密度存在定理。

## EI-4：Tonelli 与 Fubini

- **本书版本。** 设 $(S,\mathcal S,\mu)$、$(T,\mathcal T,\nu)$ 为 $\sigma$-有限测度空间。若 $f:S\times T\to[0,\infty]$ 可测，则乘积积分等于两种次序的迭代积分，允许为 $+\infty$；若 $f$ 可积，则两种迭代积分几乎处处有定义并都等于乘积积分，且可交换次序。
- **用途。** 第 4、7 章的核复合和 Chapman--Kolmogorov 结合性，以及独立乘积积分。
- **定位。** Folland 2nd ed. (1999), Chapter 2, section “Product Measures”；Durrett 5th ed. (2019), Chapter 1, §1.7。
- **未重证边界。** 对一般符号函数，绝对可积性不可省略。

## EI-5：Kolmogorov 扩张

- **本书版本。** 设 $T$ 为任意索引集，$E$ 为标准 Borel 空间。若每个非空有限 $J\subseteq T$ 上给定概率测度 $\mu_J$，且对 $J\subseteq K$ 的坐标投影满足 $(\pi_{K,J})_\#\mu_K=\mu_J$，则在 $(E^T,\mathcal E^{\otimes T})$ 上存在唯一概率测度具有这些有限维边缘。
- **用途。** 第 7 章与附录 A 从一致有限维分布得到路径空间过程。
- **定位。** Kallenberg 3rd ed. (2021), Chapter 4 “Processes, Distributions, and Independence”，process existence theorem。
- **未重证边界。** 结论只在乘积 $\sigma$-代数上构造过程，不给出连续、右连续或其他路径正则性。

## EI-6：标准 Borel 空间上的正则条件分布

- **本书版本。** 若 $S,T$ 为标准 Borel 空间，$X:\Omega\to S$、$Y:\Omega\to T$ 可测，则存在 Markov 核 $K:T\times\mathcal B(S)\to[0,1]$，使
  $$
  \mathbb P(X\in A,Y\in B)
  =\int_BK(y,A)\,\mathcal L(Y)(dy)
  $$
  对所有 Borel $A,B$ 成立；核对 $\mathcal L(Y)$-几乎每个 $y$ 唯一。
- **用途。** 第 5 章和附录 A 把抽象条件期望表示为随 $Y=y$ 变化的条件分布核。
- **定位。** Kallenberg 3rd ed. (2021), Chapter 3 “Kernels, Disintegration, and Invariance” 与 Chapter 8 “Conditioning and Disintegration”。
- **未重证边界。** 非标准 Borel 空间上的存在性不作断言；零概率条件点的版本不逐点唯一。

## EI-7：独立同分布强大数律

- **本书版本。** 若 $X_1,X_2,\ldots$ 为独立同分布实随机变量且 $\mathbb E|X_1|<\infty$，则
  $$
  \frac1n\sum_{k=1}^nX_k\to\mathbb E X_1
  $$
  几乎处处。
- **用途。** 第 6 章区分频率的路径收敛与书内 $L^2$ 弱大数律。
- **定位。** Durrett 5th ed. (2019), Chapter 2, §2.4 “Strong Law of Large Numbers”；Kallenberg 3rd ed. (2021), Chapter 5 “Random Sequences, Series, and Averages”。
- **未重证边界。** 本书不重写截断、独立和式控制与第二 Borel--Cantelli 型论证。

## EI-8：Lindeberg--Levy 中心极限定理

- **本书版本。** 若 $X_i$ 独立同分布，$\mathbb E X_1=\mu$，$0<\operatorname{Var}(X_1)=\sigma^2<\infty$，则
  $$
  \frac{\sum_{k=1}^nX_k-n\mu}{\sigma\sqrt n}
  \Rightarrow N(0,1).
  $$
- **用途。** 第 6 章描述样本均值在 $n^{-1/2}$ 尺度上的分布波动。
- **定位。** Durrett 5th ed. (2019), Chapter 3 “Central Limit Theorems”；Kallenberg 3rd ed. (2021), Chapter 6 “Gaussian and Poisson Convergence”。
- **未重证边界。** 本书不引入特征函数唯一性或 Lindeberg 替换法；非同分布三角阵列版本不在范围内。

## EI-9：Birkhoff 点态遍历定理

- **本书版本。** 对概率保测系统 $(\Omega,\mathcal F,\mathbb P,T)$、模零不变 $\sigma$-代数 $\mathcal I$ 及任意 $f\in L^1$，
  $$
  \frac1n\sum_{k=0}^{n-1}f\circ T^k
  \to\mathbb E[f\mid\mathcal I]
  $$
  几乎处处；若系统遍历，极限为 $\mathbb E f$。
- **用途。** 第 10 章把平稳遍历过程的一条路径上的字母频率连接到边缘概率。
- **定位。** Birkhoff (1931), pp. 656--660；Durrett 5th ed. (2019), Chapter 6 “Ergodic Theorems”；Gray, *Probability, Random Processes, and Ergodic Properties*, 2nd ed., Springer, 2009, Chapter 9 “Ergodic Theorems”。
- **未重证边界。** 本书不证明最大遍历不等式。EI-9 不直接推出变化中的块信息密度结论。

## EI-10：Shannon--McMillan--Breiman

- **本书版本。** 若 $(X_n)$ 是有限字母表上的平稳遍历过程，$p_n(x^n)=\mathbb P(X_1^n=x^n)$，$h$ 为熵率，则
  $$
  -\frac1n\log p_n(X_1^n)\to h
  $$
  几乎处处。
- **用途。** 第 10 章推出 AEP、典型集概率趋一及典型集指数规模界。
- **定位。** Gray 2nd ed. (2011), Chapter 4 “The Entropy Ergodic Theorem” and Chapter 12 “Ergodic Theorems for Densities”；Cover--Thomas 2nd ed. (2006), Chapters 3 and 4。
- **未重证边界。** 本书不重证条件信息函数的鞅/遍历收敛论证，也不把 EI-10 与 EI-9 合并。

## EI-11：DMS 固定长度信源编码 direct 部分

- **本书版本。** 对有限字母表独立同分布信源 $P$，存在确定性固定长度块码使块错误概率趋于零且 $\limsup_n n^{-1}\log M_n\le H(P)$；等价地，每个 $R>H(P)$ 都可由最终码率不超过 $R$ 的码可靠达到。
- **用途。** 与书内定理 9.1 的计数 converse 合并，识别 DMS 的最优固定长度压缩率。
- **定位。** Cover--Thomas 2nd ed. (2006), Chapter 3 “Asymptotic Equipartition Property” and Chapter 5 “Data Compression”；Shannon (1948), noiseless coding portions。
- **未重证边界。** 第 9 章给出典型集、基数界、单射编码和错误界组成的可审查路线，但保持 achievability 的状态为外部输入；通用编码和可数/一般字母表版本不在范围内。

## EI-12：DMC 信道编码 direct 部分

- **本书版本。** 对有限输入、输出字母表 DMC $W$ 及每个 $0\le R<C(W)=\max_pI_p(X;Y)$，存在确定性块码序列满足 $\liminf_n n^{-1}\log M_n\ge R$ 且最大错误概率趋于零。
- **用途。** 与书内定理 9.7 的 Fano 弱 converse 合并，得到平均错误与最大错误操作容量都等于单字母容量。
- **定位。** Cover--Thomas 2nd ed. (2006), Chapter 7 “Channel Capacity”，尤其 channel coding theorem and expurgation argument；Shannon (1948), noisy-channel coding portions。
- **未重证边界。** 第 9 章只记录随机码、信息密度阈值、并合界、确定性选择与 expurgation 的可审查路线。强 converse、错误指数和带代价约束信道不在本书范围。
