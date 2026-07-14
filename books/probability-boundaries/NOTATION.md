# 符号表

## 测度与概率

| 符号 | 含义 |
|---|---|
| $(\Omega,\mathcal F)$ | 可测空间 |
| $(\Omega,\mathcal F,\mathbb P)$ | 概率空间 |
| $(S,\mathcal S,\lambda)$ | 带参考测度 $\lambda$ 的测度空间；写密度前必须指定 $\lambda$ |
| $\mathbb E[X]$ | 随机变量 $X$ 的扩展期望；若正负部积分都无限则未定义 |
| $\mathbb P(A\mid\mathcal G)$ | 条件概率 $\mathbb E[\mathbf 1_A\mid\mathcal G]$ 的版本 |
| $\mathcal L(X)$ | $X$ 的分布，即 $\mathbb P\circ X^{-1}$ |
| $X\perp Y$ | $X$ 与 $Y$ 独立 |
| $X\perp\!\!\!\perp Y\mid Z$ | $X,Y$ 给定 $\sigma(Z)$ 条件独立 |
| $\mu\otimes K$ | 概率测度 $\mu$ 与 Markov 核 $K$ 组成的联合测度 |
| $X_n\to X$ a.s. | 几乎处处收敛 |
| $X_n\to X$ in $L^p$ | $\mathbb E|X_n-X|^p\to0$ |
| $X_n\xrightarrow{\mathbb P}X$ | 依概率收敛 |
| $X_n\Rightarrow X$ | 依分布收敛 |
| $\mathcal X$ 一致可积 | $\lim_{M\to\infty}\sup_{Z\in\mathcal X}\mathbb E[|Z|\mathbf1_{\{|Z|>M\}}]=0$ |

## 信息、预测与决策

| 符号 | 含义 |
|---|---|
| $H(p)$ | 有限概率向量 $p$ 的离散 Shannon 熵 |
| $H(p,q)$ | 有限分布 $p$ 相对于报告 $q$ 的扩展实数交叉熵 |
| $D_{\mathrm{KL}}(P\Vert Q)$ | 概率测度 $P$ 相对于 $Q$ 的 KL 散度；$P\not\ll Q$ 时为 $+\infty$ |
| $S(q,y)$ | 报告分布 $q$、结果 $y$ 的损失型评分规则 |
| $L(p,q)$ | 真实分布 $p$ 下报告 $q$ 的期望评分损失 |
| $\ell(a,\theta)$ | 行动 $a$ 在状态 $\theta$ 下的损失 |
| $\mathcal R(A)$ | 决策规则 $A$ 的总体风险 $\mathbb E[\ell(A,\Theta_0)]$ |
| $K(x,A)$ | 从 $x$ 到可测集合 $A$ 的 Markov 核 |
| $k_i(x_i\mid x_{\operatorname{pa}(i)})$ | 有限因果 DAG 中第 $i$ 个完整机制核 |
| $\operatorname{do}(X=x)$ | 因果模型中的干预记号，不是普通条件事件 |

## 语言模型与算法

| 符号 | 含义 |
|---|---|
| $V$ | 有限词表 |
| $x_{1:t}$ | 长度为 $t$ 的 token 前缀 |
| $z(x)\in\mathbb R^{|V|}$ | 给定上下文 $x$ 的 logits |
| $\operatorname{softmax}(z)$ | $i\mapsto e^{z_i}/\sum_j e^{z_j}$ |
| $T$ | 温度参数；除零温极限外取 $T>0$ |
| $F_\theta$ | 参数为 $\theta$ 的确定性计算映射 |
| $R=\{0,1\}^{\mathbb N}$ | 随机带空间；正文也用随机变量 $R$ 表示取值于该空间的随机流 |
| $\rho$ | 随机带上的公平 Bernoulli 乘积测度 |
| $Y_\bot$ | 输出空间加不终止符号 $\bot$ |
| $G(s)$ | 种子为 $s$ 的伪随机数生成器输出流 |
| $q_t(\cdot\mid x_{1:m},y_{1:t-1})$ | 给定初始上下文与已生成历史的实际解码核；可等于 softmax，也可为截断重归一化分布 |

## 约定

- $\log$ 默认是自然对数。
- 离散熵中采用 $0\log0=0$，它由 $\lim_{x\downarrow0}x\log x=0$ 给出。
- 条件期望是几乎处处等价类；写成具体函数时默认已经选定一个版本。
- 扩展实数加权和中，零概率坐标的贡献按零处理；特别地不把 $0\cdot(+\infty)$ 当作普通乘法。
- “确定性”总是相对于已列出的输入和状态而言。隐藏状态未被固定时，不声称整个物理执行是数学上的单值函数。
- 所有有限集合默认带幂集 $\sigma$-代数。
