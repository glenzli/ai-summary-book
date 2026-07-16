# 第三章 行为证据、对照与识别

假设两个提示模板都在询问同一道题，其中一个写“请逐步思考”，另一个只写“给出答案”。在一百道题上，前者准确率高出六个百分点。这个差值看似已经回答了提示是否有效，但它可能混合题目难度、解码随机性、输出长度、格式解析和挑选模板的过程。更重要的是，即使差值真实存在，它也没有告诉我们模型内部采用了哪种计算。

行为实验的价值不在于直接读取机制，而在于先把需要解释的现象做成可重复、可估计的对象。为此需要有限观察的边界、成对对照、随机化、结果指标和目标分布。

## M2.1 行为函数与实验样本

设 $(\mathcal X,\mathcal A_{\mathcal X})$ 为输入可测空间，$(\mathcal Y,\mathcal A_{\mathcal Y})$ 为输出可测空间。固定模型与执行环境后，理想行为可写成可测函数

$$
f:(\mathcal X,\mathcal A_{\mathcal X})
\longrightarrow(\mathcal Y,\mathcal A_{\mathcal Y}),
$$

或从 $\mathcal X$ 到 $\mathcal Y$ 的随机核 $K$。后者写作 $K(x,B)$：固定 $x\in\mathcal X$ 时，$B\mapsto K(x,B)$ 是 $(\mathcal Y,\mathcal A_{\mathcal Y})$ 上的概率测度；固定 $B\in\mathcal A_{\mathcal Y}$ 时，$x\mapsto K(x,B)$ 是 $\mathcal A_{\mathcal X}$-可测函数。符号 $K:\mathcal X\times\mathcal A_{\mathcal Y}\to[0,1]$ 只是这两个条件的简写，不能把 $\mathcal A_{\mathcal Y}$ 当作普通输出坐标而省略核条件。

实验只观察有限输入集 $S=\{x_1,\ldots,x_n\}$ 上的输出或采样。由有限观察推断全域行为需要抽样、平滑性、结构或模型类假设。

**定义 M2.1（相对于测试集的观察等价）.** 两个系统 $f,g:\mathcal X\to\mathcal Y$ 在 $S\subseteq\mathcal X$ 上观察等价，若对每个 $x\in S$，$f(x)=g(x)$。

**命题 M2.2（有限行为证据的非识别性）.** 若 $S$ 是 $\mathcal X$ 的真子集，存在 $x_0\in\mathcal X\setminus S$ 使 $\{x_0\}\in\mathcal A_{\mathcal X}$，且 $\mathcal Y$ 至少含两个元素，则存在两个不同的可测映射 $f,g:\mathcal X\to\mathcal Y$，但 $f,g$ 在 $S$ 上观察等价。

**证明.** 取上述 $x_0$，并取不同的 $y_0,y_1\in\mathcal Y$。令 $f(x)=y_0$ 对所有 $x$ 成立，再定义

$$
g(x)=
\begin{cases}
y_1,&x=x_0,\\
y_0,&x\ne x_0.
\end{cases}
$$

常值映射 $f$ 可测。对任意 $B\in\mathcal A_{\mathcal Y}$，$g^{-1}(B)$ 只能是 $\varnothing$、$\mathcal X$、$\{x_0\}$ 或 $\mathcal X\setminus\{x_0\}$，故 $g$ 也可测。对每个 $x\in S$ 都有 $g(x)=f(x)$，但 $g(x_0)\ne f(x_0)$。证毕。

若只讨论集合间任意函数，可删除单点可测条件；加入该条件是为了让本章前后都停留在可测对象范畴。该命题约束所有基准结论：有限测试只能在额外假设下支持范围外推广。

**命题 M2.3（随机核的有限非识别性）.** 设 $(\mathcal X,\mathcal A)$ 为可测空间，$S\subsetneq\mathcal X$，并存在 $x_0\in\mathcal X\setminus S$ 使 $\{x_0\}\in\mathcal A$。设 $(\mathcal Y,\mathcal B)$ 上至少存在两个不同概率测度。则存在两个从 $\mathcal X$ 到 $\mathcal Y$ 的随机核 $K_1,K_2$，使它们对每个 $x\in S$ 给出相同输出分布，但在 $\mathcal X$ 上不相同。

**证明.** 取上述 $x_0$ 以及不同概率测度 $\mu_0,\mu_1$。令 $K_1(x,\cdot)=\mu_0$ 对所有 $x$ 成立；令 $K_2(x_0,\cdot)=\mu_1$，而对 $x\ne x_0$ 令 $K_2(x,\cdot)=\mu_0$。

先检查 $K_1$。对每个固定 $x\in\mathcal X$，集合函数 $B\mapsto K_1(x,B)=\mu_0(B)$ 是 $\mathcal B$ 上的概率测度；对每个固定 $B\in\mathcal B$，函数 $x\mapsto K_1(x,B)=\mu_0(B)$ 是常值函数，故为 $\mathcal A$-可测。因此 $K_1$ 是随机核。

再检查 $K_2$。对每个固定 $x$，$K_2(x,\cdot)$ 等于 $\mu_0$ 或 $\mu_1$，所以是 $\mathcal B$ 上的概率测度。对任意固定 $B\in\mathcal B$，

$$
x\longmapsto K_2(x,B)
=\mu_0(B)+\mathbf 1_{\{x_0\}}(x)\bigl(\mu_1(B)-\mu_0(B)\bigr)
$$

是 $\mathcal A$-可测函数，因为 $\{x_0\}\in\mathcal A$。故 $K_2$ 也是随机核。二者在 $S$ 上相同，在 $x_0$ 处不同。证毕。

该命题说明随机重复调用也不能靠“样本更多”识别未测试输入上的输出核；需要模型类或迁移假设。

## M2.2 对照与最小差异

行为解释常比较成对输入 $x,x'$。若二者同时改变多个因素，输出差异无法归因于某一个因素。最小对照试图只改变目标属性，例如在保持句法结构时替换主语数。

但“只改变一个因素”依赖表示层级：替换一个 token 也会改变频率、语义、位置关联和后续 tokenization。严谨报告需列出已控制与未控制变量，而不是宣称完美最小。

## M2.3 随机化和配对

若研究提示措辞 $T\in\{0,1\}$ 对指标 $M$ 的影响，设 $X\sim D_X$，解码随机源位于可测空间 $(\mathcal U,\mathcal A_{\mathcal U})$ 并服从概率律 $\nu$。假设 $M_t:\mathcal X\times\mathcal U\to\mathbb R$ 可测且对 $D_X\otimes\nu$ 可积。相对于该乘积分布的边缘平均处理效应为

$$
\tau=\mathbb E_X\bigl[\mathbb E_U M_1(X,U)-\mathbb E_U M_0(X,U)\bigr].
$$

对同一实例运行两种提示可形成配对估计；随机化提示顺序可控制时间或缓存趋势。令 $\pi$ 是从 $\mathcal X$ 到 $\mathcal U\times\mathcal U$ 的随机核，并要求对 $D_X$-几乎处处的 $x$，概率测度 $\pi(\cdot\mid x)$ 的两个边缘都等于 $\nu$。若 $(X,U_0,U_1)$ 服从 $D_X(dx)\pi(du_0,du_1\mid x)$，则

$$
\mathbb E[M_1(X,U_1)-M_0(X,U_0)]=\tau,
$$

这里等式由条件耦合的边缘约束和可积性得到；逐样本差及其方差仍依赖 $\pi$。若两种解码器使用不同随机源分布，则“共享 seed”不保证对应随机变量具有相同语义，必须先定义各自边缘与耦合。

**命题 M2.4（配对均值的目标与方差）.** 固定上述耦合 $\pi$。设

$$
Z_i=M_1(X_i,U_{1i})-M_0(X_i,U_{0i}),\qquad i=1,\ldots,n,
$$

是从 $D_X(dx)\pi(du_0,du_1\mid x)$ 独立同分布抽取的配对差，且 $\mathbb E Z_1^2<\infty$。令 $\widehat\tau_n=n^{-1}\sum_iZ_i$。则

$$
\mathbb E\widehat\tau_n=\tau,
\qquad
\operatorname{Var}(\widehat\tau_n)=\frac{\operatorname{Var}(Z_1)}{n}.
$$

**证明.** 由上述边缘约束，$\mathbb EZ_i=\tau$。期望的线性性给出 $\mathbb E\widehat\tau_n=\tau$。又因各 $Z_i$ 独立且二阶矩有限，

$$
\operatorname{Var}(\widehat\tau_n)
=\frac1{n^2}\sum_{i=1}^n\operatorname{Var}(Z_i)
=\frac{\operatorname{Var}(Z_1)}n.
$$

证毕。

命题没有声称任意共享随机数都降低方差；$\operatorname{Var}(Z_1)$ 取决于耦合。若任务实例成簇、同一提示被重复使用或服务端状态跨调用相关，独立同分布假设失效，应在相应抽样层级聚类估计不确定性。置信区间还需要有限样本方法或渐近条件，不能只从“重复了 $n$ 次”推出覆盖率。

### 完整实验：两种提示模板的配对比较

现在把章首的问题写成协议。研究对象是固定模型、固定 tokenizer 与固定温度的解码器；总体 $D_X$ 是预先定义的数学题集合。对每道题 $X_i$，生成一对语义等价提示 $T=0,1$，并用耦合 $\pi$ 产生随机源 $(U_{0i},U_{1i})$。指标 $M_t(X_i,U_{ti})$ 取值于 $\{0,1\}$，只表示经冻结解析器判定的最终答案是否正确。

操作步骤如下：

1. 在试运行集上冻结两种模板、最大输出长度、解析器和排除规则；
2. 从目标题库按题型分层抽取确认集，每题同时运行两种模板，并随机化调用顺序；
3. 计算逐题差 $Z_i=M_1-M_0$，报告 $\widehat\tau_n$、按题型的异质性和与抽样设计匹配的区间；
4. 对格式失败另行计数，不把无法解析的输出在看到答案后选择性删除；
5. 在新的题型或难度层上重复确认，以检查 $D_{\mathrm{exp}}$ 到目标部署分布的迁移。

若观察到 $\widehat\tau_n=0.06$，直接结论是指定协议下的配对正确率差估计。这个范围化差异可以作为后续机制研究的被解释项 $\Xi$，相应问题例如“额外 token 是否承载了改变正确率的中间计算”；当前行为证据尚不能区分“更长计算”“不同指令遵循”或“解析器偏好”等机制。若先试了二十种提示再挑最大者，确认集还必须与这次选择隔离，否则六个百分点包含选择偏差。

## M2.4 指标与代理

准确率、logit 差、损失、人工偏好和任务效用是不同结果变量。一个内部干预提高正确答案 logit，不必提高最终采样正确率；提高基准分数也不必提高部署效用。

选择指标时应说明：

- 它如何从原始输出计算；
- 它对长度、格式和拒答如何处理；
- 是否在看到结果后选择；
- 与被解释项 $\Xi$ 的代理关系。

## M2.5 选择与多重比较

在大量层、头、神经元和输入中挑选最大效应会产生选择偏差。即使每个零效应估计量都无偏，最大值一般也不是零的无偏估计：若 $\widehat\tau_1,\widehat\tau_2$ 独立、同分布、可积、均值为零且非退化，则

$$
\mathbb E\max(\widehat\tau_1,\widehat\tau_2)
=\frac12\mathbb E|\widehat\tau_1-\widehat\tau_2|>0.
$$

等式来自 $\max(a,b)=(a+b+|a-b|)/2$。因此若先用同一数据发现组件，再用同一数据报告效应，报告值会受到赢家诅咒。至少应拆分发现集与确认集，或完整报告选择过程并使用适当的多重比较与重采样。

这不是要求所有探索都预注册，而是区分探索性图像与确认性证据。

## M2.6 分布范围

设 $D_{\mathrm{exp}}$ 与 $D_{\mathrm{dep}}$ 是同一可测输入空间上的概率测度，$\Delta$ 是可测效应函数。在前者上估计的平均效应

$$
\mathbb E_{x\sim D_{\mathrm{exp}}}[\Delta(x)]
$$

不等于后者上的效应，除非两分布相同或有可验证的迁移条件。

**命题 M2.5（有界效应的总变差迁移界）.** 若存在有限常数 $B\ge0$ 使 $|\Delta(x)|\le B$ 对所有 $x$ 成立，则按约定

$$
\operatorname{TV}(P,Q)=\sup_{A\in\mathcal A}|P(A)-Q(A)|,
$$

有

$$
\left|\mathbb E_{D_{\mathrm{exp}}}\Delta-
\mathbb E_{D_{\mathrm{dep}}}\Delta\right|
\le 2B\operatorname{TV}(D_{\mathrm{exp}},D_{\mathrm{dep}}).
$$

**证明.** 若 $B=0$，结论直接成立。若 $B>0$，写 $\Delta=\Delta_+-\Delta_-$，其中 $0\le\Delta_\pm\le B$。对任意可测 $g:\mathcal X\to[0,B]$，层蛋糕表示与 Tonelli 定理给出

$$
\mathbb E_Pg-\mathbb E_Qg
=\int_0^B\bigl(P(g>t)-Q(g>t)\bigr)\,dt,
$$

故其绝对值至多 $B\operatorname{TV}(P,Q)$。分别取 $g=\Delta_+$ 与 $g=\Delta_-$，再用三角不等式，即得所述 $2B\operatorname{TV}$ 上界。证毕。

这是分布迁移的一个充分上界，不是说实际总变差可轻易估计，也不声称常数在附加结构下最优。解释报告应同时给出平均值、异质性和失败子群，避免平均效果掩盖相反机制。

## M2.7 行为证据的正确位置

行为实验可以：

- 定义需要解释的稳定现象；
- 排除与行为不符的机制假说；
- 为内部干预选择输入与指标；
- 检查解释是否预测未见行为。

它通常不能单独唯一识别内部机制，因为不同程序可实现相同有限行为，甚至相同全域输入输出函数也可有不同内部实现。

在配对提示实验中，六个百分点的差异仍与三种竞争假说相容：额外 token 提供了有效中间计算，模板只提高了服从度，或者解析器偏爱某种答案格式。它们对新观察给出不同预测：限制中间 token 会削弱第一种效应，保持长度而改写指令可区分第二种，只比较解析前的正确答案 logit 则直接检查第三种。行为实验到这里没有选出内部机制，却已经把下一轮观察设计成能够区分假说的实验。

## 练习

**练习 M2.1.** 说明命题 M2.3 的单点修改在连续输入空间上为何需要可测性条件，并给出一个始终合法的离散输入版本。

**练习 M2.2.** 设计一个主谓一致最小对照，并列出至少三个仍可能变化的因素。

**练习 M2.3.** 比较共享 seed 与独立 seed 估计两个解码器差异时的目标量。

**练习 M2.4.** 给出一个 logit 差提高但最终准确率不变的例子。

**练习 M2.5.** 说明发现集与确认集如何减少“挑最大头”造成的偏差。
