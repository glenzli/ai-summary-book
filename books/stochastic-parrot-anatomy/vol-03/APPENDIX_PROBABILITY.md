# 附录 概率与估计的最小工具箱

本附录只提供卷三正文实际调用的离散概率与有限样本工具。它不是一般概率论或数理统计教程；测度论条件期望、随机过程、渐近统计、Bayesian 推断和因果识别应参考专门教材。

## A.1 样本空间、事件与分布

有限或可数样本空间记为 $\Omega$。概率质量函数 $p$ 满足

$$
p(\omega)\ge0,
\qquad
\sum_{\omega\in\Omega}p(\omega)=1.
$$

事件 $A\subseteq\Omega$ 的概率为

$$
P(A)=\sum_{\omega\in A}p(\omega).
$$

因此 $P(\varnothing)=0$、$P(A^c)=1-P(A)$；若 $(A_j)$ 两两互斥，则

$$
P\!\left(\bigcup_jA_j\right)=\sum_jP(A_j).
$$

语言模型下一 token 的空间通常是有限词表 $V$。完整有限 token 串构成可数集合；要使它成为总概率为 1 的回答空间，还必须规定 EOS、最大长度或其他停止机制。

概率模型总是由“样本空间＋事件＋分布”共同定义。只给一个数字而没有事件，不构成完整概率陈述。

## A.2 条件概率与链式分解

若 $P(B)>0$，定义

$$
P(A\mid B)=\frac{P(A\cap B)}{P(B)}.
$$

于是

$$
P(A\cap B)=P(A\mid B)P(B).
$$

对随机变量 $X_1,\ldots,X_n$，在所出现的条件事件具有正概率时，反复应用乘法公式得到

$$
P(x_{1:n})
=P(x_1)\prod_{t=2}^nP(x_t\mid x_{<t}).
$$

这就是自回归序列概率的基础。它是任意联合分布的恒等分解，不规定条件概率由 Transformer、RNN 还是查表实现。

条件必须一致。不能把 $P(X_1=x_1)$、$P(X_2=x_2)$ 等边缘概率相乘代替链式分解，除非额外成立相应独立性。

## A.3 全概率、Bayes 与隐藏条件

若 $(B_j)_j$ 是互斥完备划分，且相应条件有定义，则

$$
P(A)=\sum_jP(A\mid B_j)P(B_j).
$$

这解释了隐上下文混合。若 $H$ 未被观测，

$$
P(Y=y\mid X=x)
=\sum_hP(Y=y\mid X=x,H=h)P(H=h\mid X=x).
$$

Bayes 公式为

$$
P(B_j\mid A)
=\frac{P(A\mid B_j)P(B_j)}
{\sum_kP(A\mid B_k)P(B_k)}.
$$

softmax 的形式也包含“指数再归一化”，但这不使其自动成为未知参数的 Bayesian posterior。posterior 需要先验、似然和所条件的数据模型。

## A.4 独立、条件独立与交换性

$A,B$ 独立指

$$
P(A\cap B)=P(A)P(B).
$$

给定 $C$ 条件独立指

$$
P(A\cap B\mid C)
=P(A\mid C)P(B\mid C)
$$

在相应条件有定义处成立。无条件独立不蕴含给定任意 $C$ 后独立，反之亦然。

随机变量 $X_1,\ldots,X_n$ 独立同分布，简称 iid，要求联合分布分解且每个边缘相同。交换性只要求联合分布在有限置换下不变，通常弱于 iid。

重复 API 请求只有在模型版本、系统状态和解码协议固定，且随机源按设计独立时，才可近似视为 iid。共享会话、缓存、上游文档、滚动部署或 seed 会破坏该假设。

## A.5 随机变量、期望、方差与协方差

随机变量 $X:\Omega\to\mathbb R$ 的期望在 $\sum_\omega|X(\omega)|p(\omega)<\infty$ 时为

$$
\mathbb E[X]=\sum_\omega X(\omega)p(\omega),
$$

若再有有限二阶矩，方差为

$$
\operatorname{Var}(X)
=\mathbb E[(X-\mathbb E[X])^2]
=\mathbb E[X^2]-(\mathbb E[X])^2.
$$

协方差为

$$
\operatorname{Cov}(X,Y)
=\mathbb E[(X-\mathbb E[X])(Y-\mathbb E[Y])].
$$

期望线性：

$$
\mathbb E[aX+bY]
=a\mathbb E[X]+b\mathbb E[Y],
$$

不要求独立。方差满足

$$
\operatorname{Var}(X+Y)
=\operatorname{Var}(X)+\operatorname{Var}(Y)
+2\operatorname{Cov}(X,Y).
$$

独立且二阶矩有限时协方差为零；零协方差一般不推出独立。

## A.6 全期望与全方差

对条件变量 $Z$，当 $X$ 可积时，

$$
\mathbb E[X]
=\mathbb E_Z[\mathbb E[X\mid Z]].
$$

当 $X$ 二阶可积时，全方差公式为

$$
\operatorname{Var}(X)
=\mathbb E_Z[\operatorname{Var}(X\mid Z)]
+\operatorname{Var}_Z(\mathbb E[X\mid Z]).
$$

第一项是平均组内变异，第二项是组间均值变异。卷三用它区分固定 checkpoint 内的采样变异与 checkpoint 之间的训练变异。

分解恒成立不表示两个项已有特定认识论含义。$Z$ 代表何种模型集合，必须由实验设计说明。

## A.7 Monte Carlo、标准误差与中心极限定理

若 $X_1,\ldots,X_n$ iid，均值为 $\mu$、方差满足 $0<\sigma^2<\infty$，样本均值

$$
\overline X_n=\frac1n\sum_iX_i
$$

满足

$$
\mathbb E[\overline X_n]=\mu,
\qquad
\operatorname{Var}(\overline X_n)=\frac{\sigma^2}{n}.
$$

所以标准误差按 $1/\sqrt n$ 缩小。弱大数律说明 $\overline X_n$ 依概率趋于 $\mu$；中心极限定理在相应条件下给出

$$
\sqrt n\frac{\overline X_n-\mu}{\sigma}
\xrightarrow{d}\mathcal N(0,1).
$$

若 $\sigma^2=0$，则 $X_i=\mu$ 几乎处处成立，样本均值没有抽样变异。实践中当 $n\ge2$ 时以样本标准差 $s$ 估计 $\sigma$，写

$$
\widehat{SE}(\overline X_n)=\frac{s}{\sqrt n}.
$$

大样本正态区间依赖近似质量。重尾、小样本、强偏斜和数据依赖都可能使它失效。

Chebyshev 不等式只需有限方差：

$$
P(|\overline X_n-\mu|\ge\varepsilon)
\le\frac{\sigma^2}{n\varepsilon^2}.
$$

它通常较宽，但清楚展示 $n^{-1}$ 的概率上界。

## A.8 Bernoulli 比例与 Wilson 区间

对事件指示 $X_i=\mathbf 1\{Y_i\in A\}$，若 $X_i\overset{\mathrm{iid}}{\sim}\operatorname{Bernoulli}(p)$，则

$$
\widehat p=\frac1n\sum_iX_i,
\qquad
\operatorname{Var}(\widehat p)=\frac{p(1-p)}{n}.
$$

对双侧名义覆盖率 $1-\alpha$，令 $z=z_{1-\alpha/2}$ 为标准正态分位数。Wald 区间 $\widehat p\pm z\sqrt{\widehat p(1-\widehat p)/n}$ 在小样本或极端比例下表现较差。Wilson score 区间的中心为

$$
c=
\frac{\widehat p+z^2/(2n)}{1+z^2/n},
$$

半宽为

$$
h=
\frac{z}{1+z^2/n}
\sqrt{
\frac{\widehat p(1-\widehat p)}{n}
+\frac{z^2}{4n^2}
}.
$$

区间为 $[c-h,c+h]$。还可采用精确二项区间，但“精确”指覆盖控制，不表示区间最短。

若观察 $n$ 次而零失败，则 $P(0\text{ failures}\mid p)=(1-p)^n$。令其等于 $\alpha$，得到失败率的单侧 $1-\alpha$ 上界

$$
p_U=1-\alpha^{1/n}.
$$

零观察只给上界，不证明事件不可能。

## A.9 置信区间的频率解释

一个 $1-\alpha$ 置信区间程序的典型含义是：在假设和重复抽样机制成立时，重复构造的区间中至少或近似有 $1-\alpha$ 的比例覆盖固定参数。

它不意味着在频率学派模型下，“参数以 $1-\alpha$ 的后验概率位于已经算出的这个区间”。后者需要 Bayesian 模型。

覆盖保证还取决于：

- 数据是否来自声明的抽样机制；
- 区间方法的有限样本或渐近条件；
- 是否在看过数据后选择了 estimand 或方法；
- 是否进行了多重比较；
- 聚类与时间依赖是否被处理。

区间窄只表示在模型假设下抽样误差小，不覆盖标签错误、分布错设和不可识别偏差。

## A.10 相关均值与有效样本量

对不必独立的 $X_1,\ldots,X_n$，

$$
\operatorname{Var}(\overline X_n)
=\frac1{n^2}\sum_{i,j}
\operatorname{Cov}(X_i,X_j).
$$

若序列二阶平稳、$\operatorname{Var}(X_t)=\sigma^2$，且滞后 $k$ 的自相关为 $\rho_k$，则有限样本恒等式为

$$
\operatorname{Var}(\overline X_n)
=\frac{\sigma^2}{n}
\left[
1+2\sum_{k=1}^{n-1}
\left(1-\frac{k}{n}\right)\rho_k
\right].
$$

若再有 $\sum_{k\ge1}|\rho_k|<\infty$，则大样本下括号趋于积分自相关时间

$$
\tau_{\mathrm{int}}=1+2\sum_{k\ge1}\rho_k.
$$

当 $\tau_{\mathrm{int}}>0$ 时，相应渐近有效样本量写为

$$
n_{\mathrm{eff}}
=\frac{n}{\tau_{\mathrm{int}}}.
$$

负相关可使 $n_{\mathrm{eff}}>n$；因此它是方差等效量，不是实际独立观测数。只有当平稳性、相关和式及其截断估计可辩护时，该写法才可靠。层级数据通常更适合按真实聚类单位重采样，而不是估一个通用 $n_{\mathrm{eff}}$。

## A.11 配对差与 bootstrap

同一实验单位上观察 $X_i,Y_i$，目标差为

$$
\delta=\mathbb E[X-Y].
$$

配对估计量

$$
\widehat\delta
=\frac1n\sum_i(X_i-Y_i)
$$

利用了协方差，因为

$$
\operatorname{Var}(X-Y)
=\operatorname{Var}(X)+\operatorname{Var}(Y)
-2\operatorname{Cov}(X,Y).
$$

非参数 paired bootstrap 以单位 $i$ 为整体有放回重采样，每次同时保留 $(X_i,Y_i)$。若数据还有文档、checkpoint 或运行层级，重采样方案必须保留该结构。

bootstrap 近似观测经验分布下统计量的抽样分布，其一致性与区间覆盖仍依赖统计量和抽样机制的正则条件；非光滑极值、稀有事件、很少的聚类或强依赖需要专门方法。它不能从一个 checkpoint 推断训练运行方差，也不能修复样本不代表目标总体的问题。

## A.12 熵、交叉熵与 KL

对有限分布 $p,q$，定义

$$
H(p)=-\sum_xp(x)\log p(x),
$$

$$
H(p,q)=-\sum_xp(x)\log q(x),
$$

$$
D_{\mathrm{KL}}(p\Vert q)
=\sum_xp(x)\log\frac{p(x)}{q(x)}.
$$

约定 $0\log0=0$；若某个 $p(x)>0$ 而 $q(x)=0$，则交叉熵与 KL 为 $+\infty$。直接展开得

$$
H(p,q)=H(p)+D_{\mathrm{KL}}(p\Vert q).
$$

若 $q(x)=0$ 对某个 $p(x)>0$ 成立，非负性由 $D_{\mathrm{KL}}(p\Vert q)=+\infty$ 立即得到。否则令 $S_p=\{x:p(x)>0\}$，在 $S_p$ 上应用 $\log u\le u-1$：

$$
-D_{\mathrm{KL}}(p\Vert q)
=\sum_{x\in S_p}p(x)\log\frac{q(x)}{p(x)}
\le\sum_{x\in S_p}\bigl(q(x)-p(x)\bigr)
=q(S_p)-1\le0.
$$

故 $D_{\mathrm{KL}}(p\Vert q)\ge0$。若等号成立，则两步不等式都取等：$q(x)=p(x)$ 对所有 $x\in S_p$ 成立，且 $q(S_p)=1$，所以 $p=q$。反过来，若 $p=q$，则每个 $p(x)>0$ 的求和项都是 $p(x)\log 1=0$，而 $p(x)=0$ 的项按约定为 0，故 $D_{\mathrm{KL}}(p\Vert q)=0$。KL 不对称，也不是距离：一般

$$
D_{\mathrm{KL}}(p\Vert q)
\ne D_{\mathrm{KL}}(q\Vert p).
$$

## A.13 推前分布与语义事件

若随机变量 $Y$ 取值于 $\mathcal Y$，映射 $g:\mathcal Y\to\mathcal C$，则 $C=g(Y)$ 的推前分布为

$$
P_g(C=c)
=P(g(Y)=c)
=\sum_{y:g(y)=c}P(Y=y).
$$

这正是把多个 token 序列聚合为显示字符串或语义类别的数学形式。分布由原概率与映射共同决定；改变 $g$ 会改变事件概率。

若 $g$ 由有误差的分类器近似，观察频率估计的是分类器输出事件，而不是真实语义事件。样本量增大只减少抽样误差，不消除系统测量误差。

## A.14 Importance weighting

设源分布为 $P$、目标分布为 $Q$，且 $Q$ 对 $P$ 绝对连续。密度比

$$
w=\frac{dQ}{dP}
$$

满足对可积函数 $h$：

$$
\mathbb E_Q[h]
=\mathbb E_P[wh].
$$

离散情形只需写 $w(x)=Q(x)/P(x)$，并要求 $Q(x)>0$ 时 $P(x)>0$。若目标在源分布为零的区域有质量，权重不存在，源数据无法识别该区域目标期望。

有限样本估计

$$
\widehat\mu_Q
=\frac1n\sum_iw(X_i)h(X_i)
$$

可因极端权重具有很大方差。剪裁和自归一化可以稳定估计，却通常引入偏差或改变精确无偏性。

## A.15 相关、预测与因果

条件概率 $P(Y\mid X)$ 是观察分布。干预 $P(Y\mid do(X=x))$ 需要额外因果模型、随机实验或可辩护识别假设。仅凭联合分布通常不能唯一识别因果方向。

prompt 与输出一起变化可证明模型对输入敏感；明确修改内部变量并控制替代路径，可以支持计算图内的干预结论；二者都不自动成为现实世界同名变量的因果效应。

## A.16 使用范围

本附录足以支持卷三的条件分布、序列链式概率、经验频率、方差分解、Monte Carlo 区间、配对比较、熵、KL、推前事件和 importance weighting。

遇到连续高维条件分布、严格正则条件概率、鞅与自适应实验、Bayesian posterior、半参数效率或因果识别时，应转向专门教材，并在卷三结论之外明确新增假设。
