# 第三十一章：Perverse sheaves、recollement 与 t-结构

普通 cohomological t-structure 不会同时尊重分层空间的维数和 Verdier 对偶。Perverse t-structure 通过每个 strata 上的支撑与余支撑消失条件重新放置 cohomological degrees，其 heart 中的对象才是 perverse sheaves。开闭分解给出的 recollement 允许从开子空间与闭子空间粘合 t-structure，也解释 intermediate extension $j_{!*}$ 的唯一性。本章以复代数簇或合理层化空间的 constructible derived category 为模型，逐步推导这些范畴论结构。

所需语言是稳定 $\infty$-范畴、t-structure、heart、六操作和 Verdier 对偶。维数函数、stratification 与 constructibility 假设会在定理处固定；decomposition theorem 等几何深定理只作为外部输入，不从形式 recollement 误推。

## 31.1 可构造导出范畴与层化

**定义 31.1.** 设 $X$ 为带有限 Whitney stratification 的空间，层分解为

$$
X=\bigsqcup_{\alpha\in A}S_\alpha.
$$

本章固定系数域 $\Lambda$。可构造导出范畴 $D^b_c(X,\Lambda)$ 是 $D^b(X,\Lambda)$ 中那些对象 $K$，使得每个 cohomology sheaf $H^i(K)$ 在每个 stratum $S_\alpha$ 上局部常值且 stalk 为有限维 $\Lambda$-向量空间。一般 Noetherian 系数环也有相应理论，但 Verdier 对偶和有限性假设需要另行调整。

**定义 31.2.** 对 stratum 嵌入 $i_\alpha:S_\alpha\hookrightarrow X$，称

$$
i_\alpha^*K,\qquad i_\alpha^!K
$$

分别为 $K$ 沿该 stratum 的 restriction 和 corestriction。

**命题 31.3.** 对有限层化，stratum restrictions $\{i_\alpha^*\}_{\alpha\in A}$ 联合保守。

**证明.** 对 strata 数目归纳。取开 stratum $j:U\hookrightarrow X$，令闭补为 $i:Z\hookrightarrow X$。若所有 stratum restrictions 均为零，则 $j^*K\simeq0$，而归纳假设给出 $i^*K\simeq0$。Recollement 的余纤维序列

$$
j_!j^*K\to K\to i_*i^*K
$$

两端为零，故 $K\simeq0$。这些限制函子正合；对态射取余纤维后还可知它们联合反映等价。$\square$

## 31.2 标准 t-结构与 perverse t-结构

**定义 31.4.** $D^b_c(X,\Lambda)$ 的标准 t-结构由普通 sheaf cohomology 定义：

$$
D^{\le0}=\{K\mid H^i(K)=0\text{ for }i>0\},
$$

$$
D^{\ge0}=\{K\mid H^i(K)=0\text{ for }i<0\}.
$$

其 heart 是可构造 sheaves 的阿贝尔范畴。

**定义 31.5.** 设 $X$ 为复代数簇或复解析空间，取 middle perversity。定义

$$
{}^pD^{\le0}(X)=
\{K\in D^b_c(X)\mid H^i(i_\alpha^*K)=0\text{ for all }i>-\dim_\mathbb C S_\alpha\},
$$

$$
{}^pD^{\ge0}(X)=
\{K\in D^b_c(X)\mid H^i(i_\alpha^!K)=0\text{ for all }i<-\dim_\mathbb C S_\alpha\}.
$$

**外部输入定理 31.6.** 上述两类构成 $D^b_c(X)$ 上的 t-结构，称为 middle perverse t-structure。其 heart

$$
\operatorname{Perv}(X)={}^pD^{\le0}(X)\cap{}^pD^{\ge0}(X)
$$

是阿贝尔范畴。

**例子 31.7.** 若 $X$ 是光滑纯 $d$ 维复流形，则局部系统 $L[d]$ 是 perverse sheaf。这里平移 $[d]$ 反映了 perverse t-结构按复维数重新归一化。

**命题 31.8.** 若 $X$ 为一点，则 $\operatorname{Perv}(X)$ 等于有限维 $\Lambda$-模范畴。

**证明.** 唯一 stratum 的维数为 $0$，且 $i^*=i^!=\operatorname{id}$。Perverse 条件变为 $H^i(K)=0$ 对 $i>0$ 和 $i<0$，即 $K$ 只在 $0$ 次有 cohomology。因此 heart 是普通有限维 $\Lambda$-模。$\square$

## 31.3 Recollement 与 t-结构粘合

**定义 31.9.** 设 $j:U\hookrightarrow X$ 为开嵌入，$i:Z\hookrightarrow X$ 为闭补。一个 t-结构在 recollement 下由 $U$ 与 $Z$ 上 t-结构粘合，若

$$
K\in D^{\le0}(X)\iff j^*K\in D^{\le0}(U),\ i^*K\in D^{\le0}(Z),
$$

$$
K\in D^{\ge0}(X)\iff j^*K\in D^{\ge0}(U),\ i^!K\in D^{\ge0}(Z).
$$

**外部输入定理 31.10（BBD gluing）.** 在 recollement

$$
D(Z)\rightleftarrows D(X)\rightleftarrows D(U)
$$

中，任给 $D(Z)$ 和 $D(U)$ 上的 t-结构，公式 31.9 定义 $D(X)$ 上唯一的粘合 t-结构；不需另加两侧 t-结构之间的相容条件。Perverse t-结构可由 strata 上按复维数平移的标准 t-结构逐层粘合得到。

**命题 31.11.** 若 $X=U\sqcup Z$ 的 perverse t-结构由 $U,Z$ 粘合，则 $K\in\operatorname{Perv}(X)$ 当且仅当

$$
j^*K\in\operatorname{Perv}(U),\qquad i^*K\in{}^pD^{\le0}(Z),\qquad i^!K\in{}^pD^{\ge0}(Z).
$$

**证明.** $K$ perverse 等价于 $K\in{}^pD^{\le0}(X)$ 且 $K\in{}^pD^{\ge0}(X)$。由粘合定义，第一条件等价于

$$
j^*K\in{}^pD^{\le0}(U),\quad i^*K\in{}^pD^{\le0}(Z),
$$

第二条件等价于

$$
j^*K\in{}^pD^{\ge0}(U),\quad i^!K\in{}^pD^{\ge0}(Z).
$$

合并 $j^*$ 的两个条件得到 $j^*K\in\operatorname{Perv}(U)$。$\square$

## 31.4 中间延拓

**定义 31.12.** 对开嵌入 $j:U\hookrightarrow X$，perverse heart 上有左正合或右正合的延拓函子。中间延拓 $j_{!*}$ 定义为

$$
j_{!*}P=\operatorname{im}\bigl({}^pj_!P\to{}^pj_*P\bigr),
$$

其中 ${}^pj_!,{}^pj_*$ 表示取 perverse cohomology 后的相应 heart 函子。

**外部输入定理 31.13.** 若 $P\in\operatorname{Perv}(U)$，则 $j_{!*}P$ 是 $\operatorname{Perv}(X)$ 中唯一满足下列条件的对象：

1. $j^*j_{!*}P\simeq P$；
2. 没有非零 subobject 或 quotient object 支撑在 $Z=X\setminus U$ 上。

**命题 31.14.** 若 $P$ 是 simple perverse sheaf，则 $j_{!*}P$ 也是 simple，假设 $U$ 在支撑中稠密。

**证明.** 设 $0\ne Q\subseteq j_{!*}P$ 是 perverse subobject。限制到 $U$ 得 $j^*Q\subseteq P$。因 $P$ simple，$j^*Q$ 为 $0$ 或 $P$。若 $j^*Q=0$，则 $Q$ 支撑在 $Z$ 上，与中间延拓无非零闭支撑 subobject 矛盾。故 $j^*Q=P$。此时 quotient $j_{!*}P/Q$ 限制到 $U$ 为零，故支撑在 $Z$ 上；中间延拓无非零闭支撑 quotient，因此 quotient 为零，$Q=j_{!*}P$。$\square$

## 31.5 Verdier 对偶与 perverse heart

**外部输入定理 31.15.** Verdier duality $\mathbb D_X$ 交换 perverse t-结构的两半：

$$
\mathbb D_X({}^pD^{\le0})={}^pD^{\ge0},\qquad
\mathbb D_X({}^pD^{\ge0})={}^pD^{\le0}.
$$

因此 $\mathbb D_X$ 限制为反等价

$$
\operatorname{Perv}(X)^{op}\simeq\operatorname{Perv}(X).
$$

**命题 31.16.** 若 $P\in\operatorname{Perv}(U)$，则

$$
\mathbb D_X(j_{!*}P)\simeq j_{!*}(\mathbb D_U P).
$$

**证明.** Verdier 对偶交换 $j_!$ 与 $j_*$，并保持 perverse heart 到其反范畴。于是它把态射 ${}^pj_!P\to{}^pj_*P$ 对偶为

$$
{}^pj_!(\mathbb D_U P)\to{}^pj_*(\mathbb D_U P)
$$

的对应态射。反等价把 image 送为 image，因为在阿贝尔范畴中反等价交换 kernel 与 cokernel 并保持 image/coimage 同构。故得到所需等价。$\square$

**命题 31.17（闭支撑 heart）.** 在由 recollement 粘合的 t-结构下，$i_*:\operatorname{Perv}(Z)\to\operatorname{Perv}(X)$ 全忠实，且其本质像正是满足 $j^*K\simeq0$ 的 perverse sheaves。

**证明.** Recollement 给出 $j^*i_*\simeq0$，$i^*i_*\simeq\operatorname{id}$ 和 $i^!i_*\simeq\operatorname{id}$。若 $H\in\operatorname{Perv}(Z)$，则由粘合判别，

$$
j^*i_*H\simeq0,\qquad i^*i_*H\simeq H\in{}^pD^{\le0}(Z),\qquad i^!i_*H\simeq H\in{}^pD^{\ge0}(Z),
$$

故 $i_*H\in\operatorname{Perv}(X)$。全忠实性是 recollement 公理的一部分。

反过来，若 $K\in\operatorname{Perv}(X)$ 且 $j^*K\simeq0$，则 recollement 三角

$$
j_!j^*K\to K\to i_*i^*K
$$

给出 $K\simeq i_*i^*K$。应用 $i^!$ 得 $i^!K\simeq i^*K$。由 $K$ perverse 和粘合条件，$i^*K\in{}^pD^{\le0}(Z)$ 且 $i^!K\in{}^pD^{\ge0}(Z)$，于是 $i^*K\in\operatorname{Perv}(Z)$。故 $K$ 位于 $i_*$ 的本质像中。$\square$

## 31.6 Nearby cycles 与 vanishing cycles 入口

**定义 31.18.** 设 $f:X\to\mathbb C$ 为复代数或复解析映射，$i:X_0=f^{-1}(0)\hookrightarrow X$，$j:X_\eta=X\setminus X_0\hookrightarrow X$。固定方差为

$$
R\psi_f:D^b_c(X_\eta,\Lambda)\to D^b_c(X_0,\Lambda),
\qquad
R\phi_f:D^b_c(X,\Lambda)\to D^b_c(X_0,\Lambda).
$$

**外部输入定理 31.19.** 在上述复代数或复解析可构造语境中，存在标准 distinguished triangle

$$
i^*K\longrightarrow R\psi_f(j^*K)\longrightarrow R\phi_f(K)\longrightarrow i^*K[1].
$$

在 middle perversity 约定下，$R\psi_f[-1]$ 与 $R\phi_f[-1]$ 对 perverse t-结构 t-exact；monodromy 版本还带有典范自同构。

**注 31.20.** 这些函子是六操作、monodromy 和层化奇异性相互作用的入口。完整理论需要 étale 或解析 topology、monodromy action 和 vanishing cycle functor 的构造，本书当前只记录范畴论位置。

## 31.7 分层几何中的 heart

Perverse t-结构由 restriction/corestriction 的维数不等式定义，也可通过 recollement 逐层粘合。其 heart 是阿贝尔范畴，稳定地承载中间延拓、Verdier 对偶、nearby cycles 和 vanishing cycles。范畴论上，perverse sheaves 展示了 t-结构、六操作、局部-整体粘合和对偶性如何共同产生新的 abelian category。

## 练习

**练习 31.1.** 定义可构造导出范畴 $D^b_c(X,\Lambda)$。

**练习 31.2.** 证明有限层化下 stratum restrictions 联合保守。

**练习 31.3.** 写出标准 t-结构。

**练习 31.4.** 写出 middle perverse t-structure 的支撑和余支撑条件。

**练习 31.5.** 证明一点空间上的 perverse sheaves 是有限维模。

**练习 31.6.** 定义 recollement 下 t-结构的粘合。

**练习 31.7.** 用粘合条件刻画 $K\in\operatorname{Perv}(X)$。

**练习 31.8.** 定义中间延拓 $j_{!*}$。

**练习 31.9.** 证明中间延拓无非零闭支撑 subobject 或 quotient 的唯一性推出 simple 对象保持 simple。

**练习 31.10.** 陈述 Verdier 对偶对 perverse t-结构的作用。

**练习 31.11.** 证明 $\mathbb D_X(j_{!*}P)\simeq j_{!*}(\mathbb D_UP)$。

**练习 31.12.** 说明 nearby cycles 与 vanishing cycles 在 perverse sheaf 理论中的作用。

**练习 31.13.** 在 recollement 粘合的 t-结构下，证明支撑在闭补 $Z$ 上的 perverse sheaves 等价于 $\operatorname{Perv}(Z)$。
