# 第四章 梯度、积分梯度与路径归因

一段文本被分成若干 embedding 后，研究者对正确答案 logit 反向传播，得到每个坐标的梯度。颜色最深的位置常被称为“模型依赖的词”。然而梯度首先是连续函数在一个点附近的导数：它可能在饱和点为零，也可能沿一个不对应任何合法 token 的方向很大。要从导数走到有限差异，需要选择端点和路径；要从有限差异走到因果解释，还要说明这些扰动是否代表可实施的干预。

数学上，最小对象是一个可微标量函数。多元微积分给出局部语言，绝对连续路径上的微积分基本定理则说明何时可以把局部导数累积成端点差。

## M3.1 局部梯度

设 $U\subseteq\mathbb R^d$ 为开集，$F:U\to\mathbb R$ 是被解释标量，例如类别 logit、logit 差或损失。开集 $U$ 是允许连续扰动的数学定义域，不等于自然数据流形。

**定义 M3.1（梯度显著性）.** 若 $x\in U$ 且 $F$ 在 $x$ Fréchet 可微，局部梯度归因为

$$
A_i^{\nabla}(x)=\frac{\partial F}{\partial x_i}(x).
$$

它描述留在 $U$ 内的无穷小坐标扰动的一阶敏感度：

$$
F(x+h)=F(x)+\nabla F(x)^\top h+o(\|h\|).
$$

输入 token 是离散对象。对 embedding 求梯度解释的是连续嵌入空间中的局部扰动，不自动对应合法 token 替换。

**命题 M3.2（局部零梯度不蕴含有限无贡献）.** 存在光滑函数 $F$ 与点 $x,x'$，使 $\nabla F(x)=0$，但 $F(x)-F(x')\ne0$。

**证明.** 取 $d=1$、$U=\mathbb R$、$F(t)=t^3$、$x=0$、$x'=1$。$F'(0)=0$，但 $F(0)-F(1)=-1$。证毕。

这就是饱和或局部平坦造成的边界：梯度回答局部导数问题，不是从基线到输入的总差异。

## M3.2 梯度乘输入

常见归因为 $x_i\partial_iF(x)$。它依赖坐标原点：平移参数化 $u=x-c$ 后，乘数变为 $u_i$，即使表示同一物理输入。若“零向量”没有语义，梯度乘输入的解释也缺乏自然基线。

## M3.3 路径归因与积分梯度

**定义 M3.3（可容许路径）.** 给定输入 $x\in U$ 与基线 $x'\in U$，可容许路径是绝对连续映射 $\gamma:[0,1]\to U$，满足 $\gamma(0)=x'$、$\gamma(1)=x$，并满足：

1. $F$ 在 $\gamma(\alpha)$ 处对几乎处处的 $\alpha$ 可微；
2. 链式法则

   $$
   (F\circ\gamma)'(\alpha)=\nabla F(\gamma(\alpha))^\top\dot\gamma(\alpha)
   $$

   几乎处处成立；
3. 每个函数 $\alpha\mapsto \partial_iF(\gamma(\alpha))\dot\gamma_i(\alpha)$ 可积，且 $F\circ\gamma$ 绝对连续。

若 $F\in C^1(U)$ 且 $\gamma$ 绝对连续，则上述条件自动成立：紧集 $\gamma([0,1])\subset U$ 的某个邻域上 $\nabla F$ 有界，绝对连续映射的导数属于 $L^1$，而绝对连续链式法则给出第二项与 $F\circ\gamma$ 的绝对连续性。对含 ReLU 的分段仿射网络，可直接检查路径是否仅在零测集上穿过不可微分界面；若路径沿不可微分界面运行，不能只写“几乎处处可微”而省略链式法则检查。

**定义 M3.4（路径积分归因）.** 对可容许路径 $\gamma$，定义

$$
A_i^\gamma(F;x,x')
=\int_0^1\partial_iF(\gamma(\alpha))\dot\gamma_i(\alpha)\,d\alpha.
$$

**定理 M3.5（路径 completeness）.** 对定义 M3.3 的任意可容许路径，

$$
\sum_{i=1}^dA_i^\gamma(F;x,x')=F(x)-F(x').
$$

**证明.** 由可容许路径的第二项，几乎处处有

$$
(F\circ\gamma)'(\alpha)
=\sum_{i=1}^d\partial_iF(\gamma(\alpha))\dot\gamma_i(\alpha).
$$

右侧各项可积，有限和可与 Lebesgue 积分交换。又因 $F\circ\gamma$ 绝对连续，微积分基本定理给出

$$
\begin{aligned}
F(x)-F(x')
&=(F\circ\gamma)(1)-(F\circ\gamma)(0)\\
&=\int_0^1(F\circ\gamma)'(\alpha)\,d\alpha\\
&=\sum_{i=1}^d\int_0^1
\partial_iF(\gamma(\alpha))\dot\gamma_i(\alpha)\,d\alpha\\
&=\sum_{i=1}^dA_i^\gamma(F;x,x').
\end{aligned}
$$

证毕。

**定义 M3.6（直线积分梯度）.** 若线段 $[x',x]\subset U$，取直线路径

$$
\gamma(\alpha)=x'+\alpha(x-x'),\qquad\alpha\in[0,1],
$$

并假设它满足定义 M3.3。此时 $\dot\gamma_i=x_i-x_i'$，定义

$$
\operatorname{IG}_i(x;x')
=(x_i-x_i')\int_0^1
\frac{\partial F(\gamma(\alpha))}{\partial x_i}\,d\alpha.
$$

**推论 M3.7（integrated gradients 的 completeness）.** 在定义 M3.6 的假设下，

$$
\sum_{i=1}^d\operatorname{IG}_i(x;x')=F(x)-F(x').
$$

**证明.** 将直线路径代入定理 M3.5 即得。证毕。

completeness 是相对于端点和路径的守恒恒等式，不证明每个坐标归因具有唯一因果意义。换基线会改变总差，换路径可在总差不变时重新分配各项。[S01](INTERPRETABILITY_SOURCES.md#s01) 提出原始 IG 方法及其公理；[S02](INTERPRETABILITY_SOURCES.md#s02) 讨论其函数空间与唯一性主张的额外条件。本书只证明上述 completeness，不声称由它推出唯一性。

### 计算案例：同一个端点差的两种分配

取 $F(x_1,x_2)=x_1x_2$，从基线 $(0,0)$ 解释输入 $(1,1)$。沿直线 $\gamma(\alpha)=(\alpha,\alpha)$，有

$$
\partial_1F(\gamma(\alpha))=\alpha,
\qquad
\partial_2F(\gamma(\alpha))=\alpha.
$$

因此两项路径归因都是 $\int_0^1\alpha\,d\alpha=1/2$，总和为 $1=F(1,1)-F(0,0)$。

改用折线路径：先从 $(0,0)$ 走到 $(1,0)$，再走到 $(1,1)$。第一段只有 $x_1$ 变化，但此时 $x_2=0$，故两项积分都为零；第二段只有 $x_2$ 变化，且 $\partial_2F=x_1=1$，于是归因为 $(0,1)$。两条路径满足相同的 completeness，总分配却不同。这个计算把“路径依赖”具体化：守恒恒等式约束总量，却不替研究者选择语义上自然的变化顺序。

## M3.4 基线责任

基线 $x'$ 表示“缺少目标信息”或比较状态。图像黑底、文本零 embedding、padding token 和平均 embedding 是不同干预，可能都离开数据流形。报告必须展示多个有意义基线或论证基线选择。

若 $x'$ 与 $x$ 之间直线穿过不自然表示，积分梯度累积的是这些离分布点上的导数。路径方法不能仅凭积分形式获得语义自然性。

## M3.5 坐标与重参数化

设 $x=Bu$，其中 $B$ 可逆，并记 $G(u)=F(Bu)$。链式法则给出 $\nabla_uG=B^\top\nabla_xF$。路径线积分总和

$$
\int_\gamma \nabla F(x)^\top dx
$$

在同步变换路径后不变，但按坐标拆开的各项通常改变。输入像素或特定神经元坐标有工程意义，但“某坐标拥有贡献”不是一般线性重参数化不变量。

层、子空间或功能基归因可以提高稳定性，但也需要选择投影和度量。任何“自然基”主张都需独立理由。

## M3.6 归因与因果

梯度是数学函数的局部导数。若坐标可被合法操纵、其余条件可保持、路径处于目标干预范围，梯度可近似局部干预效应。若这些条件不成立，它只是一种敏感度。

对离散 token，有限替换、遮蔽或生成式反事实通常更接近可解释干预，但会同时改变多个表示因素。不存在不付假设代价的通用归因。

## M3.7 数值近似

积分梯度通常以 $m$ 个点的求积公式近似。设

$$
g_i(\alpha)=(x_i-x_i')\partial_iF(\gamma(\alpha)).
$$

若 $g_i$ 在 $[0,1]$ 上 Lipschitz，常数为 $L_i$，用右端点等距 Riemann 和 $\widehat{\operatorname{IG}}_i=m^{-1}\sum_{r=1}^m g_i(r/m)$，则

$$
\left|\widehat{\operatorname{IG}}_i-\operatorname{IG}_i\right|
\le \frac{L_i}{2m}.
$$

**证明.** 在第 $r$ 个长度 $1/m$ 的区间上，$|g_i(r/m)-g_i(t)|\le L_i(r/m-t)$。积分后该区间误差至多 $L_i/(2m^2)$；对 $m$ 个区间求和即得。证毕。

实际网络沿路径的导数可能在分段边界跳变，未必满足该 Lipschitz 条件；此时应做经验网格加密，而不能套用该界。无论采用何种求积，应报告步数、精度和 completeness 残差

$$
\varepsilon_{\mathrm{comp}}
=\left|F(x)-F(x')-\sum_i\widehat{\operatorname{IG}}_i\right|
$$

以及精度模式。小残差只验证数值 completeness，不验证解释正确。

乘法案例中的两条路径都可把 completeness 残差压到零，却分别得到 $(1/2,1/2)$ 与 $(0,1)$。因此实际报告若只给一条直线路径和一个小残差，仍没有区分“该路径具有领域语义”与“它只是方便的数值约定”。可执行的敏感性分析是预先给出若干语义可辩护路径，分别加密求积网格，再把路径间差异与数值误差分开报告；只有前者稳定，坐标分配才不依赖一次任意选路。

## 练习

**练习 M3.1.** 对 $F(x_1,x_2)=x_1x_2$、基线 $(0,0)$ 计算直线积分梯度。

**练习 M3.2.** 对同一函数改用折线路径：先改变 $x_1$，再改变 $x_2$。比较路径归因。

**练习 M3.3.** 给出梯度很大但有限合法扰动不改变离散输出的例子。

**练习 M3.4.** 证明线性函数 $F(x)=w^\top x+b$ 的积分梯度为 $w_i(x_i-x_i')$。

**练习 M3.5.** 为文本 embedding 选择两个不同基线，说明各自的语义困难。
