# 第三章 梯度与输入归因

梯度方法从标量目标向输入或内部 activation 反向传播，精确回答“在当前连续表示附近，哪个无穷小方向最快改变目标”。它们计算便宜、覆盖整张计算图，但局部敏感性、端点归因和自然因果效应是三种不同对象。

## 3.1 目标、坐标与解释单位

反向传播需要标量 $S$。常见选择包括：

- token logit $z_t$；
-候选 logit difference $z_a-z_b$；
- 序列 log likelihood；
- 某内部 feature activation；
- 任务损失或 verifier score。

softmax probability 受全词表归一化影响。二选一机制通常优先使用 logit difference。对序列目标，还要声明按 token 求和还是平均；长度变化会改变结论。

输入“维度”也需定义：embedding coordinate、token、span、图像 patch 或内部 feature。先逐坐标归因再求和与先把一组坐标视为整体，不一定在非线性方法下相同。

## 3.2 Vanilla gradient：局部敏感性

对连续表示 $e\in\mathbb R^d$，

$$
g(e)=\nabla_e S(e).
$$

若 $S$ 在包含线段 $\{e+t\delta:0\le t\le1\}$ 的开集上二阶连续可微，则把该线段限制为一元函数后，存在 $t^*\in(0,1)$，令 $e^*=e+t^*\delta$，有 Taylor 展开

$$
S(e+\delta)-S(e)
=g(e)^\top\delta
+\frac12\delta^\top H(e^*)\delta.
$$

梯度只控制一阶项。梯度大表示附近沿某方向变化快；梯度为零可能来自饱和或局部极值；有限替换若曲率大，符号甚至可能与一阶预测相反。

坐标换为 $e'=Ae$ 后，梯度按 $g'=A^{-\top}g$ 变换。逐坐标 saliency 不是换基不变量；内积 $g^\top\delta$ 才对应具体扰动的一阶效应。

## 3.3 Gradient × Input

相对于零基线的一阶贡献常写为

$$
a_i=e_i\frac{\partial S}{\partial e_i}.
$$

对 token embedding 常求和 $A_{token}=e^\top\nabla_eS$。它近似沿 $e\mapsto(1-\alpha)e$ 缩放到零时的局部效应，并不是“删除 token”的真实操作。

零 embedding 未必表示缺失 token；LayerNorm 还会使整体缩放近似被消除。必要控制是与实际 token 替换、多个基线和 norm-matched 扰动比较。

## 3.4 Integrated Gradients 与路径完整性

给定基线 $e'$ 和路径 $\gamma:[0,1]\to\mathbb R^d$，一般路径归因为

$$
\operatorname{PI}_i
=\int_0^1
\frac{\partial S(\gamma(\alpha))}{\partial e_i}
\frac{d\gamma_i(\alpha)}{d\alpha}
\,d\alpha.
$$

直线路径 $\gamma(\alpha)=e'+\alpha(e-e')$ 给出 Integrated Gradients：

$$
\operatorname{IG}_i(e;e')
=(e_i-e_i')
\int_0^1
\frac{\partial S(e'+\alpha(e-e'))}{\partial e_i}
\,d\alpha.
$$

若复合函数 $S\circ\gamma$ 绝对连续，且链式法则

$$
\frac{d}{d\alpha}S(\gamma(\alpha))
=\nabla S(\gamma(\alpha))^\top\gamma'(\alpha)
$$

几乎处处成立，则由绝对连续函数的微积分基本定理得到

$$
\sum_i\operatorname{IG}_i(e;e')=S(e)-S(e').
$$

这里的 **completeness** 只表示坐标归因之和等于两个端点的 score 差。它不表示归因覆盖了“真实机制”，也不表示每一项是无交互的独立原因。本卷后文所说 circuit completeness 是另一概念。

IG 满足实现不变性等公理，但实验结果仍依基线、路径与数值积分。文本 embedding 的直线路径大多不对应自然 token 序列。

## 3.5 数值积分与不确定性

用 $m$ 个点近似 IG：

$$
\widehat{\operatorname{IG}}_i
=(e_i-e_i')\frac1m
\sum_{k=1}^{m}
\frac{\partial S(e'+k(e-e')/m)}{\partial e_i}.
$$

应报告 completeness residual

$$
r_{IG}=S(e)-S(e')-
\sum_i\widehat{\operatorname{IG}}_i,
$$

并增加 $m$ 检查收敛。ReLU 之类连续分段光滑映射通常仍沿有限线段绝对连续；量化、离散 TopK 或硬条件路由却可能使 $f:=S\circ\gamma$ 发生跳跃。经典梯度在跳点处未定义，且区间内梯度积分不包含跳跃量，因此 completeness 可失败。

若 $f$ 是有界变差函数，其分布导数可分解为绝对连续、跳跃和奇异连续三部分。令 $J$ 为跳点集，$D^cf$ 为奇异连续部分，则

$$
f(1)-f(0)
=\int_0^1 f'(\alpha)\,d\alpha
+\sum_{\alpha\in J}\bigl(f(\alpha+)-f(\alpha-)\bigr)
+D^cf((0,1)).
$$

跳跃和的绝对收敛由有界变差保证。若 $f$ 在跳点之间绝对连续且没有奇异连续部分，则 $D^cf=0$，得到常用的“梯度积分加跳跃和”。在链式法则成立的区间，$f'(\alpha)=\nabla S(\gamma(\alpha))^\top\gamma'(\alpha)$。因此 $r_{IG}$ 可能同时包含数值积分误差、遗漏的跳跃质量与奇异连续变化，不能只靠增加 $m$ 解释。研究者应显式报告离散边界，或改用有限干预和混合系统归因。

Expected Gradients 对基线分布 $e'\sim D_0$ 再平均，可降低单基线任意性：

$$
\operatorname{EG}_i(e)
=\mathbb E_{e'\sim D_0}
[\operatorname{IG}_i(e;e')].
$$

它没有消除基线问题，只把问题改为为何选择 $D_0$。

## 3.6 SmoothGrad：邻域平均敏感性

SmoothGrad 计算

$$
\bar g(e)=\frac1M\sum_{m=1}^{M}
\nabla S(e+\epsilon_m),
\qquad
\epsilon_m\sim Q.
$$

它可减少可视化噪声，却改变估计对象：结果描述 $Q$ 定义邻域内的平均梯度。噪声协方差决定哪些方向被平滑；对 embedding 各向同性高斯噪声未必对应语义邻域。

必要报告包括 $Q$、噪声尺度、样本数、不同尺度稳定性及性能是否在噪声下保持。

## 3.7 Occlusion 与有限替换

对删除或替换操作 $R_i$，定义

$$
\Delta_i^{R}=S(x)-S(R_i(x)).
$$

它是实际有限反事实，不是导数。删除 token 会移动位置并改变语法；mask token 可能在生成模型训练中罕见；用语言模型补全又引入新的随机模型。

更可靠的设计使用多种 $R_i$：删除、同类词替换、span 重采样与保持长度的 neutral token，并把不同操作的结论分别报告。不能把它们平均成一个无条件“重要性”。

## 3.8 交互与 Shapley 分配

两个因素的有限差分交互为

$$
I_{ij}=S(x)-S(x_{-i})-S(x_{-j})+S(x_{-\{i,j\}}).
$$

Shapley value 把 coalition value $v(A)$ 的增量对所有加入顺序平均：

$$
\phi_i
=\sum_{A\subseteq N\setminus\{i\}}
\frac{|A|!(|N|-|A|-1)!}{|N|!}
\bigl(v(A\cup\{i\})-v(A)\bigr).
$$

它给出满足一组分配公理的 credit，不发现唯一物理原因。文本中“缺少某 token”的条件分布和 feature 分组决定 $v(A)$；精确计算指数昂贵，采样误差也应报告。

## 3.9 内部 gradient attribution

对内部 activation $a$，常用

$$
A_i=a_i\frac{\partial S}{\partial a_i}
$$

或对从 corrupt 值 $a_r$ 到 clean 值 $a_c$ 的变化使用一阶近似

$$
\widehat\Delta S
=(a_c-a_r)^\top\nabla_aS(a_r).
$$

后式可快速筛选 patch sites，但若替换跨越非线性、attention pattern 切换或 LayerNorm 尺度变化，近似误差会增大。应抽样运行真实 patch 来校准排名，而不是把 gradient proxy 当作已执行干预。

参数梯度 $\nabla_\theta S$ 则描述参数微扰。它可用于训练数据影响近似，却不是前向时“哪些参数被使用”的地图。几乎全部参数参与计算，Hessian 与优化路径假设决定 influence 结论。

## 3.10 Sanity checks 与基线

归因方法至少接受以下检查：

1. 随机化模型参数后，任务相关结构是否消失；
2. 随机化标签或目标后，归因是否失去对应规律；
3. 不同基线、路径、积分点和噪声尺度是否稳定；
4. 删除高归因单位是否比随机或 norm-matched 单位更影响目标；
5. 归因是否预测 held-out 反事实的效应大小和符号；
6. 与位置、token 长度、频率和 embedding norm 基线相比是否增加信息；
7. 正负归因是否都展示，是否存在抵消；
8. 多方法一致是否超出它们共享假设所能解释的程度。

参数随机化后的图仍相似，可能说明方法主要反映输入边缘或 embedding 结构。视觉上平滑、符合直觉不是有效性标准。

## 3.11 Faithfulness、Sensitivity 与 Completeness

三个词必须分开：

- **sensitivity**：目标对指定局部扰动的导数或有限差分；
- **attribution completeness**：归因项按定义加总为某端点差；
- **mechanistic faithfulness**：解释是否跟随原模型实际内部计算与干预响应。

高 sensitivity 不表示自然输入中该变量发生过变化；IG completeness 不表示路径在 activation manifold 上；删除高归因单位成功也可能是破坏而不是精准移除目标信息。

## 3.12 方法审计表

| 方法 | 问题与对象 | 操作/估计量 | 必要控制 | 能支持 | 不能支持与失效 |
|---|---|---|---|---|---|
| vanilla gradient | 当前点附近何方向敏感 | 反传；$\nabla S$ | 尺度、曲率、有限差分 | 局部一阶敏感性 | 有限因果贡献；饱和/换基 |
| grad×input | 相对零缩放的一阶份额 | $e_i\partial_iS$ | 非零基线、norm-matched 扰动 | 局部缩放近似 | token 删除效应 |
| Integrated Gradients | 基线到输入路径怎样分配端点差 | 路径积分；IG | 多基线、积分收敛、路径自然度 | 指定路径的完整归因 | 唯一机制；路径依赖 |
| SmoothGrad | 邻域平均是否更稳定 | 加噪平均梯度 | 噪声分布与性能保持 | 指定邻域敏感性 | 原输入单点原因 |
| occlusion | 有限替换怎样改变输出 | 删除/替换；$\Delta^R$ | 多替换、自然度、位置 | 指定反事实效应 | 无条件 token 贡献 |
| Shapley | 如何按 coalition 公理分配效应 | 子集重采样；$\phi_i$ | 缺失分布、分组、采样误差 | 给定 value function 的公平分配 | 唯一因果分解 |
| internal attribution | 哪些内部 site 值得真 patch | 梯度线性化 | 真实 patch 校准、曲率 | 候选路径与局部贡献 | 完整路径因果效应 |

梯度路线最适合快速定位、比较局部敏感方向和近似筛选路径。它的严谨性来自目标、基线和误差定义，而不是公式本身的复杂度。要建立内部变量的实际干预效应，仍需第七章的节点与路径实验。
