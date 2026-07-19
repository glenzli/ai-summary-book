# 第三章 梯度与输入归因

梯度方法从目标输出向输入或内部 activation 反向传播，回答“在当前点附近，哪个方向最能改变目标”。它们计算便宜、适用范围广，但局部导数不是完整因果解释。

## 3.1 选择标量目标

反向传播需要标量 $S(x)$。常见目标包括：

- 某 token logit $z_t$；
- 两个候选的 logit difference $z_a-z_b$；
- 序列 log likelihood；
- 分类损失；
- 某 SAE feature 或 neuron activation。

softmax 概率受所有词表项共同归一化，解释单个概率时可能混入不相关 logits。二选一任务通常优先使用 logit difference。

## 3.2 Vanilla Gradient

对连续输入表示 $e$，计算

$$
g=\nabla_e S(e).
$$

一阶 Taylor 展开给出

$$
S(e+\delta)
\approx S(e)+g^{\top}\delta.
$$

$g$ 描述无穷小扰动的局部敏感方向。梯度大表示附近变化快，不表示该输入在实际前向中贡献一定大；梯度为零也可能来自饱和，而非输入无关。

## 3.3 Gradient × Input

逐维乘积

$$
a_i=e_i\frac{\partial S}{\partial e_i}
$$

可视为相对于零向量的一阶贡献近似。对 token 常把 embedding 维度求和得到一个标量。

零 embedding 未必是有意义的“没有该 token”基线，LayerNorm 和非线性也会使缩放路径偏离真实输入流形。因此结果依坐标系和基线解释。

## 3.4 Integrated Gradients

给定基线 $e'$ 与输入 $e$，Integrated Gradients 定义

$$
\operatorname{IG}_i(e;e')
=(e_i-e'_i)
\int_0^1
\frac{\partial S(e'+\alpha(e-e'))}
{\partial e_i}\,d\alpha.
$$

在可微条件下满足 completeness：

$$
\sum_i\operatorname{IG}_i(e;e')
=S(e)-S(e').
$$

这个等式保证归因和等于端点差，不保证每个维度对应独立真实原因。路径与基线仍是方法的一部分；对文本 embedding 的直线路径会经过不对应任何 token 的向量。

## 3.5 SmoothGrad 与噪声平均

SmoothGrad 在输入附近加噪并平均梯度：

$$
\bar g(e)=\frac1M\sum_{m=1}^M
\nabla S(e+\epsilon_m).
$$

它可以减少视觉噪声，却改变了被解释对象：结果描述邻域平均敏感性。噪声分布、尺度和 embedding 空间几何都应报告。

## 3.6 Occlusion

删除、mask 或替换输入片段，比较

$$
\Delta_i=S(x)-S(x_{\setminus i}).
$$

这是有限反事实而非无穷小导数，更接近实际输入操作。但删除 token 会改变位置和语法，mask token 可能训练时不存在，替换也可能产生分布外文本。

更稳妥的做法是使用多种基线、成组删除和语义保持替换，检查结论是否一致。

## 3.7 交互作用

若两个输入因素共同作用，单独归因可能遗漏交互：

$$
I_{ij}=S(x)-S(x_{\setminus i})-S(x_{\setminus j})+S(x_{\setminus\{i,j\}}).
$$

Shapley value 通过对所有加入顺序平均分配交互贡献，具有良好公理性质，但精确计算指数昂贵，实际依赖采样近似和特征分组。不同 token 分组会得到不同解释单位。

## 3.8 内部 Gradient Attribution

对内部 activation $a$，可计算

$$
a_i\frac{\partial S}{\partial a_i}
$$

或沿计算图累积路径贡献。它比输入 saliency 更接近机制，但仍把非线性在当前点局部线性化。attention pattern、LayerNorm 分母和门控若随干预改变，固定梯度可能遗漏二阶与路径切换效应。

## 3.9 参数梯度不是推理解释

$\nabla_\theta S$ 表示改变参数对目标的局部影响，常用于 influence 或训练数据归因。它不是该输入前向传播时“哪些参数被使用”的简单地图：几乎所有参数都参与计算，梯度还依目标与当前参数化。

训练数据影响函数进一步需要 Hessian 近似和局部最优假设；在大模型非凸训练中应把结果视为近似证据。

## 3.10 Sanity Checks

归因图应至少接受：

- 随机化模型权重后是否明显变化；
- 随机化标签后是否失去任务结构；
- 不同基线和积分步数是否稳定；
- 删除高归因 token 是否比随机删除更影响目标；
- held-out 输入上是否预测行为变化；
- 与简单词频、位置和 token 长度基线相比是否增加信息。

视觉上平滑或符合直觉不是有效性检验。

## 3.11 结论

梯度方法精确计算局部敏感性，Integrated Gradients 又把沿路径的梯度积成端点差。它们适合快速定位和形成假说，但基线、坐标、饱和与交互决定解释边界。要声称某内部变量对行为有实际因果作用，仍需第七章的显式干预。
