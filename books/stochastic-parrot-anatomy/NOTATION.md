# 跨卷符号约定

局部变量在使用处定义。本表只约束跨卷重复出现的符号，避免把模型、训练分布、解码后分布和外部世界写成同一个对象。

## 模型与序列

| 符号 | 含义 |
|---|---|
| $V$ | 固定 tokenizer 的有限 token 词表 |
| $x_{1:n}$ | token 序列；$x_{<t}=x_{1:t-1}$ |
| $c$ | 除当前生成前缀外的条件，例如系统消息、用户输入或多模态编码 |
| $\theta$ | 一个固定模型 checkpoint 的参数 |
| $q_\theta(y\mid x)$ | 固定模型给出的条件分布 |
| $z\in\mathbb R^{|V|}$ | 词表上的 logits |
| $T>0$ | softmax temperature；greedy 另行定义，不把 $T=0$ 直接代入 softmax |
| $h_{\ell,p}$ 或 $x_{\ell,p}$ | 层 $\ell$、位置 $p$ 的 hidden/residual state |
| $W_U$ | 从最终表示投影到词表 logits 的 unembedding 矩阵 |

## 数据、训练与概率

| 符号 | 含义 |
|---|---|
| $P_D$ | 经采集与处理形成的数据分布 |
| $P_{train}$ | 来源混合和抽样权重实际定义的训练分布 |
| $\widehat P_n$ | $n$ 个观测定义的经验分布 |
| $\Theta$ | 把训练随机状态考虑在内时的参数随机变量 |
| $H(p)$ | 离散 Shannon 熵，默认使用自然对数 |
| $H(p,q)$ | 交叉熵 $-\sum p\log q$ |
| $D_{KL}(p\Vert q)$ | KL divergence；非对称，不是距离 |
| $\mathbb E[X]$、$\operatorname{Var}(X)$ | 期望与方差 |
| $\mathbf1\{A\}$ | 事件 $A$ 的指示量 |
| $do(X=x)$ | 因果模型中的干预；不同于观察条件 $X=x$ |

## 执行与系统

| 符号或术语 | 含义 |
|---|---|
| prefill | 对完整输入并行前向并建立 KV cache |
| decode step | 用当前缓存计算并选择下一 token 的一步 |
| $D$ | 已声明的解码策略，包括温度、截断、约束和停止规则 |
| $R$ | 运行时状态或记录；局部使用时须另行说明 |
| trace | 按时间排列的模型计算、运行时处理和外部事件 |
| commit boundary | 外部副作用越过后不能靠取消当前请求自动撤回的边界 |

## 可解释性

| 符号或术语 | 含义 |
|---|---|
| $S(x)$ | 被解释的标量目标，例如 logit difference |
| $a_u(x,p)$ | 单位 $u$ 在输入 $x$、位置 $p$ 的 activation |
| $\nabla_xS$ | 目标对输入的局部梯度 |
| $I$ | 明确声明的内部或输入干预算子 |
| $v$ | probe、steering 或 feature direction；构造方式必须另述 |
| $f$、$\hat x$ | 稀疏 feature activations 与重构 activation |
| $C$ | 相对于行为和输入分布定义的 circuit |

## 等式与比较

- `$=$` 表示所声明类型中的相等；文本相同、语义等价和任务成功不是默认同一关系。
- `≈` 只在局部已声明度量与容差时使用。
- 单次输出、条件分布和重复实验频率分别写作不同对象，不用一个“概率”互相替代。
- 局部章节可复用字母，但必须在首次出现处重新定义。
