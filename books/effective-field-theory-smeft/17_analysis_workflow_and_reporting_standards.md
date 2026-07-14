# 第十七章：从 Wilson 系数到可比较约束

设一项分析给出 $C_{\ell q}^{(1)}\in[-0.2,0.1]$，这个区间本身还不是物理结论：它没有说明 flavor 分量、$\Lambda_{\rm ref}$ 与定义尺度、输入方案、保留的圈阶，也没有交代高质量 bin 是否位于 $M_{\rm gap}$ 以下。缺少这些数据时，不同分析即使使用同一个系数名也不能比较。把 UV 匹配、RGE、破缺相参数重定义、振幅计算和协方差拟合连成一条可复算的链，可以明确每个数值在哪一步产生，以及截断误差如何传播。这里以一条 Wilson 限制从高尺度到数据表的生命周期为线索，建立结果所需的最小物理数据，并用混基、隐含 flavor、越过阈值和遗漏协方差的反例说明每项数据解决的具体歧义。

## 17.1 一个约束的定义域

**规则 17.1（SMEFT 结果最小元数据）.** 一个 SMEFT 约束或预测至少报告：

1.  EFT 类型：SMEFT、HEFT、LEFT 或其他；
2.  算符基；
3.  Wilson 坐标归一化 $\Lambda_{\rm ref}$ 与重整化定义尺度 $\mu$；
4.  flavor 假设；
5.  CP 假设；
6.  输入参数方案；
7.  截断阶数；
8.  是否保留维数六平方项；
9.  数据硬尺度 $Q$、条件化的物理谱隙 $M_{\rm gap}$ 与有效性切割；
10. Wilson 插入层级；
11. 理论误差和协方差处理。

## 17.2 跨尺度计算链

**定义 17.2（计算链）.**

1.  写出 UV 或高尺度 Wilson 系数，并区分 $M_{\rm gap}$ 与 $\Lambda_{\rm ref}$。
2.  在匹配尺度 $\mu_{\rm match}$ 给出 $C_i^{(d)}(\mu_{\rm match})$。
3.  用 RGE 运行到实验尺度。
4.  转到破缺相并选择输入参数方案。
5.  计算振幅或截面。
6.  线性化或保留平方项。
7.  与数据协方差比较。
8.  分别报告 $Q/M_{\rm gap}$ 局域性、Wilson 插入层级与 loop/log 检查。

**例 17.2A（重向量到 dilepton bin）.** 取第十章的单一重向量模型，
$M_X=3\,\mathrm{TeV}$、$g_X=1$，并把 flavor coupling 归一为一。该模型的最近
遗漏 pole 给出 $M_{\rm gap}=M_X$。若选择
$\Lambda_{\rm ref}=1\,\mathrm{TeV}$，树级匹配在
$\mu_{\rm match}\simeq3\,\mathrm{TeV}$ 给出
$$
C_{\ell q}^{(1)}(\mu_{\rm match})
=-\frac{\Lambda_{\rm ref}^2}{M_X^2}
=-\frac19.
$$
将系数运行到观测尺度后，若某 bin 的 $Q_{\max}=1.2\,\mathrm{TeV}$，则局域比值
$\rho_{\rm loc}=Q_{\max}/M_{\rm gap}=0.4$；忽略过程权重时，该插入的粗略大小为
$\rho_{\rm ins}=|C_{\ell q}^{(1)}|(Q_{\max}/\Lambda_{\rm ref})^2=0.16$。
改变 $\Lambda_{\rm ref}$ 并按命题 1.5 重标度 $C_{\ell q}^{(1)}$ 不会改变
$\rho_{\rm ins}$，也不会改变由物理质量确定的 $\rho_{\rm loc}$。两项数值随后与
RGE、PDF、cuts 和协方差一起进入限制。

**原则 17.3.** 如果中间任一步省略，则最终结论应降级为“估计”或“投影”，不得称为完整 SMEFT 约束。

## 17.3 使数值可比较的数据

正式结果至少应包含如下表格。

| 字段 | 示例 | 缺失后果 |
| --- | --- | --- |
| EFT | SMEFT dimension-six | 不知道自由度和对称性 |
| 基 | Warsaw | Wilson 坐标不可解释 |
| Wilson 坐标/运行尺度 | $\Lambda_{\rm ref}=1\ {\rm TeV}$，$\mu=1\ {\rm TeV}$ | 归一化与 RGE 位置不明确 |
| 物理谱隙 | 条件假设 $M_{\rm gap}\ge Q_{\max}/\rho_*$ | 无法判断局域展开域 |
| flavor | diagonal nonuniversal | 参数数不明确 |
| CP | CP-even | 是否含 EDM 方向不明确 |
| 输入方案 | $\{\alpha,G_F,m_Z\}$ | 电弱预测不可复现 |
| 截断 | linear dim-6 | 维数六平方和维数八边界不清 |
| 数据 | bins and covariance | likelihood 不可复现 |
| 有效性 | $\rho_{\rm loc}=Q/M_{\rm gap}\le\rho_*<1$，并逐 bin 报告 $Q$ | 高能 bin 可能越过遗漏 pole/threshold |
| Wilson 插入 | 逐 bin 报告 $|C_i^{(d)}(\mu)|(Q/\Lambda_{\rm ref})^{d-4}$ | 无法检查所声明的插入截断层级 |
| 工具 | version/hash | 数值结果不可追踪 |

## 17.4 常见错误

**错误 17.4（混合基）。** 从一个基读取 Wilson 系数，却用另一个基的 Feynman 规则计算。

**错误 17.5（隐含 flavor 假设）。** 报告“$C_{\ell q}$ 的限制”而不说明 flavor 指标组合。

**错误 17.6（过度使用高能 bin）。** 使用 $Q\sim M_{\rm gap}$ 或
$Q>M_{\rm gap}$ 的数据，却仍宣称局域截断 EFT 结果模型无关。若 $M_{\rm gap}$
未知，必须把 cut 和结论写成对 $M_{\rm gap}$ 的条件陈述。

**原则 17.6A（局域性与插入大小分开报告）.** 对 dimension-$d$ 算符，略去过程特有
耦合、helicity 和群论因子后，Wilson 插入大小写为
$$
\rho_{\rm ins,i}^{(d)}(Q)
=|C_i^{(d)}(\mu)|
\left(\frac{Q}{\Lambda_{\rm ref}}\right)^{d-4}.
$$
在 $\Lambda_{\rm ref}\mapsto a\Lambda_{\rm ref}$、
$C_i^{(d)}\mapsto a^{d-4}C_i^{(d)}$ 下，$\rho_{\rm ins,i}^{(d)}$ 不变；它检查 Wilson
插入层级。局域性则由 $\rho_{\rm loc}=Q/M_{\rm gap}$ 检查。只有在明确 UV matching 的模型中先
声明单一物理重尺度 $M$，并验证最近遗漏奇点满足 $M_{\rm gap}=M$，才可再选择
$\Lambda_{\rm ref}=M$。这项坐标选择不是一般 SMEFT 有效性条件。

**错误 17.7（维数六平方项误读）。** 保留
$C_i^{(6)}C_j^{(6)}/\Lambda_{\rm ref}^4$ 项，却声称结果在所声明假设下完整到
$p=4$。

**错误 17.8（把单系数图当作模型无关结论）。** 单系数限制是在其他 Wilson 系数固定为零的切片上得到的结果。若 UV 模型同时产生多个系数，该限制不能直接套用。

**错误 17.9（省略协方差）。** 多个 bins 或多个观测量共用系统误差时，忽略协方差会改变 Fisher 矩阵和 flat directions。

## 17.5 结果的可重用层级

| 等级 | 内容 |
| --- | --- |
| Level 0 | 只给文字结论或图 |
| Level 1 | 给 Wilson 限制和基本假设 |
| Level 2 | 给响应矩阵或 likelihood |
| Level 3 | 给数据、协方差、代码版本和有效性切割 |
| Level 4 | 给完整复现脚本和随机扫描配置 |

**原则 17.10.** 只有响应矩阵或 likelihood 已给出时，不同 Wilson 假设才能在同一数据上重新计算；若还给出数据、协方差、有效性切割和执行配置，结果才可独立数值复现。

## 17.6 一项约束所携带的物理信息

Wilson 区间是整条计算链的末端，而不是脱离上下文的观测量。例 17.2A 显示，即使在单重态模型中数值上可令 $M_{\rm gap}$、匹配尺度和参考尺度彼此接近，它们仍分别描述 pole 位置、重整化条件和坐标归一化。基、flavor、输入方案与截断确定响应矩阵的列，数据和协方差确定被约束的方向；缺少其中任一项，两个看似同名的系数区间便没有共同定义域。

## 练习

**练习 17.1.** 设计一页 SMEFT 拟合结果表，包含规则 17.1 的全部元数据，并把
$Q/M_{\rm gap}$ 有效性与
$|C_i^{(d)}(\mu)|(Q/\Lambda_{\rm ref})^{d-4}$ 插入层级分列。

**练习 17.2.** 找一个“单系数限制”图，列出它隐含的假设，包括是否先声明
$M_{\rm gap}=M$ 的模型依赖单尺度 UV matching，再选择 $\Lambda_{\rm ref}=M$。

**练习 17.3.** 用第 17.3 节的字段记录一个 Higgs 信号强度分析，并指出哪些字段会改变响应矩阵，哪些字段只改变 likelihood 权重。
