# 第八章：匹配、运行与可观测量的计算链

一个 UV 模型在重阈值处生成的 Wilson 向量，必须经过基投影和重整化群演化，才能与远处实验的截面或衰变率比较。中间任何一步改变 flavor 假设、输入参数方案或截断口径，最终的“单个系数限制”都会代表不同问题。为了把这条链写成可计算对象，先在 $\mu_{\rm match}$ 处确定 $C_i$，再运行到 $\mu_{\rm obs}$，并把观测量在独立实 Wilson 坐标附近线性化为响应矩阵。由协方差加权得到的 Fisher 矩阵揭示数据真正约束的组合，而不是某个基中的列名。高能 bin 同时带来灵敏度与风险：$Q/M_{\rm gap}$ 检查局域性，$|C_i|(Q/\Lambda_{\rm ref})^{d-4}$ 检查插入层级，二者不能合并成一个任意参考尺度比值。

## 8.1 从重阈值到数据

**定义 8.1（跨尺度预测链）.** 一个 SMEFT 预测由下列相容数据组成：

1.  选择 EFT：SMEFT、HEFT、LEFT 或其他；
2.  选择算符基和 flavor 假设；
3.  指定保留的逆尺度/圈 bidegrees $(p,L)$ 和多次插入规则；
4.  区分物理阈值 $M_{\rm gap}$、参考尺度 $\Lambda_{\rm ref}$，并在 $\mu_{\rm match}$ 匹配 Wilson 系数；
5.  用同一基和 scheme 的 RGE 运行到 $\mu_{\rm obs}$；
6.  计算可观测量；
7.  给出协方差、理论误差和基于 $Q/M_{\rm gap}$ 的 EFT 有效性切割；
8.  报告结果所依赖的基和假设。

例如，一个重向量在 $\mu_{\rm match}\simeq M_X$ 处产生 $C_{\ell q}^{(1)}$，并不意味着实验直接测量这一初值。RGE 会在 $\mu_{\rm obs}$ 处生成与它混合的其他系数，破缺相和输入方案把这些系数重组为振幅，最后 cuts 与协方差才决定数据约束的方向。下面的观测量展开逐层固定这条链的截断口径。

## 8.2 可观测量展开

**定义 8.2（线性 SMEFT 近似）.** 假设 exact baryon/lepton number 或其他精确选择定则排除奇数 $p$，并分别声明所保留的 SM 圈阶与一次维数六插入圈阶。令有量纲坐标 $c_i=C_i^{(6)}/\Lambda_{\rm ref}^2$，则
$$
O
=
O_{\mathrm{SM}}
+
\sum_i c_i O_i^{\mathrm{int}}
+
R_{p\ge4}+R_{\rm loops}.
$$
$R_{p\ge4}$ 表示未算的逆尺度阶，$R_{\rm loops}$ 表示在各已保留 $p$ 阶未算的圈阶；二者不得合并成一个没有口径的误差条。

**定义 8.3（含维数六平方项的部分 $p=4$ 近似）.** 若在线性式上再保留维数六振幅平方，则
$$
O
=
O_{\mathrm{SM}}
+
\sum_i c_i O_i^{\mathrm{int}}
+
\sum_{ij}c_ic_jO_{ij}^{\mathrm{quad}}
+
R_{p=4}^{\rm amp}+R_{p\ge6}+R_{\rm loops}.
$$
其中 $R_{p=4}^{\rm amp}$ 至少含 SM 与一次维数八插入的干涉、SM 与两次维数六插入振幅的干涉，以及由输入参数和相空间展开产生的同阶项。

**警告 8.4.** “含平方项”不等于完整 $p=4$ 或 $1/\Lambda_{\rm ref}^4$ 计算。若线性干涉因 helicity/CP 选择而消失，平方项可能是首个非零数值项，但这不使 $R_{p=4}^{\rm amp}$ 自动变小；必须给出额外 power-counting 或 UV 假设。

## 8.3 响应矩阵

按第四章的 Hermiticity 约束把复 Wilson 系数拆成独立实部/虚部后，取实坐标 $c\in\mathbb R^n$；令数据与预测为 $d,t(c)\in\mathbb R^m$。在线性维数六近似下
$$
t_a(c)=t_a^{\rm SM}+M_{ai}c_i,
$$
其中 $c_i$ 的质量维数为 $-2$。矩阵
$$
M_{ai}={\partial t_a\over\partial c_i}\bigg|_{c=0}
$$
称为响应矩阵。它是固定输入方案、cuts、尺度、PDF/辐射修正和 nuisance parameters 后在 SM 点的 Jacobian 线性化，不是对任意大 $c$ 都成立的全局模型。

假设协方差矩阵 $\Sigma$ 是与 $c$ 无关的实对称正定矩阵。忽略与 $c$ 无关的归一化常数，Gaussian likelihood 的二次型为
$$
\chi^2(c)=(d-t(c))_a(\Sigma^{-1})_{ab}(d-t(c))_b.
$$
Fisher 矩阵为
$$
F_{ij}=M_{ai}(\Sigma^{-1})_{ab}M_{bj}.
$$
若 $\Sigma$ 仅半正定，则 $\Sigma^{-1}$ 未定义；必须先限制到其 support，或声明使用 Moore--Penrose pseudoinverse 及相应 likelihood measure。

**命题 8.4A（Fisher 核等于响应核）.** 在上述正定假设下，$F$ 半正定且
$$
\ker F=\ker M.
$$
因此 $F$ 的零本征方向恰是线性预测不响应的 Wilson 组合。

**证明.** 对任意实坐标向量 $x$，
$$
x^TFx=(Mx)^T\Sigma^{-1}(Mx)\ge0.
$$
由于 $\Sigma^{-1}$ 正定，等号成立当且仅当 $Mx=0$。故 $x\in\ker F$ 当且仅当 $x\in\ker M$。$\square$

**命题 8.5（基变换下的 Fisher 矩阵）.** 若两个完整实线性坐标系由 $c=T c'$ 联系，其中 $T\in GL_n(\mathbb R)$ 且与数据无关，则
$$
M'=MT,\qquad F'=T^TFT.
$$
因此 flat direction 的物理子空间不依赖坐标，但其坐标表示依赖算符基。

**证明.** 由 $t=t_{\rm SM}+Mc=t_{\rm SM}+MTc'$ 得 $M'=MT$。代入 Fisher 矩阵定义即得。$\square$

**警告 8.5A（非线性或截断依赖的“换基”）.** 若转换还含 $c_i'c_j'$、输入参数位移，或从完整空间投影到单系数子空间，则它不是命题 8.5 的可逆线性坐标变换。此时 Fisher 秩、置信区域和平方项必须用完整 Jacobian/Hessian 重新计算，不能只乘一个常数矩阵。

## 8.4 数据解释

**原则 8.6（基不变陈述）.** 实验约束的最终物理内容应能在不同算符基之间转换。若某结论只在一个基的单系数开关图中成立，则它是展示方式，不是基不变结论。

**原则 8.7（EFT 有效性切割）.** 对高能尾部分布，应逐 bin 报告构造 $Q$ 的独立不变量、横动量和 cuts，并在条件化的 UV 假设下比较 $Q/M_{\rm gap}$。$Q/\Lambda_{\rm ref}$ 单独不是有效性检验；Wilson 插入大小应使用参考尺度不变的 $|C_i|(Q/\Lambda_{\rm ref})^{d-4}$ 组合。

## 8.5 外部工具边界

**外部输入 8.8（全局拟合工具）.** SMEFiT、HEPfit、flavio、DsixTools、wilson 等工具可用于拟合、RGE 和低能匹配。本书不把任何工具输出作为定义；工具结果必须回译为本书的 EFT、基、尺度和截断语言。

## 8.6 固定一次分析所需的数据

使响应矩阵具有确定含义的数据包括：

| 项目 | 内容 |
| --- | --- |
| EFT | SMEFT/HEFT/LEFT |
| 基 | Warsaw/Higgs/SILH/其他 |
| 物理/坐标尺度 | $M_{\rm gap}$ 假设、$\Lambda_{\rm ref}$、$\mu_{\rm match}$、$\mu_{\rm obs}$ |
| Wilson 空间 | flavor、CP、是否单系数 |
| 截断 | 保留的 $(p,L)$、多次插入、是否仅含维数六平方项 |
| 输入方案 | $\{\alpha,G_F,m_Z\}$ 等 |
| 数据 | 观测量、协方差、bin cuts |
| 有效性 | $Q_{\max}/M_{\rm gap}$、插入层级、loop/log 检查 |
| 工具 | 版本、设置、随机种子或扫描方法 |

## 8.7 从 Wilson 初值到数据方向

匹配给出高尺度 Wilson 初值，RGE 把它们送到观测尺度，响应矩阵再把独立实坐标映到数据空间。Fisher 核与响应核相同，所以 flat direction 是观测组合缺少灵敏度的事实，而不是某个基的命名问题。这个统一解释只有在物理谱隙、参考尺度、输入方案、$(p,L)$ 截断和协方差都固定后才成立；超出 SM 邻域或加入部分平方项时，线性 Jacobian 也必须连同高阶误差重新评估。

## 练习

**练习 8.1.** 说明为什么报告 $c_i=C_i/\Lambda_{\rm ref}^2$ 的限制仍不足以推出新粒子质量界，并列出还需给定的 $M_{\rm gap}$、UV coupling 与能区信息。

**练习 8.2.** 构造一个二维 Wilson 系数拟合的协方差矩阵，并解释主轴方向为何依赖观测量组合。

**练习 8.3.** 设 $M=\begin{pmatrix}1&1\\1&1\end{pmatrix}$ 且 $\Sigma=I$，求 Fisher 矩阵并找出 flat direction。
