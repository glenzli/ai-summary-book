# 第八章：匹配、运行与可观测量工作流

## 本章目标

本章把前文结构组织成实际 SMEFT 分析流程：选择基、匹配、运行、计算可观测量、处理截断误差和拟合。

## 依赖前置知识

需要第二章匹配、第三章 RGE、第七章 Warsaw basis。

## 8.1 工作流

**流程 8.1（SMEFT 分析）.** 一个可审查的 SMEFT 分析至少包含：

1.  选择 EFT：SMEFT、HEFT、LEFT 或其他；
2.  选择算符基和 flavor 假设；
3.  指定截断阶数；
4.  在高尺度匹配 Wilson 系数；
5.  用 RGE 运行到实验尺度；
6.  计算可观测量；
7.  给出协方差、理论误差和 EFT 有效性切割；
8.  报告结果所依赖的基和假设。

## 8.2 可观测量展开

**定义 8.2（线性 SMEFT 近似）.** 若只保留 SM 与维数六振幅的干涉，则
$$
O
=
O_{\mathrm{SM}}
+
\sum_i\frac{C_i}{\Lambda^2}O_i^{\mathrm{int}}
+
O(\Lambda^{-4}).
$$

**定义 8.3（含平方项近似）.** 若还保留维数六振幅平方，则
$$
O
=
O_{\mathrm{SM}}
+
\sum_i\frac{C_i}{\Lambda^2}O_i^{\mathrm{int}}
+
\sum_{ij}\frac{C_iC_j}{\Lambda^4}O_{ij}^{\mathrm{quad}}
+
O(\Lambda^{-4}\text{ dim-8 int},\Lambda^{-6}).
$$

**警告 8.4.** “含平方项”不等于完整 $1/\Lambda^4$ 计算，因为同阶还有维数八算符与 SM 的干涉。

## 8.3 响应矩阵

令数据向量为 $d_a$，理论预测为 $t_a(C)$。在线性维数六近似下
$$
t_a(C)=t_a^{\rm SM}+M_{ai}C_i,
$$
其中 $C_i$ 可以表示 $C_i/\Lambda^2$ 的有量纲组合。矩阵
$$
M_{ai}={\partial t_a\over\partial C_i}\bigg|_{C=0}
$$
称为响应矩阵。

若协方差矩阵为 $\Sigma$，Gaussian likelihood 为
$$
\chi^2(C)=(d-t(C))_a(\Sigma^{-1})_{ab}(d-t(C))_b.
$$
Fisher 矩阵为
$$
F_{ij}=M_{ai}(\Sigma^{-1})_{ab}M_{bj}.
$$
若 $F$ 有零本征值，则存在 flat direction，数据不能约束对应 Wilson 组合。

**命题 8.5（基变换下的 Fisher 矩阵）.** 若 Wilson 坐标变换为 $C=R C'$，则
$$
M'=MR,\qquad F'=R^TFR.
$$
因此 flat direction 的物理子空间不依赖坐标，但其坐标表示依赖算符基。

**证明.** 由 $t=t_{\rm SM}+MC=t_{\rm SM}+MRC'$ 得 $M'=MR$。代入 Fisher 矩阵定义即得。$\square$

## 8.4 数据解释

**原则 8.6（基不变陈述）.** 实验约束的最终物理内容应能在不同算符基之间转换。若某结论只在一个基的单系数开关图中成立，则它是展示方式，不是基不变结论。

**原则 8.7（EFT 有效性切割）.** 对高能尾部分布，应报告最大不变量质量、横动量或能量切割，并比较其与假定 $\Lambda$ 的关系。

## 8.5 外部工具边界

**外部输入 8.8（全局拟合工具）.** SMEFiT、HEPfit、flavio、DsixTools、wilson 等工具可用于拟合、RGE 和低能匹配。本书不把任何工具输出作为定义；工具结果必须回译为本书的 EFT、基、尺度和截断语言。

## 8.6 分析记录模板

一个最小可复核分析记录应包含：

| 项目 | 内容 |
| --- | --- |
| EFT | SMEFT/HEFT/LEFT |
| 基 | Warsaw/Higgs/SILH/其他 |
| 尺度 | 匹配尺度、运行尺度、观测尺度 |
| Wilson 空间 | flavor、CP、是否单系数 |
| 截断 | 线性、含平方项、是否估计维数八 |
| 输入方案 | $\{\alpha,G_F,m_Z\}$ 等 |
| 数据 | 观测量、协方差、bin cuts |
| 有效性 | $E_{\max}/\Lambda$ 或替代判据 |
| 工具 | 版本、设置、随机种子或扫描方法 |

## 本章小结

SMEFT 的实用价值来自统一解释不同实验数据，但这种统一只有在基、尺度、截断和误差假设全部公开时才有可比性。

## 练习

**练习 8.1.** 说明为什么单独报告 $C_i/\Lambda^2$ 的限制而不说明 $\Lambda$ 或能区切割是不完整的。

**练习 8.2.** 构造一个二维 Wilson 系数拟合的协方差矩阵，并解释主轴方向为何依赖观测量组合。

**练习 8.3.** 设 $M=\begin{pmatrix}1&1\\1&1\end{pmatrix}$ 且 $\Sigma=I$，求 Fisher 矩阵并找出 flat direction。
