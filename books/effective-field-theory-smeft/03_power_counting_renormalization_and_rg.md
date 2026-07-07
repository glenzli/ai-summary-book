# 第三章：幂计数、重整化与重整化群

## 本章目标

本章建立 EFT 的计算秩序：幂计数决定保留哪些算符，重整化群决定 Wilson 系数如何随尺度变化。

## 依赖前置知识

需要前两章的局域算符展开和 Wilson 系数。

## 3.1 幂计数

**定义 3.1（规范幂计数）.** 在四维弱耦合 EFT 中，若算符 $\mathcal O_i^{(d)}$ 的 Wilson 系数写为 $C_i^{(d)}/\Lambda^{d-4}$，则其对典型振幅的贡献按
$$
C_i^{(d)}\left(\frac{E}{\Lambda}\right)^{d-4}
$$
估计，忽略耦合常数、环因子和选择定则。

**警告 3.2.** 幂计数不是单纯量纲分析。强耦合、近阈值、手征破缺、loop suppression、flavor 选择和 helicity selection 都可能改变实际重要性排序。

**规则 3.3（振幅与截面截断）.** 若振幅写为
$$
A=A_{\rm SM}+{1\over\Lambda^2}A_6+{1\over\Lambda^4}A_8+\cdots,
$$
则线性维数六截断的截面为
$$
\sigma=\sigma_{\rm SM}+{2\over\Lambda^2}{\rm Re}(A_{\rm SM}A_6^\ast).
$$
若加入 $|A_6|^2/\Lambda^4$，则已经部分进入 $1/\Lambda^4$ 阶，必须同时估计遗漏的 $A_8$ 干涉。

**例 3.4（选择定则）.** 若 $A_{\rm SM}$ 与 $A_6$ 因 helicity、color 或 CP 选择定则不干涉，则线性项消失，领先 SMEFT 修正可能来自 $|A_6|^2$ 或维数八干涉。这时“维数六平方项”不再只是数值修正，而会改变首个非零阶。

## 3.2 重整化

**定义 3.5（Wilson 系数的重整化）.** 令裸系数和重整化系数满足
$$
C_{i,0}^{(d)}
=
\mu^{n_i\epsilon}
\left(
C_i^{(d)}(\mu)+\delta C_i^{(d)}(\mu)
\right),
\qquad d=4-2\epsilon.
$$
反项 $\delta C_i^{(d)}$ 用来吸收 EFT 圈图的 UV 发散。

**定义 3.6（反常维数矩阵）.** 若
$$
\mu\frac{d}{d\mu}C_i
=
\gamma_{ij}C_j,
$$
则 $\gamma_{ij}$ 称为 Wilson 系数的反常维数矩阵。

**命题 3.7（算符混合）.** 在重整化下，具有相同量子数的算符通常会混合；因此单个 Wilson 系数一般不是 RG 不变量。

**证明说明.** 圈图中插入 $\mathcal O_j$ 可产生与 $\mathcal O_i$ 同结构的 UV 发散。为保持重整化 Green 函数有限，必须用 $\delta C_i$ 吸收该发散，因此 $C_i$ 的 beta 函数含 $C_j$。$\square$

## 3.3 匹配与运行

**定义 3.8（匹配-运行工作流）.** 若 UV 模型在尺度 $\Lambda$ 匹配到 EFT，低能实验在尺度 $\mu$ 测量，则标准流程为：

1.  在 $\Lambda$ 处匹配得到 $C_i(\Lambda)$；
2.  用 RGE 演化到 $\mu$；
3.  在 $\mu$ 处计算矩阵元或可观测量；
4.  与数据比较。

**外部输入 3.9（SMEFT 维数六一圈 RGE）.** 完整维数六 SMEFT 一圈反常维数矩阵来自 Jenkins、Manohar、Trott 与 Alonso 等工作的系列计算。本书不重算该矩阵，只使用其结构和部分例子。

## 3.4 RGE 的解与尺度抵消

若 $\gamma$ 在考虑区间内可视为常数，则矩阵 RGE
$$
{dC\over d\log\mu}=\gamma C
$$
的解为
$$
C(\mu)=\exp\left[\gamma\log{\mu\over\Lambda}\right]C(\Lambda).
$$
一圈 leading-log 近似为
$$
C_i(\mu)=C_i(\Lambda)+\gamma_{ij}C_j(\Lambda)\log{\mu\over\Lambda}+O(\gamma^2).
$$

**命题 3.10（物理量的 $\mu$ 独立性）.** 设某可观测量到线性阶为
$$
O=O_{\rm SM}+C_i(\mu)M_i(\mu).
$$
若
$$
{dC_i\over d\log\mu}=\gamma_{ij}C_j,\qquad
{dM_i\over d\log\mu}=-\gamma_{ji}M_j,
$$
则 $dO/d\log\mu=0$ 到该阶成立。

**证明.** 直接求导：
$$
{dO\over d\log\mu}
=\gamma_{ij}C_jM_i-C_i\gamma_{ji}M_j.
$$
第二项交换 dummy 指标 $i\leftrightarrow j$ 后与第一项相消。$\square$

**解释 3.11.** Wilson 系数的尺度依赖不是物理效应本身；它必须与矩阵元的尺度依赖合并后才给出尺度无关的预测。

## 本章小结

幂计数给出截断原则，重整化给出尺度依赖，RG 把高尺度匹配与低尺度观测连接起来。

## 练习

**练习 3.1.** 设 $\mu dC/d\mu=\gamma C$ 且 $\gamma$ 为常数，解出 $C(\mu)$。

**练习 3.2.** 解释为什么“只打开一个 Wilson 系数”的说法一般不稳定于 RG。

**练习 3.3.** 对二维上三角矩阵
$$
\gamma=\begin{pmatrix}\gamma_1&a\\0&\gamma_2\end{pmatrix}
$$
写出 leading-log 解。
