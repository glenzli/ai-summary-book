# 第二十四章：基准数值响应矩阵

## 本章目标

本章给出不依赖具体实验数据集的基准数值响应矩阵。它们不是实验拟合结果，而是用于检查量纲、符号、退化方向和协方差接入方式的教材模板。真实分析应把本章的基准矩阵替换为对应数据集和工具链生成的响应矩阵。

## 依赖前置知识

需要第八章的 Fisher 矩阵、第二十一章的输入方案和第二十三章的响应矩阵定义。

## 24.1 电弱精密基准：$m_W$

取基准值
$$
s^2=0.231,\qquad c^2=0.769,\qquad c^2-s^2=0.538.
$$
第二十三章给出
$$
{\delta m_W^2\over m_W^2}
=
{s^2\over c^2-s^2}\epsilon_\alpha
-{s^2\over c^2-s^2}\epsilon_G
-{c^2\over c^2-s^2}\epsilon_Z
+\epsilon_W.
$$
因此数值响应行为
$$
M_{m_W^2}
\simeq
\begin{pmatrix}
0.429&-0.429&-1.430&1.000
\end{pmatrix}.
$$
对 $m_W$ 本身，
$$
M_{m_W}
\simeq
\begin{pmatrix}
0.215&-0.215&-0.715&0.500
\end{pmatrix}.
$$

**解释 24.1.** $\epsilon_Z$ 的系数较大，是因为 $m_W$ 预测对中性规范质量关系的位移敏感。该数值不代表实验误差，只代表输入方案的线性传播。

## 24.2 Higgs 基准：生产、衰变、总宽度

定义三个归一化振幅参数
$$
\theta=(a_g,\delta y_t,\delta_\Gamma)^T,
$$
其中 $a_g$ 是 $hgg$ contact 振幅相对 SM loop 振幅的归一化贡献，$\delta y_t$ 是 top-loop 振幅的相对位移，$\delta_\Gamma$ 是总宽度相对位移。对两个简化信号强度
$$
\mu_1=gg\to h\to ZZ,\qquad
\mu_2=gg\to h\to \gamma\gamma,
$$
若只让生产端和总宽度变化，则
$$
\Delta\mu_1=\Delta\mu_2=2a_g+2\delta y_t-\delta_\Gamma.
$$
响应矩阵为
$$
M_H=
\begin{pmatrix}
2&2&-1\\
2&2&-1
\end{pmatrix}.
$$
该矩阵秩为 $1$。它有两个 flat directions，例如
$$
(1,-1,0),\qquad (1,0,2).
$$

**解释 24.2.** 若不加入 $t\bar th$、VBF、总宽度约束或其他衰变道，这两个信号强度不能分离 $hgg$ contact、top Yukawa shift 和 total width shift。

## 24.3 高质量 dilepton 基准

取两个 bin 的无量纲能量比
$$
x_b={s_b\over\Lambda^2},\qquad x_1=0.25,\quad x_2=1.00.
$$
对一个 contact 参数 $c$，线性响应设为
$$
\Delta_b=x_b c.
$$
则
$$
M_{\ell\ell}=
\begin{pmatrix}
0.25\\
1.00
\end{pmatrix}.
$$
若保留平方项，取
$$
\Delta_b=x_bc+x_b^2c^2,
$$
则二次响应为
$$
Q_{\ell\ell}=
\begin{pmatrix}
0.0625\\
1.0000
\end{pmatrix}.
$$
第二个 bin 的灵敏度高，但其平方项也大得多，因此它对 EFT validity 更敏感。

## 24.4 Fisher 矩阵例子

设两个 dilepton bin 误差独立且
$$
\Sigma=\mathrm{diag}(0.1^2,0.2^2).
$$
单参数 Fisher 信息为
$$
F=M^T\Sigma^{-1}M
={0.25^2\over0.1^2}+{1.00^2\over0.2^2}
=6.25+25=31.25.
$$
因此一标准差误差估计为
$$
\sigma_c={1\over\sqrt F}\simeq0.179.
$$
若移除第二个高能 bin，则
$$
F=6.25,\qquad \sigma_c=0.400.
$$
这展示了高能 bin 对约束的作用，也说明必须同时报告有效性切割。

## 24.5 Flavor 链的二维基准

取两个 SMEFT Wilson 系数、两个 LEFT 系数和两个低能观测量。设
$$
R_{\rm SMEFT}=
\begin{pmatrix}
1&0.1\\
0&0.9
\end{pmatrix},\quad
T=
\begin{pmatrix}
1&1\\
0&1
\end{pmatrix},\quad
R_{\rm LEFT}=
\begin{pmatrix}
0.8&0\\
0.2&1
\end{pmatrix},
$$
并设低能矩阵元响应
$$
N=
\begin{pmatrix}
1&0\\
0&2
\end{pmatrix}.
$$
总响应矩阵为
$$
M=NR_{\rm LEFT}TR_{\rm SMEFT}
=
\begin{pmatrix}
0.8&0.8\\
0.4&4.04
\end{pmatrix}.
$$

**解释 24.3.** 即使高尺度只开第二个 SMEFT 系数，运行和匹配也会使两个低能观测量同时响应。Flavor 结果若省略 $R_{\rm SMEFT}$、$T$ 或 $R_{\rm LEFT}$，就不是可复核的 SMEFT 结果。

## 24.6 数据替换规则

把本章基准矩阵替换为真实分析时，必须替换：

1.  $M$：由事件模拟、解析公式或工具链给出的线性响应；
2.  $Q$：若保留平方项，由二次响应给出；
3.  $\Sigma$：实验和理论协方差；
4.  bin 定义和有效性切割；
5.  Wilson 参数坐标；
6.  输入方案和尺度。

## 本章小结

本章完成了数值响应矩阵的教材闭包：所有矩阵都是可手算的基准对象，展示了输入方案传播、Higgs 退化、高能 tail 灵敏度、Fisher 误差和 flavor 匹配链。真实实验数值属于外部数据接口，而不是 EFT 教材定义本身。

## 练习

**练习 24.1.** 用 $s^2=0.23$ 重算 $M_{m_W}$。

**练习 24.2.** 求 $M_H$ 的秩和两个线性无关的零方向。

**练习 24.3.** 若 dilepton 第二个 bin 的误差改为 $0.5$，重算 $\sigma_c$。

