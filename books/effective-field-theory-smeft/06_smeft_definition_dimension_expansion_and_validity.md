# 第六章：SMEFT 的定义、维数展开与适用范围

## 本章目标

本章给出 SMEFT 的正式定义、维数展开和有效性条件。

## 依赖前置知识

需要前五章的 EFT 定义、算符基和标准模型场内容。

## 6.1 定义

**定义 6.1（SMEFT）.** 标准模型有效场论是满足以下条件的 EFT：

1.  低能自由度为标准模型场；
2.  规范群为 $G_{\mathrm{SM}}$；
3.  Higgs 场 $H$ 是线性实现的 $SU(2)_L$ 双重态；
4.  拉氏量包含所有 $G_{\mathrm{SM}}$ 不变局域算符；
5.  算符按质量维数和 $1/\Lambda$ 展开。

因此
$$
\mathcal L_{\mathrm{SMEFT}}
=
\mathcal L_{\mathrm{SM}}
+
\sum_{d>4}\sum_i
\frac{C_i^{(d)}}{\Lambda^{d-4}}\mathcal O_i^{(d)}.
$$

## 6.2 维数展开

**命题 6.2（维数五的唯一类型）.** 在标准模型场内容下，最低维的非重整化算符为维数五 Weinberg 算符
$$
\mathcal O_5
=
(\ell^T C \ell)HH
$$
的规范不变收缩形式，它违反 lepton number 并在电弱破缺后产生 Majorana neutrino mass。

**证明说明.** 由质量维数 $[\ell]=3/2$、$[H]=1$ 得 $2[\ell]+2[H]=5$。要求 $SU(2)_L$ 和 $U(1)_Y$ 不变，两个 lepton 双重态和两个 Higgs 双重态可收缩为 singlet。完整 flavor 和 $SU(2)$ 指标符号见标准 SMEFT 文献。$\square$

**定义 6.3（维数六 SMEFT 截断）.** 若只保留到 $1/\Lambda^2$，则
$$
\mathcal L_{\mathrm{SMEFT}}^{(6)}
=
\mathcal L_{\mathrm{SM}}
+
\frac{C_5}{\Lambda}\mathcal O_5
+
\sum_i\frac{C_i^{(6)}}{\Lambda^2}\mathcal O_i^{(6)}.
$$
若假设 lepton number 守恒，可令 $C_5=0$。

## 6.3 SMEFT、HEFT 与 LEFT

| 理论 | 自由度 | 对称性实现 | 典型适用区间 |
| --- | --- | --- | --- |
| SMEFT | SM 场，含 Higgs 双重态 | $SU(2)_L\times U(1)_Y$ 线性实现 | 新物理高于电弱尺度且 decoupling |
| HEFT | Higgs singlet-like 标量与 Goldstone 非线性实现 | 电弱对称性非线性实现 | 强电弱破缺或非双重态 Higgs 情形 |
| LEFT | $W,Z,h,t$ 已积掉的低能场 | $SU(3)_c\times U(1)_{\rm em}$ | $\mu<m_W$ |

**原则 6.4（理论选择）.** 若低能数据低于电弱尺度，不能直接用 SMEFT Wilson 系数写低能振幅；必须先在电弱尺度匹配到 LEFT。若 Higgs 不属于线性双重态，SMEFT 不是正确主线，应使用 HEFT。

## 6.4 适用性

**原则 6.5（SMEFT 有效性条件）.** 对过程能标 $E$，SMEFT 截断可信通常要求
$$
E<\Lambda
$$
且 Wilson 展开收敛。实验分析中不能只报告 Wilson 系数，还应说明使用的数据能区和截断阶数。

**警告 6.6（维数六平方项）.** 若观测量写为
$$
\sigma
=
\sigma_{\mathrm{SM}}
+
\frac{1}{\Lambda^2}\sigma_{\mathrm{int}}
+
\frac{1}{\Lambda^4}\sigma_{\mathrm{quad}}
+\cdots,
$$
则 $\sigma_{\mathrm{quad}}$ 与维数八干涉项同阶。保留或丢弃维数六平方项是带有理论假设的截断方案。

## 6.5 可计算性条件

一个 SMEFT 问题在教材意义上可计算，至少需给出：

1.  过程的外态和能区；
2.  所用算符基；
3.  Wilson 系数定义尺度；
4.  flavor 与 CP 假设；
5.  输入参数方案；
6.  截断阶数；
7.  是否包含 RGE；
8.  理论误差估计。

**例 6.7（不完整陈述）.** “限制 $C_{HWB}$”不是完整物理命题。完整说法必须包含：在 Warsaw basis、某定义尺度、某输入方案、某数据集合、某 flavor/CP 口径和某截断规则下，限制 $C_{HWB}/\Lambda^2$ 的某个置信区间。

## 本章小结

SMEFT 是标准模型场内容和规范对称性下的系统高维算符展开。它的适用性不只由 Wilson 系数大小决定，还由能区、截断和 UV 假设决定。

## 练习

**练习 6.1.** 验证 Weinberg 算符的质量维数为五。

**练习 6.2.** 解释为什么 $E>\Lambda$ 的 LHC bin 不能直接用截断 SMEFT 无条件解释。

**练习 6.3.** 判断下列情形应优先使用 SMEFT、HEFT 还是 LEFT：低能核 beta decay、强耦合电弱破缺、高能 Higgs pair production。
