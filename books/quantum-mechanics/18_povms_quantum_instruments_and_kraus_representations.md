# 第十八章：POVM、量子仪器与 Kraus 表示

投影测量把每个结果对应到一个正交子空间，但真实装置可以有有限分辨率，
也可以借辅助系统实现本系统上并非投影的统计。此时一族 effect
$E_i$ 足以计算结果概率，却仍不能决定测后态：不同装置完全可能具有
同一 POVM，而把条件态送往不同方向。要同时描述“出现哪个结果”和
“结果出现后系统变成什么”，必须从 POVM 提升到量子仪器。

本章先定义有限结果 POVM，并用一个可调清晰度的量子比特测量计算概率。
单 Kraus 算子随后给出最简单的仪器实现，效果算子由 $M_i^*M_i$ 取得。
在定义完全正性后，再给出一般有限量子仪器的映射口径，并把任意信道的
Kraus/Stinespring 表示登记为外部输入。最后书内构造有限维 Naimark
扩张，具体证明广义概率如何来自更大空间中的投影测量。

## 18.1 POVM

**定义 18.1.** 有限结果集合上的 POVM 是一族正算子 $(E_i)_i$，满足
$$
E_i\ge0,\qquad \sum_iE_i=I.
$$
在态 $\rho$ 中得到结果 $i$ 的概率为
$$
p_i=\operatorname{tr}(\rho E_i).
$$

**命题 18.2.** POVM 概率非负且总和为 $1$。

**证明.** 因 $\rho,E_i\ge0$，有 $\operatorname{tr}(\rho E_i)\ge0$；可用 $\rho^{1/2}E_i\rho^{1/2}\ge0$ 的迹非负性证明。总和为
$$
\sum_ip_i=\operatorname{tr}\left(\rho\sum_iE_i\right)=\operatorname{tr}\rho=1.
$$
$\square$

**例子 18.2A（非锐量子比特测量）.** 对 $0\le\eta\le1$ 定义
$$
E_\pm=\frac12(I\pm\eta\sigma_z).
$$
$E_\pm$ 的本征值为 $(1\pm\eta)/2$ 与 $(1\mp\eta)/2$，故均为正，
且 $E_++E_-=I$。若
$\rho=(I+r\cdot\sigma)/2$，则
$$
p_\pm=\operatorname{tr}(\rho E_\pm)
=\frac12(1\pm\eta r_z).
$$
$\eta=1$ 时恢复锐利的 $\sigma_z$ 投影测量，$\eta=0$ 时结果是与输入态
无关的均匀随机数。中间值给出有限对比度，但这组 $E_\pm$ 仍没有指定
测后态。

要补上测后态，需要为每个结果给出一个未归一化输出映射。先从每个结果
只有一个 Kraus 算子的情形开始。

## 18.2 Kraus 表示

**定义 18.3（单 Kraus 仪器）.** 一族算子 $(M_i)_i$ 满足
$$
\sum_iM_i^*M_i=I
$$
时，定义结果 $i$ 的未归一化输出
$$
\mathcal I_i(\rho)=M_i\rho M_i^*.
$$
这族映射称为每个结果单 Kraus 的量子仪器。结果 $i$ 的概率为
$$
p_i=\operatorname{tr}(M_i\rho M_i^*),
$$
测后态为
$$
\rho_i=\frac{M_i\rho M_i^*}{p_i}
$$
若 $p_i>0$。

**命题 18.4.** Kraus 测量给出 POVM $E_i=M_i^*M_i$。

**证明.** 每个 $E_i$ 为正，因为
$$
\langle\psi,E_i\psi\rangle=\|M_i\psi\|^2\ge0.
$$
归一性由 $\sum_iM_i^*M_i=I$。概率满足
$$
\operatorname{tr}(M_i\rho M_i^*)=\operatorname{tr}(\rho M_i^*M_i)=\operatorname{tr}(\rho E_i).
$$
$\square$

同一效果算子可以由不同 $M_i$ 实现，例如在左侧再乘一个依赖结果的酉
算子并不改变 $M_i^*M_i$，却会改变条件态。这正是 POVM 与仪器信息量
不同的具体原因。一般仪器允许每个结果包含多个 Kraus 分支，需要先定义
完全正映射。

## 18.3 完全正映射

**定义 18.5.** 映射 $\Phi$ 称为完全正，若对每个 $n$，$\Phi\otimes\operatorname{id}_{M_n}$ 都保持正性。若还保持迹，称为量子信道。

**定义 18.5A（一般有限量子仪器）.** 有限结果量子仪器是一族完全正、
迹不增线性映射 $(\mathcal I_i)_i$，使
$\sum_i\mathcal I_i$ 保迹。对密度算子 $\rho$，
$$
p_i=\operatorname{tr}\mathcal I_i(\rho),\qquad
\rho_i=\frac{\mathcal I_i(\rho)}{p_i}\quad(p_i>0).
$$
其 POVM 由对偶映射给出 $E_i=\mathcal I_i^*(I)$，因为
$p_i=\operatorname{tr}(\rho E_i)$。定义 18.3 是
$\mathcal I_i(\rho)=M_i\rho M_i^*$ 的特例。

**外部输入定理 18.6（Kraus/Stinespring，QM-EXT-6）.** 有限维中，完全正保迹映射可写为
$$
\Phi(\rho)=\sum_\alpha M_\alpha\rho M_\alpha^*,
\qquad \sum_\alpha M_\alpha^*M_\alpha=I.
$$

Kraus 表示描述映射，Naimark 扩张则描述 POVM 的概率结构。下面的有限维
构造只需要每个 effect 的正平方根，并明确给出辅助空间和等距嵌入。

## 18.4 Naimark 扩张的有限维形式

**命题 18.7（有限维 Naimark 扩张）.** 设 $(E_i)_{i=1}^n$ 是有限维 Hilbert 空间 $\mathcal H$ 上的 POVM。存在辅助空间 $\mathbb C^n$、等距嵌入
$$
V:\mathcal H\to \mathcal H\otimes\mathbb C^n
$$
和辅助空间上的标准投影 $Q_i=I\otimes |i\rangle\langle i|$，使
$$
E_i=V^*Q_iV.
$$

**证明.** 令 $M_i=E_i^{1/2}$，定义
$$
V\psi=\sum_i M_i\psi\otimes |i\rangle.
$$
则
$$
\|V\psi\|^2=\sum_i\|M_i\psi\|^2
=\sum_i\langle\psi,E_i\psi\rangle
=\|\psi\|^2,
$$
故 $V$ 是等距嵌入。并且
$$
V^*Q_iV=M_i^*M_i=E_i.
$$
$\square$

**说明 18.8.** Naimark 扩张说明 POVM 可视为更大系统上的投影测量再压缩回原系统。这并不意味着 POVM 与原系统上的投影测量相同；辅助系统和嵌入是测量装置的一部分。

POVM、仪器与信道现在承担了不同层次的任务：$E_i$ 只决定结果概率，
$\mathcal I_i$ 还决定条件态，$\sum_i\mathcal I_i$ 则是忽略结果后的
信道。非锐量子比特例子说明概率本身可以连续偏离投影测量，Naimark
构造则把任意有限 POVM 实现为更大空间投影的压缩。下一章利用这些信道
和测量语言讨论量子比特几何、熵以及态可区分性。

## 练习

**练习 18.1.** 证明投影测量是 POVM 的特例。

**练习 18.2.** 若 $\Phi(\rho)=U\rho U^*$ 且 $U$ 酉，写出 Kraus 表示。

**练习 18.3.** 对例子 18.2A 的 $E_\pm$ 和
$\rho=\operatorname{diag}(p,1-p)$，计算 $p_\pm$，并分别解释
$\eta=0$ 与 $\eta=1$。
