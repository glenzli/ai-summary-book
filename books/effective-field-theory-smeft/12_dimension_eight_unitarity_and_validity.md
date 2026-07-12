# 第十二章：维数八、unitarity 与有效性边界

## 本章目标

本章固定 SMEFT 截断的边界问题：维数八算符、unitarity、positivity 和高能数据的有效性判断。

## 依赖前置知识

需要第三章的 $(p,L)$ 双重分次、第六章的截断和第八章的观测量展开。

## 12.1 维数八为什么重要

**事实 12.1（同阶问题与口径）.** 沿用定义 3.2A 的逆尺度次数 $p$，先固定所报告的圈阶，并假设精确选择定则排除奇数 $p$。在选定输入方案下重新展开参数后，写
$$
A=A_0+\Lambda_{\rm ref}^{-2}A_2+\Lambda_{\rm ref}^{-4}A_4+
O(\Lambda_{\rm ref}^{-6}).
$$
若 flux、cuts 和 phase-space 测度固定，则
$$
\begin{aligned}
\sigma
&=\langle A,A\rangle_0\\
&=\langle A_0,A_0\rangle_0
+{2\over\Lambda_{\rm ref}^2}\operatorname{Re}\langle A_0,A_2\rangle_0\\
&\quad+{1\over\Lambda_{\rm ref}^4}
\left(
\langle A_2,A_2\rangle_0
+2\operatorname{Re}\langle A_0,A_4\rangle_0
\right)
+O(\Lambda_{\rm ref}^{-6}).
\end{aligned}
$$
下标表示总逆尺度次数，不表示单个算符维数。只看高维顶点插入部分，有
$$
\begin{aligned}
A_2^{\rm ins}
&=A_2^{[6]}+A_2^{[5,5]},\\
A_4^{\rm ins}
&=A_4^{[8]}+A_4^{[6,6]}+A_4^{[5,7]}
+A_4^{[5,5,6]}+A_4^{[5,5,5,5]},
\end{aligned}
$$
其中方括号列出图中高维顶点的 canonical dimensions；不允许某类顶点时删除相应项。因此“维数六平方项” $\langle A_2^{[6]},A_2^{[6]}\rangle_0$ 与“两次维数六插入振幅” $A_4^{[6,6]}$ 是不同对象，后者还要与 $A_0$ 干涉。若 $C_5$ 与 $C_7$ 不因对称性或模型假设而消失，$A_4^{[5,7]}$ 也在同一 $p=4$ 阶；若奇数 $p$ 本身未被排除，还须加入 $A_1/\Lambda_{\rm ref}$、$A_3/\Lambda_{\rm ref}^3$，且固定测度的 $p=4$ 系数另含 $2\operatorname{Re}\langle A_1,A_3\rangle_0$。

输入参数和测度也必须按同一口径展开。具体地，把
$$
\theta=\theta_0+{\delta\theta_2\over\Lambda_{\rm ref}^2}
+{\delta\theta_4\over\Lambda_{\rm ref}^4}+\cdots
$$
代入振幅后，所产生的参数导数项分别计入 $A_2$ 和 $A_4$。若包含 flux、cuts 或 phase space 的 sesquilinear form 也依赖 EFT 参数，并写成
$$
\langle-,-\rangle
=\langle-,-\rangle_0
+\Lambda_{\rm ref}^{-2}\langle-,-\rangle_2
+\Lambda_{\rm ref}^{-4}\langle-,-\rangle_4+\cdots,
$$
则 $1/\Lambda_{\rm ref}^4$ 的系数还包含
$$
2\operatorname{Re}\langle A_0,A_2\rangle_2
+\langle A_0,A_0\rangle_4.
$$

**证明（书内推导）.** 由 $p=\sum_v n_v(d_v-4)$，$p=2$ 与 $p=4$ 的整数分拆分别给出上列插入结构。把振幅、输入参数和 sesquilinear form 的展开代入 $\sigma=\langle A,A\rangle$，按 $\Lambda_{\rm ref}^{-1}$ 收集即得。$\square$

**外部输入 12.2（维数八基）.** 完整 dimension-eight SMEFT operator basis 已有系统分类，但其规模远大于 dimension-six。本书将其作为研究边界，不在第一版逐项展开。

## 12.2 Perturbative unitarity

**定义 12.3（partial-wave unitarity 估计）.** 对 $2\to2$ 振幅，若某 partial wave $a_\ell(s)$ 满足
$$
|a_\ell(s)|\le 1,
$$
则称该能区未明显违反微扰 unitarity。实际分析常用更强的 $|\mathrm{Re}\,a_\ell|\le 1/2$ 作为保守条件。

**解释 12.4.** 高维算符振幅常随能量增长，如 $A_6\sim C E^2/\Lambda^2$。当该增长使 partial wave 接近 unitarity 界时，截断 EFT 已不能单独可信。

**例 12.5（常数角分布估计）.** 若某 $2\to2$ 振幅近似为
$$
{\cal A}(s,\cos\theta)=C{s\over\Lambda^2},
$$
则
$$
a_0(s)={1\over32\pi}\int_{-1}^{1}d\cos\theta\,{\cal A}
={C s\over16\pi\Lambda^2}.
$$
保守条件 $|{\rm Re}\,a_0|\le1/2$ 给出
$$
s\le {8\pi\Lambda^2\over |C|}.
$$
这不是新物理质量的精确界，而是 EFT 截断的自洽性警告。

## 12.3 Positivity 边界

**外部输入 12.6（positivity bounds）.** 在满足 Lorentz invariance、unitarity、analyticity、crossing symmetry 和适当 UV 行为的理论中，低能 Wilson 系数可受 positivity bounds 约束。

**使用边界.** Positivity 约束通常对某些 dimension-eight 组合更直接。将其用于 LHC 数据解释时，必须说明假设条件和过程能区。

**形式例 12.7（前向极限）.** 若弹性振幅在前向极限可写为
$$
{\cal A}(s,0)=a_0+a_1s+a_2s^2+\cdots,
$$
且满足适当解析性和 UV 有界性，则 dispersion relation 常推出 $a_2>0$ 类型的约束。EFT 中 $a_2$ 往往对应维数八 Wilson 系数组合。

## 12.4 有效性报告标准

**规则 12.8（发布 SMEFT 限制的最小信息）.** 一个 SMEFT 限制应报告：

1.  算符基；
2.  flavor 假设；
3.  截断阶数；
4.  是否包含维数六平方项；
5.  数据能区或最大不变量质量；
6.  输入参数方案；
7.  理论误差处理；
8.  是否使用 RG running。

## 12.5 截断方案的三种口径

| 口径 | 保留项 | 优点 | 风险 |
| --- | --- | --- | --- |
| 线性维数六 | 在 $C_5=0$ 或相应选择定则下，保留 SM 与一次 $d=6$ 插入的干涉 | EFT 阶数清楚 | 干涉抑制时无灵敏度 |
| 维数六平方 | 再加一次 $d=6$ 插入振幅的平方 | 数值上常更稳定 | 只是 $p=4$ 的一部分 |
| 固定假设下到 $p=4$ 一致 | 保留事实 12.1 的全部 $p\le4$ 振幅分拆、干涉及同阶输入参数和测度展开 | 逆尺度阶一致 | 需声明选择定则、圈阶、dimension-eight sector、多次插入和输入方案 |

**原则 12.9.** 若只在维数六线性结果上加入平方项，应把结果标为“dimension-six squared included”，不得称为“完整到 $1/\Lambda_{\rm ref}^4$”。只有在给定自由度、对称性、保留的 $(p,L)$ 集合、输入方案和 cuts 后，事实 12.1 中所有允许的同阶振幅、参数与测度项均已纳入，才可称为“在所声明假设下完整到 $p=4$”。

## 本章小结

SMEFT 的高能敏感性是优势也是风险。维数八、unitarity 和 positivity 不是附属细节，而是判断 EFT 解释是否可信的边界条件。

## 练习

**练习 12.1.** 在固定测度且奇数 $p$ 被排除时，对 $A=A_0+\Lambda_{\rm ref}^{-2}A_2+\Lambda_{\rm ref}^{-4}A_4$ 展开 $\langle A,A\rangle$ 到 $p=4$，并区分维数六平方项、两次维数六插入振幅和一次维数八插入振幅。

**练习 12.2.** 若某算符给出振幅 $A=C s/\Lambda^2$，估计何时可能接近 partial-wave unitarity 边界。

**练习 12.3.** 解释 positivity bound 为什么通常更自然地约束维数八而不是维数六。
