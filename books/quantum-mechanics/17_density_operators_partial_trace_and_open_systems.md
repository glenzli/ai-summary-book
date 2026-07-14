# 第十七章：密度算子、偏迹与开放系统

同一个实验制备过程可能以概率 $p_j$ 产生不同纯态，也可能是一个更大
纠缠纯态的局部部分；只靠单个射线无法统一描述这两种情况。密度算子把
它们都编码为正、迹为一的算子，并让任意局部可观测量的期望写成
$\operatorname{tr}(\rho A)$。这种统一也带来必须保留的区分：同一个
密度算子可以有许多系综分解，物理预测由算子本身而不是某个分解清单
决定。

本章先用谱分解证明有限维纯态恰好是纯度
$\operatorname{tr}\rho^2=1$ 的秩一投影，再以偏迹刻画子系统全部局部
统计。开放系统演化由系统与固定环境初态的乘积经过整体酉演化后取偏迹
得到；一个受控非门例子会把系统相干性怎样转成系统--环境纠缠逐项算出。
最后构造任意密度算子的纯化，并把 Lindblad 与 Uhlmann 型结构定理留在
清楚标记的外部边界。

## 17.1 密度算子

**定义 17.1.** 有限维 Hilbert 空间上的密度算子是满足
$$
\rho\ge0,\qquad \operatorname{tr}\rho=1
$$
的算子。若 $\rho=|\psi\rangle\langle\psi|$，称为纯态；否则称为混合态。

**命题 17.2.** 密度算子 $\rho$ 为纯态当且仅当 $\operatorname{tr}(\rho^2)=1$。

**证明.** 谱分解给出
$$
\rho=\sum_jp_j|e_j\rangle\langle e_j|,\qquad p_j\ge0,\quad \sum_jp_j=1.
$$
于是
$$
\operatorname{tr}(\rho^2)=\sum_jp_j^2\le \left(\sum_jp_j\right)^2=1.
$$
等号成立当且仅当最多一个 $p_j$ 非零；因和为 $1$，即某个 $p_j=1$，其余为 $0$，这正是秩一投影。$\square$

谱分解把混合性变成一组经典概率本征值，但子系统态往往不是由人为随机
制备产生，而是从复合态忽略另一个因子得到。偏迹正好保留所有局部期望。

## 17.2 期望值与约化态

**定义 17.3.** 在密度态 $\rho$ 中，可观测量 $A$ 的期望值定义为
$$
\langle A\rangle_\rho=\operatorname{tr}(\rho A).
$$

**命题 17.4.** 若复合系统态为 $\rho_{AB}$，则子系统 $A$ 的约化态
$$
\rho_A=\operatorname{tr}_B\rho_{AB}
$$
满足
$$
\operatorname{tr}_A(\rho_A X)=\operatorname{tr}_{AB}(\rho_{AB}(X\otimes I)).
$$

**证明.** 这正是偏迹的定义；它说明所有只作用在 $A$ 上的可观测量期望值由 $\rho_A$ 完全决定。$\square$

约化态描述一个子系统在某个固定时刻的统计。若系统与环境先从乘积态
出发，再作整体酉演化，逐时刻的系统输入输出关系便成为一个线性约化
映射。

## 17.3 开放系统

**定义 17.5.** 若系统 $S$ 与环境 $E$ 初态为 $\rho_S\otimes\rho_E$，总演化为酉 $U$，则系统约化演化为
$$
\Phi(\rho_S)=\operatorname{tr}_E\bigl(U(\rho_S\otimes\rho_E)U^*\bigr).
$$

**命题 17.6.** 映射 $\Phi$ 保迹并保持正性。

**证明.** 若 $\rho_S\ge0$、$\rho_E\ge0$，则张量积为正，酉共轭保持正性，偏迹保持正性，故 $\Phi(\rho_S)\ge0$。迹方面，
$$
\operatorname{tr}\Phi(\rho_S)=\operatorname{tr}(U(\rho_S\otimes\rho_E)U^*)
=\operatorname{tr}(\rho_S\otimes\rho_E)=\operatorname{tr}\rho_S.
$$
$\square$

**例子 17.6A（相干性流入环境）.** 令系统与环境都是量子比特，初态为
$$
|+\rangle_S\otimes|0\rangle_E
=\frac{|0,0\rangle+|1,0\rangle}{\sqrt2}.
$$
取以系统为控制位、环境为目标位的受控非门 $U$，则
$$
U(|+\rangle_S\otimes|0\rangle_E)
=\frac{|0,0\rangle+|1,1\rangle}{\sqrt2}.
$$
对环境取偏迹得到
$$
\rho_S'=\frac12|0\rangle\langle0|+\frac12|1\rangle\langle1|
=\frac I2.
$$
初始系统态的非对角项完全消失，但整体演化仍为酉，信息转移到了系统与
环境的相关性中。定义 17.5 的乘积初态假设很重要；若初始已有系统--环境
相关，一般不能在所有系统输入态上得到同一个完全正映射。

**边界 17.7.** 连续时间 Markov 开放系统的 Lindblad 生成元定理作为外部输入定理 QM-EXT-18 处理。

开放系统演化说明混合态可由忽略环境产生。反过来，每个有限维混合态都
能被实现为某个更大纯态的约化态，这就是纯化。

## 17.4 纯化

**定义 17.8.** 设 $\rho$ 是 $\mathcal H_A$ 上密度算子。若存在辅助 Hilbert 空间 $\mathcal H_B$ 和单位向量 $\Psi\in\mathcal H_A\otimes\mathcal H_B$，使
$$
\operatorname{tr}_B|\Psi\rangle\langle\Psi|=\rho,
$$
则称 $\Psi$ 为 $\rho$ 的纯化。

**命题 17.9.** 有限维中每个密度算子都有纯化。

**证明.** 取谱分解
$$
\rho=\sum_jp_j|e_j\rangle\langle e_j|.
$$
令 $\mathcal H_B$ 含正交归一族 $(f_j)$，定义
$$
\Psi=\sum_j\sqrt{p_j}\,e_j\otimes f_j.
$$
因 $\sum_jp_j=1$，$\Psi$ 已归一化。由 Schmidt 分解的偏迹公式，
$$
\operatorname{tr}_B|\Psi\rangle\langle\Psi|
=\sum_jp_j|e_j\rangle\langle e_j|=\rho.
$$
$\square$

**说明 17.10.** 纯化说明混合态可以看作更大封闭系统纯态的子系统态。不同纯化之间相差辅助系统上的酉自由度；该唯一性是 Uhlmann 定理的入口，本书将其作为外部输入定理 QM-EXT-19 的边界，不展开证明。

密度算子统一了随机制备与纠缠约化，却不把二者的制备历史视为额外可观测
数据。偏迹由局部期望唯一刻画；受控非门例子显示，局部退相干可以与整体
酉性并存。纯化又证明任意有限维密度算子都可嵌入更大的纯态。下一章将
同一系统的“输出态”按测量结果进一步分支，从而区分只给概率的 POVM
与同时给条件态的量子仪器。

## 练习

**练习 17.1.** 对 $\rho=p|0\rangle\langle0|+(1-p)|1\rangle\langle1|$ 计算 $\operatorname{tr}\rho^2$。

**练习 17.2.** 证明密度算子的凸组合仍是密度算子。
