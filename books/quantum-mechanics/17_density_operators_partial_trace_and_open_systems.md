# 第十七章：密度算子、偏迹与开放系统

## 本章目标

本章把态从纯态推广到混合态，介绍密度算子、约化态、开放系统和 Lindblad 形式的边界。

## 依赖前置知识

需要张量积、偏迹、投影和有限维矩阵迹。

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

**边界 17.7.** 连续时间 Markov 开放系统的 Lindblad 生成元定理作为外部输入定理 QM-EXT-18 处理。

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

## 本章小结

密度算子统一描述纯态、统计混合和子系统约化态。偏迹刻画局部可观测量的全部统计。开放系统演化来自更大封闭系统的酉演化后取偏迹。

## 练习

**练习 17.1.** 对 $\rho=p|0\rangle\langle0|+(1-p)|1\rangle\langle1|$ 计算 $\operatorname{tr}\rho^2$。

**练习 17.2.** 证明密度算子的凸组合仍是密度算子。
