# 第十八章：POVM、量子仪器与 Kraus 表示

## 本章目标

本章介绍广义测量：POVM、Kraus 算子、量子仪器和 Naimark/Stinespring 边界。

## 依赖前置知识

需要密度算子、正算子、迹和开放系统。

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

## 18.2 Kraus 表示

**定义 18.3.** 一族算子 $(M_i)_i$ 满足
$$
\sum_iM_i^*M_i=I
$$
时定义测量。结果 $i$ 的概率为
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

## 18.3 完全正映射

**定义 18.5.** 映射 $\Phi$ 称为完全正，若对每个 $n$，$\Phi\otimes\operatorname{id}_{M_n}$ 都保持正性。若还保持迹，称为量子信道。

**外部输入定理 18.6（Kraus/Stinespring，QM-EXT-6）.** 有限维中，完全正保迹映射可写为
$$
\Phi(\rho)=\sum_\alpha M_\alpha\rho M_\alpha^*,
\qquad \sum_\alpha M_\alpha^*M_\alpha=I.
$$

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

## 本章小结

POVM 描述测量结果概率，量子仪器同时描述结果和测后态。Kraus 表示把广义测量写成算子和，完全正性保证该演化在附加旁系统后仍保持物理正性。

## 练习

**练习 18.1.** 证明投影测量是 POVM 的特例。

**练习 18.2.** 若 $\Phi(\rho)=U\rho U^*$ 且 $U$ 酉，写出 Kraus 表示。
