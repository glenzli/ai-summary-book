# 第四章：量子公设、Born 规则与态更新

## 本章目标

本章以严格数学语言表述量子力学的基本公设，尤其是态、可观测量、Born 规则、投影测量和测后态。

## 依赖前置知识

需要前三章关于 Hilbert 空间、谱投影和谱测度的内容。

## 4.1 态与可观测量公设

**公设 4.1（纯态）.** 封闭量子系统的纯态由 Hilbert 空间 $\mathcal H$ 中的射线 $[\psi]$ 表示，其中 $\|\psi\|=1$。

**公设 4.2（可观测量）.** 实值可观测量由 $\mathcal H$ 上自伴算子 $A$ 表示。其可能测量结果位于谱 $\sigma(A)$ 中；在态 $\psi$ 中的结果分布为
$$
\mu^A_\psi(\Delta)=\langle\psi,E_A(\Delta)\psi\rangle.
$$

**命题 4.3.** Born 分布与态矢量的整体相位无关。

**证明.** 若 $\psi'=e^{i\theta}\psi$，则
$$
\langle\psi',E_A(\Delta)\psi'\rangle
=e^{-i\theta}e^{i\theta}\langle\psi,E_A(\Delta)\psi\rangle
=\langle\psi,E_A(\Delta)\psi\rangle.
$$
$\square$

## 4.2 投影测量与态更新

**定义 4.4.** 设离散可观测量
$$
A=\sum_r\lambda_rP_r
$$
带谱投影 $P_r$。若在态 $\psi$ 中测得 $\lambda_r$ 且 $\|P_r\psi\|\ne0$，Luders 态更新定义为
$$
\psi\longmapsto \frac{P_r\psi}{\|P_r\psi\|}.
$$

**命题 4.5.** 条件态 $P_r\psi/\|P_r\psi\|$ 已归一化。

**证明.** 由 $P_r^*=P_r$ 与 $P_r^2=P_r$，
$$
\left\|\frac{P_r\psi}{\|P_r\psi\|}\right\|^2
=\frac{\langle P_r\psi,P_r\psi\rangle}{\|P_r\psi\|^2}
=\frac{\langle\psi,P_r\psi\rangle}{\|P_r\psi\|^2}
=1.
$$
$\square$

## 4.3 相容可观测量

**定义 4.6.** 有限维自伴算子 $A,B$ 称为相容，若 $AB=BA$。

**命题 4.7.** 有限维中，若 $A$ 与 $B$ 为相容自伴算子，则存在共同正交归一特征基。

**证明.** 分解 $\mathcal H$ 为 $A$ 的特征子空间直和。由 $AB=BA$，若 $Av=\lambda v$，则
$$
A(Bv)=B(Av)=\lambda Bv,
$$
故每个 $A$-特征子空间被 $B$ 保持。在每个有限维特征子空间上限制 $B$，仍为自伴算子，故可正交对角化。合并各子空间中的正交基得到共同特征基。$\square$

## 4.4 简并测量与细化

**定义 4.8.** 若可观测量
$$
A=\sum_r\lambda_rP_r
$$
的某个谱投影 $P_r$ 秩大于 $1$，则称结果 $\lambda_r$ 简并。Luders 更新只把态投影到 $P_r\mathcal H$，不在该子空间内选择进一步基。

**例子 4.9.** 设 $\mathcal H=\mathbb C^3$，可观测量
$$
A=\operatorname{diag}(1,1,-1).
$$
结果 $1$ 的谱投影为
$$
P_+=\operatorname{diag}(1,1,0).
$$
若初态为
$$
\psi=(a,b,c)^T,\qquad |a|^2+|b|^2+|c|^2=1,
$$
则测得 $1$ 的概率为 $|a|^2+|b|^2$，测后态为
$$
\frac{(a,b,0)^T}{\sqrt{|a|^2+|b|^2}}.
$$
该测量没有区分前两个基向量。

**说明 4.10.** 若实验装置实际区分简并子空间内的某个正交基，则它测量的是 $A$ 的一个细化可观测量，而不是单纯的 $A$。因此“同一个谱值”与“同一个测量过程”不能混同；测量过程还包含仪器对简并空间的处理。

## 本章小结

量子公设把态、可观测量和概率联系起来。投影测量给出离散结果的概率和条件态。相容可观测量在有限维中可共同对角化，因此可同时赋予同一组特征值。

## 练习

**练习 4.1.** 对二能级系统 $A=\lambda_+P_++\lambda_-P_-$，写出 Born 概率和期望值。

**练习 4.2.** 证明若有限维自伴算子 $A,B$ 有共同正交特征基，则 $AB=BA$。
