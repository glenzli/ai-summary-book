# 第十九章：量子信息、熵与信道

## 本章目标

本章介绍量子比特、von Neumann 熵、保真度的基本口径、量子信道和若干标准例子。

## 依赖前置知识

需要密度算子、POVM、Kraus 表示和张量积。

## 19.1 量子比特

**定义 19.1.** 量子比特是 Hilbert 空间 $\mathbb C^2$ 中的量子系统。任意纯态可写为
$$
\psi=\alpha|0\rangle+\beta|1\rangle,\qquad |\alpha|^2+|\beta|^2=1.
$$

**定义 19.2.** 单量子比特密度矩阵可写为 Bloch 形式
$$
\rho=\frac12(I+r_x\sigma_x+r_y\sigma_y+r_z\sigma_z),
$$
其中 $r\in\mathbb R^3$ 且 $|r|\le1$。

**命题 19.3.** Bloch 形式中的 $\rho$ 为密度矩阵当且仅当 $|r|\le1$。

**证明.** Pauli 矩阵满足 $(r\cdot\sigma)^2=|r|^2I$，故 $r\cdot\sigma$ 的本征值为 $\pm |r|$。于是 $\rho$ 的本征值为
$$
\frac12(1\pm |r|).
$$
正性等价于这两个数非负，即 $|r|\le1$；迹恒为 $1$。$\square$

## 19.2 熵

**定义 19.4.** 密度算子 $\rho$ 的 von Neumann 熵为
$$
S(\rho)=-\operatorname{tr}(\rho\log\rho),
$$
其中 $0\log0$ 按 $0$ 处理。

**命题 19.5.** 有限维中，$S(\rho)=0$ 当且仅当 $\rho$ 为纯态。

**证明.** 令 $\rho$ 的本征值为 $p_j$，则
$$
S(\rho)=-\sum_jp_j\log p_j.
$$
每项非负，和为零当且仅当每个非零 $p_j$ 等于 $1$。因 $\sum_jp_j=1$，这等价于谱为 $(1,0,\dots,0)$，即纯态。$\square$

## 19.3 信道

**定义 19.6.** 有限维量子信道是完全正保迹线性映射
$$
\Phi:\mathcal T(\mathcal H)\to\mathcal T(\mathcal K).
$$

**例子 19.7.** 退相干信道
$$
\Phi(\rho)=\sum_jP_j\rho P_j
$$
抹去给定正交分解下的非对角相干项。

**命题 19.8.** 退相干信道幂等：$\Phi^2=\Phi$。

**证明.** 使用 $P_jP_k=\delta_{jk}P_j$：
$$
\Phi^2(\rho)=\sum_{j,k}P_jP_k\rho P_kP_j
=\sum_jP_j\rho P_j=\Phi(\rho).
$$
$\square$

## 19.4 迹距离与态区分

**定义 19.9.** 两个密度算子 $\rho,\sigma$ 的迹距离定义为
$$
D(\rho,\sigma)=\frac12\|\rho-\sigma\|_1,
$$
其中 $\|T\|_1=\operatorname{tr}\sqrt{T^*T}$。

**命题 19.10.** 若 $\rho=|\psi\rangle\langle\psi|$、$\sigma=|\phi\rangle\langle\phi|$ 为纯态，则
$$
D(\rho,\sigma)=\sqrt{1-|\langle\psi,\phi\rangle|^2}.
$$

**证明.** 迹距离只依赖 $\psi,\phi$ 张成的二维子空间。取基使
$$
\psi=(1,0),\qquad \phi=(c,\sqrt{1-|c|^2})
$$
且 $c=\langle\psi,\phi\rangle$ 可取非负实数。矩阵 $\rho-\sigma$ 在该二维空间上的迹为 $0$，行列式为
$$
-\bigl(1-c^2\bigr).
$$
故其本征值为 $\pm\sqrt{1-c^2}$。迹范数是本征值绝对值之和，得到
$$
D(\rho,\sigma)=\sqrt{1-c^2}.
$$
$\square$

**定义 19.11.** 纯态保真度为
$$
F(\psi,\phi)=|\langle\psi,\phi\rangle|^2.
$$
混合态保真度有多种等价规范；本书在需要时采用
$$
F(\rho,\sigma)=\left(\operatorname{tr}\sqrt{\sqrt\rho\,\sigma\sqrt\rho}\right)^2.
$$

**说明 19.12.** 迹距离描述最优测量区分概率，保真度描述态的重叠。二者在纯态情形满足命题 19.10 给出的直接关系。

## 本章小结

量子信息把态视为信息资源。量子比特的态空间是 Bloch 球；von Neumann 熵度量混合性；量子信道是完全正保迹映射，表示最一般的物理演化。

## 练习

**练习 19.1.** 计算完全混合单量子比特态 $I/2$ 的熵。

**练习 19.2.** 证明酉信道保持 von Neumann 熵。
