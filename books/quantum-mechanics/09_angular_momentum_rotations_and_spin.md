# 第九章：角动量、旋转与自旋

空间旋转不是给三个坐标各乘一个相位。连续旋转在量子态上由酉算子实现，其无穷小生成元有三个分量，而且不同轴的旋转次序不同，因此生成元必须满足非交换代数。轨道角动量是这一结构的一种实现，自旋则没有经典位置轨道作为前提；自旋 $1/2$ 只需二维 Hilbert 空间和三只 Pauli 矩阵，就能给出沿任意方向的两结果测量。

本章从共同不变定义域上的 $\mathfrak{su}(2)$ 对易关系开始，导出 $J^2$、$J_z$ 与升降算符之间的代数。随后用显式 $2\times2$ 矩阵计算一个任意 Bloch 方向自旋态的测量概率，再说明两个系统的角动量为何在张量积上相加。最后由范数计算得到升降系数。这里使用的只是已建立的矩阵交换子、Born 规则与张量积；不可约表示的完整分类留到需要角动量耦合时作为精确外部输入。

## 9.1 角动量代数

**定义 9.1.** 三个自伴算子 $J_1,J_2,J_3$ 称为角动量算子，若在共同不变定义域上满足
$$
[J_i,J_j]=i\sum_k\epsilon_{ijk}J_k.
$$
定义
$$
J^2=J_1^2+J_2^2+J_3^2,\qquad J_\pm=J_1\pm iJ_2.
$$

**命题 9.2.** 有
$$
[J_3,J_\pm]=\pm J_\pm,\qquad [J^2,J_i]=0.
$$

**证明.** 第一式由
$$
[J_3,J_1]=iJ_2,\qquad [J_3,J_2]=-iJ_1
$$
直接相加得到。第二式以 $i=3$ 为例：
$$
[J_1^2,J_3]=J_1[J_1,J_3]+[J_1,J_3]J_1=-i(J_1J_2+J_2J_1),
$$
而
$$
[J_2^2,J_3]=J_2[J_2,J_3]+[J_2,J_3]J_2=i(J_2J_1+J_1J_2).
$$
两项相消，$[J_3^2,J_3]=0$。其余指标同理按循环置换得到。$\square$

$J^2$ 与任一分量对易，使我们能够同时标记总角动量大小和一个选定轴的
分量。最小的非平凡实现是二维自旋空间，所有关系都可用 Pauli 矩阵逐项
核算。

## 9.2 自旋 $1/2$

**定义 9.3.** Pauli 矩阵为
$$
\sigma_x=\begin{pmatrix}0&1\\1&0\end{pmatrix},\quad
\sigma_y=\begin{pmatrix}0&-i\\i&0\end{pmatrix},\quad
\sigma_z=\begin{pmatrix}1&0\\0&-1\end{pmatrix}.
$$
自旋 $1/2$ 算子为 $S_i=\frac12\sigma_i$。

**命题 9.4.** $S_i$ 满足角动量对易关系。

**证明.** 直接计算 Pauli 矩阵满足
$$
[\sigma_i,\sigma_j]=2i\sum_k\epsilon_{ijk}\sigma_k.
$$
因此
$$
[S_i,S_j]=\frac14[\sigma_i,\sigma_j]
=i\sum_k\epsilon_{ijk}S_k.
$$
$\square$

**例子 9.4A（任意方向的自旋 $1/2$ 态）.** 取
$$
|\psi(\theta,\varphi)\rangle
=\cos\frac\theta2|\uparrow\rangle
+e^{i\varphi}\sin\frac\theta2|\downarrow\rangle,
\qquad 0\le\theta\le\pi.
$$
测量 $S_z=\sigma_z/2$ 时，
$$
\Pr(S_z=1/2)=\cos^2\frac\theta2,\qquad
\Pr(S_z=-1/2)=\sin^2\frac\theta2.
$$
直接乘矩阵还得到
$$
\langle\sigma_x\rangle=\sin\theta\cos\varphi,\qquad
\langle\sigma_y\rangle=\sin\theta\sin\varphi,\qquad
\langle\sigma_z\rangle=\cos\theta.
$$
因此 $(\theta,\varphi)$ 确实指定了 Bloch 球上的测量方向，而整体相位
没有进入任何期望值。

单个自旋给出二维表示。两个带角动量的系统组成复合系统时，旋转必须
同时作用在两个张量因子上，其生成元因而是两个局部生成元之和。

## 9.3 总角动量

**定义 9.5.** 两个系统角动量分别为 $J^{(1)}$ 与 $J^{(2)}$ 时，总角动量定义为
$$
J_i=J_i^{(1)}\otimes I+I\otimes J_i^{(2)}.
$$

**命题 9.6.** 总角动量仍满足角动量对易关系。

**证明.** 因不同张量因子上的算子交换，
$$
[J_i,J_j]=[J_i^{(1)},J_j^{(1)}]\otimes I+I\otimes[J_i^{(2)},J_j^{(2)}]
=i\sum_k\epsilon_{ijk}J_k.
$$
$\square$

总角动量仍满足同一代数后，可以沿用共同本征向量与升降算符。升降
系数不是额外经验规则，而由 $J_\mp J_\pm$ 的范数恒等式唯一确定。

## 9.4 升降系数

**命题 9.7.** 若
$$
J^2|j,m\rangle=j(j+1)|j,m\rangle,\qquad
J_z|j,m\rangle=m|j,m\rangle,
$$
并且 $|j,m\rangle$ 已归一化，则
$$
J_\pm|j,m\rangle
=\sqrt{j(j+1)-m(m\pm1)}\,|j,m\pm1\rangle
$$
至多差一个相位。按标准相位约定取上式正根。

**证明.** 由 $[J_z,J_\pm]=\pm J_\pm$，$J_\pm|j,m\rangle$ 若非零，则是 $J_z$ 本征值 $m\pm1$ 的向量。又
$$
J_\mp J_\pm=J^2-J_z^2\mp J_z.
$$
取范数：
$$
\|J_\pm|j,m\rangle\|^2
=\langle j,m|J_\mp J_\pm|j,m\rangle
=j(j+1)-m(m\pm1).
$$
归一化即得系数。$\square$

**说明 9.8.** 条件 $j(j+1)-m(m\pm1)\ge0$ 强迫 $m$ 在有限链中终止，这正是 $m=-j,-j+1,\dots,j$ 的代数来源。

角动量的可计算内容由三项结构共同决定：非交换生成元给出
$\mathfrak{su}(2)$ 代数，$J^2,J_z$ 提供共同量子数，$J_\pm$ 的范数
固定相邻磁量子数之间的系数。自旋 $1/2$ 把这些结构落实为 Pauli
矩阵，而张量积上的生成元和为后面的角动量耦合奠定基础。下一章先从
更一般的问题出发：哪些保持转移概率的变换应被称作量子对称性。

## 练习

**练习 9.1.** 计算 $\sigma_x\sigma_y$ 与 $\sigma_y\sigma_x$。

**练习 9.2.** 证明 $S^2=\frac34I$ 对自旋 $1/2$ 成立。
