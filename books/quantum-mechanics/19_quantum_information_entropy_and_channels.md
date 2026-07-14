# 第十九章：量子信息、熵与信道

量子态既是物理制备结果，也是信息载体，但“信息量”不能只靠向量坐标
计数。单量子比特的所有密度矩阵填满 Bloch 球：球面是纯态，内部点是
混合态。信道可以压缩这个球、抹去某些相干项，却必须在附加任意旁系统
后仍保持正性。比较两个输出态时，还需要知道它们能被一次最优测量区分到
什么程度，而不仅是矩阵元看起来相差多少。

本章先由 Pauli 代数推出 Bloch 球的正性条件，再用谱定义 von Neumann
熵并证明零熵恰好对应纯态。计算基中的退相干信道把非对角元显式消去，
展示一个幂等信道。最后以迹距离和保真度比较量子态，并在书内证明等先验
二元态区分的 Helstrom 公式；纯态迹距离公式则把这一操作性距离重新连回
第一章的转移概率。

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

熵只依赖密度算子的谱，而噪声信道还会改变本征向量与非对角相干。最
简单的例子是选定一组正交投影后，丢弃不同投影块之间的相位关系。

## 19.3 信道

**定义 19.6.** 有限维量子信道是完全正保迹线性映射
$$
\Phi:\mathcal T(\mathcal H)\to\mathcal T(\mathcal K).
$$

**例子 19.7.** 退相干信道
$$
\Phi(\rho)=\sum_jP_j\rho P_j
$$
抹去给定正交分解下的非对角相干项。对量子比特计算基投影
$P_0=|0\rangle\langle0|$、$P_1=|1\rangle\langle1|$，若
$$
\rho=\begin{pmatrix}a&c\\\overline c&1-a\end{pmatrix},
$$
则
$$
\Phi(\rho)=\begin{pmatrix}a&0\\0&1-a\end{pmatrix}.
$$
对角概率保持，而相干项 $c$ 被删除。

**命题 19.8.** 退相干信道幂等：$\Phi^2=\Phi$。

**证明.** 使用 $P_jP_k=\delta_{jk}P_j$：
$$
\Phi^2(\rho)=\sum_{j,k}P_jP_k\rho P_kP_j
=\sum_jP_j\rho P_j=\Phi(\rho).
$$
$\square$

信道输出之间的差别最终要由测量检验。迹范数对 Hermitian 差算子同时
记录正谱和负谱，因而自然控制二元判别的最佳成功率。

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

**命题 19.10A（等先验二元态区分）.** 设 $\rho,\sigma$ 以相同先验
概率 $1/2$ 出现。对两结果 POVM $(E,I-E)$，约定结果 $E$ 时猜
$\rho$，另一结果猜 $\sigma$。则最大成功概率为
$$
p_{\mathrm{succ}}^{\max}
=\frac12\bigl(1+D(\rho,\sigma)\bigr).
$$

**证明.** 令 $\Delta=\rho-\sigma$。给定 effect $0\le E\le I$ 时，
$$
\begin{aligned}
p_{\mathrm{succ}}(E)
&=\frac12\operatorname{tr}(E\rho)
+\frac12\operatorname{tr}((I-E)\sigma)\\
&=\frac12+\frac12\operatorname{tr}(E\Delta).
\end{aligned}
$$
写 Jordan 分解 $\Delta=\Delta_+-\Delta_-$，其中
$\Delta_\pm\ge0$ 且支撑正交。因为 $0\le E\le I$，
$$
\operatorname{tr}(E\Delta)
\le\operatorname{tr}(E\Delta_+)
\le\operatorname{tr}\Delta_+.
$$
取 $E$ 为 $\Delta$ 正谱子空间的投影时达到等号。又
$\operatorname{tr}\Delta=0$，所以
$$
\operatorname{tr}\Delta_+
=\frac12\|\Delta\|_1
=D(\rho,\sigma).
$$
代回即得结论。$\square$

**定义 19.11.** 纯态保真度为
$$
F(\psi,\phi)=|\langle\psi,\phi\rangle|^2.
$$
文献中同时使用“平方保真度”和其平方根两种命名规范；本书固定采用平方保真度
$$
F(\rho,\sigma)=\left(\operatorname{tr}\sqrt{\sqrt\rho\,\sigma\sqrt\rho}\right)^2.
$$

**说明 19.12.** 迹距离描述最优测量区分概率，保真度描述态的重叠。二者在纯态情形满足命题 19.10 给出的直接关系。

Bloch 球把单量子比特正性化为 $|r|\le1$，von Neumann 熵把混合性化为
本征值函数，完全正保迹映射则给出允许的有限维动力学。退相干信道具体
删除非对角元，而 Helstrom 公式证明迹距离恰好控制等先验二元判别。
下一章重新回到连续位置表象：酉算子将由积分核表示，并通过 Trotter
离散化与形式路径积分发生联系。

## 练习

**练习 19.1.** 计算完全混合单量子比特态 $I/2$ 的熵。

**练习 19.2.** 证明酉信道保持 von Neumann 熵。

**练习 19.3.** 用命题 19.10A 求等先验的两个相同态与两个正交纯态的
最优区分成功概率。
