# 第十四章：散射理论、Born 近似与截面

## 本章目标

本章介绍散射理论的基本对象：自由与相互作用 Hamiltonian、波算子、$S$ 矩阵、散射振幅和 Born 近似。

## 依赖前置知识

需要时间演化、Fourier 变换、一维势垒和含时扰动理论。

## 14.1 波算子

**定义 14.1.** 设 $H_0,H$ 是同一 Hilbert 空间 $\mathcal H$ 上的
自伴算子。记 $\mathcal H_{\mathrm{ac}}(H_0)$ 为 $H_0$ 的绝对连续
谱子空间，$P_{\mathrm{ac}}(H_0)$ 为到该子空间的正交投影。若强极限
存在，定义 Møller 波算子
$$
\Omega_\pm
=\operatorname{s-lim}_{t\to\pm\infty}
e^{itH}e^{-itH_0}P_{\mathrm{ac}}(H_0).
$$
该极限作为 $\mathcal H$ 上的算子理解，在
$\mathcal H_{\mathrm{ac}}(H_0)^\perp$ 上为零。若进一步
$$
\operatorname{Ran}\Omega_\pm=\mathcal H_{\mathrm{ac}}(H),
$$
则称相应波算子完备（本书也沿用物理文献中的“渐近完备”一词）。
这里必须使用绝对连续谱子空间；仅由“谱值不是本征值”定义的集合并不能
替代它。

**命题 14.1A（波算子的初始空间与值域）.** 定义 14.1 的强极限存在时
$$
\Omega_\pm^*\Omega_\pm=P_{\mathrm{ac}}(H_0),
$$
且
$$
\Omega_\pm e^{-isH_0}=e^{-isH}\Omega_\pm,
\qquad s\in\mathbb R.
$$
因此 $\Omega_\pm$ 在 $\mathcal H_{\mathrm{ac}}(H_0)$ 上等距，并且
$\operatorname{Ran}\Omega_\pm\subseteq\mathcal H_{\mathrm{ac}}(H)$。

**证明.** 对任意 $\psi\in\mathcal H$，极限中每个算子都保持
$P_{\mathrm{ac}}(H_0)\psi$ 的范数，故
$\|\Omega_\pm\psi\|=\|P_{\mathrm{ac}}(H_0)\psi\|$；极化恒等式给出
第一式。把极限参数平移 $s$ 得到酉群交织式。由谱定理，该式蕴含
$$
\Omega_\pm E_{H_0}(\Delta)=E_H(\Delta)\Omega_\pm
$$
对每个 Borel 集 $\Delta$ 成立。若
$\psi\in\mathcal H_{\mathrm{ac}}(H_0)$，再用
$\Omega_\pm^*\Omega_\pm\psi=\psi$ 可知，
$\Omega_\pm\psi$ 关于 $H$ 的谱测度等于 $\psi$ 关于 $H_0$ 的谱测度。
后者对 Lebesgue 测度绝对连续，从而值域包含关系成立。$\square$

**外部输入定理 14.2（一个可判定的短程势版本，QM-EXT-8）.** 取
$\mathcal H=L^2(\mathbb R^3)$、$m>0$，并令
$$
H_0=-\frac{\Delta}{2m},\qquad \mathcal D(H_0)=H^2(\mathbb R^3).
$$
若 $V\in L^\infty(\mathbb R^3;\mathbb R)$，且存在
$C,\varepsilon>0$ 使
$$
|V(x)|\le C\langle x\rangle^{-1-\varepsilon},
\qquad \langle x\rangle=(1+|x|^2)^{1/2},
$$
则 $H=H_0+V$ 在 $H^2(\mathbb R^3)$ 上自伴，定义 14.1 的
$\Omega_\pm$ 存在并渐近完备：
$$
\operatorname{Ran}\Omega_\pm=\mathcal H_{\mathrm{ac}}(H).
$$
自由 Laplacian 只有绝对连续谱，故本定理中
$P_{\mathrm{ac}}(H_0)=I$。证明需要短程散射的传播估计或平稳方法，
不在本书内部重证；束缚态属于
$\mathcal H_{\mathrm{ac}}(H)^\perp$，不会出现在波算子的值域中。
来源定位为 Teschl, *Mathematical Methods in Quantum Mechanics*,
2nd ed., §12.3 的短程定义及 Theorems 12.11--12.12。事实上上述点态
衰减给出
$$
\|\mathbf 1_{\{|x|\ge r\}}V(H_0-z)^{-1}\|
\le C_z\langle r\rangle^{-1-\varepsilon},
\qquad z\notin[0,\infty),
$$
所以该书短程条件中的径向范数函数可积；这说明本定理的假设可直接
检查，而不是未展开的“适当短程条件”。

**定义 14.3.** 若两个波算子存在，散射算子定义为
$$
S=\Omega_+^*\Omega_-:
\mathcal H_{\mathrm{ac}}(H_0)\longrightarrow
\mathcal H_{\mathrm{ac}}(H_0).
$$

**命题 14.4.** 若 $\Omega_\pm$ 都渐近完备，则 $S$ 是
$\mathcal H_{\mathrm{ac}}(H_0)$ 上的酉算子。

**证明.** 在 $\mathcal H_{\mathrm{ac}}(H_0)$ 上有
$\Omega_\pm^*\Omega_\pm=I$。渐近完备性又给出
$\Omega_\pm\Omega_\pm^*=P_{\mathrm{ac}}(H)$。因此
$$
S^*S
=\Omega_-^*\Omega_+\Omega_+^*\Omega_-
=\Omega_-^*P_{\mathrm{ac}}(H)\Omega_-=I,
$$
因为 $\operatorname{Ran}\Omega_-\subseteq\mathcal H_{\mathrm{ac}}(H)$。
交换 $+$ 与 $-$ 同理得到 $SS^*=I$。$\square$

## 14.2 Born 近似

**推导 14.5（第一 Born 振幅）.** 设 $m>0$。在三维中取
$$
H_0=-\frac{\Delta}{2m},
\qquad E=\frac{k^2}{2m},
$$
并采用散射态渐近归一化
$$
\psi_{\mathbf k}^{(+)}(x)
=e^{i\mathbf k\cdot x}
+f(\mathbf k',\mathbf k)\frac{e^{ikr}}{r}+o(r^{-1}),
\qquad
\mathbf k'=k\frac{x}{r}.
$$
若 $\int_{\mathbb R^3}(1+|x|)|V(x)|\,dx<\infty$，所选能量不在阈值或共振等使出射 Lippmann--Schwinger 方程失去唯一性的例外集合中，且以自由入射波替换积分中的精确散射态是一致的一阶近似，则
$$
f(\mathbf k',\mathbf k)
=-\frac{m}{2\pi}\int_{\mathbb R^3}e^{-i(\mathbf k'-\mathbf k)\cdot x}V(x)\,dx.
$$
这里采用 $\hbar=1$。

**推导.** 定态 Schrodinger 方程等价于
$$
(\Delta+k^2)\psi=2mV\psi.
$$
出射 Green 函数 $G_k^+(x)=e^{ik|x|}/(4\pi|x|)$ 满足 $(\Delta+k^2)G_k^+=-\delta$，故
$$
\psi(x)=e^{i\mathbf k\cdot x}
-2m\int G_k^+(x-y)V(y)\psi(y)\,dy.
$$
在 $r=|x|\to\infty$ 时，先对有界的 $y$ 区域展开
$$
G_k^+(x-y)
=\frac{e^{ikr}}{4\pi r}e^{-i\mathbf k'\cdot y}+o(r^{-1}).
$$
加权 $L^1$ 条件控制大 $|y|$ 尾部。再作第一 Born 替换 $\psi(y)\mapsto e^{i\mathbf k\cdot y}$，比较 $e^{ikr}/r$ 的系数即得公式。这里证明的是给定 Lippmann--Schwinger 方程后的首阶迭代；该积分方程的存在唯一性与 Born 级数收敛仍属于散射理论边界。$\square$

**定义 14.6.** 对弹性散射 $|\mathbf k'|=|\mathbf k|$，并采用上面的单位入射平面波归一化，微分截面定义为
$$
\frac{d\sigma}{d\Omega}=|f(\mathbf k',\mathbf k)|^2.
$$
若入射与出射通道的速度不同，概率流之比还会产生 $v_f/v_i$ 因子，不能沿用这一弹性公式。

**命题 14.7.** 若 $V$ 为球对称函数，则 Born 振幅只依赖动量转移大小 $q=|\mathbf k'-\mathbf k|$。

**证明.** Born 振幅是 $V$ 的 Fourier 变换在 $q=\mathbf k'-\mathbf k$ 处的值。球对称函数的 Fourier 变换仍球对称；这是由旋转不变性得到的：对任意旋转 $R$，
$$
\widehat V(Rq)=\int e^{-iRq\cdot x}V(x)\,dx
=\int e^{-iq\cdot y}V(Ry)\,dy
=\widehat V(q).
$$
故只依赖 $|q|$。$\square$

## 14.3 光学定理的边界

**外部输入定理 14.8（光学定理，QM-EXT-16）.** 在三维弹性散射的标准归一化下，总截面与前向散射振幅满足
$$
\sigma_{\operatorname{tot}}=\frac{4\pi}{k}\operatorname{Im} f(\mathbf k,\mathbf k).
$$

**说明 14.9.** 光学定理本质上是 $S$ 矩阵酉性的后果。严格推导需要固定散射态归一化、处理 delta 函数和连续谱通道。本书只记录其结构含义：散射到所有方向的总概率由前向振幅的虚部控制。

**命题 14.10（有限维酉矩阵类比）.** 若 $S=I+iT$ 为有限维酉矩阵，则
$$
i(T-T^*)=-T^*T.
$$

**证明.** 由 $S^*S=I$，
$$
(I-iT^*)(I+iT)=I.
$$
展开得
$$
I+iT-iT^*+T^*T=I.
$$
移项即得。$\square$

**说明 14.11.** 中心势散射常用 partial wave 展开，把入射平面波分解为不同角动量通道。每个通道的散射由相移 $\delta_\ell$ 描述，截面公式转化为相移的级数。完整推导需要球 Bessel 函数和渐近分析，本书把它作为散射专题的后续扩展。

**外部输入定理 14.12（partial wave 展开，QM-EXT-16）.** 对短程中心势，标准形式为
$$
f(\theta)=\frac1{k}\sum_{\ell=0}^\infty(2\ell+1)e^{i\delta_\ell}\sin\delta_\ell\,P_\ell(\cos\theta).
$$
该公式说明相移 $\delta_\ell$ 是中心势散射的核心数据。

**公式 14.13.** 同一规范下，由 partial wave 展开得到总截面
$$
\sigma_{\operatorname{tot}}
=\frac{4\pi}{k^2}\sum_{\ell=0}^\infty(2\ell+1)\sin^2\delta_\ell.
$$
这与光学定理相容，并体现每个角动量通道对总概率流损失的贡献。

## 本章小结

散射理论比较真实演化和自由演化在无穷远过去与未来的差异。波算子
从 $\mathcal H_{\mathrm{ac}}(H_0)$ 映入相互作用动力学；渐近完备性
精确地断言其值域等于 $\mathcal H_{\mathrm{ac}}(H)$。Born 近似是
Lippmann--Schwinger 方程的首阶迭代，而“势很弱”还必须排除阈值、
共振或长程效应破坏该迭代。完整存在性和完备性属于外部输入。

## 练习

**练习 14.1.** 证明若 $S$ 酉，则总概率守恒。

**练习 14.2.** 对 delta 型形式势的 Fourier 变换解释为什么 Born 振幅近似为常数。
