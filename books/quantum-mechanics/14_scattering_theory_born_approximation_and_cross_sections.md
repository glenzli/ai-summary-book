# 第十四章：散射理论、Born 近似与截面

## 本章目标

本章介绍散射理论的基本对象：自由与相互作用 Hamiltonian、波算子、$S$ 矩阵、散射振幅和 Born 近似。

## 依赖前置知识

需要时间演化、Fourier 变换、一维势垒和含时扰动理论。

## 14.1 波算子

**定义 14.1.** 设 $H_0$ 为自由 Hamiltonian，$H=H_0+V$ 为相互作用 Hamiltonian。若强极限存在，定义 Møller 波算子
$$
\Omega_\pm=\operatorname{s-lim}_{t\to\pm\infty}e^{itH}e^{-itH_0}.
$$

**外部输入定理 14.2（散射存在性边界，QM-EXT-8）.** 对短程势，在适当假设下 $\Omega_\pm$ 存在并在连续谱子空间上具有等距性；渐近完备性需要更深的散射理论。

**定义 14.3.** 散射算子定义为
$$
S=\Omega_+^*\Omega_-.
$$

**命题 14.4.** 若 $\Omega_\pm$ 为酉算子，则 $S$ 为酉算子。

**证明.** 若 $\Omega_\pm^*\Omega_\pm=I=\Omega_\pm\Omega_\pm^*$，则
$$
S^*S=\Omega_-^*\Omega_+\Omega_+^*\Omega_-=I,
$$
同理 $SS^*=I$。$\square$

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

散射理论比较真实演化和自由演化在无穷远过去与未来的差异。波算子和 $S$ 矩阵是严格对象；Born 近似是 Lippmann--Schwinger 方程的首阶迭代，而“势很弱”还必须排除阈值、共振或长程效应破坏该迭代。完整存在性和完备性属于外部输入。

## 练习

**练习 14.1.** 证明若 $S$ 酉，则总概率守恒。

**练习 14.2.** 对 delta 型形式势的 Fourier 变换解释为什么 Born 振幅近似为常数。
