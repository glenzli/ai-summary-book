# 第十四章：散射理论、Born 近似与截面

## 本章目标

本章介绍散射理论的基本对象：自由与相互作用 Hamiltonian、波算子、$S$ 矩阵、散射振幅和 Born 近似。

## 依赖前置知识

需要时间演化、Fourier 变换、一维势垒和含时扰动理论。

## 14.1 波算子

**定义 14.1.** 设 $H_0$ 为自由 Hamiltonian，$H=H_0+V$ 为相互作用 Hamiltonian。若强极限存在，定义 Moller 波算子
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

**公式 14.5.** 在三维中，对弱势 $V$，一阶 Born 散射振幅为
$$
f(\mathbf k',\mathbf k)
=-\frac{m}{2\pi}\int_{\mathbb R^3}e^{-i(\mathbf k'-\mathbf k)\cdot x}V(x)\,dx.
$$
这里采用 $\hbar=1$。

**定义 14.6.** 微分截面定义为
$$
\frac{d\sigma}{d\Omega}=|f(\mathbf k',\mathbf k)|^2.
$$

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

散射理论比较真实演化和自由演化在无穷远过去与未来的差异。波算子和 $S$ 矩阵是严格对象；Born 近似给出弱势下的可计算振幅。完整存在性和完备性属于外部输入。

## 练习

**练习 14.1.** 证明若 $S$ 酉，则总概率守恒。

**练习 14.2.** 对 delta 型形式势的 Fourier 变换解释为什么 Born 振幅近似为常数。
