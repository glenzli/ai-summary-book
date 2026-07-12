# 符号约定

## Hilbert 空间

- $\mathcal H,\mathcal K$：复 Hilbert 空间。
- $\langle \psi,\phi\rangle$：内积，对第二变量线性，对第一变量共轭线性。
- $\|\psi\|=\sqrt{\langle\psi,\psi\rangle}$：范数。
- $|\psi\rangle$：ket；$\langle\psi|$：bra，表示 $\phi\mapsto\langle\psi,\phi\rangle$。
- $|\psi\rangle\langle\phi|$：秩一算子，$x\mapsto \psi\langle\phi,x\rangle$。
- $[\psi]$：单位向量 $\psi$ 所定义的射线。

## 算子

- $\mathcal B(\mathcal H)$：$\mathcal H$ 上有界线性算子代数。
- $A^*$：有界算子的伴随；无界算子的伴随按定义域定义。
- $\mathcal D(A)$：无界算子 $A$ 的定义域。
- $[A,B]=AB-BA$：交换子；无界情形只在共同定义域上使用。
- $a,a^*,N=a^*a$：谐振子的湮灭、产生与数算子；定义域见第 7 章。
- $\Delta_\psi A$：态 $\psi$ 中可观测量 $A$ 的标准差。
- $E_A(\Delta)$：自伴算子 $A$ 的谱测度在 Borel 集 $\Delta$ 上的投影。
- $\sigma(A)$：谱。
- $\mathcal H_{\mathrm{ac}}(H)$、$P_{\mathrm{ac}}(H)$：$H$ 的绝对连续谱子空间及其正交投影。

## 态与测量

- $\rho$：密度算子，$\rho\ge0$ 且 $\operatorname{tr}\rho=1$。
- $P_\psi=|\psi\rangle\langle\psi|$：单位向量 $\psi$ 的纯态投影。
- $\operatorname{tr}_{\mathcal K}$：对 $\mathcal K$ 的偏迹。
- $\{M_i\}$：Kraus 算子族。
- $\{E_i\}$：POVM 效应，$E_i\ge0$ 且 $\sum_iE_i=I$。

## 动力学

- 默认单位 $\hbar=1$。
- $H$：Hamiltonian，自伴算子。
- $U(t)=e^{-itH}$：时间演化酉群。
- $\rho(x,t)=|\psi(x,t)|^2$：位置表象概率密度。
- $j(x,t)=m^{-1}\operatorname{Im}(\overline\psi\nabla\psi)$：概率流。
- $X_j,P_j$：位置与动量算子。
- $L=x\times P$：轨道角动量。
- $Y_\ell^m$：球谐函数。
- $A,\Phi$：电磁矢势和标势；若与抽象算子冲突，以局部说明为准。
- $\Pi=P-qA(X)$：动力学动量。
- $B=\nabla\times A$：磁场。
- $U_I(t,t_0)$：相互作用图像传播子。
- $J_\pm=J_x\pm iJ_y$：角动量升降算符。
- $\Delta=\sum_j\partial_j^2$：Laplacian。
- $S(t)=e^{-itH}$：有时也用于传播子；若有冲突，以章节局部说明为准。
