# 第九章：量子场、Wightman 公理与 Fock 空间

自由量子场可以在 Fock 空间上严格构造，但严格对象不是每个时空点上的算符，而是测试函数到共同稠密定义域上算符的分布。这个区别修复了两个常见误解：$a(\mathbf p)$ 是算符值分布而不是每个动量处的有界算符，$[\phi(x),\pi(y)]$ 也只能在对空间测试函数配对后理解。相互作用场的存在性远超本书范围，因此本章把可验证的自由场构造与外部的 Wightman 重构定理并列，而不让形式模展开承担不存在的非微扰证明。

## 9.1 对称 Fock 空间

**定义 9.1.** 给定复 Hilbert 空间 $\mathcal H_1$，bosonic Fock 空间为
$$
\mathcal F_s(\mathcal H_1)
=\bigoplus_{n=0}^\infty\operatorname{Sym}^n\mathcal H_1.
$$
真空向量为 $\Omega=(1,0,0,\ldots)$，有限粒子子空间
$\mathcal D_{\rm fin}=\bigoplus_{n=0}^{\rm alg}\operatorname{Sym}^n\mathcal H_1$ 稠密。

对 $f\in\mathcal H_1$，在 $n$ 粒子分量上定义
$$
a^\dagger(f)\psi_n=\sqrt{n+1}\,
\operatorname{Sym}_{n+1}(f\otimes\psi_n),
$$
并把 $a(f)$ 定义为 $a^\dagger(f)$ 在 $\mathcal D_{\rm fin}$ 上的代数伴随。按本书内积约定，$a^\dagger(f)$ 对 $f$ 线性，$a(f)$ 对 $f$ 反线性。

**命题 9.1 (`P`).** $\mathcal D_{\rm fin}$ 在产生、湮灭算符下不变，且对任意 $f,g\in\mathcal H_1$，
$$
[a(f),a^\dagger(g)]=\langle f,g\rangle I,
\qquad
[a(f),a(g)]=[a^\dagger(f),a^\dagger(g)]=0
$$
在 $\mathcal D_{\rm fin}$ 上成立。

**证明.** 在对称基本张量 $h_1\otimes_s\cdots\otimes_s h_n$ 上，定义给出
$$
a(f)(h_1\otimes_s\cdots\otimes_s h_n)
=\frac1{\sqrt n}\sum_{r=1}^n
\langle f,h_r\rangle
h_1\otimes_s\cdots\widehat h_r\cdots\otimes_s h_n.
$$
先产生 $g$ 再湮灭时，求和中有一项收缩新加入的 $g$，贡献
$\langle f,g\rangle$ 乘原张量；其余 $n$ 项正好是先湮灭再产生所得的项。因此
$a(f)a^\dagger(g)-a^\dagger(g)a(f)=\langle f,g\rangle I$。两个产生算符的交换性来自对称化不依赖加入次序；取代数伴随得到两个湮灭算符的交换性。基本对称张量张成每个有限粒子扇区，故等式在 $\mathcal D_{\rm fin}$ 上成立。$\square$

**例 9.2（单模粒子数）.** 若 $\mathcal H_1=\mathbb C e$ 且 $\|e\|=1$，令
$|n\rangle=(a^\dagger(e))^n\Omega/\sqrt{n!}$。命题 9.1 递推给出
$$
a(e)|n\rangle=\sqrt n\,|n-1\rangle,
\qquad
a^\dagger(e)|n\rangle=\sqrt{n+1}\,|n+1\rangle.
$$
因此粒子数算符 $N=a^\dagger(e)a(e)$ 在有限粒子域上满足 $N|n\rangle=n|n\rangle$。

## 9.2 自由 Klein--Gordon 场

设时空维数为 $d\ge2$，空间维数 $s=d-1$，质量 $m>0$，并记
$E_{\mathbf p}=\sqrt{|\mathbf p|^2+m^2}$。取一粒子空间
$\mathcal H_1=L^2(\mathbb R^s,d^s\mathbf p/(2\pi)^s)$。符号 $a(\mathbf p)$、$a^\dagger(\mathbf p)$ 只表示满足
$$
[a(\mathbf p),a^\dagger(\mathbf q)]
=(2\pi)^s\delta^{(s)}(\mathbf p-\mathbf q)
$$
的算符值分布。

**定义 9.2.** 自由实标量场的模展开为
$$
\phi(t,\mathbf x)=
\int\frac{d^s\mathbf p}{(2\pi)^s}
\frac1{\sqrt{2E_{\mathbf p}}}
\left(
a(\mathbf p)e^{-iE_{\mathbf p}t+i\mathbf p\cdot\mathbf x}
+a^\dagger(\mathbf p)e^{iE_{\mathbf p}t-i\mathbf p\cdot\mathbf x}
\right),
$$
并令 $\pi=\partial_t\phi$。该式的含义是：先对 $\mathbf x$ 与 Schwartz 函数配对，所得算符作用在 $\mathcal D_{\rm fin}$ 上；不主张 $\phi(t,\mathbf x)$ 在固定点是有定义的闭算符。

**命题 9.2 (`P`).** 对任意 $f,g\in\mathcal S(\mathbb R^s)$，等时 smeared fields 满足
$$
[\phi_t(f),\phi_t(g)]=[\pi_t(f),\pi_t(g)]=0,
\qquad
[\phi_t(f),\pi_t(g)]
=i\int_{\mathbb R^s}f(\mathbf x)g(\mathbf x)\,d^s\mathbf x
$$
在 $\mathcal D_{\rm fin}$ 上成立。等价地，
$[\phi(t,\mathbf x),\pi(t,\mathbf y)]=i\delta^{(s)}(\mathbf x-\mathbf y)$ 在算符值分布意义成立。

**证明.** 把定义 9.2 与其时间导数代入对易子。$a$ 与 $a$、$a^\dagger$ 与 $a^\dagger$ 的项由命题 9.1 为零；两个交叉项利用动量 delta 分布后给出
$$
[\phi(t,\mathbf x),\pi(t,\mathbf y)]
=\frac i2\int\frac{d^s\mathbf p}{(2\pi)^s}
\left(e^{i\mathbf p\cdot(\mathbf x-\mathbf y)}
+e^{-i\mathbf p\cdot(\mathbf x-\mathbf y)}\right).
$$
第二项作变量替换 $\mathbf p\mapsto-\mathbf p$，两项相等；Fourier 反演给出 $i\delta^{(s)}(\mathbf x-\mathbf y)$。先与 $f(\mathbf x)g(\mathbf y)$ 配对便得到命题中的严格等式。相同计算中，$[\phi,phi]$ 的两个交叉项相消；$[\pi,pi]$ 亦然。所有被 smeared 的系数属于一粒子空间，所以算符在 $\mathcal D_{\rm fin}$ 上定义良好。$\square$

## 9.3 Wightman 框架

**定义 9.3.** 一个标量 Wightman 场包括 Hilbert 空间 $\mathcal H$、稠密共同不变域 $\mathcal D$、真空 $\Omega\in\mathcal D$、Poincare 群正时向分支的强连续酉表示 $U$，以及算符值 tempered distribution
$\phi:\mathcal S(\mathbb R^d)\to\operatorname{End}(\mathcal D)$，并满足：

1. $U$ 协变性与真空不变性；
2. 平移生成元的联合谱包含于闭未来光锥；
3. 若 $\operatorname{supp}f$ 与 $\operatorname{supp}g$ 类空间分离，则 $[\phi(f),\phi(g)]=0$ 于 $\mathcal D$；
4. 对中性标量场，$\phi(\overline f)$ 是 $\phi(f)$ 在共同域上的伴随，即
   $\langle\phi(f)\psi,\chi\rangle=\langle\psi,\phi(\overline f)\chi\rangle$ 对 $\psi,\chi\in\mathcal D$ 成立；
5. $\Omega$ 对场多项式生成的子空间循环，真空在相位外唯一。

Hilbert 空间正定性已经编码了 Wightman 分布的正性条件。对带自旋场，协变律与类空间交换关系需按有限维 Lorentz 表示和 Bose/Fermi 统计修改。

**定理 9.3 (`E`, Wightman 重构).** 设一族 tempered distributions
$W_n\in\mathcal S'((\mathbb R^d)^n)$ 满足 Poincare 协变性、谱条件、Hermiticity、局域性、Wightman 正性与真空循环性条件。则存在 Wightman 场系统
$(\mathcal H,\mathcal D,U,\Omega,\phi)$，使
$$
W_n(f_1,\ldots,f_n)
=\langle\Omega,\phi(f_1)\cdots\phi(f_n)\Omega\rangle,
$$
且该系统在保持真空、场和 Poincare 表示的酉等价意义下唯一。

**证明路线（外部输入）.** 先以场符号的张量代数构造真空线性泛函；Wightman 正性给出半内积，商去零空间并完备化得到 Hilbert 空间。协变性和谱支撑用于构造 Poincare 表示，左乘场符号给出共同域上的算符值分布。闭包、连续性与唯一性需要完整的核定理和分布论论证，见 [SOURCES.md](SOURCES.md) 的 `E-9.3`。

**命题 9.4 (`P`).** 在满足谱条件的 Wightman 理论中，不存在联合平移谱位于 $p^0<0$ 的非零一粒子态。

**证明.** 谱条件按定义断言平移表示生成元的联合谱包含于
$\overline V_+=\{p:p^0\ge0,\ p^2\le0\}$（mostly-plus 号差）。任一一粒子子空间是平移表示的不变闭子空间，其谱包含于整体表示的谱。若其中存在谱支撑于 $p^0<0$ 的非零态，则相应谱投影在 $\overline V_+$ 的补集上非零，与整体联合谱包含关系矛盾。$\square$

**命题 9.5 (`P`, 自由 Fock 真空的 Wick 定理).** 设 $\Phi_r=a(u_r)+a^\dagger(v_r)$ 是 $\mathcal D_{\rm fin}$ 上有限个线性自由场算符，并令
$C_{ij}=\langle\Omega,\Phi_i\Phi_j\Omega\rangle$（$i<j$）。则奇数个算符的真空期望为零，而
$$
\langle\Omega,\Phi_1\cdots\Phi_{2r}\Omega\rangle
=\sum_{\pi\in\mathcal P_2(2r)}
\prod_{\{i,j\}\in\pi,\,i<j}C_{ij}.
$$

**证明.** 对算符个数归纳。真空满足 $a(u)\Omega=0$；而对每个 $\Psi\in\mathcal D_{\rm fin}$，
$$
\langle\Omega,a^\dagger(v)\Psi\rangle
=\langle a(v)\Omega,\Psi\rangle=0.
$$
因此 $\Phi_1$ 的产生部分对真空期望没有贡献。把其湮灭部分 $a(u_1)$ 逐次移到最右端。每越过 $\Phi_j$，命题 9.1 产生 c-number 对易子
$[a(u_1),\Phi_j]=C_{1j}$；移到最右端的项作用于真空后为零。因此
$$
\langle\Omega,\Phi_1\cdots\Phi_m\Omega\rangle
=\sum_{j=2}^m C_{1j}
\langle\Omega,\Phi_2\cdots\widehat\Phi_j\cdots\Phi_m\Omega\rangle.
$$
当 $m$ 为奇数，右端最终化为单个场的真空期望，等于零。当 $m=2r$，归纳假设把每个余下期望写成其余指标的全部配对；再选择与指标 $1$ 配对的唯一 $j$，恰好枚举全部 $\mathcal P_2(2r)$ 且无重复。$\square$

该命题是自由 Fock 表示内的完整代数证明。把它写成未构造的连续路径积分 Gaussian “测度”恒等式，或用于相互作用真空的微扰展开时，则必须回到第八章的 `S` 口径。

## 练习

**练习 9.1.** 计算自由标量场的二点 Wightman 分布，并说明为何积分先与 Schwartz 测试函数配对。

**练习 9.2.** 用命题 9.2 的模展开证明自由场对易子在类空间隔为零；可以调用 Lorentz 不变性把间隔变到等时形式。
