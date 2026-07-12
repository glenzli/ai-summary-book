# 第一章：尺度分离、局域性与有效理论

## 本章目标

本章给出 EFT 的基本定义：当低能过程不能解析重自由度时，可以用局域算符展开来吸收高能物理的影响。核心思想是尺度分离和局域性。

## 依赖前置知识

需要序章中的质量维数、自然单位和路径积分的基本记号。

## 1.1 尺度分离

**定义 1.1（三类尺度）.** 对一个固定 UV-to-EFT 问题，本书区分：

1.  $M_{\rm gap}>0$：从低能展开点到被积掉自由度所产生的最近 pole、branch point 或 anomalous threshold 的物理尺度；
2.  $\Lambda_{\rm ref}>0$：为使 Wilson 系数无量纲而选择的参考尺度；
3.  $\mu_{\rm match}>0$：在某重整化方案中施加匹配条件的尺度。

只有 $M_{\rm gap}$ 控制局域 Taylor 展开的运动学边界。$\Lambda_{\rm ref}$ 可连同 Wilson 坐标一起重标度，$\mu_{\rm match}$ 则由匹配与 RG 的尺度抵消控制。在单重尺度例子中可声明 $M_{\rm gap}=\Lambda_{\rm ref}=M$，但不能据此把 $\mu_{\rm match}$ 当作物理质量。

**定义 1.1A（低能运动学域）.** 对固定外态，令 $\{I_a\}$ 表示独立 Mandelstam invariants 及其他会进入振幅解析结构的动量组合。若存在 $0<\rho<1$ 使
$$
|I_a|\le \rho M_{\rm gap}^2,
\qquad m_{\mathrm{light}}^2\le\rho M_{\rm gap}^2,
$$
且该域不穿过被积掉自由度的奇点，则称它是相对 $M_{\rm gap}$ 的低能运动学域。记
$$
Q\coloneqq\max\!\left(\max_a\sqrt{|I_a|},m_{\rm light}\right).
$$
符号 $Q\ll M_{\rm gap}$ 只作为这些逐不变量条件的简写，不是独立的数学假设。

**定义 1.2（有效自由度）.** 对固定低能运动学域，凡在该域内具有需显式保留的 pole、背景响应或长波涨落，并被选作 EFT 路径积分变量的场，称为该 EFT 的有效自由度。此定义由谱与过程决定，不由任意重整化尺度 $\mu$ 单独决定。

**原则 1.3（局域展开）.** 若重自由度对轻场 1PI 顶点的贡献在所选低能运动学域内对外动量解析，则它可按外动量除以重尺度展开。动量多项式在位置空间对应局域导数算符；非解析的轻场传播部分必须保留在 EFT 的显式圈图中。

**推导说明.** 对树级重标量传播子
$$
\frac{i}{p^2-M^2}
=-\frac{i}{M^2}\left(1+\frac{p^2}{M^2}+\frac{p^4}{M^4}+\cdots\right),
\qquad |p^2|\ll M^2.
$$
每一项在位置空间对应有限个导数的局域相互作用。圈图情形还会产生对数和阈值；低于阈值时仍可用局域项加上轻场非解析结构表示。$\square$

**命题 1.3A（重传播子的有限阶余项）.** 若 $|p^2|/M^2\le\rho<1$，则对任意 $N\ge0$，
$$
\frac1{M^2-p^2}
=\frac1{M^2}\sum_{n=0}^{N}\left(\frac{p^2}{M^2}\right)^n
+R_N(p^2),
$$
且
$$
|R_N(p^2)|
\le\frac1{M^2}\frac{\rho^{N+1}}{1-\rho}.
$$

**证明（书内推导）.** 对 $x=p^2/M^2$ 使用有限几何级数恒等式
$$
\frac1{1-x}=\sum_{n=0}^{N}x^n+\frac{x^{N+1}}{1-x}.
$$
当 $|x|\le\rho<1$ 时，$|1-x|\ge1-\rho$，故余项满足所述一致界。$\square$

## 1.2 EFT 的定义

**定义 1.4（有效拉氏量）.** 给定低能场 $\phi$、对称群 $G$、参考尺度 $\Lambda_{\rm ref}$、系数域和冗余关系（分部积分、领先 EOM、Bianchi/Fierz 等），一个局域 EFT 拉氏量是局域算符等价类的形式展开
$$
\mathcal L_{\mathrm{EFT}}
=
\sum_i c_i\,\mathcal O_i(\phi,\partial\phi),
$$
其中 $\mathcal O_i$ 是 Hermitian 作用量中允许的 $G$-不变局域算符代表，$c_i$ 是 Wilson 系数。该无穷和通常按幂计数理解为形式或渐近展开；实际预测必须在有限阶截断。按质量维数写成
$$
\mathcal L_{\mathrm{EFT}}
=
\mathcal L_{\le 4}
+
\sum_{d>4}\sum_i
\frac{C_i^{(d)}(\mu)}{\Lambda_{\rm ref}^{d-4}}\mathcal O_i^{(d)}(\mu).
$$

**命题 1.5（Wilson 坐标的维数与参考尺度协变性）.** 若 $[\mathcal O_i^{(d)}]=d$，则有量纲系数
$$
c_i^{(d)}(\mu)
\coloneqq
\frac{C_i^{(d)}(\mu)}{\Lambda_{\rm ref}^{d-4}}
$$
满足 $[c_i^{(d)}]=4-d$。若把参考尺度改为 $\Lambda'_{\rm ref}=a\Lambda_{\rm ref}$，其中 $a>0$，则同一作用量要求
$$
C_i^{(d)\prime}=a^{d-4}C_i^{(d)}.
$$
因此 $C_i^{(d)}$ 与 $\Lambda_{\rm ref}$ 各自都不是物理量，只有 $c_i^{(d)}$ 或与矩阵元组成的预测有不依赖该重标度的意义。

**证明（书内推导）.** 拉氏量密度质量维数为 $4$，故乘在维数 $d$ 算符前的系数维数为 $4-d$。参考尺度变换后，
$$
\frac{C_i^{(d)\prime}}{(\Lambda'_{\rm ref})^{d-4}}
=
\frac{a^{d-4}C_i^{(d)}}{a^{d-4}\Lambda_{\rm ref}^{d-4}}
=c_i^{(d)},
$$
所以作用量不变。$\square$

## 1.3 Decoupling 与例外

**外部输入定理 1.6（Appelquist--Carazzone 型 decoupling，EFT-DEC）.** 在可重整化 UV 理论、固定低能重整化条件、外动量远低于重质量且不存在由重质量同时放大的耦合时，重场对低能 Green 函数的影响可吸收到低能参数重定义和按 $1/M$ 抑制的局域算符中。若重质量来自对称性破缺并使耦合随 $M$ 增长，或理论处于强耦合/非解析极限，则该结论的朴素幂计数不适用。

**使用边界.** 本书使用这一外部定理来解释 decoupling EFT 的存在，不在书内重证一般多圈重整化论证。树级传播子例子由命题 1.3A 独立覆盖。

**边界 1.7.** 本书的 SMEFT 主线默认采用 decoupling 设定：最近 BSM 质量隙 $M_{\rm gap}$ 高于电弱尺度，Higgs 属于 $SU(2)_L$ 双重态，电弱对称性线性实现。非线性实现属于 HEFT，不作为本书主线。

## 1.4 局域展开的数学含义

设重场交换在动量空间给出核
$$
K(p^2)={1\over M^2-p^2}.
$$
在 $|p^2|<M^2$ 的圆盘内，$K$ 是解析函数，故有收敛幂级数
$$
K(p^2)={1\over M^2}\sum_{n=0}^\infty \left({p^2\over M^2}\right)^n.
$$
Fourier 反变换时，$p^{2n}$ 变成 $(-\partial^2)^n$，所以非局域核
$$
\int d^4x\,d^4y\,J(x)K(x-y)J(y)
$$
在低能区等价于局域级数
$$
\int d^4x\,\left[
{1\over M^2}J^2
-{1\over M^4}J\partial^2J
+{1\over M^6}J(\partial^2)^2J
+\cdots
\right].
$$

**命题 1.8（解析顶点的局域导数展开）.** 固定一个 $n$ 点 1PI 顶点并消去动量守恒后的冗余动量。若其重场贡献 $\Gamma_{\mathrm{heavy}}(p_1,\ldots,p_{n-1})$ 在原点的某个复邻域内解析，则在该邻域内存在收敛的多重 Taylor 展开；每个有限阶 Taylor 多项式对应有限个局域导数算符。

**证明.** 多复变量解析性给出在某个 polydisc 内绝对收敛的 Taylor 展开。动量守恒已用于删除一个外动量，因此每个 Taylor 单项式都是独立外动量分量的有限次多项式。Fourier 变换下，乘以 $p_{a\mu}$ 对应对第 $a$ 个场插入作用 $-i\partial_\mu$；故每个单项式对应同一点处有限个场及有限阶导数的局域算符。截断 Taylor 总次数后只剩有限个 Lorentz/internal-index contractions。$\square$

**警告 1.9（非解析项）.** 轻场圈图会产生 $\log(-p^2/\mu^2)$、$\sqrt{1-4m^2/p^2}$ 等非解析结构。这些不是重物理的局域 Wilson 系数，而是 EFT 内部轻自由度传播产生的长程效应，必须在 EFT 中显式计算。

## 1.5 截断误差

**定义 1.9A（一致渐近截断）.** 固定不穿越阈值、软端点或共线端点的紧致无量纲运动学域 $\mathcal D$，并令 $\lambda=Q/M_{\rm gap}\to0^+$。给定满足 $\varphi_{n+1}(\lambda)=o(\varphi_n(\lambda))$ 的渐近尺度，例如按既定次序排列的 $\lambda^n(\log\lambda)^k$，若对每个 $N$ 都存在 $K_N,\lambda_N>0$ 使
$$
\sup_{z\in\mathcal D}
\left|
A(\lambda,z)-\sum_{n=0}^{N}A_n(z)\varphi_n(\lambda)
\right|
\le K_N|\varphi_{N+1}(\lambda)|
$$
对 $0<\lambda<\lambda_N$ 成立，则称该展开在 $\mathcal D$ 上是一致渐近展开。常数 $K_N$ 可以随截断阶 $N$ 增长；该定义不蕴含无穷级数收敛。

**例 1.9B（渐近不等于收敛）.** 对 $x>0$ 定义
$$
f(x)=\int_0^\infty\frac{e^{-t}}{1+xt}\,dt.
$$
有限几何恒等式给出
$$
f(x)=\sum_{n=0}^{N}(-1)^n n!x^n+R_N(x),
\qquad
|R_N(x)|\le (N+1)!x^{N+1}.
$$
故该级数在 $x\to0^+$ 时对每个固定 $N$ 都是渐近的；但对任何固定 $x\ne0$，通项 $n!x^n$ 最终不趋于零，所以无穷级数不收敛。

**证明（书内推导）.** 将
$$
\frac1{1+xt}
=\sum_{n=0}^{N}(-xt)^n+\frac{(-xt)^{N+1}}{1+xt}
$$
代入积分。逐项积分使用 $\int_0^\infty e^{-t}t^n\,dt=n!$；又因 $1+xt\ge1$，余项绝对值不超过 $x^{N+1}\int_0^\infty e^{-t}t^{N+1}dt$。最后用通项判别即得不收敛性。$\square$

**物理边界 1.9C.** 命题 1.3A 对单个树级传播子给出真正的一致余项界；一般重整化量子场论的 EFT 级数则通常只按渐近与幂计数解释。本书不把后者冒充为已经证明收敛的函数级数。轻场产生的非解析项应保留在系数函数 $A_n$ 中或单独因子化，不能塞入局域 Wilson 系数。

若保留到维数 $D$，令 $d_{\mathrm{next}}>D$ 为对称性和选择定则允许的最低遗漏维数。在单重尺度 UV 计数、Wilson 系数没有额外增强、运动学远离阈值且既定幂计数有效时，典型振幅的首个遗漏量估计为
$$
{\Delta A\over A_{\rm ref}}
=O\!\left(|C_{\mathrm{next}}|
\left({Q\over\Lambda_{\rm ref}}\right)^{d_{\mathrm{next}}-4}\right)
=O\!\left(|\widehat C_{\mathrm{next}}|
\left({Q\over M_{\rm gap}}\right)^{d_{\mathrm{next}}-4}\right),
$$
其中 $\widehat C_{\mathrm{next}}=C_{\mathrm{next}}(M_{\rm gap}/\Lambda_{\rm ref})^{d_{\mathrm{next}}-4}$，并省略了耦合、选择定则和可能的环因子。这是 power-counting 误差模型，不是脱离 UV 假设的严格数值上界。

**原则 1.10（误差声明）.** EFT 计算必须同时声明：

1.  保留到哪个维数；
2.  振幅还是截面层面截断；
3.  是否保留平方项；
4.  构造 $Q_{\max}$ 所用的全部硬不变量与 cuts，以及采用的 $M_{\rm gap}$；
5.  用什么量估计遗漏项。

没有误差声明的 Wilson 系数结果不是可复核 EFT 结果。

## 本章小结

EFT 的基本对象是给定自由度、对称性和冗余商后的局域渐近作用量。$M_{\rm gap}$ 控制解析边界，$\Lambda_{\rm ref}$ 只归一化 Wilson 坐标，$\mu_{\rm match}$ 组织阈值计算；预测能力来自在固定运动学域上的有限阶截断，而不是无穷级数必然收敛。

## 练习

**练习 1.1.** 展开 $1/(M^2-p^2)$ 到 $p^4/M^6$，并写出对应的导数算符阶数。

**练习 1.2.** 说明为什么接近重粒子阈值时不能只用局域算符展开。

**练习 1.3.** 对非局域核 $1/(M^2-p^2)$ 写出到 $p^6/M^8$ 的局域导数展开。

**练习 1.4.** 对维数六项保持 $c^{(6)}=C^{(6)}/\Lambda_{\rm ref}^2$ 不变，把 $\Lambda_{\rm ref}$ 放大十倍，求 $C^{(6)}$ 的变换，并核验作用量不变。

**练习 1.5.** 用例 1.9B 的余项界说明：对固定 $N$，$x\to0^+$ 时截断受控；但该结论为何不能交换成“对固定 $x$ 令 $N\to\infty$”。
