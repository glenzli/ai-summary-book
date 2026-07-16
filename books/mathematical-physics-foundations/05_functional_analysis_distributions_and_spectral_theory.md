# 第五章：泛函分析、分布与谱理论

量子力学把可观测量表示为算符，但位置、动量和 Hamiltonian 几乎从来不是处处定义的有界算符。仅写出微分表达式 $-i\,d/dx$ 并没有定义一个可观测量：定义域决定分部积分的边界项，也决定算符是否自伴以及生成哪一种时间演化。场论还要求把点场改写为算符值分布，因为测试函数之外的逐点乘积通常没有定义。本章建立后续量子论实际调用的分析口径，同时把需要完整泛函分析或 PDE 理论的结果明确留在外部边界。

## 5.1 定义域、图与伴随

**定义 5.1.** Hilbert 空间是完备内积空间。本书内积 $\langle\psi,\phi\rangle$ 对第二个变量线性。若 $\mathcal H$ 是复 Hilbert 空间，线性算符是一个线性映射
$$
A:\mathcal D(A)\longrightarrow\mathcal H,
$$
其中定义域 $\mathcal D(A)$ 是 $\mathcal H$ 的线性子空间。只有当 $\mathcal D(A)$ 稠密时，本章才定义 Hilbert 空间伴随。

**定义 5.2.** 算符 $A$ 的图为
$$
\mathcal G(A)=\{(\psi,A\psi):\psi\in\mathcal D(A)\}
\subset\mathcal H\oplus\mathcal H.
$$
若图在直和 Hilbert 空间中闭，则称 $A$ 为闭算符。若 $\overline{\mathcal G(A)}$ 仍是某个算符的图，则称 $A$ 可闭；该算符记为 $\overline A$。图范数为
$$
\|\psi\|_A=(\|\psi\|^2+\|A\psi\|^2)^{1/2}.
$$
$A$ 闭当且仅当 $\mathcal D(A)$ 关于图范数完备。

**定义 5.3.** 设 $A$ 稠密定义。其伴随 $A^*$ 的定义域是所有满足下述条件的 $\phi\in\mathcal H$：存在 $\eta\in\mathcal H$，使
$$
\langle A\psi,\phi\rangle=\langle\psi,\eta\rangle
\quad\text{对每个 }\psi\in\mathcal D(A)
$$
成立。稠密性保证这样的 $\eta$ 唯一，定义 $A^*\phi=\eta$。若 $A\subset A^*$，即 $\mathcal D(A)\subset\mathcal D(A^*)$ 且两者在 $\mathcal D(A)$ 上相同，则称 $A$ 对称；若 $A=A^*$，包括定义域相等，则称 $A$ 自伴。

**命题 5.1 (`P`).** 每个稠密定义算符 $A$ 的伴随 $A^*$ 都是闭算符。若 $A$ 对称，则 $A\subset A^*$；因此每个自伴算符都闭。

**证明.** 设 $\phi_n\in\mathcal D(A^*)$，且 $\phi_n\to\phi$、$A^*\phi_n\to\eta$。对任意 $\psi\in\mathcal D(A)$，内积连续性给出
$$
\langle A\psi,\phi\rangle
=\lim_n\langle A\psi,\phi_n\rangle
=\lim_n\langle\psi,A^*\phi_n\rangle
=\langle\psi,\eta\rangle.
$$
故 $\phi\in\mathcal D(A^*)$ 且 $A^*\phi=\eta$，这正是 $\mathcal G(A^*)$ 闭。

若 $A$ 对称，则对 $\phi\in\mathcal D(A)$ 和所有 $\psi\in\mathcal D(A)$，
$\langle A\psi,\phi\rangle=\langle\psi,A\phi\rangle$。按伴随定义，$\phi\in\mathcal D(A^*)$ 且 $A^*\phi=A\phi$，所以 $A\subset A^*$。自伴时 $A=A^*$，第一部分立即给出闭性。$\square$

**命题 5.2 (`P`, 可闭性判据).** 算符 $A$ 可闭，当且仅当对每个序列 $\psi_n\in\mathcal D(A)$，
$$
\psi_n\to0,\quad A\psi_n\to\eta
\quad\Longrightarrow\quad \eta=0.
$$
在此情形，$\overline{\mathcal G(A)}$ 是唯一最小闭扩张 $\overline A$ 的图。

**证明.** 若 $A$ 可闭，而上述序列存在，则 $(0,\eta)\in\overline{\mathcal G(A)}=\mathcal G(\overline A)$。算符的图在第一分量为零时只能含 $(0,0)$，故 $\eta=0$。

反之，设序列条件成立。若 $(\psi,\eta_1)$ 与 $(\psi,\eta_2)$ 都属于 $\overline{\mathcal G(A)}$，分别取图中序列逼近它们并相减，得到 $\chi_n\to0$ 且 $A\chi_n\to\eta_1-\eta_2$；判据给出 $\eta_1=\eta_2$。因此闭子空间 $\overline{\mathcal G(A)}$ 对每个第一分量至多有一个第二分量，确实是某个算符 $\overline A$ 的图。任何闭扩张的图都包含 $\mathcal G(A)$，从而包含其闭包，所以 $\overline A$ 是唯一最小闭扩张。$\square$

**定理 5.3 (`E`, 闭图定理).** 若 $X,Y$ 是 Banach 空间，线性算符 $T:X\to Y$ 在整个 $X$ 上有定义且图闭，则 $T$ 有界。

**外部输入边界.** 该定理说明真正无界的闭算符不可能在整个 Hilbert 空间上有定义。证明依赖 Baire 范畴定理，本书不重证；来源见 [SOURCES.md](SOURCES.md) 的 `E-5.3`。

**定理 5.4 (`E`, von Neumann 自伴扩张判据).** 设 $A$ 是复 Hilbert 空间上稠密定义、闭且对称的算符，并令
$$
\mathcal N_+=\ker(A^*-iI),\qquad
\mathcal N_-=\ker(A^*+iI).
$$
$A$ 存在自伴扩张，当且仅当 $\dim\mathcal N_+=\dim\mathcal N_-$；自伴扩张由从 $\mathcal N_+$ 到 $\mathcal N_-$ 的酉映射参数化。

**证明路线（外部输入）.** 核心步骤是把 $\mathcal D(A^*)$ 按图内积正交分解为 $\mathcal D(A)\oplus\mathcal N_+\oplus\mathcal N_-$，再判断哪些扩张定义域使伴随边界型消失。完整分解与扩张参数化见 [SOURCES.md](SOURCES.md) 的 `E-5.4`；本书只调用“边界条件是算符定义的一部分”这一结论。

**例 5.5（区间上的动量与边界条件）.** 取 $\mathcal H=L^2([0,L],dx)$，先在 $C_c^\infty(0,L)$ 上定义 $P_0f=-if'$。分部积分给出
$$
\langle P_0f,g\rangle-\langle f,-ig'\rangle
=i\,[\overline f g]_0^L=0,
$$
所以 $P_0$ 对称。其伴随定义域为 $H^1([0,L])$，且 $P_0^*g=-ig'$；这里端点值由 $H^1$ 的连续代表解释。于是 $P_0$ 并非自伴，因为其定义域远小于 $H^1$。

对任意 $\theta\in[0,2\pi)$，令
$$
\mathcal D(P_\theta)=
\{f\in H^1([0,L]):f(L)=e^{i\theta}f(0)\},
\qquad P_\theta f=-if'.
$$
若 $f,g\in\mathcal D(P_\theta)$，则
$\overline{f(L)}g(L)=e^{-i\theta}\overline{f(0)}e^{i\theta}g(0)=\overline{f(0)}g(0)$，故边界型为零，$P_\theta$ 对称。反过来，若 $g\in\mathcal D(P_\theta^*)$，先以紧支撑 $f$ 测试可知 $g\in H^1$ 且伴随仍为 $-ig'$；再让 $f(0)$ 任意，边界型消失迫使 $g(L)=e^{i\theta}g(0)$。因此 $\mathcal D(P_\theta^*)=\mathcal D(P_\theta)$，即 $P_\theta$ 自伴。不同 $\theta$ 给出不同量子边界条件，而微分表达式完全相同。

## 5.2 谱测度与函数演算

**定理 5.6 (`E`, 无界自伴算符谱定理).** 若 $A$ 是复 Hilbert 空间 $\mathcal H$ 上的自伴算符，则存在唯一投影值测度
$$
E_A:\mathcal B(\mathbb R)\longrightarrow\mathcal B(\mathcal H)
$$
使得对 $\mu_\psi(\Delta)=\langle\psi,E_A(\Delta)\psi\rangle$，
$$
\mathcal D(A)=
\left\{\psi\in\mathcal H:
\int_{\mathbb R}\lambda^2\,d\mu_\psi(\lambda)<\infty\right\},
\qquad
A\psi=\int_{\mathbb R}\lambda\,dE_A(\lambda)\psi.
$$
更一般地，对 Borel 函数 $f:\mathbb R\to\mathbb C$，
$$
\mathcal D(f(A))=
\left\{\psi:\int_{\mathbb R}|f(\lambda)|^2\,d\mu_\psi(\lambda)<\infty\right\}.
$$

**外部输入边界.** 积分是投影值测度积分，不是把 $A$ 假设成具有可数本征向量基。连续谱时，$E_A(\Delta)$ 仍有定义，而“广义本征函数”通常只在 rigged Hilbert space 或分布意义下使用。第六章的测量概率与时间演化只调用上述 PVM 和 Borel 函数演算。完整证明见 [SOURCES.md](SOURCES.md) 的 `E-5.6`。

**例 5.7（乘法算符的谱测度）.** 设 $(X,\Sigma,\mu)$ 为 $\sigma$-有限测度空间，$a:X\to\mathbb R$ 可测。在 $L^2(X,\mu)$ 上定义
$$
(M_af)(x)=a(x)f(x),\qquad
\mathcal D(M_a)=\{f\in L^2:af\in L^2\}.
$$
它的谱投影是
$$
(E_{M_a}(\Delta)f)(x)=\mathbf1_{a^{-1}(\Delta)}(x)f(x).
$$
这些投影满足可数强可加性，且
$\int\lambda^2\,d\mu_f(\lambda)=\int_X|a|^2|f|^2\,d\mu$，所以谱定理给出的定义域正是 $\mathcal D(M_a)$。为直接检查自伴性，若 $g\in\mathcal D(M_a^*)$ 且 $M_a^*g=h$，取递增的有限测度集合 $X_m\uparrow X$。在 $X_m\cap\{|a|\le n\}$ 上，每个 $L^2$ 测试函数都属于 $\mathcal D(M_a)$；伴随等式于是给出 $h=ag$ 几乎处处于该集合。令 $m,n\to\infty$，得 $h=ag$ 几乎处处于 $X$。因 $h\in L^2$，便有 $ag\in L^2$。故 $\mathcal D(M_a^*)=\mathcal D(M_a)$ 且 $M_a^*=M_a$。

**定理 5.8 (`E`, Stone 定理).** 强连续一参数酉群 $U:\mathbb R\to\mathcal U(\mathcal H)$ 唯一写成
$$
U(t)=e^{-itA}=\int_{\mathbb R}e^{-it\lambda}\,dE_A(\lambda),
$$
其中 $A$ 自伴；反之，每个自伴 $A$ 都由该式生成强连续酉群。对 $\psi\in\mathcal D(A)$，强导数满足
$$
\lim_{t\to0}\frac{U(t)\psi-\psi}{t}=-iA\psi.
$$

**外部输入边界.** Stone 定理保证自伴性而不只是对称性正是酉时间演化的正确条件。证明使用谱定理与强连续群理论，见 [SOURCES.md](SOURCES.md) 的 `E-5.8`。

## 5.3 Schwartz 空间与分布

**定义 5.4.** Schwartz 空间 $\mathcal S(\mathbb R^n)$ 由所有满足
$$
p_{\alpha,\beta}(f)=
\sup_{x\in\mathbb R^n}|x^\alpha\partial^\beta f(x)|<\infty
$$
的光滑函数组成，并赋予半范数族 $p_{\alpha,\beta}$ 定义的 Fréchet 拓扑。其连续线性对偶 $\mathcal S'(\mathbb R^n)$ 称为 tempered distributions。

**命题 5.9 (`P`).** 对 $f\in\mathcal S(\mathbb R^n)$，本书 Fourier 约定满足
$$
\widehat{\partial_jf}(k)=ik_j\widehat f(k),
\qquad
\widehat{x_jf}(k)=i\partial_{k_j}\widehat f(k),
$$
且 $\widehat f\in\mathcal S(\mathbb R^n)$。

**证明.** 第一式由分部积分得到；$f$ 快速下降使无穷远边界项为零。第二式由
$\partial_{k_j}e^{-ik\cdot x}=-ix_je^{-ik\cdot x}$ 并在积分号下求导得到，支配函数可取 $|x_jf(x)|\in L^1$。反复使用两式可把
$k^\alpha\partial_k^\beta\widehat f(k)$ 写成常数乘以某个 Schwartz 函数 $g_{\alpha,\beta}$ 的 Fourier 变换。于是
$$
|k^\alpha\partial_k^\beta\widehat f(k)|
\le \|g_{\alpha,\beta}\|_{L^1}<\infty.
$$
这对所有多重指标成立，故 $\widehat f\in\mathcal S$。$\square$

**定理 5.10 (`E`, Schwartz 空间上的 Fourier 反演).** Fourier 变换是 $\mathcal S(\mathbb R^n)$ 的连续线性自同构，其逆为
$$
f(x)=\frac1{(2\pi)^n}
\int_{\mathbb R^n}e^{ik\cdot x}\widehat f(k)\,d^nk.
$$

**外部输入边界.** 命题 5.9 已在书内证明变换保持 Schwartz 空间；反演、拓扑连续性和满射性作为调和分析输入使用，见 [SOURCES.md](SOURCES.md) 的 `E-5.10`。

**定义 5.5.** 对 $T\in\mathcal S'$，其 Fourier 变换由
$$
\langle\widehat T,\varphi\rangle
=\langle T,\widehat\varphi\rangle,
\qquad \varphi\in\mathcal S,
$$
定义。定理 5.10 保证右端连续，故定义良好；常数与逆变换按 [NOTATION.md](NOTATION.md) 的约定。

## 5.4 Sobolev 与 PDE 输入

**定理 5.11 (`E`, Sobolev 嵌入).** 对整数 $k\ge0$ 和实数 $s>k+n/2$，Fourier 定义的 Sobolev 空间
$$
H^s(\mathbb R^n)=
\left\{u\in\mathcal S':
\begin{array}{l}
\widehat u\text{ 由一个可测函数 }v\text{ 表示，且}\\[2pt]
\displaystyle\int_{\mathbb R^n}(1+|k|^2)^s|v(k)|^2\,d^nk<\infty
\end{array}\right\}
$$
连续嵌入 $C_b^k(\mathbb R^n)$。

**外部输入边界.** 本书只用该定理判断弱解何时具有经典导数，不展开椭圆或双曲正则性。临界指标 $s=k+n/2$ 不包含在陈述中；完整证明见 [SOURCES.md](SOURCES.md) 的 `E-5.11`。

## 练习

**练习 5.1.** 对例子 5.5，计算 $P_\theta$ 的本征值与归一化本征函数，并说明 $\theta$ 如何改变谱。

**练习 5.2.** 证明 $\delta$ 分布的 Fourier 变换在本书约定下为常数 $1$。

**练习 5.3.** 对例子 5.7，证明 $E_{M_a}(\Delta_1)E_{M_a}(\Delta_2)=E_{M_a}(\Delta_1\cap\Delta_2)$，并写出归一化态 $f$ 的测量概率。
