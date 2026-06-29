# 附录 R：Trace Formula 的项、截断、稳定化和应用接口

收口归一化回指：本附录的核函数、orbital integrals、truncation、stable distributions 和 transfer 比较对 Haar 测度敏感；本书只固定接口，基础 convention 见 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 3、4 节。

## R.1 紧商情形的核和 trace

设 $G$ 为数域 $K$ 上 connected reductive group，$\mathbb A=\mathbb A_K$。先假设
$$
G(K)\backslash G(\mathbb A)
$$
在模中心后紧。令 $f\in C_c^\infty(G(\mathbb A))$。

**定义 R.1.** 积分核定义为
$$
K_f(x,y)=\sum_{\gamma\in G(K)}f(x^{-1}\gamma y).
$$

**命题 R.2.** 在紧商假设下，右卷积算子 $R(f)$ 的 trace 可写为
$$
\operatorname{tr}R(f)=\int_{G(K)\backslash G(\mathbb A)}K_f(x,x)\,dx.
$$

**证明.** 紧商和 $f$ 紧支撑保证核在商上为平滑且可积。积分算子 trace 等于核在对角线上的积分，这是紧空间上平滑核算子的标准性质。$\square$

**命题 R.3.** 在同一假设下，
$$
\operatorname{tr}R(f)
=\sum_{\{\gamma\}}\operatorname{vol}(G_\gamma(K)\backslash G_\gamma(\mathbb A))O_\gamma(f),
$$
其中 $\{\gamma\}$ 遍历 $G(K)$-共轭类，$G_\gamma$ 为 centralizer，
$$
O_\gamma(f)=\int_{G_\gamma(\mathbb A)\backslash G(\mathbb A)}f(x^{-1}\gamma x)\,dx.
$$

**证明草图.** 将 $K_f(x,x)$ 的求和按共轭类分组。固定代表 $\gamma$ 后，$\delta^{-1}\gamma\delta$ 的和等价于 $G_\gamma(K)\backslash G(K)$ 上求和。再把 $G(K)\backslash G(\mathbb A)$ 与该离散商合并，得到 $G_\gamma(K)\backslash G(\mathbb A)$ 的积分，最后分解为 $G_\gamma(K)\backslash G_\gamma(\mathbb A)$ 的体积乘以 orbital integral。测度归一化由 Weil 积分公式控制。$\square$

## R.2 非紧商和 Arthur 截断

实际自守商通常非紧，原因是 proper parabolic subgroups 产生 cusps。

**外部输入定理 R.4（Arthur truncation）.** 存在截断算子 $\Lambda^T$，依赖足够 regular 的参数 $T$，使得截断核
$$
K_f^T(x,y)=\Lambda^T K_f(x,y)
$$
在对角线上可积，并且
$$
J^T(f)=\int_{G(K)\backslash G(\mathbb A)}K_f^T(x,x)\,dx
$$
是 $T$ 的 exponential-polynomial 函数。其常数项定义 Arthur trace formula 的分布 $J(f)$。

**注 R.5.** 截断不是技术细节，而是非紧谱中 Eisenstein series、continuous spectrum 和 proper Levi contributions 同时出现的来源。

## R.3 几何侧：加权轨道积分

**定义 R.6.** Arthur 几何侧按 Levi subgroups 和半单共轭类组织，形式为
$$
J_{\operatorname{geom}}^G(f)
=\sum_M\frac{|W_0^M|}{|W_0^G|}
\sum_{\gamma\in M(K)_{\operatorname{ss}}/\sim}
a^M(\gamma)J_M^G(\gamma,f).
$$
这里 $J_M^G(\gamma,f)$ 是 weighted orbital integral，$a^M(\gamma)$ 是全局体积、Tamagawa 和 centralizer 数据组成的系数。

**外部输入定理 R.7（几何展开）.** 对 Arthur invariant trace formula 的测试函数空间中的 $f$，Arthur 分布 $J(f)$ 有定义 R.6 所示的几何展开。若 $G$ anisotropic modulo center，则 weighted orbital integrals 退化为普通 orbital integrals，并回到命题 R.3。

**命题 R.8.** 普通 orbital integral 是 weighted orbital integral 的 $M=G$ 特例。

**证明.** 当 $M=G$ 时，没有 proper parabolic direction 需要截断，weight function 为 $1$。因此 $J_G^G(\gamma,f)$ 的定义化为
$$
\int_{G_\gamma(\mathbb A)\backslash G(\mathbb A)}f(x^{-1}\gamma x)\,dx.
$$
$\square$

## R.4 谱侧：离散谱、连续谱和 intertwining

**定义 R.9.** 谱侧形式上由 Levi subgroups 的离散自守表示 $\pi$、诱导表示、intertwining operators 和 logarithmic derivatives 组成：
$$
J_{\operatorname{spec}}^G(f)
=\sum_M\frac{|W_0^M|}{|W_0^G|}
\int_{\Pi_{\operatorname{disc}}(M)}
a^M(\pi)J_M^G(\pi,f)\,d\pi.
$$

**外部输入定理 R.10（谱展开）.** Arthur 分布 $J(f)$ 有谱展开 R.9，并满足
$$
J_{\operatorname{geom}}^G(f)=J_{\operatorname{spec}}^G(f).
$$
其中 continuous spectrum 的项由 Eisenstein series 和 normalized intertwining operators 的解析性质控制。

**命题 R.11.** Cuspidal test functions 可消去许多 proper Levi contribution。

**证明草图.** 若 $f$ 的某个局部分量在所有 proper parabolic 的常数项方向上为 cuspidal，则由 unfolding 和 Jacquet module 的消失，来自 proper Levi 的诱导谱项在该位置给出零贡献。几何侧相应地消去与 proper Levi 相关的退化轨道项。完整陈述需固定 strongly cuspidal 或 simple trace formula 的测试函数条件。$\square$

## R.5 Invariant trace formula

**外部输入定理 R.12（Invariant trace formula）.** Arthur 的非不变分布 $J(f)$ 可修正为 invariant distribution
$$
I^G(f)
$$
使其只依赖 $f$ 的 orbital integrals，并具有相应 invariant geometric expansion 和 spectral expansion。

**注 R.13.** Invariant 化的目的，是让 trace formula 能与 endoscopic transfer 比较。未 invariant 化的加权项对共轭和 transfer 不具备足够好的函子性。

## R.6 稳定化和 Endoscopy

**定义 R.14.** 稳定 orbital integral 是普通 orbital integrals 在稳定共轭类内的加权和：
$$
SO_\delta(f)=\sum_{\gamma\in\operatorname{st}(\delta)/\sim}e(G_\gamma)O_\gamma(f).
$$

**外部输入定理 R.15（稳定 trace formula）.** 对 quasi-split $G$，invariant trace formula 可写为
$$
I^G(f)
=S^G(f)+\sum_{H\ne G}\iota(G,H)S^H(f^H),
$$
其中 $H$ 遍历 elliptic endoscopic data，$f^H$ 为 transfer，$S^H$ 为稳定分布。

**命题 R.16.** 稳定化把 endoscopic transfer 从几何侧传递到谱侧。

**证明草图.** 若 $f^H$ 与 $f^G$ orbital-integral matching，则稳定几何侧满足相应等式。Trace formula 给出几何侧等于谱侧。若稳定 characters 能分离 packets，则谱侧分布恒等式推出 packets 或稳定虚表示的转移关系。完整证明依赖局部 transfer、fundamental lemma、稳定 trace formula 和 packet 字符恒等式。$\square$

## R.7 常见应用接口

**外部输入定理 R.17（Base change trace formula interface）.** 对 cyclic base change，比较 twisted trace formula 和普通 trace formula 可构造 `GL(n)` 的 base change lift，并证明非分歧位置的局部参数限制相容。

**外部输入定理 R.18（Endoscopic classification interface）.** 对若干 classical groups，稳定 trace formula 和 twisted endoscopy 给出离散自守谱的 Arthur packet decomposition 和到 `GL(N)` 的标准转移。

**命题 R.19.** Trace formula 应用的逻辑形状是“测试函数匹配 + 分布恒等式 + 线性无关性”。

**证明.** 测试函数匹配提供几何侧等式。Trace formula 把几何侧等式转为谱侧等式。若目标谱侧由可分离的 characters、stable characters 或 pseudo-coefficients 展开，则分布等式迫使相应表示或 packets 的系数相等。该逻辑不依赖具体群；具体定理的困难在于匹配、稳定化和线性无关性的证明。$\square$

## 练习

**练习 R.1.** 在紧商情形，从核 $K_f(x,y)$ 推导几何侧 orbital integral。

**练习 R.2.** 解释 Arthur truncation 为什么和 Eisenstein series 的连续谱相关。

**练习 R.3.** 说明 ordinary orbital integral 与 stable orbital integral 的差别。

**练习 R.4.** 解释 fundamental lemma 在稳定 trace formula 中出现的位置。

**练习 R.5.** 把 base change 的 trace formula 证明框架拆成三步。
