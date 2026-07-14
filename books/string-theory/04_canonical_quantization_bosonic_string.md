# 第四章：玻色弦的正则量子化

经典模展开把弦写成无穷多个简谐振子，但“把每个模量子化”还不足以得到物理态：
协变 Fock pairing 含有负范数方向，$L_0$ 又把连续动量限制在质量壳上，而普通
Schwartz 波包不可能以非零函数的方式精确支撑在余维一的壳面。这里先由正则辛形式
逐项推出开、闭弦振子的对易关系，再区分离壳公共算符域、固定动量广义态和质量壳
直积分，随后施加 Virasoro constraints、求出质量谱，并用 no-ghost 与 light-cone
量子化说明负范数如何被消除。所用经典解来自第二章，Virasoro/OPE 语言来自第三章；
$\alpha'$、截距和质量公式均遵循 [全书归一化表](NORMALIZATION_TABLE.md)。

## 4.1 闭弦模展开和正则量子化

**定义 4.1（闭弦模展开）.** 闭弦的经典解写为
$$
X^\mu(\tau,\sigma)=x^\mu+\alpha'p^\mu\tau
+i\sqrt{\frac{\alpha'}2}\sum_{n\ne0}\frac1n
\left(\alpha_n^\mu e^{-in(\tau-\sigma)}
+\tilde\alpha_n^\mu e^{-in(\tau+\sigma)}\right).
$$
现实条件要求
$$
(\alpha_n^\mu)^\dagger=\alpha_{-n}^\mu,\qquad
(\tilde\alpha_n^\mu)^\dagger=\tilde\alpha_{-n}^\mu.
$$

**定义 4.2（canonical commutator）.** 正则动量为
$$
P_\mu(\tau,\sigma)=\frac{1}{2\pi\alpha'}\partial_\tau X_\mu.
$$
量子化条件为
$$
[X^\mu(\tau,\sigma),P_\nu(\tau,\sigma')]
=i\delta^\mu_{\ \nu}\delta_{2\pi}(\sigma-\sigma').
$$

**命题 4.3（oscillator commutators）.** Oscillator commutators 为
$$
[x^\mu,p^\nu]=i\eta^{\mu\nu},
$$
$$
[\alpha_m^\mu,\alpha_n^\nu]=m\delta_{m+n,0}\eta^{\mu\nu},
\qquad
[\tilde\alpha_m^\mu,\tilde\alpha_n^\nu]=m\delta_{m+n,0}\eta^{\mu\nu},
$$
且左右 movers 互相对易。

**证明.** 将模展开代入 equal-time commutator，并使用周期 delta function
$$
\delta_{2\pi}(\sigma-\sigma')=\frac1{2\pi}\sum_{n\in\mathbb Z}e^{in(\sigma-\sigma')}.
$$
为看清每个系数，先把正则动量写成
$$
P_\mu(\tau,\sigma)=\frac{p_\mu}{2\pi}
+\frac1{2\pi\sqrt{2\alpha'}}\sum_{n\ne0}
\left(\alpha_{n,\mu}e^{-in(\tau-\sigma)}
+\widetilde\alpha_{n,\mu}e^{-in(\tau+\sigma)}\right).
$$
在固定 $\tau$ 的经典解空间上，正则辛形式为
$$
\begin{aligned}
\Omega_{\mathrm{cl}}
&=\int_0^{2\pi}d\sigma\,\delta P_\mu\wedge\delta X^\mu\\
&=\delta p_\mu\wedge\delta x^\mu
+i\sum_{n=1}^{\infty}\frac1n
\left(
\delta\alpha_{-n,\mu}\wedge\delta\alpha_n^\mu
+\delta\widetilde\alpha_{-n,\mu}\wedge
 \delta\widetilde\alpha_n^\mu
\right).
\end{aligned}
$$
这里使用
$\int_0^{2\pi}e^{i(m+n)\sigma}d\sigma=2\pi\delta_{m+n,0}$；左右交叉项在
$\delta P\wedge\delta X$ 中成对抵消。反演 $\Omega_{\mathrm{cl}}$ 得
$$
\{x^\mu,p^\nu\}_{\mathrm{PB}}=\eta^{\mu\nu},\qquad
\{\alpha_m^\mu,\alpha_n^\nu\}_{\mathrm{PB}}
=-im\delta_{m+n,0}\eta^{\mu\nu},
$$
tilde 部分相同，左右 Poisson brackets 为零。按
$[A,B]=i\{A,B\}_{\mathrm{PB}}$ 量子化即得命题中的关系。

最后直接代回可检查 delta function 的归一化。零模贡献为
$i\eta^{\mu\nu}/(2\pi)$，每个 chiral sector 的非零模贡献为
$$
\frac{i\eta^{\mu\nu}}{4\pi}
\sum_{n\ne0}e^{in(\sigma-\sigma')}.
$$
两份相加并补上 $n=0$ 项，恰为
$i\eta^{\mu\nu}\delta_{2\pi}(\sigma-\sigma')$。$\square$

## 4.2 开弦模展开

**定义 4.4（开弦 Neumann 模展开）.** 所有方向取 Neumann 条件时，
$$
X^\mu(\tau,\sigma)=x^\mu+2\alpha'p^\mu\tau
+i\sqrt{2\alpha'}\sum_{n\ne0}\frac{\alpha_n^\mu}{n}
e^{-in\tau}\cos n\sigma,
\qquad 0\le\sigma\le\pi.
$$

**命题 4.4A（开弦 oscillator algebra）.** 开弦 modes 满足
$$
[x^\mu,p^\nu]=i\eta^{\mu\nu},
\qquad
[\alpha_m^\mu,\alpha_n^\nu]=m\delta_{m+n,0}\eta^{\mu\nu}.
$$

**证明.** Neumann 边界条件对应区间上的 delta kernel
$$
\delta_N(\sigma,\sigma')=\frac1\pi+\frac2\pi
\sum_{n=1}^\infty\cos(n\sigma)\cos(n\sigma').
$$
将模展开及
$$
P_\mu=\frac{p_\mu}{\pi}
+\frac1{\pi\sqrt{2\alpha'}}
\sum_{n\ne0}\alpha_{n,\mu}e^{-in\tau}\cos n\sigma
$$
代入正则辛形式，并使用
$\int_0^\pi\cos(m\sigma)\cos(n\sigma)d\sigma=(\pi/2)\delta_{mn}$（$m,n>0$），得到
$$
\Omega_{\mathrm{op}}
=\delta p_\mu\wedge\delta x^\mu
+i\sum_{n=1}^{\infty}\frac1n
\delta\alpha_{-n,\mu}\wedge\delta\alpha_n^\mu.
$$
反演后量子化给出
$$
[\alpha_m^\mu,\alpha_n^\nu]
=m\delta_{m+n,0}\eta^{\mu\nu}.
$$
相应 equal-time commutator 的零模为 $i\eta^{\mu\nu}/\pi$，非零模为
$$
\frac{2i\eta^{\mu\nu}}\pi\sum_{n=1}^{\infty}
\cos(n\sigma)\cos(n\sigma'),
$$
两者之和正是 $i\eta^{\mu\nu}\delta_N(\sigma,\sigma')$。此外，
$\int_0^\pi P^\mu d\sigma=p^\mu$ 验证了零模系数
$2\alpha'p^\mu\tau$。$\square$

**定义 4.4B（离壳域与固定动量广义态）.** 令
$\mathcal F_{\mathrm{osc,fin}}$ 为由有限个 creation operators 作用在形式真空上
生成的代数振子空间。离壳 Schwartz 波包域定义为
$$
\mathcal D_{\mathrm{off}}
=\mathcal S(\mathbb R^D)
\otimes_{\mathrm{alg}}\mathcal F_{\mathrm{osc,fin}},
$$
闭弦含左右两份振子空间。动量 $p^\mu$ 在第一因子上作乘法，$x^\mu$ 作
$i\partial/\partial p_\mu$；振子多项式与 matter Virasoro modes 都先定义在这个
共同不变域上。

对固定 $p\in\mathbb R^D$，形式动量本征态 $|p\rangle$ 是
$\mathcal S'(\mathbb R^D)$ 中的广义态，而不是 $L^2$ 或 Schwartz 向量。相应有限
激发 module 记为
$$
\mathcal F_{\mathrm{fin}}(p)
=\operatorname{span}\left\{
\alpha_{-n_1}^{\mu_1}\cdots\alpha_{-n_r}^{\mu_r}|p\rangle:
r<\infty,\ n_i>0
\right\}.
$$
闭弦再与一份有限激发的 tilde module 作代数张量积。由于 target metric 为
Lorentzian，这些协变 Fock modules 带不定 pairing；在应用 no-ghost theorem 前，
它们都不是已经完成的物理 Hilbert space。

**定义 4.4C（质量壳广义态与直积分）.** 设 $\mathcal F_r(p)$ 是开弦 oscillator
level $r$ 的 fixed-momentum fiber，并令
$$
\mathscr P_r^{\mathrm{op}}
=\left\{p\in\mathbb R^D:\alpha'p^2+r-a=0\right\}.
$$
精确壳上态可按两种标准且在选定测度、测试函数配对后互相对应的方式处理：在
$\mathcal S'(\mathbb R^D)\otimes\mathcal F_{\mathrm{osc,fin}}$ 中使用支撑于
$\mathscr P_r^{\mathrm{op}}$ 的分布，或在选定壳分支及壳上测度 $d\mu_r$ 后使用
$$
\int_{\mathscr P_r^{\mathrm{op}}}^{\oplus}
d\mu_r(p)\,\mathcal F_r(p).
$$
闭弦逐纤维同时施加
$\alpha'p^2/4+r-a=0$、$\alpha'p^2/4+\widetilde r-a=0$，因而还要求
$r=\widetilde r$。零动量纤维可以单独定义，但不自动满足后述非零动量 no-ghost
定理的假设。

不能把上述壳上空间替换为 $\mathcal D_{\mathrm{off}}$ 中乘法算符的 kernel。事实上，
若 $f\in\mathcal S(\mathbb R^D)$ 满足
$(\alpha'p^2+r-a)f(p)=0$，则 $f$ 在质量壳补集上为零；该补集稠密，连续性又迫使
$f$ 在壳上也为零。因此这个 Schwartz kernel 只有零向量，而分布或质量壳直积分
才承载非零物理波包。

**引理 4.4D（Virasoro 和的局部有限性）.** 对每个固定 $m\in\mathbb Z$，
正规序二次和
$$
\frac12\sum_{n\in\mathbb Z}:\alpha_{m-n}\cdot\alpha_n:
$$
在 $\mathcal D_{\mathrm{off}}$ 的每个向量上只有有限多个非零项，并把
$\mathcal D_{\mathrm{off}}$ 映到自身；同一公式也逐纤维作用在
$\mathcal F_{\mathrm{fin}}(p)$ 上。

**证明.** 给定有限激发态，含 annihilation operator 的项只有当其 mode 出现在该态中
时才非零，故这类项有限。两个因子均为 creation operators 时，$n<0$ 且
$m-n<0$；对固定 $m$，满足这两个不等式的整数 $n$ 也只有有限多个。零模只在
Schwartz 动量变量上作乘法。因此该和逐向量有限且保持定义域。$\square$

## 4.3 Virasoro generators 和 number operators

**定义 4.5（matter Virasoro generators）.** 为把 center-of-mass momentum 纳入
统一 mode 求和，闭弦约定
$$
\alpha_0^\mu=\widetilde\alpha_0^\mu
=\sqrt{\frac{\alpha'}2}\,p^\mu,
$$
开弦约定
$$
\alpha_0^\mu=\sqrt{2\alpha'}\,p^\mu.
$$
在定义 4.4B 的共同定义域上，闭弦 matter Virasoro generators 为
$$
L_m=\frac12\sum_{n\in\mathbb Z}:\alpha_{m-n}\cdot\alpha_n:,
\qquad
\tilde L_m=\frac12\sum_{n\in\mathbb Z}:\tilde\alpha_{m-n}\cdot\tilde\alpha_n:.
$$
开弦只有一份同样形式的 $L_m$。

**命题 4.6（Virasoro algebra）.** 作为 $\mathcal D_{\mathrm{off}}$ 上以及每个
固定动量有限激发 fiber 上的算符恒等式，$L_m$ 满足 central charge $c=D$ 的
Virasoro algebra：
$$
[L_m,L_n]=(m-n)L_{m+n}
+\frac{D}{12}m(m^2-1)\delta_{m+n,0}.
$$

**证明.** 首先由振子交换关系直接得到
$$
[L_m,\alpha_n^\mu]=-n\alpha_{m+n}^\mu.
$$
因此 $[L_m,L_n]-(m-n)L_{m+n}$ 与所有振子对易，只能是中心元；由 mode number 可知它仅在 $m+n=0$ 时非零。取零动量 Fock vacuum，并令 $m>0$。由于 $L_m|0\rangle=0$，中心项可由
$$
\langle0|L_mL_{-m}|0\rangle
$$
计算。只有 $L_{-m}$ 中两个 creation operators 的部分有贡献，Wick 配对给出
$$
\langle0|L_mL_{-m}|0\rangle
=\frac D2\sum_{r=1}^{m-1}r(m-r)
=\frac D{12}m(m^2-1).
$$
这里使用了
$$
\sum_{r=1}^{m-1}r(m-r)=\frac{m(m^2-1)}6.
$$
所有中间和在引理 4.4D 的意义下逐向量有限。故中心元为命题所示；对负 $m$ 的
公式由交换子的反对称性得到。该等式尚未声称 $L_m$ 在某个 Hilbert completion 上
自伴或其闭包唯一。$\square$

**定义 4.7A（number operators）.** 闭弦 number operators 定义为
$$
N=\sum_{n=1}^\infty \alpha_{-n}\cdot\alpha_n,
\qquad
\tilde N=\sum_{n=1}^\infty \tilde\alpha_{-n}\cdot\tilde\alpha_n.
$$
开弦定义为同样的单份 $N$。Normal ordering 后，
$$
L_0^{\mathrm{closed}}=\frac{\alpha'}4p^2+N,
\qquad
\tilde L_0^{\mathrm{closed}}=\frac{\alpha'}4p^2+\tilde N,
$$
开弦为
$$
L_0^{\mathrm{open}}=\alpha'p^2+N.
$$

**注 4.7B（intercept）.** 物理态条件实际使用 $L_0-a$。玻色弦临界量子化中 $a=1$。该常数可由 light-cone zero-point energy 或 BRST nilpotency 固定。

**计算 4.7C（截距的正规化边界）.** Light-cone gauge 中每个横向 boson 的真空能
形式上为 $(1/2)\sum_{n\ge1}n$。采用 exponential cutoff，
$$
\sum_{n=1}^{\infty}ne^{-\varepsilon n}
=\frac{e^{-\varepsilon}}{(1-e^{-\varepsilon})^2}
=\frac1{\varepsilon^2}-\frac1{12}+O(\varepsilon^2).
$$
减去局部发散项 $\varepsilon^{-2}$ 后，有限部分与 $\zeta(-1)=-1/12$ 相同。因此
$$
E_0^{\perp}=-\frac{D-2}{24},
\qquad a=-E_0^{\perp}=\frac{D-2}{24}.
$$
这是保持世界面平移与 mode grading 的正规序方案；有限 counterterm 会移动 $a$，
但 Lorentz algebra closure 或 BRST nilpotency 随后固定临界方案中的 $a=1$，所以
$a$ 不是可任意调节的物理参数。

## 4.4 物理态条件和质量公式

**定义 4.8（old covariant physical states）.** Old covariant quantization 中，先在
固定壳上动量 $p$ 的广义 fiber $\mathcal F_{\mathrm{fin}}(p)$ 内取满足下列条件的
向量：
$$
(L_0-a)|\psi\rangle=0,\quad
(\tilde L_0-a)|\psi\rangle=0,
$$
以及
$$
L_n|\psi\rangle=\tilde L_n|\psi\rangle=0\quad(n>0).
$$
开弦相应地只有一份 Virasoro constraints；最后还要商去其中的 null states。这里
只施加 $n>0$ 的正频约束，负频 modes 是其形式伴随，并非零算符。物理 Hilbert
completion 的正定性依赖外部输入定理 4.10。具有动量展宽的物理态是定义 4.4C
所述质量壳直积分中的可测截面，约束逐纤维成立；它不是
$\mathcal D_{\mathrm{off}}$ 中满足同一乘法方程的 Schwartz 函数。

**命题 4.9（质量公式）.** 开弦质量公式为
$$
M^2=\frac1{\alpha'}(N-a).
$$
闭弦质量公式为
$$
M^2=\frac4{\alpha'}(N-a)=\frac4{\alpha'}(\tilde N-a),
\qquad N=\tilde N.
$$
等价地，在 level matching 已成立时，
$$
M^2=\frac2{\alpha'}(N+\tilde N-2a).
$$

**证明.** 使用 $M^2=-p^2$。开弦由
$$
0=(L_0-a)|\psi\rangle=(\alpha'p^2+N-a)|\psi\rangle
$$
得
$$
M^2=\frac1{\alpha'}(N-a).
$$
闭弦由左右两式
$$
0=\frac{\alpha'}4p^2+N-a,\qquad
0=\frac{\alpha'}4p^2+\tilde N-a
$$
得到两个相同的 $M^2$ 表达式，并要求
$$
N=\tilde N.
$$
$\square$

## 4.5 协变量子化的正定性

Virasoro 约束从不定 Fock module 中选出候选物理态，但仅靠定义 4.8 还看不出取商后
是否仍有 timelike 负范数。下面的 no-ghost theorem 精确回答非零动量纤维上的这个
问题；它把 $D=26$、$a=1$ 作为假设，而不是从正定性陈述中暗中推出这两个数值。

**后续等价表述（第 5 章证明）.** 加入标准 $bc$ ghost module 后，命题 5.9 将由
BRST current 的 OPE 独立证明：正规序 $Q_B$ 的 nilpotency 当且仅当
$D=26$、$a=1$。该结果在这里不进入证明链，也不作为 no-ghost theorem 的证明；
本章稍后由 light-cone Poincare algebra 的 closure 给出另一项独立一致性条件。

**外部输入定理 4.10（非零动量 no-ghost theorem）.** 固定平直临界玻色弦
$D=26$、$a=1$ 和非零 on-shell target momentum $p^\mu\ne0$（在所选 light-cone
frame 中取 $p^+\ne0$）。结论必须按下列物理商空间理解。

1. 对开弦，令 $\mathcal P_{\mathrm{op}}(p)$ 为满足
   $(L_0-1)|\psi\rangle=0$ 与 $L_n|\psi\rangle=0$（$n>0$）的 fixed-momentum
   old-covariant states，并令 $\mathcal N_{\mathrm{op}}(p)$ 为其中的 null/spurious
   subspace。诱导 pairing 在
   $\mathcal P_{\mathrm{op}}(p)/\mathcal N_{\mathrm{op}}(p)$ 上正定，且该商与同一
   mass level 的 transverse light-cone Fock space 同构。
2. 对闭弦，必须同时施加左右 Virasoro constraints、两个 intercept conditions 与
   level matching，再商去左右 null subspaces。所得物理商与 level-matched 左右
   transverse Fock spaces 的张量积同构；这不是把开弦陈述在未施加 level matching
   的完整张量积上直接重复一次。

在第五章的 ghost-number convention 中，相应 BRST 同构只针对指定的 cohomology
商。开弦传播外态使用
$$
H^1_{\rm rel}(Q_B;p),
\qquad
\mathcal C_{\rm rel}(p)
=\ker b_0\cap\ker L_0^{\rm tot},
$$
其中 cohomology 的分母是在同一 relative subcomplex 内的 $Q_B$-exact states。闭弦
未积分传播外态使用总 ghost number $2$ 的 semi-relative cohomology
$$
H^2_{\rm sr}(Q_B^{\rm cl};p),
\qquad
\mathcal C_{\rm sr}(p)
=\ker b_0^-\cap\ker L_0^-;
$$
更强的 relative complex 还分别施加 $b_0=\widetilde b_0=0$，不能不加说明地与
semi-relative complex 互换。这里 $L_0^-=0$ 正是 level matching。定理不声称
unrestricted absolute BRST cohomology、其他 ghost numbers 或不同 zero-mode 商都等于
transverse spectrum。

**边界 4.10A（零动量 cohomology 另列）.** $p^\mu=0$ 不在定理 4.10 的假设内。
标准 filtration/contracting-homotopy 证明需要除以非零 light-cone momentum；在
$p=0$ 时该步骤失效。开弦一级已给出直接反例：
$$
|\psi\rangle=\epsilon_\mu\alpha_{-1}^\mu|0;p=0\rangle
$$
满足 $N=1$ 的质量壳与全部正模约束，但通常生成 gauge/null 方向的
$L_{-1}|0;p\rangle\propto p\!\cdot\!\alpha_{-1}|0;p\rangle$ 在 $p=0$ 时消失，故
$\epsilon_\mu$ 的 timelike 负范数方向没有被该商去掉。BRST 语言中，relative、
semi-relative 与 absolute cohomology 也可出现不属于传播 transverse oscillator
spectrum 的额外 classes，gauge parameters 与 global/zero-mode states 须单独分析。
因此定理 4.10 不能用来宣称零动量 cohomology 已被分类。

**注 4.11.** 本书不证明 no-ghost theorem，也不计算零动量 exceptional cohomology。
定理 4.10 的作用是保证指定 physical quotient 没有留下负范数传播激发；第五章给出
BRST complex 和 nilpotency，但 $Q_B^2=0$ 本身不等于上述 cohomology theorem。
同样，条件 $D=26,a=1$ 单独并不构成所有 genus 的 Polyakov measure、unitarity 与
背景稳定性的充分条件。

## 4.6 低能谱

**例 4.12（开弦前两层）.** 玻色开弦 $a=1$。基态 $N=0$ 有
$$
M^2=-\frac1{\alpha'}
$$
为 tachyon。第一激发
$$
\zeta_\mu\alpha_{-1}^\mu|0;k\rangle
$$
有 $N=1$，故 $M^2=0$。Virasoro constraint $L_1|\psi\rangle=0$ 给出
$$
k\cdot\zeta=0,
$$
并且 null state 实现 $\zeta_\mu\sim\zeta_\mu+\lambda k_\mu$。

**例 4.13（闭弦第一激发层）.** 闭弦第一激发层
$$
\epsilon_{\mu\nu}\alpha_{-1}^\mu\tilde\alpha_{-1}^\nu|0;k\rangle
$$
有 $N=\tilde N=1$，故 $M^2=0$。极化张量分解为
$$
\epsilon_{\mu\nu}
=\epsilon_{(\mu\nu)}^{\mathrm{traceless}}
+\epsilon_{[\mu\nu]}
+\frac1D\eta_{\mu\nu}\epsilon^\rho_{\ \rho},
$$
分别对应 graviton、Kalb-Ramond field 和 dilaton。

## 4.7 Light-cone quantization 对照

**定义 4.14（light-cone coordinates）.** 定义
$$
X^\pm=\frac1{\sqrt2}(X^0\pm X^{D-1}).
$$
Light-cone gauge 固定
$$
X^+(\tau,\sigma)=x^+ + \alpha'p^+\tau
$$
在闭弦 convention 下成立。开弦零模系数相应取 $2\alpha'p^+\tau$。

**命题 4.15（light-cone physical oscillators）.** Light-cone gauge 中，Virasoro constraints 解出 $X^-$ 的非零模，独立振子只剩 transverse oscillators
$$
\alpha_n^i,\qquad i=1,\ldots,D-2.
$$
因此 physical Fock space 无负范数振子。

**证明.** Gauge 条件使 $\alpha_n^+=\tilde\alpha_n^+=0$（$n\ne0$），并固定 $\alpha_0^+=\tilde\alpha_0^+=\sqrt{\alpha'/2}\,p^+\ne0$。以
$$
\alpha_m\cdot\alpha_n
=-\alpha_m^+\alpha_n^- -\alpha_m^-\alpha_n^+
+\alpha_m^i\alpha_n^i
$$
代入 $L_n=0$，可逐模唯一解得
$$
\alpha_n^-
=\frac1{\sqrt{2\alpha'}p^+}
\sum_{m\in\mathbb Z}:\alpha_{n-m}^i\alpha_m^i:
$$
（$n=0$ 时另含正规序常数）；闭弦的 tilde 部分同理。因此 $X^+$ 已由 gauge 固定，$X^-$ 由约束决定，独立振子恰为 $D-2$ 个横向方向。$\square$

**外部输入定理 4.16（Lorentz algebra closure）.** 玻色弦 light-cone
quantization 的标准正规序 Poincare generators 在共同有限激发域上闭合且无 anomaly
当且仅当
$$
D=26,\qquad a=1.
$$

**证明路线（外部输入）.** 不含 $J^{-i}$ 的交换关系在横向 Fock space 上直接闭合；潜在异常只出现在 $[J^{-i},J^{-j}]$。把约束解出的 $\alpha_n^-$ 代入 Lorentz generators 并作 creation-annihilation normal ordering 后，异常部分在一个公共非零 prefactor 之外为
$$
\sum_{n=1}^{\infty}
\left[
\left(\frac{D-2}{24}-1\right)n
+\frac1n\left(a-\frac{D-2}{24}\right)
\right]
\left(\alpha_{-n}^i\alpha_n^j-\alpha_{-n}^j\alpha_n^i\right).
$$
不同 $n$ 的双线性振子算子线性独立，故异常消失要求方括号对每个 $n$ 都为零。其 $n$ 与 $n^{-1}$ 系数分别给出
$$
\frac{D-2}{24}=1,
\qquad
a=\frac{D-2}{24},
$$
即 $D=26$、$a=1$。反之这两个值使显示的异常和逐项消失。完整 calculation 还需
展开 $J^{-i}$ 的零模项并验证其余 Poincare 交换关系；本书引用该标准 calculation，
本段只展示决定临界条件的唯一异常项，不以路线代替完整证明。

正则量子化最终给出的不是“Schwartz 函数上的质量壳 kernel”，而是离壳算符域之上的
固定动量广义 fibers 及其壳上直积分。Virasoro 正模逐纤维选择候选态，$L_0$ 给出
质量公式，闭弦左右条件给出 level matching；no-ghost theorem 再把非零动量的协变
商空间识别为横向 Fock space。Light-cone 计算说明这些横向振子在
$D=26,a=1$ 时才能承载无 anomaly 的 Poincare 作用，第 5 章会从 BRST nilpotency
重新得到同一临界条件。

## 练习

**练习 4.1.** 从闭弦模展开推导 oscillator commutators。

**练习 4.2.** 推导开弦质量公式。

**练习 4.3.** 对闭弦第一激发层推导横向条件和 gauge equivalence。

**练习 4.4.** 在 light-cone gauge 中计数玻色弦的横向 oscillator 数，并解释为什么没有负范数振子。
