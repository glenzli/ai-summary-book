# 第四章：玻色弦的正则量子化

## 本章目标

本章在 conformal gauge 中量子化玻色弦，推导 oscillator algebra、Virasoro generators、物理态条件、质量公式、临界维数和低能谱。第六章的顶点算子和散射振幅将直接使用本章的 Fock space 与谱公式。

## 依赖前置知识

需要第二章的 conformal gauge 和第三章的 Virasoro algebra。质量公式 convention 见 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md)。

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
比较 Fourier coefficients 得 oscillator commutators；零模部分给出 $[x,p]$。$\square$

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

**证明草图.** 与闭弦相同，但使用区间 $[0,\pi]$ 上的 cosine basis 和开弦正则动量。零模系数 $2\alpha'p^\mu\tau$ 正是为了使总动量为 $p^\mu$。$\square$

## 4.3 Virasoro generators 和 number operators

**定义 4.6（matter Virasoro generators）.** 闭弦 matter Virasoro generators 为
$$
L_m=\frac12\sum_{n\in\mathbb Z}:\alpha_{m-n}\cdot\alpha_n:,
\qquad
\tilde L_m=\frac12\sum_{n\in\mathbb Z}:\tilde\alpha_{m-n}\cdot\tilde\alpha_n:.
$$
开弦只有一份同样形式的 $L_m$。

**命题 4.5（Virasoro algebra）.** $L_m$ 满足 central charge $c=D$ 的 Virasoro algebra：
$$
[L_m,L_n]=(m-n)L_{m+n}
+\frac{D}{12}m(m^2-1)\delta_{m+n,0}.
$$

**证明草图.** 使用 oscillator commutators 和 normal ordering。中心项来自移动 annihilation operators 穿过 creation operators 时产生的 c-number；结果等价于第三章 free boson $T(z)T(w)$ 的 central charge $D$。$\square$

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

## 4.4 物理态条件和质量公式

**定义 4.8（old covariant physical states）.** Old covariant quantization 中闭弦物理态满足
$$
(L_0-a)|\psi\rangle=0,\quad
(\tilde L_0-a)|\psi\rangle=0,
$$
以及
$$
L_n|\psi\rangle=\tilde L_n|\psi\rangle=0\quad(n>0).
$$
开弦相应地只有一份 Virasoro constraints。

**命题 4.7（质量公式）.** 开弦质量公式为
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

## 4.5 临界维数和 no-ghost theorem

**命题 4.9（玻色弦临界条件）.** Covariant bosonic string 的 BRST 一致性要求
$$
D=26,\qquad a=1.
$$

**证明草图.** 第五章将从 ghost CFT 和 BRST charge 证明：matter central charge $D$ 与 ghost central charge $-26$ 相加必须为零，且 $Q_B^2=0$ 同时固定 intercept。$\square$

**外部输入定理 4.10（no-ghost theorem）.** 对 covariant quantized bosonic string，当 $D=26$ 且 $a=1$ 时，物理态空间在商去 null states 后具有非负范数，并与 light-cone quantization 的 transverse Fock space 同构。

**注 4.11.** 本书不证明 no-ghost theorem。其作用是保证 covariant 约束没有留下负范数物理激发。BRST 章节会给出更系统的 cohomological 表述。

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

**证明草图.** 约束 $T_{++}=T_{--}=0$ 在 $p^+\ne0$ 时可逐模解出 $\alpha_n^-$ 和 $\tilde\alpha_n^-$，其右端由 transverse oscillators 的二次组合给出。$X^+$ 被 gauge 固定，$X^-$ 被约束解出，故只剩 $D-2$ 个横向方向。$\square$

**命题 4.16（Lorentz algebra closure）.** 玻色弦 light-cone quantization 中，量子 Lorentz algebra 无 anomaly 当且仅当
$$
D=26,\qquad a=1.
$$

**证明草图.** 需要检查含有 $J^{-i}$ 的 Lorentz generators commutators。Normal ordering 产生 anomalous terms；它们同时消失要求 $D=26$ 与 intercept $a=1$。这是 light-cone 量子化给出的临界条件，与 BRST nilpotency 条件一致。$\square$

## 本章小结

正则量子化把弦变成无限多个 oscillator。Virasoro constraints 选择物理态，质量公式来自 $L_0$ 条件，level matching 来自左右闭弦约束相容。临界维数和 no-ghost theorem 保证 covariant quantization 的一致性。

## 练习

**练习 4.1.** 从闭弦模展开推导 oscillator commutators。

**练习 4.2.** 推导开弦质量公式。

**练习 4.3.** 对闭弦第一激发层推导横向条件和 gauge equivalence。

**练习 4.4.** 在 light-cone gauge 中计数玻色弦的横向 oscillator 数，并解释为什么没有负范数振子。

