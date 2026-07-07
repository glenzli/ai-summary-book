# 附录 C：Lie algebras、Kac-Moody algebras 和 anomaly polynomials

## 目标

本附录提供 heterotic strings 和 anomaly cancellation 所需的代数接口。

## C.1 Kac-Moody algebra

**定义 C.1.** Affine current modes 满足
$$
[J_m^a,J_n^b]=if^{ab}_{\ \ c}J_{m+n}^c+km\delta^{ab}\delta_{m+n,0}.
$$

等价的 OPE 形式为
$$
J^a(z)J^b(w)\sim
\frac{k\delta^{ab}}{(z-w)^2}
+\frac{if^{ab}_{\ \ c}J^c(w)}{z-w}.
$$

若 finite-dimensional Lie algebra $\mathfrak g$ 的 dual Coxeter number 为 $h^\vee$，Sugawara stress tensor 为
$$
T(z)=\frac1{2(k+h^\vee)}:J^aJ^a:(z),
$$
central charge 为
$$
c=\frac{k\dim\mathfrak g}{k+h^\vee}.
$$
Heterotic string 中 level-one simply-laced current algebras 与 even unimodular lattice construction 兼容。

## C.1A Trace conventions

异常公式依赖 trace convention。本书使用：

1. $\operatorname{tr}$：fundamental 或 minimal defining representation 的 trace；
2. $\operatorname{Tr}$：adjoint representation 的 trace；
3. 不同群之间的换算必须在使用处声明。

例如 $SO(32)$ 与 $E_8\times E_8$ 的十维 anomaly cancellation 使用特定 trace identities。正文只使用其因式分解结论，不把完整 group-theoretic verification 展开为主线证明。

## C.2 Anomaly polynomial

**定义 C.2.** Anomaly polynomial 是比时空维数高两阶的 characteristic class，用 descent procedure 产生 anomaly variation。

**注 C.3.** 十维 Green-Schwarz mechanism 要求 anomaly polynomial 因式分解为可由 $B_2$ 变换抵消的形式。

## C.3 Descent formalism

若 $I_{2n+2}$ 是 closed gauge-invariant polynomial，并可局部写为
$$
I_{2n+2}=dI_{2n+1},
$$
则 gauge variation 给出
$$
\delta I_{2n+1}=dI_{2n}^{(1)}.
$$
量子有效作用的 anomaly 具有形式
$$
\delta\Gamma=2\pi i\int_{M_{2n}}I_{2n}^{(1)}.
$$

Green-Schwarz cancellation 的抽象结构为：
$$
I_{12}=X_4X_8,\qquad
H_3=dB_2-\omega_3^{YM}+\omega_3^L,\qquad
dH_3=X_4.
$$
令 $B_2$ 在 gauge/Lorentz transformation 下非平凡变换，则
$$
\delta\int B_2\wedge X_8
$$
可抵消由 descent 给出的 anomaly variation。

## C.4 Common characteristic forms

常用四形式为
$$
X_4=\operatorname{tr}R^2-\operatorname{tr}F^2
$$
或其归一化变体。十维公式中的 $X_8$ 是 $R$ 与 $F$ 的八形式多项式；不同教材会吸收 $(2\pi)$ 与 trace normalization。使用时必须以 anomaly polynomial 的整体归一化为准。

本附录不给出所有群的 trace identity 证明。它们属于 Lie algebra invariant theory；本书只需要明确哪些步骤是外部输入。
