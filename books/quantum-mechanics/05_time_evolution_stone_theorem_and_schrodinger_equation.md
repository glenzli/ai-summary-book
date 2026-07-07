# 第五章：时间演化、Stone 定理与 Schrodinger 方程

## 本章目标

本章给出封闭量子系统的时间演化公设，说明强连续酉群、自伴 Hamiltonian 和 Schrodinger 方程之间的关系。

## 依赖前置知识

需要 Hilbert 空间、有界算子、自伴算子和指数函数演算。

## 5.1 酉演化

**定义 5.1.** 有界算子 $U$ 称为酉算子，若
$$
U^*U=I=UU^*.
$$

**命题 5.2.** 酉算子保持内积和转移概率。

**证明.** 对任意 $\psi,\phi$，
$$
\langle U\psi,U\phi\rangle=\langle\psi,U^*U\phi\rangle=\langle\psi,\phi\rangle.
$$
取绝对值平方即得转移概率保持。$\square$

**公设 5.3（封闭系统演化）.** 封闭系统的时间演化由强连续一参数酉群 $U(t)$ 给出：
$$
U(0)=I,\qquad U(t+s)=U(t)U(s),
$$
且对每个 $\psi$，$t\mapsto U(t)\psi$ 连续。

## 5.2 Stone 定理与 Hamiltonian

**外部输入定理 5.4（Stone 定理，QM-EXT-2）.** 强连续一参数酉群唯一写成
$$
U(t)=e^{-itH},
$$
其中 $H$ 是自伴算子。

**定义 5.5.** Stone 定理中的自伴算子 $H$ 称为 Hamiltonian。若 $\psi_0\in\mathcal D(H)$，令 $\psi(t)=U(t)\psi_0$，则
$$
i\frac{d}{dt}\psi(t)=H\psi(t)
$$
称为 Schrodinger 方程。

**命题 5.6.** 若 $H$ 为有界自伴算子，则 $U(t)=e^{-itH}$ 满足 Schrodinger 方程。

**证明.** 有界算子的指数由范数收敛级数定义：
$$
e^{-itH}=\sum_{n=0}^\infty\frac{(-itH)^n}{n!}.
$$
逐项求导在算子范数中合法，得
$$
\frac{d}{dt}e^{-itH}=-iHe^{-itH}.
$$
于是 $\psi(t)=e^{-itH}\psi_0$ 满足 $i\dot\psi(t)=H\psi(t)$。$\square$

## 5.3 Heisenberg 图像

**定义 5.7.** Schrodinger 图像中态随时间变：
$$
\psi(t)=U(t)\psi(0).
$$
Heisenberg 图像中可观测量随时间变：
$$
A_H(t)=U(t)^*AU(t).
$$

**命题 5.8.** 若 $H,A$ 有界自伴，则
$$
\frac{d}{dt}A_H(t)=i[H,A_H(t)].
$$

**证明.** 由 $U'(t)=-iHU(t)$ 与 $(U^*)'(t)=iU(t)^*H$，
$$
\frac{d}{dt}U^*AU=iU^*HAU-iU^*AHU=i[H,U^*AU],
$$
其中 $H$ 与 $U(t)$ 交换，因为 $U(t)$ 是 $H$ 的函数。$\square$

## 5.4 谱表示中的相位演化

**命题 5.9.** 若有限维 Hamiltonian 有谱分解
$$
H=\sum_rE_rP_r,
$$
则
$$
U(t)=e^{-itH}=\sum_re^{-itE_r}P_r.
$$

**证明.** 对指数函数使用有限维函数演算。也可直接验证右边满足
$$
U(0)=I,\qquad \frac d{dt}U(t)=-iHU(t),
$$
并由线性常微分方程解的唯一性得到。$\square$

**推论 5.10.** 若初态分解为
$$
\psi=\sum_r\psi_r,\qquad \psi_r=P_r\psi,
$$
则
$$
\psi(t)=\sum_re^{-itE_r}\psi_r.
$$
能量本征分量的概率 $\|\psi_r\|^2$ 不随时间变。

## 本章小结

时间演化由强连续酉群给出。Stone 定理把这种群等价地编码为自伴 Hamiltonian。Schrodinger 图像演化态，Heisenberg 图像演化可观测量，二者给出相同期望值。

## 练习

**练习 5.1.** 证明酉演化保持态的归一化。

**练习 5.2.** 若 $[A,H]=0$ 且算子有界，证明 $\langle A\rangle_{\psi(t)}$ 与时间无关。
