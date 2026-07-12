# 第二十七章：标准精算例题：Gaussian 波包、自旋进动与 Rabi 振荡

## 本章目标

本章把前面建立的形式主义用于三个可完整计算的模型：自由 Gaussian 波包、自旋在恒定磁场中的进动、二能级系统的 Rabi 振荡。三者分别代表连续变量、有限维自旋和受驱跃迁，是正式量子力学教材中最常用的检验例题。

## 依赖前置知识

需要 Fourier 变换、自由粒子传播子、Pauli 矩阵、酉演化和二能级系统。

## 27.1 自由 Gaussian 波包

**设定 27.1.** 在 $L^2(\mathbb R)$ 上取 $m>0$，一维自由 Hamiltonian 为
$$
H=\frac{P^2}{2m}.
$$
取初态
$$
\psi_0(x)=\left(\frac1{\pi\sigma^2}\right)^{1/4}
\exp\left(-\frac{x^2}{2\sigma^2}+ik_0x\right),
\qquad \sigma>0,\quad k_0\in\mathbb R.
$$

**命题 27.2.** 自由演化后的波函数为
$$
\psi_t(x)=\left(\frac1{\pi\sigma^2}\right)^{1/4}
\left(1+\frac{it}{m\sigma^2}\right)^{-1/2}
\exp\left[
-\frac{(x-k_0t/m)^2}{2\sigma^2(1+it/(m\sigma^2))}
+ik_0x-\frac{ik_0^2t}{2m}
\right].
$$
这里 $t\in\mathbb R$，复平方根取沿实 $t$ 连续且在 $t=0$ 等于 $1$
的分支。

**证明.** Fourier 表象中
$$
\widehat\psi_t(p)=e^{-itp^2/2m}\widehat\psi_0(p).
$$
初态的 Fourier 变换仍为 Gaussian，中心在 $p=k_0$：
$$
\widehat\psi_0(p)=\left(\frac{\sigma^2}{\pi}\right)^{1/4}
\exp\left(-\frac{\sigma^2(p-k_0)^2}{2}\right)
$$
按本书 Fourier 规范成立。反变换为
$$
\psi_t(x)=\frac1{\sqrt{2\pi}}\left(\frac{\sigma^2}{\pi}\right)^{1/4}
\int_{\mathbb R}
\exp\left(ipx-\frac{itp^2}{2m}-\frac{\sigma^2(p-k_0)^2}{2}\right)\,dp.
$$
令 $p=k_0+q$，指数中与 $q$ 有关部分为
$$
-\frac12\left(\sigma^2+\frac{it}{m}\right)q^2
+iq\left(x-\frac{k_0t}{m}\right).
$$
使用复 Gaussian 积分
$$
\int_{\mathbb R}e^{-aq^2/2+bq}\,dq
=\sqrt{\frac{2\pi}{a}}\exp\left(\frac{b^2}{2a}\right),
\qquad \operatorname{Re}a>0,
$$
其中 $a=\sigma^2+it/m$、$b=i(x-k_0t/m)$，整理即得公式。$\square$

**推论 27.3.** 位置概率密度仍为 Gaussian，中心为 $k_0t/m$，宽度满足
$$
(\Delta X)_t^2=\frac{\sigma^2}{2}\left(1+\frac{t^2}{m^2\sigma^4}\right).
$$

**证明.** 取命题 27.2 的模平方，复相位中虚部不影响密度，实部给出中心 $k_0t/m$ 的 Gaussian。与标准密度
$$
\frac1{\sqrt{\pi}w}\exp\left(-\frac{(x-x_c)^2}{w^2}\right)
$$
比较得 $w^2=\sigma^2(1+t^2/(m^2\sigma^4))$，其方差为 $w^2/2$。$\square$

## 27.2 自旋进动

**设定 27.4.** 自旋 $1/2$ 粒子在 $z$ 方向恒定磁场中的 Hamiltonian 取
$$
H=-\frac{\omega}{2}\sigma_z.
$$
其中 $\omega\in\mathbb R$。

**命题 27.5.** 若初态为 $\sigma_x$ 的 $+1$ 本征态
$$
|+x\rangle=\frac{|\uparrow\rangle+|\downarrow\rangle}{\sqrt2},
$$
则
$$
\langle\sigma_x\rangle_t=\cos\omega t,\qquad
\langle\sigma_y\rangle_t=-\sin\omega t,\qquad
\langle\sigma_z\rangle_t=0.
$$

**证明.** 演化算子为
$$
U(t)=e^{-itH}=e^{i\omega t\sigma_z/2}
=\begin{pmatrix}e^{i\omega t/2}&0\\0&e^{-i\omega t/2}\end{pmatrix}.
$$
因此
$$
\psi(t)=\frac{e^{i\omega t/2}|\uparrow\rangle+e^{-i\omega t/2}|\downarrow\rangle}{\sqrt2}.
$$
直接计算
$$
\langle\sigma_x\rangle_t
=\overline a b+\overline b a=\cos\omega t,
$$
$$
\langle\sigma_y\rangle_t
=-i\overline a b+i\overline b a=-\sin\omega t,
$$
其中 $a=e^{i\omega t/2}/\sqrt2$、$b=e^{-i\omega t/2}/\sqrt2$。最后
$$
\langle\sigma_z\rangle_t=|a|^2-|b|^2=0.
$$
$\square$

## 27.3 Rabi 振荡

**设定 27.6.** 在旋转波近似后的二能级系统中，Hamiltonian 取
$$
H=\frac12\begin{pmatrix}\delta&\Omega\\ \Omega&-\delta\end{pmatrix}
=\frac12(\Omega\sigma_x+\delta\sigma_z),
$$
其中 $\Omega,\delta\in\mathbb R$，分别为耦合强度与失谐；该实性保证
$H$ 自伴。

**命题 27.7.** 假设 $\Omega_R:=\sqrt{\Omega^2+\delta^2}>0$。若初态为 $|0\rangle=(1,0)^T$，则测得 $|1\rangle=(0,1)^T$ 的概率为
$$
P_{0\to1}(t)
=\frac{\Omega^2}{\Omega_R^2}\sin^2\frac{\Omega_Rt}{2},
\qquad
\Omega_R=\sqrt{\Omega^2+\delta^2}.
$$

**证明.** 令 $n=(\Omega,0,\delta)/\Omega_R$。则
$$
H=\frac{\Omega_R}{2}n\cdot\sigma.
$$
由 Pauli 指数公式，
$$
U(t)=e^{-itH}
=\cos\frac{\Omega_Rt}{2}I
-i\sin\frac{\Omega_Rt}{2}\,n\cdot\sigma.
$$
作用在 $|0\rangle$ 上，$|1\rangle$ 分量只来自 $\sigma_x|0\rangle=|1\rangle$，振幅为
$$
-i\frac{\Omega}{\Omega_R}\sin\frac{\Omega_Rt}{2}.
$$
取模平方即得概率公式。$\square$

**退化情形.** 若 $\Omega=\delta=0$，则 $H=0$，所以
$U(t)=I$ 且 $P_{0\to1}(t)=0$；命题中的商式在这一点不作定义。

**推论 27.8.** 共振情形 $\delta=0$ 且 $\Omega\ne0$ 下，
$$
P_{0\to1}(t)=\sin^2\frac{|\Omega|t}{2}.
$$
特别地，第一次全反转发生在正时间 $t=\pi/|\Omega|$。

**证明.** 在命题 27.7 的公式中取 $\delta=0$，则
$\Omega_R=|\Omega|$ 且 $\Omega^2/\Omega_R^2=1$，即得显示公式。令
$|\Omega|t/2=\pi/2$，得到第一次正时间全反转。
$\square$

## 本章小结

自由 Gaussian 波包展示连续变量量子态的扩散和群速度；自旋进动展示有限维酉演化如何转化为 Bloch 球旋转；Rabi 振荡展示二能级受驱系统的跃迁概率。这些例题把本书的 Hilbert 空间、Fourier 变换、Pauli 代数和时间演化统一起来。

## 练习

**练习 27.1.** 验证设定 27.1 中 $\psi_0$ 已归一化，并计算 $t=0$ 时的 $\Delta X$。

**练习 27.2.** 在命题 27.5 中，若初态为 $|\uparrow\rangle$，计算三个 Pauli 矩阵的期望值。

**练习 27.3.** 在 Rabi 公式中令 $\delta=0$ 且 $\Omega\ne0$，求第一次
达到 $P_{0\to1}=1$ 的正时间。
