# 第二十七章：标准精算例题：Gaussian 波包、自旋进动与 Rabi 振荡

一套形式主义只有在完整计算中同时守住归一化、相位、单位和近似条件，
才真正成为可用的量子力学。自由 Gaussian 波包检验 Fourier 演化能否
给出正确群速度与展宽；恒定磁场中的自旋检验矩阵指数是否转化为 Bloch
向量进动；受驱二能级系统则检验旋转坐标系和旋转波近似何时允许把含时
问题化为常矩阵。三个模型覆盖连续变量、有限维自治演化与近共振驱动。

以下计算各自从归一化初态和自伴 Hamiltonian 出发，一直推进到可测
概率。Gaussian 积分采用全书固定的 Fourier 规范；自旋进动直接乘
Pauli 矩阵；Rabi 公式则先写出实验室驱动，明确近共振、弱驱动和有限
时间窗条件，再对有效 Hamiltonian 精确指数化。这样，最终结果既可作为
前面章节的交叉核对，也不会把有效二能级公式误当成任意强驱动下的精确
实验室动力学。

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

自由粒子没有回复力，位置宽度随时间增长；有限维自旋没有空间波包，
同一酉性改为保持 Bloch 向量长度。下一例把 Fourier 积分换成一个
$2\times2$ 对角矩阵指数。

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

恒定磁场只产生自治进动。若再施加横向周期驱动，实验室 Hamiltonian
显含时间；在近共振弱驱动区可以转入旋转坐标系并舍去快速反旋项，得到
可精确求解的有效常矩阵。

## 27.3 Rabi 振荡

考虑实验室系中的受驱二能级模型
$$
H_{\mathrm{lab}}(t)=\frac{\omega_0}{2}\sigma_z
+A\cos(\omega t)\sigma_x.
$$
在以驱动频率旋转的坐标系中，横向驱动分成常数项与频率约为
$\omega_0+\omega$ 的反旋项。旋转波近似（rotating-wave approximation,
RWA）舍去后者；它要求近共振 $|\omega_0-\omega|\ll\omega_0+\omega$
且弱驱动 $|A|\ll\omega_0+\omega$。在不积累长期共振误差的有限时间窗内，
被舍去的快速振荡贡献相对于保留项受量级
$|A|/(\omega_0+\omega)$ 控制。下述 $\Omega$ 是采用本书归一化后保留下来的
有效耦合，$\delta=\omega_0-\omega$ 是失谐；若改变实验室驱动项中 $A$ 的
归一化，$\Omega$ 与 $A$ 之间可能相差因子 $2$。

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

三个模型现在给出一组互相独立的量纲与极限检查。Gaussian 中心以
$k_0/m$ 移动而方差按 $t^2$ 展宽；静磁场使 Bloch 向量以 $\omega$
进动且长度不变；Rabi 驱动在失谐时把最大跃迁概率压低到
$\Omega^2/(\Omega^2+\delta^2)$，共振时才发生全反转。最后一个结论只
属于已声明的 RWA 参数区，前两个则是相应自治 Hamiltonian 的精确演化。
至此，全书从 Hilbert 空间与谱开始的抽象结构已经落到连续传播、自旋
测量和受驱跃迁三类可复算预测上。

## 练习

**练习 27.1.** 验证设定 27.1 中 $\psi_0$ 已归一化，并计算 $t=0$ 时的 $\Delta X$。

**练习 27.2.** 在命题 27.5 中，若初态为 $|\uparrow\rangle$，计算三个 Pauli 矩阵的期望值。

**练习 27.3.** 在 Rabi 公式中令 $\delta=0$ 且 $\Omega\ne0$，求第一次
达到 $P_{0\to1}=1$ 的正时间。
