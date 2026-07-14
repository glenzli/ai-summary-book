# 第二十五章：相互作用图像、Dyson 展开与跃迁率

一个弱驱动同时包含两种运动：未扰动 Hamiltonian 产生的已知快速相位，
以及相互作用真正造成的能级间跃迁。相互作用图像把前者精确剥离，只让
$V_I(t)$ 出现在积分方程中。不同时刻的 $V_I$ 一般不对易，普通指数会
丢失作用顺序；Dyson 单纯形积分正是记录这一顺序。固定有限时间的级数
可以按算子范数控制，但把振幅平方除以时间并取长时间极限，还必须有
连续末态密度和低耗尽窗口。

为使跃迁率推导自身闭合，本章从 Schrodinger 方程重新推导相互作用图像，
独立构造范数连续有界相互作用的二参数 Dyson 传播子，并证明其唯一性、
酉性和复合律。一级矩阵元及显式指数尾项随后给出有限时间误差。常扰动
的 sinc 型跃迁线形连接到 tempered distribution 的 delta 极限；Fermi
黄金规则则严格区分标签归一化与能量归一化，并把连续谱弱耦合动力学
保留在外部散射理论边界。

## 25.1 相互作用图像

**设定 25.1.** 设
$$
H_\lambda(t)=H_0+\lambda V(t),
$$
其中 $H_0$ 为时间无关自伴算子，$\lambda\in\mathbb R$ 是无量纲展开
参数；$V(t)$ 对每个 $t$ 都是有界自伴算子。正文内部证明假设下文
定义的 $V_I(t)$ 在所考察有限时间区间上算子范数连续。于是
$H_\lambda(t)$ 在 $\mathcal D(H_0)$ 上自伴。对强可微且满足
$\psi_S(t)\in\mathcal D(H_0)$ 的 Schrodinger 图像解，方程为
$$
i\frac d{dt}\psi_S(t)=H_\lambda(t)\psi_S(t).
$$
定义相互作用图像态
$$
\psi_I(t)=e^{itH_0}\psi_S(t)
$$
和相互作用 Hamiltonian
$$
V_I(t)=e^{itH_0}V(t)e^{-itH_0}.
$$

**命题 25.2.** 在设定 25.1 的可微性与定义域假设下，相互作用图像态满足
$$
i\frac d{dt}\psi_I(t)=\lambda V_I(t)\psi_I(t).
$$

**证明.** 对 $\psi_I=e^{itH_0}\psi_S$ 求导：
$$
\dot\psi_I=iH_0e^{itH_0}\psi_S+e^{itH_0}\dot\psi_S.
$$
由 Schrodinger 方程 $\dot\psi_S=-i(H_0+\lambda V)\psi_S$，且 $H_0$ 与 $e^{itH_0}$ 交换，$H_0$ 项相消，得
$$
\dot\psi_I=-i\lambda e^{itH_0}V(t)\psi_S=-i\lambda V_I(t)\psi_I.
$$
$\square$

自由相位消去以后，微分方程仍需在有限时间区间上构造成真正传播子。
积分方程适合 Picard 迭代，而时间有序区域的体积 $|t-t_0|^n/n!$ 将直接
给出级数收敛。

## 25.2 Dyson 级数

**定义 25.3.** 相互作用图像传播子 $U_I(t,t_0)$ 定义为
$$
\psi_I(t)=U_I(t,t_0)\psi_I(t_0).
$$
它满足积分方程
$$
U_I(t,t_0)=I-i\lambda\int_{t_0}^t V_I(s)U_I(s,t_0)\,ds.
$$
在设定 25.1 下，该积分是算子范数 Bochner 积分；下一步的时间有序
级数将直接构造此积分方程的唯一解。

**命题 25.4（Dyson 展开，范数连续有界情形）.** 设 $t\ge t_0$。
若 $V_I(t)$ 在包含 $[t_0,t]$ 的有限区间上算子范数连续，且
$\|V_I(s)\|\le M$，则
$$
U_I(t,t_0)=I+\sum_{n=1}^\infty(-i\lambda)^n
\int_{t_0\le s_n\le\cdots\le s_1\le t}
V_I(s_1)\cdots V_I(s_n)\,ds_1\cdots ds_n.
$$

**证明.** 从积分方程迭代一次得到
$$
U_I(t,t_0)=I-i\lambda\int_{t_0}^tV_I(s_1)\,ds_1
 -\lambda^2\int_{t_0}^t\int_{t_0}^{s_1}V_I(s_1)V_I(s_2)U_I(s_2,t_0)\,ds_2ds_1.
$$
继续迭代得到 $n$ 阶时间有序积分。若 $\|V_I(s)\|\le M$，第 $n$ 阶范数至多
$$
\frac{(|\lambda|M|t-t_0|)^n}{n!},
$$
故级数按算子范数绝对收敛，并可逐项代入积分方程。若 $U_1,U_2$
都是解，则在 $[t_0,t]$ 上
$$
\|U_1(r,t_0)-U_2(r,t_0)\|
\le |\lambda|M\int_{t_0}^r
\|U_1(s,t_0)-U_2(s,t_0)\|\,ds;
$$
Gronwall 不等式给出二者相等，故该级数就是唯一传播子。$\square$

**命题 25.4A（Dyson 传播子的酉性与复合律）.** 在命题 25.4 的假设下，
$U_I(t,t_0)$ 是酉算子，并且对 $r\le s\le t$ 有
$$
U_I(t,s)U_I(s,r)=U_I(t,r).
$$

**证明.** 范数连续性使积分方程的解按算子范数可微，且
$$
\partial_tU_I(t,s)=-i\lambda V_I(t)U_I(t,s).
$$
由 $V_I(t)^*=V_I(t)$，
$$
\partial_t\bigl(U_I(t,s)^*U_I(t,s)\bigr)=0,
$$
所以 $U_I(t,s)^*U_I(t,s)=I$。对
$X(t)=U_I(t,s)U_I(t,s)^*$ 有
$$
\dot X=-i\lambda V_I X+i\lambda X V_I,\qquad X(s)=I.
$$
$X(t)=I$ 是该有界线性方程的解，命题 25.4 使用的唯一性估计给出它是
唯一解，故 $U_IU_I^*=I$。最后，
$U_I(t,s)U_I(s,r)$ 与 $U_I(t,r)$ 对变量 $t$ 满足同一方程，并在
$t=s$ 取同一值；唯一性即给出复合律。$\square$

传播子的结构已经由本节内部证明。取未扰动能量本征态之间的矩阵元，
Dyson 一级项便成为一个带 Bohr 频率相位的普通时间积分。

## 25.3 跃迁振幅

**命题 25.5（一阶跃迁振幅及余项）.** 设 $t\ge t_0$，令
$\Delta t=t-t_0$，并假设 $\|V_I(s)\|\le M$。若
$H_0|n\rangle=E_n|n\rangle$，并从 $H_0$ 的一个正交归一本征系中选取
$|i\rangle,|f\rangle$，系统初态为 $|i\rangle$，则
$$
\langle f|U_I(t,t_0)|i\rangle
=\delta_{fi}+\lambda c_f^{(1)}(t)+R_f^{(\ge2)}(t),
$$
其中
$$
c_f^{(1)}(t)
=-i\int_{t_0}^t e^{i(E_f-E_i)s}\langle f|V(s)|i\rangle\,ds.
$$
余项满足显式估计
$$
\bigl|R_f^{(\ge2)}(t)\bigr|
\le e^{|\lambda|M\Delta t}-1-|\lambda|M\Delta t.
$$

**证明.** Dyson 展开的一级项给出
$$
\langle f|U_I(t,t_0)|i\rangle
=\delta_{fi}-i\lambda\int_{t_0}^t\langle f|V_I(s)|i\rangle\,ds
+R_f^{(\ge2)}(t).
$$
由 $V_I(s)=e^{isH_0}V(s)e^{-isH_0}$，
$$
\langle f|V_I(s)|i\rangle=e^{i(E_f-E_i)s}\langle f|V(s)|i\rangle.
$$
比较 $\lambda$ 的一次项得到 $c_f^{(1)}$。Dyson 级数从二阶起的尾项范数不超过
$$
\sum_{n=2}^{\infty}\frac{(|\lambda|M\Delta t)^n}{n!}
=e^{|\lambda|M\Delta t}-1-|\lambda|M\Delta t;
$$
矩阵元的绝对值不超过算子范数，故得余项估计。$\square$

**例子 25.5A（常扰动的有限时间线形）.** 取 $t_0=0$，令 $V$ 不显含
时间，$f\ne i$，并记
$\omega_{fi}=E_f-E_i$、$V_{fi}=\langle f|V|i\rangle$。若
$\omega_{fi}\ne0$，则一级跃迁振幅为
$$
\lambda c_f^{(1)}(t)
=\lambda V_{fi}\frac{1-e^{i\omega_{fi}t}}{\omega_{fi}},
$$
相应首个非零概率项是
$$
P_{i\to f}^{(2)}(t)
=4\lambda^2|V_{fi}|^2
\frac{\sin^2(\omega_{fi}t/2)}{\omega_{fi}^2}.
$$
当 $\omega_{fi}\to0$ 时，连续极限为
$P_{i\to f}^{(2)}(t)=\lambda^2|V_{fi}|^2t^2$。这是一条宽度约为
$1/t$ 的有限时间谱线，而不是到单个离散末态的恒定速率；同时还需
$|\lambda|Mt$ 足够小，使高阶尾项与初态耗尽尚未主导。

连续末态通道中，要从这条 sinc 型谱线得到近似恒定速率，需要先对末态
测度积分，再取长时间分布极限。点态极限不存在，正确对象是 delta 分布。

## 25.4 黄金规则边界

**命题 25.6（长时间核的分布极限）.** 在 tempered distribution 意义下，
$$
\frac1T\left|\int_0^T e^{i\omega t}\,dt\right|^2
\longrightarrow 2\pi\delta(\omega)
$$
当 $T\to\infty$。

**证明.** 直接改变变量得
$$
\frac1T\left|\int_0^T e^{i\omega t}\,dt\right|^2
=\int_{-T}^T\left(1-\frac{|\tau|}{T}\right)e^{i\omega\tau}\,d\tau.
$$
对任意 Schwartz 测试函数 $\varphi(\omega)$ 配对，并先对 $\omega$ 积分。所得 Fourier 变换快速衰减，而三角权重逐点趋于 $1$ 且绝对值不超过 $1$；支配收敛定理给出极限
$$
\int_{\mathbb R}\int_{\mathbb R}e^{i\omega\tau}
\varphi(\omega)\,d\omega\,d\tau
=2\pi\varphi(0).
$$
这正是 $2\pi\delta$。$\square$

**物理推导 25.7（Fermi 黄金规则及其归一化边界）.** 设时间无关
弱扰动 $\lambda V$ 把离散初态 $|i\rangle$ 耦合到一个连续末态通道。
下列连续态 ket 是广义本征态，等式均按分布意义理解。先用标签
$\alpha$ 归一化，并假设能量壳附近 $E(\alpha)$ 有 $C^1$ 局部逆
$\alpha(E)$，且 $0<|d\alpha/dE|<\infty$：
$$
\langle f,\alpha|f,\alpha'\rangle=\delta(\alpha-\alpha'),
\qquad
\rho_{\mathcal F}(E)=\left|\frac{d\alpha}{dE}\right|,
\qquad d\mu_\alpha=\rho_{\mathcal F}(E)\,dE,
$$
其中 $d\mu_\alpha$ 是标签坐标的 Lebesgue 测度在能量变量下的表示。
并假设态密度与矩阵元在宽度 $O(T^{-1})$ 的能量窗内变化缓慢。把
一阶跃迁概率对 $d\mu_\alpha$ 积分，并使用命题 25.6，得到长时间、
低耗尽近似
$$
\Gamma_{i\to\mathcal F}
=2\pi\lambda^2
\bigl|\langle f(E_i)|V|i\rangle\bigr|^2
\rho_{\mathcal F}(E_i).
$$
这里 $|f(E)\rangle$ 表示标签归一化族 $|f,\alpha(E)\rangle$。若改用能量归一化态
$$
|\widetilde f,E\rangle
:=\rho_{\mathcal F}(E)^{1/2}|f,\alpha(E)\rangle,
\qquad
\langle\widetilde f,E|\widetilde f,E'\rangle=\delta(E-E'),
$$
同一公式写成
$$
\Gamma_{i\to\mathcal F}
=2\pi\lambda^2
\bigl|\langle\widetilde f,E_i|V|i\rangle\bigr|^2,
$$
此时不得再额外乘一次 $\rho_{\mathcal F}(E_i)$。
能量壳在 $E=E_i$，而不是一个任意的 $E_f$。该式不描述到单个离散末态的恒定不可逆速率；它还要求观测时间足以分辨连续谱、但不能长到初态耗尽使一阶微扰失效。连续谱归一化、阈值奇性和严格弱耦合极限依赖谱测度与散射理论（QM-EXT-8），不由分布恒等式单独保证。

相互作用图像把自由相位与真实跃迁分开；Dyson 级数在范数连续有界假设
下给出唯一、酉且满足复合律的二参数传播子，并有显式有限时间尾项界。
常扰动产生 sinc 型离散跃迁线，只有先对连续末态积分并使用分布极限，
才得到黄金规则的速率形式。标签归一化中的态密度与能量归一化振幅不得
重复计数。下一章回到有限维旋转表示，把两个角动量的张量积改写为总
角动量通道，并由此读取选择定则。

## 练习

**练习 25.1.** 若 $[V_I(t),V_I(s)]=0$ 对所有 $s,t$ 成立，证明 Dyson 展开化为普通指数。

**练习 25.2.** 对常扰动 $V$，写出一阶振幅中的积分并化简为 $\sin$ 函数形式。

**练习 25.3.** 说明时间有序乘积为什么在 $[V_I(t),V_I(s)]\ne0$ 时不可省略。
