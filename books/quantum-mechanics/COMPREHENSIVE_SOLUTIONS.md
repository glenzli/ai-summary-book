# 综合习题答案

本文件与 [COMPREHENSIVE_EXERCISES.md](COMPREHENSIVE_EXERCISES.md) 一一对应。每个答案保留主要推导链条；提示见 [HINTS.md](HINTS.md)，需要章内定理细节时回到对应章节正文。

## 综合题 1

Born 概率为 $p_r=\|P_r\psi\|^2$。若测得 $\lambda_r$ 且 $p_r>0$，条件态为 $P_r\psi/\sqrt{p_r}$。密度算子形式为
$$
p_r=\operatorname{tr}(P_\psi P_r),\qquad
\rho_r=\frac{P_rP_\psi P_r}{\operatorname{tr}(P_\psi P_r)}.
$$

## 综合题 2

若
$$
\Psi=\sum_rs_re_r\otimes f_r,
$$
则
$$
\rho_A=\sum_rs_r^2|e_r\rangle\langle e_r|.
$$
态纠缠当且仅当至少两个 Schmidt 系数非零。纠缠熵为
$$
S(\rho_A)=-\sum_rs_r^2\log s_r^2.
$$

## 综合题 3

Schrodinger 图像中 $\psi(t)=e^{-itH}\psi$，
$$
\frac d{dt}\langle A\rangle_{\psi(t)}
=i\langle\psi(t),[H,A]\psi(t)\rangle.
$$
Heisenberg 图像 $A_H(t)=e^{itH}Ae^{-itH}$ 满足
$$
\dot A_H(t)=i[H,A_H(t)].
$$
若 $[A,H]=0$，则 $A_H(t)=A$，期望值守恒。

## 综合题 4

定义
$$
a=\sqrt{\frac{m\omega}{2}}X+\frac{i}{\sqrt{2m\omega}}P.
$$
则
$$
H=\omega(a^*a+I/2),
$$
能级为 $E_n=\omega(n+1/2)$。基态由 $a\psi_0=0$ 给出：
$$
\psi_0(x)=\left(\frac{m\omega}{\pi}\right)^{1/4}e^{-m\omega x^2/2}.
$$

## 综合题 5

Pauli 矩阵见第九章，$S_i=\sigma_i/2$。因 $\sigma_i^2=I$，
$$
S^2=\frac14(\sigma_x^2+\sigma_y^2+\sigma_z^2)=\frac34I.
$$
绕 $z$ 轴旋转角 $\theta$ 的酉算子为
$$
U_z(\theta)=e^{-i\theta\sigma_z/2}.
$$

## 综合题 6

平移酉群为
$$
(U(a)\psi)(x)=\psi(x-a).
$$
生成元为 $P=-i\nabla$。自由 Hamiltonian $H=P^2/2m$ 是 $P$ 的函数，故与 $U(a)=e^{-ia\cdot P}$ 交换。

## 综合题 7

一阶方程为
$$
(H_0-E_n^{(0)})\psi_n^{(1)}=(E_n^{(1)}-V)\psi_n^{(0)}.
$$
左乘 $\psi_n^{(0)}$ 得
$$
E_n^{(1)}=\langle\psi_n^{(0)},V\psi_n^{(0)}\rangle.
$$
左乘 $\psi_m^{(0)}$ 得
$$
\psi_n^{(1)}=\sum_{m\ne n}
\frac{\langle\psi_m^{(0)},V\psi_n^{(0)}\rangle}{E_n^{(0)}-E_m^{(0)}}\psi_m^{(0)}.
$$
简并时先在简并子空间 $M$ 中对角化压缩算子 $P_MVP_M|_M$；只有在
$V(M)\subseteq M$ 时才能把它写成 $V|_M$。

## 综合题 8

波算子为
$$
\Omega_\pm
=\operatorname{s-lim}_{t\to\pm\infty}
e^{itH}e^{-itH_0}P_{\mathrm{ac}}(H_0).
$$
它们渐近完备是指
$\operatorname{Ran}\Omega_\pm=\mathcal H_{\mathrm{ac}}(H)$。
$S$ 矩阵为
$S=\Omega_+^*\Omega_-$，作用在
$\mathcal H_{\mathrm{ac}}(H_0)$ 上。一阶 Born 振幅为
$$
f(\mathbf k',\mathbf k)
=-\frac{m}{2\pi}\int e^{-i(\mathbf k'-\mathbf k)\cdot x}V(x)\,dx,
$$
微分截面为 $d\sigma/d\Omega=|f|^2$。

## 综合题 9

绝热方程为
$$
i\varepsilon\dot\psi=H(s)\psi.
$$
非简并本征向量 $u_n(s)$ 的 Berry 连接为
$$
\mathcal A_n=i\langle u_n,\dot u_n\rangle.
$$
规范变换 $u_n\mapsto e^{i\chi}u_n$ 使 $\mathcal A_n\mapsto \mathcal A_n-\dot\chi$。

## 综合题 10

POVM 为 $E_i=M_i^*M_i$。概率为
$$
p_i=\operatorname{tr}(M_i\rho M_i^*)=\operatorname{tr}(\rho E_i).
$$
若 $p_i>0$，测后态为
$$
\rho_i=M_i\rho M_i^*/p_i.
$$
其迹为 $1$，因为分子迹正是 $p_i$。

## 综合题 11

保迹条件为
$$
\sum_\alpha M_\alpha^*M_\alpha=I.
$$
若 $\rho\ge0$，则每项 $M_\alpha\rho M_\alpha^*\ge0$，和仍为正。完全正性要求 $\Phi\otimes\operatorname{id}_n$ 对任意旁系统维数仍保持正性，排除只在孤立系统上正但与纠缠旁系统合用时失败的映射。

## 综合题 12

传播子卷积律为
$$
K(t+s;x,y)=\int K(t;x,z)K(s;z,y)\,dz.
$$
Trotter 公式把 $e^{-it(T+V)}$ 写成许多短时自由传播与势能相位的乘积，插入位置分辨率后得到路径离散和。一般路径空间上没有平移不变 Lebesgue 测度，因此路径积分需解释为离散极限或振荡积分。

## 综合题 13

中心势 $H=-\Delta/(2m)+V(r)$ 中，$\Delta$ 与旋转生成元 $L_i$ 对易，径向乘法算子 $V(r)$ 也与 $L_i$ 对易，因此 $[H,L_i]=0$，进而 $[H,L^2]=[H,L_z]=0$。取
$$
\psi(r,\Omega)=R(r)Y_\ell^m(\Omega)
$$
并用
$$
\Delta=\frac1{r^2}\frac d{dr}r^2\frac d{dr}-\frac{L^2}{r^2}
$$
得到
$$
-\frac1{2m}\left(R''+\frac2rR'-\frac{\ell(\ell+1)}{r^2}R\right)+VR=ER.
$$
令 $u=rR$ 后为
$$
-\frac1{2m}u''+\left(V+\frac{\ell(\ell+1)}{2mr^2}\right)u=Eu.
$$
氢原子能级公式还依赖 Coulomb Hamiltonian 自伴性、球谐完备性、径向 Laguerre 解和束缚态完备性。

## 综合题 14

最小耦合 Hamiltonian 为
$$
H_A=\frac1{2m}(P-qA(X))^2+q\Phi(X),
$$
动力学动量为 $\Pi=P-qA(X)$。时间无关规范变换 $A\mapsto A+\nabla\chi$、$\psi\mapsto e^{iq\chi}\psi$ 下，
$$
(P-q(A+\nabla\chi))e^{iq\chi}\psi=e^{iq\chi}(P-qA)\psi,
$$
故 $H_{A+\nabla\chi}=e^{iq\chi}H_Ae^{-iq\chi}$。匀强磁场中 $[\Pi_x,\Pi_y]=iqB$，定义
$$
a=\frac{\Pi_x+i\Pi_y}{\sqrt{2qB}}
$$
时须取 $qB>0$；此时 $[a,a^*]=1$，于是
$$
H=\frac{qB}{m}(a^*a+1/2).
$$
对 $qB<0$ 交换升降算子的选取，谱统一为
$E_n=|qB|(n+1/2)/m$；$qB=0$ 时退化为自由粒子，不能使用该升降
算子商式。
Aharonov-Bohm 相位依赖闭环磁通
$$
\exp\left(iq\oint A\cdot dx\right),
$$
在允许的单值规范变换下不变，因此不违背规范不变性。

## 综合题 15

对归一化态 $\psi\in\mathcal D(AB)\cap\mathcal D(BA)$，令
$A_0=A-\langle A\rangle$、$B_0=B-\langle B\rangle$。Cauchy-Schwarz 给出
$$
\Delta A\,\Delta B\ge |\langle A_0\psi,B_0\psi\rangle|.
$$
取虚部得
$$
\Delta A\,\Delta B\ge \frac12|\langle[A,B]\rangle|.
$$
若
$$
i\partial_t\psi=-\frac1{2m}\Delta\psi+V\psi
$$
且 $V$ 实值，则
$$
\partial_t|\psi|^2+\nabla\cdot\left(m^{-1}\operatorname{Im}(\overline\psi\nabla\psi)\right)=0.
$$
对 $H=P^2/2m+V(X)$，Heisenberg 方程给出
$$
\frac d{dt}\langle X\rangle=\frac1m\langle P\rangle,\qquad
\frac d{dt}\langle P\rangle=-\langle V'(X)\rangle.
$$

## 综合题 16

设 $H_\lambda(t)=H_0+\lambda V(t)$，并假设 $V_I(t)$ 在所考察的有限
时间区间上按算子范数连续。定义
$\psi_I=e^{itH_0}\psi_S$ 与
$V_I(t)=e^{itH_0}V(t)e^{-itH_0}$。代入 Schrodinger 方程得
$$
i\dot\psi_I=\lambda V_I\psi_I.
$$
传播子满足
$$
U_I(t,t_0)=I-i\lambda\int_{t_0}^tV_I(s)U_I(s,t_0)\,ds,
$$
由命题 25.4，算子范数 Bochner 积分的迭代给出时间有序 Dyson 级数
$$
I+\sum_{n\ge1}(-i\lambda)^n\int_{t_0\le s_n\le\cdots\le s_1\le t}V_I(s_1)\cdots V_I(s_n)\,ds_1\cdots ds_n.
$$
一阶跃迁振幅的 $\lambda$ 系数为
$$
c_f^{(1)}(t)=-i\int_{t_0}^t e^{i(E_f-E_i)s}\langle f|V(s)|i\rangle\,ds.
$$
若 $\|V_I(s)\|\le M$ 且 $\Delta t=t-t_0\ge0$，二阶及以上的
Dyson 尾项范数不超过
$$
e^{|\lambda|M\Delta t}-1-|\lambda|M\Delta t.
$$
黄金规则需要连续谱态密度和长时间分布极限。标签归一化态给出
$2\pi\lambda^2|\langle f(E_i)|V|i\rangle|^2\rho(E_i)$；若改用
能量归一化态 $|\widetilde f,E\rangle=\rho(E)^{1/2}|f,\alpha(E)\rangle$，
则不得再乘 $\rho(E_i)$。严格极限依赖谱测度或散射理论。

## 综合题 17

两个自旋 $1/2$ 的 triplet 为
$$
|1,1\rangle=|\uparrow\uparrow\rangle,\quad
|1,0\rangle=\frac{|\uparrow\downarrow\rangle+|\downarrow\uparrow\rangle}{\sqrt2},\quad
|1,-1\rangle=|\downarrow\downarrow\rangle,
$$
singlet 为
$$
|0,0\rangle=\frac{|\uparrow\downarrow\rangle-|\downarrow\uparrow\rangle}{\sqrt2}.
$$
Clebsch-Gordan 系数由
$$
|j,m\rangle=\sum_{m_1+m_2=m}
\langle j_1m_1,j_2m_2|jm\rangle
|j_1,m_1\rangle|j_2,m_2\rangle
$$
定义。因张量积基的 $J_z$ 本征值为 $m_1+m_2$，非零系数必须满足 $m=m_1+m_2$。偶极算子为秩 $1$ 球张量，Wigner-Eckart 定理给出 $m'=m+q$，其中 $q=-1,0,1$，所以 $\Delta m=0,\pm1$。

## 综合题 18

自由 Gaussian 波包中心按群速度移动：
$$
x_c(t)=\frac{k_0}{m}t,
$$
宽度满足
$$
(\Delta X)_t^2=\frac{\sigma^2}{2}\left(1+\frac{t^2}{m^2\sigma^4}\right).
$$
对 Hamiltonian $H=-\omega\sigma_z/2$，若初态为 $|+x\rangle$，则
$$
\langle\sigma_x\rangle_t=\cos\omega t,\qquad
\langle\sigma_y\rangle_t=-\sin\omega t,\qquad
\langle\sigma_z\rangle_t=0.
$$
对
$$
H=\frac12(\Omega\sigma_x+\delta\sigma_z),
$$
其中 $\Omega,\delta\in\mathbb R$。当 $\Omega^2+\delta^2>0$ 时，从
$|0\rangle$ 到 $|1\rangle$ 的概率为
$$
P_{0\to1}(t)=\frac{\Omega^2}{\Omega^2+\delta^2}
\sin^2\frac{\sqrt{\Omega^2+\delta^2}\,t}{2}.
$$
当 $\Omega=\delta=0$ 时 $H=0$ 且该概率为 $0$，上面的商式不作定义。
这三个例题分别检验 Fourier 自由演化、Pauli 代数酉旋转和二能级谱分解/指数公式。
