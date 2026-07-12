# 量子力学答案手册

本文件给出 `books/quantum-mechanics/` 当前版本全部章末练习的参考答案。答案以写出算子、内积、谱投影和迹公式为原则；涉及大型外部定理时只使用其标准结论。

## 使用说明

答案编号与正文练习编号一一对应。正文第 $n$ 章的“练习 $n.k$”在本手册中标为“答案 $n.k$”；附录练习使用 `A.k`、`B.k`、`C.k` 编号。解题提示见 [HINTS.md](HINTS.md)，跨章节综合题答案另见 [COMPREHENSIVE_SOLUTIONS.md](COMPREHENSIVE_SOLUTIONS.md)。

## 序章

**答案 0.1.** 恢复 $\hbar$ 后
$$
U(t)=e^{-itH/\hbar}.
$$
指数必须无量纲，所以 $Ht/\hbar$ 无量纲。因 $\hbar$ 的量纲为能量乘时间，$H$ 的量纲为能量。

**答案 0.2.** 由交换子双线性性，
$$
[Q,K]=[aX,bP]=ab[X,P]=iab\hbar I.
$$
要使其等于 $iI$，需 $ab\hbar=1$。

## 第一章

**答案 1.1.** Cauchy-Schwarz 等号情形给出 $\phi=\lambda\psi$。因二者单位，$|\lambda|=1$，故 $\lambda=e^{i\theta}$，二者表示同一射线。

**答案 1.2.** 有
$$
\langle\psi,\phi\rangle=1/\sqrt2,
$$
故转移概率为 $1/2$。

## 第二章

**答案 2.1.** 若 $P^2=P$ 且 $P^*=P$，则
$$
(I-P)^2=I-2P+P^2=I-P,\qquad (I-P)^*=I-P.
$$
所以 $I-P$ 是正交投影。

**答案 2.2.** $\sigma_z$ 的本征值为 $1,-1$。对应谱投影为
$$
P_+=\begin{pmatrix}1&0\\0&0\end{pmatrix},\qquad
P_-=\begin{pmatrix}0&0\\0&1\end{pmatrix}.
$$

## 第三章

**答案 3.1.** 取
$$
f_n(x)=\mathbf 1_{[n,n+1]}(x).
$$
则 $\|f_n\|_2=1$，但 $\|Xf_n\|_2^2=\int_n^{n+1}x^2dx\ge n^2$，无统一有界常数。

**答案 3.2.** 若 $A=\sum_r\lambda_rP_r$，则谱测度满足
$$
E_A(\Delta)=\sum_{\lambda_r\in\Delta}P_r.
$$
因此
$$
\mu_\psi^A(\Delta)=\sum_{\lambda_r\in\Delta}\|P_r\psi\|^2.
$$

## 第四章

**答案 4.1.** 概率为
$$
p_\pm=\|P_\pm\psi\|^2.
$$
期望值为
$$
\langle A\rangle_\psi=\lambda_+p_++\lambda_-p_-.
$$

**答案 4.2.** 若共同特征基为 $e_j$，$Ae_j=a_je_j$、$Be_j=b_je_j$，则
$$
ABe_j=a_jb_je_j=BAe_j.
$$
在基上相等故算子相等。

## 第五章

**答案 5.1.** 若 $U(t)$ 酉，则
$$
\|U(t)\psi\|^2=\langle U(t)\psi,U(t)\psi\rangle
=\langle\psi,U(t)^*U(t)\psi\rangle=\|\psi\|^2.
$$

**答案 5.2.** 有界情形中
$$
\frac d{dt}\langle\psi(t),A\psi(t)\rangle
=i\langle\psi(t),[H,A]\psi(t)\rangle=0.
$$

## 第六章

**答案 6.1.** $\psi_n(0)=\psi_n(L)=0$。并且
$$
-\frac1{2m}\psi_n''=\frac{n^2\pi^2}{2mL^2}\psi_n.
$$

**答案 6.2.** 若左侧为 $e^{ikx}+re^{-ikx}$，右侧为 $te^{iqx}$，在 $x=0$ 的连续条件为
$$
1+r=t,\qquad ik(1-r)=iqt.
$$

## 第七章

**答案 7.1.** 由 $N=a^*a$ 与 $[a,a^*]=I$，
$$
[N,a]=a^*a^2-aa^*a=(a^*a-aa^*)a=-a,
$$
并且
$$
[N,a^*]=a^*aa^*-a^*a^*a=a^*(aa^*-a^*a)=a^*.
$$

**答案 7.2.** 恢复 $\hbar$ 后
$$
E_n=\hbar\omega\left(n+\frac12\right).
$$

## 第八章

**答案 8.1.** 对
$$
\Phi^+=\frac{|00\rangle+|11\rangle}{\sqrt2},
$$
Schmidt 系数均为 $1/\sqrt2$，故
$$
\rho_A=\frac12|0\rangle\langle0|+\frac12|1\rangle\langle1|=\frac12I.
$$

**答案 8.2.** 若 $P_\psi=|\psi\rangle\langle\psi|$、$P_\phi=|\phi\rangle\langle\phi|$，则
$$
|\psi\otimes\phi\rangle\langle\psi\otimes\phi|
=(|\psi\rangle\langle\psi|)\otimes(|\phi\rangle\langle\phi|).
$$
二者作用在单纯张量 $x\otimes y$ 上均给出 $\psi\otimes\phi\,\langle\psi,x\rangle\langle\phi,y\rangle$。

## 第九章

**答案 9.1.** 直接乘法得
$$
\sigma_x\sigma_y=i\sigma_z,\qquad \sigma_y\sigma_x=-i\sigma_z.
$$

**答案 9.2.** 因 $\sigma_i^2=I$，
$$
S^2=S_x^2+S_y^2+S_z^2=\frac14(3I)=\frac34I.
$$

## 第十章

**答案 10.1.** 反酉算子满足
$$
\langle U\psi,U\phi\rangle=\overline{\langle\psi,\phi\rangle}
$$
或等价约定下取共轭，因此绝对值平方保持。

**答案 10.2.** 有限维中若 $A$ 与 $H$ 对易，则 $A$ 的每个特征子空间被 $H$ 保持。故对应正交投影 $P_\lambda$ 与 $H$ 对易。由谱投影求和，所有 $E_A(\Delta)$ 与 $H$ 对易。

## 第十一章

**答案 11.1.** 在共同定义域上
$$
[X,P^2]=[X,P]P+P[X,P]=iP+iP=2iP.
$$

**答案 11.2.** 由附录 B，
$$
\widehat{Pf}(p)=\widehat{-if'}(p)=p\widehat f(p).
$$
所以动量表象中 $P$ 是乘以 $p$ 的算子。

## 第十二章

**答案 12.1.** 对 $E$ 附近本征值，非简并一阶修正为扰动矩阵在对应本征态上的对角元。这里扰动为
$$
V=\begin{pmatrix}0&v\\ \overline v&0\end{pmatrix},
$$
故一阶修正为 $0$。更具体地，$E$ 附近的分支为
$$
E(\lambda)=E+\lambda^2\frac{|v|^2}{E-F}+O(\lambda^4).
$$
因此 $v\ne0$ 时能级移动从二阶开始；$v=0$ 时该分支恒为 $E$。

**答案 12.2.** 简并时 $E_n^{(0)}-E_m^{(0)}=0$ 可能出现在分母中，
非简并公式无定义。必须先在简并子空间 $M$ 中对角化压缩算子
$P_MVP_M|_M$，得到正确的零阶基和一阶能量；除非 $V(M)\subseteq M$，
不能把它简写成 $V|_M$。

## 第十三章

**答案 13.1.** Rayleigh 原理给出
$$
E_0=\min_{\|\psi\|=1}\langle\psi,H\psi\rangle.
$$
因此任意归一试探态的能量期望均不低于 $E_0$。

**答案 13.2.** 无限深方势阱中 $p=\sqrt{2mE}$ 为常数，但
$0,L$ 是 Dirichlet 硬壁，不是公式 13.5 所要求的简单转折点。硬壁
相位条件给出
$$
L\sqrt{2mE}=\pi\hbar n,
\qquad n=1,2,\ldots,
$$
所以
$$
E_n=\frac{\pi^2\hbar^2n^2}{2mL^2}.
$$
这恰好等于精确能级，因而尤其给出高能主项。简单转折点的 Maslov
$1/2$ 修正不能直接用于无限高硬壁；两类边界的相位条件不同。

## 第十四章

**答案 14.1.** 若 $S$ 酉，则对任意入射态 $\psi$，
$$
\|S\psi\|^2=\langle\psi,S^*S\psi\rangle=\|\psi\|^2.
$$
因此总概率守恒。

**答案 14.2.** 若形式上 $V(x)=g\delta(x)$，则 Fourier 变换为常数 $g$。Born 振幅与 Fourier 变换成正比，所以近似为常数。

## 第十五章

**答案 15.1.** 对两个粒子，交换算子为 $\tau(\psi\otimes\phi)=\phi\otimes\psi$。因此
$$
P_+=\frac12(I+\tau),\qquad P_-=\frac12(I-\tau).
$$

**答案 15.2.** 若两个单粒子态同为 $\psi$，反对称化为
$$
\frac12(\psi\otimes\psi-\psi\otimes\psi)=0.
$$

## 第十六章

**答案 16.1.** 由 $\langle u,u\rangle=1$ 求导得
$$
\langle\dot u,u\rangle+\langle u,\dot u\rangle=0.
$$
而 $\langle\dot u,u\rangle=\overline{\langle u,\dot u\rangle}$，故 $\langle u,\dot u\rangle$ 的实部为零。

**答案 16.2.** 规范变换使 $\mathcal A\mapsto\mathcal A-\dot\chi$。开路径积分会改变端点项 $-\chi(1)+\chi(0)$；闭合回路若规范单值，则改变为 $2\pi$ 的整数倍，物理相位模 $2\pi$ 不变。

## 第十七章

**答案 17.1.**
$$
\operatorname{tr}\rho^2=p^2+(1-p)^2.
$$
它等于 $1$ 当且仅当 $p=0$ 或 $p=1$。

**答案 17.2.** 若 $\rho_j\ge0$ 且 $\operatorname{tr}\rho_j=1$，$q_j\ge0$、$\sum_jq_j=1$，则
$$
\rho=\sum_jq_j\rho_j\ge0,\qquad \operatorname{tr}\rho=\sum_jq_j=1.
$$

## 第十八章

**答案 18.1.** 投影测量的谱投影 $P_i$ 满足 $P_i\ge0$ 且 $\sum_iP_i=I$，所以是 POVM。此时 $E_i=P_i$。

**答案 18.2.** 酉信道只有一个 Kraus 算子 $M_1=U$。归一条件为 $U^*U=I$。

## 第十九章

**答案 19.1.** $I/2$ 的本征值为 $1/2,1/2$，故
$$
S(I/2)=-2\cdot\frac12\log\frac12=\log2.
$$

**答案 19.2.** 若 $\rho=\sum_jp_j|e_j\rangle\langle e_j|$，则
$$
U\rho U^*=\sum_jp_j|Ue_j\rangle\langle Ue_j|
$$
有相同本征值，因此熵相同。

## 第二十章

**答案 20.1.** 若 $U(t)$ 与 $U(s)$ 分别有核 $K(t)$ 与 $K(s)$，则复合算子的核为
$$
\int K(t;x,z)K(s;z,y)\,dz,
$$
这正是 $U(t+s)$ 的核。

**答案 20.2.** 一维自由核为
$$
K_0(t;x,y)=\left(\frac{m}{2\pi it}\right)^{1/2}
\exp\left(\frac{im(x-y)^2}{2t}\right).
$$
$t\to0$ 时它在分布意义下趋向 $\delta(x-y)$，即对测试函数积分给出函数值。

## 第二十一章

**答案 21.1.** 形式替换 $E\mapsto i\hbar\partial_t$、$p\mapsto-i\hbar\nabla$ 得
$$
-\hbar^2\partial_t^2\phi=(-\hbar^2c^2\Delta+m^2c^4)\phi.
$$
移项并除以 $\hbar^2c^2$ 得 Klein-Gordon 方程。

**答案 21.2.** 在 $H_D^2$ 中，动量交叉项含
$$
\alpha_i\alpha_jp_ip_j+\alpha_j\alpha_ip_jp_i
=(\alpha_i\alpha_j+\alpha_j\alpha_i)p_ip_j=0
$$
对 $i\ne j$。质量与动量交叉项含 $\alpha_i\beta+\beta\alpha_i=0$，故消失。

## 第二十二章

**答案 22.1.** 若 $R=u/r$，则
$$
R'=\frac{u'}r-\frac u{r^2},
$$
并且
$$
R''=\frac{u''}r-\frac{2u'}{r^2}+\frac{2u}{r^3}.
$$
因此
$$
R''+\frac2rR'
=\frac{u''}r-\frac{2u'}{r^2}+\frac{2u}{r^3}
+\frac2r\left(\frac{u'}r-\frac u{r^2}\right)
=\frac{u''}r.
$$

**答案 22.2.** 当 $\ell=0$ 时离心势项消失，径向方程为
$$
-\frac1{2m}u''(r)+V(r)u(r)=Eu(r).
$$
边界条件通常要求 $u(0)=0$ 且 $u$ 在 $(0,\infty)$ 上平方可积。

**答案 22.3.** 对固定 $n$，$\ell=0,\dots,n-1$，每个 $\ell$ 有 $2\ell+1$ 个 $m$ 值。因此简并度为
$$
\sum_{\ell=0}^{n-1}(2\ell+1)=n^2.
$$

## 第二十三章

**答案 23.1.** 因旋度满足 $\nabla\times\nabla\chi=0$，
$$
\nabla\times(A+\nabla\chi)=\nabla\times A.
$$
故磁场不变。

**答案 23.2.** 设
$$
a=\frac{\Pi_x+i\Pi_y}{\sqrt{2qB}},\qquad
a^*=\frac{\Pi_x-i\Pi_y}{\sqrt{2qB}}.
$$
这里按命题 23.7 的该组升降算子取 $qB>0$。则
$$
[a,a^*]=\frac1{2qB}[\Pi_x+i\Pi_y,\Pi_x-i\Pi_y]
=\frac1{2qB}(-i[\Pi_x,\Pi_y]+i[\Pi_y,\Pi_x]).
$$
由 $[\Pi_x,\Pi_y]=iqB$ 得 $[a,a^*]=1$。

**答案 23.3.** 规范变换会改变 $A$ 但不改变磁场 $B=\nabla\times A$，并且 Hamiltonian 由波函数相位酉共轭联系。可观测概率和谱不变，所以单独的 $A$ 依赖规范选择，不是直接可观测量；规范不变量如磁通或 Wilson 环路才有直接物理意义。

## 第二十四章

**答案 24.1.** 在题设共同乘积定义域上，Robertson 关系给出
$$
\Delta A\,\Delta B\ge \frac12|\langle[A,B]\rangle|.
$$
若 $[A,B]\psi=ic\psi$，且态已归一化，则
$$
|\langle[A,B]\rangle|=|ic|=|c|.
$$
因此 $\Delta A\,\Delta B\ge |c|/2$。

**答案 24.2.** 一维中
$$
j=\frac1m\operatorname{Im}(\overline\psi\,\partial_x\psi).
$$
若 $\psi=Ae^{i(kx-\omega t)}$，则 $\partial_x\psi=ik\psi$，所以
$$
j=\frac1m\operatorname{Im}(ik|\psi|^2)=\frac{k}{m}|A|^2.
$$
平面波是动量本征态，动量为 $k$，故概率流等于概率密度乘速度 $k/m$。
这里平面波不属于 $L^2(\mathbb R)$；$|A|^2$ 与 $j$ 是广义态的常数
密度和流，不能把它们解释为整条实线上的归一化概率。

**答案 24.3.** Ehrenfest 定理给出
$$
\frac d{dt}\langle X\rangle=\frac1m\langle P\rangle,\qquad
\frac d{dt}\langle P\rangle=-m\omega^2\langle X\rangle.
$$
对第一式再求导并代入第二式：
$$
\frac{d^2}{dt^2}\langle X\rangle+\omega^2\langle X\rangle=0.
$$

## 第二十五章

**答案 25.1.** 若所有 $V_I(t)$ 两两对易，则时间有序积分中可交换算子顺序。第 $n$ 阶单纯形积分等于立方体积分的 $1/n!$：
$$
\int_{t_0\le s_n\le\cdots\le s_1\le t}V_I(s_1)\cdots V_I(s_n)
=\frac1{n!}\left(\int_{t_0}^tV_I(s)\,ds\right)^n.
$$
因此 Dyson 级数化为
$$
\exp\left(-i\lambda\int_{t_0}^tV_I(s)\,ds\right).
$$

**答案 25.2.** 若 $V$ 时间无关，则
$$
c_f^{(1)}(t)=-iV_{fi}\int_0^t e^{i\omega_{fi}s}\,ds
=V_{fi}\frac{1-e^{i\omega_{fi}t}}{\omega_{fi}}.
$$
其模平方为
$$
|c_f^{(1)}(t)|^2
=|V_{fi}|^2\frac{4\sin^2(\omega_{fi}t/2)}{\omega_{fi}^2}.
$$
当 $\omega_{fi}=0$ 时不能使用显示商式；直接积分或取连续极限得
$$
c_f^{(1)}(t)=-itV_{fi},
\qquad
|c_f^{(1)}(t)|^2=t^2|V_{fi}|^2.
$$

**答案 25.3.** 若 $V_I(t)$ 与 $V_I(s)$ 不对易，则 $V_I(t)V_I(s)$ 和 $V_I(s)V_I(t)$ 是不同算子。演化方程要求较晚时间的算子按演化顺序作用，普通指数会把不同顺序混合为对称组合，不能满足原微分方程。

## 第二十六章

**答案 26.1.** 四个态均由张量积正交基线性组合而成。$|\uparrow\uparrow\rangle$ 与 $|\downarrow\downarrow\rangle$ 的归一性直接成立；两个 $m=0$ 态范数为
$$
\frac12(\langle\uparrow\downarrow|\uparrow\downarrow\rangle+\langle\downarrow\uparrow|\downarrow\uparrow\rangle)=1.
$$
它们内积为
$$
\frac12(1-1)=0.
$$
与 $m=\pm1$ 态正交也由张量积基正交性得到。

**答案 26.2.** 对
$$
|0,0\rangle=\frac{|\uparrow\downarrow\rangle-|\downarrow\uparrow\rangle}{\sqrt2},
$$
$J_z$ 作用给出
$$
\frac1{\sqrt2}\left((1/2-1/2)|\uparrow\downarrow\rangle-( -1/2+1/2)|\downarrow\uparrow\rangle\right)=0.
$$
$J_+$ 作用在第一项只升高第二个自旋，得 $|\uparrow\uparrow\rangle$；作用在第二项只升高第一个自旋，也得 $|\uparrow\uparrow\rangle$，二者因负号相消。$J_-$ 同理相消。

**答案 26.3.** 偶极算子为秩 $1$ 球张量，分量 $q=-1,0,1$。由 $m'=m+q$ 得
$$
\Delta m=0,\pm1.
$$

## 第二十七章

**答案 27.1.** 有
$$
|\psi_0(x)|^2=\frac1{\sqrt{\pi}\sigma}e^{-x^2/\sigma^2}.
$$
积分
$$
\int_{\mathbb R}\frac1{\sqrt{\pi}\sigma}e^{-x^2/\sigma^2}\,dx=1
$$
给出归一化。该密度均值为 $0$，方差为
$$
(\Delta X)^2=\frac{\sigma^2}{2}.
$$
故 $\Delta X=\sigma/\sqrt2$。

**答案 27.2.** 若初态为 $|\uparrow\rangle$，它是 $\sigma_z$ 的 $+1$ 本征态。因 Hamiltonian 与 $\sigma_z$ 对易，态只获得相位。故
$$
\langle\sigma_x\rangle_t=0,\qquad
\langle\sigma_y\rangle_t=0,\qquad
\langle\sigma_z\rangle_t=1.
$$

**答案 27.3.** 共振时
$$
P_{0\to1}(t)=\sin^2\frac{|\Omega|t}{2},
\qquad \Omega\ne0.
$$
第一次达到 $1$ 时 $|\Omega|t/2=\pi/2$，故
$$
t=\frac{\pi}{|\Omega|}.
$$

## 附录 A

**答案 A.1.** 闭算子的定义正是：若 $(\psi_n,A\psi_n)$ 在 $\mathcal H\oplus\mathcal H$ 中收敛到 $(\psi,\eta)$，则 $\psi\in\mathcal D(A)$ 且 $A\psi=\eta$。这等价于图像为闭子空间。

**答案 A.2.** 若 $A=A^*$ 且 $z\notin\mathbb R$，则
$$
\|(A-z)\psi\|\ge |\operatorname{Im}z|\,\|\psi\|.
$$
因此 $A-z$ 单射且逆有界；标准 Hilbert 空间论进一步给出满射，故非实点在 resolvent 集中。

## 附录 B

**答案 B.1.** 在 Schwartz 空间上，Plancherel 公式为
$$
\langle \widehat f,\widehat g\rangle=\langle f,g\rangle.
$$
可先对双积分使用
$$
(2\pi)^{-d}\int e^{ip\cdot(x-y)}\,dp=\delta(x-y)
$$
的分布恒等式，再由密度推广到 $L^2$。

**答案 B.2.** Fourier 变换后自由方程
$$
i\partial_t\psi=-\frac1{2m}\Delta\psi
$$
变为
$$
i\partial_t\widehat\psi(t,p)=\frac{|p|^2}{2m}\widehat\psi(t,p).
$$
所以
$$
\widehat\psi(t,p)=e^{-it|p|^2/2m}\widehat\psi(0,p).
$$

## 附录 C

**答案 C.1.** 直接相乘：
$$
\begin{pmatrix}0&1\\1&0\end{pmatrix}
\begin{pmatrix}0&-i\\i&0\end{pmatrix}
=\begin{pmatrix}i&0\\0&-i\end{pmatrix}=i\sigma_z.
$$

**答案 C.2.** 因 $\sigma_z$ 对角，
$$
e^{-i\theta\sigma_z/2}
=\begin{pmatrix}e^{-i\theta/2}&0\\0&e^{i\theta/2}\end{pmatrix}.
$$
