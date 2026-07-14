# 第十六章：绝热定理、Berry 相位与有效动力学

若磁场方向缓慢转一圈，自旋可以始终接近瞬时能量本征态，末态却不只
积累能量积分产生的动力学相位。沿参数回路选择本征向量时还会积累一个
只由路径几何决定的相位；它在局部可以通过相位规范消去，在闭合回路上
却留下不可消除的 holonomy。这里“缓慢”不是口头描述，而是方程
$i\varepsilon\dot\psi=H(s)\psi$ 中的小参数，并且只有持续存在的谱隙才能
抑制向其他瞬时谱子空间的跃迁。

本章在有限维光滑矩阵族中固定绝热设定，把带谱隙的绝热定理作为有明确
范数误差的外部输入。非简并本征线上的 Berry 连接随后由归一化条件自然
出现，其规范变换律说明开路径积分为什么依赖端点选择。平行移动规范把
局部连接设为零，而自旋 $1/2$ 沿定纬圆的显式计算将给出半个立体角，
展示局部规范与闭路相位如何同时成立。

## 16.1 绝热设定

**设定 16.1.** 令 $H(s)$ 为光滑自伴矩阵族，$s\in[0,1]$。慢时间演化满足
$$
i\varepsilon\frac{d}{ds}\psi_\varepsilon(s)=H(s)\psi_\varepsilon(s),
\qquad 0<\varepsilon\ll1.
$$

**定义 16.2.** 若 $E_n(s)$ 是孤立本征值，且与其余谱之间有正距离下界，则称该能级有谱隙。

**外部输入定理 16.3（绝热定理，QM-EXT-7）.** 设 $H(s)$ 具有绝热
定理所需的充分光滑性，$P_n(s)$ 是与其余谱有一致正谱隙且秩不变的谱
投影。若 $\psi_\varepsilon(0)\in\operatorname{Ran}P_n(0)$ 且
$\|\psi_\varepsilon(0)\|=1$，则存在与 $\varepsilon$ 无关的常数 $C$，
使
$$
\sup_{0\le s\le1}
\|(I-P_n(s))\psi_\varepsilon(s)\|\le C\varepsilon.
$$
这里 $\psi_\varepsilon$ 解设定 16.1 的方程。定理控制的是离开瞬时谱
子空间的范数，不直接选择该子空间内部的相位或简并基。

绝热定理保住谱子空间，却没有给出非简并本征线内的相位。把解投影到一
个光滑归一化本征向量上，会同时出现能量积分和由基向量自身变化产生的
连接项。

## 16.2 Berry 相位

**定义 16.4.** 设 $E_n(s)$ 非简并，取光滑归一化本征向量 $u_n(s)$。Berry 连接定义为
$$
\mathcal A_n(s)=i\langle u_n(s),\dot u_n(s)\rangle.
$$
沿闭合回路的 Berry 相位为
$$
\gamma_n=\oint \mathcal A_n(s)\,ds.
$$

**命题 16.5.** 在规范变换 $u_n(s)\mapsto e^{i\chi(s)}u_n(s)$ 下，
$$
\mathcal A_n\mapsto \mathcal A_n-\dot\chi.
$$

**证明.** 令 $v=e^{i\chi}u$，则
$$
\dot v=e^{i\chi}(i\dot\chi\,u+\dot u).
$$
于是
$$
i\langle v,\dot v\rangle
=i\langle u,i\dot\chi\,u+\dot u\rangle
=-\dot\chi+i\langle u,\dot u\rangle.
$$
$\square$

**推论 16.6.** 闭合回路的 Berry 相位在模 $2\pi$ 意义下规范不变。

**证明.** 由前一命题，规范变换使 Berry connection 的回路积分改变
$-[\chi(1)-\chi(0)]$。闭合回路上的规范选择必须给出同一个末端射线；
若选取单值归一化本征向量，则
$e^{i\chi(1)}=e^{i\chi(0)}$，所以
$\chi(1)-\chi(0)\in2\pi\mathbb Z$。因此相位类在
$\mathbb R/2\pi\mathbb Z$ 中不变。$\square$

把动力学相位与几何相位重新合并，便得到绝热子空间中的首阶有效演化。
这一表达式仍继承定理 16.3 的谱隙与光滑性条件。

## 16.3 有效 Hamiltonian

**定义 16.7.** 在瞬时本征态 $u_n(s)$ 的绝热子空间中，有效相位由动力学相位和几何相位组成：
$$
\psi(s)\approx
\exp\left(-\frac{i}{\varepsilon}\int_0^sE_n(r)\,dr\right)
\exp\left(i\int_0^s\mathcal A_n(r)\,dr\right)u_n(s).
$$

连接依赖本征向量的相位选择。沿一个不闭合的参数区间总能解一个标量
微分方程把连接设为零；闭合回路的障碍恰好由末端相对于初端的相位记录。

## 16.4 平行移动规范

**定义 16.8.** 若归一化本征向量族满足
$$
\langle u_n(s),\dot u_n(s)\rangle=0,
$$
则称其处于平行移动规范。

**命题 16.9.** 任意光滑归一化本征向量族局部可通过相位变换化为平行移动规范。

**证明.** 令 $v(s)=e^{i\chi(s)}u(s)$。由第 16.2 节计算，
$$
i\langle v,\dot v\rangle=i\langle u,\dot u\rangle-\dot\chi.
$$
要求 $\langle v,\dot v\rangle=0$ 等价于
$$
\dot\chi=i\langle u,\dot u\rangle.
$$
因 $\langle u,\dot u\rangle$ 纯虚，右边为实函数，局部积分即可得到实值 $\chi$。$\square$

**说明 16.10.** 对闭合回路，局部平行移动规范未必能使末端向量等于初端向量；二者的相位差正是 Berry 相位。这说明 Berry 相位不是单纯的规范选择错误，而是线丛沿回路的 holonomy。

**例子 16.11（自旋 $1/2$ 的定纬回路）.** 取
$$
H(\theta,\varphi)=\frac B2\,n(\theta,\varphi)\cdot\sigma,
\qquad B>0,
$$
并固定 $0\le\theta<\pi$，让 $\varphi$ 从 $0$ 缓慢增加到 $2\pi$。
本征值 $-B/2$ 的一个归一化本征向量可取
$$
u_-(\theta,\varphi)=
\begin{pmatrix}
-e^{-i\varphi}\sin(\theta/2)\\
\cos(\theta/2)
\end{pmatrix}.
$$
直接求导得到
$$
i\langle u_-,\partial_\varphi u_-\rangle
=\sin^2\frac\theta2
=\frac{1-\cos\theta}{2}.
$$
因此闭路 Berry 相位为
$$
\gamma_-=\int_0^{2\pi}\frac{1-\cos\theta}{2}\,d\varphi
=\pi(1-\cos\theta)=\frac{\Omega_{\mathrm{solid}}}{2}
\pmod{2\pi},
$$
其中 $\Omega_{\mathrm{solid}}=2\pi(1-\cos\theta)$ 是该定纬圆围成的有向
立体角。本征向量规范或回路方向改变时符号约定会相应改变，但模
$2\pi$ 的闭路相位不变。

**定义 16.12.** 若参数空间坐标为 $R=(R^1,\dots,R^d)$，Berry 曲率形式写作
$$
F=d\mathcal A,
$$
局部坐标中
$$
F_{ab}=\partial_a\mathcal A_b-\partial_b\mathcal A_a.
$$
闭合曲线上的 Berry 相位可在可取曲面 $\Sigma$ 时写成
$$
\gamma=\int_\Sigma F
$$
模 $2\pi$。

绝热定理以 $O(\varepsilon)$ 范数界控制态离开带隙瞬时谱子空间的部分；
动力学相位来自能量积分，Berry 相位则来自本征线的连接。平行移动可以
局部消去连接，却不能消去闭路 holonomy，自旋定纬回路把它算成半个
立体角。下一章不再假设所研究系统封闭，而是把环境自由度取偏迹，观察
纯整体态怎样产生混合约化态。

## 练习

**练习 16.1.** 证明归一化本征向量满足 $\langle u,\dot u\rangle$ 纯虚。

**练习 16.2.** 解释 Berry 相位为何只在闭合回路上具有规范不变量意义。
