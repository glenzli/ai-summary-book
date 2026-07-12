# 第十六章：绝热定理、Berry 相位与有效动力学

## 本章目标

本章介绍缓慢变化 Hamiltonian 的绝热近似、几何相位和有效动力学。

## 依赖前置知识

需要时间演化、谱投影、扰动理论和微分方程。

## 16.1 绝热设定

**设定 16.1.** 令 $H(s)$ 为光滑自伴矩阵族，$s\in[0,1]$。慢时间演化满足
$$
i\varepsilon\frac{d}{ds}\psi_\varepsilon(s)=H(s)\psi_\varepsilon(s),
\qquad 0<\varepsilon\ll1.
$$

**定义 16.2.** 若 $E_n(s)$ 是孤立本征值，且与其余谱之间有正距离下界，则称该能级有谱隙。

**外部输入定理 16.3（绝热定理，QM-EXT-7）.** 在光滑性和谱隙假设下，若初态位于 $E_n(0)$ 的本征空间，则演化到时刻 $s$ 后仍在 $E_n(s)$ 的瞬时本征空间附近，误差为 $O(\varepsilon)$。

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

## 16.3 有效 Hamiltonian

**定义 16.7.** 在瞬时本征态 $u_n(s)$ 的绝热子空间中，有效相位由动力学相位和几何相位组成：
$$
\psi(s)\approx
\exp\left(-\frac{i}{\varepsilon}\int_0^sE_n(r)\,dr\right)
\exp\left(i\int_0^s\mathcal A_n(r)\,dr\right)u_n(s).
$$

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

**例子 16.11.** 自旋 $1/2$ 在缓慢旋转的磁场方向 $n(s)\in S^2$ 中，瞬时低能态沿球面闭合曲线获得 Berry 相位。该相位等于相应立体角的一半并带符号，精确推导需要球面自旋本征向量的局部规范；本书在此只记录其几何含义。

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

## 本章小结

绝热理论说明有谱隙的慢变系统保留瞬时能级。Berry 相位是本征向量线丛的几何相位，不由能量本征值单独决定。严格误差估计依赖绝热定理。

## 练习

**练习 16.1.** 证明归一化本征向量满足 $\langle u,\dot u\rangle$ 纯虚。

**练习 16.2.** 解释 Berry 相位为何只在闭合回路上具有规范不变量意义。
