# 第七章：谐振子、升降算符与 Hermite 基

## 本章目标

本章计算一维量子谐振子的能谱、基态、升降算符和数算符结构。

## 依赖前置知识

需要正则对易关系、微分方程和 Hilbert 空间正交分解。

## 7.1 Hamiltonian 与升降算符

**定义 7.1.** 在 $\hbar=1$ 下，一维谐振子 Hamiltonian 为
$$
H=\frac{P^2}{2m}+\frac12m\omega^2X^2.
$$
定义
$$
a=\sqrt{\frac{m\omega}{2}}X+\frac{i}{\sqrt{2m\omega}}P,\qquad
a^*=\sqrt{\frac{m\omega}{2}}X-\frac{i}{\sqrt{2m\omega}}P.
$$

**命题 7.2.** 在共同不变核心上，
$$
[a,a^*]=I,\qquad H=\omega\left(a^*a+\frac12I\right).
$$

**证明.** 使用 $[X,P]=iI$。令 $\alpha=\sqrt{m\omega/2}$、$\beta=1/\sqrt{2m\omega}$。则
$$
[a,a^*]=[\alpha X+i\beta P,\alpha X-i\beta P]
=-i\alpha\beta[X,P]+i\alpha\beta[P,X]=2\alpha\beta I=I.
$$
展开 $a^*a$：
$$
a^*a=\alpha^2X^2+\beta^2P^2+i\alpha\beta(XP-PX)
=\frac{m\omega}{2}X^2+\frac{1}{2m\omega}P^2-\frac12I.
$$
乘以 $\omega$ 得 Hamiltonian 公式。$\square$

## 7.2 能级

**定义 7.3.** 数算符为 $N=a^*a$。若 $N\psi=n\psi$，称 $\psi$ 为数态。

**命题 7.4.** 若 $N\psi=n\psi$ 且 $\|\psi\|=1$，则 $n\ge0$。若 $a\psi\ne0$，则 $a\psi$ 是本征值 $n-1$ 的本征向量；若 $a^*\psi\ne0$，则 $a^*\psi$ 是本征值 $n+1$ 的本征向量。

**证明.** 首先
$$
n=\langle\psi,N\psi\rangle=\langle a\psi,a\psi\rangle=\|a\psi\|^2\ge0.
$$
由 $[N,a]=-a$ 与 $[N,a^*]=a^*$，
$$
N(a\psi)=aN\psi-a\psi=(n-1)a\psi,
$$
并且
$$
N(a^*\psi)=a^*N\psi+a^*\psi=(n+1)a^*\psi.
$$
$\square$

**命题 7.5（本征值的量子化）.** 设 $\psi$ 属于 $a,a^*$ 的共同不变核心，所有有限次降阶仍在该核心中，且 $N\psi=n\psi$、$\|\psi\|=1$。则 $n\in\mathbb Z_{\ge0}$，相应能量只能为
$$
E_n=\omega\left(n+\frac12\right),\qquad n=0,1,2,\dots.
$$

**证明.** 由命题 7.4，$n\ge0$。对每个 $k\ge1$，反复使用
$aa^*=a^*a+I$ 得
$$
\|a^k\psi\|^2=n(n-1)\cdots(n-k+1).
$$
若 $n$ 不是非负整数，取 $k=\lfloor n\rfloor+2$，右端恰有一个负
因子而其余因子为正，矛盾于范数平方非负。因此 $n$ 是非负整数。
最后代入 $H=\omega(N+I/2)$。$\square$

## 7.3 基态波函数

**命题 7.6.** 基态满足 $a\psi_0=0$，在位置表象中可取
$$
\psi_0(x)=\left(\frac{m\omega}{\pi}\right)^{1/4}e^{-m\omega x^2/2}.
$$

**证明.** 在位置表象 $P=-i\,d/dx$，方程 $a\psi_0=0$ 化为
$$
\sqrt{\frac{m\omega}{2}}x\psi_0+\frac{1}{\sqrt{2m\omega}}\psi_0'=0.
$$
即 $\psi_0'=-m\omega x\psi_0$，解为 $Ce^{-m\omega x^2/2}$。归一化常数由 Gaussian 积分
$$
\int_{\mathbb R}e^{-m\omega x^2}\,dx=\sqrt{\frac{\pi}{m\omega}}
$$
确定。$\square$

## 7.4 相干态

**定义 7.7.** 谐振子的相干态是湮灭算符 $a$ 的本征态：
$$
a|\alpha\rangle=\alpha|\alpha\rangle,\qquad \alpha\in\mathbb C.
$$
在数态基中形式上写作
$$
|\alpha\rangle=e^{-|\alpha|^2/2}\sum_{n=0}^\infty\frac{\alpha^n}{\sqrt{n!}}|n\rangle.
$$

**命题 7.8.** 上述级数定义归一化态，并满足 $a|\alpha\rangle=\alpha|\alpha\rangle$。

**证明.** 范数为
$$
e^{-|\alpha|^2}\sum_{n=0}^\infty\frac{|\alpha|^{2n}}{n!}=1.
$$
又 $a|n\rangle=\sqrt n\,|n-1\rangle$，故
$$
a|\alpha\rangle
=e^{-|\alpha|^2/2}\sum_{n=1}^\infty
\frac{\alpha^n\sqrt n}{\sqrt{n!}}|n-1\rangle
=\alpha e^{-|\alpha|^2/2}\sum_{m=0}^\infty
\frac{\alpha^m}{\sqrt{m!}}|m\rangle.
$$
这正是 $\alpha|\alpha\rangle$。$\square$

**说明 7.9.** 相干态在谐振子演化下保持相干态形式，中心按经典谐振子轨道运动。因此它们是连接量子波包与经典极限的标准例子。

**外部输入定理 7.10（Hermite 函数完备性，QM-EXT-14）.** 由基态反复施加 $a^*$ 得到的 Hermite 函数族构成 $L^2(\mathbb R)$ 的正交归一基。因此谐振子的数态展开对任意平方可积态完备。

**推论 7.10A（完整谱）.** 一维谐振子自伴 Hamiltonian 的谱恰为
$\{\omega(n+\tfrac12):n\in\mathbb Z_{\ge0}\}$，每个本征值重数为一。

**证明.** 命题 7.6 给出 $n=0$ 的归一化本征态；反复施加 $a^*$
给出每个 $n\ge0$ 的非零本征态。命题 7.5 排除其他点谱，而外部输入
7.10 的完备性排除与这些 Hermite 本征态正交的连续谱或剩余谱部分。
一维 Hermite 基中每个次数只有一个归一化向量到相位，故重数为一。
$\square$

## 本章小结

谐振子是量子力学中最重要的可解模型。升降算符把二阶微分算子谱问题转化为代数问题，给出等间距能级、零点能和 Hermite 函数基。

## 练习

**练习 7.1.** 证明 $[N,a]=-a$ 与 $[N,a^*]=a^*$。

**练习 7.2.** 恢复 $\hbar$，写出谐振子能级。
