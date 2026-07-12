# 第二十四章：不确定性、Ehrenfest 定理与概率流

## 本章目标

本章补齐量子力学基础理论的三个核心定理：Robertson 不确定性关系、Schrodinger 方程的概率守恒和 Ehrenfest 定理。为避免无界算子技术遮蔽主线，正文在共同不变稠密核心或光滑快速衰减波函数上证明公式；闭算子层面的推广列入外部边界。

## 依赖前置知识

需要 Hilbert 空间内积、Schrodinger 方程、位置与动量算子、交换子和偏微分方程的分部积分。

## 24.1 方差与不确定性

**定义 24.1.** 设 $A$ 为自伴算子，单位态 $\psi$ 属于 $\mathcal D(A)$。$A$ 在态 $\psi$ 中的期望和方差为
$$
\langle A\rangle_\psi=\langle\psi,A\psi\rangle,\qquad
(\Delta_\psi A)^2=\|(A-\langle A\rangle_\psi)\psi\|^2.
$$

**定理 24.2（Robertson 不确定性关系）.** 设 $A,B$ 为自伴算子，
$\psi\in\mathcal D(AB)\cap\mathcal D(BA)$ 且 $\|\psi\|=1$。则
$$
\Delta_\psi A\,\Delta_\psi B
\ge \frac12\left|\langle\psi,[A,B]\psi\rangle\right|.
$$

**证明.** 令
$$
A_0=A-\langle A\rangle_\psi I,\qquad
B_0=B-\langle B\rangle_\psi I.
$$
由 Cauchy-Schwarz 不等式，
$$
|\langle A_0\psi,B_0\psi\rangle|
\le \|A_0\psi\|\,\|B_0\psi\|
=\Delta_\psi A\,\Delta_\psi B.
$$
另一方面，因为 $A_0,B_0$ 仍对称，
$$
2i\,\operatorname{Im}\langle A_0\psi,B_0\psi\rangle
=\langle A_0\psi,B_0\psi\rangle-\langle B_0\psi,A_0\psi\rangle
=\langle\psi,[A_0,B_0]\psi\rangle.
$$
常数项与一切算子对易，故 $[A_0,B_0]=[A,B]$。于是
$$
\frac12|\langle\psi,[A,B]\psi\rangle|
=|\operatorname{Im}\langle A_0\psi,B_0\psi\rangle|
\le |\langle A_0\psi,B_0\psi\rangle|.
$$
合并两式得到结论。$\square$

**推论 24.3.** 设 $\psi$ 归一化，属于 $XP$ 与 $PX$ 的共同定义域，且位置与动量方差有限。在 $[X,P]\psi=i\psi$ 时，
$$
\Delta_\psi X\,\Delta_\psi P\ge \frac12.
$$

**证明.** 把定理 24.2 应用于 $A=X$、$B=P$。共同定义域假设保证
交换子期望有意义，并且
$\langle\psi,[X,P]\psi\rangle=i\langle\psi,\psi\rangle=i$。
归一化使其模为 $1$，故右端为 $1/2$。$\square$

## 24.2 概率流与连续性方程

**设定 24.4.** 取 $m>0$、$d\ge1$。设 $\psi(t,x)$ 为光滑快速衰减解：
$$
i\partial_t\psi=-\frac1{2m}\Delta\psi+V(x)\psi,
$$
其中 $V$ 为实值函数。定义概率密度和概率流为
$$
\rho=|\psi|^2,\qquad
j=\frac1m\operatorname{Im}(\overline\psi\,\nabla\psi).
$$

**命题 24.5（连续性方程）.** 上述 $\rho,j$ 满足
$$
\partial_t\rho+\nabla\cdot j=0.
$$

**证明.** 由 Schrodinger 方程，
$$
\partial_t\psi=\frac i{2m}\Delta\psi-iV\psi,
\qquad
\partial_t\overline\psi=-\frac i{2m}\Delta\overline\psi+iV\overline\psi.
$$
因此
$$
\partial_t|\psi|^2
=\overline\psi\,\partial_t\psi+\psi\,\partial_t\overline\psi
=\frac i{2m}\overline\psi\Delta\psi-\frac i{2m}\psi\Delta\overline\psi.
$$
势能项因 $V$ 实值相消。另一方面
$$
\nabla\cdot j
=\frac1m\operatorname{Im}(\nabla\overline\psi\cdot\nabla\psi+\overline\psi\Delta\psi)
=\frac1m\operatorname{Im}(\overline\psi\Delta\psi),
$$
而
$$
\frac i{2m}(\overline\psi\Delta\psi-\psi\Delta\overline\psi)
=-\frac1m\operatorname{Im}(\overline\psi\Delta\psi).
$$
故 $\partial_t\rho+\nabla\cdot j=0$。$\square$

**推论 24.6.** 若 $\psi$ 足够快衰减，则总概率守恒：
$$
\frac d{dt}\int_{\mathbb R^d}|\psi(t,x)|^2\,dx=0.
$$

**证明.** 对连续性方程积分并用散度定理。无穷远边界项由快速衰减为零。$\square$

## 24.3 Ehrenfest 定理

**命题 24.7（Ehrenfest 定理，一维形式）.** 设
$$
H=\frac{P^2}{2m}+V(X)
$$
是自伴 Hamiltonian，其中 $V\in C^1(\mathbb R;\mathbb R)$。设归一化解
$\psi(t)=e^{-itH}\psi_0$ 在所考察时间区间内属于一个共同不变稠密子空间
$\mathcal D$，并且在 $\mathcal D$ 上：

1. $HX,XH,HP,PH$ 均有定义；
2. 下文的乘积法则与分部积分成立；
3. $t\mapsto\langle X\rangle_{\psi(t)}$ 与
   $t\mapsto\langle P\rangle_{\psi(t)}$ 可微，且
   $V'(X)\psi(t)\in\mathcal H$。

则
$$
\frac d{dt}\langle X\rangle_{\psi(t)}
=\frac1m\langle P\rangle_{\psi(t)},
\qquad
\frac d{dt}\langle P\rangle_{\psi(t)}
=-\langle V'(X)\rangle_{\psi(t)}.
$$

**证明.** 上述定义域与可微性假设允许在 $\mathcal D$ 上使用 Heisenberg 方程
$$
\frac d{dt}\langle A\rangle=i\langle[H,A]\rangle
$$
对时间无关算子成立。先算
$$
[P^2,X]=P[P,X]+[P,X]P=-2iP,
$$
故
$$
i[H,X]=i\frac1{2m}[P^2,X]=\frac Pm.
$$
再算 $[P,V(X)]$。在测试函数上，
$$
PV\psi=-i(V'\psi+V\psi'),\qquad VP\psi=-iV\psi',
$$
故 $[P,V(X)]=-iV'(X)$。于是
$$
[H,P]=[V(X),P]=iV'(X),
$$
从而
$$
i[H,P]=-V'(X).
$$
取期望值得结论。$\square$

**说明 24.7A（假设的作用）.** “在 Schwartz 空间上形式计算”本身不足以推出
$e^{-itH}$ 保持 Schwartz 空间。命题把所需的不变域和可微性明确列为假设；在给定
势函数类下验证这些条件，属于自伴算子与传播子正则性理论。一个常用的充分框架是：
$V$ 光滑、各阶导数具有适当的多项式增长，并另行证明 Schrodinger 演化保持
$\mathcal S(\mathbb R)$。本书只在条件成立后使用 Ehrenfest 恒等式。

**说明 24.8.** Ehrenfest 定理显示期望值满足类似经典 Hamilton 方程的关系，但一般并不等同于经典轨道，因为
$$
\langle V'(X)\rangle\ne V'(\langle X\rangle)
$$
除非势为至多二次或态足够局域。

## 本章小结

不确定性关系来自内积空间的 Cauchy-Schwarz 不等式和交换子。Schrodinger 方程给出局部概率守恒，其流密度由相位梯度控制。Ehrenfest 定理把量子期望的演化与经典力学联系起来，但不把量子态简化为点粒子轨道。

## 练习

**练习 24.1.** 设 $c\in\mathbb R$，并在包含归一化态 $\psi$ 的
共同乘积定义域上有 $[A,B]\psi=ic\psi$。证明
$\Delta_\psi A\,\Delta_\psi B\ge |c|/2$。

**练习 24.2.** 把平面波
$\psi=Ae^{i(kx-\omega t)}$ 视为广义态，计算其常数概率密度与概率流
$j$，并说明它与动量的关系。

**练习 24.3.** 对谐振子 $V(X)=m\omega^2X^2/2$，用 Ehrenfest 定理推出 $\langle X\rangle$ 满足经典谐振子方程。
