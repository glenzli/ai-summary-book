# 第二十三章：电磁耦合、规范变换与磁场

电磁场进入量子力学时，$P$ 不再等于质量乘速度；可测的动力学动量是
$\Pi=P-qA(X)$。矢势 $A$ 又不是唯一的，改变
$A\mapsto A+\nabla\chi$ 必须同时改变波函数相位，才能保持所有物理
预测。这使最小耦合不仅是把经典公式中的 $p$ 换成算子，还必须回答平方
微分表达式在哪个定义域上自伴，以及含时相位变换为何给 Hamiltonian
多出 $i\dot U U^{-1}$ 项。

本章先用闭二次型给磁 Schrodinger 表达式指定自伴实现，再分别证明时间
无关与含时规范协变性。一个常矢势的广义平面波计算会区分规范相关的
正则动量和不变的动力学动量。匀强磁场随后把二维问题纤维化为一族平移
后的谐振子，从而得到完整 Landau 谱与无限简并。最后的
Aharonov--Bohm 讨论说明，在非单连通区域中局部为零的磁场仍可能留下
全局环路相位。

## 23.1 最小耦合

**定义 23.1（最小耦合微分表达式）.** 在
$\mathcal H=L^2(\mathbb R^d)$ 上取 $m>0$、$q\in\mathbb R$ 和
$P=-i\nabla$。对足够光滑的实势
$A:\mathbb R^d\to\mathbb R^d$、$\Phi:\mathbb R^d\to\mathbb R$，
最小耦合的形式微分表达式为
$$
H=\frac1{2m}(P-qA(X))^2+q\Phi(X).
$$
其中 $A(X)$ 和 $\Phi(X)$ 是乘法算子，平方表示各分量平方之和。记
$$
\Pi=P-qA(X)
$$
为动力学动量。该表达式本身没有指定自伴算子的定义域；下述二次型
口径才定义 Hamiltonian。

**外部输入定理 23.2（磁 Schrodinger 算子自伴性，QM-EXT-11）.**
设 $A\in L^2_{\mathrm{loc}}(\mathbb R^d;\mathbb R^d)$，
$\Phi\in L^1_{\mathrm{loc}}(\mathbb R^d;\mathbb R)$，
并在 $C_c^\infty(\mathbb R^d)$ 上定义磁二次型
$$
q_{A,\Phi}[\psi]
=\frac1{2m}\int_{\mathbb R^d}|(-i\nabla-qA)\psi|^2\,dx
+\int_{\mathbb R^d}q\Phi|\psi|^2\,dx.
$$
若该形式稠定、下有界且可闭，则其闭包唯一表示一个下有界自伴算子
$H_{A,\Phi}$。QM-EXT-11 提供保证这些形式假设成立的具体局部可积性
与负部相对形式有界条件；正文后续凡写 $H_{A,\Phi}$，均指这一自伴
实现，而不是未指定定义域的形式微分表达式。

自伴实现给出演化，但不同的势对 $(A,\Phi)$ 可以表示同一电磁场。规范
变换必须连同态矢量一起作用；下面先在共同测试域验证动力学动量的协变，
再把等式提升到闭二次型与自伴算子。

## 23.2 规范协变性

**定义 23.3.** 规范变换由光滑实函数 $\chi$ 给出：
$$
A\mapsto A+\nabla\chi,\qquad
\Phi\mapsto \Phi-\partial_t\chi.
$$
波函数同时变换为
$$
\psi\mapsto e^{iq\chi}\psi.
$$

**命题 23.4.** 设 $\chi$ 与 $A$ 足够光滑。对
$\psi\in C_c^\infty(\mathbb R^d)$，时间无关规范变换下的动力学动量满足
$$
(P-q(A+\nabla\chi))e^{iq\chi}\psi
=e^{iq\chi}(P-qA)\psi.
$$

**证明.** 使用 $P=-i\nabla$：
$$
P(e^{iq\chi}\psi)
=-i(iq\nabla\chi\,e^{iq\chi}\psi+e^{iq\chi}\nabla\psi)
=e^{iq\chi}(q\nabla\chi\,\psi+P\psi).
$$
再减去 $q(A+\nabla\chi)e^{iq\chi}\psi$，$\nabla\chi$ 项相消，得到结论。$\square$

**推论 23.5.** 设 $\chi$ 足够光滑，乘法酉算子
$U_\chi\psi=e^{iq\chi}\psi$ 满足
$$
U_\chi\mathcal Q(q_{A,\Phi})
=\mathcal Q(q_{A+\nabla\chi,\Phi}),
$$
其中 $\mathcal Q(q)$ 表示闭二次型的定义域。
则时间无关规范变换下 Hamiltonian 由酉共轭联系：
$$
H_{A+\nabla\chi,\Phi}
=U_\chi H_{A,\Phi}U_\chi^{-1}.
$$
因此谱不变。

**证明.** 命题 23.4 以及标量势与 $U_\chi$ 的对易性给出
$$
q_{A+\nabla\chi,\Phi}[U_\chi\psi]
=q_{A,\Phi}[\psi]
$$
对 $C_c^\infty$ 中的 $\psi$ 成立。由闭包与所声明的形式定义域等式，
该恒等式延拓到整个闭形式域。闭半有界二次型表示定理的唯一性于是
给出相应自伴算子的酉共轭式；酉等价的自伴算子具有相同谱。$\square$

**命题 23.5A（含时规范协变性）.** 设 $\chi(t,x)$ 足够光滑，令
$$
A'=A+\nabla\chi,
\qquad
\Phi'=\Phi-\partial_t\chi.
$$
设 $U_\chi(t)=e^{iq\chi(t,X)}$ 满足
$$
U_\chi(t)\mathcal D(H_{A,\Phi}(t))
=\mathcal D(H_{A',\Phi'}(t)),
$$
且下式涉及的强导数存在。
则在本书 $\hbar=1$ 的单位下
$$
H_{A',\Phi'}(t)
=U_\chi(t)H_{A,\Phi}(t)U_\chi(t)^{-1}
+i\dot U_\chi(t)U_\chi(t)^{-1}.
$$
因此若 $i\partial_t\psi=H_{A,\Phi}\psi$，则 $\psi'=U_\chi\psi$ 满足 $i\partial_t\psi'=H_{A',\Phi'}\psi'$。

**证明.** 命题 23.4 的逐时刻计算给出动能项的酉共轭，而
$$
i\dot U_\chi U_\chi^{-1}=-q\,\partial_t\chi
$$
恰好把标势项 $q\Phi$ 变为 $q\Phi'$。对 $\psi'=U_\chi\psi$ 求导，
$$
i\partial_t\psi'
=i\dot U_\chi\psi+U_\chi H_{A,\Phi}\psi
=H_{A',\Phi'}\psi',
$$
即得协变性。$\square$

**例子 23.5B（常矢势与动量标签）.** 在一维取初始
$A=0$、$\Phi=0$，并令 $\chi(x)=ax$，其中 $a\in\mathbb R$。规范变换后
$A'=a$，而
$$
U_\chi e^{ipx}=e^{iqax}e^{ipx}=e^{i(p+qa)x}.
$$
这些平面波按广义态理解。变换后的正则动量本征值从 $p$ 变为 $p+qa$，
但
$$
(P-qa)e^{i(p+qa)x}=p\,e^{i(p+qa)x},
$$
所以动力学动量和能量 $p^2/(2m)$ 不变。正则动量标签的改变本身不是
可观测效应。

常矢势例子只有规范重标记；真正非零的磁场使动力学动量不同分量不再
对易。这个交换子把二维轨道问题转化为谐振子代数。

## 23.3 匀强磁场与 Landau 能级

**设定 23.6.** 在 $L^2(\mathbb R^2)$ 上取垂直于平面的匀强磁场
$B$，并采用命题 23.2 给出的标准自伴实现。在
$\mathcal S(\mathbb R^2)$ 上，动力学动量满足
$$
[\Pi_x,\Pi_y]=iqB.
$$

**命题 23.7（Landau 谱）.** 若 $qB\ne0$，Hamiltonian
$$
H_B=\frac1{2m}(\Pi_x^2+\Pi_y^2)
$$
的谱为
$$
\sigma(H_B)
=\left\{\frac{|qB|}{m}\left(n+\frac12\right):
n\in\mathbb Z_{\ge0}\right\},
$$
且每个谱点都有无限重数。对 $qB>0$，回旋升降算子可取
$$
a=\frac{1}{\sqrt{2qB}}(\Pi_x+i\Pi_y),\qquad
a^*=\frac{1}{\sqrt{2qB}}(\Pi_x-i\Pi_y).
$$

**证明.** 对 $qB>0$，由 $[\Pi_x,\Pi_y]=iqB$ 得
$[a,a^*]=I$。并且在 Schwartz 核心上
$$
a^*a=\frac1{2qB}(\Pi_x^2+\Pi_y^2-qB).
$$
故
$$
H_B=\frac{qB}{m}\left(a^*a+\frac12\right).
$$
为确认这条代数恒等式给出完整谱，取 Landau 规范
$A=(0,Bx)$，并对 $y$ 作部分 Fourier 变换。此时
$$
H_B\cong\int_{\mathbb R}^{\oplus}h(k)\,dk,
\qquad
h(k)=\frac1{2m}\left(P_x^2+(k-qBx)^2\right).
$$
平移 $x\mapsto x-k/(qB)$ 后，每个 $h(k)$ 都酉等价于频率
$\omega_c=|qB|/m$ 的一维谐振子。由 Hermite 完备性
QM-EXT-14，其谱恰为 $|qB|(n+1/2)/m$，且与 $k$ 无关。直积分
因此没有其他谱；任取 $L^2(\mathbb R_k)$ 系数叠加同一 $n$ 的纤维
本征函数，得到该谱点的无限维本征空间。$qB<0$ 的情形由同一纤维
论证给出，或交换升降算子的选取。
$\square$

**说明 23.7A.** 若 $qB=0$，则 $H_B=P^2/(2m)$，其谱为
$[0,\infty)$，不能把命题 23.7 的升降算子延拓到该点。有限区域中的
Landau 简并计数还依赖边界条件与磁通。

Landau 能级由局部非零磁场决定。若粒子可达区域的磁场处处为零，单连通
区域内通常可局部消去 $A$；带洞区域却还保留不能由单值规范函数消去的
环路积分。

## 23.4 Aharonov-Bohm 边界

**说明 23.8.** 在非单连通区域中，即使磁场 $B=\nabla\times A$ 在粒子可达区域为零，环路积分
$$
\oint A\cdot dx
$$
仍可通过 $\exp(iq\oint A\cdot dx)$ 给出可观测相位。这是
Aharonov-Bohm 效应。严格处理需要带洞区域上的自伴扩张、规范丛或
边界条件分析，本书只记录其规范相位机制。

最小耦合的严格对象是闭二次型所表示的自伴算子。时间无关规范变换给出
酉等价，含时规范变换还包含 $i\dot U U^{-1}$，所以只比较逐时刻谱并不
具有规范不变意义。常矢势计算展示了正则动量标签与动力学动量的区别，
Landau 纤维分解则把非零磁场化为完整谐振子谱；非单连通区域还可能保留
Aharonov--Bohm 环路相位。下一章从这些模型中抽出三条普遍结论：方差
下界、局部概率守恒和期望值运动方程。

## 练习

**练习 23.1.** 证明磁场 $B=\nabla\times A$ 在规范变换 $A\mapsto A+\nabla\chi$ 下不变。

**练习 23.2.** 验证命题 23.7 中 $[a,a^*]=1$。

**练习 23.3.** 说明为什么规范相关的 $A$ 本身不是直接可观测量。
