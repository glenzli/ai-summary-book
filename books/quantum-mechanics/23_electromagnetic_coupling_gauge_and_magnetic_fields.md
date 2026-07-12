# 第二十三章：电磁耦合、规范变换与磁场

## 本章目标

本章介绍非相对论粒子在外电磁场中的 Hamiltonian、最小耦合、规范协变性、Landau 能级和 Aharonov-Bohm 效应的数学边界。

## 依赖前置知识

需要动量算子、酉变换、对易关系、谐振子和角动量。

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

## 23.4 Aharonov-Bohm 边界

**说明 23.8.** 在非单连通区域中，即使磁场 $B=\nabla\times A$ 在粒子可达区域为零，环路积分
$$
\oint A\cdot dx
$$
仍可通过 $\exp(iq\oint A\cdot dx)$ 给出可观测相位。这是
Aharonov-Bohm 效应。严格处理需要带洞区域上的自伴扩张、规范丛或
边界条件分析，本书只记录其规范相位机制。

## 本章小结

电磁场通过最小耦合进入 Hamiltonian。时间无关规范变换给出 Hamiltonian 的酉等价，因而保持谱；含时规范变换则保持完整 Schrodinger 演化的协变性，但两个逐时刻 Hamiltonian 一般不由单纯酉共轭联系，其瞬时谱无须相同。匀强磁场给出 Landau 能级；非单连通空间中的规范势可产生 Aharonov-Bohm 相位。

## 练习

**练习 23.1.** 证明磁场 $B=\nabla\times A$ 在规范变换 $A\mapsto A+\nabla\chi$ 下不变。

**练习 23.2.** 验证命题 23.7 中 $[a,a^*]=1$。

**练习 23.3.** 说明为什么规范相关的 $A$ 本身不是直接可观测量。
