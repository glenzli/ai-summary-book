# 第二十二章：中心势、氢原子与球谐函数

氢原子的三维波函数依赖三个空间变量，直接求解偏微分方程看似比一维
势阱困难得多；Coulomb 势只依赖 $r=|x|$，却提供了决定性的旋转对称。
Hamiltonian 与 $L^2,L_z$ 对易后，球谐函数承担全部角向变化，剩余问题
化为带离心势的一维半轴方程。量子数 $\ell,m$ 因而不是分离变量时随意
引入的标签，而是共同谱投影的本征值数据。

本章先在明确共同测试域上验证中心势与角动量对易，再调用球谐完备性把
角向 Hilbert 空间分解。替换 $u=rR$ 消去径向方程的一阶导数，并显示
$\ell(\ell+1)/(2mr^2)$ 的离心项。Coulomb 全谱、自伴性与 Laguerre
完备性仍作为外部输入，但氢型基态会在书内完成归一化并逐项代回微分
方程，直接核对能量 $-\mu Z^2e^4/2$ 和 Bohr 长度之间的关系。

## 22.1 中心势与角动量守恒

**定义 22.1.** 三维中心势的形式微分表达式是
$$
H=-\frac1{2m}\Delta+V(r),\qquad r=|x|,
$$
其中 $V$ 只依赖径向变量。轨道角动量为
$$
L=x\times P,\qquad P=-i\nabla.
$$
只有在另行指定自伴定义域或闭二次型后，才把该表达式称为
Hamiltonian。下述交换子先在避开势奇点的共同测试域上理解。

**命题 22.2.** 设
$V\in C^1((0,\infty);\mathbb R)$。在共同定义域
$C_c^\infty(\mathbb R^3\setminus\{0\})$ 上，中心势微分表达式与
$L^2$、$L_z$ 对易。若某个自伴实现的定义域在旋转下不变，则相应谱
意义的旋转对称性由该实现继承。

**证明.** Laplacian 在旋转下不变，故与旋转生成元 $L_i$ 对易。可直接按分量验证：
$$
L_i=-i\sum_{j,k}\epsilon_{ijk}x_j\partial_k.
$$
因 $\Delta$ 与所有旋转生成元交换，$[-\Delta,L_i]=0$。又 $V(r)$ 为径向乘法算子。沿旋转生成元求导径向函数为零：
$$
L_i(V(r)f)=V(r)L_if-i\sum_{j,k}\epsilon_{ijk}x_j(\partial_kV(r))f.
$$
这里 $\partial_kV(r)=V'(r)x_k/r$，而 $\sum_{j,k}\epsilon_{ijk}x_jx_k=0$，故 $[V(r),L_i]=0$。于是 $[H,L_i]=0$，从而 $[H,L^2]=0$ 且 $[H,L_z]=0$。$\square$

对易关系允许把定态解同时选为 $L^2,L_z$ 本征函数。球谐完备性保证这
不只是找到一批特殊角向解，而是覆盖整个 $L^2(S^2)$ 角向空间。

## 22.2 球谐函数

**定义 22.3.** 球谐函数 $Y_\ell^m$ 是单位球面 $S^2$ 上同时满足
$$
L^2Y_\ell^m=\ell(\ell+1)Y_\ell^m,\qquad
L_zY_\ell^m=mY_\ell^m
$$
的归一化函数，其中 $\ell=0,1,2,\dots$，$m=-\ell,\dots,\ell$。

**外部输入定理 22.4（球谐完备性，QM-EXT-10）.** 族 $\{Y_\ell^m\}$ 构成 $L^2(S^2)$ 的正交归一基。

**命题 22.5.** 若 $\psi(r,\Omega)=R(r)Y_\ell^m(\Omega)$，则
$$
\Delta\psi=
\left(\frac1{r^2}\frac d{dr}r^2\frac{dR}{dr}
-\frac{\ell(\ell+1)}{r^2}R\right)Y_\ell^m.
$$

**证明.** 球坐标中 Laplacian 分解为
$$
\Delta=\frac1{r^2}\frac\partial{\partial r}r^2\frac\partial{\partial r}
-\frac{L^2}{r^2}.
$$
作用在 $R(r)Y_\ell^m(\Omega)$ 上时，径向部分只作用于 $R$，角向部分用 $L^2Y_\ell^m=\ell(\ell+1)Y_\ell^m$，得到公式。$\square$

分离变量后仍有径向一阶导数和三维体积元 $r^2dr\,d\Omega$。令
$u=rR$ 同时把径向范数化为 $\int_0^\infty|u|^2dr$，并把方程改写成
半轴上的一维 Schrodinger 形式。

## 22.3 径向方程

**定义 22.6.** 对中心势定态方程 $H\psi=E\psi$，令
$$
\psi(r,\Omega)=R(r)Y_\ell^m(\Omega),\qquad u(r)=rR(r).
$$
则径向方程为
$$
-\frac1{2m}u''(r)+\left(V(r)+\frac{\ell(\ell+1)}{2mr^2}\right)u(r)=Eu(r).
$$

**命题 22.7.** 上述径向方程由三维定态方程推出。

**证明.** 由命题 22.5，
$$
-\frac1{2m}\left(R''+\frac2rR'-\frac{\ell(\ell+1)}{r^2}R\right)+V R=ER.
$$
代入 $R=u/r$。直接计算
$$
R''+\frac2rR'=\frac{u''}{r}.
$$
乘以 $r$ 后得到
$$
-\frac1{2m}u''+\frac{\ell(\ell+1)}{2mr^2}u+Vu=Eu.
$$
$\square$

一般中心势的径向方程仍不可解。Coulomb 势的特殊尺度对称与多项式终止
条件给出 $1/n^2$ 谱；完整谱论作为外部输入，但最低态可以直接验证。

## 22.4 氢原子能级

**定义 22.8.** 氢型原子的 Coulomb Hamiltonian 为
$$
H=-\frac1{2\mu}\Delta-\frac{Z e^2}{r},
$$
其中 $\mu$ 为约化质量，且沿用序章的 $\hbar=1$ 约定。这里的 $e^2$ 表示库仑耦合常数；若使用 SI 单位，应把它替换为 $e^2/(4\pi\varepsilon_0)$，并恢复相应的 $\hbar$ 因子。

**公式 22.9（氢型束缚态能级）.** 束缚态能级为
$$
E_n=-\frac{\mu Z^2e^4}{2n^2},\qquad n=1,2,\dots.
$$
每个 $n$ 对应 $\ell=0,\dots,n-1$ 与 $m=-\ell,\dots,\ell$。

**说明 22.10.** 公式 22.9 的严格推导需要 Coulomb Hamiltonian 的自伴性、径向方程的边界条件和 Laguerre 多项式解的完备性。本书把自伴性与完备性列为外部输入；径向方程和量子数关系在本章内部给出。

**例子 22.10A（氢型基态核对）.** 令
$$
a=\frac1{\mu Ze^2},\qquad
\psi_{100}(x)=\frac1{\sqrt{\pi a^3}}e^{-r/a}.
$$
利用
$\int_0^\infty r^2e^{-2r/a}\,dr=a^3/4$ 得
$$
\|\psi_{100}\|^2
=4\pi\frac1{\pi a^3}\frac{a^3}{4}=1.
$$
对 $r>0$，
$$
\Delta e^{-r/a}
=\left(\frac1{a^2}-\frac2{ar}\right)e^{-r/a}.
$$
因此
$$
\left(-\frac1{2\mu}\Delta-\frac{Ze^2}{r}\right)\psi_{100}
=-\frac1{2\mu a^2}\psi_{100}
+\left(\frac1{\mu a}-Ze^2\right)\frac{\psi_{100}}r.
$$
由 $1/(\mu a)=Ze^2$，奇异的 $1/r$ 项相消，得到
$$
H\psi_{100}
=-\frac{\mu Z^2e^4}{2}\psi_{100}.
$$
该等式先在 $r>0$ 逐点成立；$\psi_{100}$ 属于标准 Coulomb 自伴实现的
定义域这一闭包事实包含在 QM-EXT-10 中。

中心势的旋转不变性把三维问题分成球谐角向通道与半轴径向方程，离心势
精确记录每个 $\ell$ 通道的角动量代价。氢型基态的直接代回展示了
Coulomb 奇异项怎样与指数波函数的径向导数相消；完整 $1/n^2$ 谱及其
完备性仍由 QM-EXT-10 承担。下一章加入外电磁场，普通动量将被动力学
动量取代，而规范相关的势必须通过酉协变性保持物理预测。

## 练习

**练习 22.1.** 证明径向替换 $R=u/r$ 时有 $R''+2R'/r=u''/r$。

**练习 22.2.** 对 $\ell=0$ 写出中心势径向方程。

**练习 22.3.** 计算固定主量子数 $n$ 时忽略自旋的氢型束缚态简并度。
