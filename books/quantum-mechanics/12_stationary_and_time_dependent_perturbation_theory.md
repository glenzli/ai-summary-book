# 第十二章：定态与含时扰动理论

精确可解的 Hamiltonian 很少，但“接近可解”只有在误差能够控制时才是
数学陈述。把 $H(\lambda)=H_0+\lambda V$ 写下来以后，非简并能级的
一阶移动看似只需取对角矩阵元；一旦两个未扰动能级重合，分母公式立即
失效，正确的零阶基必须由简并子空间中的压缩扰动重新选择。含时驱动又
带来另一种小量：固定有限时间上的 Dyson 截断可以是 $O(\lambda^2)$，
但误差会随观察时间增长，不能无条件外推到长时间跃迁率。

本章先在有限维中证明孤立简单本征值的局部解析分支存在，再推导一阶与
二阶定态修正，并用可精确对角化的二能级矩阵核对展开。简并情形由压缩
$P_MVP_M|_M$ 处理。对于含时扰动，本章不借用后续章节，而是在算子范数
连续、有界的假设下直接用 Picard 迭代构造传播子、证明 Dyson 级数一致
收敛并给出一阶余项界。无限维推广所需的解析扰动理论则明确保留为外部
输入。

## 12.1 非简并定态扰动

**设定 12.1.** 设
$$
H(\lambda)=H_0+\lambda V
$$
为有限维复 Hilbert 空间上的矩阵族，其中 $H_0,V$ 自伴且
$\lambda\in\mathbb R$。假设
$H_0\psi_n^{(0)}=E_n^{(0)}\psi_n^{(0)}$、
$\|\psi_n^{(0)}\|=1$，且 $E_n^{(0)}$ 非简并。

**命题 12.1A（有限维局部展开的存在性）.** 在设定 12.1 下，存在 $\lambda=0$ 的邻域以及实解析函数 $E_n(\lambda)$ 和可取归一化的解析本征向量 $\psi_n(\lambda)$，使
$$
H(\lambda)\psi_n(\lambda)=E_n(\lambda)\psi_n(\lambda),
\qquad E_n(0)=E_n^{(0)}.
$$
在固定相位条件 $\langle\psi_n^{(0)},\psi_n(\lambda)\rangle>0$ 下，该局部分支唯一。

**证明.** 记 $\psi_0=\psi_n^{(0)}$，并按
$$
\mathcal H=\mathbb C\psi_0\oplus Q\mathcal H,
\qquad
Q=I-|\psi_0\rangle\langle\psi_0|
$$
分解空间。定义
$$
a(\lambda)=\langle\psi_0,H(\lambda)\psi_0\rangle,
\quad
b(\lambda)=QH(\lambda)\psi_0,
\quad
D(\lambda)=QH(\lambda)Q\big|_{Q\mathcal H}.
$$
由于 $E_n^{(0)}$ 非简并，$D(0)-E_n^{(0)}I$ 可逆。因此在 $(E_n^{(0)},0)$ 的某个邻域内，$D(\lambda)-EI$ 仍可逆。寻找满足中间归一化条件
$$
\langle\psi_0,\widetilde\psi(\lambda)\rangle=1
$$
的本征向量，并写
$$
\widetilde\psi(\lambda)=\psi_0+\chi(\lambda),
\qquad
\chi(\lambda)\in Q\mathcal H.
$$
本征方程的 $Q$ 分量唯一给出
$$
\chi(E,\lambda)
=-[D(\lambda)-EI]^{-1}b(\lambda).
$$
其 $\mathbb C\psi_0$ 分量等价于标量方程
$$
F(E,\lambda)
\coloneqq
a(\lambda)-E
-\left\langle b(\lambda),[D(\lambda)-EI]^{-1}b(\lambda)\right\rangle
=0.
$$
在 $\lambda=0$ 时有 $b(0)=0$，故
$$
F(E_n^{(0)},0)=0,
\qquad
\partial_EF(E_n^{(0)},0)=-1.
$$
实解析隐函数定理给出唯一实解析解 $E=E_n(\lambda)$；代回上式得到唯一实解析的 $\chi(\lambda)$。最后令
$$
\psi_n(\lambda)
=\frac{\widetilde\psi(\lambda)}{\|\widetilde\psi(\lambda)\|}.
$$
其与 $\psi_0$ 的内积为正，因而同时固定了归一化与相位。$\square$

**命题 12.2.** 若
$$
E_n(\lambda)=E_n^{(0)}+\lambda E_n^{(1)}+O(\lambda^2),
\qquad
\psi_n(\lambda)=\psi_n^{(0)}+\lambda\psi_n^{(1)}+O(\lambda^2)
$$
并取规范 $\langle\psi_n^{(0)},\psi_n^{(1)}\rangle=0$，则
$$
E_n^{(1)}=\langle\psi_n^{(0)},V\psi_n^{(0)}\rangle
$$
且
$$
\psi_n^{(1)}=\sum_{m\ne n}
\frac{\langle\psi_m^{(0)},V\psi_n^{(0)}\rangle}
{E_n^{(0)}-E_m^{(0)}}\psi_m^{(0)}.
$$

**证明.** 将展开代入 $H(\lambda)\psi_n(\lambda)=E_n(\lambda)\psi_n(\lambda)$，取一阶项：
$$
(H_0-E_n^{(0)})\psi_n^{(1)}=(E_n^{(1)}-V)\psi_n^{(0)}.
$$
左乘 $\langle\psi_n^{(0)}|$ 得能量公式。左乘 $\langle\psi_m^{(0)}|$，$m\ne n$，得
$$
(E_m^{(0)}-E_n^{(0)})\langle\psi_m^{(0)},\psi_n^{(1)}\rangle
=-\langle\psi_m^{(0)},V\psi_n^{(0)}\rangle.
$$
整理即得向量修正公式。$\square$

**例子 12.2A（精确二能级展开）.** 取 $\Delta>0$、$g\in\mathbb C$，
$$
H(\lambda)=
\begin{pmatrix}0&\lambda g\\
\lambda\overline g&\Delta\end{pmatrix}.
$$
靠近未扰动能级 $0$ 的精确本征值为
$$
E_-(\lambda)
=\frac{\Delta-\sqrt{\Delta^2+4\lambda^2|g|^2}}2
=-\frac{\lambda^2|g|^2}{\Delta}+O(\lambda^4).
$$
扰动的对角矩阵元为零，所以一阶修正确实消失；本章的二阶推导将重新给出
$|g|^2/(0-\Delta)=-|g|^2/\Delta$。精确展开已经独立验证该系数，并
表明小参数展开的分母实际测量了未扰动谱隙。

简单本征值允许用谱隙反演 $H_0-E_n^{(0)}$。若谱隙在简并子空间内为
零，这一步不再合法，必须先找出扰动在该子空间中真正区分的方向。

## 12.2 简并扰动

**命题 12.3.** 若 $E$ 是 $H_0$ 的简并本征值，对应本征子空间为
$M$，并记 $P_M$ 为到 $M$ 的正交投影。设某一本征分支具有展开
$$
E(\lambda)=E+\lambda E^{(1)}+O(\lambda^2),
\qquad
\psi(\lambda)=\psi^{(0)}+O(\lambda),
$$
其中 $\psi^{(0)}\in M$ 且 $\|\psi^{(0)}\|=1$。则 $E^{(1)}$ 是压缩算子
$$
P_MVP_M\big|_M:M\longrightarrow M
$$
的本征值，$\psi^{(0)}$ 是相应本征向量。因此一阶计算必须先在
$M$ 中对角化该压缩。这里不能把 $V|_M$ 写成 $M$ 上的算子，除非
已经知道 $V(M)\subseteq M$。有限维简并分支的存在性可视为
QM-EXT-20 的有限维情形；本命题只证明给定分支必须满足的一阶方程。

**证明.** 一阶方程投影到 $M$ 上。因 $(H_0-E)$ 在 $M$ 上为零，得到
$$
P_MV P_M\psi^{(0)}=E^{(1)}\psi^{(0)}.
$$
因此必须先在简并子空间中对角化扰动。$\square$

**例子 12.3A（两重简并的劈裂）.** 若
$H_0=E I_{\mathbb C^2}$ 且
$$
V=\begin{pmatrix}0&v\\\overline v&0\end{pmatrix},
\qquad v\ne0,
$$
则整个空间就是 $M$，压缩仍为 $V$。其本征值为 $\pm|v|$，所以
$H_0+\lambda V$ 的两个精确能级是
$$
E_\pm(\lambda)=E\pm\lambda|v|.
$$
原标准基并不是正确的零阶分支；应改取 $V$ 的归一本征向量。非简并
分母公式在这里出现 $E-E=0$，正好暴露了错误。

定态扰动追踪的是谱分支，含时扰动追踪的则是同一初态在不同未扰动
能量子空间之间怎样积累振幅。为隔离自由相位，下面转入相互作用图像。

## 12.3 含时扰动

**设定 12.4.** 令 $H_0$ 为自伴算子，$\lambda\in\mathbb R$。令
$V(t)$ 为有界自伴算子，并要求相互作用图像算子族
$$
V_I(t)=e^{itH_0}V(t)e^{-itH_0}
$$
在每个所考虑的有限时间区间上按算子范数连续。有限维时，$V_I(t)$
的强连续性已蕴含这一条件。取 $H(t)=H_0+\lambda V(t)$；由有界扰动
定理，每个 $H(t)$ 都在 $\mathcal D(H_0)$ 上自伴。若 Schrodinger
图像解强可微且保持在 $\mathcal D(H_0)$ 中，则对
$\psi_I(t)=e^{itH_0}\psi_S(t)$ 使用乘积法则，$H_0$ 项相消，相互作用图像态
满足
$$
i\dot\psi_I(t)=\lambda V_I(t)\psi_I(t).
$$
本节实际把演化定义为这个有界方程的积分解。在上述范数连续假设下，
积分方程使用算子范数 Bochner 积分；下面在本章独立完成存在唯一性和
Dyson 级数的范数估计，不调用第二十五章。后者会以二参数传播子记号
重新组织同一构造并进一步讨论跃迁率。
仅假设强连续和一致有界时可以
另作逐向量的强积分构造，但不能直接引用下面的算子范数级数证明；本章
不采用那条更一般的路线。

**命题 12.4A（有界相互作用的传播子）.** 固定 $T>0$。若
$V_I:[0,T]\to\mathcal B(\mathcal H)$ 按算子范数连续且
$\sup_{0\le s\le T}\|V_I(s)\|\le M$，则积分方程
$$
U_I(t)=I-i\lambda\int_0^tV_I(s)U_I(s)\,ds
$$
在 $C([0,T],\mathcal B(\mathcal H))$ 中有唯一解，并且
$$
U_I(t)=I+\sum_{n=1}^{\infty}(-i\lambda)^n
\int_{0\le s_n\le\cdots\le s_1\le t}
V_I(s_1)\cdots V_I(s_n)\,ds_1\cdots ds_n,
$$
级数在 $[0,T]$ 上按算子范数一致收敛。

**证明.** 从 $U_0(t)=I$ 出发作 Picard 迭代。第 $n$ 次迭代新增的项正是
上式的 $n$ 重时间有序积分，其范数一致满足
$$
\left\|\int_{0\le s_n\le\cdots\le s_1\le t}
V_I(s_1)\cdots V_I(s_n)\,ds_1\cdots ds_n\right\|
\le \frac{(Mt)^n}{n!}\le\frac{(MT)^n}{n!}.
$$
因而 Weierstrass 判别法给出一致收敛；逐项代回积分方程即得一个解。
若 $U$、$W$ 都是解，令
$D(t)=\sup_{0\le r\le t}\|U(r)-W(r)\|$，则逐次代入
$$
D(t)\le |\lambda|M\int_0^tD(s)\,ds
$$
得到 $D(t)\le C(|\lambda|Mt)^n/n!$ 对所有 $n$ 成立，其中
$C=\sup_{[0,T]}D<\infty$；令 $n\to\infty$ 得 $D(t)=0$。故解唯一。$\square$

**命题 12.4B（酉性与复合律）.** 在命题 12.4A 的假设下，对
$0\le s\le t\le T$ 从积分方程
$$
U_I(t,s)=I-i\lambda\int_s^tV_I(r)U_I(r,s)\,dr
$$
构造的解是酉算子，并满足
$$
U_I(t,s)U_I(s,r)=U_I(t,r),\qquad 0\le r\le s\le t\le T.
$$

**证明.** 命题 12.4A 的证明平移积分下限后仍适用，并给出唯一的算子
范数连续解。因 $V_I$ 按算子范数连续，积分方程还给出
$$
\partial_tU_I(t,s)=-i\lambda V_I(t)U_I(t,s)
$$
的算子范数导数。利用 $V_I(t)^*=V_I(t)$，
$$
\partial_t\bigl(U_I(t,s)^*U_I(t,s)\bigr)=0,
$$
所以 $U_I(t,s)^*U_I(t,s)=I$。令
$X(t)=U_I(t,s)U_I(t,s)^*$，则
$$
\dot X(t)=-i\lambda V_I(t)X(t)+i\lambda X(t)V_I(t),
\qquad X(s)=I.
$$
常值函数 $I$ 满足同一方程；对该有界线性积分方程使用命题 12.4A 中
相同的唯一性估计，得到 $X(t)=I$，故 $U_I(t,s)$ 酉。最后，
$U_I(t,s)U_I(s,r)$ 与 $U_I(t,r)$ 作为 $t$ 的函数满足同一微分方程，
并在 $t=s$ 取相同值 $U_I(s,r)$；唯一性给出复合律。$\square$

**命题 12.5.** 设 $t\ge0$。若 $\psi_I(0)=|i\rangle$，且 $|i\rangle,|f\rangle$
是 $H_0$ 的归一化本征态，定义
$c_f(t)=\langle f,\psi_I(t)\rangle$。则一阶 Dyson 截断给出的振幅为
$$
c_f(t)=\delta_{fi}-i\lambda\int_0^t
e^{i(E_f-E_i)s}\langle f,V(s)i\rangle\,ds
+O(\lambda^2).
$$
这里的 $O(\lambda^2)$ 在固定有限时间区间上按紧接本命题给出的显式
范数估计理解。

**证明.** 积分方程为
$$
\psi_I(t)=\psi_I(0)-i\lambda\int_0^tV_I(s)\psi_I(s)\,ds.
$$
命题 12.4A 保证该积分方程及其 Dyson 展开按算子范数成立。一阶截断
把积分中的 $\psi_I(s)$ 替换为初态 $i$；再取 $f$ 分量
并使用 $H_0$ 的本征方程即可。$\square$

**命题 12.5A（有限时间的一阶余项界）.** 在设定 12.4 下，设 $t\ge0$、$\|\psi_I(0)\|=1$，且 $\sup_{0\le s\le t}\|V_I(s)\|\le M$。则截断 Dyson 级数到一阶后的态向量余项满足
$$
\left\|\psi_I(t)-\left(I-i\lambda\int_0^tV_I(s)\,ds\right)\psi_I(0)\right\|
\le e^{|\lambda|Mt}-1-|\lambda|Mt.
$$
特别地，在固定有限 $t$ 上余项为 $O(\lambda^2M^2t^2)$。

**证明.** 由命题 12.4A 证明中的单纯形估计，第 $r$ 阶时间有序 Bochner 积分的算子范数
至多为
$$
|\lambda|^rM^r\frac{t^r}{r!},
$$
因为有序单纯形 $0\le s_r\le\cdots\le s_1\le t$ 的体积为 $t^r/r!$。对 $r\ge2$ 求和得到所述指数余项。$\square$

上述有界含时链已经由积分方程、Picard 唯一性和显式尾项界闭合。为了
与例子 12.2A 的精确谱展开比较，下面回到定态问题，补出在那里首次
出现的二阶能量修正。

## 12.4 二阶能量修正

**命题 12.6.** 在命题 12.2 的非简并有限维设定下，若取规范
$$
\langle\psi_n^{(0)},\psi_n(\lambda)\rangle=1+O(\lambda^2),
$$
则二阶能量修正为
$$
E_n^{(2)}
=\sum_{m\ne n}
\frac{|\langle\psi_m^{(0)},V\psi_n^{(0)}\rangle|^2}
{E_n^{(0)}-E_m^{(0)}}.
$$

**证明.** 二阶方程为
$$
(H_0-E_n^{(0)})\psi_n^{(2)}
=(E_n^{(1)}-V)\psi_n^{(1)}+E_n^{(2)}\psi_n^{(0)}.
$$
左乘 $\langle\psi_n^{(0)}|$，左边为零，且规范使 $\psi_n^{(1)}$ 与 $\psi_n^{(0)}$ 正交，得
$$
E_n^{(2)}=\langle\psi_n^{(0)},V\psi_n^{(1)}\rangle.
$$
代入一阶向量修正
$$
\psi_n^{(1)}=\sum_{m\ne n}
\frac{\langle\psi_m^{(0)},V\psi_n^{(0)}\rangle}
{E_n^{(0)}-E_m^{(0)}}\psi_m^{(0)}
$$
即得公式。$\square$

## 12.5 无限维边界

本章公式在有限维中由代数展开证明。若 $H_0$ 是无限维 Hilbert 空间上的自伴算子，$V$ 为无界或相对有界扰动，则存在性、谱投影的可微性和级数收敛不能由上述代数计算推出。教材采用如下外部边界：

**外部输入说明 12.7.** 若孤立本征值与其余谱隔离，且 $H(\lambda)=H_0+\lambda V$ 满足解析扰动族假设，则本征投影和本征值的局部展开由 Kato 解析扰动理论保证，见外部输入定理 QM-EXT-20。正文只使用其有限维后果，不把该定理的解析 Fredholm 理论证明纳入量子力学主线。

本章得到的是三种口径不同而边界清楚的近似。有限维简单能级具有书内
证明的局部解析分支，一、二阶系数由谱隙分母控制；简并能级必须先对角化
$P_MVP_M|_M$；范数连续有界的含时相互作用则具有书内构造的有限时间
Dyson 传播子，一阶余项受
$e^{|\lambda|Mt}-1-|\lambda|Mt$ 控制。无界无限维扰动不由这些代数
结论自动覆盖，而要满足 QM-EXT-20 的解析族假设。下一章考察另外两类
近似：用试探空间控制能量，以及用小 $\hbar$ 控制局部波形。

## 练习

**练习 12.1.** 对二能级矩阵 $\begin{pmatrix}E&\lambda v\\ \lambda\overline v&F\end{pmatrix}$，在 $E\ne F$ 时求 $E$ 附近本征值的一阶修正。

**练习 12.2.** 说明为什么简并扰动中不能直接使用非简并分母公式。

**练习 12.3.** 在设定 12.4 中若 $V_I(t)=V_0$ 为常值有界自伴算子，
证明
$U_I(t,s)=e^{-i\lambda(t-s)V_0}$，并说明它如何从时间有序 Dyson
级数得到。
