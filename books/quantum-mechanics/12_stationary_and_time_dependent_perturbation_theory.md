# 第十二章：定态与含时扰动理论

## 本章目标

本章推导非简并定态扰动公式、简并扰动的矩阵对角化原则和含时扰动的一阶跃迁振幅。

## 依赖前置知识

需要自伴矩阵谱分解、Schrodinger 方程和内积计算。

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

## 12.3 含时扰动

**设定 12.4.** 令 $H_0$ 为自伴算子，$\lambda\in\mathbb R$。令
Hilbert 空间有限维，或更一般地令 $V(t)$ 为强连续且在所考虑有限
时间区间上一致有界的自伴算子族。取
$H(t)=H_0+\lambda V(t)$，相互作用图像中的态满足
$$
i\dot\psi_I(t)=\lambda V_I(t)\psi_I(t),\qquad
V_I(t)=e^{itH_0}V(t)e^{-itH_0}.
$$

**命题 12.5.** 若 $\psi_I(0)=|i\rangle$，且 $|i\rangle,|f\rangle$
是 $H_0$ 的归一化本征态，定义
$c_f(t)=\langle f,\psi_I(t)\rangle$。则一阶 Dyson 截断给出的振幅为
$$
c_f(t)=\delta_{fi}-i\lambda\int_0^t
e^{i(E_f-E_i)s}\langle f,V(s)i\rangle\,ds
+O(\lambda^2).
$$
这里的 $O(\lambda^2)$ 在固定有限时间区间上按命题 12.5A 的范数估计理解。

**证明.** 积分方程为
$$
\psi_I(t)=\psi_I(0)-i\lambda\int_0^tV_I(s)\psi_I(s)\,ds.
$$
一阶截断把积分中的 $\psi_I(s)$ 替换为初态 $i$。再取 $f$ 分量并使用 $H_0$ 的本征方程即可。$\square$

**命题 12.5A（有限时间的一阶余项界）.** 在设定 12.4 下，设 $t\ge0$、$\|\psi_I(0)\|=1$，且 $\sup_{0\le s\le t}\|V_I(s)\|\le M$。则截断 Dyson 级数到一阶后的态向量余项满足
$$
\left\|\psi_I(t)-\left(I-i\lambda\int_0^tV_I(s)\,ds\right)\psi_I(0)\right\|
\le e^{|\lambda|Mt}-1-|\lambda|Mt.
$$
特别地，在固定有限 $t$ 上余项为 $O(\lambda^2M^2t^2)$。

**证明.** 第 $r$ 阶时间有序积分的算子范数至多为
$$
|\lambda|^rM^r\frac{t^r}{r!},
$$
因为有序单纯形 $0\le s_r\le\cdots\le s_1\le t$ 的体积为 $t^r/r!$。对 $r\ge2$ 求和得到所述指数余项。$\square$

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

## 本章小结

扰动理论把难解 Hamiltonian 展开为可解 Hamiltonian 加小扰动。非简并情形有显式分母公式；简并情形必须先在简并子空间中对角化；含时扰动给出跃迁振幅和选择定则的起点。无限维推广需要解析扰动理论作为外部输入。

## 练习

**练习 12.1.** 对二能级矩阵 $\begin{pmatrix}E&\lambda v\\ \lambda\overline v&F\end{pmatrix}$，在 $E\ne F$ 时求 $E$ 附近本征值的一阶修正。

**练习 12.2.** 说明为什么简并扰动中不能直接使用非简并分母公式。
