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
为有限维自伴矩阵族。假设 $H_0\psi_n^{(0)}=E_n^{(0)}\psi_n^{(0)}$，且 $E_n^{(0)}$ 非简并。

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

**命题 12.3.** 若 $E$ 是 $H_0$ 的简并本征值，对应本征子空间为 $M$，则一阶能量修正为 $V|_M$ 的本征值。

**证明.** 一阶方程投影到 $M$ 上。因 $(H_0-E)$ 在 $M$ 上为零，得到
$$
P_MV P_M\psi^{(0)}=E^{(1)}\psi^{(0)}.
$$
因此必须先在简并子空间中对角化扰动。$\square$

## 12.3 含时扰动

**设定 12.4.** 令 $H(t)=H_0+\lambda V(t)$，相互作用图像中的态满足
$$
i\dot\psi_I(t)=\lambda V_I(t)\psi_I(t),\qquad
V_I(t)=e^{itH_0}V(t)e^{-itH_0}.
$$

**命题 12.5.** 一阶近似下，从初态 $i$ 到末态 $f$ 的跃迁振幅为
$$
c_f^{(1)}(t)=-i\lambda\int_0^t
e^{i(E_f-E_i)s}\langle f,V(s)i\rangle\,ds.
$$

**证明.** 积分方程为
$$
\psi_I(t)=\psi_I(0)-i\lambda\int_0^tV_I(s)\psi_I(s)\,ds.
$$
一阶近似把积分中的 $\psi_I(s)$ 替换为初态 $i$。再取 $f$ 分量并使用 $H_0$ 的本征基即可。$\square$

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
