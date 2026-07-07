# 第二十五章：相互作用图像、Dyson 展开与跃迁率

## 本章目标

本章补齐含时量子力学的标准形式：相互作用图像、时间有序指数、Dyson 级数、跃迁振幅和 Fermi 黄金规则的边界。正文证明有限维有界情形；连续谱极限的严格性作为外部输入。

## 依赖前置知识

需要 Schrodinger 演化、含时扰动理论、Fourier 变换和分布极限。

## 25.1 相互作用图像

**设定 25.1.** 设
$$
H(t)=H_0+V(t),
$$
其中 $H_0$ 为时间无关自伴算子。Schrodinger 图像态满足
$$
i\frac d{dt}\psi_S(t)=H(t)\psi_S(t).
$$
定义相互作用图像态
$$
\psi_I(t)=e^{itH_0}\psi_S(t)
$$
和相互作用 Hamiltonian
$$
V_I(t)=e^{itH_0}V(t)e^{-itH_0}.
$$

**命题 25.2.** 相互作用图像态满足
$$
i\frac d{dt}\psi_I(t)=V_I(t)\psi_I(t).
$$

**证明.** 对 $\psi_I=e^{itH_0}\psi_S$ 求导：
$$
\dot\psi_I=iH_0e^{itH_0}\psi_S+e^{itH_0}\dot\psi_S.
$$
由 Schrodinger 方程 $\dot\psi_S=-i(H_0+V)\psi_S$，且 $H_0$ 与 $e^{itH_0}$ 交换，$H_0$ 项相消，得
$$
\dot\psi_I=-ie^{itH_0}V(t)\psi_S=-iV_I(t)\psi_I.
$$
$\square$

## 25.2 Dyson 级数

**定义 25.3.** 相互作用图像传播子 $U_I(t,t_0)$ 定义为
$$
\psi_I(t)=U_I(t,t_0)\psi_I(t_0).
$$
它满足积分方程
$$
U_I(t,t_0)=I-i\int_{t_0}^t V_I(s)U_I(s,t_0)\,ds.
$$

**命题 25.4（Dyson 展开，有界有限维情形）.** 若 $V_I(t)$ 连续且有界，则
$$
U_I(t,t_0)=I+\sum_{n=1}^\infty(-i)^n
\int_{t_0\le s_n\le\cdots\le s_1\le t}
V_I(s_1)\cdots V_I(s_n)\,ds_1\cdots ds_n.
$$

**证明.** 从积分方程迭代一次得到
$$
U_I=I-i\int V_I
 -\int_{t_0}^t\int_{t_0}^{s_1}V_I(s_1)V_I(s_2)U_I(s_2,t_0)\,ds_2ds_1.
$$
继续迭代得到 $n$ 阶时间有序积分。若 $\|V_I(s)\|\le M$，第 $n$ 阶范数至多
$$
\frac{(M|t-t_0|)^n}{n!},
$$
故级数按算子范数绝对收敛，并由逐项代入积分方程验证为唯一解。$\square$

## 25.3 跃迁振幅

**命题 25.5（一阶跃迁振幅）.** 若 $H_0|n\rangle=E_n|n\rangle$，系统初态为 $|i\rangle$，则到一阶
$$
c_f^{(1)}(t)
=-i\int_{t_0}^t e^{i(E_f-E_i)s}\langle f|V(s)|i\rangle\,ds.
$$

**证明.** Dyson 展开的一级项给出
$$
\langle f|U_I(t,t_0)|i\rangle
=\delta_{fi}-i\int_{t_0}^t\langle f|V_I(s)|i\rangle\,ds+O(V^2).
$$
由 $V_I(s)=e^{isH_0}V(s)e^{-isH_0}$，
$$
\langle f|V_I(s)|i\rangle=e^{i(E_f-E_i)s}\langle f|V(s)|i\rangle.
$$
对 $f\ne i$ 得公式。$\square$

## 25.4 黄金规则边界

**公式 25.6（Fermi 黄金规则）.** 若扰动近似为常量 $V$，末态形成连续谱且态密度为 $\rho(E_f)$，长时间平均跃迁率形式上为
$$
\Gamma_{i\to f}=2\pi |\langle f|V|i\rangle|^2\rho(E_f)
$$
并满足能量守恒 $E_f=E_i$。

**说明 25.7.** 黄金规则使用极限
$$
\lim_{T\to\infty}\frac1T\left|\int_0^T e^{i\omega t}\,dt\right|^2=2\pi\delta(\omega)
$$
的分布意义。将离散谱替换为连续谱和态密度需要散射理论或谱测度口径，因此本书把完整严格化列为外部输入。

## 本章小结

相互作用图像把可解自由演化剥离出去，只留下扰动驱动。Dyson 展开是时间有序积分级数，给出系统化的含时微扰论。黄金规则是连续谱和长时间极限下的一阶跃迁率公式，其严格解释依赖谱测度和散射理论。

## 练习

**练习 25.1.** 若 $[V_I(t),V_I(s)]=0$ 对所有 $s,t$ 成立，证明 Dyson 展开化为普通指数。

**练习 25.2.** 对常扰动 $V$，写出一阶振幅中的积分并化简为 $\sin$ 函数形式。

**练习 25.3.** 说明时间有序乘积为什么在 $[V_I(t),V_I(s)]\ne0$ 时不可省略。

