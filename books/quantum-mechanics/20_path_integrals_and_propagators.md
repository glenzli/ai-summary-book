# 第二十章：路径积分与传播子

## 本章目标

本章说明传播子、Trotter 离散化、自由粒子核和路径积分的严格边界。

## 依赖前置知识

需要 Schrodinger 演化、Fourier 变换、Gaussian 积分和半经典作用量。

## 20.1 传播子

**定义 20.1.** 若 $U(t)=e^{-itH}$ 在位置表象中有可测函数核 $K(t;x,y)$，使对某个稠密测试函数空间（例如 $\mathcal S(\mathbb R^d)$）中的每个 $\psi$ 有
$$
(U(t)\psi)(x)=\int K(t;x,y)\psi(y)\,dy,
$$
则称 $K$ 为传播子核。更一般地，若 $K(t)\in\mathcal S'(\mathbb R_x^d\times\mathbb R_y^d)$，且上式按分布配对解释后给出 $U(t):\mathcal S\to\mathcal S'$，则称 $K$ 为分布传播子核。

**命题 20.2.** 若两个核的乘积满足 Fubini/Tonelli 条件，则传播子满足卷积公式
$$
K(t+s;x,y)=\int K(t;x,z)K(s;z,y)\,dz.
$$

**证明.** 由 $U(t+s)=U(t)U(s)$，
$$
(U(t+s)\psi)(x)=U(t)(U(s)\psi)(x)
=\int K(t;x,z)\left(\int K(s;z,y)\psi(y)\,dy\right)dz.
$$
由 Fubini 定理交换积分后与核表示比较，得到几乎处处成立的公式。若核只作为分布存在，则右侧必须理解为已定义的分布核复合；该复合的存在来自算子乘积及其核，不等于任意两个分布都可逐点相乘。$\square$

## 20.2 自由粒子核

**公式 20.3.** 对 $H=P^2/2m$、$m>0$ 和 $t\ne0$，自由粒子传播子为
$$
K_0(t;x,y)=\left(\frac{m}{2\pi it}\right)^{d/2}
\exp\left(\frac{im|x-y|^2}{2t}\right)
$$
在振荡积分或 tempered distribution 意义下成立。复幂采用 Fresnel 分支，即
$$
(it)^{-d/2}
=|t|^{-d/2}\exp\left(-\frac{i\pi d}{4}\operatorname{sgn}t\right).
$$
并且
$$
K_0(t;\cdot,y)\longrightarrow\delta_y
$$
当 $t\to0$ 时在 $\mathcal S'(\mathbb R^d)$ 中成立。

**证明.** Fourier 变换下 $H$ 变为乘法 $|p|^2/2m$，故
$$
K_0(t;x,y)=\frac1{(2\pi)^d}\int_{\mathbb R^d}
e^{ip\cdot(x-y)}e^{-it|p|^2/2m}\,dp.
$$
配方并使用 Fresnel Gaussian 积分得到公式。$\square$

## 20.3 路径积分边界

**外部输入定理 20.4（Lie--Trotter 乘积公式，QM-EXT-9）.** 设 $T,V$ 自伴，$\mathcal D(T)\cap\mathcal D(V)$ 稠密，且 $T+V$ 在该交定义域上本质自伴；记其自伴闭包仍为 $T+V$。则对每个固定 $t\in\mathbb R$，
$$
e^{-it(T+V)}=\operatorname{s-lim}_{n\to\infty}
\left(e^{-itT/n}e^{-itV/n}\right)^n.
$$

**说明 20.5.** 形式路径积分
$$
K(t;x,y)=\int_{q(0)=y}^{q(t)=x}e^{iS[q]}\mathcal Dq
$$
应理解为 Trotter 离散化、振荡积分和极限过程的记号。一般情形不存在平移不变的“Lebesgue 路径测度”。

## 20.4 短时核与作用量

**命题 20.6（split 一步核）.** 令 $V:\mathbb R^d\to\mathbb R$ 为有限几乎处处的可测函数，使 $V(X)$ 按最大乘法定义域成为自伴算子。对形式 Hamiltonian
$$
H=\frac{P^2}{2m}+V(X)
$$
定义 Lie--Trotter 的单步算子
$$
U_\epsilon^{\mathrm{split}}
=e^{-i\epsilon P^2/2m}e^{-i\epsilon V(X)}.
$$
及任意 $\epsilon\ne0$，该 split 算子的分布核恰为
$$
K_\epsilon^{\mathrm{split}}(x,y)=
\left(\frac{m}{2\pi i\epsilon}\right)^{d/2}
\exp\left(i\frac{m|x-y|^2}{2\epsilon}-i\epsilon V(y)\right).
$$

**证明.** 第二个算子在位置表象中乘以 $e^{-i\epsilon V(y)}$；第一个算子的核是自由粒子短时核
$$
\left(\frac{m}{2\pi i\epsilon}\right)^{d/2}
\exp\left(i\frac{m|x-y|^2}{2\epsilon}\right).
$$
相乘得到公式。注意该等式描述 split operator 的精确核；在没有有界交换子等额外假设时，不应把它单独写成 $e^{-i\epsilon H}$ 的逐点 $O(\epsilon^2)$ 核近似。若和算子满足定理 20.4 的假设，该定理保证的是 $n$ 步乘积在强算子拓扑中收敛，而不是单步核逐点收敛。$\square$

**说明 20.7.** 将许多短时核相乘并对中间位置积分，会得到离散作用量和
$$
\sum_j\left(\frac{m|x_{j+1}-x_j|^2}{2\epsilon}-\epsilon V(x_j)\right),
$$
这是形式作用量积分的离散来源。

**说明 20.8.** 若把时间替换为虚时间 $t=-i\tau$，振荡因子形式上变为
$$
e^{-S_E[q]},
$$
其中 $S_E$ 是 Euclidean action。虚时间方法与热核、配分函数和 Feynman-Kac 公式有关；严格概率测度解释通常在虚时间而非实时间中成立。

## 本章小结

传播子是酉演化的位置表象核。自由粒子核可由 Fourier 变换计算。路径积分是强大的形式语言，但严格解释依赖 Trotter 公式、振荡积分和极限分析。

## 练习

**练习 20.1.** 用传播子卷积公式解释 $U(t+s)=U(t)U(s)$。

**练习 20.2.** 写出一维自由粒子核，并指出 $t\to0$ 时它趋向 delta 分布的意义。
