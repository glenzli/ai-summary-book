# 第二十一章：相对论一粒子方程与适用边界

## 本章目标

本章介绍 Klein-Gordon 方程、Dirac 方程和非相对论极限，并说明为什么完整相对论量子理论需要量子场论。

## 依赖前置知识

需要 Schrodinger 方程、动量算子、矩阵和自旋。本章为显示非相对论极限而暂时恢复 $\hbar$ 与 $c$；所有公式均按这一局部单位约定书写。

## 21.1 Klein-Gordon 方程

**定义 21.1.** 由相对论能量关系
$$
E^2=p^2c^2+m^2c^4
$$
形式替换 $E\mapsto i\hbar\partial_t$、$p\mapsto -i\hbar\nabla$ 得到 Klein-Gordon 方程
$$
\left(\frac1{c^2}\partial_t^2-\Delta+\frac{m^2c^2}{\hbar^2}\right)\phi=0.
$$

**边界 21.2.** Klein-Gordon 方程不是普通概率振幅的一阶时间演化方程；其自然解释属于场论或带不定结构的相对论波方程。

## 21.2 Dirac 方程

**定义 21.3.** Dirac 方程为
$$
i\hbar\partial_t\psi=
\left(c\,\alpha\cdot(-i\hbar\nabla)+\beta mc^2\right)\psi,
$$
其中矩阵满足
$$
\alpha_i\alpha_j+\alpha_j\alpha_i=2\delta_{ij}I,\qquad
\alpha_i\beta+\beta\alpha_i=0,\qquad \beta^2=I.
$$

**命题 21.4.** 在 Schwartz 核心 $\mathcal S(\mathbb R^3;\mathbb C^N)$ 上，Dirac Hamiltonian 的平方给出相对论能量关系：
$$
H_D^2=(c^2P^2+m^2c^4)I.
$$

**证明.** 令 $H_D=c\alpha\cdot p+\beta mc^2$。使用反对易关系，
$$
H_D^2=c^2\sum_{i,j}\alpha_i\alpha_jp_ip_j
+mc^3\sum_i(\alpha_i\beta+\beta\alpha_i)p_i
+m^2c^4\beta^2.
$$
中间项为零，第一项因 $p_ip_j=p_jp_i$ 化为 $c^2p^2I$，最后一项为 $m^2c^4I$。$\square$

## 21.3 非相对论极限

**说明 21.5.** 设 $m>0$。当 $|p|/(mc)\ll1$ 时，
$$
\sqrt{p^2c^2+m^2c^4}
=mc^2+\frac{p^2}{2m}-\frac{p^4}{8m^3c^2}
+O\!\left(\frac{p^6}{m^5c^4}\right).
$$
去掉静止能相位 $e^{-imc^2t/\hbar}$ 后，低能极限恢复 Schrodinger 动力学，并伴随自旋和磁矩修正。

## 21.4 概率解释的边界

**命题 21.6.** Dirac Hamiltonian
$$
H_D=c\alpha\cdot P+\beta mc^2
$$
若矩阵 $\alpha_i,\beta$ Hermitian，则对光滑且具有足够空间衰减的解给出守恒密度
$$
\rho=\psi^\dagger\psi
$$
和流
$$
j=c\,\psi^\dagger\alpha\psi.
$$

**证明.** Dirac 方程写作
$$
i\hbar\partial_t\psi=H_D\psi.
$$
其伴随为
$$
-i\hbar\partial_t\psi^\dagger=(H_D\psi)^\dagger.
$$
在无外场且矩阵 Hermitian 情形，质量项在 $\partial_t(\psi^\dagger\psi)$ 中相消，动量项给出
$$
\partial_t(\psi^\dagger\psi)
=-c\nabla\cdot(\psi^\dagger\alpha\psi).
$$
因此满足连续性方程。$\square$

**说明 21.7.** Dirac 方程虽有正定密度，但其负能解、外场中的谱不稳定和粒子产生问题说明单粒子解释不是完整相对论量子理论。完整处理需要量子场论中的场算符和反粒子解释。

**说明 21.8.** Klein-Gordon 方程也有守恒流，但其时间分量不正定，不能直接解释为普通概率密度。这一点是从非相对论波函数理论过渡到场论的第一个结构性障碍。

**命题 21.9（Klein-Gordon 流及其非正定性）.** 设 $m>0$。对光滑且具有足够空间衰减的复 Klein-Gordon 解 $\phi$，定义
$$
\rho_{\mathrm{KG}}
=\frac{i\hbar}{2mc^2}
\left(\overline\phi\,\partial_t\phi
-\phi\,\partial_t\overline\phi\right),
\qquad
\mathbf j_{\mathrm{KG}}
=-\frac{i\hbar}{2m}
\left(\overline\phi\,\nabla\phi
-\phi\,\nabla\overline\phi\right).
$$
则
$$
\partial_t\rho_{\mathrm{KG}}+\nabla\cdot\mathbf j_{\mathrm{KG}}=0.
$$
然而 $\rho_{\mathrm{KG}}$ 可取正也可取负，因此不同于 Schrodinger 理论中的 $|\psi|^2$。

**证明.** 用 $\overline\phi$ 乘 Klein-Gordon 方程，再减去其复共轭方程乘以 $\phi$。质量项相消，剩余项为
$$
\frac1{c^2}\partial_t
\left(\overline\phi\,\partial_t\phi
-\phi\,\partial_t\overline\phi\right)
-\nabla\cdot
\left(\overline\phi\,\nabla\phi
-\phi\,\nabla\overline\phi\right)=0.
$$
乘以 $i\hbar/(2m)$ 即得连续性方程。对频率相反的平面波，$\rho_{\mathrm{KG}}$ 符号相反，故不正定。$\square$

## 本章小结

相对论一粒子方程揭示了非相对论量子力学的边界。Klein-Gordon 和 Dirac 方程可以形式上量子化相对论能量关系，但粒子产生湮灭、因果性和局域性要求进入量子场论。

## 练习

**练习 21.1.** 从 $E^2=p^2c^2+m^2c^4$ 推导 Klein-Gordon 方程的形式替换。

**练习 21.2.** 验证 Dirac 矩阵反对易关系如何消去 $H_D^2$ 中的交叉项。
