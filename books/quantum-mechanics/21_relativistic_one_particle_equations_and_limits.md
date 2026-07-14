# 第二十一章：相对论一粒子方程与适用边界

把非相对论色散关系 $E=p^2/(2m)$ 换成
$E^2=p^2c^2+m^2c^4$ 后，直接量子化得到的 Klein--Gordon 方程对时间是
二阶的，其守恒流时间分量可以为负；若要求一阶时间演化，Dirac 方程
必须引入满足 Clifford 反对易关系的矩阵和多分量波函数。两种方程都能
恢复相对论色散，却不能仅凭“一粒子波函数”处理粒子产生、负能谱和
反粒子。

本章暂时恢复 $\hbar$ 与 $c$，所有公式均服从这一局部单位约定。先从
形式算子替换得到 Klein--Gordon 方程，再验证 Dirac Hamiltonian 的平方
和正定密度连续性方程。低动量展开说明去掉静止能相位后如何回到
Schrodinger 动力学。最后对正、负频 Klein--Gordon 平面波直接计算电荷型
密度的相反符号，从而把“为何需要场论”的边界落实到可检查公式，而不是
只作概念宣告。

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

Klein--Gordon 方程的二阶时间结构允许正、负频两支。Dirac 的思路是把
色散关系线性化为一阶时间方程，代价是引入不能同时对角化的矩阵自由度。

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

平方恒等式保证 Dirac 方程具有正确色散。要与本书的非相对论主线连接，
还需在 $|p|/(mc)\ll1$ 的量纲明确极限中展开正能分支。

## 21.3 非相对论极限

**说明 21.5.** 设 $m>0$。当 $|p|/(mc)\ll1$ 时，
$$
\sqrt{p^2c^2+m^2c^4}
=mc^2+\frac{p^2}{2m}-\frac{p^4}{8m^3c^2}
+O\!\left(\frac{p^6}{m^5c^4}\right).
$$
去掉静止能相位 $e^{-imc^2t/\hbar}$ 后，低能极限恢复 Schrodinger 动力学，并伴随自旋和磁矩修正。

能量展开说明低能动力学，却没有回答波函数能否按 Born 规则解释。下面
分别检查 Dirac 与 Klein--Gordon 方程的局部守恒流及其正定性。

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

**例子 21.9A（Klein--Gordon 正负频密度）.** 令
$$
\omega_k=\sqrt{c^2|k|^2+\frac{m^2c^4}{\hbar^2}}
$$
并取平面波广义解
$$
\phi_\pm(t,x)=A e^{\mp i\omega_kt+i k\cdot x}.
$$
代入命题 21.9 的密度得到
$$
\rho_{\mathrm{KG}}[\phi_+]
=\frac{\hbar\omega_k}{mc^2}|A|^2,\qquad
\rho_{\mathrm{KG}}[\phi_-]
=-\frac{\hbar\omega_k}{mc^2}|A|^2.
$$
二者空间模平方相同，守恒密度却符号相反，因此该密度不能作为对所有
一粒子态非负的 Born 概率。

Klein--Gordon 与 Dirac 方程都实现了相对论色散，Dirac 密度还保持正定；
但负频支、负能谱和外场中的粒子数变化表明，固定一粒子 Hilbert 空间
不是完整相对论理论。低动量展开只说明 Schrodinger 方程怎样作为正能
分支的近似出现，并不消除这些结构性障碍。下一章回到可靠的非相对论
范围，用旋转对称性求解中心势与氢型束缚态。

## 练习

**练习 21.1.** 从 $E^2=p^2c^2+m^2c^4$ 推导 Klein-Gordon 方程的形式替换。

**练习 21.2.** 验证 Dirac 矩阵反对易关系如何消去 $H_D^2$ 中的交叉项。
