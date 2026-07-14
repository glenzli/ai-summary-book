# 第五章：时间演化、Stone 定理与 Schrodinger 方程

没有测量发生时，封闭系统仍会改变：叠加态的不同能量分量积累不同相位，随后对另一可观测量的测量概率便随时间振荡。演化必须保持归一化和所有内积，因此由酉算子实现；若系统自治，先演化 $s$ 再演化 $t$ 与直接演化 $t+s$ 相同，这才产生一参数群。把这一群写成 $e^{-itH}$ 并非矩阵记号的直接推广，因为无限维 Hamiltonian 通常无界，指数的存在与生成元的自伴性需要 Stone 定理。

本章先从酉算子的几何性质进入，再在自治口径下连接强连续群、Hamiltonian 与 Schrodinger 方程。Schrodinger 图像和 Heisenberg 图像随后把同一统计变化分别归因于态或可观测量。最后的二能级计算会显示一个关键现象：能量测量概率保持不变，并不意味着所有测量概率都不变。显含时间的 Hamiltonian 不满足一参数群口径，而由二参数传播子描述，相关构造在含时扰动章节中另行完成。

## 5.1 酉演化

**定义 5.1.** 有界算子 $U$ 称为酉算子，若
$$
U^*U=I=UU^*.
$$

**命题 5.2.** 酉算子保持内积和转移概率。

**证明.** 对任意 $\psi,\phi$，
$$
\langle U\psi,U\phi\rangle=\langle\psi,U^*U\phi\rangle=\langle\psi,\phi\rangle.
$$
取绝对值平方即得转移概率保持。$\square$

**公设 5.3（自治封闭系统演化）.** 时间平移齐次、由时间无关
Hamiltonian 支配的封闭系统，其时间演化由强连续一参数酉群 $U(t)$ 给出：
$$
U(0)=I,\qquad U(t+s)=U(t)U(s),
$$
且对每个 $\psi$，$t\mapsto U(t)\psi$ 连续。若 Hamiltonian 显含时间，
一般只要求二参数酉传播子满足
$U(t,s)U(s,r)=U(t,r)$ 与 $U(s,s)=I$，不把它称为一参数群。

## 5.2 Stone 定理与 Hamiltonian

**外部输入定理 5.4（Stone 定理，QM-EXT-2）.** 强连续一参数酉群唯一写成
$$
U(t)=e^{-itH},
$$
其中 $H$ 是自伴算子。

**定义 5.5.** Stone 定理中的自伴算子 $H$ 称为 Hamiltonian。若 $\psi_0\in\mathcal D(H)$，令 $\psi(t)=U(t)\psi_0$，则
$$
i\frac{d}{dt}\psi(t)=H\psi(t)
$$
称为 Schrodinger 方程。

**命题 5.6.** 若 $H$ 为有界自伴算子，则 $U(t)=e^{-itH}$ 满足 Schrodinger 方程。

**证明.** 有界算子的指数由范数收敛级数定义：
$$
e^{-itH}=\sum_{n=0}^\infty\frac{(-itH)^n}{n!}.
$$
逐项求导在算子范数中合法，得
$$
\frac{d}{dt}e^{-itH}=-iHe^{-itH}.
$$
于是 $\psi(t)=e^{-itH}\psi_0$ 满足 $i\dot\psi(t)=H\psi(t)$。$\square$

生成元形式把时间演化变成微分方程。预测同一个实验统计时，可以让态按该方程演化，也可以固定态而把可观测量作酉共轭；两种选择只是记账位置不同。

## 5.3 Heisenberg 图像

**定义 5.7.** Schrodinger 图像中态随时间变：
$$
\psi(t)=U(t)\psi(0).
$$
Heisenberg 图像中可观测量随时间变：
$$
A_H(t)=U(t)^*AU(t).
$$

**命题 5.8.** 若 $H,A$ 有界自伴，则
$$
\frac{d}{dt}A_H(t)=i[H,A_H(t)].
$$

**证明.** 由 $U'(t)=-iHU(t)$ 与 $(U^*)'(t)=iU(t)^*H$，
$$
\frac{d}{dt}U^*AU=iU^*HAU-iU^*AHU=i[H,U^*AU],
$$
其中 $H$ 与 $U(t)$ 交换，因为 $U(t)$ 是 $H$ 的函数。$\square$

Heisenberg 方程把交换子变成变化率，但最直接的计算往往仍来自
Hamiltonian 的谱分解：每个能量子空间只积累一个相位。

## 5.4 谱表示中的相位演化

**命题 5.9.** 若有限维 Hamiltonian 有谱分解
$$
H=\sum_rE_rP_r,
$$
则
$$
U(t)=e^{-itH}=\sum_re^{-itE_r}P_r.
$$

**证明.** 对指数函数使用有限维函数演算。也可直接验证右边满足
$$
U(0)=I,\qquad \frac d{dt}U(t)=-iHU(t),
$$
并由线性常微分方程解的唯一性得到。$\square$

**推论 5.10.** 若初态分解为
$$
\psi=\sum_r\psi_r,\qquad \psi_r=P_r\psi,
$$
则
$$
\psi(t)=\sum_re^{-itE_r}\psi_r.
$$
能量本征分量的概率 $\|\psi_r\|^2$ 不随时间变。

**证明.** 由 $\psi_r=P_r\psi$ 与命题 5.9，
$U(t)\psi=\sum_re^{-itE_r}P_r\psi$。每个相位因子的模为 $1$，故
$\|e^{-itE_r}\psi_r\|^2=\|\psi_r\|^2$。$\square$

**例子 5.10A（二能级相位与可见振荡）.** 令
$$
H=\frac{\omega}{2}\sigma_z,\qquad
|+x\rangle=\frac1{\sqrt2}\begin{pmatrix}1\\1\end{pmatrix}.
$$
则
$$
U(t)|+x\rangle
=\frac1{\sqrt2}
\begin{pmatrix}e^{-i\omega t/2}\\e^{i\omega t/2}\end{pmatrix}.
$$
能量的两个结果概率始终各为 $1/2$。若在时刻 $t$ 测量
$\sigma_x$，得到 $+1$ 的振幅和概率分别为
$$
\langle+x|U(t)|+x\rangle=\cos\frac{\omega t}{2},
\qquad
\Pr_t(\sigma_x=+1)=\cos^2\frac{\omega t}{2}.
$$
相对相位因此可以不改变能量分布，却改变另一组谱投影的概率。

自治演化的完整链条现在是：强连续一参数酉群由唯一自伴
Hamiltonian 生成，定义域中的初态满足 Schrodinger 方程，酉共轭则给出
Heisenberg 图像。二能级例子说明真正可观测的是能量分量之间的相对
相位。下一章把抽象 Hamiltonian 具体化为一维微分算子，并在那里同时
面对自伴实现、边界条件和概率流匹配。

## 练习

**练习 5.1.** 证明酉演化保持态的归一化。

**练习 5.2.** 若 $[A,H]=0$ 且算子有界，证明 $\langle A\rangle_{\psi(t)}$ 与时间无关。
