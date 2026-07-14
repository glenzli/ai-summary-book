# 第十七章 后 Newton 近似、参数化检验与数值相对论

正式的广义相对论教材不能只停在精确解。多数可观测系统既不是完全静态球对称，也不是完全线性波动：太阳系要求高精度弱场展开，双星 inspiral 要求后 Newton 波形，黑洞并合要求数值相对论。本章把这些方法放到统一框架中。

统一它们的不是一套万能公式，而是对误差来源的分层控制：后 Newton 方法按 $v/c$ 与势深展开，参数化后 Newton 形式把不同理论的弱场偏离编码为可测参数，有效一体方法重组双体动力学，数值相对论则在满足约束的 $3+1$ 初值上直接演化。本章说明这些方法如何衔接，并把完整高阶系数、稳定离散算法和波形目录明确保留为专门文献与计算框架的输入。

## 17.1 弱场慢速展开

设典型速度满足

$$
\epsilon\sim \frac{v^2}{c^2}\ll1,
$$

引力势满足

$$
\frac{|\Phi|}{c^2}\sim\epsilon.
$$

取 $c=1$ 后，后 Newton 展开按 $\epsilon$ 的阶数组织。固定一个
渐近惯性坐标系，假设场在 Newton 阶静态，测试粒子满足
$v^i=O(\epsilon^{1/2})$，且无标量各向异性应力时最低阶度规为

$$
g_{00}=-(1+2\Phi)+O(\epsilon^2),
$$

$$
g_{0i}=O(\epsilon^{3/2}),
$$

$$
g_{ij}=(1-2\Phi)\delta_{ij}+O(\epsilon^2).
$$

**命题 17.1（Newton 极限）.** 在上述坐标、阶数、静态性和慢速假设
下，测试粒子的坐标时间测地线方程在 Newton 阶为

$$
\frac{d^2x^i}{dt^2}=-\partial_i\Phi.
$$

**证明.** 设典型空间变化尺度为 $L$。由 $g_{00}=-(1+2\Phi)$、
$g^{ij}=\delta^{ij}+O(\epsilon)$ 和静态性，

$$
\Gamma^i{}_{00}
=-\frac12g^{ij}\partial_jg_{00}
=\partial_i\Phi+O(\epsilon^2/L).
$$

坐标时间形式的测地线方程中，$g_{0i}=O(\epsilon^{3/2})$ 的项还要
乘一个 $v^i=O(\epsilon^{1/2})$；含两个空间速度或非仿射参数修正的项
也至少再多一个 $O(\epsilon)$ 因子。因此其余项均为
$O(\epsilon^2/L)$，高于 Newton 阶，并有

$$
\frac{d^2x^i}{dt^2}
=-\Gamma^i{}_{00}+O(\epsilon^2/L)
=-\partial_i\Phi+O(\epsilon^2/L).
$$

保留最低阶即得结论。若再令 $\Lambda=0$，假设物质非相对论且
$T_{00}=\rho+O(\epsilon\rho)$，Einstein 方程的 $00$ 分量给出

$$
\nabla^2\Phi=4\pi G\rho.
$$

因此粒子方程与场方程一起恢复 Newton 引力。$\square$

## 17.2 1PN 形式的物理含义

后 Newton 近似中的 $1$PN 意味着在 Newton 理论之后保留 $O(v^2/c^2)$ 的相对论修正。它包含：

- 运动质量和动能的相对论修正。
- 空间曲率对粒子轨道和光线传播的影响。
- 引力场自身能量的贡献。
- 多体系统中的速度相关相互作用。

完整 $1$PN 多体方程较长，本书不全文列出，但保留两个最常用结果：

水星近日点进动：

$$
\Delta\phi
=\frac{6\pi GM}{a(1-e^2)c^2}
$$

每轨道。

Shapiro 时间延迟主项：

$$
\Delta t
\sim
\frac{2GM}{c^3}
\ln
\frac{4r_Er_R}{b^2}.
$$

其中 $a,e$ 是轨道半长轴和偏心率，$b$ 是光线冲击参数。

## 17.3 参数化后 Newton 形式

为了比较不同引力理论，常用 PPN 形式。最简单的静态弱场度规写为

$$
g_{00}=-1+2U-2\beta U^2+O(\epsilon^3),
$$

$$
g_{ij}=(1+2\gamma U)\delta_{ij}+O(\epsilon^2).
$$

上述 PPN 度规仍取 $c=1$，并采用点质量外部 $U=GM/r>0$ 的约定，
因此与前两节取负值的 Newton 势满足 $U=-\Phi$。恢复单位后，度规中
出现的是无量纲组合 $U/c^2=GM/(rc^2)$；两套公式的空间度规号差并不
矛盾。

广义相对论预言

$$
\beta=1,\qquad \gamma=1.
$$

光线偏折主项与 $(1+\gamma)$ 成正比：

$$
\alpha
=\frac{2(1+\gamma)GM}{bc^2}.
$$

当 $\gamma=1$ 时恢复 GR 的

$$
\alpha=\frac{4GM}{bc^2}.
$$

PPN 的价值在于把“检验 GR”变成测量一组参数是否等于 GR 预言。

## 17.4 双星能量损失

考虑两体系统约化质量 $\mu$、总质量 $M$、近圆轨道半径 $r$。Newton 轨道能量为

$$
E=-\frac{G\mu M}{2r}.
$$

四极矩公式给出引力波辐射功率主项

$$
P
=-\frac{dE}{dt}
=\frac{32}{5}
\frac{G^4}{c^5}
\frac{\mu^2M^3}{r^5}.
$$

因此轨道半径随时间减小：

$$
\frac{dr}{dt}
=-\frac{64}{5}
\frac{G^3}{c^5}
\frac{\mu M^2}{r^3}.
$$

频率升高、周期缩短，形成 chirp。双中子星 Hulse-Taylor 系统和现代引力波探测都依赖这种能量损失图像。

观测中更常用的是 chirp 质量

$$
\mathcal M
=\mu^{3/5}M^{2/5}.
$$

对近圆轨道，Kepler 关系给出

$$
\Omega^2=\frac{GM}{r^3}.
$$

主导四极矩辐射使引力波频率 $f$ 的演化满足

$$
\frac{df}{dt}
=\frac{96}{5}\pi^{8/3}
\left(\frac{G\mathcal M}{c^3}\right)^{5/3}
f^{11/3}.
$$

这说明 inspiral 波形相位首先测到的是 $\mathcal M$，而不是两个质量各自的数值。质量比、自旋和潮汐效应需要更高阶相位修正来区分。

## 17.5 有效一体问题

有效一体问题 EOB 的思想是把两体相对论动力学重写为一个有效粒子在变形黑洞背景中的运动，并加入辐射反作用。它把以下输入合并：

- 后 Newton 展开。
- 黑洞微扰理论。
- 数值相对论校准。
- ringdown 准正规模。

本书不构造 EOB Hamiltonian，但强调其角色：它是从解析近似通向实用引力波模板的桥梁。

## 17.6 数值相对论的 3+1 形式

四维度规可写成

$$
ds^2
=-N^2dt^2
+\gamma_{ij}(dx^i+N^idt)(dx^j+N^jdt),
$$

其中 $N$ 是 lapse，$N^i$ 是 shift，$\gamma_{ij}$ 是空间度规。外曲率定义为

$$
K_{ij}
=-\frac{1}{2N}
\left(
\partial_t\gamma_{ij}
-D_iN_j-D_jN_i
\right).
$$

对应的未来单位法向量为

$$
n^\mu\partial_\mu
=\frac1N(\partial_t-N^i\partial_i),
$$

所以上式正是 $K_{ij}=-\tfrac12\mathcal L_n\gamma_{ij}$；它与第十三章
的动量密度号差必须成对使用。

Einstein 方程分为约束方程和演化方程。约束已在第十三章给出；演化方程可写为

$$
\partial_t\gamma_{ij}
=-2NK_{ij}+D_iN_j+D_jN_i.
$$

$K_{ij}$ 的演化方程包含三维 Ricci 张量、$K_{ij}$ 的二次项、lapse 的 Hessian 和物质源项。

## 17.7 BSSN 变量

原始 ADM 形式数值稳定性不理想。BSSN 形式在选定的空间坐标中引入
共形分解：

$$
\gamma_{ij}=e^{4\varphi}\tilde\gamma_{ij},
\qquad
\det\tilde\gamma_{ij}=1.
$$

外曲率分解为迹和无迹部分：

$$
K=\gamma^{ij}K_{ij},
$$

$$
\tilde A_{ij}
=e^{-4\varphi}
\left(
K_{ij}-\frac13\gamma_{ij}K
\right).
$$

再引入共形联络函数

$$
\tilde\Gamma^i=-\partial_j\tilde\gamma^{ij}.
$$

$\det\tilde\gamma=1$ 与上述偏导表达式依赖所选坐标密度约定；
$\tilde\Gamma^i$ 不是任意空间坐标变换下的普通向量。

BSSN 的目的不是改变 Einstein 方程，而是选择更适合数值演化的变量，使约束控制和强双曲性质更好。

## 17.8 波形提取

数值模拟得到的是有限半径处的时空数据。取 null tetrad
$(\ell,n,m,\bar m)$ 满足 $\ell\cdot n=-1$、
$m\cdot\bar m=1$，并在本书曲率约定下定义

$$
\Psi_4=-C_{\alpha\beta\gamma\delta}
n^\alpha\bar m^\beta n^\gamma\bar m^\delta.
$$

再选择与外向波传播方向及 $(+,\times)$ 偏振基相配的 tetrad，则远区有

$$
\Psi_4
\approx
\ddot h_+-i\ddot h_\times
$$

这里同时选定了 Newman--Penrose 标量定义、外向 null tetrad 的取向和
偏振复组合；改变曲率或 tetrad 约定可能使右侧整体变号或复共轭，故该式
不是脱离这些选择的裸符号恒等式。实际波形还需外推到
无穷远或使用 Cauchy-characteristic extraction。然后把波形分解成自旋权球谐模式

$$
h(t,\theta,\phi)
=\sum_{\ell,m}h_{\ell m}(t)\,{}_{-2}Y_{\ell m}(\theta,\phi).
$$

最强模式通常是 $\ell=2,m=\pm2$。

## 17.9 正式教材中的边界

本章给出后 Newton、PPN、EOB 和数值相对论的入口，但完整理论分别需要：

- 高阶 PN 计算和正则化。
- PPN 全参数体系。
- EOB Hamiltonian 的系统构造。
- 数值 PDE 稳定性、网格和边界条件。
- 黑洞微扰与准正规模谱。

这些属于专题课程。相对论主教材需要让读者知道这些方法如何连接第八至第十三章的核心方程。

## 习题

1. 从弱场度规的测地线方程恢复 Newton 方程。
2. 用 PPN 参数 $\gamma$ 写出光线偏折主项，并代入 GR 值。
3. 由 $E=-G\mu M/(2r)$ 和四极矩辐射功率推导 $dr/dt$。
4. 写出 $3+1$ 度规分解中 lapse 和 shift 的几何含义。
5. 解释为什么数值相对论需要变量重写，而不是直接演化原始 Einstein 方程。
