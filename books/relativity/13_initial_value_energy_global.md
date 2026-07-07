# 第十三章 初值问题、能量与整体结构

## 13.1 为什么需要初值问题

Einstein 方程是非线性偏微分方程。要把它当作动力学理论，必须回答：给定某个空间切片上的数据，是否存在唯一的时空演化？

将四维时空按空间超曲面 $\Sigma_t$ 分解，诱导三维度规为 $\gamma_{ij}$，外曲率为 $K_{ij}$。初始数据为

$$
(\Sigma,\gamma_{ij},K_{ij},\text{matter data}).
$$

这些数据不能任意给定，必须满足约束方程。

## 13.2 ADM 约束

**命题 13.1（ADM 约束的投影来源）.** Einstein 方程沿空间切片单位法向 $n^\mu$ 和切向投影 $\gamma^\mu{}_i$ 的分量给出 Hamiltonian constraint 和 momentum constraint。

在无 $\Lambda$ 情况下，约束为

$$
{}^{(3)}R+K^2-K_{ij}K^{ij}=16\pi G\rho,
$$

以及

$$
D_j(K^{ij}-\gamma^{ij}K)=8\pi G j^i.
$$

其中 $D_i$ 是 $\gamma_{ij}$ 的 Levi-Civita 联络，${}^{(3)}R$ 是三维标量曲率，$\rho$ 和 $j^i$ 是相对于切片法向观察者测得的能量密度和动量密度。

第一式称为 Hamiltonian constraint，第二式称为 momentum constraint。

这些约束来自 Einstein 方程沿切片法向和切向的投影。设 $n^\mu$ 是未来指向单位法向量，$\gamma^\mu{}_\nu=\delta^\mu{}_\nu+n^\mu n_\nu$ 是投影到空间切片的投影算子。Gauss-Codazzi 关系给出

$$
G_{\mu\nu}n^\mu n^\nu
=\frac12
\left(
{}^{(3)}R+K^2-K_{ij}K^{ij}
\right),
$$

以及

$$
G_{\mu\nu}n^\mu\gamma^\nu{}_i
=D_jK^j{}_i-D_iK.
$$

将 Einstein 方程投影到这两个方向，就得到 Hamiltonian constraint 和 momentum constraint。其余空间-空间分量给出演化方程。

## 13.3 外部输入 A：局部适定性

**外部输入 A.** 给定满足约束的足够光滑初始数据，Einstein 方程在适当规范选择下存在局部唯一发展，并且存在最大 Cauchy 发展。

本书不证明该定理。证明需要把 Einstein 方程在谐和坐标等规范中化为准线性双曲系统，并处理约束传播。

约束传播的逻辑是：若初始切片满足约束，并且演化方程成立，则由 Bianchi 恒等式

$$
\nabla^\mu G_{\mu\nu}=0
$$

和物质守恒方程可推出约束违背量满足齐次演化系统。因此初始时为零的约束违背在局部演化中保持为零。严格证明需要固定规范和函数空间，本书只记录这一机制。

## 13.4 ADM 能量

对渐近平直时空，可以定义 ADM 质量。若空间度规在无穷远接近 Euclidean 度规：

$$
\gamma_{ij}=\delta_{ij}+O(r^{-1}),
$$

则 ADM 质量形式上为

$$
M_{\mathrm{ADM}}
=\frac{1}{16\pi G}
\lim_{r\to\infty}
\int_{S_r}
(\partial_j\gamma_{ij}-\partial_i\gamma_{jj})n^i\,dS.
$$

它测量孤立引力系统的总能量。

在 Minkowski 初始数据中 $\gamma_{ij}=\delta_{ij}$，积分中被积函数为零，所以 $M_{\mathrm{ADM}}=0$。对 Schwarzschild 初始数据，在合适渐近平直坐标下该积分给出参数 $M$。因此 ADM 质量把度规在无穷远的一阶偏离转化为总能量。

**外部输入 B (正质量定理).** 在适当能量条件和渐近平直假设下，

$$
M_{\mathrm{ADM}}\ge0,
$$

且等号只在 Minkowski 时空中达到。

## 13.5 能量条件

常见能量条件包括：

- Null energy condition: $T_{\mu\nu}k^\mu k^\nu\ge0$ 对所有类光 $k^\mu$。
- Weak energy condition: $T_{\mu\nu}u^\mu u^\nu\ge0$ 对所有类时 $u^\mu$。
- Dominant energy condition: 能量流不超光。
- Strong energy condition: $(T_{\mu\nu}-\frac12Tg_{\mu\nu})u^\mu u^\nu\ge0$。

它们不是数学必然，而是物质模型的物理假设。宇宙学常数正值违反 strong energy condition。

## 13.6 奇点定理

**外部输入 C (Penrose-Hawking 奇点定理).** 在适当因果条件、能量条件和困陷面或宇宙膨胀假设下，时空测地线不完备。

该定理说明 GR 的奇点不是球对称解的偶然产物，而是在广泛条件下出现的整体现象。它并不直接说明曲率一定发散；严格结论是测地线不完备。

## 13.7 全局双曲性

时空若存在 Cauchy 超曲面，即每条不可延拓因果曲线恰好与其相交一次，则称为全局双曲。全局双曲性保证初值描述有清楚意义。

非全局双曲时空可能存在 Cauchy horizon、闭合类时曲线或边界信息缺失，使初值问题不再良好。

## 13.8 时间对称初值数据

一个重要简化是时间对称初值数据，即

$$
K_{ij}=0.
$$

真空约束退化为

$$
{}^{(3)}R=0.
$$

若取共形形式

$$
\gamma_{ij}=\psi^4\tilde\gamma_{ij},
$$

则三维标量曲率满足

$$
{}^{(3)}R(\gamma)
=\psi^{-5}
\left(
-8\tilde\Delta\psi+\tilde R\psi
\right).
$$

在 $\tilde\gamma_{ij}=\delta_{ij}$ 的情形，真空约束变成

$$
\Delta\psi=0.
$$

Schwarzschild 时间对称切片在各向同性坐标中可写为

$$
\psi=1+\frac{M}{2r}.
$$

这给出一个简单例子：求解约束方程本身已经是非平凡椭圆型问题；它不是随便指定三维几何和外曲率即可。

## 13.9 整体结构的最低要求

在本书层面，讨论整体结构时至少要说明四件事：

1. 时空是否全局双曲。
2. 是否存在渐近平直、渐近 de Sitter 或渐近反 de Sitter 边界。
3. 是否存在事件视界、Cauchy 视界或闭合类时曲线。
4. 相关能量条件是否被物质模型满足。

这些条件决定“初值是否决定未来”“总能量是否可定义”“奇点定理是否可用”等问题。它们不是技术装饰，而是广义相对论作为动力学理论时不可省略的边界条件。

## 习题

1. 解释为什么 Einstein 方程初始数据必须满足约束。
2. 写出真空时间对称初始数据的 Hamiltonian constraint。
3. 说明 ADM 质量为什么只适合渐近平直时空。
4. 比较 weak energy condition 和 null energy condition。
5. 说明奇点定理的结论为什么是测地线不完备，而不直接是曲率发散。
