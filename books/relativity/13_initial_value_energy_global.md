# 第十三章 初值问题、能量与整体结构

## 13.1 为什么需要初值问题

Einstein 方程是非线性偏微分方程。要把它当作动力学理论，必须回答：给定某个空间切片上的数据，是否存在唯一的时空演化？

将四维时空按空间超曲面 $\Sigma_t$ 分解，诱导三维度规为 $\gamma_{ij}$，外曲率为 $K_{ij}$。初始数据为

$$
(\Sigma,\gamma_{ij},K_{ij},\text{matter data}).
$$

这些数据不能任意给定，必须满足约束方程。

## 13.2 ADM 约束

**命题 13.1（ADM 约束的投影来源）.** 采用
$(-,+,+,+)$、$K_{ij}=-\tfrac12\mathcal L_n\gamma_{ij}$ 与
$j_i=-T_{\mu\nu}n^\mu\gamma^\nu{}_i$。当 $\Lambda=0$ 时，Einstein
方程沿空间切片未来单位法向 $n^\mu$ 和切向投影 $\gamma^\mu{}_i$ 的
分量分别给出下述 Hamiltonian constraint 和 momentum constraint。

在无 $\Lambda$ 情况下，约束为

$$
{}^{(3)}R+K^2-K_{ij}K^{ij}=16\pi G\rho,
$$

以及

$$
D_j(K^{ij}-\gamma^{ij}K)=8\pi G j^i.
$$

其中 $D_i$ 是 $\gamma_{ij}$ 的 Levi-Civita 联络，${}^{(3)}R$ 是三维标量曲率。配合第十七章
$$
K_{ij}=-\frac12\mathcal L_n\gamma_{ij}
$$
的号差，本章定义法向观察者测得的能量密度与空间动量密度为
$$
\rho=T_{\mu\nu}n^\mu n^\nu,
\qquad
j_i=-T_{\mu\nu}n^\mu\gamma^\nu{}_i.
$$

第一式称为 Hamiltonian constraint，第二式称为 momentum constraint。

**证明.** 采用本书 mostly-plus 号差和上面的 $K_{ij},j_i$ 定义。这些约束来自 Einstein 方程沿切片法向和切向的投影。设 $n^\mu$ 是未来指向单位法向量，$\gamma^\mu{}_\nu=\delta^\mu{}_\nu+n^\mu n_\nu$ 是投影到空间切片的投影算子。Gauss--Codazzi 关系给出

$$
G_{\mu\nu}n^\mu n^\nu
=\frac12
\left(
{}^{(3)}R+K^2-K_{ij}K^{ij}
\right),
$$

以及

$$
-G_{\mu\nu}n^\mu\gamma^\nu{}_i
=D_jK^j{}_i-D_iK.
$$

将 Einstein 方程投影到这两个方向，并使用
$$
-G_{\mu\nu}n^\mu\gamma^\nu{}_i
=-8\pi G T_{\mu\nu}n^\mu\gamma^\nu{}_i
=8\pi G j_i,
$$
就得到 Hamiltonian constraint 和 momentum constraint。其余空间-空间分量给出演化方程。
$\square$

若显式保留宇宙学常数，则法向--法向投影改为

$$
{}^{(3)}R+K^2-K_{ij}K^{ij}
=16\pi G\rho+2\Lambda,
$$

而动量约束不变，因为 $g_{\mu\nu}n^\mu\gamma^\nu{}_i=0$。这一
$2\Lambda$ 的位置也可作为 ADM 号差的交叉检查。

## 13.3 外部输入 A：最大 Cauchy 发展

**外部输入 A（真空最大 Cauchy 发展）.** 设 $\Sigma$ 是光滑三维流形，
$(\gamma_{ij},K_{ij})$ 是其上满足真空约束的光滑初始数据。则存在一个
以 $\Sigma$ 为 Cauchy 超曲面的真空、全局双曲发展；所有这类发展中
存在最大者，并在保持初始嵌入的等距意义下唯一。这里的唯一性不是固定
坐标分量逐点相等，而是模微分同胚的几何唯一性。

本书不证明该定理。局部存在性需要把 Einstein 方程在谐和坐标等规范
中化为准线性双曲系统；最大化与几何唯一性还需要
Choquet-Bruhat--Geroch 的发展拼接论证。加入物质时，只有当相应物质
方程与 Einstein 方程组成适定的双曲系统并满足兼容约束时，才能使用
对应版本，不能由真空定理自动推出。

约束传播的逻辑是：若初始切片满足约束，并且演化方程成立，则由 Bianchi 恒等式

$$
\nabla^\mu G_{\mu\nu}=0
$$

和物质守恒方程可推出约束违背量满足齐次演化系统。因此初始时为零的约束违背在局部演化中保持为零。严格证明需要固定规范和函数空间，本书只记录这一机制。

## 13.4 ADM 能量

固定一个渐近平直端，并在该端选择一个渐近 Lorentz 系：空间切片上的
渐近 Cartesian 坐标固定其空间轴，切片法向固定渐近时间方向。至少要求

$$
\gamma_{ij}=\delta_{ij}+O(r^{-1}),
\qquad
\partial_k\gamma_{ij}=O(r^{-2}),
\qquad
K_{ij}=O(r^{-2}),
$$

连同使极限存在并在容许渐近坐标变换下具有正确变换律的标准可积性与
奇偶条件。在这一选定渐近系中，ADM 能量定义为

$$
E_{\mathrm{ADM}}
=\frac{1}{16\pi G}
\lim_{r\to\infty}
\int_{S_r}
(\partial_j\gamma_{ij}-\partial_i\gamma_{jj})n^i\,dS.
$$

同一渐近系中的 ADM 空间动量为
$$
P_i=P_i^{\mathrm{ADM}}
=\frac{1}{8\pi G}
\lim_{r\to\infty}
\int_{S_r}(K_{ij}-K\gamma_{ij})n^j\,dS.
$$
这里 $n^i$ 是 $S_r\subset\Sigma$ 在三维度规中的向外单位法向；上式的
号差与本章
$K_{ij}=-\tfrac12\mathcal L_n\gamma_{ij}$ 的约定配套。仅写
$\gamma_{ij}=\delta_{ij}+O(r^{-1})$ 而不控制导数、$K_{ij}$、渐近坐标
类和可积性，不足以保证这些积分定义良好。

在容许的渐近 Lorentz 变换下，
$$
P_{\mathrm{ADM}}^\mu=(E_{\mathrm{ADM}},P^i)
$$
作为四向量变换；因此 $E_{\mathrm{ADM}}$ 是依赖渐近观察者的能量，
不是 Lorentz 标量。若该四动量未来非类空，则定义 ADM 不变质量
$$
m_{\mathrm{ADM}}
=\sqrt{-\eta_{\mu\nu}P_{\mathrm{ADM}}^\mu P_{\mathrm{ADM}}^\nu}
=\sqrt{E_{\mathrm{ADM}}^2-\delta^{ij}P_iP_j}.
$$
若四动量类时，存在渐近静止系使 $P_i'=0$，并且只在该系中有
$E_{\mathrm{ADM}}'=m_{\mathrm{ADM}}$。非零类光四动量没有静止系；零
四动量则不选出唯一静止系。一般初值数据中把上述能量积分直接称为
“ADM 质量”会混淆这三个对象。

在 Minkowski 标准初始数据中
$\gamma_{ij}=\delta_{ij}$、$K_{ij}=0$，所以
$E_{\mathrm{ADM}}=0$、$P_i=0$ 与 $m_{\mathrm{ADM}}=0$。对
Schwarzschild 的标准时间对称静止切片，$P_i=0$，能量积分给出参数
$M$，故 $E_{\mathrm{ADM}}=m_{\mathrm{ADM}}=M$。这两个例子都处在
零空间动量的渐近系中，不能用来把一般系中的能量与不变质量混同。

**外部输入 B（三维 Riemannian 正质量定理）.** 设
$(\Sigma,\gamma)$ 是完备、连通、无边界的光滑三维 Riemannian 流形，
具有满足上述标准衰减条件的渐近平直端，并且
${}^{(3)}R(\gamma)\ge0$。把这些 Riemannian 数据视为时间对称初始
数据 $K_{ij}=0$ 时，每个渐近平直端满足 $P_i=0$，且

$$
E_{\mathrm{ADM}}=m_{\mathrm{ADM}}\ge0,
$$

若某端等号成立，则 $(\Sigma,\gamma)$ 等距于 Euclidean
$(\mathbb R^3,\delta)$。在 $\Lambda=0$ 的时间对称数据
$K_{ij}=0$ 中，Hamiltonian constraint 与非负能量密度给出
${}^{(3)}R=16\pi G\rho\ge0$，故该版本可直接使用。一般
$K_{ij}\ne0$ 的正能量定理比较完整 ADM 四动量，并在 dominant energy
condition 及标准完备性、衰减与约束假设下给出
$E_{\mathrm{ADM}}\ge(\delta^{ij}P_iP_j)^{1/2}$；本书把该更强版本作为
外部边界，不把它压缩进上面的 Riemannian 陈述。没有这样的因果性结论
时，根号公式不应被无条件宣称为实数质量。

## 13.5 能量条件

常见能量条件包括：

- Null energy condition: 对每个非零类光向量 $k^\mu$，
  $T_{\mu\nu}k^\mu k^\nu\ge0$。
- Weak energy condition: 对每个未来类时向量 $u^\mu$，
  $T_{\mu\nu}u^\mu u^\nu\ge0$。
- Dominant energy condition: 对每个未来类时向量 $u^\mu$，能流
  $-T^\mu{}_{\nu}u^\nu$ 为未来因果向量；这同时包含其能量密度非负。
- Strong energy condition: 对每个类时向量 $u^\mu$，
  $(T_{\mu\nu}-\frac12Tg_{\mu\nu})u^\mu u^\nu\ge0$。

它们不是数学必然，而是物质模型的物理假设。宇宙学常数正值违反 strong energy condition。

## 13.6 奇点定理

**外部输入 C（Penrose 1965 奇点定理的一种标准版本）.** 设四维、
时间定向、全局双曲时空具有非紧 Cauchy 超曲面，满足 null convergence
condition

$$
R_{\mu\nu}k^\mu k^\nu\ge0
$$

对每个类光向量 $k^\mu$ 成立，并含有一个闭合未来困陷二维曲面。则该
时空未来类光测地线不完备。

该版本不声称存在曲率发散点，也不声称困陷面本身就是事件视界；严格
结论是至少一条未来类光测地线只具有有限仿射长度。Hawking 的宇宙学
版本使用不同的初始膨胀与能量假设，本书不把两类定理混成一个省略假设
的“Penrose--Hawking 定理”。

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
\psi=1+\frac{GM}{2r}.
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
3. 说明 ADM 四动量为什么只适合渐近平直时空，并区分
   $E_{\mathrm{ADM}}$、$P_i$ 与 $m_{\mathrm{ADM}}$。
4. 比较 weak energy condition 和 null energy condition。
5. 说明奇点定理的结论为什么是测地线不完备，而不直接是曲率发散。
