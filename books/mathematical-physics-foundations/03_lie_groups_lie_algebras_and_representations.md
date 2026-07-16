# 第三章：Lie 群、Lie 代数与表示

对称性只有作用在对象上才有数学内容。经典系统中，Lie 群作用在相空间上并产生守恒量；量子系统中，它通过 Hilbert 空间的酉表示产生量子数和选择定则；场论中，局域对称性进一步变成主丛联络。本章建立从 Lie 群到 Lie 代数、从抽象对称到矩阵表示的基本桥梁。

## 3.1 Lie 群与 Lie 代数

**定义 3.1.** Lie 群是同时为光滑流形和群的对象，使乘法与取逆为光滑映射。

**定义 3.2.** Lie 群 $G$ 的 Lie 代数 $\mathfrak g=T_eG$，括号由左不变向量场的 Lie 括号诱导。

**命题 3.1 (`P`).** $\mathfrak g$ 与 $G$ 上左不变向量场空间自然同构。

**证明.** 给定 $\xi\in T_eG$，令 $X_\xi(g)=(dL_g)_e\xi$。乘法的光滑性说明 $X_\xi$ 光滑，而且
$$
(dL_h)_gX_\xi(g)=(dL_{hg})_e\xi=X_\xi(hg),
$$
故它左不变。反过来，若 $X$ 左不变，则 $X(g)=(dL_g)_eX(e)$，所以 $X$ 由 $X(e)$ 唯一决定。这两个线性构造互逆。

若 $X,Y$ 左不变，则自然性公式 $(L_h)_*[X,Y]=[(L_h)_*X,(L_h)_*Y]=[X,Y]$ 表明 $[X,Y]$ 仍左不变。因此把 $[X_\xi,X_\eta](e)$ 定义为 $[\xi,\eta]$ 后，上述同构也保持括号。$\square$

**定义 3.3.** 指数映射 $\exp:\mathfrak g\to G$ 定义为左不变向量场 $X_\xi$ 从单位元出发的时间 $1$ 流。该流对所有实时间存在：局部 ODE 定理先给出单位元附近的一小段积分曲线；左平移保持 $X_\xi$，所以可把任意已存在的小段逐段左平移并拼接，唯一性保证重叠处一致。因而时间 $1$ 的取值对每个 $\xi$ 都有定义。

## 3.2 表示与微分

**定义 3.4.** $G$ 在向量空间 $V$ 上的表示是群同态 $\rho:G\to GL(V)$。若 $V$ 为 Hilbert 空间且 $\rho(g)$ 酉，则称为酉表示。

**命题 3.2 (`P`).** 若 $\rho:G\to GL(V)$ 为有限维光滑表示，则
$$
d\rho:\mathfrak g\to\mathfrak{gl}(V),\qquad
d\rho(\xi)=\left.\frac{d}{dt}\right|_{t=0}\rho(\exp t\xi)
$$
是 Lie 代数表示。

**证明.** 把 $\rho$ 看作 Lie 群光滑同态 $G\to GL(V)$。对 $\xi\in\mathfrak g$，上一命题给出左不变向量场 $X_\xi$。同态性质给出 $\rho\circ L_g=L_{\rho(g)}\circ\rho$，故链式法则说明 $X_\xi$ 与 $GL(V)$ 上由 $(d\rho)_e\xi$ 生成的左不变向量场 $Y_{d\rho(\xi)}$ 是 $\rho$-相关的。相关向量场的 Lie 括号仍相关，因此
$$
\begin{aligned}
(d\rho)_e[\xi,\eta]
&=(d\rho)_e[X_\xi,X_\eta](e)\\
&=[Y_{d\rho(\xi)},Y_{d\rho(\eta)}](I)
=[d\rho(\xi),d\rho(\eta)],
\end{aligned}
$$
最后一个括号是矩阵交换子。线性来自微分的线性，故 $d\rho$ 是 Lie 代数表示。$\square$

**命题 3.3 (`P`, Schur 引理).** 若 $V,W$ 是复数域上的有限维不可约表示，且 $T:V\to W$ 交错表示作用，则 $T=0$ 或 $T$ 为同构。特别地，若 $V=W$，则 $T=\lambda I$。

**证明.** 对任意 $g\in G$，$Tv=0$ 蕴含 $T\rho_V(g)v=\rho_W(g)Tv=0$，故 $\ker T$ 是不变子空间；同理 $\operatorname{im}T$ 不变。若 $T\ne0$，不可约性给出 $\ker T=0$ 与 $\operatorname{im}T=W$，所以 $T$ 为同构。

现在令 $V=W$。有限维复线性算符 $T$ 有特征值 $\lambda$。算符 $T-\lambda I$ 仍与表示交错，且其核含有对应特征向量，因而非零。第一部分迫使 $T-\lambda I=0$，即 $T=\lambda I$。$\square$

## 3.3 紧群和粒子标签

**定理 3.4 (`E`, Peter-Weyl).** 对紧 Lie 群 $G$，其不可约有限维酉表示的矩阵系数在 $L^2(G)$ 中张成稠密子空间，并给出正则表示的 Hilbert 直和分解。

**外部输入边界.** 本书使用该定理解释紧对称群的 Fourier 分解与量子数离散性，不证明紧群调和分析细节；精确定位见 [SOURCES.md](SOURCES.md) 的 `E-3.4`。

**定理 3.5 (`E`, $SU(2)$ 有限维表示分类).** 每个有限维复不可约光滑表示 $\rho:SU(2)\to GL(V)$ 同构于唯一一个最高权为 $m\in\mathbb Z_{\ge0}$ 的表示；令 $j=m/2$，则 $\dim_{\mathbb C}V=2j+1$。反之，每个 $j\in\frac12\mathbb Z_{\ge0}$ 都出现。

**证明路线（外部输入）.** 先把表示微分并复化为 $\mathfrak{sl}_2(\mathbb C)$ 表示；最高权理论给出权链 $m,m-2,\ldots,-m$ 及其唯一性。随后使用 $SU(2)$ 单连通性，把 Lie 代数表示积分回群表示。第六章会在有限维酉角动量表示内完整证明升降算符和 Casimir 公式，但这里不重建“微分与积分互逆”以及全部最高权分类。精确来源见 [SOURCES.md](SOURCES.md) 的 `E-3.5`。

**例 3.6（平面旋转及其微分）.** 写
$$
R(\theta)=
\begin{pmatrix}\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta\end{pmatrix},
\qquad
J=\begin{pmatrix}0&-1\\1&0\end{pmatrix}.
$$
直接相乘得 $R(\theta_1)R(\theta_2)=R(\theta_1+\theta_2)$。其微分为
$dR(1)=\left.\frac d{dt}\right|_{t=0}R(t)=J$。由于 $J^2=-I$，指数级数的偶、奇部分给出
$e^{\theta J}=I\cos\theta+J\sin\theta=R(\theta)$。这完整展示了群表示、Lie 代数生成元和指数映射的关系。

## 练习

**练习 3.1.** 证明 $SO(3)$ 的 Lie 代数可与 $(\mathbb R^3,\times)$ 同构。

**练习 3.2.** 对自旋 $j$ 表示，计算二次 Casimir $J^2=J_1^2+J_2^2+J_3^2$ 的本征值。
