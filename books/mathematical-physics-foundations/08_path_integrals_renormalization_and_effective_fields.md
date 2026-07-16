# 第八章：路径积分、重整化与有效场论

路径积分把量子振幅写成“所有历史的相位和”，但无穷维场空间上并不存在自动可用的 Lebesgue 测度。可靠的计算链条应从有限维 Gaussian 积分开始：格点或模截止把自由度变成有限个，Wick 展开成为普通积分恒等式；只有在讨论移除 cutoff、微扰重整化和连续极限时，论证才进入标准物理口径。本章沿这条顺序区分可证明的有限维事实、外部的微扰重整化定理和仍属形式主义的连续场路径积分。

## 8.1 有限维 Gaussian 与 Wick 展开

**命题 8.1 (`P`, Euclidean Gaussian 积分).** 设 $A\in M_n(\mathbb R)$ 为实对称正定矩阵，$J\in\mathbb R^n$。取 $(\det A)^{1/2}>0$，则绝对收敛积分满足
$$
\int_{\mathbb R^n}
e^{-\frac12x^TAx+J^Tx}\,d^nx
=(2\pi)^{n/2}(\det A)^{-1/2}
e^{\frac12J^TA^{-1}J}.
$$

**证明.** 先计算一维积分 $I=\int_{\mathbb R}e^{-u^2/2}\,du$。被积函数非负，Tonelli 定理与极坐标换元给出
$$
I^2=\int_{\mathbb R^2}e^{-(u^2+v^2)/2}\,du\,dv
=2\pi\int_0^\infty e^{-r^2/2}r\,dr=2\pi,
$$
故 $I=\sqrt{2\pi}$。由有限维实谱定理，$A=O^TDO$，其中 $O$ 正交、$D=\operatorname{diag}(\lambda_1,\ldots,\lambda_n)$ 且 $\lambda_i>0$。令 $y=D^{1/2}Ox$；Jacobi 行列式为 $(\det A)^{1/2}$，于是
$$
\int e^{-x^TAx/2}\,d^nx
=(\det A)^{-1/2}\prod_{i=1}^n
\int_{\mathbb R}e^{-y_i^2/2}\,dy_i
=(2\pi)^{n/2}(\det A)^{-1/2}.
$$
最后配方
$$
-\frac12x^TAx+J^Tx
=-\frac12(x-A^{-1}J)^TA(x-A^{-1}J)
+\frac12J^TA^{-1}J
$$
并作平移 $x\mapsto x+A^{-1}J$。Lebesgue 测度的平移不变性与前式给出结论。$\square$

**定义 8.1.** 令 $C=A^{-1}$，以
$$
d\gamma_C(x)=
\frac{(\det A)^{1/2}}{(2\pi)^{n/2}}
e^{-x^TAx/2}\,d^nx
$$
表示中心 Gaussian 概率测度，期望记为 $\mathbb E_C$。

**命题 8.2 (`P`, 有限维 Wick 公式).** 对指标 $i_1,\ldots,i_m\in\{1,\ldots,n\}$，若 $m$ 为奇数，则
$\mathbb E_C[x_{i_1}\cdots x_{i_m}]=0$；若 $m=2r$，则
$$
\mathbb E_C[x_{i_1}\cdots x_{i_{2r}}]
=\sum_{\pi\in\mathcal P_2(2r)}
\prod_{\{a,b\}\in\pi}C_{i_ai_b},
$$
其中 $\mathcal P_2(2r)$ 是 $\{1,\ldots,2r\}$ 的全部无序配对。

**证明.** 命题 8.1 给出矩母函数
$$
M(J)=\mathbb E_C[e^{J^Tx}]
=e^{\frac12J^TCJ}.
$$
Gaussian 尾部支配任意多项式乘 $e^{J^Tx}$，所以可在 $J=0$ 邻域逐次把导数移入积分号；因此所求矩等于
$\partial_{i_1}\cdots\partial_{i_m}M(0)$。指数的幂级数为
$$
M(J)=\sum_{q=0}^\infty\frac1{q!2^q}
\left(\sum_{a,b}C_{ab}J_aJ_b\right)^q.
$$
每项次数为偶数，故奇阶导数在零点为零。次数 $2r$ 时只有 $q=r$ 项贡献。展开其中的 $r$ 个二次因子并求导，每一种把 $2r$ 个导数分配成 $r$ 对的方式给出相应协方差乘积；因 $C$ 对称，$r!2^r$ 个有序与定向重复恰被系数 $1/(r!2^r)$ 消去，剩下每个无序配对一次。$\square$

**例 8.3（相关二元 Gaussian 的四阶矩）.** 对 $n=2$，命题 8.2 给出
$$
\mathbb E_C[x_1^2x_2^2]
=C_{11}C_{22}+2C_{12}^2.
$$
三个配对分别是 $(11)(22)$ 以及两个把每个 $x_1$ 与一个 $x_2$ 配对的方式。这个有限维等式正是四点自由场关联函数出现三种 Wick 收缩的原型。

## 8.2 正规化模型与形式路径积分

**定义 8.2.** 有限自由度 Euclidean 正规化模型由 $N<\infty$、坐标 $\phi\in\mathbb R^N$、下有界作用 $S_\Lambda:\mathbb R^N\to\mathbb R$ 和 Lebesgue 积分
$$
Z_\Lambda[J]=\int_{\mathbb R^N}
e^{-S_\Lambda(\phi)+J\cdot\phi}\,d^N\phi
$$
组成。下标 $\Lambda$ 可以表示有限格点、有限体积加 Fourier 模截止或其他使自由度有限的 regulator。只有在该积分绝对收敛时，它才是本章中的真实测度论对象。

连续场论的记号
$$
Z[J]=\int\mathcal D\phi\,
e^{iS[\phi]+i\int J\phi}
\quad\text{或}\quad
Z_E[J]=\int\mathcal D\phi\,
e^{-S_E[\phi]+\int J\phi}
$$
只是形式生成泛函，除非另行给出 Gaussian 测度、格点极限或构造性定义。本书不把 $\mathcal D\phi$ 视为无穷维 Lebesgue 测度，也不从它推出换元或平移不变性。

**命题 8.3 (`S`, 正规化微扰展开).** 在固定有限 regulator 下，设
$S_\Lambda(\phi)=\frac12\phi^TA_\Lambda\phi+gV_\Lambda(\phi)$，其中 $A_\Lambda$ 正定，$V_\Lambda$ 为多项式。形式地按 $g$ 展开到 $R$ 阶，得到
$$
Z_\Lambda[J]
=\sum_{r=0}^{R}\frac{(-g)^r}{r!}
V_\Lambda(\partial_J)^r Z_{0,\Lambda}[J]
+O(g^{R+1}),
$$
每一项由命题 8.2 的有限维 Wick 配对计算。恢复 $\hbar$ 后，连通图的环数给出相应的 $\hbar$ 次数。

**推导说明（标准物理口径）.** 在有限维中，等式来自对 $e^{-gV}$ 作 Taylor 展开，并用 $\phi_i e^{J\cdot\phi}=\partial_{J_i}e^{J\cdot\phi}$。若 $g$ 位于使积分收敛的区间且余项有统一可积控制，这也是普通积分的渐近展开；但场论所需的 $\Lambda\to\infty$、体积极限、级数求和以及 Minkowski 振荡积分都不由逐项 Wick 计算自动保证。因此本命题只作为固定 regulator、固定微扰阶数的计算规则使用。$\square$

## 8.3 重整化条件与 Callan--Symanzik 方程

正规化只让中间表达式有定义；重整化还要规定哪些有限量被称为质量、耦合和场归一化。以四维 massive Euclidean $\phi^4$ 理论为例，在 cutoff $\Lambda$ 下写
$$
S_\Lambda[\phi]=\int d^4x\left[
\frac12(Z_\phi+\delta Z_\Lambda)(\partial\phi)^2
+\frac12(m_R^2+\delta m_\Lambda^2)\phi^2
+\frac{g_R+\delta g_\Lambda}{4!}\phi^4
\right].
$$
在非例外 Euclidean 动量减法点选择尺度 $\mu>0$。一种重整化方案要求 1PI 顶点满足
$$
\left.\frac{d\Gamma_R^{(2)}}{dp^2}\right|_{p^2=\mu^2}=1,
\qquad
\left.\Gamma_R^{(2)}(p)\right|_{p^2=\mu^2}=m_R^2+\mu^2,
$$
以及在 $p_i^2=\mu^2$、$p_i\cdot p_j=-\mu^2/3$（$i\ne j$）的对称四点构型上
$$
\Gamma_R^{(4)}(p_1,p_2,p_3,p_4)=g_R.
$$
这些等式是方案的定义，不是实验定律。换减法点或改用 minimal subtraction 会改变有限部分以及 $m_R,g_R,Z_\phi$ 的数值，但同一精度下的可观测量应在参数匹配后相同。

**定理 8.4 (`E`, massive $\phi^4_4$ 的逐阶微扰重整化).** 对带 ultraviolet regulator 的四维 massive Euclidean $\phi^4$ 理论，在任意固定微扰阶数，可选择局部 counterterms $\delta Z_\Lambda$、$\delta m_\Lambda^2$、$\delta g_\Lambda$，使所有带正外腿数的重整化 1PI 振幅在满足上述非例外动量减法条件后，于移除 ultraviolet regulator 时逐阶有限。若还要重整化真空泛函，则须另加真空能 counterterm。

**证明路线（外部输入）.** BPHZ 森林公式先对每个发散子图减去其外动量 Taylor 多项式；幂计数表明所需局部单项式只有 $(\partial\phi)^2$、$\phi^2$ 和 $\phi^4$。重叠发散的组合消去与 regulator 移除需要完整的收敛定理，本书不重证。精确版本和章节定位见 [SOURCES.md](SOURCES.md) 的 `E-8.4`。该定理是关于形式微扰级数逐阶系数的陈述，不证明四维连续相互作用测度的非微扰存在性。

**命题 8.5 (`S`, Callan--Symanzik 方程).** 设 $r^a$ 表示所选方案中的全部重整化参数（包括需要时的质量参数），并定义
$$
\beta^a(r)=\left.\mu\frac{dr^a}{d\mu}\right|_{\rm bare},
\qquad
\gamma_\phi(r)=
\left.\frac12\mu\frac{d\log Z_\phi}{d\mu}\right|_{\rm bare}.
$$
若裸场与重整化场满足 $\phi_0=Z_\phi^{1/2}\phi_R$，则无复合算符插入的重整化 $n$ 点函数形式上满足
$$
\boxed{
\left(
\mu\frac{\partial}{\partial\mu}
+\beta^a(r)\frac{\partial}{\partial r^a}
+n\gamma_\phi(r)
\right)G_R^{(n)}=0.}
$$

**推导说明（标准物理口径）.** 裸关联函数与重整化关联函数的关系为
$G_0^{(n)}=Z_\phi^{n/2}G_R^{(n)}$。固定裸参数对 $\mu$ 求导，左端为零；对右端使用乘积与链式法则，除以 $Z_\phi^{n/2}$，便得到框中方程。这里三个算子之间的加号不可省略。若使用 1PI 顶点、不同的 $Z_\phi$ 定义或把质量的经典维数从 $\beta^a$ 中拆出，$n\gamma_\phi$ 的符号和质量项会按约定改写；复合算符插入还会出现异常维数矩阵。该方程只在已选 regulator、重整化方案和给定微扰阶数内使用。$\square$

## 8.4 Wilson 有效作用与能标截断

**定义 8.3.** 在有限维正规化模型中若自由度正交分解为
$\mathbb R^N=V_<\oplus V_>$，则 Wilson 有效作用可由普通积分严格定义：
$$
e^{-S_{{\rm eff},\Lambda'}(\phi_<)}
=\int_{V_>}e^{-S_\Lambda(\phi_<+\phi_>)}\,d\phi_>.
$$
连续场论中把 $V_>$ 解释为 $\Lambda'<|k|<\Lambda$ 的模并写 $\mathcal D\phi_>$，仍是依赖 regulator 的形式记号；真实极限必须逐模型证明。

设轻场外部能量满足 $E\ll M$，其中 $M$ 是已积分掉粒子的质量或新物理尺度。在 $d$ 维时空中，有效 Lagrangian 按局部算符维数组织为
$$
\mathcal L_{\rm EFT}
=\mathcal L_{\rm light}^{(\le d)}
+\sum_i c_i(\mu)M^{d-\Delta_i}\mathcal O_i(\mu),
\qquad \Delta_i>d,
$$
并只允许轻场理论的对称性相容的算符。匹配通常在 $\mu\simeq M$ 进行，再用 RG 演化到 $\mu\simeq E$ 以控制 $\log(M/E)$。

**命题 8.6 (`S`, EFT 局部展开).** 若被积掉的自由度有质量隙 $M>0$，外动量满足 $|p_i|\le E<M$，且阈值以下的重振幅在外动量附近解析，则其重场贡献可按 $p/M$ 展开为局部算符。若幂计数规定保留到相对阶 $(E/M)^N$，遗漏项为 $O((E/M)^{N+1})$，允许乘微扰耦合与对数；轻的无质量传播子造成的非解析项必须保留在 EFT 环图中，不能吸收到局部 Wilson 系数。

**推导说明（标准物理口径）.** 有质量传播子
$(M^2+p^2)^{-1}$ 在 $|p|<M$ 内有收敛几何级数。一般重子图在无阈值穿越且 regulator 保持局部性的条件下，可对外动量作 Taylor 展开；动量多项式 Fourier 变换为局部导数算符。复杂理论中的一致幂计数、算符混合与 decoupling 定理依赖具体模型，本命题只在所列解析性和能标条件下使用。$\square$

**例 8.7（树级积分掉重标量）.** 在 Euclidean 有限体积并加模 cutoff，令
$$
S[\phi,\chi]=S_{\rm light}[\phi]
+\frac12\langle\chi,(-\partial^2+M^2)\chi\rangle
+\kappa\langle\chi,\phi^2\rangle.
$$
这是有限维 Gaussian。配方并积分 $\chi$ 后，除去与 $\phi$ 无关的行列式，精确得到
$$
S_{\rm eff}[\phi]=S_{\rm light}[\phi]
-\frac{\kappa^2}{2}
\langle\phi^2,(-\partial^2+M^2)^{-1}\phi^2\rangle.
$$
若所有外动量满足 $p^2\le E^2<M^2$，则
$$
\frac1{M^2+p^2}
=\frac1{M^2}-\frac{p^2}{M^4}+R_1(p),
\qquad
|R_1(p)|\le
\frac{E^4}{M^6(1-E^2/M^2)}.
$$
回到坐标空间并分部积分，
$$
S_{\rm eff}=S_{\rm light}
-\frac{\kappa^2}{2M^2}\int\phi^4
+\frac{\kappa^2}{2M^4}\int[\partial_\mu(\phi^2)]^2
+O\!\left(\frac{\kappa^2E^4}{M^6}\right).
$$
这个例子同时给出自由度、匹配尺度、局部算符、截断阶和显式余项；没有把连续极限中的形式测度当成证明的一部分。

## 8.5 规范固定

**命题 8.7 (`S`, Faddeev--Popov 形式).** 若规范条件 $G(A)=0$ 在某个场构型邻域内横截规范轨道，且线性化算符
$$
M_A=\left.\frac{\delta G(A^g)}{\delta\alpha}\right|_{\alpha=0}
$$
在所选函数空间和边界条件下可逆，则形式恒等式
$$
1=\Delta_{\rm FP}[A]
\int\mathcal Dg\,\delta(G(A^g)),
\qquad
\Delta_{\rm FP}[A]=\det M_A,
$$
把局部规范群体积转化为 ghost 行列式。

**推导说明（标准物理口径）.** 在有限维自由群作用与局部切片中，这是普通换元公式的 Jacobi 行列式。场论表达把它类比到无穷维场空间，并用 Grassmann ghost 表示行列式。该步骤假设局部切片存在、零模已处理且 regulator 与规范对称相容；Gribov copies、异常、行列式相位和测度构造都可能破坏朴素公式，所以这里只作为局部微扰规则使用。$\square$

## 练习

**练习 8.1.** 用命题 8.1 对 $J$ 求导，证明 $\mathbb E_C[x_ix_j]=C_{ij}$。

**练习 8.2.** 列出 $\mathbb E_C[x_1x_2x_3x_4]$ 的全部 Wick 配对，并在 $C=I$ 时化简。

**练习 8.3.** 在例子 8.7 中把传播子展开到 $M^{-6}$，写出下一个局部算符，并给出相对阶数。

**练习 8.4.** 从 $G_0^{(n)}=Z_\phi^{n/2}G_R^{(n)}$ 逐步推出命题 8.5，并解释若定义 $\widetilde\gamma_\phi=-\gamma_\phi$ 时公式如何变化。
