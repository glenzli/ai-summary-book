# 术语表

作者：Dr. Stochastic Parrot

本表只记录本书中反复使用、容易因约定不同而混淆的术语。符号约定以 [NOTATION.md](NOTATION.md) 为准。

## A

**ADM 分解**：把一个四维时空按空间超曲面族分解为 lapse、shift 和三维度量的形式。它是初值问题和 Hamilton 形式的基本语言。

**ADM 质量**：渐近平直初值数据在无穷远处定义的总能量。严格定义依赖衰减条件，本书只使用其标准边界积分形式和正质量定理作为外部输入。

## B

**Bianchi 恒等式**：曲率张量满足的微分恒等式。缩并 Bianchi 恒等式给出 $\nabla^\mu G_{\mu\nu}=0$，从而解释 Einstein 方程左端的自动守恒性。

**boost**：沿某一空间方向的 Lorentz 变换。它混合时间坐标与该方向空间坐标，并保持 Minkowski 度量不变。

## C

**Carter 常数**：Kerr 测地线中的额外守恒量，来自隐藏对称性或二阶 Killing 张量。它使 Kerr 测地线方程可分离。

**Cauchy 超曲面**：每条不可延伸因果曲线恰好穿过一次的类空超曲面。具有 Cauchy 超曲面的时空称为全局双曲时空。

**Christoffel 符号**：坐标基下 Levi-Civita 联络的系数 $\Gamma^\rho_{\mu\nu}$。它不是张量，但其组合可形成曲率张量。

**cosmological constant**：宇宙学常数 $\Lambda$。Einstein 方程写作
$$
G_{\mu\nu}+\Lambda g_{\mu\nu}=8\pi G T_{\mu\nu}.
$$

## E

**Einstein 张量**：$G_{\mu\nu}=R_{\mu\nu}-\frac12 Rg_{\mu\nu}$。它对称并满足 $\nabla^\mu G_{\mu\nu}=0$。

**Einstein 方程**：把几何曲率与物质能动张量联系起来的场方程。其真空形式为 $R_{\mu\nu}=\Lambda g_{\mu\nu}$；当 $\Lambda=0$ 时为 $R_{\mu\nu}=0$。

**EOB**：有效一体问题。它把两体相对论动力学重写为一个有效粒子在变形背景中的运动，并结合后 Newton、黑洞微扰和数值相对论波形。

**ergoregion**：能层。Kerr 黑洞外视界和静止极限面之间的区域，其中定常 Killing 向量不再类时，可出现负 Killing 能量轨道。

**等效原理**：在足够小的邻域内，引力效应可以通过选择局部惯性系被一阶消去；潮汐效应由曲率刻画，不能一般消去。

## F

**FLRW 度量**：满足空间齐性和各向同性假设的宇宙学度量。其动力学由 Friedmann 方程控制。

**Friedmann 方程**：在 FLRW 度量和理想流体能动张量下由 Einstein 方程推出的宇宙尺度因子方程。

## G

**geodesic**：测地线。对 Levi-Civita 联络而言，它在仿射参数下满足
$\nabla_{\dot\gamma}\dot\gamma=0$。它是长度泛函的驻定曲线；只有足够
短、未越过共轭点的类时段才有局部固有时极大性。

**global hyperbolicity**：全局双曲性。它保证因果结构良好，并是 Einstein 方程初值问题中常用的时空条件。

## H

**harmonic gauge**：调和规范，也称 de Donder 规范。在线性化引力中常写作 $\partial^\mu \bar h_{\mu\nu}=0$。

**horizon**：视界。对最大延拓的正质量 Schwarzschild 时空和选定的
渐近平直端，未来事件视界的面积半径为 $r=2GM$；它不是曲率奇点。
事件视界由全局因果结构定义，不是“坐标系的边界”。

## K

**Kerr 度规**：描述四维真空、定常、轴对称旋转黑洞的精确解。第十五章
取 $G=c=1$，用几何化质量 $M=GM_{\rm phys}/c^2$ 和
$a=J_{\rm phys}/(M_{\rm phys}c)$ 参数化；此时几何化角动量满足
$J=Ma$。

**Killing 向量场**：满足 $\mathcal L_X g=0$ 的向量场。它表示时空对称性，并沿测地线给出守恒量。

## L

**Levi-Civita 联络**：唯一同时满足无挠性和度量相容性的联络。其存在唯一性是 Riemann/Lorentz 几何的基础命题。

**Lorentz 变换**：保持 Minkowski 度量的线性变换，即 $\Lambda^T\eta\Lambda=\eta$。

**Lorentzian manifold**：Lorentz 流形，带有符号为 $(-,+,+,+)$ 的非退化度量的光滑流形。

## M

**Minkowski 时空**：带平直 Lorentz 度量的四维仿射空间，是狭义相对论的几何背景。

**mass shell**：质壳关系。对质量 $m$ 的自由粒子，四动量满足 $p_\mu p^\mu=-m^2$。

## N

**Noether 定理**：连续对称性给出守恒流或守恒量的定理。本书使用其有限维和场论中的基本形式。

## P

**PPN**：参数化后 Newton 形式。它用一组参数描述弱场慢速引力理论的可观测偏离，便于比较广义相对论和其他引力理论。

**proper time**：固有时，类时世界线上由度量诱导的时间
$$
d\tau^2=-ds^2.
$$

**post-Newtonian approximation**：后牛顿近似。按 $v/c$ 和 $GM/(rc^2)$ 展开广义相对论效应的系统方法。本书只使用最低阶解释，不展开完整高阶形式。

## R

**Ricci 张量**：曲率张量的缩并 $R_{\mu\nu}=R^\rho{}_{\mu\rho\nu}$。

**Riemann 曲率张量**：联络二阶不可交换性的张量化度量。本书采用
$$
R^\rho{}_{\sigma\mu\nu}X^\sigma=(\nabla_\mu\nabla_\nu-\nabla_\nu\nabla_\mu)X^\rho
$$
的约定。

## S

**Schwarzschild 半径**：在 $c=1$ 且保留 $G$ 时
$r_s=2GM$，恢复单位为 $r_s=2GM/c^2$。在最大延拓的正质量
Schwarzschild 时空中，它是相对于所选渐近平直端的未来事件视界面积
半径。

**stress-energy tensor**：应力能量张量 $T_{\mu\nu}$。它描述能量密度、动量密度、能流与应力，并作为 Einstein 方程的源。

## T

**tensor**：张量。它是多重线性对象，其坐标分量按协变/逆变规则变换。指标记号只是表示方式，不是定义本身。

**TT 规范**：横迹规范。在线性化引力波中，物理自由度可写为横向且无迹的空间扰动。

## W

**worldline**：世界线。粒子在时空中的曲线；类时世界线可用固有时参数化。

**world model**：本书不用这个术语描述相对论。相对论中的“模型”通常指满足场方程和边界/初值条件的几何-物质系统。
