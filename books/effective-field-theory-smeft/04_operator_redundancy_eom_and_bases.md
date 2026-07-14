# 第四章：算符冗余、EOM 与算符基

对称性允许的局域单项式远多于独立的散射信息。比如 $\phi^2(\partial\phi)^2$ 与 $\phi^3\Box\phi$ 可由分部积分联系，后者又能借助领先运动方程换成势能项；若把三者当成独立参数，匹配矩阵会人为出现冗余方向。真正的算符基因而不是一张“看起来不同”的单项式清单，而是局域作用量在总导数、逐阶场重定义和张量恒等式下的商空间坐标。本章从标量例子出发，先区分逐点等式、作用量等价与 on-shell S-matrix 等价，再处理二阶场重定义、Hermitian conjugation 和 flavor 类型。圈级计算还迫使我们暂时扩大到 EOM、BRST-exact 与 evanescent 方向，完成重整化后才能投影回物理基。

## 4.1 分部积分冗余

**定义 4.1（总导数等价）.** 固定时空 $M$、允许的场构型、拓扑 sector 和边界条件。若两个拉氏量密度相差一个在该 sector 全局定义的局域流的散度
$$
\mathcal L_1-\mathcal L_2=\partial_\mu K^\mu,
$$
且 $\int_{\partial M}d\Sigma_\mu K^\mu=0$，则称它们在该作用量问题中等价。这里的等价是积分泛函的等价，不是两个局域密度逐点相等。

**命题 4.2（分部积分删除导数分配冗余）.** 在平直时空的固定微扰真空 sector，若 $K^\mu$ 全局定义且散射边界条件使其边界积分为零，则上述总导数项不改变作用量，因而不改变由该作用量计算的 S-matrix。

**证明（书内推导）.** 作用量差为
$$
\Delta S=\int d^4x\,\partial_\mu K^\mu
=\int_{\partial M} d\Sigma_\mu K^\mu.
$$
若场在边界充分衰减或采用相容的周期边界条件，则边界积分为零，所以两个作用量相同。其微扰路径积分权重和由 LSZ 得到的 S-matrix 因而相同。$\square$

**外部输入反例 4.2A（局部总导数不保证全局可删，EFT-TOPOLOGY）.** 非阿贝尔规范理论中 $\operatorname{tr}(F\wedge F)$ 在局部可写成 Chern--Simons 3-form 的外微分，但该 3-form 在非平凡规范丛上未必全局定义，且 $\int\operatorname{tr}(F\wedge F)$ 可测量拓扑数。因此 $\theta F\widetilde F$ 不能仅凭“局部是总导数”从包含非平凡拓扑 sector 的量子理论删除。命题 4.2 的全局性和边界假设不可省略；来源边界见附录 B。

## 4.2 EOM 冗余

**定义 4.3（给定截断阶的 EOM 冗余）.** 设 $S_0[\phi]$ 是所选幂计数的领先、规范不变作用量。使用 DeWitt 求和约定，若一个积分局域泛函可写为
$$
\mathscr O_F
=F^a[\phi]\frac{\delta S_0}{\delta\phi^a}
$$
再加一个满足定义 4.1 的总导数，则称它在该阶相对于 $S_0$ 是 EOM 冗余的。$F^a$ 必须是局域的，并与 $\phi^a$ 的 Lorentz、规范和 Grassmann 类型相容。改变 $S_0$ 或改变截断阶会改变这条冗余关系。

**命题 4.4（局域场重定义删除 EOM 算符）.** 设 $F[\phi]$ 是局域泛函，并且变换在所考虑的场构型邻域中可扰动求逆。在一阶微扰中，场重定义
$$
\phi\mapsto \phi+\epsilon F[\phi]
$$
使领先作用量变化为
$$
S_0[\phi+\epsilon F]
=
S_0[\phi]
+
\epsilon\int d^4x\,F[\phi]\frac{\delta S_0}{\delta\phi}
+
O(\epsilon^2).
$$
因此 EOM 冗余算符可在固定截断阶内被场重定义移入其他算符系数；该变换同时会在 $O(\epsilon^2)$ 及更高阶生成新算符。

**证明（书内推导）.** 对泛函 $S_0$ 作一阶 Taylor 展开：
$$
S_0[\phi+\epsilon F]-S_0[\phi]
=\epsilon\int d^4x\,
\frac{\delta S_0}{\delta\phi(x)}F[\phi](x)+O(\epsilon^2).
$$
若原 EFT 含 $-\epsilon\int F\,\delta S_0/\delta\phi$，则该项在新变量中与上述一阶变化抵消；剩余变化只重新分配同阶其他算符并产生更高阶项。量子 S-matrix 不变性使用第二章外部输入定理 2.7A。$\square$

**警告 4.5.** EOM 删除不是说算符在所有场构型上为零，而是说它对 on-shell 物理可观测量可由场重定义吸收。off-shell Green 函数和中间 Wilson 系数会改变。

**命题 4.5A（场重定义的二阶项）.** 设
$$
S[\phi]=S_0[\phi]+\epsilon S_1[\phi]+\epsilon^2S_2[\phi]+O(\epsilon^3),
$$
并作局域扰动可逆变换，其中 $F,G$ 与 $\epsilon$ 无关，
$$
\phi^a\mapsto\phi^a+\epsilon F^a[\phi]+\epsilon^2G^a[\phi].
$$
则变换后的作用量到二阶为
$$
\begin{aligned}
S'[\phi]
={}&S_0
+\epsilon\left(S_1+S_{0,a}F^a\right)\\
&+\epsilon^2\left(
S_2+S_{1,a}F^a
+\frac12F^aS_{0,ab}F^b
+S_{0,a}G^a
\right)
+O(\epsilon^3),
\end{aligned}
$$
其中 $S_{r,a}=\delta S_r/\delta\phi^a$，重复指标包含时空积分；显示公式采用 bosonic 记号，含费米场时改用固定的左/右泛函导数和 graded 符号。因此用领先 EOM 删除 $S_1$ 的一个方向只在 $O(\epsilon)$ 完成；到 $O(\epsilon^2)$ 必须保留 $S_{1,a}F^a+\tfrac12F^aS_{0,ab}F^b$，并再用 $G^a$ 处理其中新的 EOM 方向。

**证明（书内推导）.** 对 $S_0[\phi+\epsilon F+\epsilon^2G]$ 作二阶泛函 Taylor 展开，对 $\epsilon S_1[\phi+\epsilon F+\epsilon^2G]$ 作一阶展开，并保留 $\epsilon^2S_2[\phi]$ 的零阶项。按 $\epsilon$ 次数收集即得显示公式。$\square$

**警告 4.5B（平方项与换基）.** 若在 $p=2$ 用 EOM 删除一个维数六算符，却在可观测量中保留该振幅的平方或其他 $p=4$ 项，就必须同时变换命题 4.5A 生成的 $p=4$ 作用量。只变换线性 Wilson 系数而保留原平方项，不是同一理论在两个基中的比较。该高阶边界使用外部输入 EFT-EQ-HO 的完整量子讨论，见附录 B。

**例 4.6（标量 EOM 冗余）.** 设领先拉氏量为
$$
{\cal L}_0={1\over2}(\partial\phi)^2-{1\over2}m^2\phi^2-{\lambda\over4!}\phi^4.
$$
其 EOM 为
$$
\Box\phi+m^2\phi+{\lambda\over6}\phi^3=0.
$$
考虑维数六形式的算符
$$
{\cal O}_{\rm EOM}=\phi^3\Box\phi.
$$
利用 EOM 得
$$
\phi^3\Box\phi
=-m^2\phi^4-{\lambda\over6}\phi^6.
$$
这里的等式只表示 $O(\epsilon)$ 作用量商中的关系，不是 off-shell 场构型上的恒等式。含 $\phi^3\Box\phi$ 的 EFT 可在该阶通过场重定义改写为 $\phi^4$ 和 $\phi^6$ 系数的移动；$m^2\phi^4$ 项还会诱导低维参数的 $m^2/\Lambda_{\rm ref}^2$ 位移。若继续到 $O(\epsilon^2)$，必须使用命题 4.5A 的诱导项。

**例 4.7（分部积分与 EOM 的组合）.**
$$
\partial_\mu(\phi^3\partial^\mu\phi)
=3\phi^2(\partial\phi)^2+\phi^3\Box\phi.
$$
故
$$
\phi^2(\partial\phi)^2
\simeq
-{1\over3}\phi^3\Box\phi,
$$
其中 $\simeq$ 表示相差总导数。再用 EOM 可把它换成势能型算符。这说明算符冗余常需同时使用分部积分和 EOM。

## 4.3 算符基

**定义 4.8（算符商空间与算符基）.** 固定时空维数、场及其表示、精确对称性、flavor 口径、作用量阶 $p$ 和领先作用量 $S_0$。对每个 field-monomial canonical dimension $d_{\rm op}$，令 $V_{d_{\rm op}}^{\rm op}$ 为只由场与协变导数构成的规范不变局域多项式积分泛函的复向量空间；算符维数按 Warsaw convention 不包含质量参数。再令 $\mathbb K_{\rm SM}$ 为由 dimensionless SM couplings 和非负次质量 spurions 生成的 graded 系数环，其中质量 spurion 的 degree 等于其质量维数，并定义
$$
\mathscr V
\coloneqq
\mathbb K_{\rm SM}\otimes_{\mathbb C}
\bigoplus_{d_{\rm op}}V_{d_{\rm op}}^{\rm op}.
$$
张量的总质量分次是 spurion degree 与 $d_{\rm op}$ 之和。于是 $\phi^3\Box\phi\in V_6^{\rm op}$，而 $m^2\phi^4\in\mathbb K_{\rm SM}^{(2)}\otimes V_4^{\rm op}$；二者都属于 $\mathscr V$ 的总分次六，但不被误称为同一 field-monomial dimension。固定总分次 $D$ 与作用量阶 $p$ 后，令 $\mathscr R_{D,p}\subset\mathscr V_{D,p}$ 为下列关系生成的子空间：

1.  满足定义 4.1 假设的总导数；
2.  由 $S_0$ 产生并只在该作用量阶使用的 EOM 方向；
3.  Bianchi、Schouten、Fierz 及其他代数张量恒等式。

EFT 算符空间是积分局域泛函的商
$$
\mathscr V_{\mathrm{phys}}^{D,p}
=\mathscr V_{D,p}/\mathscr R_{D,p}.
$$
在固定 SM 参数值后，一组算符代表若其等价类构成该复商空间的基，则称为该阶、该 sector 的算符基；Wilson 系数属于其复对偶，换基规律由命题 2.7 给出。Hermitian conjugation 在 $\mathscr V_{D,p}$ 上给出反线性 involution：自伴算符的系数为实数，非自伴代表必须以 $C\mathcal O+C^*\mathcal O^\dagger$ 出现。因而“复基的维数”和“Hermitian 拉氏量的实参数数目”是两个需分别声明的计数。Warsaw 列表只列新的 $d_{\rm op}=6$ field monomials；EOM 产生的 $m^2V_4^{\rm op}$ 方向吸收到受抑制的 SM 参数位移中，这一 bookkeeping 必须与输入参数方案一起保留。

**例 4.9（基与破缺相参数化不是同一类型）.** Warsaw basis 与一个在相同场内容、对称性、flavor 和截断假设下完整约化的 SILH-like 集合，可以作为同一未破缺相商空间的不同基。通常所谓 Higgs basis 则是电弱破缺、场重归一化和输入参数选择后的耦合/伪可观测量参数化；除非另行证明完备性与可逆性，它不自动是同一个 $\mathscr V_{\mathrm{phys}}^{D,p}$ 的线性算符基。二者转换必须记录输入方案、规范关系和高阶截断。

**命题 4.9A（重整化下降到商空间的判据）.** 设 $Z:\mathscr V^{\rm ext}\to\mathscr V^{\rm ext}$ 是在包含计算所需冗余结构的扩大空间上的线性重整化映射，$R\subset\mathscr V^{\rm ext}$ 是待 quotient 的冗余子空间。公式
$$
\overline Z([\mathcal O])\coloneqq[Z\mathcal O]
$$
在 $\mathscr V^{\rm ext}/R$ 上良定义，当且仅当
$$
Z(R)\subseteq R.
$$

**证明.** 若 $\mathcal O'\sim\mathcal O$，则 $\mathcal O'=\mathcal O+r$，其中 $r\in R$。映射在商上良定义恰好要求 $[Z\mathcal O']=[Z\mathcal O]$，即 $[Zr]=0$。这对所有 $r\in R$ 成立当且仅当 $Z(R)\subseteq R$。$\square$

**外部输入 4.9B（SMEFT 线性维数六的 quotient 闭合，SMEFT-EOM-RG）.** 在维数正规化与 $\overline{\mathrm{MS}}$ 的线性维数六计算中，EOM 算符子空间在算符 RGE 下保持不变，因此物理 Wilson 系数的 RGE 可下降到 EOM 商；但 EOM 算符仍会出现在中间 counterterm 中，并可在投影时产生物理算符 mixing。Jenkins--Manohar--Trott I 的 Sec. 3、Eqs. (3.8)--(3.11) 是本书使用的精确版本。该输入不自动覆盖双插入、维数八或不同 evanescent projection scheme。

## 4.4 其他冗余关系

**Bianchi 恒等式。** 对非阿贝尔场强，
$$
D_{[\mu}X_{\nu\rho]}=0.
$$
含 $D_\mu\widetilde X^{\mu\nu}$ 的算符可因此与其他导数结构相关。

**Fierz/Schouten 恒等式。** 四费米子算符存在 Lorentz 和内部指标重排恒等式。对二维 Weyl spinor 指标，基本的无约定歧义张量恒等式是
$$
\epsilon_{\alpha\beta}\epsilon_{\gamma\delta}
+\epsilon_{\alpha\gamma}\epsilon_{\delta\beta}
+\epsilon_{\alpha\delta}\epsilon_{\beta\gamma}=0.
$$
把 Grassmann-valued spinors 代入后还需按固定场顺序交换，额外负号取决于所选 contraction 和排序约定。Warsaw basis 的四费米子独立性依赖这类恒等式，不能在未声明约定时只抄写一个符号公式。

**警告 4.10（四维 Fierz 与 $d_{\rm DR}$ 维重整化）.** 四维 Fierz/Schouten 恒等式在 $d_{\rm DR}=4-2\epsilon$ 中一般不完整，二者之差定义 evanescent 方向。故四维商 $\mathscr V_{\rm phys}^{D,p}$ 适合陈述最终基，但 loop renormalization 必须先在扩大空间中闭合，再按给定有限方案投影；否则 $1/\epsilon$ pole 乘 $O(\epsilon)$ evanescent 差异会漏掉有限项。

**原则 4.11（基选择与圈级投影次序）.** 构造或重整化算符基时依次执行：

1.  写出所有满足规范对称性和 Lorentz 对称性的局域结构；
2.  固定总导数的边界与拓扑 sector；
3.  用 Bianchi 等代数恒等式化简，并在 loop 计算中保留 evanescent 补空间；
4.  从规范不变 $S_0$ 写出协变领先 EOM，标出只在何种 $p$ 阶可用；
5.  在扩大空间完成 matching/renormalization，确认命题 4.9A 的不变子空间条件；
6.  再投影 EOM、BRST-exact、IBP 和四维 Fierz 冗余；
7.  最后检查 Hermitian conjugation、flavor 交换对称和 Wilson 实参数计数。

## 4.5 规范协变、BRST 与 EOM 投影

**约定 4.12（规范不变基的 EOM）.** 构造 Warsaw 等规范不变基时，EOM 来自规范固定前的 $S_0$，并以协变形式书写。允许的场重定义须把每个物质场映到同一规范表示，并把规范连接的修正写成相应协变类型。直接使用某个 $R_\xi$ gauge 的 gauge-fixed EOM 来定义规范不变算符商，会把物理冗余与 gauge-fixing 伪影混合。

**外部输入边界 4.13（BRST 口径，EFT-REN）.** Gauge-fixed off-shell Green 函数的重整化在包含 ghost、gauge-fixing、EOM 和 BRST-exact 结构的扩大空间中进行；物理 gauge-invariant 插入由相应 BRST cohomology 类表示。本书不重建 algebraic renormalization，只使用其后果：完整 on-shell 振幅与最终物理商空间预测不依赖 gauge parameter，而中间 off-shell Green 函数、冗余系数和有限投影可以依赖它。该边界与第二章警告 2.7E、第三章命题 3.7 的假设共同使用。

## 4.6 从单项式到物理坐标

分部积分、EOM 和张量恒等式删除的是作用量或 S-matrix 中的重复描述，不是把某个局域密度逐点设为零。标量例子还显示，一阶场重定义会在下一逆尺度阶生成新项，所以换基必须与所报告的平方项和多次插入同步。圈级重整化先在含 EOM、BRST-exact 与 evanescent 方向的扩大空间中进行；只有冗余子空间在重整化下保持不变，映射才下降到物理商。由此得到的算符基只是 Wilson 对偶空间的坐标，真正不变的是完整振幅与可观测量。

## 练习

**练习 4.1.** 用分部积分证明 $\phi^2\Box\phi^2$ 与 $(\partial_\mu\phi^2)(\partial^\mu\phi^2)$ 只差总导数和符号。

**练习 4.2.** 举例说明 off-shell Green 函数会依赖 EOM 冗余算符选择。

**练习 4.3.** 对例 4.6，写出场重定义 $\phi\mapsto \phi+\epsilon\phi^3$ 对 ${\cal L}_0$ 的一阶影响。

**练习 4.4.** 对命题 4.5A 取 $G=0$，写出 $F=\phi^3$ 时由 $\tfrac12F S_{0,\phi\phi}F$ 产生的局域结构，并说明其逆尺度次数。

**练习 4.5.** 给定 $V=\operatorname{span}\{Q_1,Q_2,E\}$ 与 $R=\operatorname{span}\{E\}$，构造一个满足 $Z(R)\subset R$ 的矩阵和一个不满足该条件的矩阵，并直接检查哪一个能定义 $V/R$ 上的映射。
