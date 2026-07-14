# 第五章：路径积分、ghost 和 BRST 量子化

Polyakov 路径积分表面上是对 $X$ 与世界面 metric 的积分，实际上却把整条
$\operatorname{Diff}(\Sigma)\times\operatorname{Weyl}(\Sigma)$ gauge orbit 重复
计算了无穷多次。选定 conformal gauge 后，这个冗余不会凭空消失：Faddeev--Popov
Jacobian 变成 $bc$ ghost CFT，metric 的剩余变形变成 moduli，而 gauge 等价关系由
一个 nilpotent BRST charge 编码。以下从第三章的 OPE 与第四章的 Virasoro constraints
出发，逐步构造 ghost action、BRST complex 和零模选择规则，并说明总 central charge
为何必须消失。所用 ghost number、正规序与 $Q_B$ 归一化固定在
[全书归一化表](NORMALIZATION_TABLE.md) 中。

## 5.1 Polyakov path integral

**定义 5.1（形式 Polyakov path integral）.** Euclidean Polyakov path integral 写为
$$
Z=\int\frac{\mathcal D h\,\mathcal D X}
{\operatorname{Diff}(\Sigma)\times\operatorname{Weyl}(\Sigma)}
e^{-S_E[X,h]}.
$$

**注 5.2（形式性）.** 该表达式是物理定义式。严格定义需要选择 regulator、处理 gauge group 体积、处理 Riemann surface moduli 和 operator insertions。本书使用其标准扰动论接口。

更具体地说，本章同时使用三种不同层级：局部 CFT 的 OPE 采用 point splitting
normal ordering；Gaussian determinant 采用 heat-kernel/zeta determinant 的标准物理
方案；moduli space 上的积分只按 genus 逐阶定义，并在退化边界另行给出
$i\varepsilon$、解析延拓或红外 cutoff。三者不得互相替代。

**定义 5.3（genus expansion）.** 常 dilaton 背景下，闭弦 genus $g$ 世界面权重为
$$
g_s^{2g-2}.
$$
带 $n$ 个闭弦外态的振幅常整体写成 $g_s^{2g-2+n}$，其中外态归一化可吸收部分 $g_s$ convention。

## 5.2 Gauge fixing、ghost action 和 central charge

**定义 5.4（metric fluctuation 分解）.** 在 conformal gauge 附近，metric fluctuation 可分解为
$$
\delta h_{ab}
=(\mathcal L_v h)_{ab}+2\omega h_{ab}
+\sum_I \delta m^I\,\mu_{I,ab},
$$
其中前三项分别是 diffeomorphism、Weyl 和 moduli 方向。

令 traceless diffeomorphism operator 为
$$
(P_1v)_{ab}=\nabla_av_b+\nabla_bv_a-h_{ab}\nabla_cv^c.
$$
Conformal Killing vectors 属于 $\ker P_1$，而与 $\operatorname{im}P_1$ 正交的
traceless tensors 给出 moduli/cokernel 方向。

**定义 5.5A（ghost action）.** Faddeev-Popov determinant 可由 anticommuting ghosts $b_{ab},c^a$ 表示。复坐标中其 action 为
$$
S_{bc}=\frac1{2\pi}\int d^2z\,
\left(b_{zz}\bar\partial c^z+b_{\bar z\bar z}\partial c^{\bar z}\right).
$$
Holomorphic fields 记为 $b(z),c(z)$，权重分别为 $2,-1$。

**推导说明 5.5B（Faddeev--Popov 输入边界）.** 在选定 gauge slice 和
$L^2$ pairing 后，线性化 Jacobian 的非零模部分为 $\det'P_1$。Grassmann Gaussian
恒等式把选定 regulator 下的 determinant 表示为
$$
\det'P_1\ \widehat=\
\int\mathcal D'b\,\mathcal D'c\,
\exp\left[-\frac1{2\pi}\int b\,P_1c\right].
$$
撇号表示先移除 kernel/cokernel zero modes，再由显式 $c$ 插入和 Beltrami
$b$ 插入处理它们。有限维 Grassmann 恒等式是代数事实；把它推广到无限维并选择
determinant line、相位及 heat-kernel/zeta regulator 是标准路径积分输入，不在本书中
声称为已构造的测度论定理。

**命题 5.5（ghost central charge）.** Reparametrization ghost system 的 central charge 为
$$
c_{bc}=-26.
$$

**证明.** 这是第三章命题 3.11 的 $bc$ system central charge 计算，权重为 $(2,-1)$。$\square$

**推论 5.6（局部 Weyl/BRST 临界条件）.** 在 free-boson matter CFT 与
point-split $bc$ CFT 的 conformal-gauge 微扰量子化中，局部 Weyl anomaly 消失要求
$$
c_{\mathrm{matter}}+c_{\mathrm{ghost}}=D-26=0.
$$
因此临界维数为
$$
D=26.
$$

**推导说明（标准物理口径）.** 正规化后的二维 Ward identity 把局部 trace anomaly
与 total central charge 联系起来；这一 Ward identity 是 worldsheet QFT 输入。
Matter free bosons 的 operator OPE 精确给出 $D$，ghost operator OPE 精确给出
$-26$，相加得到 $D-26$。所以 $D=26$ 是局部 anomaly cancellation 的必要条件；
它不单独保证 modular invariance、tadpole cancellation 或 moduli 边界收敛。$\square$

## 5.3 BRST symmetry

**定义 5.7（BRST current、mode charge 与定义域）.** 在临界玻色弦的局部
operator algebra 中，BRST current 可写为
$$
j_B(z)=c(z)\left(T_m(z)+\frac12T_{bc}(z)\right)+\frac32\partial^2c(z)
$$
在标准玻色弦 convention 下成立，其围道积分为
$$
Q_B^{\mathrm{crit}}=\oint\frac{dz}{2\pi i}\,j_B(z).
$$
为了同时检验一般 intercept，定义正规序 mode operator
$$
Q_B(a)=\sum_n c_{-n}L_n^{m}
-\frac12\sum_{m,n}(m-n):c_{-m}c_{-n}b_{m+n}:
-a c_0.
$$
临界取 $a=1$，并简记为 $Q_B$；闭弦总 charge 为
$Q_B^{\mathrm{cl}}=Q_B+\widetilde Q_B$。定义域取 matter 与标准 ghost module 的
有限 $L_0$-level 直和 $\mathcal D_{\mathrm{BRST}}$。由于 $Q_B$ 保持总 level，
上述正规序和在每个齐次 level 上是有限的，并给出
$\mathcal D_{\mathrm{BRST}}\to\mathcal D_{\mathrm{BRST}}$ 的奇算符。这里不声称其在
Hilbert completion 上已有唯一闭延拓。

**定义 5.8（BRST cohomology）.** 在 $Q_B^2=0$ 后，代数 BRST cohomology 定义为
$$
\mathcal H_{\mathrm{phys}}=H^\bullet(Q_B)
=\frac{\ker Q_B}{\operatorname{im}Q_B}.
$$
即
$$
Q_B|\psi\rangle=0,\qquad
|\psi\rangle\sim|\psi\rangle+Q_B|\chi\rangle.
$$
标准开玻色弦外态取 ghost number $1$。闭弦未积分外态取总 ghost number $2$，并在
semi-relative complex
$$
b_0^-|\psi\rangle=0,
\qquad L_0^-|\psi\rangle=0,
\qquad b_0^-=b_0-\widetilde b_0,
\quad L_0^-=L_0-\widetilde L_0
$$
上取 $Q_B^{\mathrm{cl}}$ cohomology。泛写的 $H^\bullet(Q_B)$ 是分次复形，不应与
特定 ghost number 的 on-shell 外态空间混同。

**命题 5.9（nilpotency 条件）.** 在玻色弦中，$Q_B^2=0$ 要求总 central charge 为零，并固定 normal ordering constant 为临界值。

**推导说明（标准物理口径）.** 使用
$$
\{b_m,c_n\}=\delta_{m+n,0},\qquad
[L_m^m,L_n^m]=(m-n)L_{m+n}^m
+\frac D{12}(m^3-m)\delta_{m+n,0},
$$
将定义 5.7 的 mode 表达式平方。含有非中心 Virasoro 项的部分与 ghost 三次项逐项抵消；剩余项是 ghost number $2$ 的异常，其各 mode 的系数为
$$
A(n)=\frac{D-26}{12}(n^3-n)+2(a-1)n,
\qquad n>0.
$$
换言之，$Q_B^2$ 在 $\mathcal D_{\mathrm{BRST}}$ 上是由
$A(n)c_{-n}c_n$ 组成的逐 level 有限和，整体符号取决于 ghost mode 的排序
convention，但零点条件不受影响。若 $Q_B^2=0$，则 $A(n)=0$ 对所有 $n>0$
成立；三次项和一次项分别要求
$$
D=26,\qquad a=1.
$$
反之，在这两个条件下异常多项式恒为零，故 $Q_B$ 幂零。OPE 语言中的同一计算表现为 $j_B(z)j_B(w)$ 的异常高阶极点消失。该推导以正规序后的 mode algebra 为起点，不声称给出路径积分测度的非微扰构造。$\square$

**命题 5.10（BRST exact state 的 decoupling）.** 若外态之一为 BRST exact，即 $|\psi\rangle=Q_B|\chi\rangle$，则在无 anomaly 且 moduli 边界项受控的振幅中该外态 decouple。

**推导说明（标准物理口径）.** 设其余顶点 $V_i$ 均为 BRST closed。把 exact 插入写为
$$
\{Q_B,W\}=\oint_W\frac{dz}{2\pi i}j_B(z)W.
$$
由第三章的 contour deformation，围绕 $W$ 的 contour 可向外移动。穿过 $V_i$ 时所得 residue 是 $\{Q_B,V_i\}=0$；若世界面没有边界，且模空间积分不存在未控制的边界贡献，contour 最终收缩，振幅为零。高 genus 或退化极限中，移动 contour 还可能产生模空间边界项，因此命题明确把“边界项受控”列为假设；这些项在一致的 on-shell 扰动论中按物理中间态因子化。$\square$

## 5.4 Ghost zero modes 和顶点插入

**定义 5.11（unintegrated 与 integrated insertions）.** 在具有 conformal Killing vectors 的曲面上，需要用 $c$ ghost 插入吸收 ghost zero modes。Sphere 上闭弦 tree amplitude 通常取三个未积分顶点
$$
c\bar c V
$$
和其余积分顶点
$$
\int d^2z\,V.
$$

**命题 5.11A（sphere ghost zero mode counting）.** Riemann sphere 的 holomorphic conformal Killing vectors 维数为 $3$，因此闭弦 sphere amplitude 需要三个 $c$ 与三个 $\bar c$ zero modes。

**证明.** Sphere 的全纯自同构群为 $PSL_2(\mathbb C)$。其 holomorphic vector fields 由
$$
1,\quad z,\quad z^2
$$
张成，对应三个 $c$ ghost zero modes。Antiholomorphic 部分同理。$\square$

## 5.5 Moduli 分解

**外部输入定理 5.12（Polyakov path integral 的 moduli 分解）.** 固定 genus $g$ 的闭弦扰动论振幅可写成 Riemann surface moduli space $\mathcal M_{g,n}$ 上的积分，积分测度由 matter CFT、ghost determinant、vertex operator insertions 和 Beltrami differentials 给出。

**注 5.13.** 该定理依赖 Riemann surface theory、ghost zero modes 和 gauge slice 的选择。第十五章将回到高 genus 扰动论。

## 5.6 Genus-one measure 示例

**定义 5.14（带红外 cutoff 的 torus vacuum expression）.** 令
$$
\mathcal F_T=\{\tau\in\mathcal F:\tau_2\le T\}.
$$
闭弦 genus-one vacuum expression 在 cutoff $T<\infty$ 时写为
$$
\mathcal Z_1(T)
=\int_{\mathcal F_T}\frac{d^2\tau}{2\tau_2^2}\,
Z_X(\tau,\bar\tau)Z_{bc}(\tau,\bar\tau),
$$
其中 $\mathcal F$ 是 $SL(2,\mathbb Z)$ fundamental domain。对 $D$ 个非紧 free bosons，
$$
Z_X=V_D(4\pi^2\alpha'\tau_2)^{-D/2}|\eta(\tau)|^{-2D}.
$$
Ghost contribution 为
$$
Z_{bc}=\tau_2|\eta(\tau)|^4
$$
在常用 zero-mode convention 下成立。

**命题 5.15（critical bosonic torus integrand）.** 对 $D=26$ 的临界玻色弦，
$$
\mathcal Z_1(T)
=\frac{V_{26}}2\int_{\mathcal F_T}\frac{d^2\tau}{\tau_2}
(4\pi^2\alpha'\tau_2)^{-13}
|\eta(\tau)|^{-48}
$$
至整体规范化 convention 成立。

**证明.** 将定义 5.14 中的 $Z_X$ 与 $Z_{bc}$ 相乘，并取 $D=26$：
$$
|\eta|^{-52}|\eta|^4=|\eta|^{-48},
\qquad
\frac{d^2\tau}{2\tau_2^2}\tau_2
=\frac12\frac{d^2\tau}{\tau_2}.
$$
$\square$

**注 5.16（tachyon 红外发散）.** 当 $\tau_2\to\infty$ 时，
$|\eta(\tau)|^{-48}\sim e^{4\pi\tau_2}$，这是闭弦 tachyon 沿长管传播的红外
发散。因此玻色弦中 $\lim_{T\to\infty}\mathcal Z_1(T)$ 不存在，命题 5.15 只给出
正规化后的被积函数和有限 cutoff 表达式，不能称为有限真空能。Modular fundamental
domain 去除了 worldsheet UV 重复计数，但不会消除物理 tachyon 的 IR instability。

截断 torus 积分的例子把 gauge fixing 的几层后果集中展示出来：ghost determinant
消去两个非物理 bosonic directions，modular fundamental domain 避免重复计数，
但任何一项都不会消除 tachyon 的长管红外发散。局部层面，总 central charge 为零
使 Weyl/BRST anomaly 消失并固定 $D=26$；态空间层面，closed states 还必须模去
exact states。由此得到的 BRST cohomology 才能在下一章通过顶点算子进入散射振幅。

## 练习

**练习 5.1.** 解释为什么 $c_{\mathrm{matter}}+c_{\mathrm{ghost}}=0$ 是 Weyl anomaly cancellation 条件。

**练习 5.2.** 说明 BRST exact state 为什么应被视为零物理态。

**练习 5.3.** 解释 sphere tree amplitude 为什么需要三个未积分闭弦顶点。

**练习 5.4.** 从 $Z_X$ 和 $Z_{bc}$ 推导临界玻色弦 torus integrand 中的 $|\eta|^{-48}$。
