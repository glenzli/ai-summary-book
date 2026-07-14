# 核心习题解答

本文给出主线核心习题解答。它不是出版版全量题库，但覆盖本书内容收口所需的核心计算与概念检验。

## 练习 0.1

**题目.** 说明为什么 string perturbation theory 的展开参数不是 $\hbar$，而是 worldsheet topology 或 string coupling。

**解答.** 在一阶量子化的弦论中，固定世界面拓扑后的路径积分已经是量子振幅；不同拓扑的世界面给出不同 loop order。常 dilaton 背景满足
$$
S_\Phi=\frac{\Phi_0}{4\pi}\int_\Sigma\sqrt h\,R^{(2)}
=\Phi_0\chi(\Sigma).
$$
闭合 genus $g$ 世界面有 $\chi=2-2g$，因此 Euclidean 权重为
$$
e^{-S_\Phi}=g_s^{2g-2},\qquad g_s=e^{\Phi_0}.
$$
所以扰动展开按 genus 或等价地按 $g_s$ 组织；$\hbar$ 只是在把路径积分整体写成 $e^{iS/\hbar}$ 时的量子计数参数，不是弦微扰论的拓扑展开参数。$\square$

## 练习 0.2

**题目.** 举例说明同一个质量公式在不同 $\alpha'$ convention 下如何改变。

**解答.** 本书采用
$$
T=\frac1{2\pi\alpha'},\qquad
X^\mu(z,\bar z)X^\nu(w,\bar w)\sim
-\frac{\alpha'}2\eta^{\mu\nu}\log|z-w|^2.
$$
开玻色弦物理态条件给出
$$
M^2=\frac1{\alpha'}(N-a),\qquad a=1.
$$
若另一套 convention 把 Regge slope 记为 $\alpha'_{\mathrm{new}}=2\alpha'$，而 oscillator number $N$ 和 normal ordering constant $a$ 不重定义，则同一公式可写成
$$
M^2=\frac{2}{\alpha'_{\mathrm{new}}}(N-a).
$$
物理质量不变；变化的是符号 $\alpha'$ 的定义和 propagator、张力、mode expansion 中与之配套的系数。因此比较不同教材时必须同时检查 $T$、OPE 和 $L_0$ convention。$\square$

## 练习 1.1

**题目.** 从作用量 $S[q]=\int L(q,\dot q,t)\,dt$ 推导 Euler-Lagrange 方程。

**解答.** 令 $q\mapsto q+\epsilon\eta$，其中 $\eta$ 在端点为零。则
$$
\delta S=\int\left(\frac{\partial L}{\partial q}\eta+\frac{\partial L}{\partial\dot q}\dot\eta\right)dt
=
\int\left(\frac{\partial L}{\partial q}-\frac{d}{dt}\frac{\partial L}{\partial\dot q}\right)\eta\,dt.
$$
任意 $\eta$ 下 $\delta S=0$ 等价于 Euler-Lagrange 方程。$\square$

## 练习 1.2

**题目.** 对自由标量场计算 canonical stress tensor 和 Hilbert stress tensor，并说明二者的关系。

**解答.** 平坦时空自由标量
$$
\mathcal L=-\frac12\partial_\rho\phi\,\partial^\rho\phi
$$
的 canonical stress tensor 为
$$
T^\mu_{\ \nu,\mathrm{can}}
=
\frac{\partial\mathcal L}{\partial(\partial_\mu\phi)}\partial_\nu\phi
-\delta^\mu_{\ \nu}\mathcal L
=
-\partial^\mu\phi\,\partial_\nu\phi
+\frac12\delta^\mu_{\ \nu}\partial_\rho\phi\partial^\rho\phi.
$$
Hilbert stress tensor由 curved action
$$
S=-\frac12\int d^dx\sqrt{|g|}\,g^{\mu\nu}\partial_\mu\phi\partial_\nu\phi
$$
对 $g^{\mu\nu}$ 变分给出
$$
T_{\mu\nu}
=\partial_\mu\phi\partial_\nu\phi
-\frac12g_{\mu\nu}\partial_\rho\phi\partial^\rho\phi
$$
按本书 mostly-plus 与 Hilbert 定义约定。两者相差指标位置和可能的 improvement term；对 Poincare 不变理论，Hilbert tensor 是对称且直接耦合引力的张量。$\square$

## 练习 1.3

**题目.** 证明常 dilaton coupling 在 genus $g$ 闭合世界面上给出 $g_s^{2g-2}$。

**解答.** 常 dilaton 作用量为
$$
S_\Phi=\frac{\Phi_0}{4\pi}\int_\Sigma\sqrt h\,R^{(2)}.
$$
Gauss-Bonnet theorem 给出
$$
\frac1{4\pi}\int_\Sigma\sqrt h\,R^{(2)}=\chi(\Sigma)=2-2g.
$$
Euclidean 权重为
$$
e^{-S_\Phi}=e^{-\Phi_0(2-2g)}.
$$
令 $g_s=e^{\Phi_0}$，得到
$$
e^{-S_\Phi}=g_s^{2g-2}.
$$
$\square$

## 练习 2.1

**题目.** 证明 Nambu-Goto 作用量在世界面重参数化下不变。

**解答.** 重参数化 $\sigma^a\mapsto\sigma'^a$ 下诱导度量 $\gamma_{ab}$ 按二阶张量变换，面积元满足
$$
d^2\sigma'\sqrt{-\det\gamma'}=d^2\sigma\sqrt{-\det\gamma}.
$$
因此 $S_{NG}=-T\int\sqrt{-\det\gamma}$ 不变。$\square$

## 练习 2.2

**题目.** 在 conformal gauge 下写出闭弦的一般经典解。

**解答.** Conformal gauge 下运动方程为
$$
(\partial_\tau^2-\partial_\sigma^2)X^\mu=0.
$$
引入 $\sigma^\pm=\tau\pm\sigma$，通解为
$$
X^\mu=X_L^\mu(\sigma^+)+X_R^\mu(\sigma^-).
$$
闭弦周期性 $X(\tau,\sigma+2\pi)=X(\tau,\sigma)$ 在非紧方向要求 oscillator modes 为整数，得到
$$
X^\mu=x^\mu+\alpha'p^\mu\tau
+i\sqrt{\frac{\alpha'}2}\sum_{n\ne0}\frac1n
\left(\alpha_n^\mu e^{-in(\tau-\sigma)}
+\tilde\alpha_n^\mu e^{-in(\tau+\sigma)}\right).
$$
$\square$

## 练习 2.3

**题目.** 从开弦边界变分推出 Neumann 与 Dirichlet 条件。

**解答.** Conformal gauge 中 $X$ 变分的边界项为
$$
\delta S_{\partial\Sigma}
=-\frac1{2\pi\alpha'}\int d\tau\,
\delta X_\mu\,\partial_\sigma X^\mu
\bigg|_{\sigma=0}^{\sigma=\pi}.
$$
要使变分问题良定，可要求端点变分任意但
$$
\partial_\sigma X^\mu|_{\partial\Sigma}=0,
$$
这就是 Neumann 条件；也可要求
$$
\delta X^\mu|_{\partial\Sigma}=0,
$$
这就是 Dirichlet 条件。$\square$

## 练习 3.1

**题目.** 用 free boson OPE 计算 $\partial X^\mu(z)e^{ik\cdot X(w)}$ 的奇异部分。

**解答.** 由
$$
\partial X^\mu(z)X^\nu(w)\sim-\frac{\alpha'}{2}\frac{\eta^{\mu\nu}}{z-w}
$$
和 Wick contraction 得
$$
\partial X^\mu(z)e^{ik\cdot X(w)}
\sim
-i\frac{\alpha'}{2}\frac{k^\mu}{z-w}e^{ik\cdot X(w)}.
$$
$\square$

## 练习 3.2

**题目.** 推导 primary field 在 $L_0$ 和 $L_{-1}$ 下的变换。

**解答.** 对 weight $h$ 的 primary field，
$$
T(z)\mathcal O(w)\sim
\frac{h\mathcal O(w)}{(z-w)^2}
+\frac{\partial\mathcal O(w)}{z-w}.
$$
由
$$
L_n=\oint_w\frac{dz}{2\pi i}z^{n+1}T(z)
$$
得
$$
[L_n,\mathcal O(w)]
=
\left(w^{n+1}\partial_w+h(n+1)w^n\right)\mathcal O(w).
$$
取 $n=-1$ 得
$$
[L_{-1},\mathcal O]=\partial\mathcal O,
$$
取 $n=0$ 得
$$
[L_0,\mathcal O]=(w\partial+h)\mathcal O.
$$
$\square$

## 练习 3.3

**题目.** 用 $T(z)T(w)$ OPE 的 contour integral 推导 Virasoro algebra 的中心项。

**解答.** 取
$$
T(z)T(w)\sim\frac{c/2}{(z-w)^4}
+\frac{2T(w)}{(z-w)^2}
+\frac{\partial T(w)}{z-w}.
$$
将
$$
[L_m,L_n]=
\oint_0\frac{dw}{2\pi i}w^{n+1}
\oint_w\frac{dz}{2\pi i}z^{m+1}T(z)T(w)
$$
代入。四阶极点贡献
$$
\frac c2\oint_0\frac{dw}{2\pi i}w^{n+1}
\frac1{3!}\partial_w^3w^{m+1}
=\frac c{12}m(m^2-1)\delta_{m+n,0}.
$$
二阶和一阶极点给出 $(m-n)L_{m+n}$。$\square$

## 练习 3.4

**题目.** 计算 level $1$ descendant $L_{-1}|h\rangle$ 的范数，并说明 $h=0$ 时发生什么。

**解答.** 对 highest-weight state，
$$
L_n|h\rangle=0\quad(n>0),\qquad L_0|h\rangle=h|h\rangle.
$$
因此
$$
\|L_{-1}|h\rangle\|^2
=\langle h|L_1L_{-1}|h\rangle
=\langle h|[L_1,L_{-1}]|h\rangle
=2h\langle h|h\rangle.
$$
若 $h=0$，该 descendant 范数为零。在 vacuum module 中 $L_{-1}|0\rangle$ 对应 identity 的导数，应作为 null direction 商去。$\square$

## 练习 4.1

**题目.** 从闭弦模展开推导 oscillator commutators。

**解答.** 将闭弦模展开代入
$$
[X^\mu(\tau,\sigma),P_\nu(\tau,\sigma')]
=i\delta^\mu_{\ \nu}\delta_{2\pi}(\sigma-\sigma')
$$
并使用
$$
\delta_{2\pi}(\sigma-\sigma')
=\frac1{2\pi}\sum_{n\in\mathbb Z}e^{in(\sigma-\sigma')}.
$$
零模比较给出 $[x^\mu,p^\nu]=i\eta^{\mu\nu}$。非零 Fourier modes 的比较给出
$$
[\alpha_m^\mu,\alpha_n^\nu]=m\delta_{m+n,0}\eta^{\mu\nu},
\qquad
[\tilde\alpha_m^\mu,\tilde\alpha_n^\nu]=m\delta_{m+n,0}\eta^{\mu\nu}.
$$
$\square$

## 练习 4.2

**题目.** 推导开弦质量公式。

**解答.** 开弦 Virasoro 零模为
$$
L_0=\alpha'p^2+N
$$
在本书归一化下的物理条件是 $(L_0-a)|\psi\rangle=0$。由于 $M^2=-p^2$，得
$$
\alpha'(-M^2)+N-a=0,
$$
即
$$
M^2=\frac1{\alpha'}(N-a).
$$
$\square$

## 练习 4.3

**题目.** 对闭弦第一激发层推导横向条件和 gauge equivalence。

**解答.** 闭弦第一激发为
$$
|\epsilon;k\rangle
=\epsilon_{\mu\nu}\alpha_{-1}^\mu\tilde\alpha_{-1}^\nu|0;k\rangle.
$$
$L_0,\tilde L_0$ 给出 $k^2=0$。约束 $L_1|\epsilon;k\rangle=0$ 使用
$$
[L_1,\alpha_{-1}^\mu]=\alpha_0^\mu\propto k^\mu
$$
得到
$$
k^\mu\epsilon_{\mu\nu}=0.
$$
同理由 $\tilde L_1$ 得
$$
k^\nu\epsilon_{\mu\nu}=0.
$$
Null states 由 $L_{-1}$ 或 $\tilde L_{-1}$ 作用在低一层态上产生，使
$$
\epsilon_{\mu\nu}\sim
\epsilon_{\mu\nu}+k_\mu\xi_\nu+k_\nu\tilde\xi_\mu.
$$
$\square$

## 练习 4.4

**题目.** 在 light-cone gauge 中计数玻色弦的横向 oscillator 数，并解释为什么没有负范数振子。

**解答.** Light-cone gauge 固定 $X^+$，Virasoro constraints 解出 $X^-$。因此独立振子只剩
$$
\alpha_n^i,\qquad i=1,\ldots,D-2.
$$
这些方向的 metric 是正定 transverse metric $\delta_{ij}$，所以由它们生成的 Fock space 没有时间方向 oscillator 带来的负范数。临界维数下 Lorentz algebra closure 保证该 gauge-fixed spectrum 与 covariant quantization 的物理谱一致。$\square$

## 练习 5.1

**题目.** 解释为什么 $c_{\text{matter}}+c_{\text{ghost}}=0$ 是 Weyl anomaly cancellation 条件。

**解答.** 在二维 CFT 中，曲面背景上的量子 trace anomaly 与总 central charge 成正比：
$$
\langle T^a_{\ a}\rangle
\propto
c_{\mathrm{tot}}R^{(2)}.
$$
Conformal gauge 固定后，Weyl transformation 仍应是 gauge redundancy；若 $c_{\mathrm{tot}}\ne0$，路径积分测度在 Weyl 变换下产生 anomaly，规范固定后的理论依赖代表度量而非只依赖 conformal class。因此必须有
$$
c_{\mathrm{tot}}=c_{\mathrm{matter}}+c_{\mathrm{ghost}}=0.
$$
$\square$

## 练习 5.2

**题目.** 说明 BRST exact state 为什么应被视为零物理态。

**解答.** 物理态定义为 cohomology：
$$
\mathcal H_{\mathrm{phys}}=\ker Q_B/\operatorname{im}Q_B.
$$
若
$$
|\psi\rangle=Q_B|\chi\rangle,
$$
则 $|\psi\rangle$ 是纯 gauge 方向。任意 BRST-closed bra $\langle\varphi|$ 下，
$$
\langle\varphi|\psi\rangle
=\langle\varphi|Q_B|\chi\rangle
=\pm\langle Q_B\varphi|\chi\rangle=0.
$$
在振幅语言中，$Q_B$ contour 可从 exact 插入移走并在其他 BRST-closed 插入处不给贡献，因此 exact states decouple。$\square$

## 练习 5.3

**题目.** 解释 sphere tree amplitude 为什么需要三个未积分闭弦顶点。

**解答.** Riemann sphere 的 conformal Killing group 是 $PSL_2(\mathbb C)$，holomorphic conformal Killing vectors 由
$$
1,\quad z,\quad z^2
$$
张成，因此有三个 $c$ ghost zero modes；antiholomorphic 部分也有三个 $\bar c$ zero modes。路径积分中 ghost zero modes 必须被插入吸收，否则 Grassmann 积分为零。每个未积分闭弦顶点 $c\bar cV$ 吸收一对 zero modes，所以 sphere tree amplitude 需要三个未积分闭弦顶点。$\square$

## 练习 5.4

**题目.** 从 $Z_X$ 和 $Z_{bc}$ 推导临界玻色弦 torus integrand 中的 $|\eta|^{-48}$。

**解答.** $D$ 个非紧 free bosons 的 oscillator contribution 为
$$
|\eta|^{-2D}.
$$
临界玻色弦 $D=26$，所以 matter 给出 $|\eta|^{-52}$。$bc$ ghosts 给出
$$
|\eta|^4
$$
以及 zero-mode 相关的 $\tau_2$ 因子。因此总的 $\eta$ 幂次为
$$
|\eta|^{-52}|\eta|^4=|\eta|^{-48}.
$$
$\square$

## 练习 6.1

**题目.** 用第三章 OPE 推导闭弦 tachyon 的 conformal weights，并验证其质量公式。

**解答.** 对
$$
V_k=:e^{ik\cdot X}:
$$
使用
$$
T(z)=-\frac1{\alpha'}:\partial X^\mu\partial X_\mu:
$$
和
$$
\partial X^\mu(z)e^{ik\cdot X(w)}
\sim
-i\frac{\alpha'}2\frac{k^\mu}{z-w}e^{ik\cdot X(w)}.
$$
双收缩给出
$$
T(z)V_k(w,\bar w)\sim
\frac{\alpha'k^2/4}{(z-w)^2}V_k
+\frac1{z-w}\partial V_k.
$$
故 $h=\alpha'k^2/4$，同理 $\bar h=\alpha'k^2/4$。闭弦积分顶点要求 $(h,\bar h)=(1,1)$，所以 $k^2=4/\alpha'$，即
$$
M^2=-k^2=-\frac4{\alpha'}.
$$
$\square$

## 练习 6.2

**题目.** 推导开弦边界指数算子的 Koba-Nielsen 因子。

**解答.** 边界上由 doubling trick 得
$$
\langle X^\mu(x_i)X^\nu(x_j)\rangle
=-2\alpha'\eta^{\mu\nu}\log|x_i-x_j|.
$$
Wick theorem 给出
$$
\left\langle\prod_i:e^{ik_i\cdot X(x_i)}:\right\rangle
=(2\pi)^D\delta^{(D)}\!\left(\sum_i k_i\right)
\prod_{i<j}|x_i-x_j|^{2\alpha'k_i\cdot k_j}.
$$
$\square$

## 练习 6.3

**题目.** 从 Veneziano amplitude 的 gamma function 表达式读出开弦第 $N$ 层的 pole 位置。

**解答.** Ordered 四点 tachyon 振幅含因子
$$
\Gamma(-1-\alpha's).
$$
Gamma function 在非正整数处有 simple pole：
$$
-1-\alpha's=-m,\qquad m=0,1,2,\ldots.
$$
因此
$$
\alpha's=m-1.
$$
令 $N=m$，得到
$$
M_N^2=s=\frac{N-1}{\alpha'},
$$
这正是开弦质量公式 $M^2=(N-1)/\alpha'$ 的谱位置。$\square$

## 练习 6.4

**题目.** 解释 worldsheet degeneration 如何给出 target-space propagator pole。

**解答.** 当 punctured worldsheet 退化为由长管连接的两部分时，长管可用 plumbing parameter $q$ 描述。CFT 在长管上传播给出
$$
\sum_\alpha q^{L_0^\alpha-a}\bar q^{\tilde L_0^\alpha-a}
|\alpha\rangle\langle\alpha|.
$$
对 $q$ 的径向积分等价于 Schwinger proper time 积分。积分在
$$
L_0-a=0,\qquad \tilde L_0-a=0
$$
处产生 pole，即 target-space 中间态 on-shell propagator
$$
\frac1{k^2+M_\alpha^2}.
$$
Residue 是左右两个低点振幅的乘积。$\square$

## 练习 7.1

**题目.** 证明 T-duality 下 $p_L$ 不变而 $p_R$ 变号。

**解答.** T-duality 取
$$
R'=\frac{\alpha'}R,\qquad n'=w,\qquad w'=n.
$$
则
$$
p'_L=\frac{n'}{R'}+\frac{w'R'}{\alpha'}
=\frac w{\alpha'/R}+\frac{n(\alpha'/R)}{\alpha'}
=\frac{wR}{\alpha'}+\frac nR=p_L,
$$
而
$$
p'_R=\frac{n'}{R'}-\frac{w'R'}{\alpha'}
=\frac{wR}{\alpha'}-\frac nR=-p_R.
$$
$\square$

## 练习 7.2

**题目.** 从 $L_0=\tilde L_0=a$ 推导圆紧化闭弦的 level matching 条件。

**解答.** 圆紧化时
$$
L_0=\frac{\alpha'}4(-M^2+p_L^2)+N,\qquad
\tilde L_0=\frac{\alpha'}4(-M^2+p_R^2)+\tilde N.
$$
两式相减得
$$
0=L_0-\tilde L_0
=\frac{\alpha'}4(p_L^2-p_R^2)+N-\tilde N.
$$
由于
$$
p_L^2-p_R^2=
\left(\frac nR+\frac{wR}{\alpha'}\right)^2
-\left(\frac nR-\frac{wR}{\alpha'}\right)^2
=\frac{4nw}{\alpha'},
$$
所以
$$
N-\tilde N+nw=0.
$$
$\square$

## 练习 7.3

**题目.** 用 dual coordinate 的定义证明 Neumann 条件变为 Dirichlet 条件。

**解答.** T-dual coordinate 满足
$$
\partial_\tau\widetilde X=\partial_\sigma X,\qquad
\partial_\sigma\widetilde X=\partial_\tau X.
$$
开弦端点的 Neumann 条件为
$$
\partial_\sigma X|_{\partial\Sigma}=0.
$$
因此
$$
\partial_\tau\widetilde X|_{\partial\Sigma}=0.
$$
这表示端点处 $\widetilde X$ 沿世界面时间不变，即端点固定在 dual circle 的某一点上，正是 Dirichlet 条件。$\square$

## 练习 7.4

**题目.** 说明为什么 orbifold 闭弦理论需要 twisted sectors。

**解答.** Orbifold torus path integral 需要允许场沿两个 torus cycles 分别带有 group elements $g,h$ 的 twisted boundary conditions。Modular transformation $S:\tau\mapsto-1/\tau$ 会交换两个 cycles，因此会把 temporal twist 和 spatial twist 互换。若只保留 untwisted sector，即 spatial twist 为 identity 的 sector，这个集合在 modular group 下不闭合。加入所有 twisted sectors 后，partition function 才能形成 modular invariant 的求和。$\square$

## 练习 8.1

**题目.** 由 fermion OPE 推导 $\{\psi_r^\mu,\psi_s^\nu\}=\eta^{\mu\nu}\delta_{r+s,0}$。

**解答.** 模展开为
$$
\psi^\mu(z)=\sum_r\psi_r^\mu z^{-r-1/2},
$$
因此
$$
\psi_r^\mu=\oint_0\frac{dz}{2\pi i}\,z^{r-1/2}\psi^\mu(z).
$$
对两个 modes 使用径向有序 contour，并代入
$$
\psi^\mu(z)\psi^\nu(w)\sim\frac{\eta^{\mu\nu}}{z-w},
$$
内层 contour 取 residue 后得到
$$
\{\psi_r^\mu,\psi_s^\nu\}
=\eta^{\mu\nu}\oint_0\frac{dw}{2\pi i}\,w^{r+s-1}
=\eta^{\mu\nu}\delta_{r+s,0}.
$$
$\square$

## 练习 8.2

**题目.** 用 central charge 计数推导 RNS 临界维数。

**解答.** $D$ 个 free bosons 贡献 $c=D$，$D$ 个 Majorana fermions 贡献 $c=D/2$，故 matter central charge 为
$$
c_m=\frac{3D}{2}.
$$
RNS ghost sector 中
$$
c_{bc}=-26,\qquad c_{\beta\gamma}=11,
$$
所以
$$
c_{\mathrm{gh}}=-15.
$$
量子 super-Weyl anomaly cancellation 要求
$$
c_m+c_{\mathrm{gh}}=0,
$$
即
$$
\frac{3D}{2}-15=0.
$$
因此 $D=10$。$\square$

## 练习 8.3

**题目.** 说明为什么 R sector ground states 构成 spacetime Clifford algebra 表示。

**解答.** R sector 中 fermion modes 为整数，特别有零模 $\psi_0^\mu$。由 oscillator algebra，
$$
\{\psi_0^\mu,\psi_0^\nu\}=\eta^{\mu\nu}.
$$
若定义
$$
\Gamma^\mu=\sqrt2\,\psi_0^\mu,
$$
则
$$
\{\Gamma^\mu,\Gamma^\nu\}=2\eta^{\mu\nu}.
$$
这就是 target-space Clifford algebra。因此 R ground state space 必须承载该 Clifford algebra 的表示，也就是 spacetime spinor 表示。$\square$

## 练习 8.4

**题目.** 用 bosonization 计算十维 RNS spin field 的 conformal weight。

**解答.** 十个 real worldsheet fermions 可两两 bosonize 为五个 free bosons $H_I$。Spin field 形如
$$
S\sim\exp\left(\frac i2\sum_{I=1}^5s_IH_I\right),
\qquad s_I=\pm1.
$$
对一个 canonically normalized boson，$e^{iaH}$ 的 conformal weight 为 $a^2/2$。每个 $a=s_I/2$ 给出 $1/8$，五个相加得到
$$
h(S)=5\cdot\frac18=\frac58.
$$
$\square$

## 练习 9.1

**题目.** 说明 type IIA 与 type IIB 的 R-R potential 次数 parity 为什么不同。

**解答.** R-R states 是左右 Ramond ground states 的张量积。十维中 spinor bilinear 可用 antisymmetric gamma matrices 展开为 differential forms。相同 chirality 的 Majorana-Weyl spinors 只给出偶次数 potentials；相反 chirality 的张量积只给出奇次数 potentials。因此 IIB 有偶次数 R-R potentials，IIA 有奇次数 R-R potentials。$\square$

## 练习 9.2

**题目.** 用 light-cone gauge 计数十维 GS string 的 transverse bosons 与 fermions。

**解答.** 十维坐标 $X^\mu$ 中，light-cone gauge 固定 $X^+$，Virasoro constraints 解出 $X^-$，剩余
$$
D-2=8
$$
个 transverse bosons。十维 Majorana-Weyl spinor 有 $16$ 个实分量；kappa symmetry 消去一半，留下 $8$ 个物理 fermionic components。因此 light-cone gauge 下 bosonic 与 fermionic degrees of freedom 匹配。$\square$

## 练习 9.3

**题目.** 根据 R-R potential 的 form degree 判断 IIA/IIB 中允许的 D-brane parity。

**解答.** D$p$-brane 电耦合于 R-R potential $C_{p+1}$。IIA 的 R-R potentials 为奇次数 $C_1,C_3,\ldots$，故 $p+1$ 为奇数，$p$ 为偶数。IIB 的 R-R potentials 为偶次数 $C_0,C_2,C_4,\ldots$，故 $p+1$ 为偶数，$p$ 为奇数。于是 IIA 允许偶 $p$ D-branes，IIB 允许奇 $p$ D-branes。$\square$

## 练习 10.1

**题目.** 解释为什么左移动额外自由度需要 $16$ 维 lattice。

**解答.** Heterotic string 的左移动部分是 bosonic string，需要 matter central charge $26$。十个非紧坐标 $X^\mu$ 已贡献 $10$，因此内部左移动 CFT 必须贡献
$$
26-10=16.
$$
若用 compact bosons 实现，每个 boson 贡献 $c=1$，故需要 $16$ 个左移动 compact bosons，也就是一个 $16$ 维 lattice compactification。$\square$

## 练习 10.2

**题目.** 验证 $P^2=2$ 的内部 lattice vertex 具有 left conformal weight $1$。

**解答.** 对归一化为
$$
Y^I(z)Y^J(w)\sim-\delta^{IJ}\log(z-w)
$$
的内部 compact bosons，vertex
$$
V_P(z)=:e^{iP\cdot Y(z)}:
$$
的 conformal weight 为
$$
h=\frac{P^2}{2}.
$$
若 $P^2=2$，则 $h=1$，因此该 vertex 可作为 Kac-Moody current。$\square$

## 练习 10.3

**题目.** 说明 $\Theta_\Lambda(\tau)/\eta(\tau)^{16}$ 中 theta function 和 eta function 分别来自哪些自由度。

**解答.** $16$ 个左移动 compact bosons 的 Hilbert space 分为零模 momentum 部分和 oscillator 部分。零模 $P\in\Lambda$ 对 $L_0$ 的贡献为 $P^2/2$，求和给出
$$
\Theta_\Lambda(\tau)=\sum_{P\in\Lambda}q^{P^2/2}.
$$
每个 chiral boson 的 oscillator modes 贡献 $\eta(\tau)^{-1}$，$16$ 个 bosons 给出 $\eta(\tau)^{-16}$。因此内部 lattice CFT 的 contribution 为
$$
\frac{\Theta_\Lambda(\tau)}{\eta(\tau)^{16}}.
$$
$\square$

## 练习 11.1

**题目.** 说明 string frame 与 Einstein frame 的区别。

**解答.** String frame 是世界面 sigma model 直接耦合的 metric；fundamental string action 中出现的是 $g^S_{\mu\nu}$。在低能有效作用中，gravitational term 带有整体因子 $e^{-2\Phi}$。Einstein frame 通过
$$
g^E_{\mu\nu}=e^{-4\Phi/(D-2)}g^S_{\mu\nu}
$$
定义，使 Einstein-Hilbert 项具有标准规范化，不再带整体 dilaton factor。两者是场变量选择不同，不改变 on-shell 物理量。$\square$

## 练习 11.2

**题目.** 从 $S_{\mathrm{NS}}$ 对 $B$ 变分，推出 $d(e^{-2\Phi}*H)=0$。

**解答.** $B$ 只通过 $H=dB$ 进入
$$
S_H=-\frac1{4\kappa_0^2}\int e^{-2\Phi}H\wedge *H.
$$
变分得
$$
\delta S_H
=-\frac1{2\kappa_0^2}\int e^{-2\Phi}d(\delta B)\wedge *H.
$$
分部积分并忽略边界项：
$$
\delta S_H
=\frac1{2\kappa_0^2}\int \delta B\wedge d(e^{-2\Phi}*H).
$$
任意 $\delta B$ 下作用量驻定，故
$$
d(e^{-2\Phi}*H)=0.
$$
$\square$

## 练习 11.3

**题目.** 从 string-frame NS-NS action 对 dilaton 变分，推导最低阶 dilaton equation。

**解答.** 取
$$
S=\int\sqrt{-g}\,e^{-2\Phi}
\left(R+4|\nabla\Phi|^2-\frac1{12}H^2\right).
$$
对 $\Phi$ 变分。$e^{-2\Phi}$ 给出
$$
-2\delta\Phi\left(R+4|\nabla\Phi|^2-\frac1{12}H^2\right).
$$
动能项变分为
$$
8\nabla_\mu\Phi\nabla^\mu\delta\Phi,
$$
分部积分并包含 $e^{-2\Phi}$ 后得到
$$
-8\delta\Phi\,\nabla^2\Phi
+16\delta\Phi|\nabla\Phi|^2.
$$
合并并除去公共因子，得
$$
4\nabla^2\Phi-4|\nabla\Phi|^2+R-\frac1{12}H^2=0.
$$
$\square$

## 练习 12.1

**题目.** 将 DBI action 展开到 $F^2$ 阶。

**解答.** 在平坦背景、$B=0$ 和常 dilaton 下，
$$
S_{\mathrm{DBI}}
=-\tau_p\int d^{p+1}\xi
\sqrt{-\det(\eta_{ab}+2\pi\alpha'F_{ab})}.
$$
利用
$$
\sqrt{\det(1+M)}
=1+\frac12\operatorname{tr}M
+\frac18(\operatorname{tr}M)^2
-\frac14\operatorname{tr}M^2+\cdots.
$$
对 antisymmetric $F$ 有 $\operatorname{tr}F=0$，并得到
$$
\sqrt{-\det(\eta+2\pi\alpha'F)}
=1+\frac{(2\pi\alpha')^2}{4}F_{ab}F^{ab}+\cdots.
$$
因此
$$
S_{\mathrm{DBI}}
=-\tau_p\int d^{p+1}\xi
\left(1+\frac{(2\pi\alpha')^2}{4}F_{ab}F^{ab}+\cdots\right).
$$
$\square$

## 练习 12.2

**题目.** 由 WZ coupling 说明 worldvolume flux 如何诱导 lower-dimensional D-brane charge。

**解答.** WZ coupling 为
$$
S_{\mathrm{WZ}}
=\mu_p\int P\left[\sum_q C_q\right]\wedge e^{B+2\pi\alpha'F}.
$$
取 $B=0$ 并展开指数：
$$
e^{2\pi\alpha'F}=1+2\pi\alpha'F+\frac12(2\pi\alpha'F)^2+\cdots.
$$
其中
$$
\mu_p\int C_{p-1}\wedge 2\pi\alpha'F
$$
是对 $C_{p-1}$ 的电耦合。由于 $C_{p-1}$ 电耦合于 D$(p-2)$-brane，该项说明 worldvolume flux 携带 D$(p-2)$ charge。$\square$

## 练习 12.3

**题目.** 解释为什么重合 D-branes 的 transverse scalars 取 adjoint 表示，并说明对角 vev 的几何意义。

**解答.** $N$ 个重合 D-branes 上，开弦端点带 Chan-Paton labels $i,j$，因此 massless fields 是 $N\times N$ 矩阵，取值于 $\mathfrak u(N)$ 的 adjoint 表示。沿 Dirichlet 方向的 massless modes 是 transverse scalars $\Phi^i$。若
$$
\Phi^i=\operatorname{diag}(\phi^i_1,\ldots,\phi^i_N),
$$
则第 $r$ 个 brane 的 transverse 位置为
$$
Y^i_r=2\pi\alpha'\phi^i_r.
$$
off-diagonal entries 对应连接不同 branes 的开弦；当 branes 分离时这些 modes 获得与弦长成正比的质量。$\square$

## 练习 13.1

**题目.** 说明 $c_1(X)=0$ 与 canonical bundle 平凡之间的关系。

**解答.** Canonical bundle 为
$$
K_X=\Lambda^nT^{*(1,0)}X.
$$
其第一 Chern class 满足
$$
c_1(K_X)=-c_1(TX).
$$
因此 $c_1(TX)=0$ 等价于 $c_1(K_X)=0$。若进一步存在 nowhere-vanishing holomorphic $n$-form $\Omega$，则 $\Omega$ 给出 $K_X$ 的 holomorphic trivialization。反过来，$K_X$ holomorphically trivial 意味着存在这样的 $\Omega$，从而 $c_1(X)=0$。$\square$

## 练习 13.2

**题目.** 对 Calabi-Yau threefold，解释为什么 complex structure moduli 由 $H^{2,1}(X)$ 计数。

**解答.** Complex structure 的 infinitesimal deformation 由
$$
H^1(X,TX)
$$
表示。Calabi-Yau threefold 有 nowhere-vanishing holomorphic three-form $\Omega$。用 Beltrami differential $\mu\in H^1(TX)$ 与 $\Omega$ 收缩：
$$
\mu\lrcorner \Omega\in H^{2,1}(X).
$$
该映射在 Calabi-Yau 情形给出同构，因此 complex structure moduli 的复维数为
$$
h^{2,1}(X).
$$
$\square$

## 练习 13.3

**题目.** 用五次齐次多项式计数推导 quintic 的 $h^{2,1}=101$。

**解答.** $\mathbb P^4$ 中五次齐次多项式的单项式数为
$$
\binom{5+4}{4}=126.
$$
整体缩放不改变 hypersurface，减去 $1$。射影坐标变换群 $\operatorname{PGL}(5)$ 维数为 $25-1=24$，对应冗余 complex structure 参数。故
$$
h^{2,1}=126-1-24=101.
$$
$\square$

## 练习 14.1

**题目.** 比较 T-duality 与 S-duality 的耦合常数行为。

**解答.** T-duality 首先作用在紧化半径和 momentum/winding 上：
$$
R\leftrightarrow\frac{\alpha'}R,\qquad n\leftrightarrow w.
$$
在保持低维 Newton constant 的规范下，dilaton 会相应平移，但 T-duality 不把弱耦合普遍映到强耦合。S-duality 则直接作用于 string coupling，例如
$$
g_s\mapsto \frac1{g_s},
$$
因此把弱耦合描述映到强耦合描述。$\square$

## 练习 14.2

**题目.** 用 D0-brane 质量推导 $R_{11}=g_s\ell_s$。

**解答.** Type IIA 中 D0-brane 质量为
$$
M_{D0}=\frac1{g_s\ell_s}.
$$
若 D0-brane 是十一维圆上的单位 KK momentum，则其质量应为
$$
M_{KK}=\frac1{R_{11}}.
$$
令 $M_{D0}=M_{KK}$，得到
$$
R_{11}=g_s\ell_s.
$$
$\square$

## 练习 14.3

**题目.** 用 M2-brane 张力推导 fundamental string tension。

**解答.** M2-brane 张力为
$$
T_{\mathrm{M2}}=\frac1{(2\pi)^2\ell_{11}^3}.
$$
包裹半径 $R_{11}=g_s\ell_s$ 的十一维圆后，所得 string 张力为
$$
T=2\pi R_{11}T_{\mathrm{M2}}
=\frac{R_{11}}{2\pi\ell_{11}^3}.
$$
代入 $\ell_{11}=g_s^{1/3}\ell_s$，得
$$
T=\frac{g_s\ell_s}{2\pi g_s\ell_s^3}
=\frac1{2\pi\ell_s^2}
=\frac1{2\pi\alpha'}.
$$
$\square$

## 练习 15.1

**题目.** 说明 torus modular parameter $\tau$ 的基本区域为什么避免重复计数。

**解答.** Torus
$$
E_\tau=\mathbb C/(\mathbb Z+\tau\mathbb Z)
$$
的 homology cycle basis 可由 $SL(2,\mathbb Z)$ 变换重新选择。该变换把
$$
\tau\mapsto\frac{a\tau+b}{c\tau+d}
$$
但不改变复环面同构类。若在整个 upper half-plane 上积分，会把同一个 torus 的不同 cycle basis 重复计数。取 $SL(2,\mathbb Z)$ 的 fundamental domain 正是对这些等价描述取商。$\square$

## 练习 15.2

**题目.** 用 degeneration 图像解释为什么高 genus 振幅边界应出现低阶振幅的因子化。

**解答.** 当 Riemann surface 出现长细管时，可用 plumbing parameter $q$ 描述连接区域。CFT 在细管上传播的贡献为
$$
\sum_\alpha q^{L_0^\alpha-a}\bar q^{\tilde L_0^\alpha-a}
|\alpha\rangle\langle\alpha|.
$$
这等价于在两侧曲面之间插入一组完备中间态。对 $q$ 积分后，在中间态 on-shell 处产生 pole，residue 等于左右两个低阶振幅的乘积。因此 moduli space 边界编码 perturbative unitarity 的因子化。$\square$

## 练习 15.3

**题目.** 说明 compact boson torus partition function 中 momentum/winding lattice sum 如何体现 T-duality。

**解答.** 半径 $R$ 的 compact boson 零模贡献为
$$
\sum_{n,w\in\mathbb Z}
q^{\alpha'p_L^2/4}\bar q^{\alpha'p_R^2/4},
$$
其中
$$
p_L=\frac nR+\frac{wR}{\alpha'},\qquad
p_R=\frac nR-\frac{wR}{\alpha'}.
$$
T-duality 取 $R\mapsto\alpha'/R$ 且 $n\leftrightarrow w$，于是 $p_L\mapsto p_L$、$p_R\mapsto -p_R$。由于 partition function 只依赖 $p_L^2,p_R^2$ 并对所有 $n,w$ 求和，该 lattice sum 不变。$\square$

## 练习 16.1

**题目.** 说明 A-model 为什么只依赖 Kahler moduli 而不依赖 complex structure moduli。

**解答.** A-twist 后，sigma model action 可分为 $Q$-exact 项和 topological term。Complex structure 或 metric 中不改变 Kahler class 的变形只改变 $Q$-exact 部分；$Q$-closed observables 的 correlators 对 $Q$-exact deformation 不变。非平凡依赖来自
$$
\int_\Sigma f^*(B+iJ),
$$
即 complexified Kahler moduli。因此 A-model 只依赖 Kahler moduli。$\square$

## 练习 16.2

**题目.** 解释 mirror symmetry 如何把 curve counting 转化为 period 计算。

**解答.** Mirror symmetry 给出
$$
A\text{-model on }X\simeq B\text{-model on }Y.
$$
$X$ 的 A-model genus-zero prepotential 含有 holomorphic curve counting 信息；$Y$ 的 B-model prepotential 由 holomorphic three-form periods
$$
\int_\Gamma\Omega_Y
$$
和 Picard-Fuchs equations 计算。Mirror map 把 $Y$ 的 complex structure flat coordinates 对应到 $X$ 的 Kahler coordinates。将 B-model prepotential 用这些 coordinates 展开，即可读出 $X$ 上的 Gromov-Witten invariants。$\square$

## 练习 16.3

**题目.** 写出 quintic mirror Picard-Fuchs operator，并说明 holomorphic period 与 logarithmic period 在 mirror map 中的作用。

**解答.** Quintic mirror 的 Picard-Fuchs operator 可写为
$$
\mathcal L=
\theta^4
-5z(5\theta+1)(5\theta+2)(5\theta+3)(5\theta+4),
\qquad
\theta=z\frac{d}{dz}.
$$
Periods 满足 $\mathcal L\Pi=0$。在 large complex structure point 附近，有 holomorphic period $\Pi_0$ 和含 logarithm 的 period $\Pi_1$。Mirror map 为
$$
t(z)=\frac{\Pi_1(z)}{\Pi_0(z)},\qquad q=e^{2\pi it}.
$$
这里 $t$ 是 A-model 的 flat Kahler coordinate。$\square$

## 练习 17.1

**题目.** 说明 BPS 条件为什么有助于跨耦合常数比较态数。

**解答.** BPS 态饱和
$$
M=|Z(Q)|
$$
并被部分 supercharges 湮灭，因此属于短表示。短表示不能在一般连续变形下变成长表示，除非发生 wall crossing 或与其他短表示重组。故 BPS index 在同一 chamber 内不随耦合常数变化，可在弱耦合 D-brane 描述中计算，再与强耦合 black hole 描述比较。$\square$

## 练习 17.2

**题目.** 取 17.8B 的 K3 D1--D5 elliptic genus 为外部输入，用 17.8C 的
fixed-index Jacobi--Rademacher 渐近推导 $\ell=0$ BPS index 的 leading exponent；
写出 Cardy 区，并说明何时可把该指数增长解释为绝对 BPS degeneracy 的 leading
entropy。

**解答.** 记 $N=Q_1Q_5$。K3 D1--D5 elliptic genus
$$
\Phi_N(\tau,z)=\sum_{n,\ell}\Omega_N(n,\ell)q^ny^\ell
$$
是 weight $0$、index $N$ 的 weak Jacobi form；其 trace 中右动非基态成对抵消，故
$\Omega_N(n,\ell)$ 计数的是固定右动 Ramond ground sector 的带符号 index，而不是
完整 CFT 总态数。DMVV 精确乘积及 $K3$ 的
$q^0$ 项给出
$$
[q^0y^{\pm N}]\Phi_N=N+1,
\qquad \Delta_{\rm p}=-N^2.
$$
对判别式
$$
\Delta=4Nn-\ell^2
$$
固定 $N$ 并取 $\Delta\to\infty$。EZ85/DMZ12 的外部 Jacobi/Rademacher 输入在
leading modular coupling 非零的 residue sector 给出
$$
\log|\Omega_N(n,\ell)|
=\frac{\pi}{N}\sqrt{|\Delta_{\rm p}|\Delta}+o(\sqrt\Delta)
=\pi\sqrt\Delta+o(\sqrt\Delta).
$$
最大 polar component 到 $\ell=0$ residue 的 $S$ coupling 非零，故取 $\ell=0$ 得
$$
S_{\mathrm{index}}
=\log|\Omega_{Q_1Q_5}(n,0)|
=2\pi\sqrt{Q_1Q_5n}
+o\!\left(\sqrt{Q_1Q_5n}\right).
$$
严格的 fixed-index Cardy 区是固定 $N$、$n\to\infty$；若 $N$ 同时放大，还须另加
leading coefficient 和余项一致受控的 uniform-saddle 假设，并取
$\Delta/N^2\to\infty$，对 $\ell=0$ 即
$n/(Q_1Q_5)\to\infty$。普通 Cardy 定理只控制完整 modular-invariant partition
function，不能替代这个 BPS-index 输入。

若 $d_{\mathrm{BPS}}$ 是同一扇区的绝对态数，则只有
$|\Omega_N|\le d_{\mathrm{BPS}}$。要把上式升级为绝对简并的 leading entropy，必须
另行验证或假设
$$
\log d_{\mathrm{BPS}}-\log|\Omega_N|=o(\sqrt\Delta),
$$
即不存在指数级符号 cancellation。$\square$

## 练习 17.3

**题目.** 说明为什么 BPS index 可能在 wall of marginal stability 上跳变。

**解答.** BPS bound state 的稳定性取决于 constituent central charges 的相位。当两个或多个 constituents 的 central charges 相位对齐时，bound state 的结合能可趋于零，态可在保持总电荷的情况下衰变。此时 Hilbert space 中对应 BPS sector 的短表示可重组，BPS index 在穿过该 wall 时发生跳变。在同一 chamber 内没有这种 marginal decay，index 保持不变。$\square$

## 练习 18.1

**题目.** 解释大 $N$ 和大 't Hooft coupling 极限为何对应 classical supergravity。

**解答.** AdS/CFT 字典中
$$
\lambda=g_{\mathrm{YM}}^2N,\qquad
\frac{R^4}{\alpha'^2}\sim\lambda,\qquad
g_s\sim\frac{\lambda}{N}.
$$
因此 $\lambda\gg1$ 意味着 $R\gg \ell_s$，string-scale curvature corrections 被抑制；同时需要 $g_s\sim\lambda/N\ll1$，即在给定大 $\lambda$ 后仍取足够大的 $N$，以抑制 string loop corrections。两者同时成立时，bulk theory 可由 classical supergravity 近似。$\square$

## 练习 18.2

**题目.** 推导 $\operatorname{AdS}_{d+1}$ 中标量质量和 CFT scaling dimension 的关系。

**解答.** 在 Poincare 坐标
$$
ds^2=R^2\frac{dz^2+dx_idx^i}{z^2}
$$
中，忽略边界方向动量，令
$$
\phi(z)\sim z^\alpha.
$$
Klein-Gordon 方程
$$
(\nabla^2-m^2)\phi=0
$$
的 near-boundary 主导项给出
$$
\alpha(\alpha-d)=m^2R^2.
$$
取 $\alpha=\Delta$ 或 $\alpha=d-\Delta$，得到
$$
m^2R^2=\Delta(\Delta-d).
$$
$\square$

## 练习 18.3

**题目.** 用 conformal invariance 固定 scalar primary two-point function 的幂次。

**解答.** Translation 和 rotation invariance 要求
$$
\langle\mathcal O(x)\mathcal O(0)\rangle=f(|x|).
$$
若 $\mathcal O$ 的 scaling dimension 为 $\Delta$，scale transformation $x\mapsto\lambda x$ 下
$$
\mathcal O(x)\mapsto\lambda^{-\Delta}\mathcal O(\lambda x).
$$
因此
$$
f(\lambda |x|)=\lambda^{-2\Delta}f(|x|),
$$
解得
$$
f(|x|)=\frac{C}{|x|^{2\Delta}}.
$$
Special conformal invariance 进一步要求两个算子的 scaling dimensions 相同，否则二点函数为零。$\square$

## 练习 19.1

**题目.** 说明 flux quantization 为什么使连续 moduli potential 依赖离散数据。

**解答.** Flux quantization 要求
$$
\int_{\Gamma_p}F_p
$$
落在整数 lattice 中。因此 harmonic flux 不再是任意连续参数，而由整数向量标记。把高维 action 中的 $|F_p|^2$ 在内部空间积分时，所得四维势能依赖 moduli，同时也依赖这些整数 flux quanta。故不同 flux sector 给出不同的离散势能族。$\square$

## 练习 19.2

**题目.** 解释为什么 GVW superpotential 通常先固定 complex structure moduli 而不是 Kahler moduli。

**解答.** GVW superpotential 为
$$
W_{\mathrm{GVW}}=\int_X\Omega\wedge G_3.
$$
其中 $\Omega$ 依赖 complex structure moduli，$G_3=F_3-\tau H_3$ 依赖 axio-dilaton $\tau$ 和 flux quanta。Kahler moduli 不进入 $\Omega$，也不在 tree-level $W_{\mathrm{GVW}}$ 中出现。因此 F-term equations 首先约束 complex structure moduli 与 $\tau$；Kahler moduli 通常需要 nonperturbative effects 或 $\alpha'$ corrections 才能固定。$\square$

## 练习 19.3

**题目.** 用 no-scale identity 说明 tree-level flux potential 为什么不固定 Kahler moduli。

**解答.** 四维 $\mathcal N=1$ F-term potential 为
$$
V=e^K\left(K^{I\bar J}D_IW D_{\bar J}\overline W-3|W|^2\right).
$$
若 tree-level flux superpotential $W$ 不依赖 Kahler moduli $T^a$，则
$$
D_aW=K_aW.
$$
若 Kahler potential 满足 no-scale identity
$$
K^{a\bar b}K_aK_{\bar b}=3,
$$
则 Kahler sector 对势能的贡献
$$
e^K K^{a\bar b}D_aW D_{\bar b}\overline W
=3e^K|W|^2
$$
正好抵消 $-3e^K|W|^2$。因此最低阶势能不依赖这些 Kahler moduli。$\square$

## 练习 20.1

**题目.** 两张平行 D$p$-branes 相距 $L$。从 Dirichlet classical solution
计算其对 $L_0$ 的贡献，并验证最低 GSO-allowed NS vector 的质量为
$L/(2\pi\alpha')$。

**解答.** 取一个 Dirichlet 方向，并令两个端点分别位于 $0,L$。满足边界条件的
零模部分为
$$
X_{\mathrm{cl}}(\sigma)=\frac{L}{\pi}\sigma,\qquad
\partial_\sigma X_{\mathrm{cl}}=\frac{L}{\pi}.
$$
代入 open-string Hamiltonian 或 Virasoro zero mode，得到 stretching contribution
$$
\Delta L_0
=\frac1{4\pi\alpha'}\int_0^\pi
d\sigma\,(\partial_\sigma X_{\mathrm{cl}})^2
=\frac{L^2}{4\pi^2\alpha'}.
$$
因此 NS sector 的物理态条件为
$$
0=L_0-\frac12
=\alpha'k_\parallel^2
+\frac{L^2}{4\pi^2\alpha'}
+N_{\mathrm{osc}}-\frac12.
$$
最低 GSO-allowed vector 有 $N_{\mathrm{osc}}=1/2$。使用
$M^2=-k_\parallel^2$ 得
$$
M^2=\frac{L^2}{4\pi^2\alpha'^2},
\qquad
M=\frac{L}{2\pi\alpha'}.
$$
这也等于 fundamental-string tension
$T_F=(2\pi\alpha')^{-1}$ 乘以弦长 $L$。$\square$

## 练习 20.2

**题目.** 设 genus-$g$ Riemann surface 上的 line bundle $L$ 满足
$\deg L>2g-2$。用 Riemann--Roch 与 Serre duality 证明
$h^0(X,L)=\deg L+1-g$，并解释这个等式为何只固定净手征零模而非完整相互作用。

**解答.** Riemann--Roch 给出
$$
h^0(X,L)-h^1(X,L)=\deg L+1-g.
$$
Serre duality 把第二项写成
$$
h^1(X,L)=h^0(X,K_X\otimes L^{-1}).
$$
右侧 line bundle 的 degree 为
$$
\deg K_X-\deg L=2g-2-\deg L<0.
$$
负次数 holomorphic line bundle 没有非零 holomorphic section，故
$h^1(X,L)=0$，从而
$$
h^0(X,L)=\deg L+1-g.
$$
在内部 fermion 方程由该 Dolbeault complex 实现时，index
$h^0-h^1$ 给出两种 chirality 零模数之差。它不确定 Yukawa couplings、massive
Kaluza--Klein modes、gauge representations 或 quantum corrections；这些还依赖
bundle embedding、作用量和 overlap integrals。因此 index 固定净手征数，不固定
完整低能相互作用。$\square$

## 练习 20.3

**题目.** 从
$F_0^{\mathrm{inst}}(Q)=\operatorname{Li}_3(Q)$ 出发，计算前三次
$t=\log Q$ 导数，并展开 $Q/(1-Q)$ 的前四项；说明这些项如何记录 multiple covers。

**解答.** 因为 $Q=e^t$，有
$$
\frac{d}{dt}\operatorname{Li}_s(Q)
=Q\frac{d}{dQ}\operatorname{Li}_s(Q)
=\operatorname{Li}_{s-1}(Q).
$$
所以
$$
\frac{dF_0^{\mathrm{inst}}}{dt}
=\operatorname{Li}_2(Q),
$$
$$
\frac{d^2F_0^{\mathrm{inst}}}{dt^2}
=\operatorname{Li}_1(Q)=-\log(1-Q),
$$
以及
$$
\frac{d^3F_0^{\mathrm{inst}}}{dt^3}
=\operatorname{Li}_0(Q)
=\frac{Q}{1-Q}
=Q+Q^2+Q^3+Q^4+O(Q^5).
$$
原势
$$
\operatorname{Li}_3(Q)=\sum_{d\ge1}\frac{Q^d}{d^3}
$$
中 $Q^d/d^3$ 是 primitive $\mathbb P^1$ 的 degree-$d$ multiple-cover weight。
三次 $t$ 导数各带来一个 $d$，恰消去 $d^{-3}$，于是 Yukawa coupling 中每个
positive cover degree 的系数均为 $1$。$\square$

## 练习 20.4

**题目.** 直接按 oscillator partitions 计算 $Z_{24}(q)$ 的 $q^3$ 系数，并把
贡献分成 mode partitions $3$、$2+1$ 与 $1+1+1$ 三类。

**解答.** 总 level $3$ 有三种整数分拆。

1. 对 $3$，态为 $\alpha_{-3}^i|0\rangle$，共有 $24$ 个。
2. 对 $2+1$，态为
   $\alpha_{-2}^i\alpha_{-1}^j|0\rangle$。两个 mode numbers 不同，故
   $i,j$ 可独立选择，共 $24^2=576$ 个。
3. 对 $1+1+1$，三个 mode-$1$ bosons 构成 $24$ 维 species space 的三次对称幂，
   维数为
   $$
   \binom{24+3-1}{3}=\binom{26}{3}=2600.
   $$

相加得到
$$
d_3=24+576+2600=3200,
$$
所以
$$
Z_{24}(q)=1+24q+324q^2+3200q^3+O(q^4).
$$
$\square$

## 练习 20.5

**题目.** 逐项指出命题 20.16 的证明在哪些地方使用了离散谱、trace-class
与 Fredholm 假设，并说明非紧 sigma model 的 continuum 为什么可能产生额外边界项。

**解答.** 离散谱与有限重保证每个 $E>0$ eigenspace 都可单独分解，并在有限维空间
内用 $Q+Q^\dagger$ 配对两种 fermion parity。Trace-class 假设保证
$$
\operatorname{Tr}((-1)^Fe^{-\beta H})
$$
收敛，并允许对 eigenspaces 求和以及使用 supertrace 的循环性。Fredholm 性保证
零模有限维、像闭合，使
$\dim\ker H_{\bar0}-\dim\ker H_{\bar1}$ 定义良好且在小变形下稳定。

非紧 sigma model 常有从 $E=0$ 开始或逼近 $E=0$ 的连续谱。此时正能
boson--fermion states 虽在局部配对，其散射相移或谱密度之差仍可在 continuum
端点留下边界贡献；态也可能从 target 的无穷远流入或流出。于是朴素 holomorphic
index 可能不再是 trace-class，需加入 regulator 或非全纯 completion。命题 20.16
因此不能在缺少这些假设时直接套用。$\square$
