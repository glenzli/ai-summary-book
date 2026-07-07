# 主线例题集

本文档补充正文中的标准计算例题。例题只服务 string theory 主线，不展开成外部学科教材。

## 例题 1：由 Polyakov action 得到波动方程

取 conformal gauge 下的 Lorentzian Polyakov action
$$
S=-\frac1{4\pi\alpha'}\int d\tau d\sigma\,
\eta^{ab}\partial_aX^\mu\partial_bX_\mu.
$$
变分得
$$
\delta S
=\frac1{2\pi\alpha'}\int d\tau d\sigma\,
\delta X_\mu\,\partial_a\partial^aX^\mu
+\text{boundary terms}.
$$
若先忽略边界项，任意 $\delta X$ 给出
$$
\partial_a\partial^aX^\mu=(-\partial_\tau^2+\partial_\sigma^2)X^\mu=0.
$$
边界项决定开弦端点的 Neumann 或 Dirichlet 条件；闭弦则要求周期性消去边界贡献。

## 例题 2：开弦质量公式的最短推导

开玻色弦量子化后
$$
L_0=\alpha'p^2+N
$$
在本书 convention 下等价于物理态条件
$$
(L_0-a)|\phi\rangle=0,\qquad a=1.
$$
由于 $p^2=-M^2$，得到
$$
M^2=\frac1{\alpha'}(N-a).
$$
$N=0$ 给出 tachyon，$N=1$ 给出 massless vector。该计算说明 normal ordering constant 不是可选细节，而是谱的核心输入。

## 例题 3：T-duality 的质量谱检验

圆紧化半径 $R$ 上
$$
p_L=\frac mR+\frac{nR}{\alpha'},\qquad
p_R=\frac mR-\frac{nR}{\alpha'}.
$$
闭弦质量含
$$
M^2=\frac12(p_L^2+p_R^2)+\frac2{\alpha'}(N+\tilde N-2a).
$$
在
$$
R\mapsto \frac{\alpha'}R,\qquad m\leftrightarrow n
$$
下，$p_L$ 不变而 $p_R$ 变号。因此 $p_L^2+p_R^2$ 不变，level matching 也保持不变。这是 T-duality 最直接的谱检验。

## 例题 4：BRST exact state 的退耦

设一个外态为 BRST exact：
$$
|V\rangle=Q_B|\Lambda\rangle.
$$
树图振幅中的对应插入可写为 contour integral：
$$
\langle (Q_B\Lambda)\prod_i V_i\rangle
=
\left\langle \oint j_B\,\Lambda \prod_i V_i\right\rangle.
$$
若其余顶点都是 BRST closed，contour 可从 $\Lambda$ 周围移到其他插入点而不给出贡献。moduli space 边界项在良好因子化和无 anomaly 条件下消失。故 exact state 不改变物理振幅，应在 physical Hilbert space 中商去。

## 例题 5：DBI action 到 Yang-Mills kinetic term

单个 D$p$-brane 的 DBI action 为
$$
S_{DBI}=-\tau_p\int d^{p+1}\xi\,e^{-\Phi}
\sqrt{-\det(G+2\pi\alpha'F)}.
$$
在平坦背景、常 dilaton 和小 $F$ 下，
$$
\det(\eta+2\pi\alpha'F)
=\det\eta\left(1+\frac{(2\pi\alpha')^2}{2}F_{ab}F^{ab}+O(F^4)\right).
$$
因此
$$
S_{DBI}
=-\tau_pe^{-\Phi}\int d^{p+1}\xi
\left(1+\frac{(2\pi\alpha')^2}{4}F_{ab}F^{ab}+O(F^4)\right).
$$
去掉张力常数项后得到 Yang-Mills kinetic term。规范耦合由 $\tau_p$、$g_s=e^\Phi$ 和 $\alpha'$ 的 convention 共同决定。

## 例题 6：Quintic 的 complex-structure moduli 计数

Quintic threefold 是 $\mathbb P^4$ 中五次齐次多项式 $P_5=0$ 的零点。五次单项式数为
$$
\binom{5+4}{4}=126.
$$
整体缩放去掉 $1$ 个参数，$\operatorname{PGL}(5,\mathbb C)$ 坐标变换去掉
$$
5^2-1=24
$$
个参数，故
$$
h^{2,1}=126-1-24=101.
$$
这与正文中 quintic 的 Hodge number 一致。

## 例题 7：AdS scalar 的 scaling dimension

在 $\operatorname{AdS}_{d+1}$ 中，标量场近边界解满足
$$
\phi(z,x)\sim z^{d-\Delta}\phi_0(x)+z^\Delta A(x).
$$
将该形式代入 Klein-Gordon 方程
$$
(\nabla^2-m^2)\phi=0
$$
的 leading radial part，得到
$$
\Delta(\Delta-d)=m^2L^2.
$$
因此
$$
\Delta=\frac d2+\sqrt{\frac{d^2}{4}+m^2L^2}
$$
对应标准量子化下的 CFT operator dimension。

## 例题 8：No-scale potential 的抵消

四维 $\mathcal N=1$ supergravity 标量势为
$$
V=e^K\left(K^{I\bar J}D_IW D_{\bar J}\overline W-3|W|^2\right).
$$
若 Kahler moduli $T^a$ 不出现在 $W$ 中，则
$$
D_aW=K_aW.
$$
若 Kahler potential 满足 no-scale identity
$$
K^{a\bar b}K_aK_{\bar b}=3,
$$
则 Kahler sector 的贡献
$$
e^K K^{a\bar b}K_aK_{\bar b}|W|^2
$$
正好抵消 $-3e^K|W|^2$。因此 tree-level flux superpotential 可固定 complex structure 与 axio-dilaton，却通常不固定 Kahler moduli。
