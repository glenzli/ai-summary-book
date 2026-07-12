# 习题答案与提示

本文件给出各章习题的关键步骤。读者应先独立完成计算，再对照这里检查符号和逻辑。

## 第零章

1. 设 $v'^\mu=(A^{-1})^\mu{}_\nu v^\nu$，$w'_\mu=A^\rho{}_\mu w_\rho$，则
   $$w'_\mu v'^\mu=A^\rho{}_\mu(A^{-1})^\mu{}_\nu w_\rho v^\nu=w_\nu v^\nu.$$
2. $p^\mu=(E,\mathbf p)$，$p_\mu=(-E,\mathbf p)$，故 $p_\mu p^\mu=-E^2+\mathbf p^2=-m^2$。恢复单位为 $E^2=p^2c^2+m^2c^4$。
3. $L=\frac12m\dot q^2-V(q)$，Euler-Lagrange 方程给出 $m\ddot q+\partial_qV=0$。
4. 对 $\phi$ 变分并分部积分得 $(\Box-m^2)\phi=0$，其中 $\Box=\partial_\mu\partial^\mu$。
5. 类光曲线满足 $ds^2=0$，因此沿曲线 $d\tau^2=-ds^2=0$，无法提供单调参数。

## 第一章

1. 由 $\Lambda^T\eta\Lambda=\eta$ 取行列式：
   $$(\det\Lambda)^2\det\eta=\det\eta,$$
   所以 $(\det\Lambda)^2=1$，即 $\det\Lambda=\pm1$。
2. 类时矢量 $V^\mu$ 满足 $V^2<0$。令三速度 $\mathbf{v}=\mathbf{V}/V^0$，则 $|\mathbf{v}|<1$。作速度为 $\mathbf{v}$ 的 boost，可使空间分量消失。
3. $u^\mu=\gamma(1,\mathbf{v})$，故
   $$u^\mu u_\mu=-\gamma^2+\gamma^2\mathbf{v}^2=-\gamma^2(1-\mathbf{v}^2)=-1.$$
4. Lorentz 变换保持内积，故若 $k^2=0$，则 $k'^2=k^2=0$。

## 第二章

1. 将 $v$ 换成 $-v$ 得逆变换：
   $$t=\gamma(t'+vx'),\qquad x=\gamma(x'+vt').$$
2. 由 $y'=y$ 和 $dt'=\gamma(dt-vdx)$，
   $$u'_y=\frac{dy}{\gamma(dt-vdx)}=\frac{u_y}{\gamma(1-vu_x)}.$$
3. 若 $u=\tanh\phi_u$，$v=\tanh\phi_v$，则
   $$\tanh(\phi_u-\phi_v)=\frac{u-v}{1-uv}.$$
4. 恒固有加速度世界线给出
   $$v=\frac{dx/d\tau}{dt/d\tau}=\tanh(\alpha\tau).$$
   又 $t=\alpha^{-1}\sinh(\alpha\tau)$，故
   $$v(t)=\frac{\alpha t}{\sqrt{1+\alpha^2t^2}}.$$
5. 在两端同地惯性系中，任意非直线路径固有时为 $\int\sqrt{1-v^2}\,dt\le\int dt$，等号只在 $v=0$ 时取得。

## 第三章

1. 由 $E=\gamma m$，$\mathbf{p}=\gamma m\mathbf{v}$，直接得
   $$\mathbf{v}=\frac{\mathbf{p}}{E}.$$
2. 无质量粒子 $E^2=\mathbf{p}^2$，Hamilton 速度满足 $|\mathbf{v}|=|\partial E/\partial\mathbf{p}|=1$。
3. $L=-m\sqrt{1-v^2}$，$\mathbf{p}=\gamma m\mathbf{v}$。Hamilton 量
   $$H=\mathbf{p}\cdot\mathbf{v}-L=\gamma m.$$
4. 静止系中 $u^\mu=(1,0,0,0)$，代入
   $$T^{\mu\nu}=(\rho+p)u^\mu u^\nu+p\eta^{\mu\nu}$$
   得 $T^{00}=\rho$，$T^{ij}=p\delta^{ij}$。
5. 质心系中 $p_1=(E,\mathbf{p})$，$p_2=(E,-\mathbf{p})$，故 $P=(2E,0)$，$s=-(P^2)=4E^2$。

## 第四章

1. 采用正文约定：
   $$
   F^{\mu\nu}=
   \begin{pmatrix}
   0&E_x&E_y&E_z\\
   -E_x&0&B_z&-B_y\\
   -E_y&-B_z&0&B_x\\
   -E_z&B_y&-B_x&0
   \end{pmatrix}.
   $$
2. 对 $\partial_\mu F^{\nu\mu}=j^\nu$ 取 $\partial_\nu$，利用偏导交换和 $F^{\nu\mu}$ 反对称，得到 $\partial_\nu j^\nu=0$。
3. $F_{\mu\nu}$ 是二阶张量，完全缩并 $F_{\mu\nu}F^{\mu\nu}$ 是标量。
4. $u_\mu F^\mu{}_\nu u^\nu=F_{\mu\nu}u^\mu u^\nu=0$，因为 $F_{\mu\nu}$ 反对称而 $u^\mu u^\nu$ 对称。
5. 写 $F^{\nu\mu}=\partial^\nu A^\mu-\partial^\mu A^\nu$，代入本书约定的 Maxwell 方程 $\partial_\mu F^{\nu\mu}=j^\nu$ 得
   $$\partial^\nu(\partial_\mu A^\mu)-\Box A^\nu=j^\nu.$$
   Lorenz 规范下第一项为零，因此 $\Box A^\nu=-j^\nu$。

## 第五章

1. 见第三章：$L=-m\sqrt{1-v^2}$，$\partial L/\partial v^i=\gamma mv^i$。
2. 对 $q\int A_\mu dx^\mu$ 变分，积分分部后得到 $q(\partial_\mu A_\nu-\partial_\nu A_\mu)u^\nu=qF_{\mu\nu}u^\nu$。
3. 对标量场作用量分部积分，任意 $\delta\phi$ 的系数为 $(\Box-m^2)\phi$。
4. 代入 Noether 公式：
   $$T^\mu{}_\nu=-\partial^\mu\phi\,\partial_\nu\phi-\delta^\mu{}_\nu\mathcal L$$
   按 Lagrangian 号差可等价调整整体写法，关键是与守恒律一致。
5. 规范变换改变作用量端点项 $q[\chi(B)-\chi(A)]$，端点固定时不改变 Euler-Lagrange 方程。

## 第六章

1. 球面非零 Christoffel：
   $$\Gamma^\theta{}_{\phi\phi}=-\sin\theta\cos\theta,\qquad
   \Gamma^\phi{}_{\theta\phi}=\Gamma^\phi{}_{\phi\theta}=\cot\theta.$$
2. 将 Levi-Civita 公式代入 $\partial_\rho g_{\mu\nu}-\Gamma^\sigma{}_{\rho\mu}g_{\sigma\nu}-\Gamma^\sigma{}_{\rho\nu}g_{\mu\sigma}$，三项抵消。
3. 对标量 $\nabla_\mu\phi=\partial_\mu\phi$，二阶中 Christoffel 下指标对称，故交换 $\mu,\nu$ 不变。
4. 坐标变换下 $\Gamma$ 含二阶坐标导数项，因此不按张量变换。
5. 平直 Minkowski 坐标中 $\Gamma=0$，故 $R^\rho{}_{\sigma\mu\nu}=0$。

## 第七章

1. 对 $S=-m\int ds$ 使用 Euler-Lagrange 方程，并取仿射参数，得到 $\ddot{x}^\mu+\Gamma^\mu{}_{\rho\sigma}\dot{x}^\rho\dot{x}^\sigma=0$。
2. 若 $\lambda'=a\lambda+b$，二阶导数和速度二次项同乘 $a^{-2}$，方程保持。
3. 正规坐标使 $\partial_\rho g_{\mu\nu}(p)=0$，Levi-Civita 公式给出 $\Gamma(p)=0$。
4. 测地线偏离方程右侧含 Riemann 张量，说明不能由局部坐标变换消去的相对加速度就是潮汐效应。
5. 静态度规中 $\omega=E/\sqrt{-g_{tt}}$，弱场 $g_{tt}=-(1+2\Phi)$ 展开即可。

## 第八章

1. $g^{\mu\nu}G_{\mu\nu}=R-\frac12R g^{\mu\nu}g_{\mu\nu}=R-2R=-R$。
2. Ricci 张量只是 Riemann 的缩并。真空 Schwarzschild 有 $R_{\mu\nu}=0$，但 Weyl 曲率非零。
3. 展开 $\nabla_\mu(T^{\mu\nu}\xi_\nu)$，用 $\nabla_\mu T^{\mu\nu}=0$ 和 Killing 方程。
4. $T^{(\Lambda)}_{\mu\nu}=-(\Lambda/8\pi G)g_{\mu\nu}$。在共动系中 $T_{00}=\rho_\Lambda$，$T_{ij}=p_\Lambda g_{ij}$，得 $p_\Lambda=-\rho_\Lambda$。
5. 要使 $G_{tt}\approx2\nabla^2\Phi$ 与 $\nabla^2\Phi=4\pi G\rho$ 一致，右侧系数必须为 $8\pi G$。

## 第九章

1. $t,\phi$ 不显含于 Lagrangian，故共轭动量守恒，得到 $E=(1-2GM/r)\dot t$ 和 $\ell=r^2\dot\phi$。
2. 将守恒量代入归一化条件 $u^2=-1$，整理为 $\dot r^2+V_{\rm eff}=E^2$。
3. 类光有效势 $V=(1-2GM/r)\ell^2/r^2$，求 $dV/dr=0$ 得 $r=3GM$。
4. 轨道方程多出 $3GMu^2$，破坏 Newton 椭圆的闭合性，导致近日点进动。
5. 恢复单位：$\Delta\phi=4GM/(bc^2)$。

## 第十章

1. $K=48G^2M^2/r^6$ 在 $r=2GM$ 有限，在 $r=0$ 发散。
2. 用 $v=t+r_*$，且 $dr_*=(1-2GM/r)^{-1}dr$，代入 Schwarzschild 度规整理。
3. 是否能到达未来无穷远依赖整个时空的未来结构，因此事件视界是全局概念。
4. $r_s=2GM/c^2$。
5. Kerr 有角动量、能层和框架拖曳；Schwarzschild 静态且球对称。

## 第十一章

1. $\dot\rho+3H(1+w)\rho=0$，积分得 $\rho\propto a^{-3(1+w)}$。
2. 尘埃平直无 $\Lambda$：$H^2\propto a^{-3}$，故 $a(t)\propto t^{2/3}$。
3. 辐射平直无 $\Lambda$：$H^2\propto a^{-4}$，故 $a(t)\propto t^{1/2}$。
4. 相邻波峰沿 null geodesic 传播，共动距离相同，得 $\Delta t_0/a(t_0)=\Delta t_e/a(t_e)$，故 $1+z=a_0/a_e$。
5. 若把正宇宙学常数吸收到流体，
   $\rho_\Lambda=\Lambda/(8\pi G)$、$p_\Lambda=-\rho_\Lambda$，故
   $\rho_\Lambda+3p_\Lambda=-2\rho_\Lambda<0$。等价地，在显式保留
   $\Lambda$ 且无其他物质时 $\ddot a/a=\Lambda/3>0$；再取 $H>0$
   才称为加速膨胀支。

## 第十二章

1. 展开 Ricci 张量一阶项，定义迹反转扰动并取 Lorenz 规范，得到 $\Box\bar h_{\mu\nu}=-16\pi GT_{\mu\nu}$。
2. 真空中 $\Box\bar h_{\mu\nu}=0$，平面波解 $e^{ik_\mu x^\mu}$ 要求 $k^\mu k_\mu=0$，即以光速传播。
3. $+$ 偏振使 $x$ 方向伸长时 $y$ 方向压缩，半周期后反过来。
4. 质量单极守恒，偶极对应质心运动和动量守恒，不能辐射；主辐射来自四极矩变化。
5. 源需弱场、慢速、孤立，观察点在波区。

## 第十三章

1. Einstein 方程含约束分量，不包含二阶时间导数，限制初始切片上的 $\gamma_{ij},K_{ij}$。
2. 真空时间对称 $K_{ij}=0$，约束化为 ${}^{(3)}R=0$。
3. ADM 质量使用空间无穷远的渐近平直结构定义，对一般宇宙学时空没有该无穷远。
4. WEC 要求所有类时观察者测得能量密度非负；NEC 只要求所有类光方向收缩非负。
5. 测地线不完备表示自由落体观察者或光线在有限仿射参数内无法延拓；曲率发散是常见但更强的现象。

## 第十四章

1. 弱场中高处 $\Phi$ 较大，$d\tau/dt$ 较大，高处钟走得更快。
2. $D_S$ 是观察者到源，$D_{LS}$ 是透镜到源，$D_L$ 通常是观察者到透镜。
3. 后 Newton 小量来自慢速 $v^2/c^2$ 和弱引力势 $GM/(rc^2)$。
4. 并合阶段强场、快速、非线性，不能用线性化或低阶 PN 控制。
5. 例：$ds^2$ 控制因果结构，$p^\mu p_\mu$ 控制质量壳，$F_{\mu\nu}F^{\mu\nu}$ 控制电磁场分类。

## 第十五章

1. 令 $a=0$，则 $\Sigma=r^2$、$\Delta=r^2-2Mr$，交叉项 $dt\,d\phi$ 消失，角向部分为 $r^2d\Omega^2$，径向项变为 $(1-2M/r)^{-1}dr^2$。
2. $\Delta=r^2-2Mr+a^2=0$，解为
   $r_\pm=M\pm\sqrt{M^2-a^2}$。当 $0<|a|<M$ 时两个不同正根分别
   对应外 Killing/事件视界和内 Cauchy 视界；$|a|=M$ 时是单个退化
   重根；$|a|>M$ 无实根。$a=0$ 时虽有代数根 $0,2M$，但 $r=0$
   是 Schwarzschild 曲率奇点，只有 $r=2M$ 是视界。
3. 静止极限面由 $g_{tt}=0$ 给出，即 $\Sigma=2Mr$。代入 $\Sigma=r^2+a^2\cos^2\theta$，解得 $r=M+\sqrt{M^2-a^2\cos^2\theta}$。
4. 对 Killing 向量 $K^\mu$，沿测地线有 $d(K_\mu p^\mu)/d\lambda=p^\mu p^\nu\nabla_\mu K_\nu=0$，因为 $p^\mu p^\nu$ 对称而 $\nabla_{(\mu}K_{\nu)}=0$。
5. 在能层内 $\partial_t$ 不再是类时向量，Killing 能量 $E=-p_t$ 对未来类时动量不必正定，因此可有负能量轨道。

## 第十六章

1. 因为无穷小坐标变换会使 $\delta g_{\mu\nu}\mapsto\delta g_{\mu\nu}-\mathcal L_\xi\bar g_{\mu\nu}$，一部分扰动只是坐标效应。
2. Newton 规范下
   $$ds^2=a^2(\eta)[-(1+2\Phi)d\eta^2+(1-2\Psi)\delta_{ij}dx^idx^j].$$
3. 对连续性方程取时间导数，用 Euler 方程替换 $\dot{\mathbf v}$，再用 Poisson 方程替换 $\nabla^2\Phi$，得到
   $$\ddot\delta+2H\dot\delta-4\pi G\bar\rho\delta=0.$$
4. 物质主导时 $a\propto t^{2/3}$，代入后设 $\delta\propto t^p$，得 $p=2/3,-1$；增长模为 $\delta\propto a$。
5. 声学峰位置和相对高度依赖声速、视界尺度、空间曲率、重子密度、暗物质密度和暗能量，因此可反推宇宙学参数。

## 第十七章

1. 弱场慢速下 Christoffel 主项为 $\Gamma^i{}_{00}\approx\partial_i\Phi$，测地线方程给出 $d^2x^i/dt^2=-\partial_i\Phi$。
2. PPN 中 $\alpha=2(1+\gamma)GM/(bc^2)$。GR 取 $\gamma=1$，得到 $\alpha=4GM/(bc^2)$。
3. 由 $E=-G\mu M/(2r)$ 得 $dE/dr=G\mu M/(2r^2)$。令 $-dE/dt=P$，代入四极矩功率，解得
   $$\frac{dr}{dt}=-\frac{64}{5}\frac{G^3\mu M^2}{c^5r^3}.$$
4. lapse $N$ 描述相邻空间切片之间的固有时间间隔；shift $N^i$ 描述空间坐标线从一片到下一片的横向滑移。
5. 原始 ADM 变量下方程的双曲性和约束传播不够适合稳定演化；BSSN 或 generalized harmonic 变量能改善数值稳定性和规范控制。
