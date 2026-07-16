# 核心练习解答

本文件给出每章代表性练习的解答。解答只依赖正文和 A-C 附录中已经定义的对象；若使用外部输入，会明确标出。

## 第 0 章

**练习 0.1.** 谱定理为 `E`，因为本书不证明投影值测度构造。Noether 第一定理为 `P`，第七章给出变分证明。Faddeev-Popov 行列式为 `S`，因为它是无穷维规范轨道上的形式换元推导。$d^2=0$ 为 `P`，第一章用交错偏导证明。

**练习 0.2.** 无穷维空间通常没有与有限维 Lebesgue 测度类似的平移不变局部有限测度。路径积分中的 $\mathcal D\phi$ 是形式符号；只有在 Gaussian 测度、格点 cutoff 或构造性框架中才可严格化。因此“测度平移不变”应标为 `S` 或在具体构造下另证。

## 第 1 章

**练习 1.1.** 对 $\alpha=\alpha_i dx^i$，
$$
d\alpha=d\alpha_i\wedge dx^i
=\partial_j\alpha_i\,dx^j\wedge dx^i
=\frac12(\partial_i\alpha_j-\partial_j\alpha_i)\,dx^i\wedge dx^j.
$$

**练习 1.2.** 对
$$
L=\frac12m g_{ij}(q)\dot q^i\dot q^j-V(q),
$$
有
$$
\frac{\partial L}{\partial\dot q^k}=mg_{kj}\dot q^j,
\quad
\frac{\partial L}{\partial q^k}=\frac12m\partial_kg_{ij}\dot q^i\dot q^j-\partial_kV.
$$
代入 Euler-Lagrange 方程并乘以 $g^{\ell k}$ 得
$$
\ddot q^\ell+\Gamma^\ell_{ij}\dot q^i\dot q^j=-\frac1m g^{\ell k}\partial_kV.
$$

## 第 2 章

**练习 2.1.** 令
$X_f=a^i\partial_{q^i}+b_i\partial_{p_i}$。由
$$
\iota_{X_f}(dq^i\wedge dp_i)=a^i dp_i-b_i dq^i=df
$$
得
$$
a^i=\frac{\partial f}{\partial p_i},\qquad
b_i=-\frac{\partial f}{\partial q^i}.
$$

**练习 2.2.** $SO(3)$ 的无穷小作用为 $\xi_{\mathbb R^3}(q)=\xi\times q$，在余切丛提升后 moment map 满足
$$
\langle\mu(q,p),\xi\rangle=p\cdot(\xi\times q)=\xi\cdot(q\times p).
$$
故 $\mu=q\times p$。

## 第 3 章

**练习 3.1.** 任意 $A\in\mathfrak{so}(3)$ 反对称，可唯一写成 $A_\omega v=\omega\times v$。矩阵对易子满足
$[A_\omega,A_\eta]v=(\omega\times\eta)\times v$，所以 $\mathfrak{so}(3)\cong(\mathbb R^3,\times)$。

**练习 3.2.** 自旋 $j$ 表示中最高权态满足 $J_+|j,j\rangle=0$。由
$J^2=J_-J_+ + J_3(J_3+1)$ 得
$J^2|j,j\rangle=j(j+1)|j,j\rangle$。Casimir 与所有 $J_i$ 对易，因此整个不可约表示上本征值相同。

## 第 4 章

**练习 4.1.** $U(1)$ Abelian，故共轭平凡且 $[A\wedge A]=0$。规范函数 $g=e^{i\lambda}$ 时 $g^{-1}dg=i\,d\lambda$，曲率
$F'=dA'=dA+d(g^{-1}dg)=F$。

**练习 4.2.** $\delta F_A=D_Aa$。于是
$$
\delta S=2\int\operatorname{tr}(D_Aa\wedge *F_A)
=2\int\operatorname{tr}(a\wedge D_A*F_A)
$$
其中使用
$d\operatorname{tr}(a\wedge *F_A)=\operatorname{tr}(D_Aa\wedge *F_A)-\operatorname{tr}(a\wedge D_A*F_A)$，且边界项取零。任意紧支撑 $a$ 给出 $D_A*F_A=0$。

## 第 5 章

**练习 5.1.** $P=-i\,d/dx$ 的对称性依赖分部积分边界项。定义在 $C_c^\infty(\mathbb R)$ 上它对称但不等于其伴随定义域；在 $H^1(\mathbb R)$ 上可得到标准自伴动量。区间上还需边界条件，例如周期边界。

对正文的 $P_\theta$，本征方程 $-if'=\lambda f$ 给出
$f(x)=Ce^{i\lambda x}$。边界条件要求
$$
e^{i\lambda L}=e^{i\theta},
\qquad
\lambda_n=\frac{\theta+2\pi n}{L},
\quad n\in\mathbb Z.
$$
取 $C=L^{-1/2}$ 得归一化本征函数
$f_n(x)=L^{-1/2}e^{i\lambda_nx}$。因此改变 $\theta$ 会把整个整数动量格平移 $\theta/L$；微分表达式不变，而谱随自伴定义域改变。

**练习 5.2.** 按定义
$$
\langle\widehat\delta,\varphi\rangle
=\langle\delta,\widehat\varphi\rangle
=\widehat\varphi(0)
=\int_{\mathbb R^n}\varphi(k)\,d^nk
$$
对每个 $\varphi\in\mathcal S(\mathbb R^n)$ 成立。因此 $\widehat\delta$ 是常函数 $1$ 所定义的 tempered distribution。

**练习 5.3.** 对任意 $f\in L^2(X,\mu)$，
$$
\begin{aligned}
E_{M_a}(\Delta_1)E_{M_a}(\Delta_2)f
&=\mathbf1_{a^{-1}(\Delta_1)}
\mathbf1_{a^{-1}(\Delta_2)}f\\
&=\mathbf1_{a^{-1}(\Delta_1\cap\Delta_2)}f
=E_{M_a}(\Delta_1\cap\Delta_2)f.
\end{aligned}
$$
若 $\|f\|_2=1$，谱测量落在 Borel 集 $\Delta$ 的概率为
$$
\langle f,E_{M_a}(\Delta)f\rangle
=\int_{a^{-1}(\Delta)}|f(x)|^2\,d\mu(x).
$$

## 第 6 章

**练习 6.1.** 以下等式都在题设的共同不变核上理解。由 $[p^2,q]=-2ip$ 与 $[V(q),p]=iV'(q)$，Heisenberg 方程给出
$$
\dot q=i[H,q]=p/m,\qquad
\dot p=i[H,p]=-V'(q).
$$
对留在该核内的可微演化态取期望，得
$d\langle q\rangle/dt=\langle p\rangle/m$，
$d\langle p\rangle/dt=-\langle V'(q)\rangle$。

**练习 6.2.** Pauli 矩阵满足 $[\sigma_i,\sigma_j]=2i\epsilon_{ijk}\sigma_k$。令 $J_i=\sigma_i/2$，则 $[J_i,J_j]=i\epsilon_{ijk}J_k$。

## 第 7 章

**练习 7.1.** 去掉共同的无穷小参数后，$\delta\phi=i\phi$、$\delta\phi^*=-i\phi^*$。把 $\phi$ 与 $\phi^*$ 当作独立场变量，题设 Lagrange 密度给出
$$
\frac{\partial\mathcal L}{\partial(\partial_\mu\phi)}=-\partial^\mu\phi^*,
\qquad
\frac{\partial\mathcal L}{\partial(\partial_\mu\phi^*)}=-\partial^\mu\phi.
$$
由于 $B^\mu=0$，Noether 公式严格给出
$$
j^\mu=i\left(\phi\partial^\mu\phi^*-\phi^*\partial^\mu\phi\right)
$$
并在 Klein--Gordon 方程壳上满足 $\partial_\mu j^\mu=0$。

**练习 7.2.** $S=-\frac14\int F_{\mu\nu}F^{\mu\nu}d^dx$，$\delta F_{\mu\nu}=\partial_\mu\delta A_\nu-\partial_\nu\delta A_\mu$。分部积分得
$$
\partial_\mu F^{\mu\nu}=0.
$$
规范变换 $A\mapsto A+d\lambda$ 下 $F\mapsto dA+d^2\lambda=F$。

## 第 8 章

**练习 8.1.** 归一化矩母函数为
$$
M(J)=\frac{Z(J)}{Z(0)}=e^{J^TCJ/2},
\qquad C=A^{-1}.
$$
一次求导得 $\partial_iM=(CJ)_iM$；再求导，
$$
\partial_j\partial_iM
=C_{ij}M+(CJ)_i(CJ)_jM.
$$
令 $J=0$，得到
$\mathbb E_C[x_ix_j]=\partial_j\partial_iM(0)=C_{ij}$。

**练习 8.2.** 四个指标的配对只有
$(12)(34)$、$(13)(24)$、$(14)(23)$，所以
$$
\mathbb E_C[x_1x_2x_3x_4]
=C_{12}C_{34}+C_{13}C_{24}+C_{14}C_{23}.
$$
若 $C=I_4$ 且四个下标互异，三个乘积都为零，故该矩为零。更一般地，重复下标时应直接代入 Kronecker delta；例如
$\mathbb E[x_1^2x_2^2]=1$（$1\ne2$）。

**练习 8.3.** 对 $p^2<M^2$，
$$
\frac1{M^2+p^2}
=\frac1{M^2}-\frac{p^2}{M^4}
+\frac{p^4}{M^6}
+O\!\left(\frac{p^6}{M^8}\right).
$$
令 $f=\phi^2$。Fourier 反演与分部积分把 $p^4$ 项变成
$\langle f,(\partial^2)^2f\rangle=\int[\partial^2(\phi^2)]^2$，故下一个有效作用项为
$$
-\frac{\kappa^2}{2M^6}
\int[\partial^2(\phi^2)]^2.
$$
相对于无导数的 $M^{-2}\phi^4$ 项，它是 $(E/M)^4$ 阶；若保留到这里，下一项相对为 $O((E/M)^6)$。

**练习 8.4.** 固定裸参数求导：
$$
\begin{aligned}
0
&=\left.\mu\frac d{d\mu}
\left(Z_\phi^{n/2}G_R^{(n)}\right)\right|_{\rm bare}\\
&=Z_\phi^{n/2}
\left[
\mu\frac{\partial}{\partial\mu}
+\left.\mu\frac{dr^a}{d\mu}\right|_{\rm bare}
\frac{\partial}{\partial r^a}
+\frac n2\left.\mu\frac{d\log Z_\phi}{d\mu}\right|_{\rm bare}
\right]G_R^{(n)}.
\end{aligned}
$$
除以 $Z_\phi^{n/2}$ 并代入 $\beta^a,\gamma_\phi$ 的定义，即得
$$
\left(\mu\partial_\mu+\beta^a\partial_{r^a}
+n\gamma_\phi\right)G_R^{(n)}=0.
$$
若改定义 $\widetilde\gamma_\phi=-\gamma_\phi$，最后一项相应写成
$-n\widetilde\gamma_\phi$；这是约定变化，不是物理差异。

## 第 9 章

**练习 9.1.** 对自由真空，在算符值分布意义有
$$
\langle\Omega,\phi(x)\phi(y)\Omega\rangle
=\int\frac{d^{d-1}\mathbf p}{(2\pi)^{d-1}}\frac1{2E_{\mathbf p}}
e^{-iE_{\mathbf p}(x^0-y^0)+i\mathbf p\cdot(\mathbf x-\mathbf y)}
$$
其中 $E_{\mathbf p}=\sqrt{|\mathbf p|^2+m^2}$。

该表达式先作为 tempered distribution 理解：与
$f(x)g(y)\in\mathcal S(\mathbb R^{2d})$ 配对后，时空 Fourier 变换限制到正能量质量壳，Schwartz 衰减使质量壳积分收敛。未配对的点场乘积不是本书声称有定义的算符。

**练习 9.2.** 若 $x-y$ 类空间，可取正时向 Lorentz 变换 $\Lambda$ 使
$(\Lambda x)^0=(\Lambda y)^0$。标量场协变性给出
$$
[\phi(x),\phi(y)]
=U(\Lambda)^{-1}
[\phi(\Lambda x),\phi(\Lambda y)]U(\Lambda).
$$
等时场对易子由命题 9.2 为零，因此原对易子也为零。严格地说，应先取支撑彼此类空间分离的测试函数并使用协变性；这避免把点场当成普通算符。

## 第 10 章

**练习 10.1.** $\delta S=-\frac12\int F^{\mu\nu}(\partial_\mu\delta A_\nu-\partial_\nu\delta A_\mu)d^dx
=-\int F^{\mu\nu}\partial_\mu\delta A_\nu d^dx$。分部积分得
$\int(\partial_\mu F^{\mu\nu})\delta A_\nu d^dx$，故方程为 $\partial_\mu F^{\mu\nu}=0$。

**练习 10.2.** 无穷小规范变换为 $\delta_\epsilon A=D_A\epsilon$。BRST 变换写作 $sA=D_Ac$，形式上就是把偶的规范参数 $\epsilon$ 换成奇的 ghost 场 $c$，并配合 $sc=-\frac12[c,c]$ 保证幂零性。
