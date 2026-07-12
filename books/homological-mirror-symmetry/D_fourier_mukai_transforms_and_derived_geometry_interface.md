# 附录 D：Fourier--Mukai transforms 与导出代数几何接口

## D.1 Kernel 与 convolution

**定义 D.1.** 设 $X,Y$ 是 smooth proper finite-type $k$-schemes，
$p_X,p_Y$ 是 $X\times_kY$ 的投影。对
$K\in\operatorname{Perf}(X\times_kY)$，定义
$$
\Phi_K(E)=\mathbf R p_{Y*}
(\mathbf Lp_X^*E\otimes^{\mathbf L}K).
$$
按定理 2.12A，它既表示 triangulated functor，也表示固定 B-side models
之间的 dg quasi-functor。

**命题 D.2（kernel convolution）.** 再设 $Z$ smooth proper finite type，
$L\in\operatorname{Perf}(Y\times_kZ)$。令
$p_{12},p_{23},p_{13}$ 为 $X\times Y\times Z$ 到相应二因子乘积的投影，则
$$
K\star L=\mathbf R p_{13*}
(\mathbf Lp_{12}^*K\otimes^{\mathbf L}\mathbf Lp_{23}^*L)
\in\operatorname{Perf}(X\times_kZ),
\tag{D.1}
$$
并有 enhanced functors 的自然等价
$$
\Phi_L\circ\Phi_K\simeq\Phi_{K\star L}.
\tag{D.2}
$$

**证明.** 为避免投影重名，写
$p_Y^{XY}:X\times Y\to Y$、
$p_Y^{YZ}:Y\times Z\to Y$，其余类似。对
$A=\mathbf L(p_X^{XY})^*E\otimes^{\mathbf L}K$，Cartesian square
$$
\begin{matrix}
X\times Y\times Z&\xrightarrow{p_{12}}&X\times Y\\
\downarrow p_{23}&&\downarrow p_Y^{XY}\\
Y\times Z&\xrightarrow{p_Y^{YZ}}&Y
\end{matrix}
$$
满足 derived base change 的假设：$p_Y^{XY}$ proper，而 over a field 的
投影给出所需 Tor-independence。故
$$
\mathbf L(p_Y^{YZ})^*\mathbf R(p_Y^{XY})_*A
\simeq\mathbf R p_{23*}\mathbf Lp_{12}^*A.
\tag{D.3}
$$
把 (D.3) 代入 $\Phi_L(\Phi_K(E))$，再用 projection formula，得到
$$
\mathbf R(p_Z^{YZ}\circ p_{23})_*
\left(
\mathbf Lq_X^*E\otimes^{\mathbf L}
\mathbf Lp_{12}^*K\otimes^{\mathbf L}\mathbf Lp_{23}^*L
\right),
\tag{D.4}
$$
其中 $q_X:X\times Y\times Z\to X$。另一方面，把 (D.1) 代入
$\Phi_{K\star L}(E)$ 并对 $p_{13}$ 再用 projection formula，得到同一个
表达式，因为
$p_Z^{XZ}\circ p_{13}=p_Z^{YZ}\circ p_{23}$ 且
$p_X^{XZ}\circ p_{13}=q_X$。所有同构对 $E$ 自然，故给出 (D.2)。最后，
$p_{13}$ 是 proper perfect morphism，因而把括号中的 perfect complex 推为
perfect，证明 (D.1) 的值域。证毕。

## D.2 Adjoints

**外部输入定理 D.3（adjoint kernels）.** 设 $X,Y$ 为 smooth proper
$k$-varieties，维数分别为 $m,n$。令
$\sigma:Y\times X\xrightarrow{\sim}X\times Y$ 交换因子，且
$K^\vee=\mathbf R\mathcal Hom(K,\mathcal O_{X\times Y})$。则
$\Phi_K:\operatorname{Perf}(X)\to\operatorname{Perf}(Y)$ 的 right、left
adjoints 分别由 $Y\times X$ 上的 kernels
$$
K^R=\sigma^*(K^\vee\otimes p_X^*\omega_X[m]),
\qquad
K^L=\sigma^*(K^\vee\otimes p_Y^*\omega_Y[n])
\tag{D.5}
$$
给出。

**证明路线（外部输入）.** 先用 tensor--Hom adjunction 把 $K$ 换成
$K^\vee$，再对 proper smooth projections 应用 Grothendieck duality；
$p^!(-)=p^*(-)\otimes\omega_p[\dim p]$ 产生 (D.5) 的 canonical bundle 与
shift。本书不重建 Grothendieck duality。来源：Huybrechts 的
Fourier--Mukai adjoint formulas。

**解释 D.4.** 在 HMS 中，B-side functor 若由 kernel 给出，其 adjunction、
spherical-functor 条件和 twist 可通过 (D.1)、(D.5) 计算。省略 canonical
bundle 或 dimension shift 会给出错误 adjoint。

## D.3 Derived intersections

**定义 D.5.** 对 affine maps
$\operatorname{Spec}A\to\operatorname{Spec}R\leftarrow
\operatorname{Spec}B$，derived fiber product 的函数 dg algebra 是
$$
A\otimes_R^{\mathbf L}B.
$$
其负同调记录 $\operatorname{Tor}^R_i(A,B)$。只有
$\operatorname{Tor}^R_i(A,B)=0$（$i>0$）时，ordinary fiber product
$\operatorname{Spec}(A\otimes_RB)$ 才没有丢失 derived intersection data。

**例 D.6（自交的 Tor 项）.** 令 $R=k[x]$、$A=B=k=R/(x)$。普通 tensor
product 是 $k$，但用 resolution
$[R\xrightarrow{x}R]$ 计算得到
$$
H^0(k\otimes_R^{\mathbf L}k)\cong k,
\qquad H^{-1}(k\otimes_R^{\mathbf L}k)\cong k.
$$
第二个 $k=\operatorname{Tor}_1^R(k,k)$ 是点在 $\mathbb A^1$ 中 derived
self-intersection 的额外方向。用普通 fiber product 代替 derived one 会漏掉
它，也会改变相应 kernel composition 的 endomorphism complex。

**警告 D.7.** 本书不把 derived algebraic geometry 全部纳入主线。只在
kernel base change、derived intersections、matrix factorizations、
singularity categories 或 moduli of objects 需要时使用上述接口；每次仍需
检查 properness、Tor-amplitude 与 enhancement model。

## 本附录小结

Fourier--Mukai transforms 在 B-side 是 enhanced kernel functors。
Convolution 的证明依赖 derived base change 与 projection formula；adjoints
依赖 Grothendieck duality；非横截交必须保留 derived tensor product 中的
Tor 信息。
