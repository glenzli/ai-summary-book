# 附录 B：二维 CFT 公式表和 Virasoro 表示论

## 目标

本附录集中记录 CFT 公式，供第 3、5、8、15 章引用。

## B.1 基本 OPE

$$
T(z)\mathcal O(w)\sim\frac{h\mathcal O(w)}{(z-w)^2}+\frac{\partial\mathcal O(w)}{z-w}.
$$

$$
T(z)T(w)\sim\frac{c/2}{(z-w)^4}+\frac{2T(w)}{(z-w)^2}+\frac{\partial T(w)}{z-w}.
$$

Free boson convention:
$$
X^\mu(z,\bar z)X^\nu(w,\bar w)
\sim-\frac{\alpha'}2\eta^{\mu\nu}\log|z-w|^2,
$$
$$
\partial X^\mu(z)\partial X^\nu(w)
\sim-\frac{\alpha'}2\frac{\eta^{\mu\nu}}{(z-w)^2}.
$$

Free fermion convention:
$$
\psi^\mu(z)\psi^\nu(w)\sim\frac{\eta^{\mu\nu}}{z-w}.
$$

## B.2 Virasoro highest weights

**定义 B.1.** Highest-weight state $|h\rangle$ 满足
$$
L_0|h\rangle=h|h\rangle,\qquad L_n|h\rangle=0\quad(n>0).
$$

Descendants 由 $L_{-n}$ 作用生成。

Virasoro algebra:
$$
[L_m,L_n]=(m-n)L_{m+n}
+\frac{c}{12}m(m^2-1)\delta_{m+n,0}.
$$

Level $1$ Gram matrix:
$$
\langle h|L_1L_{-1}|h\rangle=2h.
$$
Level $2$ basis $\{L_{-2}|h\rangle,L_{-1}^2|h\rangle\}$ 的 Gram matrix 为
$$
\begin{pmatrix}
4h+\frac c2 & 6h\\
6h & 4h(2h+1)
\end{pmatrix}.
$$

## B.3 First-order systems

对 anticommuting $bc$ system，若 $b$ 的 conformal weight 为 $\lambda$，$c$ 的 weight 为 $1-\lambda$，则
$$
c_{bc}=1-3(2\lambda-1)^2.
$$
Reparametrization ghosts 对应 $\lambda=2$，故
$$
c_{bc}=-26.
$$

对 commuting $\beta\gamma$ system，若 $\beta$ 的 weight 为 $\lambda$，$\gamma$ 的 weight 为 $1-\lambda$，则
$$
c_{\beta\gamma}=-1+3(2\lambda-1)^2.
$$
Superconformal ghosts 对应 $\lambda=3/2$，故
$$
c_{\beta\gamma}=11.
$$

## B.4 Characters and modular functions

Dedekind eta function:
$$
\eta(\tau)=q^{1/24}\prod_{n\ge1}(1-q^n),\qquad q=e^{2\pi i\tau}.
$$
其 modular transformations 为
$$
\eta(\tau+1)=e^{\pi i/12}\eta(\tau),\qquad
\eta(-1/\tau)=(-i\tau)^{1/2}\eta(\tau).
$$

Compact boson on circle 的 lattice sum 具有形式
$$
Z_R(\tau,\bar\tau)
=\frac1{|\eta(\tau)|^2}
\sum_{m,n\in\mathbb Z}
q^{\frac{\alpha'}4p_L^2}\bar q^{\frac{\alpha'}4p_R^2}.
$$
在 $R\leftrightarrow \alpha'/R$ 下，$(m,n)$ 交换给出 T-duality invariance。

## B.5 Ward identity template

若 $\mathcal O_i$ 是 primary fields，则
$$
\langle T(z)\prod_i\mathcal O_i(z_i)\rangle
=\sum_i\left(
\frac{h_i}{(z-z_i)^2}
+\frac1{z-z_i}\partial_{z_i}
\right)
\langle\prod_i\mathcal O_i(z_i)\rangle.
$$
这是第三章 contour argument 的标准输出，也是顶点算子权重条件和散射振幅 conformal covariance 的基本工具。
