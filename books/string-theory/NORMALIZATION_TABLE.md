# 全书归一化总表

本文固定 string theory 全书使用的 convention。任何涉及作用量、OPE、Virasoro 代数、质量公式、BRST 或散射振幅的段落必须与本文相容。

## 1. 时空与世界面号差

| 对象 | 默认 convention |
|---|---|
| target metric | $\eta_{\mu\nu}=\operatorname{diag}(-,+,\ldots,+)$ |
| Lorentzian worldsheet | $\eta_{ab}=\operatorname{diag}(-,+)$ |
| closed string coordinate | $\sigma\sim\sigma+2\pi$ |
| open string coordinate | $\sigma\in[0,\pi]$ |

## 2. 张力与 Polyakov 作用量

String tension 和 Regge slope 的关系为
$$
T=\frac{1}{2\pi\alpha'}.
$$
Lorentzian Polyakov 作用量取
$$
S_P=-\frac{1}{4\pi\alpha'}\int_\Sigma d^2\sigma\sqrt{-h}\,h^{ab}\partial_aX^\mu\partial_bX_\mu.
$$
Euclidean worldsheet 下取
$$
S_E=\frac{1}{4\pi\alpha'}\int_\Sigma d^2\sigma\sqrt{h}\,h^{ab}\partial_aX^\mu\partial_bX_\mu.
$$

## 3. OPE 与 propagator

在平面复坐标中，free boson OPE 为
$$
X^\mu(z,\bar z)X^\nu(w,\bar w)
\sim
-\frac{\alpha'}{2}\eta^{\mu\nu}\log|z-w|^2.
$$
因此
$$
\partial X^\mu(z)\partial X^\nu(w)
\sim
-\frac{\alpha'}{2}\frac{\eta^{\mu\nu}}{(z-w)^2}.
$$

## 4. Virasoro 代数

Virasoro generators 满足
$$
[L_m,L_n]=(m-n)L_{m+n}+\frac{c}{12}m(m^2-1)\delta_{m+n,0}.
$$
Matter free bosons 给出 $c=D$。Reparametrization ghosts 给出 $c_{bc}=-26$。

## 5. 质量公式

开弦在本书归一化下满足
$$
M^2=\frac{1}{\alpha'}(N-a).
$$
闭弦满足
$$
M^2=\frac{4}{\alpha'}(N-a)=\frac{4}{\alpha'}(\tilde N-a),
\qquad
N=\tilde N.
$$
等价地，在 level matching 成立时可写为
$$
M^2=\frac{2}{\alpha'}(N+\tilde N-2a).
$$
玻色弦临界维数 $D=26$ 时 $a=1$。

## 6. BRST

BRST charge 记为 $Q_B$。物理态定义为
$$
Q_B|\psi\rangle=0,\qquad
|\psi\rangle\sim|\psi\rangle+Q_B|\chi\rangle.
$$
Nilpotency $Q_B^2=0$ 要求总 central charge 为零。

## 7. D-brane tension 与 DBI convention

Type II BPS D$p$-brane 的物理张力记为
$$
\tau_p=\frac1{(2\pi)^p g_s(\alpha')^{(p+1)/2}}.
$$
若 $g_s=e^{\Phi_0}$，DBI action 写为
$$
S_{\mathrm{DBI}}
=-\tau_p\int d^{p+1}\xi\,
e^{-(\Phi-\Phi_0)}
\sqrt{-\det(P[g+B]+2\pi\alpha'F)}.
$$
这样在常 dilaton 背景中不会重复计算 $g_s^{-1}$。
WZ coupling 中若使用 string-frame R-R potentials，则
$$
\mu_p=g_s\tau_p
=\frac1{(2\pi)^p(\alpha')^{(p+1)/2}}.
$$
