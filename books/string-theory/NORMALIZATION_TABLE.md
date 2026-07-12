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

上述 mode operators 先定义在有限 $L_0$-level 的 algebraic BRST complex 上。标准
开弦外态取 ghost number $1$；闭弦未积分外态取 total ghost number $2$，并施加
$b_0^-=L_0^-=0$ 的 semi-relative 条件。`$H^\bullet(Q_B)$` 是分次复形，不等同于
未注明 ghost number 与 zero-mode 条件的物理 Hilbert completion。

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

## 8. 散射振幅边界

全动量 incoming，并写
$$
\mathscr A_n
=i(2\pi)^D\delta^{(D)}\!\left(\sum_i k_i\right)\mathcal M_n.
$$
$\mathcal M_n$ 是 reduced amplitude。第六章的 beta/Gamma function 公式固定其
kinematic factor 和 coupling 幂次，但用 $\propto$ 保留 sphere/disk vacuum、外态
LSZ、Chan--Paton trace 与 $g_o^2/g_s$ 的 convention-dependent 常数。Euler/complex
beta integrals 先在绝对收敛域计算，再以亚纯延拓定义物理运动学；这不是 UV cutoff。

## 9. 正规化与渐近方案

| 对象 | 本书采用的方案 | 不可混同的边界 |
|---|---|---|
| 局部 CFT composite operators | OPE point splitting / mode normal ordering | 不构造 Polyakov functional measure |
| oscillator zero-point energy | exponential cutoff 的有限部，等价于 zeta/Hurwitz-zeta bookkeeping | 发散级数不按通常意义收敛；BRST/Lorentz closure 固定 finite subtraction |
| FP 与 worldsheet determinants | heat-kernel/zeta determinant，zero modes 分离 | determinant line、相位和 moduli measure 是路径积分输入 |
| tree moduli integrals | 收敛域积分后亚纯延拓，Lorentzian poles 加 $i0$ | 不等同于 loop IR regularization |
| loop degeneration | modular fundamental domain 加显式 IR cutoff/prescription | modular invariance 不消除 tachyon 或 massless IR divergence |
| sigma-model beta functions | background-field $\alpha'$ expansion，scheme 由局部 field redefinitions 联系 | 高阶系数不是 scheme-independent 收敛级数 |
| low-energy target action | $g_s$ genus expansion 与 $\alpha' E^2$ derivative expansion | supergravity 只是双重渐近截断 |
| holographic on-shell action | radial cutoff 加 local boundary counterterms | finite local terms 改变 contact terms，不改变非局部 scaling |

## 10. 等号与渐近号

- `$=$` 只用于声明的 operator domain、固定 regulator/cutoff 或有限截断内的等式。
- `$\sim$` 必须由正文说明是短距离 OPE、pole leading part、saddle approximation 还是
  large-charge/large-$N$ ratio asymptotic。
- `AdS/CFT`、nonperturbative S/U-duality 与 quantum black-hole entropy matching 保持
  `C`/研究边界状态；free compact-boson T-duality 则是 `E` 类 exact CFT input，二者
  不得因都称“duality”而合并状态。
