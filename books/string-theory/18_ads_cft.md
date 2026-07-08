# 第十八章：AdS/CFT 的精确定式和基本检验

## 本章目标

本章给出 AdS/CFT 的标准例子、参数字典和基本检验，明确区分：

1. brane near-horizon 几何计算；
2. large $N$ 与 large 't Hooft coupling 极限；
3. GKPW dictionary；
4. 已检验的 protected quantities；
5. 尚无完整数学证明的非微扰等价陈述。

## 依赖前置知识

需要第十二章 D-branes、第十四章 dualities、第十七章 brane/black brane 逻辑和第十一章低能 supergravity。

## 18.1 标准对偶

**物理猜想 18.1（AdS/CFT）.** Type IIB string theory on
$$
\operatorname{AdS}_5\times S^5
$$
with $N$ units of self-dual five-form flux 等价于四维 $\mathcal N=4$ supersymmetric Yang-Mills theory with gauge group $\operatorname{SU}(N)$。

**定义 18.2（参数字典）.** 记
$$
\lambda=g_{\mathrm{YM}}^2N
$$
为 't Hooft coupling。AdS 半径 $R$ 满足
$$
\frac{R^4}{\alpha'^2}=4\pi g_sN.
$$
若采用常见 AdS/CFT trace convention $g_{\mathrm{YM}}^2=4\pi g_s$，则
$$
\frac{R^4}{\alpha'^2}=\lambda.
$$
若采用第十二章 DBI trace convention，右侧差一个固定的 trace-normalization 常数；参数依赖不变。

**命题 18.3（large $N$ 与 classical gravity）.** Classical type IIB supergravity 近似要求
$$
N\gg1,\qquad \lambda\gg1,
$$
并且还需 $g_s\sim\lambda/N\ll1$，使 string loop corrections 与 $\alpha'$ corrections 均可忽略。

**证明草图.** String loop expansion 由 $g_s$ 控制，而在固定 $\lambda=g_{\mathrm{YM}}^2N$ 时 $g_s\sim \lambda/N$，故 large $N$ 抑制 loops。曲率半径满足 $R^2/\alpha'\sim \sqrt\lambda$，故 $\lambda\gg1$ 使 AdS 曲率远小于 string scale，从而抑制 $\alpha'$ corrections。$\square$

## 18.2 D3-branes 与 near-horizon limit

**命题 18.4（D3-brane near-horizon geometry）.** $N$ 个重合 D3-branes 的低能闭弦几何在 near-horizon limit 中给出
$$
\operatorname{AdS}_5\times S^5
$$
且半径由 five-form flux number $N$ 决定。

**证明草图.** D3-brane supergravity 解含 harmonic function
$$
H(r)=1+\frac{R^4}{r^4}.
$$
near-horizon limit $r\ll R$ 中 $H(r)\sim R^4/r^4$，metric 化为 $\operatorname{AdS}_5\times S^5$。Flux quantization 给出 $R^4=4\pi g_sN\alpha'^2$。$\square$

**注 18.5（两种低能描述）.** 同一 D3-brane 系统的低能开弦描述是 $\mathcal N=4$ SYM，闭弦 near-horizon 描述是 IIB string on $\operatorname{AdS}_5\times S^5$。AdS/CFT 猜想把这两种描述提升为完整等价。

## 18.3 GKPW dictionary

**原则 18.6（GKPW dictionary）.** Bulk field $\phi$ 的边界值作为 CFT operator $\mathcal O$ 的 source：
$$
Z_{\mathrm{string}}[\phi|_{\partial}=\phi_0]
=
\left\langle
\exp\left(\int_{\partial AdS}\phi_0\mathcal O\right)
\right\rangle_{\mathrm{CFT}}.
$$
Classical supergravity 极限中，
$$
Z_{\mathrm{string}}\approx e^{-S_{\mathrm{sugra,on-shell}}}.
$$

**命题 18.7（bulk scalar mass 与 CFT dimension）.** 在 $\operatorname{AdS}_{d+1}$ 中，标量场质量 $m$ 与 dual scalar primary 的 conformal dimension $\Delta$ 满足
$$
m^2R^2=\Delta(\Delta-d).
$$

**证明草图.** 在 Poincare coordinate
$$
ds^2=R^2\frac{dz^2+dx_idx^i}{z^2}
$$
中，标量方程 near boundary $z\to0$ 的径向主导项给出解
$$
\phi(z,x)\sim z^{d-\Delta}\phi_0(x)+z^\Delta A(x).
$$
代入 Klein-Gordon 方程的 indicial equation 即得 $m^2R^2=\Delta(\Delta-d)$。$\square$

## 18.4 基本检验

**命题 18.8（symmetry matching）.** $\operatorname{AdS}_5\times S^5$ 的 bosonic isometry
$$
SO(4,2)\times SO(6)
$$
匹配四维 $\mathcal N=4$ SYM 的 conformal group 与 R-symmetry group。

**证明.** $\operatorname{AdS}_5$ 的 isometry group 为 $SO(4,2)$，这正是四维 conformal group。$S^5$ 的 isometry group 为 $SO(6)\cong SU(4)$，匹配 $\mathcal N=4$ SYM 的 R-symmetry。$\square$

**注 18.9（protected data）.** Chiral primary dimensions、anomaly coefficients、supersymmetric indices 和某些 Wilson loop 期望值是 AdS/CFT 的重要检验对象。非 protected observables 通常只能在强/弱耦合不同区域分别近似计算。

## 18.5 Two-point function 的标准结构

**命题 18.10（scalar two-point function scaling）.** 若 bulk scalar $\phi$ dual to scalar primary $\mathcal O$，且 conformal dimension 为 $\Delta$，则 CFT two-point function 具有形式
$$
\langle\mathcal O(x)\mathcal O(0)\rangle
=\frac{C_\Delta}{|x|^{2\Delta}}.
$$

**证明.** CFT 中 scalar primary 的 two-point function 由 translation、rotation 和 scale invariance 固定为 $C|x|^{-2\Delta}$；special conformal invariance 排除其他函数形式。AdS 计算通过 GKPW dictionary 给出同一幂次，并固定 normalization convention dependent 的 $C_\Delta$。$\square$

**命题 18.11（bulk on-shell action 给出二点函数）.** Classical supergravity 中，对 quadratic bulk scalar action 求解 Dirichlet problem 并将 on-shell action 对 boundary source $\phi_0$ 二次变分，可得到 $\mathcal O$ 的 connected two-point function。

**证明草图.** GKPW dictionary 给出
$$
W_{\mathrm{CFT}}[\phi_0]=-S_{\mathrm{on-shell}}[\phi_0].
$$
因此
$$
\langle\mathcal O(x)\mathcal O(y)\rangle
=\frac{\delta^2W}{\delta\phi_0(x)\delta\phi_0(y)}.
$$
Bulk-to-boundary propagator 的 near-boundary scaling 由命题 18.7 的 $\Delta$ 决定。$\square$

## 18.6 Wilson loops 与最小曲面

**命题 18.12（Wilson loop 的 string worldsheet 近似）.** 在 large $N$、large $\lambda$ 极限中，fundamental representation Wilson loop 的强耦合期望值由以该 loop 为边界的 fundamental string worldsheet 面积近似：
$$
\langle W(C)\rangle\sim e^{-S_{\mathrm{NG}}(\Sigma_C)}.
$$

**证明草图.** Wilson loop 插入对应边界上的 fundamental string source。Classical string limit 中，path integral 由极小 Nambu-Goto 面主导。严格规范化和 Legendre boundary term 依赖具体 contour。$\square$

## 本章小结

AdS/CFT 是 string theory 最精确的非微扰定义候选之一。标准例子由 D3-branes 的两种低能描述产生；参数字典解释 classical gravity 极限；GKPW dictionary 给出 correlation functions 的操作定义。全局等价在本书中保持为物理猜想。

## 练习

**练习 18.1.** 解释大 $N$ 和大 't Hooft coupling 极限为何对应 classical supergravity。

**练习 18.2.** 推导 $\operatorname{AdS}_{d+1}$ 中标量质量和 CFT scaling dimension 的关系。

**练习 18.3.** 用 conformal invariance 固定 scalar primary two-point function 的幂次。
