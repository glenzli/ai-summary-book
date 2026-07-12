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

“等价”在这里意指选定相容的 global form、AdS boundary conditions、operator/source
字典与 flux sector 后，Hilbert spaces、observables 和 correlation functions 的完整
对应。D3-brane open-string 描述原本给出 $U(N)$；自由 center-of-mass $U(1)$ 在
decoupling limit 中分离，本章讨论 interacting $SU(N)$ sector。不同 global gauge
groups 与 bulk discrete data 需要另行匹配。由于 bulk type-IIB string theory 尚无
独立完成的普适非微扰构造，本猜想不是两个已分别严格定义对象之间的数学同构定理。

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

**命题 18.3（classical supergravity 的参数区）.** 在
$g_{\mathrm{YM}}^2=4\pi g_s$ convention 下，classical type-IIB supergravity 近似要求
$$
\lambda\gg1,
\qquad \frac{\lambda}{N}\ll1,
$$
并要求所研究 observable 不探测 string-scale local structure。

**推导说明（标准物理口径）.** 参数字典逐项给出
$$
\frac{\alpha'}{R^2}=\lambda^{-1/2},
\qquad
g_s=\frac{\lambda}{4\pi N}.
$$
所以 generic higher-derivative expansion 由 $\lambda^{-1/2}$ 组织；type IIB 首个
受保护 $\alpha'^3R^4$ 项相对两导数作用量按 $\lambda^{-3/2}$ 抑制。Closed-string
loop 至少带 $g_s^2\sim\lambda^2/N^2$。因此仅有 $N\gg1$ 而让
$\lambda/N$ 不小，或仅有 $\lambda\gg1$ 而 $N$ 不够大，都不足以得到 classical
supergravity。$\square$

## 18.2 D3-branes 与 near-horizon limit

**外部输入定理 18.4（extremal D3-brane solution 与 flux normalization）.** 十维
type-IIB 两导数 supergravity 存在 $N$ 个重合 extremal D3-branes 的解，其 metric 为
$$
ds^2=H(r)^{-1/2}\eta_{ab}dx^adx^b
+H(r)^{1/2}\left(dr^2+r^2d\Omega_5^2\right),
\qquad
H(r)=1+\frac{R^4}{r^4},
$$
并带 self-dual five-form flux。Flux quantization 固定
$R^4=4\pi g_sN\alpha'^2$。本书引用该 supergravity solution，不重解其 coupled
Einstein--five-form equations。

**计算 18.4A（near-horizon geometry）.** 在几何区域 $r\ll R$，上述外部输入
逐项给出
$$
\operatorname{AdS}_5\times S^5
$$
的共同半径 $R$。事实上
$$
H^{-1/2}\sim\frac{r^2}{R^2},
\qquad
H^{1/2}\sim\frac{R^2}{r^2},
$$
所以
$$
ds^2\sim
\frac{r^2}{R^2}\eta_{ab}dx^adx^b
+\frac{R^2}{r^2}dr^2+R^2d\Omega_5^2,
$$
前两项正是 Poincare patch 的 $\operatorname{AdS}_5$ metric。$\square$

**注 18.4B（几何极限与 decoupling limit）.** $r\ll R$ 只证明 metric 的局部
near-horizon 形式。把 throat excitations 与 asymptotically flat bulk 解耦还使用
$\alpha'\to0$、$U=r/\alpha'$ 与四维能标固定的 decoupling limit。由“两种低能
描述”进一步推出完整对偶，仍是物理猜想 18.1，而不是计算 18.4A 的逻辑推论。

**注 18.5（两种低能描述）.** 同一 D3-brane 系统的低能开弦描述是 $\mathcal N=4$ SYM，闭弦 near-horizon 描述是 IIB string on $\operatorname{AdS}_5\times S^5$。AdS/CFT 猜想把这两种描述提升为完整等价。

## 18.3 GKPW dictionary

**物理猜想 18.6（GKPW dictionary）.** 在物理猜想 18.1 的框架内，bulk field
$\phi$ 的 renormalized boundary datum 作为 CFT operator $\mathcal O$ 的 source：
$$
Z_{\mathrm{string}}[\phi|_{\partial}=\phi_0]
=
\left\langle
\exp\left(\int_{\partial AdS}\phi_0\mathcal O\right)
\right\rangle_{\mathrm{CFT}}.
$$
Classical supergravity 极限中，
$$
Z_{\mathrm{string}}\sim e^{-S_{\mathrm{sugra,ren}}[\phi_0]}.
$$
这里 $S_{\mathrm{sugra,ren}}$ 是加 radial cutoff、局部 boundary counterterms 后取极限
得到的 renormalized on-shell action；裸 $S_{\mathrm{on-shell}}$ 一般发散。等式还依赖
bulk boundary condition、operator normalization 与 quantization choice，不能脱离
AdS/CFT 猜想当成独立证明。

**命题 18.7（AdS scalar 的 indicial roots）.** 在固定 Euclidean
$\operatorname{AdS}_{d+1}$ 上，令 free scalar 满足
$(\nabla^2-m^2)\phi=0$。其边界 indicial exponents 为
$$
\Delta_\pm=\frac d2\pm
\sqrt{\frac{d^2}{4}+m^2R^2},
\qquad
m^2R^2=\Delta_\pm(\Delta_\pm-d).
$$
在 standard quantization 与 GKPW 猜想成立时，dual scalar primary 取
$\Delta=\Delta_+$；允许 alternative quantization 时可取 $\Delta_-$。

**证明.** 在 Poincare coordinates
$$
ds^2=R^2\frac{dz^2+\delta_{ij}dx^idx^j}{z^2}
$$
中直接计算
$$
\nabla^2\phi
=\frac1{R^2}\left[
z^2(\partial_z^2+\partial_i\partial_i)
-(d-1)z\partial_z
\right]\phi.
$$
令 $\phi\sim z^\delta f(x)$ 并只保留 $z\to0$ 的最低径向次幂，则
$$
(\nabla^2-m^2)\phi
\sim R^{-2}\left[\delta(\delta-d)-m^2R^2\right]
z^\delta f(x).
$$
故 indicial equation 为 $\delta(\delta-d)=m^2R^2$，解二次方程即得
$\Delta_\pm$。取 $\Delta=\Delta_+$ 时，两支写成
$$
\phi(z,x)\sim z^{d-\Delta}\phi_0(x)+z^\Delta A(x).
$$
根为实数要求 Breitenlohner--Freedman bound
$m^2R^2\ge-d^2/4$。当
$0<\sqrt{d^2/4+m^2R^2}<1$ 且满足相应 unitarity/boundary conditions 时，两种
quantization 都可能；端点会出现 logarithmic branches，须另行处理。上述代数只
推出 indicial threshold；“满足该 bound 的适当能量泛函给出稳定 evolution”是
`BF82` 的外部谱/边界条件结果。这一证明也不证明 bulk mode 与 CFT operator 的
对偶。$\square$

## 18.4 基本检验

**命题 18.8（symmetry matching 检验）.** $\operatorname{AdS}_5\times S^5$ 的
bosonic isometry Lie algebra
$$
\mathfrak{so}(4,2)\oplus\mathfrak{so}(6)
$$
匹配四维 $\mathcal N=4$ SYM 的 conformal 与 R-symmetry Lie algebras。

**证明.** $\operatorname{AdS}_5$ 的 local isometry algebra 为
$\mathfrak{so}(4,2)$，匹配四维 conformal algebra；$S^5$ 的 isometry algebra 为
$\mathfrak{so}(6)\cong\mathfrak{su}(4)$，匹配 $\mathcal N=4$ SYM 的 R-symmetry。
加入 fermionic generators 后匹配 $\mathfrak{psu}(2,2|4)$。Global covers 与 quotients
需和物理猜想 18.1 的 global data 一起选择。对称性匹配是必要检验，不是对偶性的
充分证明。$\square$

**注 18.9（protected data）.** Chiral primary dimensions、anomaly coefficients、supersymmetric indices 和某些 Wilson loop 期望值是 AdS/CFT 的重要检验对象。非 protected observables 通常只能在强/弱耦合不同区域分别近似计算。

## 18.5 Two-point function 的标准结构

**命题 18.10（CFT scalar two-point function scaling）.** 在 Euclidean CFT 的
translation/rotation/scale invariant vacuum 中，若 $\mathcal O$ 是 scalar primary，
dimension 为 $\Delta$，则对 $x\ne0$，
$$
\langle\mathcal O(x)\mathcal O(0)\rangle
=\frac{C_\Delta}{|x|^{2\Delta}}.
$$

**证明.** Translation invariance 使 correlator 只依赖 $x$，rotation invariance 使其
只依赖 $|x|$。Scale covariance 给出 functional equation
$f(\lambda|x|)=\lambda^{-2\Delta}f(|x|)$，故
$f(r)=C_\Delta r^{-2\Delta}$；special conformal covariance 与此形式相容并要求
两个算子 dimensions 相同。$x=0$ 处可另有 scheme-dependent contact terms。
这个证明完全在 CFT 内，不使用 GKPW，也不固定 $C_\Delta$。$\square$

**推导说明 18.11（条件式 holographic two-point prescription）.** 假设物理猜想
18.6、选定 standard/alternative quantization，并能完成 holographic
renormalization。则对 quadratic bulk scalar action 求解 Dirichlet problem，并将
renormalized on-shell action 对 boundary source $\phi_0$ 二次变分，可得到
$\mathcal O$ 的 connected two-point function。

具体地，在 $z=\varepsilon$ 截断曲面上，利用方程运动后，bulk quadratic action 化为
边界项
$$
S_{\mathrm{reg}}
=\frac12\int_{z=\varepsilon}d^dx\sqrt\gamma\,
\phi\,n^M\partial_M\phi.
$$
其 $\varepsilon\to0$ 展开一般含幂发散与 logarithmic divergence。选取局部
$S_{\mathrm{ct}}[\phi,\gamma]$ 后定义
$$
S_{\mathrm{ren}}
=\lim_{\varepsilon\to0}(S_{\mathrm{reg}}+S_{\mathrm{ct}}).
$$
GKPW dictionary 条件式地给出
$$
W_{\mathrm{CFT}}[\phi_0]=-S_{\mathrm{ren}}[\phi_0].
$$
因此
$$
\langle\mathcal O(x)\mathcal O(y)\rangle
=\frac{\delta^2W}{\delta\phi_0(x)\delta\phi_0(y)}.
$$
Bulk-to-boundary propagator 的 near-boundary scaling 由命题 18.7 的 $\Delta$ 决定，
从而复现命题 18.10 的非局部幂次。$C_\Delta$ 还依赖 bulk kinetic normalization 与
source normalization；有限 local counterterms 只改变 contact terms。该计算检验
GKPW，但不证明猜想本身。$\square$

## 18.6 Wilson loops 与最小曲面

**推导说明 18.12（Wilson loop 的 semiclassical string saddle）.** 条件于
AdS/CFT Wilson-loop dictionary，并在 $\lambda\gg1$、$\lambda/N\ll1$ 且存在主导
光滑 saddle 时，fundamental representation Wilson loop 满足
$$
\log\langle W(C)\rangle
=-S_{\mathrm{NG}}^{\mathrm{ren}}(\Sigma_C)
+O(\lambda^0)+O(g_s^2),
$$
其中 $S_{\mathrm{NG}}^{\mathrm{ren}}\sim\sqrt\lambda$，其边界是 contour $C$。

**推导说明（标准物理口径）.** Dictionary 把 Wilson loop 插入对应到边界上的
fundamental-string source。Classical string limit 中，worldsheet path integral 由
极小 Nambu--Goto 面主导；near-boundary 面积有 perimeter divergence，必须用
radial cutoff 与 boundary/Legendre counterterm 定义
$S_{\mathrm{NG}}^{\mathrm{ren}}$。$O(\lambda^0)$ 来自 worldsheet fluctuations，
$O(g_s^2)=O(\lambda^2/N^2)$ 来自 closed-string handles；若有多个 saddles 还需比较其 action 与 Stokes
phenomena。该式是双重渐近展开，不是有限 $N,\lambda$ 的精确等号。$\square$

## 本章小结

AdS/CFT 是 string theory 最精确的非微扰定义候选之一。标准例子由 D3-branes 的两种低能描述产生；参数字典解释 classical gravity 极限；GKPW dictionary 在猜想成立时给出 correlation functions 的操作 prescription。全局等价在本书中保持为物理猜想。

## 练习

**练习 18.1.** 解释大 $N$ 和大 't Hooft coupling 极限为何对应 classical supergravity。

**练习 18.2.** 推导 $\operatorname{AdS}_{d+1}$ 中标量质量和 CFT scaling dimension 的关系。

**练习 18.3.** 用 conformal invariance 固定 scalar primary two-point function 的幂次。
