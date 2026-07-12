# 第十二章：D-brane 有效理论、DBI 作用量和 Wess-Zumino 耦合

## 本章目标

本章从第七章的 D-brane 边界条件出发，建立 D-brane 的低能有效理论：

1. open-string massless modes 给出 worldvolume gauge fields 与 transverse scalars；
2. Dirac-Born-Infeld action 组织 $F$ 和 embedding 的非线性 $\alpha'$ 修正；
3. Wess-Zumino coupling 表示 D-brane 的 R-R charge；
4. 多个重合 branes 的低能极限给出 nonabelian gauge theory。

## 依赖前置知识

需要第七章 D-brane 定义、第十一章低能有效作用和第九章 type II R-R potentials。

## 12.1 Worldvolume fields

**命题 12.1（重合 branes 上的 gauge field）.** 一叠 $N$ 个重合 D$p$-branes 的低能开弦谱包含 $U(N)$ gauge field $A_a$，以及 $9-p$ 个 adjoint scalar fields $\Phi^i$，描述 brane 的 transverse fluctuations。

**推导说明（标准物理口径）.** 开弦 massless vector 沿 Neumann 方向的极化给出 worldvolume gauge field。沿 Dirichlet 方向的极化不对应 gauge field，而对应端点位置的波动，即 transverse scalar。Chan-Paton labels 使这些 fields 取值于 $N\times N$ 矩阵；重合时开弦 stretched mass 为零，矩阵自由度完整恢复为 $U(N)$ adjoint。$\square$

**定义 12.2（pullback）.** 若 D-brane embedding 为
$$
X^\mu=X^\mu(\xi^a),\qquad a=0,\ldots,p,
$$
则 target tensor $E_{\mu\nu}=g_{\mu\nu}+B_{\mu\nu}$ 的 pullback 为
$$
P[E]_{ab}
=E_{\mu\nu}(X)\partial_aX^\mu\partial_bX^\nu.
$$

## 12.2 DBI action

为避免 $g_s$ 计数重复，本章采用物理张力
$$
\tau_p=\frac1{(2\pi)^p g_s(\alpha')^{(p+1)/2}},
$$
并把 dilaton fluctuation 写成 $\Phi-\Phi_0$，其中 $g_s=e^{\Phi_0}$。

**定义 12.3（DBI action）.** 单个 D$p$-brane 的 string-frame Dirac-Born-Infeld action 为
$$
S_{\mathrm{DBI}}
=-\tau_p\int d^{p+1}\xi\,
e^{-(\Phi-\Phi_0)}
\sqrt{-\det\left(P[g+B]_{ab}+2\pi\alpha'F_{ab}\right)}.
$$

**输入边界.** 该公式是 disk/tree level、单个 Abelian brane 且 fields 缓慢变化时的
标准有效作用。Determinant 重求和无导数的 $\alpha'F$ 幂与 embedding 的一阶导数，
但不包含 $\nabla F$、高 target curvature、massive open strings 或 string loops；
所以它不是完整 open-string effective action 的非微扰定义。

**命题 12.4（Yang-Mills kinetic term）.** 在平坦背景、常 dilaton、static gauge、$B=0$ 和小场强近似下，DBI action 展开为
$$
S_{\mathrm{DBI}}
=-\tau_p\int d^{p+1}\xi
\left[
1+\frac{(2\pi\alpha')^2}{4}F_{ab}F^{ab}
+\frac12\partial_aY^i\partial^aY_i
+\cdots
\right].
$$
因此 worldvolume Yang-Mills coupling 满足
$$
\frac1{g_{\mathrm{YM}}^2}
=\tau_p(2\pi\alpha')^2,
$$
即
$$
g_{\mathrm{YM}}^2
=(2\pi)^{p-2}g_s(\alpha')^{(p-3)/2}.
$$

**证明.** 使用矩阵恒等式
$$
\det(1+M)
=1+\operatorname{tr}M
+\frac12\left((\operatorname{tr}M)^2-\operatorname{tr}M^2\right)+\cdots.
$$
对 antisymmetric $F_{ab}$ 有 $\operatorname{tr}F=0$，二阶项给出 $F_{ab}F^{ab}/4$ 的标准规范化。Static gauge 中 transverse embedding $X^i=Y^i(\xi)$ 使 induced metric 产生 $\partial_aY^i\partial_bY_i$，展开 square root 得 scalar kinetic term。代入 $\tau_p$ 得 $g_{\mathrm{YM}}$ 公式。$\square$

**注 12.5（nonabelian DBI）.** 多个 branes 的非阿贝尔 DBI action 在高阶 commutator 与导数修正下没有简单闭式。本书只使用其低能 Yang-Mills 极限和受保护的 WZ coupling。

## 12.3 Wess-Zumino coupling

**定义 12.6（WZ action）.** D$p$-brane 与 R-R potentials 的基本耦合为
$$
S_{\mathrm{WZ}}
=\mu_p\int_{W_{p+1}}
P\left[\sum_q C_q\right]\wedge e^{B+2\pi\alpha'F}.
$$
在 string-frame R-R potential 的常用规范中
$$
\mu_p=g_s\tau_p
=\frac1{(2\pi)^p(\alpha')^{(p+1)/2}}.
$$
转到 canonical R-R fields 后，BPS brane 的 tension 与 charge 满足相同 supersymmetry bound。

**命题 12.7（lower-dimensional brane charge）.** 若 D$p$-brane worldvolume 上有 gauge flux $F$，则 WZ coupling 中的展开项
$$
\mu_p\int C_{p-1}\wedge (2\pi\alpha'F)
$$
表示该 D$p$-brane 携带 D$(p-2)$-brane charge。

**证明.** 将指数
$$
e^{B+2\pi\alpha'F}
$$
按形式次数展开。要在 $W_{p+1}$ 上积分，总形式次数必须为 $p+1$。$C_{p-1}\wedge F$ 正好是 $(p+1)$-form，因此给出对 $C_{p-1}$ 的电耦合；按 R-R potential 与 brane charge 的定义，这就是 D$(p-2)$ charge。$\square$

**注 12.8（curvature corrections）.** 完整 WZ coupling 含有
$$
\sqrt{\frac{\widehat A(TW)}{\widehat A(NW)}}
$$
等曲率修正，用于 anomaly inflow。本书在第十章和附录 C 中只保留 anomaly cancellation 所需接口。

## 12.4 Open-string effective theory

**命题 12.9（D3-brane 低能极限）.** 在 decoupling/low-energy limit 中忽略
massive open strings、closed-string interactions 与 higher-derivative terms 后，
type IIB 中 $N$ 个重合 D3-branes 的 worldvolume theory 为四维 $\mathcal N=4$
$U(N)$ super Yang--Mills theory，且
$$
g_{\mathrm{YM}}^2=2\pi g_s
$$
在本书 DBI 规范下成立。

**推导说明（标准物理口径）.** 对 $p=3$ 代入命题 12.4 得 gauge coupling。D3-brane 有 $6$ 个 transverse scalars，open superstring massless fermions 与 gauge field 一起组成十维 $N=1$ super Yang-Mills 约化到四维后的 multiplet，即四维 $\mathcal N=4$。$\square$

**注 12.10（规范差异）.** 文献中常见 $g_{\mathrm{YM}}^2=4\pi g_s$，差异来自 gauge kinetic term 写作 $-\frac1{4g^2}\operatorname{Tr}F^2$ 或 generator trace 规范的选择。后续使用 AdS/CFT 时必须同时固定 't Hooft coupling convention。

## 12.5 Nonabelian low-energy limit

**命题 12.11（dimensionally reduced SYM）.** $N$ 个重合 D$p$-branes 的最低阶低能作用量是十维 $U(N)$ super Yang-Mills dimensionally reduced 到 $(p+1)$ 维：
$$
S_{\mathrm{YM}}
=-\frac1{g_{\mathrm{YM}}^2}\int d^{p+1}\xi\,
\operatorname{Tr}\left(
\frac14F_{ab}F^{ab}
+\frac12D_a\Phi^iD^a\Phi_i
-\frac14[\Phi^i,\Phi^j]^2
+\text{fermions}
\right).
$$

**推导说明（标准物理口径）.** 开弦 massless modes 给出 $A_a$、transverse scalars $\Phi^i$ 和 fermions，Chan-Paton factors 使其取值于 $\mathfrak u(N)$。Disk amplitudes 的三点和四点低能极限匹配 Yang-Mills cubic/quartic interactions；T-duality 把高维 gauge field 的内部 components 映为 transverse scalars，从而得到 dimensional reduction 形式。$\square$

**命题 12.12（scalar vev 与 brane separation）.** 对角化的 scalar expectation values 表示 brane 在 transverse directions 的位置。若
$$
\Phi^i=\operatorname{diag}(\phi^i_1,\ldots,\phi^i_N),
$$
则第 $r$ 与第 $s$ 个 brane 的分离满足
$$
Y^i_r-Y^i_s=2\pi\alpha'(\phi^i_r-\phi^i_s).
$$
off-diagonal open strings 的质量与该分离成正比。

**推导说明（标准物理口径）.** T-duality 下 Wilson line eigenvalues 变为 dual circle 上的 brane positions。DBI action 中 static gauge 的 transverse fluctuation 归一化为 $Y^i=2\pi\alpha'\Phi^i$。off-diagonal modes 是连接不同 branes 的开弦，其 classical stretching energy 给出质量项。$\square$

## 12.6 Anomaly inflow 接口

**外部输入定理 12.13（anomaly inflow）.** D-brane worldvolume chiral fields 的 anomaly 可由 bulk R-R/WZ coupling 的 gauge variation 抵消。完整 WZ action 中的
$$
\sqrt{\frac{\widehat A(TW)}{\widehat A(NW)}}
$$
曲率因子正是为匹配 anomaly polynomial 所需。

**使用边界.** 本书不证明 index theorem 形式的 anomaly inflow，只使用其说明 WZ 曲率修正和 D-brane charge quantization 的必要性。

## 本章小结

D-branes 的低能理论由 open-string massless modes 控制。DBI action 给出 Abelian brane 的非线性电磁和几何响应；WZ action 编码 R-R charge 与 lower brane charges。重合 branes 的最低阶理论是 supersymmetric Yang-Mills，这条主线直接通向 AdS/CFT。

## 练习

**练习 12.1.** 将 DBI action 展开到 $F^2$ 阶。

**练习 12.2.** 由 WZ coupling 说明 worldvolume flux 如何诱导 lower-dimensional D-brane charge。

**练习 12.3.** 解释为什么重合 D-branes 的 transverse scalars 取 adjoint 表示，并说明对角 vev 的几何意义。
