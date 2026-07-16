# 附录 B：范畴、同调与指标工具

现代数学物理经常把“对象及其保持结构的映射”放在同一层级处理。规范场的关联丛、BRST 复形、特征类和指标定理都需要最低限度的范畴与同调语言。本附录只提供正文使用的工具。

## 范畴语言

**定义 B.1.** 范畴 $\mathcal C$ 由对象、态射、态射复合和单位态射组成，并满足结合律与单位律。

**定义 B.2.** 函子 $F:\mathcal C\to\mathcal D$ 把对象和态射送到对象和态射，并保持复合与单位。

**例子 B.3.** 向量空间与线性映射构成范畴；光滑流形与光滑映射构成范畴；主 $G$-丛与丛同态也可形成相应范畴。

## 链复形和同调

**定义 B.4.** 一个向量空间链复形 $(C_\bullet,\partial)$ 是一族向量空间 $C_n$ 与线性映射 $\partial_n:C_n\to C_{n-1}$，满足 $\partial_{n-1}\partial_n=0$。第 $n$ 个同调空间为
$$
H_n(C)=\ker\partial_n/\operatorname{im}\partial_{n+1};
$$
商空间有定义是因为 $\partial^2=0$ 给出 $\operatorname{im}\partial_{n+1}\subseteq\ker\partial_n$。链映射 $f:C_\bullet\to D_\bullet$ 是一族线性映射 $f_n:C_n\to D_n$，满足
$$
\partial_n^D f_n=f_{n-1}\partial_n^C.
$$

**命题 B.1 (`P`).** 链映射诱导同调映射 $H_n(f):H_n(C)\to H_n(D)$。

**证明.** 对 cycle $x\in\ker\partial_C$，链映射条件给出
$\partial_Df_n(x)=f_{n-1}(\partial_Cx)=0$，故 $f_n(x)$ 仍是 cycle。定义
$$
H_n(f)([x])=[f_n(x)].
$$
若 $[x]=[x']$，则 $x-x'=\partial_Cy$，从而
$f_n(x)-f_n(x')=f_n(\partial_Cy)=\partial_Df_{n+1}(y)$；两个像相差 boundary，所以定义与代表元无关。线性来自 $f_n$ 的线性。若 $g:D_\bullet\to E_\bullet$ 也是链映射，则代表元计算给出
$H_n(g\circ f)=H_n(g)\circ H_n(f)$，恒等链映射诱导恒等映射。因此同调映射不仅存在，而且满足函子性。$\square$

## Hodge 与指标

**定理 B.2 (`E`, Hodge 定理).** 闭定向 Riemann 流形上，每个 de Rham 上同调类有唯一 harmonic 代表。

**外部输入边界.** 正文只用它解释自由场零模、Betti 数和 Laplacian 谱的关系，不证明椭圆正则性；定位见 [SOURCES.md](SOURCES.md) 的 `E-B.2`。

**定义 B.5.** 设 $V,W$ 为 Banach 空间。Fredholm 算符 $D:V\to W$ 是核与余核有限维且像闭的有界线性算符。其指标为
$$
\operatorname{ind}D=\dim\ker D-\dim\operatorname{coker}D.
$$

**外部输入 B.3 (`E`).** 闭流形上的椭圆微分算子延拓为适当 Sobolev 空间之间的 Fredholm 算符；若流形有边界，则还必须指定椭圆边界条件。Fredholm 指标在 Fredholm 算符的范数连续变形下局部不变。

**外部输入边界.** 第十章的异常-指标接口只使用指标稳定性和 Atiyah--Singer 公式的陈述，不展开椭圆估计；定位见 [SOURCES.md](SOURCES.md) 的 `E-B.3`。

## BRST 复形口径

**定义 B.6.** 一个 BRST 复形由分次向量空间 $\mathcal V^\bullet$ 和 degree $+1$ 的微分 $Q$ 组成，满足 $Q^2=0$。其第 $k$ 个上同调为
$$
H^k_Q(\mathcal V)=
\frac{\ker(Q:\mathcal V^k\to\mathcal V^{k+1})}
{\operatorname{im}(Q:\mathcal V^{k-1}\to\mathcal V^k)}.
$$
物理态候选通常取指定 ghost 数上的 BRST 上同调。

**边界说明.** 本书只使用局部 BRST 幂零性和同调语言解释规范冗余；不证明完整 gauge-fixed Hilbert 空间上的物理态定理。
