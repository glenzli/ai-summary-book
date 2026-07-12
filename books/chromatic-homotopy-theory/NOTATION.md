# 符号与约定

## 0. 集合论和范畴口径

**约定 N.1.** 全书固定 Grothendieck universes
$$
\mathcal U\in\mathcal V\in\mathcal W.
$$
若不特别说明，“小范畴”和“集合”指 $\mathcal U$-小。稳定 infinity-范畴、presentable 范畴和谱范畴在 $\mathcal V$ 中处理。

**约定 N.2.** $\mathbf{Sp}$ 表示谱的稳定 infinity-范畴，$\mathbf{Sp}_{(p)}$ 表示 $p$-局部谱的全子范畴，$\mathbf{Sp}^{\wedge}_p$ 表示 $p$-完备谱的全子范畴。若使用点集模型，例如 symmetric spectra 或 orthogonal spectra，该模型只用于构造，不改变最终等价类。

**约定 N.3.** 全书固定素数 $p$。若某个命题对所有素数成立，会写作“对任意素数 $p$”。若命题依赖 $p\ge 5$、$p>n+1$ 或类似范围，必须在命题名处标出。

## 1. 谱和同调

| 符号 | 含义 |
| --- | --- |
| $\mathbb S$ | 球谱 |
| $X_{(p)}$ | 谱 $X$ 的 $p$-局部化 |
| $X^\wedge_p$ | $p$-完备化，除非说明，否则在谱范畴中取 derived completion |
| $\Sigma^dX$ | $d$ 次悬挂；$d<0$ 时为脱悬挂 |
| $F(X,Y)$ | function spectrum |
| $DX$ | 有限谱 $X$ 的 Spanier-Whitehead dual $F(X,\mathbb S)$ |
| $E_*X$ | $\pi_*(E\otimes X)$ |
| $E^*X$ | $\pi_{-*}F(X,E)$ |

**约定 N.4.** 本书使用 homological grading 书写 $E_*X$。若进入 cohomological grading，会显式说明符号反转。

## 2. Bousfield 局部化

| 符号 | 含义 |
| --- | --- |
| $\langle E\rangle$ | 谱 $E$ 的 Bousfield 类，即所有 $E$-acyclic 谱的等价类 |
| $L_E$ | 关于 $E$ 的 Bousfield localization |
| $C_E$ | $E$-acyclization fiber，$C_EX\to X\to L_EX$ |
| $X$ is $E$-acyclic | $E\otimes X\simeq 0$ |
| $X$ is $E$-local | 对任意 $E$-acyclic $A$，$F(A,X)\simeq 0$ |

**约定 N.5.** 本书不用单个不加说明的偏序符号比较 Bousfield 类。需要比较时直接写：
$$
E\otimes X\simeq 0\Rightarrow F\otimes X\simeq 0
$$
或其反向，避免不同文献中偏序方向不同造成歧义。

## 3. 色层理论

| 符号 | 含义 |
| --- | --- |
| $K(0)$ | $H\mathbb Q$ |
| $K(n)$ | 第 $n$ 个 Morava K-theory，$n\ge 1$ |
| $E(0)$ | $H\mathbb Q$；因此 $L_0$ 为有理化 |
| $E(n)$ | Johnson-Wilson theory，$n\ge1$ |
| $BP$ | $p$-局部 Brown-Peterson spectrum |
| $BP\langle n\rangle$ | truncated Brown-Peterson spectrum |
| $E_n$ | 高度 $n$ 的 Morava E-theory/Lubin-Tate theory |
| $L_n$ | $L_{E(n)}$ |
| $M_nX$ | $\operatorname{fib}(L_nX\to L_{n-1}X)$，其中 $L_{-1}X=0$ |
| $L_{K(n)}$ | Morava $K(n)$-localization |
| $T(n)$ | 某个 type $n$ 有限谱的 $v_n$ self-map 的 telescope 的 Bousfield 类代表 |
| $L_n^f$ | finite/telescopic localization，具体模型需随章节声明 |

**约定 N.6.** 系数环写作
$$
K(n)_*\cong \mathbb F_p[v_n^{\pm 1}],\qquad |v_n|=2(p^n-1).
$$
Johnson-Wilson theory 写作
$$
E(n)_*\cong \mathbb Z_{(p)}[v_1,\ldots,v_n,v_n^{-1}],
$$
其中 $|v_i|=2(p^i-1)$。

**约定 N.7.** Morava E-theory 的 homotopy 系数采用
$$
(E_n)_*\cong W(k)[[u_1,\ldots,u_{n-1}]][u^{\pm 1}],\qquad |u|=2,\quad |u_i|=0,
$$
其中 $k$ 是高度 $n$ 形式群的完美剩余域。若改用 cohomological convention $|u|=-2$，必须在局部说明。

## 4. 有限谱和 type

**定义 N.8.** 非零有限 $p$-局部谱 $X$ 的 type 是首次非消失的
Morava 高度：
$$
\operatorname{type}(X)=\min\{n\ge0\mid K(n)_*X\ne0\}.
$$
finite detection 保证该集合非空。零谱的 type 记作 $\infty$。因此
type $n$ 按定义满足 $K(i)_*X=0$ 对所有 $i<n$，而“所有 $i\ge n$
均非零”是外部高度单调性定理，不是定义。

**约定 N.9.** $\mathcal C_n$ 表示有限 $p$-局部谱的 thick 子范畴
$$
\mathcal C_n=\{X\mid K(i)_*X=0\text{ 对所有 }0\le i<n\},
\qquad n\ge1,
$$
并令 $\mathcal C_0$ 为全部有限 $p$-局部谱、
$\mathcal C_\infty=\{0\}$。接受有限谱高度单调性后，$n\ge1$ 时可等价
写为 $\mathcal C_n=\{X\mid K(n-1)_*X=0\}$。于是
$$
\mathcal C_0\supseteq \mathcal C_1\supseteq \mathcal C_2\supseteq\cdots.
$$

**约定 N.9A.** 对 $n\ge1$，$v_n$-self-map 写作
$v:\Sigma^dF\to F$ 且 $d>0$。高度零单独取
$p:\mathbb S_{(p)}\to\mathbb S_{(p)}$（次数 $0$），其 telescope 为
$T(0)\simeq H\mathbb Q$。

## 5. 群和 descent

| 符号 | 含义 |
| --- | --- |
| $\mathbb G_n$ | extended Morava stabilizer group |
| $\mathbb S_n$ | height $n$ Morava stabilizer group |
| $H_c^s(G;M)$ | profinite group $G$ 对连续离散或 profinite 模 $M$ 的连续群上同调，具体拓扑随上下文声明 |
| $X^{hG}$ | homotopy fixed points |
| $X^{tG}$ | Tate construction |

**约定 N.10.** 对 profinite group 的同伦固定点和连续群上同调不按离散有限群公式处理。每次使用 descent spectral sequence 时必须声明 $G$ 的拓扑、模的连续性和收敛条件。

## 6. 计算和对偶符号

| 符号 | 含义 |
| --- | --- |
| $I_n$ | invariant prime ideal $(p,v_1,\ldots,v_{n-1})\subset BP_*$；注意第十章也用 $I_n$ 表示 dualizing object 时会局部改名或显式声明 |
| $\operatorname{Ext}_{BP_*BP}$ | $\operatorname{Ext}_{(BP_*,BP_*BP)}$ 的缩写，即 Hopf algebroid comodules 中的 Ext |
| $H_c^s(G;M)$ | profinite group 的连续群上同调 |
| $I$ | Brown-Comenetz dualizing spectrum |
| $I_{\mathbb Q/\mathbb Z}X$ | $F(X,I)$，需要避免与 invariant ideal $I_n$ 混淆 |
| $\operatorname{Pic}_{K(n)}$ | $\operatorname{Pic}(\mathbf{Sp}_{K(n)})$ 的 $\pi_0$ |
| $\kappa_n$ | $K(n)$-local exotic Picard subgroup，使用前需指定文献 convention |

**约定 N.11.** 为避免冲突，正文中 invariant prime ideal 优先写作 $I_n^{BP}$，Brown-Comenetz/Gross-Hopkins dualizing object 优先写作 $I_n^{GH}$，除非局部上下文已经固定。

**约定 N.12.** $v_n$ self-map 的 telescope 总写成
$$
v^{-1}F=\operatorname*{colim}\left(F\to\Sigma^{-d}F\to\Sigma^{-2d}F\to\cdots\right),
$$
其中第一箭头是 $v:\Sigma^dF\to F$ 的脱悬挂伴随。禁止省略悬挂方向。

## 7. Equivariant 和 motivic 符号

| 符号 | 含义 |
| --- | --- |
| $\mathbf{Sp}^G$ | genuine $G$-spectra 的稳定 infinity-范畴 |
| $\Phi^H$ | $H$-geometric fixed point functor |
| $\mathbf{SH}(S)$ | 基 $S$ 上 motivic spectra 的稳定 infinity-范畴 |
| $MGL$ | algebraic cobordism spectrum |
| $tmf,TMF$ | connective 与 periodic topological modular forms，具体版本需随章节声明 |
