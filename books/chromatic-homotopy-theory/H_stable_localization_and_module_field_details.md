# 附录 H：稳定局部化与 $K(n)$-module 细节

## H.1 Localizing subcategories

**定义 H.1.** 稳定 presentable infinity-范畴 $\mathcal C$ 的 localizing subcategory 是全稳定子范畴 $\mathcal L\subseteq\mathcal C$，对所有小 colimits 封闭。

**命题 H.2.** 若 $E$ 是谱，则 $E$-acyclic 谱构成 $\mathbf{Sp}$ 的 localizing subcategory。

**证明.** 记
$$
\mathcal A_E=\{X\mid E\otimes X\simeq0\}.
$$
张量函子 $E\otimes-$ 是 exact functor，故保持 fiber/cofiber 序列；因此 $\mathcal A_E$ 稳定。它也是左伴随，保持所有 colimits。若 $\{X_i\}$ 均在 $\mathcal A_E$ 中，则
$$
E\otimes\operatorname*{colim}_iX_i\simeq\operatorname*{colim}_i(E\otimes X_i)\simeq0.
$$
所以 $\mathcal A_E$ 对 colimits 封闭。证毕。

**定义 H.3.** 谱 $Y$ 为 $E$-local，当且仅当对所有 $A\in\mathcal A_E$ 有
$$
F(A,Y)\simeq0.
$$
记 $E$-local 谱全子范畴为 $\mathbf{Sp}_E$。

**命题 H.4.** $\mathbf{Sp}_E$ 对 limits 封闭。

**证明.** 设图 $Y_i$ 全为 $E$-local。对任意 $E$-acyclic $A$，
$$
F(A,\lim_iY_i)\simeq\lim_iF(A,Y_i)\simeq\lim_i0\simeq0.
$$
故 $\lim_iY_i$ 是 $E$-local。证毕。

## H.2 Acyclic-local orthogonality

**命题 H.5.** 若 $A$ 是 $E$-acyclic 且 $Y$ 是 $E$-local，则任意映射 $A\to Y$ 在 mapping space 中为零，即
$$
\operatorname{Map}(A,Y)\simeq *.
$$

**证明.** $\operatorname{Map}(A,Y)=\Omega^\infty F(A,Y)$。按 $E$-local 定义，$F(A,Y)\simeq0$，其零空间为 contractible。证毕。

**命题 H.6.** 若 fiber 序列 $A\to X\to Y$ 中 $A$ 是 $E$-acyclic 且 $Y$ 是 $E$-local，则 $X\to Y$ 是 $E$-localization。

**证明.** 定义 1.12 要求两点：目标 $Y$ 为 $E$-local，fiber $A$ 为 $E$-acyclic。二者正是假设。证毕。

## H.3 Smashing localization 的等价判据

**命题 H.7.** localization $L$ 是 smashing，当且仅当自然变换
$$
L\mathbb S\otimes X\to LX
$$
对所有 $X$ 为等价。

**证明.** 这是定义 1.15。实际使用时要检查该自然变换由单位 $\mathbb S\to L\mathbb S$ 和 localization map $X\to LX$ 诱导，且与 colimits 相容。证毕。

**命题 H.8.** 若 $L$ smashing，则 $L$-acyclics 是 tensor ideal：若 $LX\simeq0$，则对任意 $Y$ 有 $L(X\otimes Y)\simeq0$。

**证明.** 由 smashing 性，
$$
L(X\otimes Y)\simeq L\mathbb S\otimes X\otimes Y\simeq (L\mathbb S\otimes X)\otimes Y\simeq LX\otimes Y\simeq0.
$$
证毕。

## H.4 $K(n)$-modules 的 field-like 性质

**外部输入 H.9.** Morava K-theory $K(n)$ 的 module category 具有 graded field behavior：任意 $K(n)$-module spectrum 由其 graded homotopy groups 控制，特别地可分解为若干悬挂的 $K(n)$ 的 wedge。该结论依赖 $K(n)_*$ 为 graded field 以及 $K(n)$-module category 的标准结构。

**证明草图 H.10.** 设 $M$ 是 $K(n)$-module。对每个 homogeneous basis element $x\in\pi_dM$，由 $K(n)$-module 结构得到映射
$$
\Sigma^dK(n)\to M.
$$
取所有 basis elements 的 wedge，得到
$$
\bigvee_{\alpha}\Sigma^{d_\alpha}K(n)\to M.
$$
该映射在 $\pi_*$ 上是 $K(n)_*$-模同构。若采用 $K(n)$-modules 中弱等价由同伦群检测的模型，则该映射为等价。最后一句是 module category 的模型输入，不是普通代数自动推出。

**推论 H.11.** 对 $K(n)$-module $M$，若 $\pi_*M=0$，则 $M\simeq0$。

**证明.** 由 H.10，$M$ 分解为由 $\pi_*M$ 的 basis 给出的 wedge。若 $\pi_*M=0$，basis 为空，wedge 为零对象。证毕。

## H.5 与 $K(n)$-acyclic 的区别

**警告 H.12.** $K(n)\otimes X$ 是 $K(n)$-module，因此可用 H.9。谱 $X$ 本身通常不是 $K(n)$-module。由 $K(n)_*X=0$ 得到的是 $K(n)\otimes X\simeq0$，也就是 $X$ 为 $K(n)$-acyclic；不能推出 $X\simeq0$，除非 $X$ 属于有限谱并调用 finite detection。

## 本附录小结

本附录把第一章中局部化的形式细节补齐，并把 $K(n)$ 的“场性”限制在 $K(n)$-module category 中。核心风险是：field-like module behavior 不能直接升级为普通谱范畴中的对象分类。
