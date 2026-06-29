# 附录 I：Čech 超上同调与谱序列

## I.0 目标

第三卷多次使用以下推理：

1. 取一个有限 Stein 覆盖。
2. 在所有有限交上使用 Cartan B 或 Dolbeault acyclicity。
3. 用 Čech 复形计算 $R\Gamma$。
4. 把计算结果送入 Serre duality、GAGA 或 Riemann-Roch 的公式。

附录 C 已写出 Čech 复形和 acyclic 覆盖的结论，但证明仍过于压缩。本附录给出 Čech-to-derived 谱序列、超上同调和 acyclic 覆盖定理的完整同调代数证明。Cartan B、Dolbeault lemma、椭圆正则性等复几何深层定理仍作为输入；本附录只证明“从局部消失到全局复形计算”的代数部分。

## I.1 Čech 双复形

设 $X$ 是拓扑空间，$\mathfrak U=\{U_i\}_{i\in I}$ 是开覆盖，$\mathcal F$ 是阿贝尔群 sheaf。为避免符号负担，本附录假设 $I$ 可良序，并使用递增指标 $i_0<\cdots<i_p$。记

$$
U_{i_0\cdots i_p}=U_{i_0}\cap\cdots\cap U_{i_p}.
$$

取 $\mathcal F$ 的 injective resolution

$$
0\to \mathcal F\to \mathcal I^0\to\mathcal I^1\to\cdots.
$$

定义双复形

$$
C^{p,q}(\mathfrak U,\mathcal I^\bullet)
=
\prod_{i_0<\cdots<i_p}
\mathcal I^q(U_{i_0\cdots i_p}).
$$

水平方向微分为 Čech 微分 $\delta:C^{p,q}\to C^{p+1,q}$：

$$
(\delta c)_{i_0\cdots i_{p+1}}
=
\sum_{a=0}^{p+1}(-1)^a
c_{i_0\cdots\widehat{i_a}\cdots i_{p+1}}
|_{U_{i_0\cdots i_{p+1}}}.
$$

竖直方向微分为 resolution 微分 $d:C^{p,q}\to C^{p,q+1}$。总复形取符号

$$
D(c)=\delta c+(-1)^pdc
$$

对 $c\in C^{p,q}$。

**引理 I.1.** $D^2=0$。

**证明.** 已知 $\delta^2=0$，$d^2=0$。对 $c\in C^{p,q}$，

$$
D^2c
=
\delta^2c+(-1)^{p+1}d\delta c+(-1)^p\delta dc+(-1)^{2p}d^2c.
$$

由于 sheaf restriction 与 resolution 微分相容，$d\delta=\delta d$。中间两项符号相反，故相消。证毕。

## I.2 Čech-to-derived 谱序列

记总复形为

$$
\operatorname{Tot}^\bullet C^{\bullet,\bullet}(\mathfrak U,\mathcal I^\bullet).
$$

按竖直次数过滤：

$$
F^q\operatorname{Tot}^n
=
\bigoplus_{b\ge q}C^{n-b,b}.
$$

在有限覆盖或每个总次数只有有限多个非零项的情形中，这是有界过滤。无限覆盖时需使用乘积总化和收敛条件；第三卷实际使用有限 Stein 覆盖，故本附录采用有限覆盖版本。

**定理 I.2（Čech-to-derived 谱序列）.** 若 $\mathfrak U$ 是有限开覆盖，则存在第一象限谱序列

$$
E_1^{p,q}
=
\prod_{i_0<\cdots<i_p}
H^q(U_{i_0\cdots i_p},\mathcal F)
\Rightarrow
H^{p+q}(X,\mathcal F).
$$

其 $d_1$ 微分由 Čech 微分诱导。

**证明.** 双复形 $C^{p,q}$ 的竖直上同调为

$$
H_v^q(C^{p,\bullet})
=
\prod_{i_0<\cdots<i_p}
H^q(U_{i_0\cdots i_p},\mathcal F),
$$

因为 $\mathcal I^\bullet|_{U_{i_0\cdots i_p}}$ 是 $\mathcal F|_{U_{i_0\cdots i_p}}$ 的 injective 或 acyclic resolution，且有限乘积在阿贝尔群中正合。过滤总复形给出标准谱序列，其 $E_1$ 页即竖直上同调，$d_1$ 由水平微分 $\delta$ 诱导。

剩下要说明极限为 $H^\bullet(X,\mathcal F)$。对固定 $q$，$\mathcal I^q$ 是 injective sheaf，因而 flasque。flasque sheaf 对任意开覆盖的增强 Čech 复形

$$
0\to \mathcal I^q(X)
\to C^0(\mathfrak U,\mathcal I^q)
\to C^1(\mathfrak U,\mathcal I^q)
\to\cdots
$$

正合。于是双复形的水平增广

$$
\Gamma(X,\mathcal I^\bullet)
\to
\operatorname{Tot}^\bullet C^{\bullet,\bullet}(\mathfrak U,\mathcal I^\bullet)
$$

是 quasi-isomorphism。因此总复形上同调等于

$$
H^\bullet(\Gamma(X,\mathcal I^\bullet))
=
H^\bullet(X,\mathcal F).
$$

证毕。

**注 I.3.** 若不想引用“injective sheaf flasque”，可改用 flasque resolution 定义 sheaf cohomology；对 paracompact 空间也可用 fine resolution。第三卷的 Dolbeault 证明正使用 fine resolution。

## I.3 Acyclic 覆盖定理

**定义 I.4.** 覆盖 $\mathfrak U$ 对 $\mathcal F$ 称为 acyclic，如果对所有 $p\ge0$ 和所有非空有限交 $U_{i_0\cdots i_p}$，

$$
H^q(U_{i_0\cdots i_p},\mathcal F)=0,\qquad q>0.
$$

**定理 I.5.** 若 $\mathfrak U$ 是有限 acyclic 覆盖，则自然映射

$$
H^n(C^\bullet(\mathfrak U,\mathcal F))
\to
H^n(X,\mathcal F)
$$

为同构。

**证明.** 由定理 I.2，

$$
E_1^{p,q}=0,\qquad q>0.
$$

因此谱序列只剩 $q=0$ 一行。该行为普通 Čech 复形

$$
C^p(\mathfrak U,\mathcal F)
=
\prod_{i_0<\cdots<i_p}
\mathcal F(U_{i_0\cdots i_p}),
$$

其 $d_1$ 即 Čech 微分。于是

$$
E_2^{p,0}=H^p(C^\bullet(\mathfrak U,\mathcal F)),
\qquad
E_2^{p,q}=0\ (q>0).
$$

没有非零高阶微分能进入或离开 $q=0$ 行，故 $E_2=E_\infty$。过滤的 associated graded 给出 $H^n(X,\mathcal F)$ 的唯一非零分级片，因此得到同构。证毕。

## I.4 超上同调版本

设 $\mathcal K^\bullet$ 是下有界 sheaf 复形。取 injective resolution 的 Cartan-Eilenberg 版本，或取 K-injective resolution

$$
\mathcal K^\bullet\to \mathcal I^\bullet.
$$

定义

$$
\mathbb H^n(X,\mathcal K^\bullet)
=
H^n(\Gamma(X,\mathcal I^\bullet)).
$$

**定义 I.6.** 覆盖 $\mathfrak U$ 对复形 $\mathcal K^\bullet$ 称为 hyper-acyclic，如果对所有有限交 $V=U_{i_0\cdots i_p}$ 和所有 $m$，自然复形 $\mathcal K^\bullet|_V$ 的上同调 sheaf $\mathcal H^m(\mathcal K^\bullet)|_V$ 满足

$$
H^q(V,\mathcal H^m(\mathcal K^\bullet))=0,\qquad q>0.
$$

**定理 I.7（Čech 超上同调计算）.** 若 $\mathfrak U$ 是有限 hyper-acyclic 覆盖，则 Čech 总复形

$$
\operatorname{Tot}^n C^\bullet(\mathfrak U,\mathcal K^\bullet)
=
\bigoplus_{p+r=n}
\prod_{i_0<\cdots<i_p}
\mathcal K^r(U_{i_0\cdots i_p})
$$

计算超上同调：

$$
H^n\operatorname{Tot} C^\bullet(\mathfrak U,\mathcal K^\bullet)
\cong
\mathbb H^n(X,\mathcal K^\bullet).
$$

**证明.** 对双复形先取竖直 cohomology，得到

$$
E_1^{p,m}
=
\prod_{i_0<\cdots<i_p}
\mathcal H^m(\mathcal K^\bullet)(U_{i_0\cdots i_p})
$$

在 hyper-acyclic 假设下，其水平 cohomology 计算 $H^p(X,\mathcal H^m(\mathcal K^\bullet))$。等价地，Čech-to-hyperderived 谱序列在局部高上同调消失后收敛到 $\mathbb H^{p+m}(X,\mathcal K^\bullet)$。由于覆盖有限，过滤有界，谱序列强收敛。总复形即上述 Čech 总复形。证毕。

## I.5 Dolbeault 复形的代数后果

设 $X$ 是复流形，$E$ 是全纯向量丛。Dolbeault 复形

$$
\mathcal A_X^{0,\bullet}(E)
$$

是 $\mathcal O(E)$ 的 fine resolution，这是第三卷附录 F 的输入定理 F.1。

**命题 I.8.** 对任意开覆盖 $\mathfrak U$，若每个 $U_{i_0\cdots i_p}$ 上 Dolbeault lemma 成立，则

$$
H^n(X,\mathcal O(E))
\cong
H^n\Gamma(X,\mathcal A_X^{0,\bullet}(E)).
$$

**证明.** fine sheaf 在 paracompact 空间上 acyclic。因此 $\mathcal A_X^{0,\bullet}(E)$ 是 $\mathcal O(E)$ 的 acyclic resolution。sheaf cohomology 可由 acyclic resolution 的全局截面计算。证毕。

**命题 I.9.** 若 $\mathfrak U$ 是有限 Stein 覆盖，且所有有限交仍为 Stein，则对相干层 $\mathcal F$，

$$
R\Gamma(X,\mathcal F)
\simeq
C^\bullet(\mathfrak U,\mathcal F)
$$

在导出范畴 $D(\mathbf C)$ 中成立。

**证明.** Cartan B 给出

$$
H^q(U_{i_0\cdots i_p},\mathcal F)=0,\qquad q>0
$$

对所有有限交成立。故 $\mathfrak U$ 是 acyclic 覆盖。由定理 I.5，Čech 复形的同调等于 sheaf cohomology。更强地，定理 I.2 中的总复形与 $\Gamma(X,\mathcal I^\bullet)$ quasi-isomorphic，因此在导出范畴中得到 $R\Gamma(X,\mathcal F)\simeq C^\bullet(\mathfrak U,\mathcal F)$。证毕。

## I.6 有限性推论的严格形式

**命题 I.10.** 设 $\mathfrak U$ 是有限 acyclic 覆盖，且每个 $\mathcal F(U_{i_0\cdots i_p})$ 是有限维 $\mathbf C$-向量空间。则 $H^n(X,\mathcal F)$ 有限维。

**证明.** Čech 复形每一项是有限多个有限维向量空间的乘积，故有限维。有限维复形的上同调是 kernel 除以 image；二者为有限维空间的子空间，因此上同调有限维。由定理 I.5 得 $H^n(X,\mathcal F)$ 有限维。证毕。

**警告 I.11.** 对一般 Stein 开集，$\mathcal F(U)$ 通常是无限维 Fréchet 空间。因此命题 I.10 不能直接用于相干上同调有限性。经典有限性定理需要 Grauert finiteness、椭圆 Fredholm 理论或其他深层输入。第三卷把这些结果列为输入定理，而不是从有限 Stein 覆盖形式推出。

## I.7 本附录小结

本附录证明了第三卷可自足使用的同调代数部分：

1. Čech 双复形的符号和 $D^2=0$。
2. Čech-to-derived 谱序列。
3. acyclic 覆盖计算 sheaf cohomology。
4. hyper-acyclic 覆盖计算超上同调。
5. Stein 覆盖加 Cartan B 推出 Čech 计算。

尚未在本书证明的部分是复几何输入：Cartan A/B、Dolbeault lemma、Grauert finiteness、Serre duality 完美性、GAGA 和 Riemann-Roch。

## 练习

**练习 I.1.** 检查引理 I.1 中总微分符号。如果改用 $D=d+(-1)^q\delta$，写出对应的总复形约定。

**练习 I.2.** 证明 flasque sheaf 对任意开覆盖的增强 Čech 复形正合。

**练习 I.3.** 对 $\mathbb P^1=U_0\cup U_\infty$，把附录 H 的线丛计算写成定理 I.5 的一个特例。

**练习 I.4.** 解释警告 I.11 为什么阻止我们用有限 Stein 覆盖直接证明紧复流形上相干上同调有限维。
