# 第三章：Dolbeault resolution 与 liquid 复形

全纯层 $\mathcal O(E)$ 很难直接取导出全局截面，光滑 $(0,q)$-形式层却有 partition of
unity，因而是全局截面函子的 acyclic 对象。Dolbeault 方法用 $\bar\partial$ 把前者
嵌入后者组成的有限复形；真正需要核对的不是一句“Dolbeault 定理”，而是局部正合、
fine acyclicity 和全局截面计算三步如何衔接。进入凝聚语言后还多出第四步：每个
Fréchet 空间及 $\bar\partial$ 必须在 liquid 范畴中有正确实现，且 cokernel 不能由
普通拓扑商想当然地替代。

本章把前三步的形式证明写在正文，并展示 Cauchy--Green 算子的实际输入与输出。局部
积分估计保留为外部输入；第二卷的 liquid 主定理只负责把 Fréchet 项送入 liquid
范畴。经典 Dolbeault 复形的 liquid 实现可以由连续 Hodge--Green 分裂控制，但它与
Clausen--Scholze 解析结构层的比较是另一项义务，不能由“每一项 liquid”自动推出。

## 3.1 $(0,q)$-形式与微分

设 $X$ 是复维数 $n$ 的复流形，$E\to X$ 是全纯向量丛。记
$\mathcal A_X^{0,q}(E)$ 为光滑 $E$-值 $(0,q)$-形式 sheaf。在局部全纯坐标
$z_1,\ldots,z_n$ 和 $E$ 的全纯标架中，截面唯一写成

$$
\alpha
=
\sum_{|J|=q}\alpha_J\,
d\bar z_{j_1}\wedge\cdots\wedge d\bar z_{j_q},
$$

其中 $\alpha_J$ 是光滑 $\mathbb C^r$-值函数。定义

$$
\bar\partial\alpha
=
\sum_{|J|=q}\sum_{k=1}^n
\frac{\partial\alpha_J}{\partial\bar z_k}
\,d\bar z_k\wedge d\bar z_J.
$$

全纯坐标变换和全纯标架变换与 $\bar\partial$ 相容，因此定义不依赖所选局部表示。

**命题 3.1.** 有 $\bar\partial^2=0$，故
$\mathcal A_X^{0,\bullet}(E)$ 是位于次数 $0,\ldots,n$ 的 sheaf 复形。

**证明.** 对每个系数 $\alpha_J$，$\bar\partial^2$ 中 $(k,\ell)$ 与
$(\ell,k)$ 两项的二阶偏导相同：

$$
\frac{\partial^2\alpha_J}
{\partial\bar z_k\partial\bar z_\ell}
=
\frac{\partial^2\alpha_J}
{\partial\bar z_\ell\partial\bar z_k},
$$

而外微分因子满足
$d\bar z_k\wedge d\bar z_\ell
=-d\bar z_\ell\wedge d\bar z_k$；两项相消。$k=\ell$ 时楔积为零。证毕。

次数零的 kernel 是全纯截面：

$$
\ker\bigl(
\bar\partial:\mathcal A_X^{0,0}(E)\to
\mathcal A_X^{0,1}(E)
\bigr)=\mathcal O(E),
$$

因为在全纯平凡化中，$\bar\partial s=0$ 正是每个分量满足 Cauchy--Riemann 方程。

## 3.2 局部正合从哪里来

局部正合性的解析核心是一变量 Cauchy--Green 算子。若
$D'\Subset D\subset\mathbb C$ 是圆盘且 $f\in C_c^\infty(D)$，置

$$
(Tf)(z)=
\frac{1}{2\pi i}
\int_D\frac{f(\zeta)}{\zeta-z}\,
d\zeta\wedge d\bar\zeta.
$$

**外部输入定理 3.2（Cauchy--Green 公式与估计）.** 在 $D'$ 上有

$$
\frac{\partial}{\partial\bar z}Tf=f,
$$

且 $T:C_c^\infty(D)\to C^\infty(D')$ 对相应 Fréchet 拓扑连续。

从这个一变量输入可以逐变量消去多圆盘上的 $d\bar z_j$ 分量。具体地，在第 $j$ 个变量
应用 $T_j$，并令 $H_j=T_j\iota_j$，其中 $\iota_j$ 对含 $d\bar z_j$ 的项作收缩。
Cauchy--Green 公式给出同伦恒等式

$$
\bar\partial_jH_j+H_j\bar\partial_j=\Pi_j,
$$

$\Pi_j$ 是含 $d\bar z_j$ 的分量投影。依次从 $j=n$ 到 $1$ 应用该恒等式，每一步都
把一个方向的分量写成 $\bar\partial$-边界；缩小多圆盘和 cutoff 保证积分核只在内部
使用。全部符号与参数连续性见
[附录 R.2--R.3](R_dolbeault_local_poincare_details.md)。

**外部输入定理 3.3（带系数 Dolbeault 局部引理）.** 对 $q>0$ 和
$x\in X$，若 $\alpha\in\mathcal A_X^{0,q}(E)(U)$ 满足
$\bar\partial\alpha=0$，则存在 $x$ 的邻域 $V\subset U$ 和
$\beta\in\mathcal A_X^{0,q-1}(E)(V)$，使

$$
\bar\partial\beta=\alpha|_V.
$$

这里保留为外部输入的是奇核积分的光滑性与连续估计；从一变量公式到逐变量同伦的
代数步骤已由上式和附录 R 展开。定理 3.3 连同次数零 kernel 给出 sheaf 正合列

$$
0\to\mathcal O(E)
\to\mathcal A_X^{0,0}(E)
\xrightarrow{\bar\partial}\cdots
\xrightarrow{\bar\partial}
\mathcal A_X^{0,n}(E)
\to0.
$$

## 3.3 Partition of unity 产生 fine sheaf

**定义 3.4.** paracompact Hausdorff 空间 $X$ 上的 sheaf $\mathcal G$ 称为 fine，
如果对每个局部有限开覆盖 $\{U_i\}$，存在 endomorphisms
$\theta_i:\mathcal G\to\mathcal G$，满足
$\operatorname{supp}\theta_i\subset U_i$，和式 $\sum_i\theta_i$ 局部有限，且

$$
\sum_i\theta_i=\operatorname{id}_{\mathcal G}.
$$

**外部输入定理 3.5（光滑 partition of unity）.** 光滑流形上每个局部有限开覆盖都有
从属于它的光滑 partition of unity $\{\rho_i\}$。

**命题 3.6.** 每个 $\mathcal A_X^{0,q}(E)$ 都是 fine sheaf。

**证明.** 对给定覆盖取 $\{\rho_i\}$，并定义
$\theta_i(\alpha)=\rho_i\alpha$。乘以光滑函数保持 $(0,q)$ 型，
$\operatorname{supp}\theta_i\subset\operatorname{supp}\rho_i\subset U_i$；
局部有限性保证和式有定义，且

$$
\sum_i\theta_i(\alpha)
=
\left(\sum_i\rho_i\right)\alpha
=
\alpha.
$$

故满足定义 3.4。证毕。

## 3.4 Fine resolution 计算 $R\Gamma$

fine sheaf 的 acyclicity 有一个可见的 Čech 同伦。对局部有限覆盖 $\mathfrak U$，
记 $C^p(\mathfrak U,\mathcal G)$ 为交替 Čech cochains。若
$c\in C^p$、$p>0$，定义

$$
(Kc)_{i_0\cdots i_{p-1}}
=
\sum_j
\theta_j
\left(
c_{j,i_0\cdots i_{p-1}}
\big|_{U_{j,i_0\cdots i_{p-1}}}
\right),
$$

并利用 $\theta_j$ 的支撑把每项延拓到
$U_{i_0\cdots i_{p-1}}$。

**引理 3.7.** 对 $p>0$，有

$$
\delta K+K\delta=\operatorname{id}_{C^p}.
$$

因此 $\check H^p(\mathfrak U,\mathcal G)=0$。

**证明.** 展开两个交替和。删除某个 $i_a$ 后再插入 $j$ 的项，与先插入 $j$ 再删除
同一 $i_a$ 的项符号相反，逐项相消。唯一未配对的项为

$$
\sum_j\theta_j(c_{i_0\cdots i_p})
=c_{i_0\cdots i_p}.
$$

局部有限性保证所有和式合法。证毕。

使用 paracompact Hausdorff 空间上的 Čech--sheaf cohomology 比较定理这一标准外部
输入，引理 3.7 推出 fine sheaf 对 $\Gamma(X,-)$ acyclic。

**定理 3.8（Dolbeault 计算）.** 对复流形 $X$ 和全纯向量丛 $E$，有自然
quasi-isomorphism

$$
R\Gamma(X,\mathcal O(E))
\simeq
\Gamma(X,\mathcal A_X^{0,\bullet}(E)),
$$

因而

$$
H^q(X,\mathcal O(E))
\cong
H^q\bigl(
\Gamma(X,\mathcal A_X^{0,\bullet}(E)),\bar\partial
\bigr).
$$

**证明.** 定理 3.3 给出 $\mathcal O(E)$ 的 resolution；命题 3.6 与引理 3.7 说明其中
每个 $\mathcal A_X^{0,q}(E)$ 都是 $\Gamma$-acyclic。对该 resolution 使用
hypercohomology 谱序列

$$
E_1^{p,q}=H^q(X,\mathcal A_X^{0,p}(E))
\Longrightarrow
\mathbb H^{p+q}(X,\mathcal A_X^{0,\bullet}(E)).
$$

$q>0$ 各行均为零，所以谱序列退化，abutment 等于全局截面复形的同调。resolution
又是 quasi-isomorphism
$\mathcal O(E)\simeq\mathcal A_X^{0,\bullet}(E)$，故 abutment 等于
$H^*(X,\mathcal O(E))$。证毕。

附录 N 保留 fine Čech 同伦、acyclic resolution 和 hypercohomology 的更完整版本；
附录 I 处理有限 Stein 覆盖与 Čech-to-derived 谱序列。

## 3.5 Worked example：实际解一个 $\bar\partial$ 方程

取圆盘 $D\subset\mathbb C$，输入
$\alpha=f(z)\,d\bar z$，其中 $f\in C_c^\infty(D)$。维数一中不存在 $(0,2)$-形式，
所以 $\bar\partial\alpha=0$。定义

$$
u(z)=
\frac{1}{2\pi i}
\int_D\frac{f(\zeta)}{\zeta-z}\,
d\zeta\wedge d\bar\zeta.
$$

输入定理 3.2 给出输出

$$
\bar\partial u=f(z)\,d\bar z=\alpha
$$

在较小圆盘 $D'\Subset D$ 上成立。因此 $\alpha|_{D'}$ 的局部 Dolbeault 上同调类为
零。计算的失败条件也很具体：若不缩小圆盘且 $f$ 不紧支撑，积分会遇到边界项，不能
直接沿用同一公式；局部引理通过 cutoff 与缩小邻域处理这一点。

## 3.6 Liquid 提升需要额外的正合输入

$\Gamma(U,\mathcal A_X^{0,q}(E))$ 在紧集上的 $C^\infty$ 半范数所定义的拓扑下是
Fréchet 空间，
$\bar\partial$ 是连续线性映射。第二卷第五章的外部输入把每一项实现为 $p$-liquid
对象。然而“每一项 liquid”并不自动说明其 cohomology 等于经典拓扑商；还必须控制
$\operatorname{im}\bar\partial$ 的闭性或凝聚局部提升。

**定义 3.9（liquid Dolbeault 模型）.** 固定 $0<p\le1$。对紧复流形 $X$ 和全纯
向量丛 $E$，逐项凝聚化定义 $p$-liquid 复形

$$
R\Gamma_{\mathrm{Dol},p}(X,E)
:=
\underline{\Gamma(X,\mathcal A_X^{0,\bullet}(E))}.
$$

每一项是 $p$-liquid 由第二卷定理 5.5--5.6 保证，连续线性算子 $\bar\partial$
给出 liquid 态射。若选定 Hermitian 度量后存在连续 Hodge--Green 分裂，则该复形在
liquid 范畴中连续同伦等价于零微分 harmonic 复形，因而

$$
H^q\bigl(R\Gamma_{\mathrm{Dol},p}(X,E)\bigr)
\cong
\underline{\mathcal H^{0,q}(X,E)}
\cong
\underline{H^q(X,\mathcal O(E))}.
$$

这里有限维空间均取通常的欧氏拓扑；它们是有限自由
$\underline{\mathbb C}$-模，而不是把底层向量空间离散化所得的凝聚对象。连续
Hodge--Green 分裂及其范畴后果在第四章和附录 L、AR.2 中展开。

**比较边界 3.10.** 若另有一种由 analytic ring 或解析结构层定义的对象
$R\Gamma_{\mathrm{an}}(X,\mathcal O(E))$，要把它与
$R\Gamma_{\mathrm{Dol},p}(X,E)$ 识别，必须构造比较态射并证明它是等价。本书不把这项
比较误称为 Clausen--Scholze 的具名“Dolbeault 建模定理”；他们的复几何理论采用
nuclear/Fredholm 的范畴路线，逐项 liquid membership 本身也不足以给出该比较。
第四章将加入椭圆 Fredholm 输入，解释这个各项无限维的复形为何有有限维同调。

## 3.7 从 resolution 到有限性问题

Dolbeault resolution 已经把相干向量丛上同调变成一个明确复形，却没有证明该复形的
像闭或同调有限维。有限 Stein 覆盖也有同样边界：它能计算上同调，但各交集上的全纯
截面通常无限维。下一章将把 Hodge 分解的解析输入与有限 resolution 的谱序列传播
分开处理，并以 $\mathbb P^1$ 上 $\mathcal O(d)$ 的 Čech 复形给出完全可算的输出。

## 练习

**练习 3.1.** 在局部坐标中逐项展开 $\bar\partial^2f=0$，并指出混合偏导相等与楔积
反对称各自消去哪些项。

**练习 3.2.** 补全引理 3.7 在 $p=1$ 时的计算，验证
$(\delta K+K\delta)c=c$。

**练习 3.3.** 对 $f\in C_c^\infty(D)$，列出 Cauchy--Green 输出 $u$ 的定义域、值域、
所得方程以及不能直接覆盖到 $\partial D$ 的原因。
