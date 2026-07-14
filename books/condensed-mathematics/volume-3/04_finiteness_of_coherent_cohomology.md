# 第四章：相干上同调为何变成有限维

紧复流形上的 Dolbeault 复形每一项通常都是无限维 Fréchet 空间，有限 Stein 覆盖的
Čech 复形也由无限维全纯函数空间组成；有限维上同调绝不来自“复形项有限”。真正的
压缩机制是椭圆算子的 Fredholm 性：Hodge 分解把上同调类选成 harmonic representative，
而紧流形上的椭圆 Laplacian 只有有限维 kernel。对一般相干层，还需沿有限 resolution
或 Grauert 直接像定理把向量丛结论传播出去。

本章保留椭圆正则性和 Grauert 有限性为深层外部输入，但完整证明 Hodge 分解如何推出
有限维、有限 resolution 的谱序列如何传播有限性，以及有限维 cohomology 为何在
analytic 派生范畴中给出 perfect 对象。最后对 $\mathbb P^1$ 上 $\mathcal O(d)$ 从
输入、微分到输出逐项计算，展示一般机制在最小模型中的具体形状。

## 4.1 Čech 计算不自动给有限性

若 $\mathfrak U=\{U_i\}$ 是有限 Stein 覆盖，所有非空有限交仍 Stein，则 Cartan B
这一经典外部输入给出

$$
R\Gamma(X,\mathcal F)
\simeq
C^\bullet(\mathfrak U,\mathcal F)
$$

对相干层 $\mathcal F$ 成立。然而即使 $U=\Delta$ 是圆盘，

$$
\mathcal O(\Delta)
=
\left\{
\sum_{m\ge0}a_mz^m:
\text{幂级数在 }\Delta\text{ 收敛}
\right\}
$$

也包含无限多个线性无关单项式。因而“覆盖有限”和“高阶局部上同调消失”只说明 Čech
复形计算正确，不说明其项或同调有限维。有限性必须来自额外的紧性或 Fredholm 输入。

## 4.2 Hodge--Fredholm 压缩机制

设

$$
\cdots\to H^{q-1}\xrightarrow{d_{q-1}}H^q
\xrightarrow{d_q}H^{q+1}\to\cdots
$$

是 Hilbert 复形，$d_q$ 为闭稠定算子且 $d_qd_{q-1}=0$。定义

$$
\Delta_q=d_{q-1}d_{q-1}^*+d_q^*d_q,
\qquad
\mathcal H^q=\ker\Delta_q.
$$

由 adjoint 定义，对 $x$ 在 $\Delta_q$ 的定义域中有

$$
\langle\Delta_qx,x\rangle
=
\|d_qx\|^2+\|d_{q-1}^*x\|^2,
$$

故 $\mathcal H^q=\ker d_q\cap\ker d_{q-1}^*$。

**外部输入定理 4.1（椭圆 Hodge--Fredholm）.** 对紧 Hermitian 复流形 $X$ 和配备
Hermitian 度量的全纯向量丛 $E$，Dolbeault Laplacian
$\Delta_{\bar\partial,q}$ 在 Sobolev 完成上是
Fredholm，并有正交分解

$$
H^q
=
\operatorname{im}\bar\partial_{q-1}
\oplus
\mathcal H^{0,q}(X,E)
\oplus
\operatorname{im}\bar\partial_q^*.
$$

两个像闭，且 $\mathcal H^{0,q}(X,E)$ 有限维；harmonic Sobolev 向量由椭圆正则性
自动光滑。

**命题 4.2.** 在输入定理 4.1 下，自然映射

$$
\mathcal H^{0,q}(X,E)
\longrightarrow
H^q\bigl(\Gamma(X,\mathcal A_X^{0,\bullet}(E)),\bar\partial\bigr)
$$

是同构。

**证明.** 若 harmonic 形式 $h$ 是边界 $h=\bar\partial a$，则
$h$ 同时属于正交直和中的 harmonic 分量和
$\operatorname{im}\bar\partial$ 分量，故 $h=0$，所以映射单射。

反之，取 $x\in\ker\bar\partial_q$。Hodge 分解的第三个分量
$\operatorname{im}\bar\partial_q^*$ 与 $\ker\bar\partial_q$ 正交：若
$v=\bar\partial_q^*b$ 且 $y\in\ker\bar\partial_q$，则
$\langle y,v\rangle=\langle\bar\partial_qy,b\rangle=0$。前两个分量均落在
$\ker\bar\partial_q$，所以把正交分解与该闭子空间相交得到

$$
x=\bar\partial a+h.
$$

于是 $x$ 与 $h$ 相差边界，任一上同调类都有 harmonic 代表。这个论证只使用闭值域
给出的正交分解，避免了对未声明属于复合无界算子定义域的向量施加
$\bar\partial\bar\partial^*$。证毕。

第三章的 Dolbeault 计算与上述 harmonic 代表定理合在一起，给出向量丛情形：

**推论 4.3.** 若 $X$ 紧且 $E$ 为全纯向量丛，则
$H^q(X,\mathcal O(E))$ 有限维。

**证明.** 第三章定理 3.8 把 sheaf cohomology 识别为 Dolbeault cohomology；命题 4.2
再把后者识别为输入定理 4.1 中的有限维 harmonic space。证毕。

## 4.3 有限 resolution 如何传播有限性

设相干层 $\mathcal F$ 有全局有限局部自由 resolution

$$
E^{-m}\to\cdots\to E^0\to\mathcal F\to0.
$$

将 $E^\bullet$ 放在次数 $[-m,0]$，则
$E^\bullet\simeq\mathcal F$。

**定理 4.4.** 若每个 $H^q(X,E^p)$ 有限维，则每个
$H^k(X,\mathcal F)$ 有限维。

**证明.** 使用有界 hypercohomology 谱序列

$$
E_1^{p,q}=H^q(X,E^p)
\Longrightarrow
\mathbb H^{p+q}(X,E^\bullet)
=
H^{p+q}(X,\mathcal F).
$$

$E^\bullet$ 有界，所以固定总次数只涉及有限个 $p$，也就只涉及有限项。
$E_1^{p,q}$ 按假设有限维。每个后续页满足

$$
E_{r+1}^{p,q}
=
\ker(d_r:E_r^{p,q}\to E_r^{p+r,q-r+1})
/
\operatorname{im}(d_r:E_r^{p-r,q+r-1}\to E_r^{p,q}),
$$

故仍为有限维子商。收敛给 $H^k(X,\mathcal F)$ 一个有限过滤，其分级片为相应
$E_\infty^{p,k-p}$；有限个有限维分级片的扩张仍有限维。证毕。

长度一时，这个机制就是短正合列

$$
0\to E^{-1}\to E^0\to\mathcal F\to0
$$

的长正合上同调列：$H^q(\mathcal F)$ 是两个有限维 kernel/cokernel 的扩张。附录 M
给出一般有限过滤和 two-out-of-three 版本，附录 X 保留上式的完整超上同调展开。

全局 resolution 是额外假设，不能由第二章的局部 resolution 自动推出。因此一般结论
仍须使用下面的深层输入。

**外部输入定理 4.5（Cartan--Serre--Grauert 有限性）.** 若 $X$ 是紧复空间，
$\mathcal F$ 是相干解析层，则每个 $H^q(X,\mathcal F)$ 是有限维复向量空间，且仅
有限多个 $q$ 非零。

对复流形上有全局有限向量丛 resolution 的 $\mathcal F$，定理 4.4 与推论 4.3 已经给出
书内化约；输入定理 4.5 覆盖没有这种全局 resolution 的一般情形。

## 4.4 有限同调等价于点上的 perfect 复形

记 $D(\mathbb C)$ 为复向量空间导出范畴。

**命题 4.6.** 若 $C^\bullet$ 是有界复形，且每个 $H^q(C^\bullet)$ 有限维，则在
$D(\mathbb C)$ 中有非典范同构

$$
C^\bullet
\simeq
\bigoplus_q H^q(C^\bullet)[-q].
$$

特别地，$C^\bullet$ 是 perfect。

**证明.** 记 $Z^q=\ker d^q$、$B^q=\operatorname{im}d^{q-1}$。向量空间短正合列可
分裂，选择

$$
Z^q=B^q\oplus H_q,
\qquad
C^q=Z^q\oplus L^q,
$$

其中 $H_q\cong H^q(C^\bullet)$。微分限制
$d^q:L^q\to B^{q+1}$ 是同构：它按定义满射到 $B^{q+1}$，kernel 为
$L^q\cap Z^q=0$。因此 $C^\bullet$ 分解为零微分复形
$\bigoplus_qH_q[-q]$ 与若干两项同构复形
$L^q\xrightarrow{\sim}B^{q+1}$ 的直和。后者可缩，故在导出范畴中为零，得到所述
同构。有限个 $H_q$ 均有限维，所以右侧是有限维向量空间组成的有界复形，即 perfect。
证毕。

**推论 4.7（凝聚形式后果）.** 对向量丛 $E$，接受第三章定义 3.9 所述连续
Hodge--Green 分裂后，$R\Gamma_{\mathrm{Dol},p}(X,E)$ 是 perfect liquid 对象。对一般
相干层 $\mathcal F$，若另给出
$R\Gamma_{\mathrm{an}}(X,\mathcal F)$ 与经典 $R\Gamma$ 相容的比较定理，则输入定理
4.5 同样推出该 analytic 对象 perfect；仅有 Grauert 的有限维结论并不自行构造这项
比较。

**证明.** 输入定理 4.5 给出有限个有限维 cohomology，命题 4.6 给出有限个
$\mathbb C[-q]$ 的有限直和模型。单位对象可对偶，有限直和、shift、cone 和 retract
保持可对偶；强幺半实现保持 evaluation、coevaluation 与三角恒等式。证毕。

这里“finite-dimensional cohomology”到“perfect”的推导已经完成；椭圆有限性和
analytic 强幺半实现仍分别是外部输入，二者不可用一句“紧对象”互相替代。

## 4.5 Worked example：$\mathbb P^1$ 上的 $\mathcal O(d)$

在解析射影直线上取 $U_0\cong\mathbb C_z$、
$U_\infty\cong\mathbb C_w$，交集为 $\mathbb C^\times$，且 $w=z^{-1}$。
$\mathcal O(d)$ 的转移函数为 $e_\infty=z^de_0$。Cartan B 说明该 Stein 覆盖的 Čech
复形计算上同调：

$$
C^0=\mathcal O(\mathbb C)\oplus\mathcal O(\mathbb C),
\qquad
C^1=\mathcal O(\mathbb C^\times),
$$

$$
\delta(f_0,f_\infty)
=
z^df_\infty(z^{-1})-f_0(z).
$$

任意 $g\in\mathcal O(\mathbb C^\times)$ 有 Laurent 展开
$g=\sum_{k\in\mathbb Z}c_kz^k$。其非负幂部分在 $\mathbb C$ 上整，因而可由 $f_0$
消去；其 $k\le d$ 部分写成 $z^df_\infty(z^{-1})$，也可消去。商中只留下

$$
z^{d+1},z^{d+2},\ldots,z^{-1}
$$

这些幂。当 $d\ge-1$ 时区间为空；当 $d\le-2$ 时有 $-d-1$ 项。因此

$$
h^1(\mathbb P^1,\mathcal O(d))
=
\begin{cases}
0,&d\ge-1,\\
-d-1,&d\le-2.
\end{cases}
$$

kernel 条件
$f_0(z)=z^df_\infty(z^{-1})$ 表明 $f_0$ 在无穷远最多有 $d$ 阶极点。若 $d\ge0$，
它是次数不超过 $d$ 的多项式；若 $d<0$，只能为零。故

$$
h^0(\mathbb P^1,\mathcal O(d))
=
\begin{cases}
d+1,&d\ge0,\\
0,&d<0.
\end{cases}
$$

输入是整数 $d$、标准覆盖与转移函数；步骤是 Laurent 展开后按指数消去；输出是两个
有限维群以及

$$
\chi(\mathbb P^1,\mathcal O(d))=d+1.
$$

若覆盖交集不是 acyclic，朴素两项 Čech 复形未必计算 sheaf cohomology；这里 Cartan B
正是不可省略的适用条件。附录 H 给出代数 Laurent 多项式版本，附录 S 推广到
$\mathbb P^n$。

## 4.6 有限维性为对偶准备了什么

有限性使线性对偶与 cohomology 交换，也使 trace 配对的非退化性可以等价地写成一个
perfect-complex quasi-isomorphism。它本身尚未构造配对。第五章将从
$\alpha\wedge\beta$ 的积分开始，先证明配对尊重 $\bar\partial$-边界，再把深层的
Serre perfectness 与书内的 derived/六函子形式推导分离。

## 练习

**练习 4.1.** 在命题 4.2 的满射证明中，详细验证
$\operatorname{im}\bar\partial_q^*\perp\ker\bar\partial_q$，并说明闭值域为何允许从
Hodge 分解得到
$\ker\bar\partial_q=\operatorname{im}\bar\partial_{q-1}\oplus\mathcal H^{0,q}$。

**练习 4.2.** 对长度一 resolution 写出完整长正合列，并直接证明定理 4.4 的该特例。

**练习 4.3.** 用 $d=-3$ 实际计算例 4.5 的 kernel、cokernel 基和 Euler
characteristic。
