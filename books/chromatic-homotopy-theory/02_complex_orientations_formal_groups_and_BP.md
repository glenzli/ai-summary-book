# 第二章：复定向、形式群律与 Brown-Peterson theory

## 本章目标

本章说明 chromatic height 的来源。我们从复定向同调理论得到形式群律，再通过 $p$-typical 化进入 $BP$、Hazewinkel generators、Johnson-Wilson theories 和 Morava K-theories 的系数形式。

## 依赖前置知识

需要熟悉 $\mathbb CP^\infty$、线丛的 classifying space、幂级数环和一维形式群律的基本定义。复杂的 Lazard theorem 和 Quillen theorem 作为外部输入。

## 2.1 复定向

**定义 2.1.** 设 $E$ 是乘法谱。一个复定向（complex orientation）是元素
$$
x\in E^2(\mathbb CP^\infty)
$$
使得在包含 $i:\mathbb CP^1\hookrightarrow \mathbb CP^\infty$ 下，$i^*x$ 是 $\widetilde E^2(\mathbb CP^1)$ 中由单位决定的标准生成元。

**命题 2.2.** 若 $E$ 复定向，则有自然同构
$$
E^*(\mathbb CP^\infty)\cong E^*[[x]].
$$

**证明草图.** 对有限射影空间 $\mathbb CP^m$，复定向给出 Thom class 和 projective bundle formula，得到
$$
E^*(\mathbb CP^m)\cong E^*[x]/(x^{m+1}).
$$
取逆极限并使用 $\mathbb CP^\infty=\operatorname*{colim}_m\mathbb CP^m$ 的 CW-filtration，得到 $x$-adic 完备幂级数环。完整证明依赖 generalized cohomology 的 Milnor exact sequence 和 projective bundle theorem，作为基础外部输入。证毕。

**定义 2.3.** 复定向 $x$ 对应的形式群律定义为
$$
F_E(x_1,x_2)=\mu^*x\in E^*[[x_1,x_2]],
$$
其中 $\mu:\mathbb CP^\infty\times \mathbb CP^\infty\to\mathbb CP^\infty$ classifies 两个 universal line bundles 的张量积。

## 2.2 形式群律和高度

**定义 2.4.** 交换环 $R$ 上的一维交换形式群律是幂级数
$$
F(X,Y)\in R[[X,Y]]
$$
满足：
$$
F(X,0)=X,\quad F(0,Y)=Y,
$$
$$
F(F(X,Y),Z)=F(X,F(Y,Z)),
$$
$$
F(X,Y)=F(Y,X).
$$

**定义 2.5.** 设 $R$ 是特征 $p$ 的环。形式群律 $F$ 的 $p$-series 定义为
$$
[p]_F(X)=\underbrace{F(X,F(X,\ldots,F(X,X)\ldots))}_{p\text{ 次}}.
$$
若在域 $k$ 上
$$
[p]_F(X)=aX^{p^n}+\text{higher terms},\qquad a\ne 0,
$$
则 $F$ 的高度为 $n$。若 $[p]_F(X)=0$，高度为 $\infty$。

**例 2.6.** 加法形式群 $F_a(X,Y)=X+Y$ 在特征 $p$ 下满足 $[p]_{F_a}(X)=0$，高度为 $\infty$。乘法形式群
$$
F_m(X,Y)=X+Y+XY
$$
满足
$$
[p]_{F_m}(X)=(1+X)^p-1=X^p
$$
在特征 $p$ 下高度为 $1$。

**证明.** 加法情形中 $pX=0$。乘法情形由 $1+F_m(X,Y)=(1+X)(1+Y)$ 得
$$
1+[p]_{F_m}(X)=(1+X)^p.
$$
在特征 $p$ 下二项式中间系数为零，得到 $[p]_{F_m}(X)=X^p$。证毕。

## 2.3 $MU$ 与 Quillen theorem

**外部输入定理 2.7 (Quillen).** 复 cobordism spectrum $MU$ 的标准复定向给出 Lazard ring 与 $MU_*$ 的同构，使得 $MU$ 上的形式群律为 universal one-dimensional commutative formal group law。

**使用说明.** 本书不会重证 Quillen theorem。其作用是把复定向同调理论的自然变换问题转化为形式群律的代数问题，并允许用形式群高度分层研究稳定同伦论。

**推论 2.8.** 若 $E$ 是复定向环谱，则复定向诱导环同态
$$
MU_*\to E_*,
$$
从而给出 $E_*$ 上的形式群律。

**证明.** 复定向等价于乘法上同调理论中的 Thom classes，因而由 $MU$ 的 universal property 诱导环谱映射 $MU\to E$。取同伦群得到 $MU_*\to E_*$。Quillen theorem 识别该映射对应的形式群律。证毕。

## 2.4 Brown-Peterson theory

**定义 2.9.** 固定素数 $p$。Brown-Peterson spectrum $BP$ 是 $p$-局部复 cobordism 的 $p$-typical summand。其系数环写作
$$
BP_*\cong \mathbb Z_{(p)}[v_1,v_2,\ldots],\qquad |v_i|=2(p^i-1),
$$
其中 $v_i$ 可取 Hazewinkel generators。

**外部输入 2.10.** $BP$ 的存在、$MU_{(p)}$ 的 $p$-typical splitting 和上式系数计算作为 Brown-Peterson-Quillen-Hazewinkel 体系外部输入。

**定义 2.11.** Johnson-Wilson theory $E(n)$ 的系数为
$$
E(n)_*\cong \mathbb Z_{(p)}[v_1,\ldots,v_n,v_n^{-1}].
$$
在本书中 $L_n$ 表示 $L_{E(n)}$。

**定义 2.12.** Morava K-theory $K(n)$ 的系数为
$$
K(n)_*\cong \mathbb F_p[v_n^{\pm1}],\qquad |v_n|=2(p^n-1),
$$
并令 $K(0)=H\mathbb Q$。

**命题 2.13.** $K(n)_*$ 是 graded field，即每个非零 homogeneous 元素可逆。

**证明.** $K(n)_*$ 的 homogeneous 非零元素形如 $a v_n^m$，其中 $a\in\mathbb F_p^\times$，$m\in\mathbb Z$。其逆为 $a^{-1}v_n^{-m}$。证毕。

**警告 2.14.** 命题 2.13 只是系数环的代数事实。它不等同于说任意 $K(n)$-module spectrum 的结构完全平凡；后者需要 module category 层面的定理。

## 2.5 截断 Brown-Peterson spectra

**定义 2.15.** $BP\langle n\rangle$ 表示截断 Brown-Peterson spectrum，其 homotopy groups 为
$$
BP\langle n\rangle_*\cong \mathbb Z_{(p)}[v_1,\ldots,v_n].
$$

**警告 2.16.** $BP\langle n\rangle$ 的乘法结构高度依赖模型和素数。近期 redshift 文献中使用的 $\mathbb E_3$-$BP$ algebra structure 是深层定理，不能由系数环商自动推出。

## 2.6 Landweber exactness 的教材口径

**定义 2.17.** 设 $M$ 是 $BP_*$-模。对 $n\ge0$，记
$$
I_n=(p,v_1,\ldots,v_{n-1}),\qquad I_0=(0).
$$
称 $M$ 满足 Landweber exactness 条件，若对每个 $n$，乘以 $v_n$ 的映射
$$
v_n:M/I_nM\to M/I_nM
$$
是单射，其中 $v_0=p$。

**外部输入定理 2.18 (Landweber exact functor theorem).** 若 $BP_*$-代数 $R$ 满足 Landweber exactness，则函子
$$
X\longmapsto BP_*X\otimes_{BP_*}R
$$
定义一个同调理论。在合适条件下该同调理论由谱表示。

**例 2.19.** $E(n)_*=\mathbb Z_{(p)}[v_1,\ldots,v_n,v_n^{-1}]$ 满足 Landweber exactness。

**证明草图.** 对 $k\le n$，在依次模去 $p,v_1,\ldots,v_{k-1}$ 后，$v_k$ 仍是多项式或 Laurent 多项式环中的非零因子。对 $k=n$，$v_n$ 已可逆。对 $k>n$，相应 quotient 在 $v_n$ 已可逆同时又处于 ideal 中时退化为零，单射条件自动满足。完整证明需处理 $BP_*$-代数结构和 regular sequence，作为 Landweber theorem 的标准例子。证毕。

**警告 2.20.** Landweber exactness 产生同调理论，不产生唯一的 $\mathbb E_\infty$-ring spectrum。tmf 的构造需要 sheaf of structured ring spectra，而不仅是 Landweber exact local charts。

## 2.7 $p$-typical 化和高度

**定义 2.21.** 形式群律 $F$ 在 $p$-局部环上称为 $p$-typical，若其 logarithm 或 $p$-series 在合适坐标下只含 $p$-power 次项。Hazewinkel generators $v_i$ 是 $p$-typical 坐标下 $p$-series 的标准参数。

**例 2.22.** 在 $BP_*$ 上，universal $p$-typical formal group law 的 $p$-series 可写成
$$
[p]_F(x)=px+_F v_1x^p+_F v_2x^{p^2}+_F\cdots
$$
其中 $+_F$ 表示形式群律加法。该公式是 convention，具体展开依赖坐标选择。

**警告 2.23.** 不能把 $[p]_F(x)=px+v_1x^p+\cdots$ 当作普通幂级数加法下的恒等式而忽略 $+_F$。许多低阶计算可在指定坐标中化简，但必须声明 convention。

## 本章小结

复定向把稳定同伦论连接到形式群律。Quillen theorem 说明 $MU$ 表示 universal formal group law；$BP$ 抽取 $p$-typical 信息；$E(n)$ 和 $K(n)$ 则把高度 $\le n$ 和高度 $n$ 的信息分别变成同调理论。这一章给出系数和定义，深层存在性和 universal property 已登记为外部输入。

## 练习

**练习 2.1.** 验证乘法形式群 $F_m(X,Y)=X+Y+XY$ 的结合律。

**练习 2.2.** 在 $K(n)_*$ 中写出 $v_n^3+v_n^3$ 是否可逆，分别讨论 $p=2$ 与 $p\ne2$。

**练习 2.3.** 说明为什么从 $BP_*$ 到 $BP\langle n\rangle_*$ 的环商不自动给出唯一的 $\mathbb E_\infty$-ring quotient。
