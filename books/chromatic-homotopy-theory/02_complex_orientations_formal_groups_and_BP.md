# 第二章：复定向、形式群律与 Brown-Peterson theory

第一章只说明怎样相对于一个同调理论局部化，却没有解释“高度”从何而来。复定向提供缺失的代数坐标：线丛的张量积在 $E^*(\mathbb CP^\infty)$ 上诱导一维形式群律，而在固定素数处，$p$-级数的首个非零层正好记录高度。沿这条链可以从 $MU$ 进入 $BP$、Hazewinkel 生成元、Johnson--Wilson 理论和 Morava $K$-理论的系数环。本章假定读者熟悉 $\mathbb CP^\infty$、线丛分类空间与幂级数环；Lazard 分类和 Quillen 对 $MU$ 的识别作为精确外部输入，后续只使用其明确陈述。

## 2.1 复定向

**定义 2.1.** 设 $E$ 是带单位的 homotopy-commutative ring spectrum，
并用
$$
E^q(X)=\pi_{-q}F(\Sigma^\infty_+X,E)
$$
作为未约化 cohomological grading；约化版本用 based suspension
spectrum $\Sigma^\infty X$。一个复定向（complex orientation）是元素
$$
x\in \widetilde E^2(\mathbb CP^\infty)
$$
使得在包含 $i:\mathbb CP^1\hookrightarrow \mathbb CP^\infty$ 下，
$i^*x$ 经 suspension isomorphism
$$
\widetilde E^2(\mathbb CP^1)\cong
\widetilde E^2(S^2)\cong E^0(*)=\pi_0E
$$
对应于单位 $1\in\pi_0E$。若只给 $mathbb E_1$-乘法而无交换性，后文
的交换形式群律结论不在本定义的适用范围内。

**外部基础定理 2.2（projective bundle theorem）.** 若 $E$ 满足定义
2.1 且选定复定向 $x$，则有 complete graded $E^*$-algebra 的自然同构
$$
E^*(\mathbb CP^\infty)\cong E^*[[x]].
$$

**证明路线与边界（外部输入）.** Projective bundle theorem 对每个
有限射影空间给出
$$
E^*(\mathbb CP^m)\cong E^*[x]/(x^{m+1}).
$$
这些 quotient 的限制映射满射，所以 Milnor exact sequence 的
$\lim^1 E^{*-1}(\mathbb CP^m)$ 项消失；取逆极限得到 $x$-adic 完备
幂级数环。Projective bundle theorem 与 Thom isomorphism 的证明不在
本书重建；上述逆极限步骤只说明外部定理如何产生所示 complete ring，
不冒充完整书内证明。

**定义 2.3.** 复定向 $x$ 对应的形式群律定义为
$$
F_E(x_1,x_2)=\mu^*x
\in E^2(\mathbb CP^\infty\times\mathbb CP^\infty)
\cong E^*[[x_1,x_2]],
$$
其中 $\mu:\mathbb CP^\infty\times \mathbb CP^\infty\to\mathbb CP^\infty$ classifies 两个 universal line bundles 的张量积。

这里 $x_1,x_2$ 的 cohomological degree 均为 $2$，系数按总次数 $2$
分次；因此严格说这是 $E^*$ 上的 graded formal group law。只有在指定
偶周期元并重分次后，才可把坐标无说明地视作 degree $0$。

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

**定义 2.11.** 对 $n\ge1$，Johnson--Wilson theory $E(n)$ 指表示下列
Landweber-exact 同调理论的一个选定 ring-spectrum 模型：
$$
E(n)_*\cong \mathbb Z_{(p)}[v_1,\ldots,v_n,v_n^{-1}].
$$
高度零单独约定
$$
E(0)=H\mathbb Q,
\qquad E(0)_*=\mathbb Q.
$$
在本书中 $L_n$ 表示 $L_{E(n)}$。Landweber exact functor theorem 先
给同调理论；representing spectrum 及所需乘法结构属于 Johnson--Wilson
理论的标准外部构造，不能只从显示的系数环推出。

**定义 2.12.** 对 $n\ge1$，Morava K-theory $K(n)$ 指一个选定的、有
单位且 homotopy-associative 的 ring-spectrum 模型，其系数为
$$
K(n)_*\cong \mathbb F_p[v_n^{\pm1}],\qquad |v_n|=2(p^n-1),
$$
并令 $K(0)=H\mathbb Q$。本书只在需要 module category 时使用上述
结合乘法；更强的交换或 $\mathbb E_k$ 结构必须另行声明。

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

**例 2.19.** 对 $n\ge1$，
$E(n)_*=\mathbb Z_{(p)}[v_1,\ldots,v_n,v_n^{-1}]$ 满足 Landweber
exactness。高度零的 $E(0)_*=\mathbb Q$ 另由 $p$ 已可逆直接检查。

**证明.** 记 $R=E(n)_*$。当 $k=0$ 时，$v_0=p$ 在
$R=\mathbb Z_{(p)}[v_1,\ldots,v_n,v_n^{-1}]$ 上是非零因子。对
$1\le k\le n$，有
$$
R/I_kR\cong
\mathbb F_p[v_k,v_{k+1},\ldots,v_n,v_n^{-1}],
$$
这是整环；故乘以 $v_k$ 单射。当 $k=n$ 时，$v_n$ 甚至可逆。若
$k\ge n+1$，则 $I_k$ 含 $v_n$，而 $v_n$ 在 $R$ 中可逆，所以
$I_kR=R$ 且 $R/I_kR=0$；零模上的乘法映射自动单射。所有 $k$ 均满足
定义 2.17，故 $E(n)_*$ Landweber exact。证毕。

**警告 2.20.** Landweber exactness 产生同调理论，不产生唯一的 $\mathbb E_\infty$-ring spectrum。tmf 的构造需要 sheaf of structured ring spectra，而不仅是 Landweber exact local charts。

## 2.7 $p$-typical 化和高度

**定义 2.21.** 形式群律 $F$ 在 $p$-局部环上称为 $p$-typical，若其 logarithm 或 $p$-series 在合适坐标下只含 $p$-power 次项。Hazewinkel generators $v_i$ 是 $p$-typical 坐标下 $p$-series 的标准参数。

**例 2.22.** 在 $BP_*$ 上，universal $p$-typical formal group law 的 $p$-series 可写成
$$
[p]_F(x)=px+_F v_1x^p+_F v_2x^{p^2}+_F\cdots
$$
其中 $+_F$ 表示形式群律加法。该公式是 convention，具体展开依赖坐标选择。

**警告 2.23.** 不能把 $[p]_F(x)=px+v_1x^p+\cdots$ 当作普通幂级数加法下的恒等式而忽略 $+_F$。许多低阶计算可在指定坐标中化简，但必须声明 convention。

## 2.8 从复定向到色层高度

复定向把稳定同伦论连接到形式群律。Quillen theorem 说明 $MU$ 表示 universal formal group law；$BP$ 抽取 $p$-typical 信息；$E(n)$ 和 $K(n)$ 则把高度 $\le n$ 和高度 $n$ 的信息分别变成同调理论。这一章给出系数和定义，深层存在性和 universal property 已登记为外部输入。

## 练习

**练习 2.1.** 验证乘法形式群 $F_m(X,Y)=X+Y+XY$ 的结合律。

**练习 2.2.** 在 $K(n)_*$ 中写出 $v_n^3+v_n^3$ 是否可逆，分别讨论 $p=2$ 与 $p\ne2$。

**练习 2.3.** 说明为什么从 $BP_*$ 到 $BP\langle n\rangle_*$ 的环商不自动给出唯一的 $\mathbb E_\infty$-ring quotient。
