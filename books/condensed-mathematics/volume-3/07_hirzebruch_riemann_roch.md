# 第七章：Euler characteristic、特征类与 Riemann--Roch

$R\Gamma(X,E)$ 是 perfect 复形时，全部上同调可以压成一个整数

$$
\chi(X,E)=\sum_i(-1)^i\dim_\mathbb C H^i(X,E).
$$

这个整数既是有限复形中恒等态射的迹，也是 $K$-理论上的可加函数。
Hirzebruch--Riemann--Roch 断言它还等于
$\operatorname{ch}(E)\operatorname{td}(T_X)$ 的顶次积分；Grothendieck 形式则要求该
恒等式与任意 proper pushforward 相容。特征类和推前相容性是深层几何输入，不能从
analytic 范畴的存在性形式推出。

本章先用 splitting principle 组织 Chern character 和 Todd class 的形式代数，再把
GRR 保留为外部输入，并完整推出到点映射的 HRR、$K^0$ 可加性和 categorical trace
解释。最后将第四章算出的 $\mathbb P^1$ 上同调与特征类积分逐项比较，给出输入、步骤
和同一个输出 $d+1$。

## 7.1 $K^0$ 与 Euler characteristic

设 $X$ 是紧复流形。向量丛 Grothendieck 群 $K^0(X)$ 由全纯向量丛的同构类生成，并对
每个短正合列

$$
0\to E'\to E\to E''\to0
$$

施加关系 $[E]=[E']+[E'']$。

**命题 7.1.** coherent cohomology finite-dimensional 时，

$$
\chi_X:K^0(X)\longrightarrow\mathbb Z,
\qquad
[E]\longmapsto\sum_i(-1)^i\dim H^i(X,E)
$$

是群同态。

**证明.** 短正合列给出有限维上同调长正合列。把长正合列截成有限个短正合列，或对其
kernel 与 image 逐项使用维数可加性，所有中间维数在交错和中两两相消，得到

$$
\chi(X,E)=\chi(X,E')+\chi(X,E'').
$$

所以 $\chi_X$ 尊重 $K^0$ 的定义关系。证毕。

## 7.2 Chern character 的形式代数

**外部输入定理 7.2（Chern 类与 splitting principle）.** 每个复向量丛 $E$ 有自然
Chern 类 $c_i(E)$，满足 Whitney 乘法公式。并存在
$\pi:Y\to X$，使 $\pi^*:H^*(X,\mathbb Q)\to H^*(Y,\mathbb Q)$ 单射，且
$\pi^*E$ 有线丛分级商。若这些线丛的一阶 Chern 类为
$x_1,\ldots,x_r$，可形式地称它们为 $E$ 的 Chern roots。

**定义 7.3.** 置

$$
\operatorname{ch}(E)=\sum_{j=1}^r e^{x_j},
\qquad
\operatorname{td}(E)
=
\prod_{j=1}^r\frac{x_j}{1-e^{-x_j}}.
$$

这些是 Chern roots 的对称幂级数，且 $X$ 维数有限，故每个 cohomological degree 只含
有限项；splitting principle 的单射性使它们从 $Y$ 唯一下降到 $X$。

**命题 7.4.** 对向量丛 $E,F$，

$$
\operatorname{ch}(E\oplus F)
=
\operatorname{ch}(E)+\operatorname{ch}(F),
$$

$$
\operatorname{ch}(E\otimes F)
=
\operatorname{ch}(E)\operatorname{ch}(F),
$$

且

$$
\operatorname{td}(E\oplus F)
=
\operatorname{td}(E)\operatorname{td}(F).
$$

**证明.** 拉回到共同的 splitting space。若 $E,F$ 的 roots 分别为 $x_i,y_j$，
直和的 roots 是二者并集，所以 Chern character 的和式相加，Todd 因子的乘积相乘。
张量积的 roots 为 $x_i+y_j$，故

$$
\operatorname{ch}(E\otimes F)
=
\sum_{i,j}e^{x_i+y_j}
=
\left(\sum_i e^{x_i}\right)
\left(\sum_j e^{y_j}\right).
$$

$\pi^*$ 单射，三式下降到 $X$。证毕。

特别地，$\operatorname{ch}$ 定义环同态

$$
\operatorname{ch}:K^0(X)\to H^{\mathrm{even}}(X,\mathbb Q).
$$

对复曲线，高于二次的 cohomology 消失，因而

$$
\operatorname{td}(T_X)=1+\frac12c_1(T_X).
$$

## 7.3 GRR 输入与到点映射

**外部输入定理 7.5（Grothendieck--Riemann--Roch，光滑绝对形式）.** 设 $X,Y$
是光滑复代数簇，$f:X\to Y$ proper，且 $E\in K^0(X)$；或处在相应的紧复几何范围，
并已知 $Rf_*E$ 定义 $K^0(Y)$ 中的类。则

$$
\operatorname{ch}(Rf_*E)\operatorname{td}(T_Y)
=
f_*\bigl(
\operatorname{ch}(E)\operatorname{td}(T_X)
\bigr)
$$

在 $H^{\mathrm{even}}(Y,\mathbb Q)$ 中成立。若 $X$ 或 $Y$ 奇异，必须改用
$G$-理论、Chow 群或 perfect complex 的版本，并把绝对切丛公式替换为相应的
virtual tangent/relative Todd class 陈述；本定理的显示公式不覆盖该情形。

deformation to the normal cone、Chern character 的几何构造和推前相容是该输入的
深层部分，见附录 AE、AK 与 AP。本书不把下列形式推论冒充为这些构造的证明。

**定理 7.6（HRR）.** 若 $X$ 光滑 proper，$E$ 为向量丛，则

$$
\chi(X,E)
=
\int_X\operatorname{ch}(E)\operatorname{td}(T_X).
$$

**证明.** 对结构映射 $f:X\to *$ 应用输入定理 7.5。点的 tangent bundle 为零，
$\operatorname{td}(T_*)=1$。在
$K^0(*)\cong\mathbb Z$ 中，

$$
[Rf_*E]
=
\sum_i(-1)^i[H^i(X,E)]
=
\chi(X,E),
$$

而点上的 Chern character把有限维向量空间送到其维数。因此 GRR 左侧为
$\chi(X,E)$，右侧的 $f_*$ 正是取顶次积分，得到公式。证毕。

**命题 7.7.** HRR 两侧都是 $K^0(X)\to\mathbb Q$ 的群同态。

**证明.** 左侧由命题 7.1。右侧中 $\operatorname{ch}$ 可加，乘以固定类
$\operatorname{td}(T_X)$ 和积分都是线性映射。因此它也尊重 $K^0$ 关系。证毕。

所以若 $K^0(X)$ 有一组已知生成元，只需在生成元上核验 HRR；这是一条接受特征类输入
后的完整形式化约。

## 7.4 Euler characteristic 是导出迹

设 $C\in D^b(\mathbb C)$ 有有限维 cohomology。第四章命题 4.6 给出

$$
C\simeq\bigoplus_iH^i(C)[-i].
$$

**命题 7.8.** categorical trace 满足

$$
\operatorname{Tr}(\operatorname{id}_C)
=
\sum_i(-1)^i\dim H^i(C).
$$

**证明.** 对次数零有限维向量空间 $V$，evaluation/coevaluation 的基计算给
$\operatorname{Tr}(\operatorname{id}_V)=\dim V$。对 shift $V[-1]$，对称幺半
导出范畴的 Koszul 符号使 evaluation 绕过一个奇次数因子，trace 变为
$-\dim V$；归纳得 $V[-i]$ 的 trace 为 $(-1)^i\dim V$。trace 对有限直和相加，代入
上述分解即得公式。证毕。

若第三章的 analytic 实现强幺半且保持 dualizable objects，它保持 evaluation、
coevaluation 和 trace。因此在 condensed/analytic 范畴中，
$R\Gamma_{\mathrm{an}}(X,E)$ 的恒等态射之迹仍为 $\chi(X,E)$。输入定理 7.5 的
进一步内容正是把这个 categorical trace 与 characteristic-class 积分比较；强幺半
形式本身不能构造 Todd correction。

## 7.5 Worked example：$\mathbb P^1$ 上的两种 $d+1$

令

$$
H=c_1(\mathcal O(1))\in H^2(\mathbb P^1,\mathbb Q),
\qquad
H^2=0,
\qquad
\int_{\mathbb P^1}H=1.
$$

对任意整数 $d$，

$$
\operatorname{ch}(\mathcal O(d))
=e^{dH}=1+dH.
$$

Euler sequence 给
$c_1(T_{\mathbb P^1})=2H$，所以

$$
\operatorname{td}(T_{\mathbb P^1})
=
1+\frac12c_1(T_{\mathbb P^1})
=
1+H.
$$

相乘并取顶次项：

$$
\int_{\mathbb P^1}
(1+dH)(1+H)
=
\int_{\mathbb P^1}
\bigl(1+(d+1)H\bigr)
=
d+1.
$$

第四章的 Laurent--Čech 计算给

$$
h^0(\mathcal O(d))-h^1(\mathcal O(d))=d+1
$$

对所有 $d\in\mathbb Z$ 成立。因此左侧的同调输入、右侧的 Chern/Todd 步骤和输出
整数完全吻合。计算失败的条件也清楚：若 $X$ 非 proper，上同调可能不有限，积分到点
和 Euler characteristic 都未必按上述形式定义；若 $X$ 奇异，则 $T_X$ 应替换为适当
tangent complex，GRR 的陈述也需相应修正。$\mathbb P^n$ 的系数提取见附录 U。

## 7.6 从数值公式回到函子关系

HRR 把一个对象的导出迹转成特征数，GRR 则要求这种比较沿 proper pushforward 相容。
要同时表达 proper 与非 proper 映射、紧支撑、对偶和内部 Hom，需要的不只是一个 trace，
而是完整的六函子关系。第八章将以开嵌入
$\mathbb C^\times\hookrightarrow\mathbb C$ 为具体模型区分 $f_!$ 与 $f_*$，再说明
base change、projection formula 和 duality 之间留下哪些精确开放问题。

## 练习

**练习 7.1.** 用 Chern roots 证明
$\operatorname{ch}(E^\vee)=\sum_ie^{-x_i}$，并写出到 cohomological degree $4$ 的展开。

**练习 7.2.** 从 Euler sequence 推出
$c_1(T_{\mathbb P^1})=2H$，补全例 7.5 中 Todd class 的计算。

**练习 7.3.** 对二项复形
$0\to\mathbb C^a\xrightarrow{d}\mathbb C^b\to0$，直接比较恒等态射的 supertrace 与
cohomology 的交错维数。
