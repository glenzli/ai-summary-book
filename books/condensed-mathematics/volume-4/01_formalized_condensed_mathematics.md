# 第一章：匹配族的机器可检规格

在纸面上，“相容局部截面唯一粘合”只有一句话；在 proof assistant 中，覆盖的索引
类型、纤维积投影、两条限制映射、空覆盖和 universe 层级都必须显式存在。最合适的
起点不是列举可以形式化的名词，而是选一个核心命题，把输入数据拆到每个中间对象都有
确定类型。本章选择有限覆盖的 sheaf 等化子，并用三点集合的二元覆盖完整执行一次。

这些规格不绑定某个 Lean 版本。它们描述的是可直接翻译到范畴论库的数学接口：
Cover 提供有限态射族，Match 提供相容截面的 subtype，glue 提供逆映射，最终输出一个
等化子或 limit 证明。紧 Hausdorff 可表 sheaf 的连续性与一般基站点比较则在这个接口
之上逐层加入。

## 1.1 输入数据与大小层级

固定 Grothendieck universe $\mathcal U$。设 $\mathcal C$ 是
$\mathcal U$-小范畴，具有初对象和有限纤维积；设

$$
F:\mathcal C^{op}\to\mathbf{Set}_{\mathcal U}
$$

为预层。一个以有限类型 $I$ 索引的覆盖数据包括对象 $U$、对象族 $U_i$、态射
$p_i:U_i\to U$，以及该族属于给定预拓扑的证明。

对 $i,j\in I$，选择拉回方块

$$
\begin{matrix}
U_{ij}&\xrightarrow{\pi_j}&U_j\\
\downarrow\pi_i&&\downarrow p_j\\
U_i&\xrightarrow{p_i}&U .
\end{matrix}
$$

机器证明必须记录这不是任意交换方块，而是 pullback；否则后续共同限制没有稳定类型。

空索引 $I=\varnothing$ 也要保留。若空族覆盖初对象 $0$，sheaf 条件将强制
$F(0)$ 为单点集合，不能用“覆盖非空”的实现悄悄删掉。

## 1.2 匹配族与 restriction 映射

**定义 1.1.** 覆盖 $\mathcal U=(U_i\to U)_{i\in I}$ 的匹配族集合为

$$
\operatorname{Match}_F(\mathcal U)
=
\left\{
(s_i)\in\prod_iF(U_i):
F(\pi_i)(s_i)=F(\pi_j)(s_j)
\text{ in }F(U_{ij})\ \forall i,j
\right\}.
$$

预层函子性给 restriction

$$
\rho:F(U)\longrightarrow\operatorname{Match}_F(\mathcal U),
\qquad
s\longmapsto(F(p_i)(s))_i.
$$

它确实落在 matching subtype，因为
$p_i\pi_i=p_j\pi_j$，施加反变函子后两条复合限制相等。

**命题 1.2.** $F$ 在覆盖 $\mathcal U$ 上满足 sheaf 条件，当且仅当 $\rho$ 为双射。

**证明.** 若满足 sheaf 条件，每个匹配族有唯一粘合 $s\in F(U)$；令
$\operatorname{glue}$ 把匹配族送到该 $s$。粘合的限制条件给
$\rho\circ\operatorname{glue}=\operatorname{id}$，唯一性给
$\operatorname{glue}\circ\rho=\operatorname{id}$，所以 $\rho$ 双射。

反之，若 $\rho$ 双射，对匹配族 $m$ 取唯一原像 $\rho^{-1}(m)$；它的限制正是 $m$，
给出存在性。若两个全局截面粘合同一匹配族，它们在 $\rho$ 下像相同，由单射性相等，
给出唯一性。证毕。

形式化时，证明可拆成三个稳定接口：restriction 落在匹配族、glue 后再 restriction 为
恒等、restriction 后再 glue 为恒等。最后两项比直接操作“唯一存在”更容易被后续
等化子证明复用。

## 1.3 Worked example：三点集合的二元覆盖

在有限集合范畴中取

$$
U=\{0,1,2\},\qquad
U_0=\{0,1\},\qquad
U_1=\{1,2\},
$$

覆盖态射为包含。对固定集合 $T$，取可表预层

$$
h_T(A)=\operatorname{Map}(A,T).
$$

交 $U_{01}=\{1\}$。输入一个匹配族就是两个函数

$$
f_0:\{0,1\}\to T,
\qquad
f_1:\{1,2\}\to T
$$

满足 $f_0(1)=f_1(1)$。粘合步骤逐点定义

$$
f(0)=f_0(0),\qquad
f(1)=f_0(1)=f_1(1),\qquad
f(2)=f_1(2).
$$

输出是唯一函数 $f:U\to T$，并给出显式双射

$$
T^3
\xrightarrow{\sim}
T^2\times_TT^2.
$$

若输入在交点不相容，即 $f_0(1)\ne f_1(1)$，则它不属于
$\operatorname{Match}$，粘合函数不存在。这一失败不是程序异常，而是等化子排除的
数学分支。

## 1.4 等化子的范畴表述

若目标范畴 $\mathcal E$ 有有限极限，集合乘积可替换为 $\mathcal E$ 中乘积。两条箭头

$$
\prod_iF(U_i)
\rightrightarrows
\prod_{i,j}F(U_{ij})
$$

分别由 $\pi_i$ 和 $\pi_j$ 的限制诱导。命题 1.2 的输出可表成：
$F(U)$ 连同 $\rho$ 是这对箭头的 equalizer cone，亦即一个 limit 的普遍性证明。

这个版本需要目标范畴的有限乘积和等化子。若只假设集合值，标准范畴结构自动提供；
若目标是阿贝尔群、链复形或谱，则必须在相应范畴中实例化 limit，并决定使用严格等化子
还是 homotopy limit。第八章解释谱值时为何需后者。

## 1.5 紧 Hausdorff 可表对象的新增义务

把例 1.3 从有限集合换成 $\mathbf{CHaus}$ 后，逐点粘合仍给集合映射，但连续性需要
额外证明。对有限联合满射覆盖 $S_i\to S$，令

$$
q:\coprod_iS_i\to S.
$$

源紧、目标 Hausdorff，所以 $q$ 是闭满射，因而为 quotient map。匹配族
$f_i:S_i\to T$ 合成连续映射
$g:\coprod_iS_i\to T$；匹配条件说明 $g$ 在 $q$ 的纤维上常值，故唯一因子化为集合
映射 $f:S\to T$。由 quotient map 的定义，$g=fq$ 连续推出 $f$ 连续。输出即

$$
h_T(S)\xrightarrow{\sim}\operatorname{Match}_{h_T}(\mathcal U).
$$

这里需分别形式化有限余并紧、紧集在 Hausdorff 空间中闭、闭满射为 quotient，以及
matching 等价于纤维上常值。任何一项缺失，都只能得到集合值粘合，不能得到
$\mathbf{CHaus}$ 中态射。

## 1.6 基站点比较的可复用分解

对全忠实 $i:\mathcal D\hookrightarrow\mathcal C$，restriction 要成为 sheaf 范畴
等价，至少需要：$\mathcal D$-对象覆盖每个 $\mathcal C$-对象；覆盖拉回和交对象仍可
由 $\mathcal D$ 覆盖；任意两个选择有共同细化。机器证明可用切片范畴上的右 Kan
延拓组织：

$$
\widetilde H(U)
=
\varprojlim_{(D\to U)\in(\mathcal D/U)^{op}}H(D).
$$

稳定基与共同细化使这个极限可由任一 $\mathcal D$-覆盖上的 matching equalizer
计算。先证明切片索引的大小合法，再由复合函子
$\mathcal D/V\to\mathcal D/U$ 定义 restriction，继而证明复合覆盖上的 matching
同时计算 $\widetilde H(U)$ 和各 $\widetilde H(U_a)$ 的粘合。最后构造 restriction 与
延拓的 unit/counit。第二章给出有限稳定基版本的数学证明，附录 A 与 F 保留逐引理规格。

大小失败会首先出现在切片范畴 $\mathcal D/U$：若站点不是小范畴，以上极限未必落在当前
universe。对 $\mathbf{CHaus}$ 必须先固定 universe 和小骨架。Gleason、Nöbeling 及
solid 张量等深层定理则应作为具有完整类型的外部参数接入，不能用未证明公理混在
等化子基础层中。

## 1.7 从规格到下一次计算

匹配族 subtype、restriction、glue 与两个逆律构成最小可复用单元。第二章会把它用于
集合值和阿贝尔群值 sheaf、Čech 微分、可表凝聚对象及基子站点比较；第三章再把同样的
“输入分解、构造复形、取同调”结构用于 Ext 与 Tor。形式化的价值由这些中间对象能否
独立复用决定，而不是由名词覆盖数量决定。

## 练习

**练习 1.1.** 对例 1.3 写出 $\rho$ 和 $\operatorname{glue}$ 的两个复合，逐点验证
它们为恒等映射。

**练习 1.2.** 在空族覆盖初对象 $0$ 的情形计算
$\operatorname{Match}_F(\varnothing)$，推出 sheaf 必须满足 $F(0)\cong *$。

**练习 1.3.** 把第 1.5 节证明拆成四个引理，并为每个引理列出定义域、值域和唯一使用
的拓扑假设。
