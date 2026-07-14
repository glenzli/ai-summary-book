# 第一章：复解析空间的凝聚语言

开圆盘上的全纯函数不只形成一个复向量空间。若一列函数在每个紧子集上一致收敛，其
极限仍全纯；求导、限制和乘法也对这种收敛连续。只保留底层向量空间会丢掉这些信息，
而把所有连续族 $S\to\mathcal O(U)$ 一并记录，才得到后续 liquid 模和导出解析几何
需要的对象。本章从局部零点模型出发，先固定复解析空间与截面拓扑，再实际计算一个由
profinite 参数空间索引的全纯函数族。

第二卷提供的 liquid 实现定理适用于 Banach 与 Fréchet 空间，但从逐开集实现到解析
结构层和派生范畴仍是深层建模输入。正文会证明局部函数空间的 Fréchet 性与参数族的
具体描述，并把更强的 sheaf 化、张量和相干嵌入精确标作 Clausen--Scholze 输入。

## 1.1 局部零点模型

**定义 1.1（复解析空间）.** 复解析空间是一个局部环化空间
$(X,\mathcal O_X)$，使每个点 $x\in X$ 都有开邻域 $V$，以及开集
$U\subset\mathbb C^n$ 和有限生成理想 sheaf
$\mathcal I\subset\mathcal O_U$，满足

$$
(V,\mathcal O_X|_V)
\cong
\bigl(Z(\mathcal I),\,
(\mathcal O_U/\mathcal I)|_{Z(\mathcal I)}\bigr).
$$

允许 $\mathcal I$ 含幂零元时可得到非约化解析空间。若 $\mathcal I=0$ 且局部模型
都是开集 $U\subset\mathbb C^n$，则 $X$ 是复流形。

**例 1.2（双点）.** 在圆盘 $\Delta$ 上取理想 $(z^2)$。底层零点集只有原点，但局部环

$$
\mathbb C\{z\}/(z^2)
$$

含有非零幂零元 $\bar z$。连续函数的点值看不见 $\bar z$，结构层却能看见；因此复
解析空间不能只由底层拓扑空间定义。

复流形上的结构层截面为全纯函数。若 $E\to X$ 是秩 $r$ 全纯向量丛，其全纯截面层记为
$\mathcal O(E)$；在全纯平凡化上，它等于 $\mathcal O_X^{\oplus r}$。

## 1.2 全纯函数的 Fréchet 拓扑

令 $\Delta^n=\{z\in\mathbb C^n:|z_i|<1\}$。取递增数列
$0<r_1<r_2<\cdots<1$ 且 $r_m\to1$，在 $\mathcal O(\Delta^n)$ 上定义半范数

$$
p_m(f)=\sup_{|z_i|\le r_m}|f(z)|.
$$

**命题 1.3.** 这些半范数使 $\mathcal O(\Delta^n)$ 成为 Fréchet 空间；所得拓扑与
紧开一致收敛拓扑相同。

**证明.** 闭多圆盘 $\{|z_i|\le r_m\}$ 穷尽 $\Delta^n$ 的紧子集，因此这些半范数
生成紧开拓扑，并给出可数可度量局部凸结构。若 $(f_j)$ 对所有 $p_m$ 都是 Cauchy，
则它在每个闭小多圆盘上一致收敛到连续函数 $f_m$；不同 $m$ 的极限由唯一性相容，故
拼成 $f:\Delta^n\to\mathbb C$。Weierstrass 一致收敛定理说明紧集上一致极限 $f$
仍全纯，而且 $p_m(f_j-f)\to0$。所以空间完备。证毕。

限制映射
$\mathcal O(U)\to\mathcal O(V)$、$V\subset U$，对紧开拓扑连续；乘法为连续双线性
映射。由此 $U\mapsto\mathcal O(U)$ 是 Fréchet 代数值的 sheaf，而不只是向量空间值
sheaf。

## 1.3 一个由测试空间参数化的函数族

对紧 Hausdorff 空间 $S$，定义

$$
\underline{\mathcal O(\Delta)}(S)
=
\operatorname{Cont}(S,\mathcal O(\Delta)).
$$

**例 1.4（输入、识别与输出）.** 输入一个连续映射
$\Phi:S\to\mathcal O(\Delta)$。令

$$
F:S\times\Delta\to\mathbb C,
\qquad
F(s,z)=\Phi(s)(z).
$$

则 $F$ 连续，且每个切片 $F(s,-)$ 全纯。反过来，若 $F$ 连续、各切片全纯，并且对每个
$r<1$，映射 $s\mapsto F(s,-)$ 在范数
$\sup_{|z|\le r}|-|$ 下连续，则它给出唯一的连续
$\Phi:S\to\mathcal O(\Delta)$。

**证明.** 求值映射
$\mathcal O(\Delta)\times\Delta\to\mathbb C$ 对紧开拓扑连续，故第一方向由复合
得到。反方向的假设恰好说明对生成 Fréchet 拓扑的每个半范数 $p_m$，$\Phi$ 都连续；
因此它对该初始拓扑连续。复合两个构造后，逐点都有
$F(s,z)=\Phi(s)(z)$，故二者互逆。证毕。

当 $S=\{1,\ldots,k\}$ 有限离散时，输出退化为

$$
\underline{\mathcal O(\Delta)}(S)
\cong
\mathcal O(\Delta)^k.
$$

当 $S$ 无限 profinite 时，连续性要求函数族在每个闭小圆盘上一致随参数变化；这比
任意集合族严格得多，也正是凝聚化保留的拓扑信息。

## 1.4 从逐开集凝聚化到解析结构层

对复流形 $X$，逐开集定义

$$
\underline{\mathcal O}_X(U)(S)
=
\operatorname{Cont}(S,\mathcal O_X(U)),
\qquad
S\in\mathbf{ProFin}_\kappa.
$$

限制映射的连续性使它在 $U$ 上反变，在 $S$ 上满足凝聚 sheaf 条件。对有限开覆盖，
经典全纯函数的粘合等化子与紧开拓扑相容，所以该公式给出局部可直接检查的模型。

**外部输入定理 1.5（Clausen--Scholze 解析建模）.** 对
[附录 AR.1](AR_clausen_scholze_complex_geometry_core_theorem_atlas.md) 登记的复解析
对象 $X$，上述局部模型可组织成结构对象 $\mathcal O_X^{\mathrm{an}}$ 和稳定
analytic 派生范畴 $D_{\mathrm{an}}(X)$。Fréchet 截面的 liquid 实现与限制、张量及导出
全局截面相容，相干解析层有忠实的派生解释。

该定理包含的内容强于命题 1.3 和例 1.4：它必须验证不同坐标图上的下降、解析环测度
公理以及导出层面的正合性。本书不重建这些深层步骤。第二章将在接受该输入后，把相干
层的有限表示和有限局部自由复形放入 $D_{\mathrm{an}}(X)$。

## 1.5 局部模型留下的问题

函数空间的凝聚化保存了连续参数族，analytic 建模输入又把这些对象沿开覆盖粘合起来。
但“一个 $\mathcal O_X$-模局部由有限个生成元和关系给出”仍是另一种有限性。它不由
Fréchet 性推出，也不等同于导出范畴中的紧性。下一章将用相干层、perfect complex 和
一个点支撑层的完整 resolution 区分这三种概念。

## 练习

**练习 1.1.** 检查 $\mathbb C\{z\}/(z^2)$ 的唯一素理想为 $(\bar z)$，并说明其
底层点空间为何不能恢复幂零元。

**练习 1.2.** 对有限离散 $S$ 证明
$\operatorname{Cont}(S,\mathcal O(\Delta))\cong\mathcal O(\Delta)^S$，并写出
乘法在两侧的对应。

**练习 1.3.** 设 $S$ 为 profinite 空间。证明若
$\Phi:S\to\mathcal O(\Delta)$ 连续，则对每个 $r<1$，集合
$\{\Phi(s)|_{|z|\le r}:s\in S\}$ 在一致范数下紧。
