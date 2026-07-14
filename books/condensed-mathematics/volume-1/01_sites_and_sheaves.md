# 第一章：站点、覆盖与 sheaf 条件

给定若干局部截面 $s_i\in F(U_i)$，两两在
$U_i\times_U U_j$ 上相等，并不单凭“局部”二字保证存在全局截面。必须先规定哪些
$U_i\to U$ 可以称为覆盖，再把相容性、存在性与唯一性写成可在任意范畴中检查的
公式。凝聚数学所用的有限联合满射正适合由覆盖族生成的 Grothendieck 预拓扑处理；
sheaf 条件则会成为一个等化子，而不是含混的粘合口号。

下面默认读者熟悉范畴、反变函子、自然变换、有限极限、纤维积和等化子。我们不从
sieve 的最大一般性出发，因为后续真正需要的是覆盖在恒等、拉回和复合下的稳定性，
以及这些公理如何让可表预层和紧 Hausdorff 测试空间进入同一套语言。

## 1.0 大小与空覆盖约定

本章的 \(\mathcal C\) 是第一卷附录 A 所固定 universe 中的小范畴，值域
\(\mathbf{Set}\) 指同一工作层级的集合范畴。覆盖族的指标集总是有限集，**允许为空集**。
若 \(\mathcal C\) 有初对象 \(0\)，空族覆盖 \(0\) 时，sheaf 条件包含
\(F(0)\cong *\)。这个退化条件在 condensed site 上给出 \(F(\varnothing)=*\)，
不能从只检查非空覆盖的版本中省略。

## 1.1 预层

**定义 1.1.** 设 $\mathcal C$ 为上述小范畴。$\mathcal C$ 上的集合值预层
（presheaf）是反变函子

$$
F:\mathcal C^{\operatorname{op}}\to \mathbf{Set}.
$$

若 $f:V\to U$ 是 $\mathcal C$ 中的态射，则 $F$ 给出限制映射

$$
F(f):F(U)\to F(V).
$$

通常记 $s|_V=F(f)(s)$。

**例 1.2.** 对任意对象 $U\in\mathcal C$，可表预层定义为

$$
h_U(V)=\operatorname{Hom}_{\mathcal C}(V,U).
$$

若 $g:V'\to V$，则限制映射为预合成：

$$
h_U(V)\to h_U(V'),\qquad
\varphi\mapsto \varphi\circ g.
$$

这给出 Yoneda 嵌入

$$
y:\mathcal C\to \widehat{\mathcal C},
\qquad U\mapsto h_U.
$$

## 1.2 覆盖族与预拓扑

**定义 1.3.** 设 $\mathcal C$ 有有限纤维积。一个 Grothendieck 预拓扑由每个对象 $U$ 的一族覆盖族

$$
\{U_i\to U\}_{i\in I},\qquad |I|<\infty,
$$

组成，并满足：

1. 恒等态射 $\{\operatorname{id}_U:U\to U\}$ 是覆盖。
2. 若 $\{U_i\to U\}$ 是覆盖，且 $V\to U$ 是任意态射，则拉回族
   $$
   \{V\times_U U_i\to V\}_{i\in I}
   $$
   是覆盖。
3. 若 $\{U_i\to U\}$ 是覆盖，且对每个 $i$，$\{U_{ij}\to U_i\}_j$ 是覆盖，则复合族
   $$
   \{U_{ij}\to U_i\to U\}_{i,j}
   $$
   是覆盖。

这里 \(I=\varnothing\) 被允许；此时第 2、3 项按空族的通常范畴论约定解释。

配备预拓扑的范畴称为站点（site），记作 $(\mathcal C,J)$。

本书中最重要的覆盖族是有限覆盖族，尤其是有限联合满射族。有限性会使 sheaf 条件写成有限乘积和有限纤维积上的等化子。

## 1.3 匹配族与 sheaf 条件

设 $F$ 是 $\mathcal C$ 上的预层，$\{U_i\to U\}_{i\in I}$ 是覆盖族。一个匹配族（matching family）是元素族

$$
(s_i)_{i\in I}\in \prod_i F(U_i)
$$

满足对任意 $i,j$，在纤维积 $U_i\times_U U_j$ 上限制相同：

$$
s_i|_{U_i\times_U U_j}
=
s_j|_{U_i\times_U U_j}.
$$

这可以写成等化子条件：

$$
\prod_i F(U_i)
\rightrightarrows
\prod_{i,j}F(U_i\times_U U_j).
$$

**定义 1.4.** 预层 $F$ 是 separated presheaf，如果对每个覆盖族 $\{U_i\to U\}$，映射

$$
F(U)\to \prod_i F(U_i)
$$

是单射到匹配族集合中。等价地，两个截面若在覆盖的每个部分上相等，则它们本身相等。

**定义 1.5.** 预层 $F$ 是 sheaf，如果对每个覆盖族 $\{U_i\to U\}$，序列

$$
F(U)\longrightarrow \prod_i F(U_i)
\rightrightarrows
\prod_{i,j}F(U_i\times_U U_j)
$$

是 $\mathbf{Set}$ 中的等化子。

换句话说，每个匹配族存在唯一粘合（gluing）：

$$
s\in F(U),\qquad s|_{U_i}=s_i.
$$

**注 1.6.** sheaf 条件包含两个部分：存在性与唯一性。separated presheaf 只要求唯一性，不要求每个匹配族都能粘合。

**边界 1.6A（空对象）.** 假设 \(\mathcal C\) 有初对象 \(0\)，且空族覆盖
\(0\)。定义 1.5 对该覆盖变成

$$
F(0)\longrightarrow \prod_{i\in\varnothing}F(U_i)=*.
$$

右侧第二个乘积也为单点集，所以等化子是 \(*\)。因此 sheaf 必须满足
\(F(0)\cong *\)。反之，只检查非空覆盖不会检测这个条件；例如在
\(\mathbf{CHaus}_\kappa\) 上，可在所有非空对象取单点集、在空对象取一个多点集，
并把所有到空对象的限制映射选为同一基点，由此得到满足所有非空覆盖条件但不满足空覆盖
条件的预层。

## 1.4 子典范性

**定义 1.7.** 若站点 $(\mathcal C,J)$ 上每个可表预层 $h_U$ 都是 sheaf，则称该拓扑是子典范的（subcanonical）。

子典范性很重要，因为它保证原范畴 $\mathcal C$ 可以通过 Yoneda 嵌入忠实地看作 sheaf 范畴的一部分：

$$
\mathcal C\hookrightarrow \operatorname{Sh}(\mathcal C,J).
$$

凝聚数学中，紧 Hausdorff 空间站点的有限联合满射拓扑是子典范的。证明会在第二章给出，因为它依赖紧 Hausdorff 空间中满射闭映射是商映射的事实。

## 1.5 等化子条件的展开

为了避免后续把 sheaf 条件当作黑箱，我们写出最常用的二元覆盖情形。设 $\{U_1\to U,U_2\to U\}$ 是覆盖。则 $F$ 的 sheaf 条件要求

$$
F(U)\to F(U_1)\times F(U_2)
\rightrightarrows
F(U_1\times_U U_1)
\times F(U_1\times_U U_2)
\times F(U_2\times_U U_1)
\times F(U_2\times_U U_2)
$$

为等化子。

若覆盖映射是单态，例如开覆盖中的开嵌入，则 $U_i\times_U U_i\simeq U_i$，条件常简化为熟悉的交集相容性。但在凝聚数学中覆盖通常不是单态，而是联合满射。因此必须保留所有纤维积项。

这点非常关键。凝聚数学的覆盖不是“开子集覆盖”的简单替换；它允许一般的满射测试对象。于是重叠不是集合交，而是纤维积。

## 1.6 例子：拓扑空间上的开覆盖 sheaf

设 $X$ 是拓扑空间，令 $\operatorname{Open}(X)$ 为开集按包含关系形成的范畴。覆盖族是开覆盖 $\{U_i\subset U\}$。则预层

$$
F:\operatorname{Open}(X)^{\operatorname{op}}\to \mathbf{Set}
$$

是通常意义上的 sheaf，当且仅当对每个开覆盖，局部截面能在交集上相容时唯一粘合。

这个例子帮助理解 sheaf 条件，但凝聚数学的站点不是 $\operatorname{Open}(X)$。
凝聚数学使用固定层级的测试空间范畴 $\mathbf{CHaus}_\kappa$，覆盖族是有限联合满射族。

## 1.7 等化子等待一个具体站点

预层的反变性把限制映射组织起来，覆盖等化子再精确区分哪些预层能够从相容局部数据
恢复全局数据。若拓扑子典范，可表预层也通过同一检查；这正是以后把测试空间视作
凝聚对象的依据。抽象框架至此只缺一项输入：一个具有有限纤维积、稳定覆盖和有效商
性质的具体测试范畴。下一章构造的紧 Hausdorff 站点将使公式

$$
\mathbf{CondSet}_\kappa
=\operatorname{Sh}(\mathbf{CHaus}_\kappa,J_{\operatorname{surj}}),
$$

中的每个符号都有确定含义。

## 练习

**练习 1.1.** 证明：若 $F$ 是 sheaf，则 $F$ 是 separated presheaf。

**练习 1.2.** 设 $\mathcal C$ 有纤维积。把三元覆盖 $\{U_1,U_2,U_3\}\to U$ 的 sheaf 条件完整写成等化子。

**练习 1.3.** 设 $X$ 为拓扑空间。说明为什么通常开覆盖 sheaf 条件中的“交集”其实是纤维积 $U_i\times_U U_j$。

**练习 1.4.** 给出一个预层在覆盖上满足唯一性但不满足存在性的例子。

**练习 1.5.** 在允许空覆盖的 condensed site 上证明
\(F(\varnothing)=*\)，并说明该结论为何不是任一非空满射覆盖的推论。
