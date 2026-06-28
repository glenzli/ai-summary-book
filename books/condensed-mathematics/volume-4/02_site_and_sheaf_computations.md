# 第二章：站点、覆盖与 sheaf 条件的计算

## 本章目标

本章把 sheaf 条件从一句定义展开成可检验的等化子、Čech 复形和站点比较命题。为避免隐藏集合论问题，本章默认站点 $\mathcal C$ 是小范畴，具有所需的有限纤维积；凝聚数学中遇到的大站点先通过 universe 或小骨架处理，见第一卷附录 A。

## 2.1 覆盖的匹配族

设 $\{U_i\to U\}_{i\in I}$ 是站点 $\mathcal C$ 中的有限覆盖，且 $F:\mathcal C^{op}\to\mathbf{Set}$ 是预层。记

$$
U_{ij}=U_i\times_U U_j .
$$

一个匹配族是元素族 $(s_i)_{i\in I}$，其中 $s_i\in F(U_i)$，并满足

$$
s_i|_{U_{ij}}=s_j|_{U_{ij}}
\quad\text{in }F(U_{ij})
$$

对所有 $i,j$ 成立。

**命题 2.1.1（有限覆盖的等化子形式）。** 预层 $F$ 在覆盖 $\{U_i\to U\}$ 上满足 sheaf 条件，当且仅当自然映射

$$
F(U)\longrightarrow
\prod_iF(U_i)
\rightrightarrows
\prod_{i,j}F(U_{ij})
$$

使 $F(U)$ 成为右侧两箭头的等化子。

**证明。** 自然映射把 $s\in F(U)$ 送到限制族 $(s|_{U_i})_i$。两条平行箭头分别把 $(s_i)_i$ 送到

$$
(s_i|_{U_{ij}})_{i,j},
\qquad
(s_j|_{U_{ij}})_{i,j}.
$$

因此等化子中的元素正是匹配族。sheaf 条件说每个匹配族存在唯一粘合 $s\in F(U)$，也就是说上面的映射在集合层面给出从 $F(U)$ 到匹配族集合的双射。反过来，若该映射是等化子，则每个匹配族有唯一原像，正是唯一粘合。证毕。

若 $F$ 取值于阿贝尔群，则同一个证明在 $\mathbf{Ab}$ 中成立，等化子是核：

$$
0\to F(U)\to\prod_iF(U_i)
\xrightarrow{d^0}
\prod_{i,j}F(U_{ij}),
\qquad
d^0((s_i))=(s_i|_{U_{ij}}-s_j|_{U_{ij}})_{i,j}.
$$

## 2.2 二元覆盖和三重交

对二元覆盖 $\{U_1\to U,U_2\to U\}$，命题 2.1.1 给出

$$
F(U)\cong
\left\{
(s_1,s_2)\in F(U_1)\times F(U_2)
\mid
s_1|_{U_{12}}=s_2|_{U_{12}}
\right\}.
$$

这里的 $U_{12}=U_1\times_U U_2$。若 $F$ 取值于阿贝尔群，则

$$
0\to F(U)\to F(U_1)\oplus F(U_2)
\xrightarrow{(a,b)\mapsto a|_{U_{12}}-b|_{U_{12}}}
F(U_{12})
$$

正合。

三元覆盖已经需要三重交来描述 Čech 微分。设 $U_{ijk}=U_i\times_UU_j\times_UU_k$。定义

$$
C^0=\prod_iF(U_i),\qquad
C^1=\prod_{i,j}F(U_{ij}),\qquad
C^2=\prod_{i,j,k}F(U_{ijk})
$$

以及

$$
(d^0s)_{ij}=s_j|_{U_{ij}}-s_i|_{U_{ij}},
$$

$$
(d^1t)_{ijk}
=t_{jk}|_{U_{ijk}}
-t_{ik}|_{U_{ijk}}
+t_{ij}|_{U_{ijk}} .
$$

**引理 2.2.1（Čech 微分平方为零）。** 对任意预层阿贝尔群 $F$，有 $d^1d^0=0$。

**证明。** 对 $s=(s_i)$，在 $U_{ijk}$ 上计算：

$$
\begin{aligned}
(d^1d^0s)_{ijk}
&=(s_k-s_j)-(s_k-s_i)+(s_j-s_i)\\
\\
&=0.
\end{aligned}
$$

这里每一项都表示进一步限制到 $U_{ijk}$ 后的截面。限制映射的函子性保证三项确实处在同一个群中。证毕。

## 2.3 可表 sheaf 的检查

设 $T\in\mathbf{CHaus}$，令

$$
h_T(S)=\operatorname{Hom}_{\mathbf{CHaus}}(S,T).
$$

凝聚数学采用的紧 Hausdorff 站点上，覆盖可取为有限族 $\{S_i\to S\}$，其诱导映射 $\coprod_iS_i\to S$ 为满射。

**命题 2.3.1（紧 Hausdorff 站点的可表预层是 sheaf）。** 对任意 $T\in\mathbf{CHaus}$，$h_T$ 是 sheaf。

**证明。** 设 $\{S_i\to S\}$ 为覆盖，并给定匹配族 $f_i:S_i\to T$。因为 $\coprod_iS_i\to S$ 满射，至多存在一个集合映射 $f:S\to T$ 使 $f|_{S_i}=f_i$。匹配条件说明若 $x_i\in S_i$ 与 $x_j\in S_j$ 映到同一点 $x\in S$，则 $f_i(x_i)=f_j(x_j)$，所以 $f$ 良定义。

剩下证明 $f$ 连续。有限余并 $\coprod_iS_i$ 是紧 Hausdorff，映射

$$
q:\coprod_iS_i\to S
$$

是从紧空间到 Hausdorff 空间的连续满射，因此是闭映射，特别是商映射。复合 $f\circ q$ 在每个 $S_i$ 上等于 $f_i$，故连续。由 $q$ 为商映射，$f$ 连续。于是匹配族唯一粘合，$h_T$ 是 sheaf。证毕。

这个证明同时解释了为什么“可表对象为 sheaf”在凝聚数学中不是形式口号，而是紧性、Hausdorff 性和覆盖定义共同作用的结果。

## 2.4 基子站点比较

许多计算不直接在 $\mathbf{CHaus}$ 上做，而在 profinite 或 extremally disconnected 对象上做。下面的命题给出常用比较形式。

**定理 2.4.1（稳定基子站点比较，有限型版本）。** 设 $i:\mathcal D\hookrightarrow\mathcal C$ 是全忠实子范畴。为使等化子公式类型完全明确，本定理采用稳定基假设。假设：

1. $\mathcal C$ 有有限纤维积；若 $D_1,D_2\in\mathcal D$ 且有映射 $D_1\to U\leftarrow D_2$，其中 $U\in\mathcal C$，则 $D_1\times_UD_2$ 仍属于 $\mathcal D$。
2. 每个 $U\in\mathcal C$ 都存在覆盖 $\{D_a\to U\}$，其中 $D_a\in\mathcal D$。
3. 若 $D\in\mathcal D$ 且 $\{U_i\to D\}$ 是 $\mathcal C$ 中覆盖，则存在 $\mathcal D$ 中覆盖 $\{D_b\to D\}$ 共同细化它。
4. $\mathcal D$ 上的拓扑由 $\mathcal C$ 限制而来。

则限制函子诱导范畴等价

$$
i^\ast:\operatorname{Sh}(\mathcal C)\xrightarrow{\sim}
\operatorname{Sh}(\mathcal D).
$$

**证明。** 先证全忠实。给定 $\mathcal C$ 上 sheaf $F,G$ 和一个在 $\mathcal D$ 上的态射 $\alpha:i^\ast F\to i^\ast G$。对任意 $U\in\mathcal C$，选覆盖 $\{D_a\to U\}$。由 sheaf 条件，

$$
F(U)\to \prod_aF(D_a)
\rightrightarrows
\prod_{a,b}F(D_a\times_U D_b)
$$

是等化子，$G$ 同理。稳定基假设保证 $D_a\times_UD_b$ 仍在 $\mathcal D$，所以 $\alpha$ 已经在这些交对象上定义；sheaf 的分离性和粘合性使 $\alpha$ 在 $D_a$ 上的定义唯一决定 $\alpha_U:F(U)\to G(U)$。共同细化假设保证不同覆盖的构造相容，所以得到唯一的 $\mathcal C$-态射 $F\to G$。

再证本质满。给定 $\mathcal D$ 上 sheaf $H$，对 $U\in\mathcal C$ 定义

$$
\widetilde H(U)=
\varprojlim_{\{D_a\to U\}}
\operatorname{Eq}
\left(
\prod_aH(D_a)
\rightrightarrows
\prod_{a,b}H(D_a\times_UD_b)
\right),
$$

其中极限沿 $\mathcal D$-覆盖及其共同细化取。假设 2 和 3 使索引范畴非空并且共同细化足够多；假设 4 使当 $U\in\mathcal D$ 时此定义还原为 $H(U)$。标准粘合检查给出 $\widetilde H$ 是 $\mathcal C$ 上 sheaf，且 $i^\ast\widetilde H\simeq H$。于是 $i^\ast$ 本质满。证毕。

形式化时，最后一步通常拆成三个 lemma：共同细化范畴滤过、构造与覆盖选择无关、扩张后的对象满足 sheaf 条件。参见附录 A。

## 2.5 本章小结

sheaf 计算的核心是等化子；Čech 计算的核心是限制映射的函子性；站点比较的核心是共同细化。凝聚数学中大量“可在 profinite 或 ED 对象上检查”的说法，本质上都依赖本章的比较命题加上第一卷的极不连通对象理论。

## 练习

**练习 2.1.** 对三个对象的覆盖写出 $C^0,C^1,C^2$ 和两个 Čech 微分。

**练习 2.2.** 在命题 2.3.1 中指出紧 Hausdorff 条件分别用在何处。

**练习 2.3.** 证明若两个 $\mathcal D$-覆盖有共同细化，则由它们构造出的 $\widetilde H(U)$ 中的匹配族给出同一个粘合。
