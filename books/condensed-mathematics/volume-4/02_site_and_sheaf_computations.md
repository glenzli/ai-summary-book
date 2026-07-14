# 第二章：站点、覆盖与 sheaf 条件的计算

三个局部截面能否粘合，不由它们分别存在决定，而由它们在每个二重交上的限制决定；
若要继续计算上同调，三重交又负责保证 Čech 微分平方为零。对凝聚站点，还必须证明
可表连续映射沿有限联合满射粘合后仍连续，并说明为何只在 profinite 或极不连通对象上
计算不会丢失 sheaf。等化子、商映射和共同细化正是这三步的可检验机制。

以下固定小范畴 $\mathcal C$，假设相关有限纤维积存在。大测试站点先在第一卷附录 A
所定 universe 中取小骨架。每个计算都会给出覆盖与预层作为输入，写出匹配条件和限制
映射，输出全局截面或比较等价；交对象不存在、覆盖不稳定或共同细化失败时，相应公式
便没有合法类型或不能重建原 sheaf。

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

## 2.2 Worked example 与三重交

先实际执行一次有限凝聚覆盖。令

$$
S=\{0,1,2\},\qquad
S_0=\{0,1\},\qquad
S_1=\{1,2\}
$$

都带离散拓扑，并取包含映射覆盖 $S$。对紧 Hausdorff 空间 $T$ 的可表预层
$h_T(A)=\operatorname{Cont}(A,T)$，输入

$$
f_0=(t_0,t_1)\in h_T(S_0),\qquad
f_1=(t'_1,t_2)\in h_T(S_1).
$$

交 $S_0\times_SS_1=\{1\}$，匹配条件正是 $t_1=t'_1$。满足时，逐点粘合输出
$f=(t_0,t_1,t_2)\in h_T(S)$，并得到

$$
h_T(S)\cong h_T(S_0)\times_{h_T(\{1\})}h_T(S_1)
\cong T^2\times_TT^2.
$$

若 $t_1\ne t'_1$，输入不在等化子中，因而没有全局输出。有限离散性使所有集合映射
连续；对一般紧 Hausdorff 覆盖，2.3 节的商映射步骤承担同一连续性检查。

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

是等化子；把同一覆盖代入 $G$ 的 sheaf 条件，则得到以 $G(D_a)$ 与
$G(D_a\times_UD_b)$ 为两层的对应等化子。稳定基假设保证
$D_a\times_UD_b$ 仍在 $\mathcal D$，所以 $\alpha$ 已经在这些交对象上定义；
两个等化子之间因而有唯一诱导映射
$\alpha_U:F(U)\to G(U)$。共同细化假设保证不同覆盖给出的诱导映射相容，
所以得到唯一的 $\mathcal C$-态射 $F\to G$。

再证本质满。给定 $\mathcal D$ 上 sheaf $H$，对 $U\in\mathcal C$ 定义右 Kan
延拓

$$
\widetilde H(U)=
\varprojlim_{(D\to U)\in(\mathcal D/U)^{op}}H(D).
$$

若 $v:V\to U$，复合给函子 $\mathcal D/V\to\mathcal D/U$；限制极限 cone 得到
$\widetilde H(U)\to\widetilde H(V)$。恒等与复合由切片函子的恒等与复合保证，所以
$\widetilde H$ 先成为预层。

现取 $\mathcal C$-覆盖 $\{U_a\to U\}$，再由假设 2 对每个 $U_a$ 取
$\mathcal D$-覆盖 $\{D_{a\alpha}\to U_a\}$。复合族
$\{D_{a\alpha}\to U\}$ 是 $U$ 的 $\mathcal D$-覆盖。稳定基假设和共同细化说明，
上述切片极限可由这个覆盖的等化子计算：

$$
\widetilde H(U)
\cong
\operatorname{Eq}\left(
\prod_{a,\alpha}H(D_{a\alpha})
\rightrightarrows
\prod_{a,b,\alpha,\beta}
H(D_{a\alpha}\times_UD_{b\beta})
\right).
$$

同样地，$\widetilde H(U_a)$ 由固定 $a$ 的那些项计算，而
$\widetilde H(U_a\times_UU_b)$ 由二重交的 $\mathcal D$-细化计算。把三组等化子展开，
在 $\{U_a\to U\}$ 上粘合 $\widetilde H$，恰好等价于在复合
$\mathcal D$-覆盖上粘合 $H$；后者由 $H$ 的 sheaf 条件成立。因此
$\widetilde H$ 是 $\mathcal C$-sheaf。

若 $U=D\in\mathcal D$，对象 $\operatorname{id}_D$ 在
$(\mathcal D/D)^{op}$ 中为初对象，故极限由 $H(D)$ 给出，得到
$i^*\widetilde H\cong H$。反过来，对 $\mathcal C$-sheaf $F$，任取
$\mathcal D$-覆盖 $\{D_a\to U\}$；$F$ 的 sheaf 条件把 $F(U)$ 识别为该覆盖的
matching equalizer，而上面的切片极限给出同一个 equalizer，所以
$\widetilde{i^*F}\cong F$。两个同构与 restriction 相容，给出拟逆的 unit 与
counit。因此 $i^*$ 全忠实且本质满，故为等价。证毕。

形式化时，最后一步通常拆成三个 lemma：共同细化范畴滤过、构造与覆盖选择无关、扩张
后的对象满足 sheaf 条件。第一卷[附录 B](../volume-1/B_site_comparison_theorem.md)
保留同一切片极限构造的逐段版本，本卷附录 A 记录形式化接口。

## 2.5 从局部匹配到较小测试站点

有限覆盖的 matching object 是一个等化子，Čech 微分平方为零来自限制映射的函子性；
紧 Hausdorff 可表预层的粘合还使用联合满射为 quotient map。稳定基比较再用覆盖、交
对象和共同细化从 $\mathcal D$ 上数据重建 $\mathcal C$-sheaf。于是“在 profinite 或
ED 对象上检查”有了明确前提：这些对象必须形成足够覆盖且对交和细化稳定的基。第三章
将把同样的有限复形计算用于投射分解，并把输出从匹配族换成 Ext 与 Tor。

## 练习

**练习 2.1.** 对三个对象的覆盖写出 $C^0,C^1,C^2$ 和两个 Čech 微分。

**练习 2.2.** 在命题 2.3.1 中指出紧 Hausdorff 条件分别用在何处。

**练习 2.3.** 证明若两个 $\mathcal D$-覆盖有共同细化，则由它们构造出的 $\widetilde H(U)$ 中的匹配族给出同一个粘合。
