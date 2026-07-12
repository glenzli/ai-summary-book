# 附录 Z：经典空间到 Liquid 对象的证明模块

## Z.0 目标

本附录把旧称“liquid realization”的接口拆成三个可核查层次：

1. 书内构造 \(E\mapsto\underline E\)；
2. 外部定理判断 \(\underline E\) 是否 \(p\)-liquid；
3. 书内用 profinite 局部提升判断短正合列与 cohomology 是否保持。

这样不再假设一个定义域不明的 \(\mathcal T_p\) 或一个额外对象
\(E_{\mathrm{liq}}\)。

## Z.1 拓扑向量空间的凝聚化

设 \(E\) 是 \(\mathcal U\)-小 Hausdorff 实拓扑向量空间，定义

$$
\underline E(S)=\operatorname{Cont}(S,E),
\qquad S\in\mathbf{CHaus}_\kappa.
$$

**命题 Z.1.** \(\underline E\) 是凝聚
\(\underline{\mathbb R}\)-模；连续线性映射 \(u:E\to F\) 诱导凝聚线性态射
\(\underline u:\underline E\to\underline F\)。

**证明.** 对有限联合满射覆盖 \(q:\coprod_iS_i\to S\)，\(q\) 是 quotient map。
相容的连续映射 \(S_i\to E\) 合成一张在 \(q\)-fibres 上常值的连续映射，因而唯一
下降为连续映射 \(S\to E\)。空覆盖给
\(\operatorname{Cont}(\varnothing,E)=*\)。加法与数乘逐点定义；连续线性映射通过后
复合给自然变换。证毕。

**边界 Z.2.** Z.1 只给凝聚模。除非再验证 Hom 延拓条件，不能称其为
\(p\)-liquid；除非 \(E\) 为 \(\kappa\)-紧生成，也不能从任意凝聚态射反推出连续映射。

## Z.2 Liquid membership 输入

固定 \(0<p\le1\)。

**外部输入定理 Z.3（CS26）.** 每个 \(p\)-Banach 空间的凝聚化是
\(p\)-liquid，逆极限保持 \(p\)-liquid。因此每个实 Fréchet 空间的凝聚化对所有
\(0<p\le1\) 都 \(p\)-liquid。

**来源与边界.** CS26 Theorem 2.14、Lemma 2.16 及其后的逆极限推论。输入只判断
membership；exactness 在下一节书内处理。

**定义 Z.4.** 在 Z.3 的适用对象上记

$$
\mathcal L_p(E):=\underline E\in\mathbf{Liquid}_p.
$$

该等号是定义，不是近似或另一次完成化。

## Z.3 Epimorphism 与局部提升

**定义 Z.5.** 连续满射 \(q:E\twoheadrightarrow F\) 称为凝聚有效，如果对每个
\(S\in\mathbf{ProFin}_\kappa\) 和 \(f:S\to F\)，存在有限联合满射覆盖
\(\{S_i\to S\}\)，使 \(f|_{S_i}\) 连续提升到 \(E\)。

**命题 Z.6.** \(\underline q:\underline E\to\underline F\) 是凝聚集合（因而凝聚
阿贝尔群）中的 epimorphism，当且仅当 \(q\) 凝聚有效。

**证明.** Sheaf 态射为 epimorphism 当且仅当每个目标截面局部来自源截面。把站点对象
和截面分别写成 \(S\) 与连续映射 \(S\to F\)，所得条件逐字就是 Z.5。这里从
\(\mathbf{CHaus}_\kappa\) 缩到 \(\mathbf{ProFin}_\kappa\) 使用第一卷第五章的站点
比较及附录 A.3 的同层级 ED 覆盖；没有该输入，只检查 profinite 对象并不足以完成证明。
证毕。

**推论 Z.7（连续截面）.** 若 \(q\) 有连续截面（不要求线性），则 \(q\) 凝聚有效。若拓扑
向量空间短正合列

$$
0\to E'\to E\xrightarrow{q}E''\to0
$$

中 \(E'\cong\ker q\) 同胚且 \(q\) 有连续截面，则凝聚化后正合。只有当截面还线性时，
该短正合列才由这个截面在凝聚阿贝尔群范畴中 split。

**证明.** 截面与任意 \(f:S\to E''\) 复合给全局提升，所以 Z.6 给 epimorphism；
凝聚化保持 kernel，故短列正合。一般连续截面只诱导凝聚集合态射，不是凝聚群同态。
若截面线性，它才诱导凝聚线性右逆，从而给 split exactness。证毕。

**边界 Z.8（closed range 不蕴含 exactness）.** 映射像闭只保证像与 quotient 是
Hausdorff Fréchet 空间。它没有构造 Z.5 所需的参数族局部提升。一般使用中必须直接
验证 Z.5；Hodge/Green operator 给出的连续 splitting 是一个充分条件。

## Z.4 闭值域复形的 cohomology

设 \(E^\bullet\) 是 Fréchet 复形，并记

$$
B^q=\operatorname{im}d^{q-1},\qquad
Z^q=\ker d^q,\qquad
H^q_{\mathrm{top}}=Z^q/B^q.
$$

**命题 Z.9.** 若 \(B^q\) 在 \(Z^q\) 中闭，则
\(H^q_{\mathrm{top}}\) 是 Hausdorff Fréchet 空间。

**证明.** \(Z^q\) 是 Fréchet 空间的闭子空间；Fréchet 空间模闭子空间仍是
Hausdorff、complete 且 metrizable。证毕。

**定义 Z.10.** 在 Z.9 假设下，称 \(E^\bullet\) 在次数 \(q\) 凝聚严格，如果

$$
E^{q-1}\twoheadrightarrow B^q,
\qquad
Z^q\twoheadrightarrow H^q_{\mathrm{top}}
$$

都凝聚有效。

**定理 Z.11（cohomology 比较）.** 若 \(E^\bullet\) 在次数 \(q\) 凝聚严格，则

$$
H^q(\mathcal L_p(E^\bullet))
\cong
\mathcal L_p(H^q_{\mathrm{top}}(E^\bullet)).
$$

**证明.** Z.1 的构造逐对象保持 finite limits，因此
\(\ker\underline d^q=\underline{Z^q}\)。第一张有效满射与 Z.6 说明
\(\operatorname{im}\underline d^{q-1}=\underline{B^q}\)。第二张有效满射说明

$$
\operatorname{coker}(\underline{B^q}\to\underline{Z^q})
=\underline{H^q_{\mathrm{top}}}.
$$

所以心脏中的 kernel modulo image 给出所示同构。Z.3 保证所有对象属于满阿贝尔子范畴
\(\mathbf{Liquid}_p\)，故计算可在该范畴或 \(\mathbf{CondAb}\) 中进行。证毕。

## Z.5 有限维与 Fredholm

**命题 Z.12.** 有限维实向量空间 \(V\) 满足

$$
\mathcal L_p(V)\cong\underline{\mathbb R}^{\oplus\dim V},
$$

因而是 perfect liquid 对象。

**证明.** 选取 \(V\cong\mathbb R^n\)；凝聚化保持有限乘积，有限乘积在阿贝尔范畴
中等于有限直和。Liquid 单位 perfect，且 perfect 性对有限直和封闭。证毕。

**推论 Z.13.** 若 \(E^\bullet\) 每次凝聚严格且
\(H^q_{\mathrm{top}}(E^\bullet)\) 有限维，则
\(H^q(\mathcal L_p(E^\bullet))\) perfect。

**证明.** 组合 Z.11 与 Z.12。证毕。

## Z.6 Dolbeault 接口

**外部输入定理 Z.14（Dolbeault--Hodge）.** 对 compact complex manifold \(X\)
和 holomorphic vector bundle \(E\)，Dolbeault Fréchet 复形

$$
\Gamma(X,\mathcal A^{0,\bullet}(E)),\bar\partial
$$

有连续 Green operators 与 Hodge projections，给连续 exact/coexact/harmonic 分解；
harmonic spaces 有限维，并计算 sheaf cohomology \(H^q(X,E)\)。

**本书不证明的边界.** Parametrix、椭圆估计、正则性与 Dolbeault lemma 均属于输入
D.8；本附录只使用连续 splitting 和 cohomology identification。

**定理 Z.15（Dolbeault liquid cohomology）.** 在 Z.14 下，Dolbeault 复形每次
凝聚严格，且

$$
H^q\!\left(\mathcal L_p
\Gamma(X,\mathcal A^{0,\bullet}(E))\right)
\cong
\mathcal L_p(H^q(X,E))
$$

为 perfect liquid 对象。

**证明.** Z.14 的连续 splittings 通过 Z.7 验证 Z.10 的两张有效满射。应用 Z.11，
再用 Z.14 的 classical cohomology identification 和 Z.12。证毕。

## Z.7 本附录闭包

**结论 Z.16.** 本附录书内证明了凝聚化、epimorphism 的局部提升判别、split
exactness、凝聚严格复形的 cohomology 比较和有限维 perfect 性。外部输入精确剩余：
Banach/Fréchet 的 \(p\)-liquid membership，以及 Dolbeault--Hodge 的连续 splitting
与有限性。

## 练习

1. 证明命题 Z.6。
2. 给出有连续非线性截面但无连续线性截面的满射，并检查 Z.7 仍适用第一项。
3. 在定理 Z.11 中分别指出两张有效满射控制 image 与 cokernel 的位置。
4. 说明 Z.14 的哪两个连续算子验证凝聚严格性。
