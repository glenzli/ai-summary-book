# 附录 AA：Scholze 与 Clausen-Scholze 核心定理图谱

## AA.0 目标

本附录把本书使用的 Scholze 与 Clausen-Scholze 定理整理成核心定理图谱。它的目的不是增加新的黑箱，而是明确：

1. 哪些定理属于凝聚数学主线；
2. 哪些部分本书已经证明；
3. 哪些部分仍必须作为外部输入；
4. 后续如果继续自足化，应从哪里展开证明。

本附录的原则是：solid、analytic、liquid 和复几何建模不是可选应用背景，而是 condensed mathematics 进入现代几何与分析的核心结构。

## AA.1 核心定理的判定标准

一个定理在本书中称为核心定理，若满足以下任一条件：

1. 它定义或构造了后续章节的主要范畴；
2. 它保证一个张量、Hom、推前或 realization 在正确范畴中存在；
3. 它使经典几何对象可进入 condensed/analytic/liquid 语言；
4. 它在后续主定理中被反复使用，而非只用于单个例子。

按此标准，以下定理属于核心：

- condensed 基础和 ED/profinite 测试理论；
- solidification、solid tensor 和 profinite 测度对象；
- analytic rings、analyticization、Huber pair rational localization 和 rational descent；
- \(p\)-liquid analytic ring 与 liquid realization；
- \(f_!\)、投影公式和 \(f^!\) 形式；
- Clausen-Scholze 的 condensed/analytic complex geometry 建模；
- coherent finiteness、Serre duality、GAGA 和 HRR/GRR 的 condensed/analytic 表述。

## AA.2 Condensed 基础核心

**核心定理 AA.1（测试站点与 sheaf 比较）。** compact Hausdorff 站点、profinite 测试子站点和极不连通测试对象给出等价的凝聚对象检测方式；在合适小性约定下，凝聚集合与凝聚阿贝尔群由这些测试对象上的 sheaf 条件控制。

**书内部分。** 第一卷证明 compact Hausdorff 的基本闭性、sheaf 等化子条件、profinite 站点比较的形式部分、Stone 对偶链和 ED 覆盖检测。

**外部部分。** Gleason lifting theorem 和某些 universe/可展示性细节仍作为输入。

**依赖位置。** 第一卷第五至八章，第一卷附录 B、D、J、K、N、O。

**核心定理 AA.2（凝聚阿贝尔群的同调代数）。** \(\mathbf{CondAb}\) 是足够好的 Grothendieck 型阿贝尔范畴，具有投射生成元、Ext、Tor、派生张量和 sheafification 的正合控制。

**书内部分。** 第一卷证明自由凝聚阿贝尔群的泛性质、ED 测试对象给投射性、Ext/Tor 的基本同调代数和工作例题。

**外部部分。** 完整可展示性、K-flat/K-injective 存在性等一般派生范畴技术作为标准输入。

**依赖位置。** 第一卷第七至十一章，附录 C、E、G、H、I、M。

## AA.3 Solid 核心

**核心定理 AA.3（Nöbeling 与 profinite 整值函数）。** 对 profinite \(S\)，连续整值函数群

$$
C(S,\mathbb Z)
$$

是自由阿贝尔群。

**书内部分。** 第一卷附录 F、P 证明有限、可数和过滤代数部分，并说明自由性如何进入 solid 测度对象计算。

**外部部分。** 任意 profinite 空间的 Nöbeling-Asgeirsson 超限过滤构造仍作为输入。

**用途。** 控制 \(\mathbb Z^\square[S]\) 的乘积型模型，并服务于 profinite 测度张量公式。

**核心定理 AA.4（solidification 反射局部化）。** Dirac-to-measure cone

$$
K_S=\operatorname{cofib}(\mathbb Z[\underline S]\to\mathbb Z^\square[S])
$$

定义的 \(K_S\)-局部对象构成 \(D(\mathbf{CondAb})\) 的反射稳定子范畴，记为

$$
D_\square(\mathbb Z).
$$

**书内部分。** 第二卷附录 V 证明集合生成局部化、局部对象判别、kernel localizing 和 \(K_S\)-正交形式。

**外部部分。** 该局部范畴与 Scholze solid theory 的完全识别，以及所需的测度对象计算性质。

**核心定理 AA.5（solid kernel 张量理想性）。** solidification 的核对普通派生张量封闭。因此

$$
M\otimes^{L,\square}N
=
L^\square(M\otimes^LN)
$$

给出 \(D_\square(\mathbb Z)\) 的闭对称幺半结构。

**书内部分。** 第二卷附录 K、L、W 证明“kernel 为张量理想 \(\Rightarrow\) 幺半下降”和张量理想的生成元归约。

**外部部分。** profinite 测度张量计算和 \(\ker L^\square=\mathcal N_\square\) 的 Scholze 证明。

**核心定理 AA.6（profinite 测度张量公式）。** 对 profinite \(S,T\)，有

$$
\mathbb Z^\square[S]\otimes^{L,\square}
\mathbb Z^\square[T]
\simeq
\mathbb Z^\square[S\times T].
$$

**书内部分。** 第二卷附录 W 证明该公式如何推出 solid kernel 张量理想性所需的生成元计算。

**外部部分。** 公式本身作为 Scholze solid theory 输入。

## AA.4 Analytic 核心

**核心定理 AA.7（analytic ring localization）。** 若 \((A,\mathcal M)\) 是 analytic ring，则由

$$
K_S^{\mathcal M}=\operatorname{cofib}(A[\underline S]\to\mathcal M[S])
$$

定义的局部对象构成反射稳定子范畴

$$
D(A,\mathcal M)\subset D(A),
$$

且 localization 与张量积相容。

**书内部分。** 第二卷附录 I、X 证明 analytic cone 判别、局部对象形式、analyticization 泛性质和张量下降的条件性证明。

**外部部分。** analytic ring 公理推出正确 localization、kernel 张量理想性和测度对象 functoriality。

**核心定理 AA.8（Huber pair 与 rational localization）。** 离散 Huber pair \((A,A^+)\) 给出 analytic ring；rational localization 与 analytic module category 相容，并满足 finite rational cover 的 Čech descent。

**书内部分。** 第二卷第六章、附录 N、Y 证明 rational descent 的范畴论形式层：Čech nerve、mapping-space descent、对象 gluing 和 compact generation descent。

**外部部分。** Huber pair 测度对象构造、rational acyclicity 和 rational intersections 相容性。

**核心定理 AA.9（\(f_!\)、投影公式与右伴随）。** 在 Scholze 的有限型仿射情形中，存在

$$
f_!:D(A^\square)\to D(\mathbb Z^\square)
$$

满足投影公式，并在 compact generation 条件下有右伴随 \(f^!\)。

**书内部分。** 第二卷第七章、附录 F、L 证明接受 \(f_!\) 和投影公式后，\(f^!\) 的内部 Hom 公式与形式伴随如何推出。

**外部部分。** \(f_!\) 的构造、边界项、compact generation 和投影公式本身。

## AA.5 Liquid 核心

**核心定理 AA.10（\(p\)-liquid analytic ring）。** 对合适的 \(p\)，\((\mathbb R,\mathcal M_{<p})\) 是 analytic ring，其 analytic module category 给出 \(p\)-liquid 实向量空间理论。

**书内部分。** 第二卷第五章、附录 S 给出 liquid 入口和接受输入后的有限维、Fredholm、Dolbeault 类型后果。

**外部部分。** \(p\)-liquid 测度对象构造和 analytic ring 条件验证。

**核心定理 AA.11（liquid realization）。** 对核 Fréchet 空间、适用的 Banach 空间、有限维空间和连续线性映射，存在进入 liquid 范畴的 realization

$$
\mathcal L_p:E\mapsto E_{\mathrm{liq}},
$$

并与有限极限、有限直和、闭值域短正合列和经典连续线性映射相容。

**书内部分。** 第二卷附录 J、P、Z 证明拓扑向量空间凝聚化、闭值域 Fréchet cohomology 的 Hausdorff 性、接受 exactness 后的 cohomology 比较。

**外部部分。** realization 的构造、Hom 判别和 exactness 范围。

## AA.6 Clausen-Scholze 复几何核心

**核心定理 AA.12（复解析对象的 condensed/analytic 建模）。** compact complex manifolds、相干解析层、Dolbeault 复形和相关 Fréchet/liquid 函数空间可在 condensed/analytic/liquid 语言中表示，并保持 classical cohomology。

**书内部分。** 第三卷第一至三章和附录 B、N、R 给出语言翻译、Dolbeault resolution 的形式层和局部解析骨架。

**外部部分。** Clausen-Scholze 的建模定理、liquid realization 与 Dolbeault-Fredholm 输入的相容性。

**核心定理 AA.13（coherent finiteness 与 compactness）。** 紧复空间或 proper 情形中，相干上同调有限维，并在 analytic/liquid 派生范畴中给出 perfect/compact 对象。

**书内部分。** 第三卷第四章、附录 L、M、X、AC、AQ 证明接受 Grauert 或 Hodge-Fredholm 输入后的有限性传播。

**外部部分。** Grauert direct image theorem、Fredholm-Hodge 定理和 Clausen-Scholze compactness statement。

**核心定理 AA.14（Serre/Grothendieck duality）。** 相干层的上同调与 dualizing complex 的 Ext 之间存在 perfect pairing；在 smooth 向量丛情形化为经典 Serre duality。

**书内部分。** 第三卷第五章、附录 J、O、AA、AD、AQ 证明链级配对、Hodge 向量丛情形和 dualizing complex 形式后果。

**外部部分。** 一般 perfectness、dualizing complex 存在性和 trace theorem。

**核心定理 AA.15（GAGA 与 HRR/GRR）。** Proper/projective 代数几何与解析几何的相干层、上同调、导出范畴和 Riemann-Roch 公式相容。

**书内部分。** 第三卷第六、七章和附录 K、Q、Y、AI、AO、AE、AK、AP、AQ 证明 exact equivalence 的导出后果、projective GAGA 骨架、形式函数路线、特征类形式代数和 GRR 的组织模块。

**外部部分。** Serre analytic finite generation、Grothendieck existence、localized Chern character、deformation specialization 和 GRR 基本因子定理。

## AA.7 总依赖表

| 核心块 | 本书位置 | 书内闭合部分 | 外部输入 |
| --- | --- | --- | --- |
| Condensed sheaf 基础 | 卷一 1-11，附录 B/C/H/K/N | sheaf、ED 检测、Ext/Tor 形式层 | Gleason、可展示性细节 |
| Nöbeling | 卷一 F/P | 有限/可数/过滤代数模块 | 任意 profinite 超限过滤 |
| Solidification | 卷二 M/Q/V | 局部对象、反射局部化形式、kernel 性质 | Scholze solid 识别 |
| Solid tensor | 卷二 K/L/Q/W | 幺半下降、生成元归约 | profinite 测度张量公式 |
| Analytic rings | 卷二 I/N/R/X/Y | cone 判别、analyticization 泛性质、descent 形式层 | analytic ring localization、rational acyclicity |
| Liquid | 卷二 J/P/S/Z | 凝聚化、闭值域、Fredholm 后果 | \(p\)-liquid 测度与 realization |
| \(f_!\)/duality 入口 | 卷二 F/L，卷三 J/AD/AJ | 伴随、投影公式后果 | \(f_!\) 构造与 trace |
| Complex geometry | 卷三 F-AQ | 形式后果、计算模型、proof modules | Clausen-Scholze 建模和经典深定理 |

## AA.8 核心闭包定理

**定理 AA.16（核心定理输入闭包）。** 接受 AA.1-AA.15 中标出的外部输入后，本书四卷中的凝聚数学主线闭合：所有后续 solid、analytic、liquid 和复几何应用均可追溯到本附录列出的核心定理块，且接受输入后的范畴论、同调代数、张量、descent、duality 和 cohomology 有限性推论均在书内证明或在对应附录中给出证明模块。

**证明.** 卷一建立 condensed 基础并登记 Gleason、Nöbeling 等输入。卷二附录 Q-Z 把 solid、analytic、liquid 主线收束为主定理包并拆成证明模块。卷三附录 F-AQ 把复几何主输入、证明模块和 condensed/analytic 闭包组织为 theorem package。全书定理索引和依赖图给出跨卷引用链。因此任何主线结论若依赖深层结果，都落在 AA.1-AA.15 的外部输入中；若不依赖深层结果，则对应形式证明已在相应附录给出。证毕。

## 练习

1. 说明为什么 solid kernel 张量理想性是 solid tensor product 的必要条件。
2. 在 AA.8 的 rational descent 中区分 ordinary sheaf descent 与 category-valued descent。
3. 解释 liquid realization 为什么不能只由拓扑向量空间的凝聚化推出。
4. 从 AA.12-AA.15 中任选一个复几何定理，列出其 classical 输入、condensed/analytic 输入和书内形式后果。
