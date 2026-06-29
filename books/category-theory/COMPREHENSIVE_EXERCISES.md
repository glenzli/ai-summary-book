# 综合习题

本文件收录跨章节综合题。每题要求同时使用多个章节的概念；答案见 [COMPREHENSIVE_SOLUTIONS.md](COMPREHENSIVE_SOLUTIONS.md)。

## 第一组：普通范畴论核心

**综合题 1.** 设 $\mathcal C$ 有终对象、二元积和等化子。

1. 构造任意有限图形 $D:\mathcal J\to\mathcal C$ 的极限。
2. 证明构造对图形自然。
3. 用该构造写出 pullback 和 equalizer 的关系。

**综合题 2.** 设 $F:\mathcal C\to\mathcal D$，$G:\mathcal D\to\mathcal C$。

1. 从 Hom 自然同构定义伴随。
2. 构造单位和余单位。
3. 证明左伴随保持余等化子。
4. 用自由阿贝尔群伴随解释张量积的右正合性。

**综合题 3.** 设 $\mathcal C$ 小。

1. 证明 Yoneda 嵌入 $y:\mathcal C\to\widehat{\mathcal C}$ 完全忠实。
2. 证明每个预层是可表预层的典范余极限。
3. 将第 2 点写成 co-Yoneda 公式。

## 第二组：结构性范畴论

**综合题 4.** 设 $\mathcal V$ 为完备闭对称幺半范畴，$\mathcal A$ 为小 $\mathcal V$-富范畴。

1. 写出富自然变换对象的 end 公式。
2. 证明 enriched Yoneda。
3. 在 $\mathcal V=\mathbf{Ab}$ 时解释该定理如何恢复加性 Yoneda。

**综合题 5.** 设 $\mathcal C$ 为小幺半范畴。

1. 写出 Day 卷积公式。
2. 用 co-Yoneda 证明 $y(a)\star y(b)\cong y(a\otimes b)$。
3. 说明为什么相干性证明需要外部输入。

**综合题 6.** 设 $(\mathcal C,J)$ 为小站点。

1. 用筛写出 sheaf 条件。
2. 在覆盖族有拉回的情形下推导等化子形式。
3. 解释 subcanonical 拓扑与 Yoneda 嵌入的关系。
4. 说明 Grothendieck topos 与 $\infty$-topos 的 sheaf 条件差别。

## 第三组：高阶范畴论

**综合题 7.** 设 $\mathcal C$ 为普通范畴。

1. 证明 $N(\mathcal C)$ 是 quasi-category。
2. 计算 $hN(\mathcal C)$。
3. 解释普通范畴中的极限如何由 $N(\mathcal C)$ 中的 slice quasi-category 恢复。

**综合题 8.** 比较 Kan complex、quasi-category 和 ordinary category nerve。

1. 说明三者的 horn 条件。
2. 证明 Kan complex 中每条边都是等价。
3. 说明为什么 $N(\mathcal C)$ 通常不是 Kan complex。

**综合题 9.** 设 $p:X\to S$ 是 Cartesian fibration。

1. 写出 Cartesian edge 的映射空间判别。
2. 说明它如何对应普通 Grothendieck fibration 的 Cartesian lift。
3. 陈述 straightening/unstraightening 定理。
4. 解释它与第六章 Kan 延拓点态公式的关系。

**综合题 10.** 设 $C$ 为稳定 $\infty$-范畴。

1. 写出稳定性的定义。
2. 解释纤维序列与余纤维序列为什么一致。
3. 说明 $hC$ 为什么有三角范畴结构。
4. 比较稳定 $\infty$-范畴与三角范畴的信息量。

## 第四组：外部输入边界

**综合题 11.** 从本书中选择三个外部输入定理。

1. 写出每个定理的精确用途。
2. 写出该定理依赖的章节和后续影响。
3. 判断若该定理不可用，正文中哪些结论需要降级。

**综合题 12.** 设计一个从普通范畴论进入 higher algebra 的学习路线。

1. 列出必须掌握的普通范畴论工具。
2. 列出从模型范畴过渡到 quasi-category 的技术点。
3. 说明为什么 $E_n$-代数需要 $\infty$-operad 语言。

## 第五组：新增结构与高阶综合

**综合题 13.** 设 $\mathcal C$ 为小范畴。

1. 证明 $\widehat{\mathcal C}$ 局部可表现。
2. 说明可表预层为何构成强生成子。
3. 若 $L:\widehat{\mathcal C}\to\mathcal E$ 是保持小余极限的可达函子，用局部可表现范畴伴随函子定理判断它是否有右伴随。

**综合题 14.** 设 $R$ 为环。

1. 证明 $R\text{-}\mathbf{Mod}$ 是 Grothendieck 范畴。
2. 对态射 $f:M\to N$ 写出 image 与 coimage，并证明二者同构。
3. 解释 Gabriel-Popescu 定理如何把一般 Grothendieck 范畴与模范畴联系起来。

**综合题 15.** 比较 $\infty$-伴随的三种口径。

1. 写出 mapping space 自然等价口径。
2. 写出 correspondence 左右可表示口径。
3. 写出 walking adjunction/scaled nerve 口径中的低维数据。
4. 说明普通伴随如何嵌入这三种口径。

**综合题 16.** 设 $p:X\to S$ 是 Cartesian fibration，对应 $F:S^{op}\to\mathcal{Cat}_\infty$。

1. 构造边 $\alpha:s\to t$ 的 restriction $\alpha^*:X_t\to X_s$。
2. 证明复合边给出 $(\beta\alpha)^*\simeq\alpha^*\beta^*$。
3. 陈述 Cartesian sections as limits。
4. 解释该定理如何组织 sheaf 或 descent data。

**综合题 17.** 连接稳定 $\infty$-范畴、谱序列与 Morita 理论。

1. 在带 t-结构的稳定 $\infty$-范畴中，说明 heart 的核和余核如何由纤维/余纤维给出。
2. 对有限滤过对象写出 $E_1$ 页和收敛目标。
3. 证明矩阵代数 $M_n(k)$ 与 $k$ Morita 等价的双模数据。
4. 说明 cobordism hypothesis 为什么使用 fully dualizable objects 而不是任意对象。

**综合题 18.** 连接 presentable $\infty$-categories、topos 和高阶代数。

1. 证明可表预层在 $\mathcal P(C)$ 中紧。
2. 说明 accessible localization 如何同时覆盖 sheaf 化和 Bousfield localization。
3. 用 presentable 伴随函子定理判断保持小余极限的函子是否为左伴随。
4. 解释 $\operatorname{Pr}^L$ 的张量积为什么是第二十二章高阶代数的背景。

**综合题 19.** 连接 coend、profunctor 和 Morita 理论。

1. 写出 profunctor 的定义和 coend 复合公式。
2. 用 co-Yoneda 证明恒等 profunctor 的单位律。
3. 说明函子 $F$ 如何给出伴随 profunctors $F_*\dashv F^*$。
4. 比较 profunctor 复合与双模相对张量积。

**综合题 20.** 连接富 profunctor、equipment 与 base change。

1. 写出富 profunctor 和富 coend 复合公式。
2. 说明 companion/conjoint 如何把垂直函子变成水平 profunctor。
3. 在集合 slice 范畴中证明拉回方块满足 Beck-Chevalley。
4. 解释 indexed category、Cartesian fibration 和 equipment 三者的关系。

**综合题 21.** 连接紧生成、Brown 表示性和 Bousfield 局部化。

1. 写出 compactly generated stable presentable $\infty$-category 的定义。
2. 说明 Brown 表示性如何推出伴随存在性。
3. 对 Bousfield localization $L$ 构造 $A_X\to X\to LX$。
4. 解释 smashing localization 与普通 Bousfield localization 的差别。

**综合题 22.** 连接 dg 范畴、紧对象和导出 Morita 理论。

1. 从 $\operatorname{Ch}(k)$-富范畴定义 small dg category，并构造 $H^0(\mathcal A)$。
2. 证明可表 dg 模 $h_a$ 在 $D(\mathcal A)$ 中 compact。
3. 比较 quasi-equivalence 与 Morita equivalence。
4. 对普通代数 $A$ 写出 Hochschild chains 的导出 trace 公式，并说明 Morita 不变性的含义。

**综合题 23.** 连接六操作、Beck-Chevalley 和 Verdier 对偶。

1. 写出稳定系数系统和六操作形式主义的基本数据。
2. 对 Cartesian 方块构造普通基变换态射 $g^*f_*\to f'_*g'^*$。
3. 证明投影公式对复合态射封闭。
4. 在开闭分解的 recollement 序列下，证明 $j^*$ 与 $i^*$ 联合保守。
5. 对 dualizable $K$ 证明 $\mathbb D_X(K)\simeq K^\vee\otimes\omega_X$。

**综合题 24.** 连接相对范畴、模型范畴和高阶范畴模型。

1. 定义 relative category 和 $\infty$-categorical localization。
2. 证明 localization 的泛性质给出唯一性。
3. 定义 Dwyer-Kan equivalence，并证明它诱导同伦范畴等价。
4. 对模型范畴 $\mathcal M$，解释 $\mathcal M_\infty$ 的映射空间如何由 cofibrant-fibrant 对象计算。
5. 比较 quasi-category、simplicial category 和 complete Segal space 模型的角色。

**综合题 25.** 连接 dg quotient、localizing invariants 和 noncommutative motives。

1. 定义 small stable idempotent-complete $\infty$-categories 的 exact sequence。
2. 说明 Drinfeld quotient 如何使子 dg 范畴对象在 $H^0$ 中变为零。
3. 证明 localizing invariant 是 additive invariant。
4. 陈述 $\operatorname{Mot}_{\operatorname{loc}}$ 的普遍性质，并推出 motives 等价蕴含所有 localizing invariants 等价。
5. 用 Morita 等价解释 $K(M_n(R))\simeq K(R)$ 和 $HH(M_n(R))\simeq HH(R)$。

**综合题 26.** 连接 perverse sheaves、recollement、Verdier 对偶和六操作。

1. 写出 middle perverse t-structure 的支撑与余支撑条件。
2. 用 recollement 证明有限层化下 stratum restrictions 联合保守。
3. 说明 t-结构如何由开闭分解粘合。
4. 定义中间延拓 $j_{!*}$，并证明 simple perverse sheaf 的中间延拓仍 simple。
5. 证明 Verdier 对偶与中间延拓相容。

**综合题 27.** 连接 Bousfield localization、compact generation 和 chromatic homotopy。

1. 定义 Bousfield class 和其偏序。
2. 证明楔和给出 Bousfield classes 的 join。
3. 陈述 Morava $K(n)$ 的系数环和有限谱 chromatic type。
4. 陈述 thick subcategory theorem 和 telescope conjecture。
5. 用 chromatic fracture square 证明 $L_nX$ 的零对象检测。

**综合题 28.** 连接 $D$-modules、perverse sheaves 和六操作。

1. 解释左 $D_X$-module 与 flat connection 的关系。
2. 定义 characteristic variety 与 holonomic 条件。
3. 计算平凡 connection 的 de Rham complex。
4. 陈述 Riemann-Hilbert correspondence，并说明 heart 层对应。
5. 解释 Kashiwara equivalence 如何表达 recollement 的闭支撑部分。

**综合题 29.** 连接 derived stacks、QCoh、cotangent complex 和 formal moduli。

1. 定义 derived affine scheme 和 derived stack。
2. 证明 derived affine 的映射空间公式。
3. 定义 $\operatorname{QCoh}(X)$，并证明 $X=\operatorname{Spec}A$ 时恢复 $\operatorname{Mod}_A$。
4. 用导子的表示性证明 cotangent complex 的 transitivity triangle。
5. 陈述 Lurie-Pridham formal moduli theorem，并解释切复形与 cotangent complex 的关系。

**综合题 30.** 连接 Barr-Beck-Lurie、Cech descent 和 faithfully flat descent。

1. 说明伴随 $F\dashv G$ 如何产生 monad 和 comparison functor。
2. 陈述 Barr-Beck-Lurie 单子性定理。
3. 证明 monadic 遗忘函子保守。
4. 解释 comonadicity 如何给出 Cech descent。
5. 对 faithfully flat $A\to B$ 写出模下降的 cocycle data。

**综合题 31.** 连接 Tannaka duality、QCoh 和 derived stacks。

1. 定义 neutral Tannakian category 和 fiber functor。
2. 写出 matrix coefficient coalgebra 的 coend 公式，并解释其意义。
3. 证明仿射情形的高阶 Tannaka 公式
   $$
   \operatorname{Map}(R,A)\simeq\operatorname{Fun}^{L,\otimes}(\operatorname{Mod}_R,\operatorname{Mod}_A).
   $$
4. 说明 Tannaka 重构与 QCoh descent 的相容性。
5. 解释为什么带 fiber functor 的 $\operatorname{QCoh}(BG)$ 可恢复 $G$。

**综合题 32.** 连接 tensor triangular geometry、perfect complexes 和 chromatic homotopy。

1. 定义 tt-category、thick tensor ideal 和 prime tensor ideal。
2. 定义 Balmer spectrum 和对象支撑，并证明张量支撑公式。
3. 陈述 Balmer 分类定理。
4. 说明 $\operatorname{Spc}(\operatorname{Perf}(R))$ 与 $\operatorname{Spec}R$ 的关系。
5. 解释有限谱的 chromatic type 如何给出 tt-geometry 的例子。

**综合题 33.** 连接 $K$-theory、$THH$、$TC$ 和 cyclotomic trace。

1. 解释 $THH$ 作为谱值 trace 的定义。
2. 说明 $THH$ 的圆作用和 cyclotomic structure。
3. 写出 $TC$ 的 Nikolaus-Scholze 型 fiber 公式。
4. 陈述 cyclotomic trace 和 Dundas-Goodwillie-McCarthy 定理。
5. 说明 trace methods 为什么是 noncommutative motives 语境中的自然变换理论。

**综合题 34.** 连接 Goodwillie calculus、稳定化和 operad。

1. 定义 $n$-excisive functor 和 Goodwillie approximation $P_nF$。
2. 定义 $D_nF$ 和 $n$-homogeneous functor。
3. 用 $1$-excisive reduced 条件证明 $\operatorname{cr}_2F=0$。
4. 写出 homogeneous layer 的 derivative 公式。
5. 陈述 chain rule，并解释它为什么产生 operad 结构。

**综合题 35.** 连接 motivic homotopy、$\mathbb A^1$-局部化和六操作。

1. 从 $\operatorname{Sm}_S$ 定义 motivic spaces $\mathbf H(S)$。
2. 证明 $\mathbf H(S)$ presentable，并刻画局部对象。
3. 定义 Tate sphere 和 $\mathbf{SH}(S)$。
4. 写出 motivic localization triangle 和 homotopy purity。
5. 说明 compact generation 如何用紧生成子检测对象与态射，并指出 realization functor 保守性为何是额外假设。

**综合题 36.** 连接范畴逻辑、依赖类型和 univalence。

1. 定义子对象纤维化，并解释谓词替换。
2. 在 regular category 中构造 $\exists_f\dashv f^*$。
3. 证明 $\Sigma_f\dashv f^*$，并解释 $\Pi_f$ 的类型论含义。
4. 用 comprehension category 解释上下文、类型和项。
5. 陈述 univalence，并说明它如何把等价对象视为相等对象。

**综合题 37.** 连接 $E_n$-代数、因子化同调和 Hochschild homology。

1. 定义 $\operatorname{Disk}_n$ 和 $E_n$-代数。
2. 写出因子化同调的左 Kan 延拓公式。
3. 证明 $\int_{\mathbb R^n}A\simeq A$ 与不交并公式。
4. 陈述因子化同调 excision，并说明其计算意义。
5. 陈述 $\int_{S^1}A\simeq HH(A)$ 和非阿贝尔 Poincare 对偶。

**综合题 38.** 连接 condensed sets、solid modules 和 derived categories。

1. 定义 $\operatorname{ProFin}$ 站点和 condensed set。
2. 证明离散集合全忠实嵌入 condensed sets。
3. 说明 condensed abelian groups 是 Grothendieck abelian category 的后果。
4. 定义 solidification、solid tensor product 和 solid module。
5. 解释 derived solid category 为什么是稳定 presentable 范畴的自然来源。

**综合题 39.** 连接语法范畴、分类 topos 和 tripos。

1. 定义有限极限理论的语法范畴，并陈述其泛性质。
2. 定义几何理论的分类 topos。
3. 证明分类 topos 在等价意义下唯一。
4. 定义泛模型，并说明任意模型如何由它拉回。
5. 定义 tripos 和 generic predicate，并解释 tripos-to-topos 的意义。

**综合题 40.** 连接关系演算、regular 逻辑和正合完成。

1. 在 regular category 中定义关系及其复合。
2. 证明集合中该复合恢复通常关系复合。
3. 证明态射图像关系满足 $\Gamma_g\circ\Gamma_f=\Gamma_{gf}$。
4. 定义 exact completion，并说明已 exact 范畴的完成。
5. 解释 allegory 如何抽象 regular 逻辑的关系演算。

**综合题 41.** 连接 cohesive topos、模态和 cohomology。

1. 写出 cohesive $\infty$-topos 的伴随串。
2. 定义 $\int,\flat,\sharp$ 三个模态。
3. 证明 $\flat$ 在 $\operatorname{Disc}$ 全忠实时幂等。
4. 说明 left exact modality 为什么与恒等类型相容。
5. 用 $\Pi\dashv\operatorname{Disc}$ 推出 shape 上的 cohomology 公式。

**综合题 42.** 连接 exit-path 范畴、可构造 sheaves 和层化因子化同调。

1. 定义 conically stratified space 和 exit path。
2. 定义 $\operatorname{Exit}(X)$，并证明单层情形恢复 fundamental $\infty$-groupoid。
3. 陈述 exit-path 分类 constructible sheaves 的定理。
4. 描述开闭分解下的 exit 粘合数据。
5. 说明层化因子化同调如何推广普通因子化同调。

**综合题 43.** 连接高阶 Morita、trace 和 $E_n$-Koszul duality。

1. 定义 $\operatorname{Alg}_n(C)$ 的对象和 1-态射。
2. 说明 $n=1$ 时如何恢复普通 Morita bicategory。
3. 证明 $M_n(k)$ 与 $k$ Morita 等价。
4. 陈述 Morita trace、$HH(A)$ 和 $\int_{S^1}A$ 的关系。
5. 定义 $E_n$-Koszul dual，并计算 $\mathbb 1^!$。

**综合题 44.** 连接 derivator、同伦 Kan 延拓和稳定性。

1. 定义预 derivator 和 derivator。
2. 解释限制函子 $u^*$ 与同伦 Kan 延拓 $u_!,u_*$。
3. 说明唯一函子 $I\to *$ 如何给出同伦极限和余极限。
4. 定义 stable derivator，并说明 pushout/pullback 的关系。
5. 从 $\infty$-category $C$ 构造 $\mathbb D_C$，并证明 $\mathbb D_C(*)\simeq hC$。

**综合题 45.** 连接 stacks、torsors、gerbes 和非阿贝尔上同调。

1. 定义 groupoid-valued prestack 和 stack。
2. 写出 descent datum，并说明 stack 条件的有效粘合含义。
3. 定义 $G$-torsor 与 classifying stack $BG$。
4. 说明 $H^1(U,G)$ 与 torsors、$H^2(U,A)$ 与 gerbes 的关系。
5. 解释 1-stacks 如何嵌入 higher stacks。

**综合题 46.** 连接有效下降、Barr-Beck 和范畴 Galois 理论。

1. 定义 $p:E\to B$ 的 descent category。
2. 定义 effective descent morphism。
3. 用 monadicity 判别 effective descent。
4. 定义 covering、trivial covering 和 normal extension。
5. 解释有限 Galois 扩张中 descent datum 与 Galois 群作用的关系。

**综合题 47.** 连接多项式函子、species、解析函子和 W-types。

1. 从 $I\leftarrow E\to B\to J$ 定义多项式函子。
2. 在 Set 中推导 $P(X)=\sum_{b\in B}X^{E_b}$。
3. 定义 species 与解析函子。
4. 证明常值 species 给出有限多重集函子。
5. 证明自然数对象是 $P(X)=1+X$ 的 W-type。

**综合题 48.** 连接 $\infty$-cosmos、homotopy 2-category 和模型无关高阶范畴论。

1. 定义 $\infty$-cosmos。
2. 定义 homotopy 2-category $\mathcal K_2$。
3. 定义 $\infty$-cosmos 中的 equivalence、isofibration 和 adjunction。
4. 证明左伴随保持表示性 colimit。
5. 说明 $\infty$-cosmos 为什么支持模型无关的高阶范畴论。

**综合题 49.** 连接正交性、因子化系统和局部对象。

1. 定义 $f\perp g$ 以及 ${}^\perp\mathcal S,\mathcal S^\perp$。
2. 定义正交因子化系统，并证明分解唯一到唯一同构。
3. 证明 $\mathbf{Set}$ 中 surjection-injection 构成正交因子化系统。
4. 用正交性刻画 $\mathcal S$-局部对象。
5. 定义弱因子化系统，并说明它与正交因子化系统的关系。

**综合题 50.** 连接 sketches、doctrines 和代数理论。

1. 定义 sketch 及其模型。
2. 说明空 sketch 的模型范畴。
3. 定义有限积理论，并解释群对象的有限积理论。
4. 定义 doctrine，并说明 doctrine 强弱与模型条件的关系。
5. 说明小范畴为何可由有限极限 sketch 表示。

**综合题 51.** 连接幂等分裂、Karoubi 包络和绝对余极限。

1. 定义幂等和幂等分裂。
2. 证明分裂对象唯一到唯一同构。
3. 定义 $\operatorname{Kar}(\mathcal C)$，并证明 $\mathcal C\to\operatorname{Kar}(\mathcal C)$ 全忠实。
4. 证明 $\operatorname{Kar}(\mathcal C)$ 幂等完备。
5. 定义绝对余极限，并说明分裂 coequalizer 为何绝对。

**综合题 52.** 连接共尾性、反射子范畴、Kan 延拓和单子伴随。

1. 定义共尾函子，并证明有左伴随的函子共尾。
2. 用共尾性说明点态左 Kan 延拓可以缩小逗号范畴。
3. 定义反射子范畴，并证明其余极限可由环境余极限再反射得到。
4. 从单子构造 Kleisli 伴随，并说明它恢复原单子。
5. 从单子构造 Eilenberg-Moore 自由-遗忘伴随，并证明 Kleisli 范畴全忠实嵌入 Eilenberg-Moore 范畴。
