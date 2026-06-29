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
