# 第二十一章：Operad 理论的开放问题与结构性边界

前二十章反复出现一种共同现象：严格对象、同伦模型与几何实现之间往往已有比较定理，但比较的自然性、可计算性和适用范围仍留下缺口。比如，ordinary operad 同时产生 dendroidal nerve 与 category-of-operators nerve；一般底环上的逐 arity 弱等价却未必诱导代数范畴等价；Fukaya 型局部模型即使具有形式上的 operadic composition，也仍需独立的紧性与 descent 定理。本章不按论文年份列目录，而把这些缺口整理成稳定的研究问题。每个问题都先固定输入对象和期望结论，再用已完成章节中的低阶例子检验其边界；近期文献只是问题背景，不决定问题的表述。

## 21.1 模型比较为何还需要可计算版本

模型比较定理通常断言两个模型范畴经过局部化后呈现等价的 infinity-categories。这个结论足以传递等价不变量，却不自动告诉我们一个给定 corolla、一个具体 algebra object 或一个 nullary operation 在比较函子下变成什么。对于计算和几何应用，后一个问题往往与等价本身同样重要。

**约定 21.1（开放问题的口径）.** 本章所谓开放问题，是一个尚未被本书所列统一定理覆盖的问题族，不表示所有特例都未知。每个问题都固定：

1. 输入对象所在的模型与大小层级；
2. 期望构造的源、靶和方差；
3. 要保持的结构，例如 operation spaces、algebras、constants 或 descent；
4. 至少一个 strict 或低 arity 特例。

已有定理覆盖某个子类时，问题只指其余范围或更强的显式性要求。

**例 21.2（同一 strict operad 的两种 nerve）.** 设 $\mathcal P$ 是允许 arity $0$ 的小 colored operad。第十六章给出
$$
N_d(\mathcal P)\in\mathbf{dSet},
$$
而第十八章给出
$$
N(\mathcal P^\otimes)\longrightarrow N(\mathbf{Fin}_*).
$$
对颜色 $c_1,\ldots,c_n,c$，前者在 corolla 上的 operation set 是
$$
\mathcal P(c_1,\ldots,c_n;c),
$$
后者 over $\mu_n:\langle n\rangle\to\langle1\rangle$ 的相应 mapping-space 分量也应恢复同一离散运算数据。这个 strict 计算是任何比较函子的最低要求。外部输入定理 18.20 覆盖 open/no-constants 的同伦理论比较，但它不直接给出本书默认含 constants 情形中逐 corolla 的单一步骤公式。

**开放问题 21.3（带 constants 的显式模型比较）.** 对一类允许 nullary operations、具有小颜色空间且 operation spaces fibrant 的 simplicial colored operads，构造并比较通往 dendroidal 与 Lurie-style 模型的显式派生函子，使其同时满足：

1. 在 strict 离散对象上恢复例 21.2；
2. 保持说明 17.7 的派生 operation spaces；
3. 与取 open 部分及重新加入 constants 的操作相容；
4. 在目标 symmetric monoidal infinity-category 满足相应完备性假设时，诱导 algebra infinity-categories 的等价。

已知 zigzag 解决了若干重要模型范围；问题在于给出覆盖 constants、enrichment 与 algebra objects 的统一且可计算版本，并辨明最弱假设。

## 21.2 一般底环上的 rectification 边界

在特征 $0$ 的良好链复形范畴中，对称群 coinvariants 常有足够的同伦正合性，许多 cofibrant resolution 能严格化代数。正特征中，这一机制会在最简单的对称幂上失败，所以“所有 homotopy commutative algebras 都能严格化”不是模型无关的结论。

**例 21.4（对称幂中的可见障碍）.** 取 $k=\mathbb F_p$ 和附录 X 的 acyclic 链复形
$$
0\longrightarrow k\cdot y\xrightarrow{d}k\cdot x\longrightarrow0,
\qquad |y|=2,\quad |x|=1,\quad dy=x.
$$
在 $\operatorname{Sym}^p(C)$ 中，
$$
d(y^p)=p\,y^{p-1}x=0,
$$
而 $y^p$ 位于最高链次数，不能是边界。因此自由严格交换代数函子不保持 trivial cofibration $0\to C$。这个单一计算没有证明所有正特征 rectification 都失败，却精确指出普通 coinvariants 无法承担无条件证明。

**开放问题 21.5（可判定的 rectification 充要条件）.** 设 $\mathcal M$ 是可组合的对称幺半模型范畴，$\varphi:\mathcal O\to\mathcal P$ 是 admissible operads 之间的 entrywise weak equivalence。寻求可从 $\mathcal M$ 的等变张量性质与 $\mathcal O,\mathcal P$ 的 cofibrancy 数据判定的充要条件，使 extension--restriction adjunction
$$
\varphi_!:\operatorname{Alg}_{\mathcal O}(\mathcal M)
\rightleftarrows
\operatorname{Alg}_{\mathcal P}(\mathcal M):\varphi^*
$$
成为 Quillen equivalence。附录 G 的 symmetric flatness 等条件给出充分条件；问题要求解释哪些部分接近必要，以及失败时由哪个有限群作用检测。

**开放问题 21.6（严格化障碍的同调对象）.** 对未满足问题 21.5 条件的 $\varphi$，构造函子性的 obstruction theory，使第一障碍由
$$
X^{\otimes n}_{h\Sigma_n}\longrightarrow X^{\otimes n}_{\Sigma_n}
$$
或相应 norm/Tate 比较的缺陷测量，并使高阶障碍与自由 operad 的树 filtration 对齐。例 21.4 应成为交换 operad、权重 $p$ 的首个非零障碍，而在特征 $0$ 的 projective 情形中这些障碍应消失。

## 21.3 Infinity-operadic Koszul 理论

经典 Koszul 对偶从二次生成关系出发，以 bar--cobar 和 twisting morphism 连接 operad 与 conilpotent cooperad。Infinity-operad 本身已经把复合放松为同伦相干数据，因此把这套理论推广过去时，必须决定“线性化”“余结构”和“完备化”分别在哪个模型中完成。

**开放问题 21.7（bar--cobar 对偶的模型与完备性）.** 固定交换环 $k$ 以及一种 linear infinity-operad 模型。确定一类对象 $X$ 和一类 conilpotent 或 complete infinity-cooperadic objects $Y$，使得：

1. convolution object 中的 Maurer--Cartan 元素表示 twisting morphisms；
2. bar 与 cobar 构成派生伴随；
3. 在连通性、有限性或完备性假设下，单位或余单位成为等价；
4. 该结论在模型比较下保持。

问题的关键不是形式上写出 $\Omega$ 与 $B$，而是给出收敛 filtration、处理无限树展开，并说明代数与余代数两侧的弱等价。

**例 21.8（strict specialization）.** 若 $X$ 来自逐 arity 有限维的 classical Koszul operad $\mathcal P$，问题 21.7 的构造必须退化为
$$
\Omega\mathcal P^{\ash}\xrightarrow{\sim}\mathcal P
$$
以及定理 9.20 的自然双射
$$
\operatorname{Hom}_{\mathrm{dgOp}}(\Omega\mathcal C,\mathcal P)
\cong
\operatorname{Tw}(\mathcal C,\mathcal P)
\cong
\operatorname{Hom}_{\mathrm{dgCoop}}(\mathcal C,B\mathcal P).
$$
若候选理论在 $\operatorname{Ass}$、$\operatorname{Com}$、$\operatorname{Lie}$ 上不能恢复这些映射及其代数范畴，它讨论的就是另一种对偶，而不是经典 Koszul 理论的扩张。

**开放问题 21.9（模型无关且可计算的 operadic homology）.** 构造 infinity-operad $X$ 的同调或 Quillen homology，使它：

1. 对 strict operad 与已知 operadic Quillen homology 一致；
2. 在 dendroidal 与 Lurie-style 比较下保持；
3. 带有由树顶点数或 skeletal filtration 产生的收敛谱序列；
4. 能检测问题 21.7 的 bar--cobar 单位何时为等价。

仅取各树值或各 operation space 的普通同调通常会丢失 face maps 所编码的复合，所以不满足这些要求。

## 21.4 Operadic categories 与广义树形

有根树不是组织代入的唯一组合对象。Operadic category 把 cardinality、fiber 和 local terminal object 直接写进一个范畴，从而有可能同时容纳树、图、轮换结构与其他代入模式。困难在于：一套足以定义 strict substitution 的公理，不一定自动产生良好的同伦模型或 tensor product。

**开放问题 21.10（operadic category 上的同伦理论）.** 给定小 operadic category $\mathsf O$，确定哪些可检验条件保证：

1. $\mathsf O$-operads 存在由底范畴转移的模型结构；
2. 存在 higher nerve，把 fibrant $\mathsf O$-operads 刻画为某种 Segal/complete objects；
3. Boardman--Vogt 型 tensor product 在派生层面存在并满足预期的 interchange 泛性质；
4. 改变 operadic category 的基函子诱导可控的 Quillen adjunction。

不同 $\mathsf O$ 可能需要不同 normality、tameness 或 polynomial-monad 假设；寻求一个不抹平这些差异的统一框架是问题的一部分。

**例 21.11（有限集 fiber 的低阶检验）.** 在 $\mathbf{Fin}$ 中，对
$$
T\xrightarrow{f}S\xrightarrow{g}R
$$
及 $r\in R$，有集合分解
$$
(gf)^{-1}(r)
=
\coprod_{s\in g^{-1}(r)}f^{-1}(s).
$$
左边表示一次取复合 fiber，右边表示先取外层 fiber 再取内层 fibers。第一章代入乘积的结合律正由这两个索引集合的典范双射给出。任何 operadic-category nerve 的 Segal map 都应在这一例中恢复同一拉平，而不能只恢复 underlying category 的普通复合。

**开放问题 21.12（广义 nerve 之间的比较）.** 对同时可由树范畴、operator category 或 operadic category 描述的结构，构造比较函子并确定其像，使以下数据在比较中可追踪：颜色、nullary operations、fiber substitution、automorphism groups 与 derived operation spaces。一个只在一元截断上等价的 nerve 不足以解决该问题，因为例 21.11 的多输入 fiber 已在线性树之外。

## 21.5 Relative operadic localization

普通 relative category $(\mathcal C,W)$ 的局部化只反演一元箭头。Infinity-operad 还含多输入运算；反演颜色之间的某些一元箭头后，所有以这些颜色为输入或输出的 operation spaces 都必须同步改变。这个相容性是 operadic localization 比 ordinary localization 多出的内容。

**开放问题 21.13（局部化与代数对象的相容性）.** 设
$$
p:\mathcal O^\otimes\to N(\mathbf{Fin}_*)
$$
是小 Lurie-style infinity-operad，$W$ 是 underlying infinity-category $\mathcal O_{\langle1\rangle}$ 中对复合封闭的一类边。研究何种假设保证存在 operadic localization
$$
\mathcal O^\otimes\longrightarrow\mathcal O^\otimes[W^{-1}]
$$
满足对保持 inert edges 的 operad maps 的泛性质，并进一步比较
$$
\operatorname{Alg}_{\mathcal O[W^{-1}]}(\mathcal C)
$$
与那些把 $W$ 送到等价的 $\mathcal O$-algebras 所成的 full subcategory。还需说明该构造与 dendroidal Rezk 型模型及 category-of-operators localization 的关系。已有构造覆盖若干模型；问题针对这些泛性质和 algebra comparison 的统一假设。

**例 21.14（一元退化）.** 若 $\mathcal O$ 只有 unary operations，它就是 ordinary category $\mathcal A$ 的 operadic 写法，且
$$
\mathcal O^\otimes[W^{-1}]
$$
的 $\langle1\rangle$-fiber 必须等价于 Dwyer--Kan localization
$$
L(\mathcal A,W).
$$
此时没有非线性树，问题 21.13 退化为定义 19.3 的泛性质。反之，只验证这个一元特例仍不够：对真正的二元运算 $\theta:(x,y)\to z$，还必须比较反演 $x\to x'$ 后 $\theta$ 所在 operation space 怎样沿输入变量派生变化。

## 21.6 Factorization descent 与 Fukaya 型几何

第二十章把 prefactorization 乘法、Weiss descent 和 factorization homology 分开。Fukaya 理论中的 gluing 还多出一层：局部链复形与高阶乘法来自伪全纯曲线计数，只有紧化模空间的边界分析成立后，代数上的 $A_\infty$ 或更高 operadic 关系才有来源。

**开放问题 21.15（几何 gluing 的统一 operadic 接口）.** 对一个明确的辛几何类别，例如带 stops 的 Liouville sectors，构造一个同时记录局部 Fukaya 型范畴、边界/缺陷模、sectorial gluing 与高阶曲面运算的 colored infinity-operad 或更一般多输入对象，并证明：

1. 局部运算在所选链级模型中良定义且与 choices 无关到相干等价；
2. gluing 满足指定的 descent 或 excision；
3. 代数模型中的 operadic composition 与紧化模空间的边界分层一致；
4. 所得全局对象在几何同伦或适当 Morita 等价下不变。

不同几何设置已有各自定理；这里的问题是统一接口以及模型间可比较性，而不是把所有 Fukaya categories 归入一个无假设结论。

**例 21.16（区间 gluing 的代数影子）.** 设 $A$ 是 $E_1$-algebra，$M$ 是右 $A$-module，$N$ 是左 $A$-module。带边界的一维 factorization homology 给出外部输入公式
$$
\int_{[0,1]}(M,A,N)\simeq M\otimes_A^{\mathbf L}N.
$$
当 $A=k[\varepsilon]/(\varepsilon^2)$ 且两端都取 augmentation module $k$ 时，附录 X 计算得
$$
H_i(k\otimes_A^{\mathbf L}k)\cong k\qquad(i\ge0),
$$
而正则边界条件给出 $A$。因此 gluing 的输出真实依赖边界标签；在 Fukaya 语境中，端点或 stop 数据不能从内部 $A_\infty$-category 遗忘后再恢复。问题 21.15 的任何高维版本都应在一维退化中保留这个 derived tensor product。

**开放问题 21.17（几何高阶运算的可比较不变量）.** 构造能从问题 21.15 的几何模型中提取、又可在不同 transversality 与 chain models 之间比较的 operadic invariants，例如带 filtration 的 Hochschild/factorization homology、中心或 deformation complexes。所需比较必须区分 quasi-equivalence、Morita equivalence 与同调群同构，并说明 compactification strata 如何诱导 invariant 上的乘法、括号或 BV 型算子。

## 21.7 一张由障碍而非年份组织的地图

这些问题并非六条互不相干的支线。模型比较决定一个构造能否换语言；rectification 决定同伦结构何时能换成严格结构；Koszul 对偶与 operadic homology 提供可计算不变量；operadic categories 扩大允许的组合形状；relative localization 控制反演后多输入运算如何变化；几何 descent 则检验这些形式机制能否承受分析与 gluing。例 21.2、21.4、21.8、21.11、21.14 和 21.16 分别给出每个问题必须通过的最低测试。未来结果无论采用何种模型，只要不能在这些特例中恢复已知计算，就不能直接接入前二十章的结论链。

## 练习

**练习 21.1.** 对 $\mathcal P=\operatorname{Com}$，分别计算 $N_d(\mathcal P)_{C_0}$ 与 $\operatorname{Com}^\otimes$ over $\langle0\rangle$，说明含 constants 的比较必须保留什么数据。

**练习 21.2.** 重做例 21.4 的 $d(y^p)$ 计算，并指出证明在哪一步使用 $\operatorname{char}k=p$、在哪一步使用 $|y|=2$ 为偶数。

**练习 21.3.** 取 $\mathcal P=\operatorname{Ass}$，写出例 21.8 的 strict specialization 在权重 $2$ 两棵二叉树上的作用，并说明结合关系如何出现。

**练习 21.4.** 把一个小范畴 $\mathcal A$ 视为只有一元运算的 colored operad，证明问题 21.13 的 operadic mapping universal property 若存在，限制到 $\langle1\rangle$ 后给出 ordinary localization 的 universal property。

**练习 21.5.** 在例 21.16 中分别取 $(M,N)=(A,A)$ 与 $(k,k)$，写出 two-sided bar construction 的前三个 simplicial degrees，并比较两种边界条件产生的链复形。
