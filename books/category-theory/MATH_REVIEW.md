# 数学审查记录

本文档记录《范畴论》教材的审查清单、当前风险和后续校订任务。

## 全书审查清单

- [ ] 每章是否声明依赖前置知识。
- [ ] 所有新增符号是否进入 `NOTATION.md`。
- [ ] 每个定义是否说明所在宇宙或范畴。
- [ ] 每个非平凡命题是否带完整证明或外部输入来源。
- [ ] 每个泛性质是否写成自然同构、终对象/始对象或表示性陈述。
- [ ] 是否避免把“同构”“等价”“同伦”等词混用。
- [ ] $\infty$-范畴章节是否说明模型口径。
- [ ] 外部输入定理是否进入 `SOURCES.md` 或章节末尾来源表。
- [ ] 每个练习是否在 `SOLUTIONS.md` 中有对应答案。
- [ ] 综合题是否在 `COMPREHENSIVE_SOLUTIONS.md` 中有对应答案。
- [ ] 新术语是否进入 `TERM_INDEX.md`。
- [ ] 新外部输入定理是否进入 `THEOREM_DEPENDENCIES.md`。
- [ ] 章节资料源是否进入 `CHAPTER_SOURCE_NOTES.md`。

## 当前初稿风险

- 第一章的“完全忠实且本质满推出存在拟逆”使用选择原则；附录 A 已记录，但后续若形式化需明确选择强度。
- 序章至第二章已进入终稿化第一轮：补入终稿阅读约定、骨架、等价边界例子、可表性边界、Yoneda 计算原则和对应答案。第三至第七章已进入终稿化第二轮：补入 Set 余等化子、创造极限、有限极限反例、偏序伴随、对角函子伴随、伴随保持性反例、生成族忠实判别、稠密与本质满边界、有伴随时 Kan 点态公式退化、Kan 延拓存在性边界、幂等单子和反射单子；相关练习答案已同步。第八至第十一章已进入终稿化第三轮：补入辫/对称幺半范畴、松幺半函子传代数对象、单子作为代数对象、非辫性例子、闭结构单位和指数律、非闭反例、Day 单位计算、富 Yoneda 全忠实、张量/余张量、集合值 coend 商公式、end/coend 形式 Yoneda 和存在性边界。第五章预层密度定理和第十一章 co-Yoneda 已补为正文证明。
- 第八章 Mac Lane 相干性、第十二章局部可表现范畴伴随函子定理、第十四章 sheaf 化/Giraud、第十六章模型范畴局部化、第十九章 straightening、第二十章稳定同伦范畴三角结构、第二十一章高阶 Giraud、第二十二章高阶代数存在性定理均为外部输入。
- 第十二至第十四章已进入终稿化第四轮：补入预层范畴局部可表现性、强生成子、紧生成对象检测自然同构、Set 有限生成与基数边界、image/coimage、核/余核判别单满、正合函子保持 image/coimage、模范畴 Grothendieck 性、阿贝尔非 Grothendieck 边界、separated 预层、plus 构造、sheaf 极限创建、sheaf 化反射泛性质和几何态射复合；plus 构造完整证明与 Giraud 定理仍作为外部输入。
- 第十五至第十九章已进入终稿化第五轮：补入 2-函子、伪函子、单对象 2-范畴/幺半范畴比较、$\mathbf{Cat}$ 中等价判别、相对范畴、离散模型结构、Quillen 伴随保持 cofibrant/fibrant 对象、nerve 全忠实、ordinary nerve 为 Kan 的群胚判别、标准单纯形计算、左右映射空间模型、ordinary nerve 中等价边判别、correspondence 表示性口径、adjunction data 低维展开、walking adjunction、scaled nerve 低维口径、ordinary pullback 恢复、普通 Grothendieck construction、基为点/$[1]$/$[2]$ 的 straightening 低维模型、Cartesian 传输函子、纤维内态射判别和 Cartesian sections；完整 scaled model structure 与 straightening 仍作为外部输入。
- 第二十至第二十三章已进入终稿化第六轮：第二十章补入有限余极限保持推出 exactness 判别和三角同伦范畴边界；第二十一章补入 sheaf 极限逐点计算、objectwise 离散 sheaf 恢复 ordinary sheaf；第二十二章补入普通幺半范畴中 $E_1$-代数恢复和幺半函子传递代数结构；第二十三章补入局部等价映射空间判别和局部对象极限创建。heart 阿贝尔性、一般谱序列收敛、hyperdescent 等价、高阶 Giraud、代数范畴 presentability、Ind 刻画、presentable 伴随函子定理和 $\operatorname{Pr}^L$ 幺半结构仍作为外部输入。
- 第二十四章已补 profunctor、coend 复合、Cauchy completion、加权余极限和 $\infty$-correspondence；$\mathbf{Prof}$ 双范畴相干性和高阶 correspondence 的 $(\infty,2)$-结构仍为外部输入。
- 第二十五章已补富 profunctor、equipment、companion/conjoint、Beck-Chevalley 条件、indexed category 与 fibration 比较；高阶 equipment/framed bicategory 模型仍为外部输入。
- 第二十六章已补 compact generation、Brown representability、localizing subcategory、Verdier quotient、Bousfield localization、smashing localization 和 Neeman-Thomason 型定理；Brown 表示性和紧对象商定理仍为外部输入。
- 第二十七章已补 dg category、dg modules、dg Yoneda、pretriangulated enhancement、Morita equivalence、dg bimodules、perfect modules 和 Hochschild chains；dg 模模型结构、dg nerve 稳定性、导出 Morita 定理和 Hochschild 型 Morita 不变性仍为外部输入。
- 第二十八章已补六操作形式主义、基变换态射、投影公式、proper compatibility、recollement、Verdier duality 和 equipment 比较；具体几何理论中的六操作存在性、基变换定理、投影公式、purity 和 Verdier 对偶仍为外部输入。
- 第二十九章已补 relative category、$\infty$-categorical localization、saturation、simplicial category、Dwyer-Kan equivalence、underlying $\infty$-category、coherent nerve、complete Segal space 和模型选择原则；simplicial localization、hammock localization、Bergner-Joyal 比较和 Rezk CSS 模型比较仍为外部输入。
- 第三十章已补 exact sequence of stable categories、flasque swindle、dg quotient、Drinfeld quotient、additive/localizing invariants、noncommutative motives、$K$-theory localization 和 Hochschild/THH localizing 性；Drinfeld quotient 构造、universal motives 和 K/HH/THH 局部化定理仍为外部输入。
- 第三十一章已补 constructible derived category、perverse t-structure、BBD gluing、intermediate extension、Verdier duality 和 nearby/vanishing cycles 入口；perverse t-结构存在性、中间延拓刻画、Verdier 对偶 t-exactness 与 nearby/vanishing cycles 构造仍为外部输入。
- 第三十二章已补 Bousfield lattice、Morava $K$-theory、chromatic type、厚子范畴定理、$v_n$-self maps、telescope conjecture、chromatic fracture square 和范畴论解释；Morava $K$ 理论存在性、Hopkins-Smith 定理、周期性定理、telescope conjecture 和 fracture square 仍为外部输入。
- 第三十三章已补 $D_X$、flat connection、characteristic variety、holonomic/regular holonomic、de Rham/solution functors、Riemann-Hilbert correspondence、$D$-module 六操作和 Kashiwara equivalence；Bernstein inequality、regular holonomic 理论、Riemann-Hilbert 和 Kashiwara equivalence 仍为外部输入。
- 第三十四章已补 derived affine schemes、derived stacks、$\operatorname{QCoh}$、perfect complexes、cotangent complex、transitivity triangle、formal moduli problems、Lurie-Pridham 定理、IndCoh 和 singular support 入口；representability、QCoh compact generation、formal moduli 等价和 IndCoh/singular support 理论仍为外部输入。
- 第三十五章已补 Barr-Beck-Lurie monadicity、comparison functor、split simplicial objects、monadic 保守性、comonadic descent、Cech nerve 和 faithfully flat descent；Barr-Beck-Lurie、comonadic Barr-Beck 与 QCoh fpqc descent 仍为外部输入。
- 第三十六章已补 neutral Tannakian categories、fiber functor、matrix coefficient coend、高阶 Tannaka、仿射 Tannaka 公式、descent 相容和 $BG$ 重构；经典 Tannaka、高阶 Tannaka 与 $\operatorname{QCoh}(BG)\simeq\operatorname{Rep}(G)$ 仍为外部输入。
- 第三十七章已补 tt-category、thick tensor ideal、Balmer spectrum、支撑公式、Thomason subsets、perfect complexes 和 chromatic primes；Balmer 分类定理、$\operatorname{Spc}(\operatorname{Perf}(R))$ 计算和有限谱 Balmer spectrum 仍为外部输入。
- 第三十八章已补 $THH$ trace 口径、圆作用、cyclotomic spectra、$TC$ 公式、cyclotomic trace、Dundas-Goodwillie-McCarthy 定理和 trace methods；THH localizing 性、cyclotomic refinement、TC 理论和 DGM 定理仍为外部输入。
- 第三十九章已补 Goodwillie calculus、reduced/excisive functors、Goodwillie tower、cross-effects、derivatives、chain rule 和收敛入口；$P_nF$ 存在性、homogeneous functor 分类、chain rule 和 analytic convergence 仍为外部输入。
- 第四十章已补 motivic spaces、Nisnevich descent、$\mathbb A^1$-localization、$\mathbf{SH}(S)$、motivic 六操作、homotopy purity、Thom spaces、motivic $H\mathbb Z$ 和 compact generation；Morel-Voevodsky 稳定化、motivic 六操作、purity、motivic spectra 和 compact generation 仍为外部输入。
- 第四十一章已补子对象纤维化、regular existential、Heyting implication、locally Cartesian closed categories、$\Sigma_f\dashv f^*\dashv\Pi_f$、comprehension categories、groupoid 恒等类型、univalence 和几何逻辑；elementary topos 内部语言、weak factorization 语义、univalent universes 和 HoTT 的 $\infty$-topos 模型仍为外部输入。
- 第四十二章已补 $\operatorname{Disk}_n$、$E_n$-代数、因子化同调 Kan 延拓定义、圆盘计算、不交并公式、excision、圆周 Hochschild homology、非阿贝尔 Poincare 对偶和 factorization algebras；excision、$\int_{S^1}A\simeq HH(A)$、非阿贝尔 Poincare 对偶和局部常值 factorization algebra 分类仍为外部输入。
- 第四十三章已补 profinite 站点、condensed sets、离散嵌入全忠实、拓扑空间凝聚化、condensed abelian groups、solidification、solid tensor product、solid modules、solid derived categories 和 analytic rings；condensed topos 结构、投射生成元、solidification 存在性和 analytic rings 理论仍为外部输入。
- 第四十四章已补语法范畴、有限极限理论、coherent 逻辑、分类 topos、泛模型、tripos、generic predicate 和 tripos-to-topos；语法范畴泛性质、分类 topos 存在性、tripos-to-topos 和 PER 表示仍为外部输入。
- 第四十五章已补关系复合、函数图像关系、regular/exact completion、effective equivalence relations、allegory 和 regular 公式的关系解释；regular completion、exact completion、allegory 与 exact category 表征仍为外部输入。
- 第四十六章已补 cohesive 伴随串、shape/flat/sharp 模态、left exact modality、modal type theory、differential cohesion、de Rham shape 和 cohesive cohomology；cohesive $\infty$-topos 实例、differential cohesion 和 modal HoTT 完整语义仍为外部输入。
- 第四十七章已补 conically stratified spaces、exit paths、exit-path $\infty$-category、constructible sheaves、exit 分类、recollement 粘合、perverse sheaf 组合骨架和层化因子化同调；exit simplicial set 的 quasi-category 性、constructible sheaves 分类和层化 excision 仍为外部输入。
- 第四十八章已补 higher Morita $(\infty,n)$-categories、smooth/proper、Morita trace、HH 与因子化同调、higher traces、增广 $E_n$-代数、Koszul dual 和 factorization/Koszul 对偶；higher Morita 构造、fully dualizable 判别、higher trace 定理和 $E_n$-Koszul duality 仍为外部输入。
- 第四十九章已补 derivator、同伦 Kan 延拓、点态公式、stable derivator、与 $\infty$-范畴的比较；derivator 来源、点态公式完整定理、稳定 derivator 三角结构仍为外部输入。
- 第五十章已补 stacks、descent datum、torsors、classifying stacks、gerbes、Cech cocycles 和 higher stacks；$BG$ stack 性、$H^1$ torsor 分类、$H^2$ gerbe 分类和 higher hyperdescent 仍为外部输入。
- 第五十一章已补 descent category、effective descent、monadicity 判别、categorical Galois structure、covering、normal extension、Galois groupoid 和有限 Galois descent；topos/regular effective descent、Galois structures 和 normal extensions/groupoid actions 对应仍为外部输入。
- 第五十二章已补 polynomial functors、containers、species、analytic functors、W-types、多项式单子和 list monad；W-types 存在性、多项式单子与 operads 的一般等价仍为外部输入。
- 第五十三章已补 $\infty$-cosmos、homotopy 2-category、equivalences、isofibrations、adjunctions、modules 和模型无关性；$\infty$-cosmos 例子、伴随定义等价、modules/weighted limits 理论仍为外部输入。
- 第五十四章已补正交性、正交因子化系统、Set 中 epi-mono 分解、局部对象、弱因子化系统和 retract 闭包；小对象论证和 cofibrantly generated weak factorization systems 仍为外部输入。
- 第五十五章已补 sketches、有限积理论、doctrines、essentially algebraic theories、小范畴的有限极限 sketch 和模型 full subcategory；sketch 模型范畴可表现性仍为外部输入。
- 第五十六章已补幂等分裂、Karoubi 包络、幂等完备、绝对余极限、分裂 coequalizer 和 Cauchy completion；富 Cauchy completion 的一般绝对权重理论仍为外部输入。
- 术语索引、章节来源注释和外部输入依赖图已经建立；后续新增章节必须同步维护这些文件。
- 范围边界已在附录 F 固定：后续扩写控制在范畴论本体；外部领域深定理只登记为外部输入，不在本书内部闭合。
- 附录 G 已固定终稿化标准；后续校订应按该附录逐章执行。

## 下一轮建议

1. 继续终稿化第二十四至第二十八章：profunctor、equipment、紧生成稳定范畴、dg 增强和六操作形式主义的内部泛性质、低维计算和外部输入边界。
2. 把第三至第二十三章的部分答案继续从要点升级为逐步证明，尤其是 Beck 定理、Day 相干性、Fubini、局部可表现伴随函子定理、Giraud 定理、模型局部化、straightening、稳定 exactness 和 presentable 伴随函子定理周边保持外部输入边界的练习。
3. 扩写可达、可表现、sketch、doctrine 和局部化之间的内部依赖链。
4. 扩写正交/弱因子化系统、小对象论证的范畴论证明口径。
5. 扩写 Karoubi、Cauchy、Ind/Pro、exact/regular completion 的泛性质与例子。
6. 扩写 2-范畴、双范畴、profunctor、equipment 和 indexed category 的纯范畴论比较。
7. 扩写 quasi-category、Cartesian fibration、$\infty$-cosmos 的模型无关接口，避免外部同伦论计算。
8. 为核心章节继续增加“例子/反例/边界条件”小节。
9. 把 `SOLUTIONS.md` 中核心章节答案从要点升级为逐步证明。
