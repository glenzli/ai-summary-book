# 章节来源注释

本文档记录各章的主要资料源、书内证明范围和外部输入边界。它补充 [SOURCES.md](SOURCES.md) 与 [D_theorem_source_index.md](D_theorem_source_index.md)。

## 第一部分：普通范畴论基础

| 章节 | 主资料源 | 书内证明范围 | 外部输入 |
|---|---|---|---|
| 00 序章 | Mac Lane, Riehl, Leinster | 说明范围、严格性标准、终稿阅读约定 | 无 |
| 01 范畴、函子与自然变换 | Mac Lane, Leinster, Riehl | 范畴、反范畴、自然变换、等价判别、骨架和边界例子 | 使用选择原则构造拟逆，见附录 A |
| 02 泛性质与 Yoneda | Mac Lane, Riehl | 终对象唯一性、可表性、Yoneda、Yoneda 完全忠实、可表性边界和 Yoneda 计算原则 | 无 |
| 03 极限与余极限 | Mac Lane, Riehl | 有限极限构造、函子范畴逐点极限、表示性刻画、共尾性定理、Set 余等化子、创造极限、有限极限反例 | 无 |
| 04 伴随函子 | Mac Lane, Riehl | Hom 定义、单位余单位、保持极限/余极限、全忠实判别、反射子范畴余极限、偏序伴随、对角函子例子、保持性反例 | 无 |
| 05 可表、密度与生成元 | Riehl, Mac Lane | 元素范畴、预层密度、生成族判别、可表投射性、稠密与本质满边界 | 无 |
| 06 Kan 延拓 | Mac Lane, Riehl | Kan 泛性质、左右点态公式、完全忠实情形、点态公式稳定性、共尾缩小、有伴随时的点态退化、存在性边界 | 存在性依赖目标范畴相应极限/余极限 |
| 07 单子与代数 | Mac Lane, Borceux, Riehl | 单子、伴随产生单子、代数、Kleisli 范畴、Kleisli/EM 伴随、幂等单子、反射子范畴单子 | Beck 单子性定理 |

## 第二部分：结构性范畴论

| 章节 | 主资料源 | 书内证明范围 | 外部输入 |
|---|---|---|---|
| 08 幺半范畴 | Mac Lane, Kelly | 幺半定义、辫和对称定义、代数对象、松/强幺半函子传代数对象、单子作为代数对象、非辫性例子 | Mac Lane 相干性 |
| 09 闭范畴与 Day 卷积 | Kelly, Day, Riehl | 闭结构、评价、笛卡尔闭唯一性、单位内部 Hom、指数律、非闭反例、可表和单位 Day 卷积计算 | Day 卷积完整相干性 |
| 10 富范畴 | Kelly | 富范畴、富自然变换 end 公式、enriched Yoneda 证明、富 Yoneda 全忠实、张量/余张量与一对象权重 | 富函子范畴的一般存在性假设 |
| 11 end 与 coend | Mac Lane, Kelly, Riehl | end/coend 公式、自然变换 end 公式、co-Yoneda、集合值 coend 商公式、end/coend 形式 Yoneda、存在性边界 | Fubini for ends/coends |
| 12 可表现范畴 | Adámek-Rosický, Borceux | 正则基数、紧对象例子、可达定义、预层范畴局部可表现、强生成子、紧生成对象检测自然同构、Set 有限生成与基数边界 | Ind 刻画、局部可表现结构定理、局部可表现范畴伴随函子定理 |
| 13 正合与 Grothendieck 范畴 | Borceux, Popescu | 核余核、image/coimage、阿贝尔范畴、核/余核判别单满、正合函子保持 image/coimage、$R$-模范畴 Grothendieck 性、阿贝尔非 Grothendieck 边界 | Gabriel-Popescu |
| 14 站点与 topos | SGA 4, Johnstone, Mac Lane-Moerdijk | Grothendieck 拓扑、sheaf 条件、separated 预层、plus 构造口径、subcanonical、预层 topos、sheaf 极限创建、sheaf 化反射泛性质、几何态射定义与复合 | sheaf 化左正合、plus 构造完整证明、Giraud 定理 |

## 第三部分：高阶与同伦范畴论

| 章节 | 主资料源 | 书内证明范围 | 外部输入 |
|---|---|---|---|
| 15 2-范畴与双范畴 | Bénabou, Mac Lane, Leinster | 严格 2-范畴、交换律、双范畴定义、2-函子、伪函子、单对象 2-范畴/幺半范畴比较、$\mathbf{Cat}$ 中等价判别 | 等价严格化为伴随等价、一般双范畴相干严格化 |
| 16 模型范畴 | Quillen, Hovey, Hirschhorn | 模型范畴公理、Quillen 伴随定义、相对范畴、离散模型结构、Quillen 伴随保持 cofibrant/fibrant 对象 | 同伦范畴计算、导出函子、Dwyer-Kan localization |
| 17 单纯集与 quasi-category | Joyal, Lurie, Riehl-Verity, Kerodon | 单纯集、标准单纯形计算、horn、nerve 内角唯一填充、nerve 全忠实、ordinary nerve 为 Kan 的群胚判别、Kan 复形基本性质 | Joyal 模型结构 |
| 18 $\infty$-极限与伴随 | Lurie HTT, Riehl-Verity, Kerodon | 左右映射空间模型、ordinary nerve 的离散 Hom 计算、ordinary nerve 中等价边判别、Kan complex 的同伦范畴群胚性、correspondence 表示性口径、adjunction data 低维展开、walking adjunction、scaled nerve 低维口径、join/slice 基础、ordinary pullback 与终对象恢复、普通极限比较、伴随保持余极限的映射空间证明 | 同伦范畴构造、映射空间 Kan 性、$\infty$-伴随刻画、adjunction data 等价、scaled model structure |
| 19 Cartesian fibration | Lurie HTT, Kerodon, Riehl-Verity | 映射空间判别、ordinary Grothendieck construction、基为点/$[1]$/$[2]$ 的低维模型、Cartesian 传输函子、Cartesian lift 复合、marked simplicial set 定义、纤维内态射判别、常值族例子、Kan 延拓高阶形式、Cartesian sections | Cartesian model structure、ordinary fibration nerve 比较、straightening、sections-as-limits |
| 20 稳定 $\infty$-范畴 | Lurie HA, Hovey-Schwede-Shipley, BBD, Boardman | 稳定定义、纤维余纤维、sequential prespectrum、$\Omega$-谱、映射谱、smash product、悬挂-环路互逆、正合函子、有限余极限保持推出 exactness 判别、t-结构、heart 核余核、heart 加性、长正合列表述、exact couple、有限滤过谱序列收敛、完备滤过和条件收敛入口 | 同伦范畴三角结构、谱富化、谱稳定化、smash product、heart 阿贝尔性、cohomology 长正合列、一般谱序列收敛、Boardman 收敛理论 |
| 21 高阶 topos | Lurie HTT, Rezk | space 值 sheaf、离散 sheaf 与 ordinary sheaf 比较、sheaf 极限逐点计算、objectwise 离散 sheaf 恢复 ordinary sheaf、超覆盖与超下降入口、effective epimorphism、groupoid object 入口、$\infty$-topos 定义、截断、Postnikov tower、hypercompletion、几何态射和点 | hyperdescent 等价表述、高阶 Giraud、groupoid objects 有效性、0-截断 topos、hypercompletion 理论 |
| 22 高阶代数 | Lurie HA, May, Boardman-Vogt, Ayala-Francis, Francis | $\infty$-operad、active/inert 分解、Segal 条件、多重映射空间、幺半 $\infty$-范畴、$E_n$-代数入口、普通幺半范畴中 $E_1$-代数恢复、幺半函子传递代数、模范畴、bar 构造、相对张量积、Morita、单位双模、矩阵代数 Morita 等价、Frobenius 代数二维 TFT 影子、中心、因子化同调、fully dualizable、cobordism hypothesis | Dunn additivity、代数范畴 presentability、模范畴和相对张量积存在性、Deligne 型定理、因子化同调 excision、smooth/proper 可对偶性判别、cobordism hypothesis |
| 23 可表现 $\infty$-范畴 | Lurie HTT, HA, Riehl-Verity, Kerodon | 预层 $\infty$-范畴、可表紧性、presentable 定义、局部化泛性质、局部等价映射空间判别、局部对象极限创建、left exact localization 与 $\infty$-topos、稳定 exact localization | $\infty$-Yoneda、Ind 刻画、presentable 伴随函子定理、accessible localization、$\operatorname{Pr}^L$ 闭对称幺半结构 |
| 24 Profunctor 与 correspondence | Bénabou, Street, Kelly, Mac Lane, Lurie | profunctor 定义、coend 复合、单位律、函子诱导 profunctor 伴随、右/左可表示 profunctor 伴随判别、离散关系例子、Cauchy completion、加权余极限的 coend 公式 | $\mathbf{Prof}$ 双范畴相干性、Cauchy completion 的普遍性质、$\infty$-correspondence 的 $(\infty,2)$-结构 |
| 25 富 profunctor 与 equipment | Kelly, Street, Shulman, Grandis-Paré, Lurie | 富 profunctor、富 coend 复合、companion/conjoint、二重胞腔、mate 对应、低维 exact square 计算、Beck-Chevalley 条件、indexed category 与 fibration 比较 | 富 $\mathbf{Prof}_{\mathcal V}$ 双范畴相干性、equipment/framed bicategory 高阶模型、six functor 形式化 |
| 26 紧生成与 Bousfield 局部化 | Brown, Neeman, Thomason, Lurie HA | compact objects、compact generation、localizing subcategory、Verdier quotient 泛性质、Bousfield localization、局部等价余纤维判别、smashing localization | Brown 表示性、presentable 稳定商存在性、Neeman-Thomason 紧对象商定理 |
| 27 dg 范畴与导出 Morita | Keller, Toën, Tabuada, Lurie HA, Canonaco-Stellari | dg category、$H^0$、dg Yoneda、dg Yoneda 全忠实、单对象 dg category/dg algebra 比较、可表模紧性、quasi-equivalence、dg bimodule、Hochschild 公式 | dg 模模型结构、dg nerve 稳定性、perfect modules 等于 compact objects、导出 Morita 定理、Hochschild 型 Morita 不变性 |
| 28 六操作形式主义 | Grothendieck, Verdier, Deligne, Ayoub, Cisinski-Déglise, Gaitsgory-Rozenblyum | 抽象六操作资料、基变换态射构造、投影公式相干、恒等态射低维检查、proper compatibility 复合、recollement 检测零对象与闭支撑恢复、Verdier 对偶的形式计算 | 具体几何理论中的六操作存在性、基变换定理、投影公式、recollement、Verdier 对偶和 purity |
| 29 相对范畴与模型比较 | Dwyer-Kan, Rezk, Bergner, Barwick-Kan, Lurie HTT, Riehl-Verity | relative category、$\infty$-localization 泛性质、saturation、relative functor 导出、只倒置同构时 nerve 恢复、simplicial category、Dwyer-Kan equivalence、$2$-out-of-$3$、Segal 条件、CSS 定义 | simplicial localization、hammock localization、coherent nerve、Bergner-Joyal 比较、Rezk CSS 模型和 Quillen 等价链 |
| 30 dg 商与非交换 motives | Drinfeld, Keller, Toën, Tabuada, Blumberg-Gepner-Tabuada | exact sequence、flasque swindle、dg quotient 泛性质、Drinfeld quotient 的后果、additive/localizing invariant、零判别、直和公式、derived Morita 不变性、motives 普遍性质应用 | Drinfeld quotient 构造、Verdier quotient 的 dg enhancement、$K$-理论局部化、universal additive/localizing motives、HH/THH localizing 性 |
| 31 Perverse sheaves 与 recollement | BBD, Goresky-MacPherson, Kashiwara-Schapira, Beilinson, Bernstein, Deligne | 可构造导出范畴定义、stratum restrictions 保守性、perverse 条件、点空间例子、recollement 判别、闭支撑 heart 等价、中间延拓 simple 性、Verdier 对偶形式推论 | perverse t-结构存在性、BBD gluing、中间延拓刻画、Verdier 对偶 t-exactness、nearby/vanishing cycles 构造 |
| 32 Chromatic homotopy | Ravenel, Hopkins-Smith, Hovey-Palmieri-Strickland, Bousfield | Bousfield class 偏序、join 计算、$E$-equivalence、Bousfield equivalent 谱的局部等价类、type 定义后果、telescope conjecture 后果、fracture square 零检测、范畴论解释 | Morava $K$-theory 存在性、厚子范畴定理、周期性定理、telescope conjecture、chromatic fracture square |
| 33 $D$-modules 与 Riemann-Hilbert | Kashiwara-Schapira, Hotta-Takeuchi-Tanisaki, Borel, Mebkhout, Beilinson-Bernstein | $D_X$ 定义、connection 解释、点空间例子、de Rham 平凡 connection 计算、proper 相容形式、Kashiwara-recollement 比较、等价运输 t-结构/heart/伴随 | Bernstein inequality、regular holonomic 理论、Riemann-Hilbert correspondence、$D$-module 六操作、Kashiwara equivalence |
| 34 导出代数几何 | Lurie DAG/SAG, Toën-Vezzosi, Gaitsgory-Rozenblyum, Illusie, Pridham | derived affine、prestack/stack 定义、仿射映射空间、派生仿射拉回、QCoh 仿射恢复、cotangent complex 表示性、cotangent 零判别、transitivity triangle、切复形说明和切映射 | derived/spectral stack representability、QCoh compact generation、Lurie-Pridham formal moduli、IndCoh 与 singular support |
| 35 Barr-Beck-Lurie 与 descent | Barr-Beck, Lurie HA, Riehl-Verity, Gaitsgory-Rozenblyum | monad 定义、伴随产生 monad、comparison functor、comparison 与遗忘函子、split simplicial realization、monadic 保守性、单子性等价不变、模范畴例子、Cech descent 说明、恒等 Cech descent | Barr-Beck-Lurie 单子性、comonadic Barr-Beck、faithfully flat descent |
| 36 Tannaka duality | Saavedra, Deligne-Milne, Lurie SAG, Gaitsgory-Rozenblyum, Toën-Vezzosi | neutral Tannakian 定义、coend 重构公式解释、仿射 Tannaka 公式证明、descent 相容说明、由 $\operatorname{QCoh}$ 判别栈等价、态射重构、$BG$ 重构和环路群解释 | 经典 Tannaka duality、高阶 Tannaka duality、$\operatorname{QCoh}(BG)\simeq\operatorname{Rep}(G)$ |
| 37 Tensor triangular geometry | Balmer, Thomason, Hopkins-Smith, Benson-Iyengar-Krause | tt-category、thick tensor ideal、prime ideal、Balmer support 性质、支撑的三角性质、张量幂支撑、Balmer spectrum 函子性、Perfect complexes 支撑零检测、商的张量下降 | Balmer 分类定理、$\operatorname{Spc}(\operatorname{Perf}(R))\cong\operatorname{Spec}R$、有限谱 chromatic prime 描述 |
| 38 THH、cyclotomic trace 与 TC | Bökstedt, Dundas-Goodwillie-McCarthy, Nikolaus-Scholze, Blumberg-Mandell, Blumberg-Gepner-Tabuada | THH trace 口径、Morita 不变性、cyclotomic 忘却、TC 函子性、cyclotomic trace 自然性、相对 $K/TC$、相对 trace、THH 零判别、相对不变量 Morita 不变性、DGM 后果 | THH localizing 性、cyclotomic refinement、TC 公式、cyclotomic trace、Dundas-Goodwillie-McCarthy 定理 |
| 39 Goodwillie calculus | Goodwillie, Arone-Ching, Lurie HA, Heuts | reduced/excisive 定义、正合函子线性、cross-effect 基本计算、收敛的有限多项式情形 | $P_nF$ 存在性、Goodwillie 层的齐次性、derivatives 分类 homogeneous functors、chain rule、tower 收敛定理 |
| 40 Motivic homotopy | Morel-Voevodsky, Ayoub, Cisinski-Déglise, Hoyois, Robalo | motivic spaces 定义、presentability、$\mathbb A^1$-局部对象识别、稳定化定义、Thom 零丛计算、compact generation 生成子检测形式 | stable motivic homotopy category、复 realization 相容性、motivic 六操作、homotopy purity、motivic Eilenberg-Mac Lane spectra、compact generation |
| 41 范畴逻辑与类型论 | Lawvere, Makkai-Reyes, Johnstone, Jacobs, Awodey, Hofmann-Streicher, Voevodsky, Shulman | 子对象纤维化、regular existential、Heyting implication、$\Sigma_f$ 伴随、comprehension category 替换、groupoid 恒等类型、几何逻辑保持性 | elementary topos 内部逻辑、identity types 的 weak factorization 语义、univalent universes、HoTT 的 $\infty$-topos 模型 |
| 42 因子化同调 | Lurie HA, Ayala-Francis, Francis, Costello-Gwilliam, Dunn, May | $\operatorname{Disk}_n$、$E_n$-代数对象值、Kan 延拓公式、圆盘计算、不交并公式、excision 后果、圆作用来源 | 因子化同调 excision、$\int_{S^1}A\simeq HH(A)$、非阿贝尔 Poincare 对偶、局部常值 factorization algebras 与 $E_n$-代数等价 |
| 43 Condensed 与 solid | Clausen-Scholze, Scholze, Barwick-Haine, Johnstone | profinite 站点、离散对象全忠实、凝聚化一致性、Grothendieck 后果、solid tensor 形式后果、导出局部化保持余极限 | condensed topos 与 condensed abelian groups 的结构、拓扑空间嵌入、投射生成元、solidification、analytic rings 和 solid derived categories |
| 44 语法范畴与分类 topos | Lawvere, Makkai-Reyes, Johnstone, Mac Lane-Moerdijk, Hyland-Johnstone-Pitts | 空有限积语法范畴计算、coherent 析取解释、分类 topos 唯一性、泛模型拉回、topos 子对象 tripos | 语法范畴泛性质、分类 topos 存在性、tripos-to-topos、PER 表示 |
| 45 正合完成与 allegory | Carboni, Vitale, Freyd-Scedrov, Johnstone, Makkai-Reyes | 关系复合、函数图像复合、exact completion 唯一性、effective quotient 稳定性、regular 公式关系解释 | regular completion、exact completion 存在性、关系 allegory、exact category 与 allegory 表征 |
| 46 Cohesive topos | Lawvere, Schreiber, Shulman, Anel-Biedermann-Finster-Joyal | cohesive 伴随串、flat 幂等、left exact modality 保 pullback、de Rham 局部对象判别、cohomology 映射空间公式 | cohesive $\infty$-topos 例子、differential cohesion、modal HoTT 模型、de Rham stack 语义 |
| 47 层化同伦与 exit-path | MacPherson, Treumann, Lurie, Ayala-Francis-Rozenblyum, Goresky-MacPherson | 单层 exit 计算、两层方向性、exit 粘合数据、perverse sheaf 组合骨架、单层因子化同调比较 | exit simplicial set 为 quasi-category、constructible sheaves 分类、层化因子化同调和 excision |
| 48 高阶 Morita 与 Koszul 对偶 | Lurie HA, Francis, Ayala-Francis, Haugseng, Toën, Ginot | $n=1$ Morita 计算、矩阵代数 Morita 等价、HH Morita 不变性、higher trace 低维退化、单位 Koszul 对偶 | 高阶 Morita $(\infty,n)$-范畴构造、smooth/proper fully dualizable 判别、higher traces、$E_n$-Koszul duality、factorization/Koszul 对偶 |
| 49 Derivator | Grothendieck, Heller, Franke, Groth, Maltsiniotis, Cisinski | 预 derivator 函子性、唯一函子的同伦极限解释、equivalence of categories 诱导等价、stable 定义展开、$\mathbb D_C(*)$ 计算 | derivator 来源、点态公式、稳定 derivator 三角结构、$\infty$-范畴到 derivator 的构造 |
| 50 Stacks 与 gerbes | Giraud, Breen, Jardine, Laumon-Moret-Bailly, Lurie HTT, Noohi | sheaf 作为离散 stack、descent datum、有效粘合解释、平凡群 $BG$、Cech 1-cocycle、1-stack 嵌入 higher stack | $BG$ 为 stack、$H^1$ torsor 分类、$H^2$ gerbe 分类、higher stack hyperdescent |
| 51 范畴 Galois 与 descent | Janelidze-Kelly, Borceux-Janelidze, Grothendieck, Barr-Beck, Johnstone | descent datum、同构 effective descent、monadicity 推论、trivial covering pullback 稳定、有限 Galois descent 计算 | regular/topos effective descent、Galois structures、normal extensions 与 groupoid actions |
| 52 多项式函子与 species | Joyal, Gambino-Kock, Kock, Abbott-Altenkirch-Ghani, Moerdijk-Palmgren | Set 中多项式公式、container 等价、species 解析函子、多重集例子、自然数对象为 W-type、list 单子 | W-types 存在性、多项式单子与 operads、同伦 species 和群胚解析函子 |
| 53 $\infty$-cosmoi | Riehl-Verity, Joyal, Lurie HTT, Rezk, Bergner | homotopy 2-category、quasi-category 同伦范畴、isofibration 公理后果、左伴随保持表示性 colimit、representable module | $\infty$-cosmos 公理系统、模型来源、adjunction 定义等价、modules/weighted limits/Kan extensions 的模型无关理论 |
| 54 正交与因子化系统 | Freyd-Kelly, Cassidy-Hébert-Kelly, Adámek-Rosický, Riehl | 正交类包含、正交分解唯一性、Set epi-mono 分解、局部对象正交刻画、正交到弱因子化、retract 闭包 | 小对象论证、cofibrantly generated weak factorization systems |
| 55 Sketches 与 doctrines | Ehresmann, Makkai-Reyes, Adámek-Rosický, Johnstone, Barr-Wells | 空 sketch 模型、群对象有限积理论、doctrine 强弱、小范畴有限极限 sketch、模型 full subcategory | sketch 模型范畴可表现性、geometric sketch 分类 topos |
| 56 Karoubi 与绝对余极限 | Karoubi, Bénabou, Street, Kelly, Cauchy completion literature | 幂等分裂唯一性、Karoubi 包络全忠实、Karoubi 幂等完备、幂等完备时等价、分裂 coequalizer 绝对性 | enriched Cauchy completion 与绝对加权余极限一般理论 |

## 附录

| 附录 | 主资料源 | 书内证明范围 | 外部输入 |
|---|---|---|---|
| A 宇宙与大小 | SGA, Mac Lane | universe 口径、预层范畴大小 | Grothendieck universe 存在性作为集合论背景 |
| B 单纯形范畴 | Goerss-Jardine, Lurie | 单纯恒等式、nerve 内角唯一填充 | 无 |
| C 泛性质模板 | 本书总结 | 证明模板 | 无 |
| D 资料源定理索引 | 本书总结 | 来源索引 | 无 |
| E 高阶技术模型 | Lurie HTT, Kerodon, Cisinski | join/slice 基础、Kan complex 与 quasi-category 比较、scaled simplicial set 低维解释 | Joyal 模型结构、marked model structure、scaled model structure、ordinary fibration nerve 比较 |
| G 终稿化审查标准 | 本书总结 | 终稿标准、逐章审查表、外部输入审查表 | 无 |
