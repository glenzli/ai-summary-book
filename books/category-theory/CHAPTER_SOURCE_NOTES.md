# 章节来源注释

本文档记录各章的主要资料源、书内证明范围和外部输入边界。它补充 [SOURCES.md](SOURCES.md) 与 [D_theorem_source_index.md](D_theorem_source_index.md)。

## 第一部分：普通范畴论基础

| 章节 | 主资料源 | 书内证明范围 | 外部输入 |
|---|---|---|---|
| 00 序章 | Mac Lane, Riehl, Leinster | 只说明范围和严格性标准 | 无 |
| 01 范畴、函子与自然变换 | Mac Lane, Leinster, Riehl | 范畴、反范畴、自然变换、等价判别 | 使用选择原则构造拟逆，见附录 A |
| 02 泛性质与 Yoneda | Mac Lane, Riehl | 终对象唯一性、可表性、Yoneda、Yoneda 完全忠实 | 无 |
| 03 极限与余极限 | Mac Lane, Riehl | 有限极限构造、函子范畴逐点极限 | 小余极限逐点计算作为标准对偶事实 |
| 04 伴随函子 | Mac Lane, Riehl | Hom 定义、单位余单位、保持极限/余极限 | 无 |
| 05 可表、密度与生成元 | Riehl, Mac Lane | 元素范畴、预层密度、生成族、可表投射性 | 无 |
| 06 Kan 延拓 | Mac Lane, Riehl | Kan 泛性质、点态公式、完全忠实情形 | 存在性依赖目标范畴相应极限/余极限 |
| 07 单子与代数 | Mac Lane, Borceux, Riehl | 单子、伴随产生单子、代数、Kleisli 范畴 | Beck 单子性定理 |

## 第二部分：结构性范畴论

| 章节 | 主资料源 | 书内证明范围 | 外部输入 |
|---|---|---|---|
| 08 幺半范畴 | Mac Lane, Kelly | 幺半定义、代数对象、强幺半函子传代数对象 | Mac Lane 相干性 |
| 09 闭范畴与 Day 卷积 | Kelly, Day, Riehl | 闭结构、评价、笛卡尔闭唯一性、可表 Day 卷积计算 | Day 卷积完整相干性 |
| 10 富范畴 | Kelly | 富范畴、富自然变换 end 公式、enriched Yoneda 证明 | 富函子范畴的一般存在性假设 |
| 11 end 与 coend | Mac Lane, Kelly, Riehl | end/coend 公式、自然变换 end 公式、co-Yoneda | Fubini for ends/coends |
| 12 可表现范畴 | Adámek-Rosický, Borceux | 正则基数、紧对象例子、可达定义、预层范畴局部可表现、强生成子 | Ind 刻画、局部可表现结构定理、局部可表现范畴伴随函子定理 |
| 13 正合与 Grothendieck 范畴 | Borceux, Popescu | 核余核、image/coimage、阿贝尔范畴、正合函子、$R$-模范畴 Grothendieck 性 | Gabriel-Popescu |
| 14 站点与 topos | SGA 4, Johnstone, Mac Lane-Moerdijk | Grothendieck 拓扑、sheaf 条件、separated 预层、plus 构造口径、subcanonical、预层 topos、几何态射定义 | sheaf 化左正合、plus 构造完整证明、Giraud 定理 |

## 第三部分：高阶与同伦范畴论

| 章节 | 主资料源 | 书内证明范围 | 外部输入 |
|---|---|---|---|
| 15 2-范畴与双范畴 | Bénabou, Mac Lane, Leinster | 严格 2-范畴、交换律、双范畴定义 | 等价严格化为伴随等价 |
| 16 模型范畴 | Quillen, Hovey, Hirschhorn | 模型范畴公理、Quillen 伴随定义 | 同伦范畴计算、导出函子、Dwyer-Kan localization |
| 17 单纯集与 quasi-category | Joyal, Lurie, Riehl-Verity, Kerodon | 单纯集、标准单纯形计算、horn、nerve 内角唯一填充、Kan 复形基本性质 | Joyal 模型结构 |
| 18 $\infty$-极限与伴随 | Lurie HTT, Riehl-Verity, Kerodon | 左右映射空间模型、ordinary nerve 的离散 Hom 计算、correspondence 表示性口径、adjunction data 低维展开、walking adjunction、scaled nerve 低维口径、join/slice 基础、ordinary pullback 与终对象恢复、普通极限比较、伴随保持余极限的映射空间证明 | 同伦范畴构造、映射空间 Kan 性、$\infty$-伴随刻画、adjunction data 等价、scaled model structure |
| 19 Cartesian fibration | Lurie HTT, Kerodon, Riehl-Verity | 映射空间判别、ordinary Grothendieck construction、基为 $[1]$ 和 $[2]$ 的 straightening 低维模型、Cartesian 传输函子、Cartesian lift 复合、marked simplicial set 定义、Kan 延拓高阶形式、Cartesian sections | Cartesian model structure、ordinary fibration nerve 比较、straightening、sections-as-limits |
| 20 稳定 $\infty$-范畴 | Lurie HA, Hovey-Schwede-Shipley, BBD, Boardman | 稳定定义、纤维余纤维、sequential prespectrum、$\Omega$-谱、映射谱、smash product、悬挂-环路互逆、正合函子、t-结构、heart 核余核、heart 加性、长正合列表述、exact couple、有限滤过谱序列收敛、完备滤过和条件收敛入口 | 同伦范畴三角结构、谱富化、谱稳定化、smash product、heart 阿贝尔性、cohomology 长正合列、一般谱序列收敛、Boardman 收敛理论 |
| 21 高阶 topos | Lurie HTT, Rezk | space 值 sheaf、离散 sheaf 与 ordinary sheaf 比较、超覆盖与超下降入口、effective epimorphism、groupoid object 入口、$\infty$-topos 定义、截断、Postnikov tower、hypercompletion、几何态射和点 | hyperdescent 等价表述、高阶 Giraud、groupoid objects 有效性、0-截断 topos、hypercompletion 理论 |
| 22 高阶代数 | Lurie HA, May, Boardman-Vogt, Ayala-Francis, Francis | $\infty$-operad、active/inert 分解、Segal 条件、多重映射空间、幺半 $\infty$-范畴、$E_n$-代数入口、模范畴、bar 构造、相对张量积、Morita、单位双模、矩阵代数 Morita 等价、Frobenius 代数二维 TFT 影子、中心、因子化同调、fully dualizable、cobordism hypothesis | Dunn additivity、代数范畴 presentability、模范畴和相对张量积存在性、Deligne 型定理、因子化同调 excision、smooth/proper 可对偶性判别、cobordism hypothesis |
| 23 可表现 $\infty$-范畴 | Lurie HTT, HA, Riehl-Verity, Kerodon | 预层 $\infty$-范畴、可表紧性、presentable 定义、局部化泛性质、left exact localization 与 $\infty$-topos、稳定 exact localization | $\infty$-Yoneda、Ind 刻画、presentable 伴随函子定理、accessible localization、$\operatorname{Pr}^L$ 闭对称幺半结构 |
| 24 Profunctor 与 correspondence | Bénabou, Street, Kelly, Mac Lane, Lurie | profunctor 定义、coend 复合、单位律、函子诱导 profunctor 伴随、Cauchy completion、加权余极限的 coend 公式 | $\mathbf{Prof}$ 双范畴相干性、Cauchy completion 的普遍性质、$\infty$-correspondence 的 $(\infty,2)$-结构 |
| 25 富 profunctor 与 equipment | Kelly, Street, Shulman, Grandis-Paré, Lurie | 富 profunctor、富 coend 复合、companion/conjoint、二重胞腔、Beck-Chevalley 条件、indexed category 与 fibration 比较 | 富 $\mathbf{Prof}_{\mathcal V}$ 双范畴相干性、equipment/framed bicategory 高阶模型、six functor 形式化 |
| 26 紧生成与 Bousfield 局部化 | Brown, Neeman, Thomason, Lurie HA | compact objects、compact generation、localizing subcategory、Verdier quotient 泛性质、Bousfield localization、smashing localization | Brown 表示性、presentable 稳定商存在性、Neeman-Thomason 紧对象商定理 |
| 27 dg 范畴与导出 Morita | Keller, Toën, Tabuada, Lurie HA, Canonaco-Stellari | dg category、$H^0$、dg Yoneda、可表模紧性、quasi-equivalence、dg bimodule、Hochschild 公式 | dg 模模型结构、dg nerve 稳定性、perfect modules 等于 compact objects、导出 Morita 定理、Hochschild 型 Morita 不变性 |
| 28 六操作形式主义 | Grothendieck, Verdier, Deligne, Ayoub, Cisinski-Déglise, Gaitsgory-Rozenblyum | 抽象六操作资料、基变换态射构造、投影公式相干、proper compatibility、recollement 检测零对象、Verdier 对偶的形式计算 | 具体几何理论中的六操作存在性、基变换定理、投影公式、recollement、Verdier 对偶和 purity |
| 29 相对范畴与模型比较 | Dwyer-Kan, Rezk, Bergner, Barwick-Kan, Lurie HTT, Riehl-Verity | relative category、$\infty$-localization 泛性质、saturation、simplicial category、Dwyer-Kan equivalence、Segal 条件、CSS 定义 | simplicial localization、hammock localization、coherent nerve、Bergner-Joyal 比较、Rezk CSS 模型和 Quillen 等价链 |

## 附录

| 附录 | 主资料源 | 书内证明范围 | 外部输入 |
|---|---|---|---|
| A 宇宙与大小 | SGA, Mac Lane | universe 口径、预层范畴大小 | Grothendieck universe 存在性作为集合论背景 |
| B 单纯形范畴 | Goerss-Jardine, Lurie | 单纯恒等式、nerve 内角唯一填充 | 无 |
| C 泛性质模板 | 本书总结 | 证明模板 | 无 |
| D 资料源定理索引 | 本书总结 | 来源索引 | 无 |
| E 高阶技术模型 | Lurie HTT, Kerodon, Cisinski | join/slice 基础、Kan complex 与 quasi-category 比较、scaled simplicial set 低维解释 | Joyal 模型结构、marked model structure、scaled model structure、ordinary fibration nerve 比较 |
