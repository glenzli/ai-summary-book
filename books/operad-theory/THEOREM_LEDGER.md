# 定理账本：内部证明、外部输入与研究边界

本文件是《Operad Theory》的审校账本。它不新增数学结论，而是把每章结论分成三类：

1. **内部证明**：当前草稿已经给出证明，后续只需压缩、校正符号或补充细节。
2. **外部输入**：正文可使用，但必须在附录 D 中定位来源、定理编号、版本和模型假设。
3. **边界说明**：只用于解释范围、失败模式或研究方向，不得作为证明步骤。

## 0. 使用规则

若正文需要调用某个结论，应按下列顺序检查：

1. 该结论是否在本文件中列为内部证明；
2. 若不是，是否在附录 D 中列为外部输入；
3. 若只出现在边界说明中，则不得用于推出后续命题；
4. 若结论涉及模型比较，必须同时检查附录 G、M、O；
5. 若结论涉及几何应用，必须同时检查附录 N。

## 1. 基础 operad 部分

| 范围 | 内部证明 | 外部输入 | 边界说明 |
| --- | --- | --- | --- |
| 附录 A | $\mathbf B_{\mathcal U}\simeq\coprod B\Sigma_n$；左右作用转换；特征 $0$ 下 invariants/coinvariants 比较 | Maschke 型事实作为基础代数输入 | 一般底环 coinvariants 不 exact |
| 附录 B | 全体有限集映射的两层拉平；代入乘积结合律与 coherence；含 arity $0$ 的单位律；arity coinvariants 公式 | 无大型外部输入 | 非空分块公式只在内层 nullary 项为初对象时成立；反例 B.10.1 |
| 第一章 | 左作用数据重建有限集函子；全映射代入乘积；operad 作为幺半对象；End、Ass、Com 的 nullary 检查 | 无大型外部输入 | 集合值结果不能自动推广到链复形；空纤维不可无条件删除 |
| 第二章 | 自由代数 coend 公式；空纤维兼容的无坐标自由代数动作；自由-遗忘伴随；monad 识别；finitary 性 | 基础 coend/filtered colimit 事实 | $T_s=\varnothing$ 仍占据外层槽，不能形成所谓“空块” |
| 第三章 | 非对称偏复合公理；树收缩顺序无关 | 无大型外部输入 | 平面树不等于 Moerdijk-Weiss 树 |
| 第四章 | 自由非对称 operad 的树递归；全映射树 grafting；自由对称 operad 的叶标号树公式；全映射 operadic congruence | 无大型外部输入 | 零叶子树实现 nullary substitution；自同构群商必须记录，附录 H 给出 $\mathcal U$-小树群胚骨架上的 colimit 口径 |
| 第五章 | colored symmetric sequence；colored substitution；endomorphism colored operad；multicategory 对照 | enriched/admissible colored 版本 | enriched 版本不能只换 Hom 集合 |
| 第七章 | PROP 定义等价；End PROP；从 operad 到 PROP；双代数 PROP；PROP 给 properad | properad 到自由 PROP；完整 directed graph groupoid 构造 | wheeled trace 需有限性/dualizability |
| 附录 H | planar、leaf-labelled、Moerdijk-Weiss 树对照；自由 operad 的 $\mathcal U$-小树群胚 colimit 与自由性 | dendroidal nerve fully faithful 已定位到 MW-2 | 单变量 $\operatorname{Dec}_E$ 取 colimit，不使用缺少反变因子的伪 coend；$\Omega(T)$ 不是单色自由 operad 的 arity 值 |
| 附录 K | colored tree 公式；自由 colored operad 泛性质；模/双模编码 | enriched colored operad admissibility | enriched 模型结构需另行检查 |
| 附录 U | PROP interchange、双代数兼容、properad 连通图复合、wheeled trace 和图替换结合律 | 自由 properad/PROP、Frobenius PROP 完整构造、wheeled graph complex | properad 不含任意水平张量 |
| 附录 P | arity $0,1,2$ 代入计算；End 结合律；Ass/Com/Lie 低阶检查 | Lie 组合模型深层结果不在本附录证明 | 特征 $2$ Lie 约定需分开 |

## 2. 线性、Koszul 与同伦代数

| 范围 | 内部证明 | 外部输入 | 边界说明 |
| --- | --- | --- | --- |
| 第六章 | $R$-模值 Schur functor；Ass/Com/Lie/Pois 定义；End 线性复合 | Lie/Pois 经典识别细节 | 一般底环反对称风险 |
| 附录 F | Ass、Com、End、Lie、Poisson 的逐项验算；自由 Ass/Com 代数 | PBW、自由 Lie 模型、little cubes 同调识别 | Poisson 与 $E_n$ 同调需系数条件 |
| 第八章 | reduced 二次数据；自由 operad 代入权重 $r+\sum_t s_t$；二次对偶定义；Koszul 性定义 | Ass/Com/Lie Koszul 性已定位到 GK-5；对偶 Koszul 性和生成函数已定位到 GK-2 | 同权内层因子给 $r+ns$，$r+s$ 只适用于一次非单位插入；含单位与 reduced 口径不可混用 |
| 第九章 | dg-operad/cooperad；$M\circ_{(1)}N=(M\circ(I_k\oplus N))_{[1]}$；由 coaugmentation 分裂构造 $\Delta_{(1)}$；typed convolution/Maurer--Cartan；bar-cobar 泛性质；直和 cofree 口径下 conilpotence | Quadratic Koszul criterion 精确定位到 LV-2；GK-3/GK-7 与 FRE-1--FRE-4 分别作 classical/model-category 交叉包 | conilpotence、完成化选择与 filtration 不可省略 |
| 附录 I | derivation/coderivation 泛性质；zero-extension 口径下的 convolution pre-Lie；Maurer--Cartan 与 twisted differential；bar 递增与 cobar 递减权重滤过及逐 arity 有界性 | Connected weight-graded 四项等价定位到 LV-1，quadratic specialization 定位到 LV-2 | unary 顶点可破坏逐 arity 有界性；反例 I.22.1 区分直和与完成化 |
| 附录 Q | 二元二次 operad 权重-arity 关系；$\operatorname{Ass}_{ns}$ rewriting 的终止性与唯一临界对合流；Koszul twisting 低权重行为；bar/cobar differential 低权重公式；bar-cobar counit 的生成元公式与权重 $2$ 抵消 | 合流二次 rewriting 推出 $\operatorname{Ass}_{ns}$ Koszul 性已定位到 LV-3；bar-cobar counit entry 已定位到 FRE-4；algebra cofibrant replacement 已定位到 FRE-5；谱序列收敛 | bar 权重 $>1$ 的单个 cobar 生成元映 $0$；低阶链映射检查不推出 exactness；LV-3 只覆盖 reduced nonsymmetric quadratic 版本 |
| 第十章 | $A_\infty/L_\infty/C_\infty$ 的 bar-cobar 定义；$E_n$ 基本边界 | May recognition；$H_\*(E_n)$；形式性 | $E_n$ 与 infinity-operad 不同 |
| 附录 L | suspended coderivation 定义；$\mathcal P_\infty$ 记号边界；additivity/rectification 边界 | Poisson 同调、形式性、Dunn additivity 已定位到 DUNN-1 | 任意 cofibrant replacement 不叫 $\mathcal P_\infty$ |
| 第十一章 | Gerstenhaber/BV 定义和基本结构说明 | Deligne 猜想已定位到 MS-1--MS-3 与 BF-1--BF-4；framed $E_2$ 同调为 BV | BV 符号需与模型匹配 |
| 第十二章 | Hochschild differential、cup、insertion、brace 基本公式 | brace/surjection operad 与 $E_2$ 链模型 locator 已定位到 MS-1--MS-3 与 BF-1--BF-3 | dg signs 需用 suspended 算法 |
| 第十三章 | contraction、minimal model、formality 定义；低阶转移说明 | HPL；Markl transfer existence 已定位到 MHT-1--MHT-8；minimal model 唯一性 | Massey product 说明不可当证明 |
| 附录 J | normalized contraction；$A_\infty$ 平面树递归；低阶恒等式检查 | Markl transfer existence 已定位；完整高阶 signs 由附录 W convention package 管理 | $L_\infty$ shuffle signs 需模型固定 |
| 附录 S | $m_2^H$、$m_3^H$、Massey 边界、$\ell_2^H/\ell_3^H$ 低阶形状、$A_\infty$-formality 低阶判据 | Markl transfer existence 已定位；minimal model 唯一性；strict dg formality rectification | $m_3$ 不等于无选择 Massey product |
| 附录 W | 同调分次、operadic suspension、suspended Hochschild cochains、brace signs、$A_\infty/L_\infty$ 度数和转移符号检查表 | 文献 convention 的逐条转换需最终核对 | unsuspended 全公式不是主定义 |
| 附录 P | suspended $A_\infty$ 低阶关系；Hochschild bracket 低阶计算 | 与 $E_2$ 模型比较 | 一般底环下 $[m,m]=0$ 需谨慎 |

## 3. 模型范畴、dendroidal 与 infinity-operad

| 范围 | 内部证明 | 外部输入 | 边界说明 |
| --- | --- | --- | --- |
| 附录 C | WFS retract 封闭；replacement；Quillen adjunction 条件等价；导出伴随 | homotopy category mapping 计算 | left/right derived functor 需 replacement |
| 第十四章 | 对称幺半模型范畴定义；对称序列作为 $\prod_n\mathcal M^{\Sigma_n}$ 的积模型；固定 operad 的 admissibility；底层 cofibrancy 的正确判据；rectification 语言 | BM-1 的 interval package、PSAR-2 的 admissibility package、PSAR-4 的 map-specific rectification package | 各 $\mathcal M^{\Sigma_n}$ 的 projective 结构仍是输入；不得把 operad 模型结构、固定 operad 代数转移和 rectification 合并；完整 $W$-construction 仅为外部边界 14.31 |
| 附录 G | T0--T3 的逐项假设包；admissibility/rectification 检查表；正特征对称幂内部反例 | Pavlov--Scholbach PSAR-1--PSAR-6、PSP-1--PSP-2 与 Berger--Moerdijk schemas | checklist 不是定理；正特征下反例 G.16/X.15--X.16 阻止无假设转移 |
| 附录 R | $\mathbf{sSet}$、Top、$\mathbf{Ch}_k$、一般 $\mathbf{Ch}_R$、spectra、colored/enriched 案例判定；rectification 正反例格式 | 逐底范畴 transferred/admissibility/rectification 定理已定位到 BM/HIN/FRE/PSAR；需假设翻译 | 案例不是全称定理，不能跨底范畴迁移 |
| 第十五章 | 全映射 simplicial/topological operad 定义；Sing/realization 的 operad-level functor 构造；well-pointed/$\Sigma$-free 条件 | Kan--Quillen；底范畴 Top--sSet Quillen equivalence；BM-1 的两类 operad model structures | Operad-level realization--Sing Quillen equivalence 仅为边界 15.18；chains on spaces 需 monoidal 相干性 |
| 第十六章 | 固定 $\mathcal U$-小树骨架 $\Omega$；dendroidal set、nerve、Segal core、horns 定义；strict Segal 性 | dendroidal nerve fully faithful 已定位到 MW-2；$\Delta\hookrightarrow\Omega$ fully faithful 已定位到 MW-1 | presheaf 与 category-of-elements colimit 的 universe 层级已固定；树方向 convention 需保持 |
| 第十七章 | inner horn fillers 的类型检查；strict nerve 唯一 filler；线性限制为 quasi-category | MW-4 的 homotopy coherent nerve inner Kan；CM-3 的 operadic model structure；HHM 比较仅限 open/no-constants 模型 | inner Kan 存在填充不等于 strict；Segal core map 不等于单个完整 inner horn |
| 附录 T | 两顶点 horn、Segal core、boundary/horn 差异、corolla automorphism、degeneracy 样例；$L_1\to\eta$ 与 $X_\eta\to X_{L_1}$ 方差检查 | inner horn inclusion normality；strict Segal 本质像；operadic model structure | 低阶 horn 样例不等于模型结构证明 |
| 第十八章 | $\mathbf{Fin}_*$ active/inert；Lurie-style infinity-operad；algebras over infinity-operads | coCartesian fibration 技术；category-of-operators 已定位到 HA-OP-1--HA-OP-3；dendroidal-Lurie comparison 已定位到 HHM-1--HHM-5 | active/inert 方向不可混用 |
| 第十九章 | relative functor induces localization functor；大模型范畴在更大 universe 中 localization；derived tensor replacement；rectification 后 localization 等价说明 | DK localization = DKR-1--DKR-7；ordinary straightening = HTT-1；underlying monoidal infinity-category = HA-MON-1--HA-MON-2；spaces-valued operadic straightening = PRA-4；strict-to-infinity = PSAR/HA-ALG | 只有本质小或给定 small DK presentation 时才能小化；Bousfield localization preservation 与原模型范畴的 monoidal localization 是不同问题；先代数后 localization 不自动交换 |
| 附录 M | 四类模型比较图；允许路径、禁止捷径和 arity $0$ 检查 | Moerdijk-Weiss strict nerve core；White/White--Yau localization preservation；HA-OP category-of-operators；HHM 的 open/no-constants zig-zag；PSAR-5 的 strict-to-infinity algebra comparison | HHM 不能覆盖本书默认 constants；模型比较不是定义同一性 |
| 附录 Y | strict operad 的树指标线性化；Segal-linear 接口；ordinary algebra 的 dendroidal 特化；Koszul extension strict specialization test | Hoffbeck-Moerdijk infinity-operadic homology/Koszul 结果保持研究边界；若升级为定理输入需新增 locator | linear infinity-operad 结果不得替代 classical Koszul theorem |
| 附录 O | 模型混用失败模式 | 无新增定理 | 仅作边界检查 |
| 附录 P | 最小 dendroidal inner horn 解释 | dendroidal nerve/model structure 背景 | 唯一填充与存在填充分离 |

## 4. 几何、factorization 与前沿

| 范围 | 内部证明 | 外部输入 | 边界说明 |
| --- | --- | --- | --- |
| 第二十章 | Framed Disk category、prefactorization/factorization algebra、factorization homology 定义；Fukaya category 结构性描述 | AF-1 excision、AF-2 圆周、DUNN-1 additivity 可调用；20.6/20.18/20.22/20.23 均降为外部或研究边界 | Locally constant comparison 和 Fukaya 构造/gluing 在无精确模型 locator 时不进入证明链 |
| 附录 N | framed/tangential setup；不交并公式；slice-final 的 disk 归一化；球面的 open-collar excision；检查表 | AF-1 excision；AF-2 圆周 Hochschild；AF-4 boundary；AF-5 commutative coefficients | 闭半球不是 $\operatorname{Disk}_n^{fr}$ 对象；N.30 是 Fukaya 研究边界而非可调用定理 |
| 附录 V | 半空间、区间、圆周 trace、分层区间、hypersurface defect、corners 和 Fukaya skeleta 边界样例 | AF-4 只作 boundary-manifold 入口；V.3/V.6/V.8 为外部边界，V.11 为研究边界 | 无边界 disk 归一化不能用于边界/分层空间；未定位几何模式不可作为定理 |
| 附录 O | factorization/Fukaya 失败模式 | 无新增定理 | 普通同调与 factorization homology 分离 |
| 附录 P | cyclic bar levels；derived relative tensor product 括号说明 | $\int_{S^1}A\simeq HH_\*(A)$ 比较 | bar 模型不替代几何 excision 证明 |
| 附录 X | arity $0$、coinvariants、特征 $2$ Lie、$\operatorname{Sym}^p$ 不保持 acyclic complex、正特征 rectification、$HH_\*(k)$、$HH_0(M_n(k))$、区间 module 条件例子 | Morita invariance、boundary factorization、rectification 深层定理 | 小例子不能替代一般定理 |
| 附录 Z | operadic category 数据包低阶检查；BV interchange 低阶形式；relative dendroidal object；Fukaya $A_\infty$ 条件性边界证明 | operadic category nerve、relative dendroidal Rezk nerve、Fukaya 高阶结构仍需外部定理 | 三类前沿均不得直接定理化 |
| 第二十一章 | 研究边界分类；版本表；模型差异表；进入正文验证流程 | 进入正文前需逐条核查 | 2025-2026 预印本不得直接定理化 |
| 前沿审计 | 2026-06-30 arXiv 条目版本、用途和进入正文前检查清单 | 当前不作为证明输入；只有升级为正文定理时才新增 locator | 只固定研究边界 |

## 5. 元文档

| 文件 | 状态 | 审校用途 |
| --- | --- | --- |
| `README.md` | 当前目录入口 | 检查章节是否可达 |
| `SKILL.md` | 写作约束 | 约束后续扩写风格 |
| `NOTATION.md` | 符号入口 | 防止同一符号多义 |
| `SOURCES.md` | 资料源入口 | 记录主要来源和边界 |
| `INTERNAL_OPERAD_CLOSURE_AUDIT.md` | 内部闭合审计 | 检查 operad theory 主体定义链、类型链和公理链 |
| `INTERNAL_NUMBERING_AND_CROSSREF_AUDIT.md` | 编号与交叉引用审计 | 检查第一至第七章编号是否可稳定引用 |
| `LABEL_LEDGER_CH01_07.md` | 稳定 label 表 | 登记第一至第七章交叉引用目标 |
| `LABEL_LEDGER_CORE_APPENDICES.md` | 核心附录 label 表 | 登记附录 A/B/H/K/P/U/X 交叉引用目标 |
| `LABEL_LEDGER_CH08_21.md` | 高级章节 label 表 | 登记第八至第二十一章的 416 个交叉引用目标 |
| `LABEL_LEDGER_REMAINING_APPENDICES.md` | 剩余附录 label 表 | 登记附录 C/D/E/F/G/I/J/L/M/N/O/Q/R/S/T/V/W/Y/Z 的 398 个交叉引用目标 |
| `CROSSREF_REWRITE_AUDIT.md` | 散文引用替换审计 | 记录两轮散文编号替换，覆盖主体章节、高级章节、主要附录和元文档 |
| `D_source_theorem_index.md` | 外部输入索引与引用包 | 外部定理定位主表 |
| `MATH_REVIEW.md` | 审查记录 | 当前风险和已完成检查 |
| `DEPENDENCY_GRAPH.md` | 依赖图 | 防止倒用高级定理 |
| `PUBLICATION_CLOSURE_MATRIX.md` | 完本闭包矩阵 | 区分基本完本和最终出版 |
| `PUBLICATION_PROOFING_LEDGER.md` | 出版校对账本 | 记录出版校对动作、locator 状态和局部指称判定 |
| `REFERENCE_LOCATOR_LEDGER.md` | 引用定位账本 | 最终出版前 P0/P1 定位入口 |
| `FINAL_OPERAD_THEORY_CLOSURE.md` | 最终数学收口账本 | 判定剩余项为 locator 已闭合、边界关闭或 production work |
| `P0_REFERENCE_LOCATORS_BATCH_9.md` | P0 locator | 记录 Pavlov--Scholbach admissibility/rectification、symmetric powers，以及 Lurie strict-to-infinity 与 underlying symmetric monoidal infinity-category comparison |
| `P0_REFERENCE_LOCATORS_BATCH_10.md` | P0 locator | 记录 Hinich Dwyer--Kan localization、HHM dendroidal-Lurie comparison、Lurie category of operators 和 Pratali operadic straightening |
| `P1_REFERENCE_LOCATORS_FINAL_SWEEP.md` | P1 locator | 记录 Dunn additivity、Deligne conjecture 和剩余拓扑/几何边界 locator |
| `SECOND_PASS_STRICTIFICATION_PLAN.md` | 严格化路线 | 判定未完成任务 |
| `THEOREM_LEDGER.md` | 本文件 | 区分内部证明、外部输入和边界说明 |

## 6. 当前缺口

本账本显示，当前草稿已经达到基本完本严格草稿态和 operad theory 数学收口态，但尚未达到 camera-ready 出版态，主要 production 剩余项是：

1. 内部 operad theory 主体已完成定义链和类型链审计；正文二十一章与附录 A--Z 的稳定 label 表已闭合，两轮散文交叉引用替换已覆盖可直接定位的跨章/跨附录指称，剩余为局部公式指称与最终出版校对。
2. 附录 D 中主要 P0/P1 theorem locator 已收口；已定位批次覆盖 Berger--Moerdijk、Cisinski--Moerdijk、Lurie HTT、Ayala--Francis、Ginzburg--Kapranov classical Koszul core、Loday--Vallette LV-1--LV-3、Fresse modern cobar/cofibrant replacement、Hinich dg-operad model context、Markl homotopy transfer existence、Moerdijk--Weiss dendroidal nerve core、White/White--Yau localization preservation、Pavlov--Scholbach admissibility/rectification、Hinich Dwyer--Kan localization、HHM dendroidal-Lurie comparison、Lurie category-of-operators/algebra/monoidal-model comparison、Pratali operadic straightening locator、Deligne locator 和 Dunn additivity locator。剩余符号翻译、假设表和 bibliography 属于 production/copy-editing。
3. 若干内部证明仍需从“证明草稿”压缩为最终教材格式。
4. $A_\infty/L_\infty$、brace 和 operadic suspension 已有交叉核对表，但完整 unsuspended 展开仍需最终锁定。
5. 模型范畴中的 transferred structure 与 rectification 已有案例层和主 locator；按底范畴逐例翻译文献假设属于 production-level hypothesis table。
6. factorization homology 的 topological manifold excision、圆周 Hochschild 计算、带边界基础版本和 Dunn additivity 已完成定位；stratified/Fukaya 版本已关闭为外部几何边界。
7. Fukaya category 相关内容只能保持为接口，除非加入具体几何模型和外部分析定理。
8. 2026 研究边界条目尚未进入正文证明链。

因此，本书当前状态是“operad theory 数学收口版”，不是 camera-ready 出版版本。
