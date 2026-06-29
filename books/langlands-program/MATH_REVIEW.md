# 数学审查记录

本文档记录《Langlands 纲领》草稿的审查清单、逐章收口状态和后续补证任务。

## 全书审查清单

- [ ] 每章是否列出本章目标和依赖前置知识。
- [ ] 每个新定义是否包含完整数据和公理。
- [ ] 每个非平凡命题是否给出证明、证明草图或“外部输入定理”标记。
- [ ] 每个外部输入定理是否能在 `SOURCES.md` 中追溯。
- [ ] 是否避免把 Langlands 对应写成无条件的一一对应；是否说明已知情形、猜想情形和归一化。
- [ ] 是否区分局部对象和整体对象。
- [ ] 是否区分 classical modular forms、adelic automorphic forms 和 automorphic representations。
- [ ] 是否区分 complex representations、l-adic representations、mod p representations。
- [ ] 是否说明 Frobenius 归一化。
- [ ] 是否说明 Haar 测度和 Fourier 变换归一化。
- [ ] 跨章节比较参数、L 因子、Hecke 本征值或 Galois 表示时，是否回指 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md)。
- [ ] 每个外部输入是否已按“核心结构、支撑接口、卫星理论”分级；是否避免把非主线外部理论无节制并入本书。

## 外部输入分级审查规则

本书不以重写所有相关数学理论为目标。外部输入按以下三类处理：

1. 核心结构：直接定义或计算 Langlands 主线对象，例如 adeles、Weil-Deligne 参数、Satake 参数、Hecke 作用、L 群、局部 L 因子、导子、packet、几何 Satake。此类可以在本书内展开到可计算层。
2. 支撑接口：证明很深但本书需要其结论，例如完整类域论、Tate thesis 解析延拓、Neron 模型存在性、Ribet 降层、模性提升、Arthur trace formula、Fargues-Scholze。此类写精确定理、假设、归一化、使用位置和来源，不追求全证明。
3. 卫星理论：本身可构成独立课程或专著，例如完整代数几何、完整 p-adic Hodge theory、完整 Bruhat-Tits 理论、完整 D-module/derived stack 六运算构造。此类只给接口，并在后续计划中标为另卷。

新增材料前必须回答：该材料是否定义 Langlands 对象，是否影响参数/L 因子/导子/Hecke 本征值计算，是否是应用章逻辑闭环必需。若不是，则不在本书主体展开。

## 收口判定

本次审查采用 [CLOSURE_STATUS.md](CLOSURE_STATUS.md) 的标准。

- 当前结论：本书已经进入主线基本收口候选状态；尚未是最终审定版。
- 后续原则：不再新增同级主线，不再把外部理论扩写成完整背景专著。
- 必需方向：逐章收口台账维护、外部输入来源维护、阅读路径和习题回指维护。
- 可后置方向：class formation 完整证明、Taylor-Wiles patching、Arthur trace formula 完整证明、D-module/derived stack 六运算、perfectoid/diamond/Fargues-Fontaine 完整技术层。

## 第一轮收口记录

- 已在第 1-22 章和第 90 章中对涉及 convention 的高风险章节加入 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 回指。
- 已新增 [MAINLINE_PROOF_CHAINS.md](MAINLINE_PROOF_CHAINS.md)，把 `GL(1)`、费马应用、一般算术 Langlands 和几何 Langlands 四条路径改写为最短证明链。
- 已新增 [EXERCISE_COVERAGE.md](EXERCISE_COVERAGE.md)，记录四条主线的已解习题覆盖和基本收口前最小新增题目。
- 已补入 `3.5`、`7.6` 两道收口题，并补齐 `15.3`、`20.4` 的解答；习题层面的四条主线基本闭合。

## 第二轮收口记录

- 已完成 `THEOREM_INDEX.md`、`E_external_input_theorem_index.md` 和 [MAINLINE_PROOF_CHAINS.md](MAINLINE_PROOF_CHAINS.md) 的状态一致性审计；结果记录于 [INDEX_CONSISTENCY_AUDIT.md](INDEX_CONSISTENCY_AUDIT.md)。
- 已把 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 回指补入高风险附录 A、B、C、D、F、G、H、I、J、K、L、M、N、O、P、Q、R、S、T、U、V、W、X、Y、Z、AA、AB、AC、AD、AE。
- 当前达到主线基本收口候选状态；下一步应做交叉引用审稿和编号一致性检查。

## 逐章收口缺口

逐章风险已经改写为 [CHAPTER_CLOSURE_AUDIT.md](CHAPTER_CLOSURE_AUDIT.md)。本轮判定如下。

| 类型 | 状态 | 后续动作 |
|---|---|---|
| 新增同级主线 | 无需新增 | 继续禁止横向扩张为第五条主线 |
| 应用链阻断项 | 未发现 | 费马应用继续只使用外部输入定理和本书已证引理 |
| 猜想误用风险 | 未发现阻断项 | 维护 `P/S/E/C` 状态标记 |
| 归一化缺口 | 高风险层已闭合 | 新增或改写段落必须回指 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) |
| 外部输入来源 | 重点条目已拆细 | 后续维护 [E_external_input_theorem_index.md](E_external_input_theorem_index.md)，新增外部输入必须先登记 |
| 编号和交叉引用 | 仍需审稿 | 下一轮检查正文、附录、定理索引和习题答案回指 |

## 第三轮收口记录

- 已新增 [CHAPTER_CLOSURE_AUDIT.md](CHAPTER_CLOSURE_AUDIT.md)，把原“当前风险”逐条改写为正文章节台账、附录层台账、非阻断后置项和下一轮优先级。
- 已确认完整 Tate thesis、class formation、Taylor-Wiles、Arthur trace formula、D-module/IndCoh、perfectoid/diamond 等理论不再作为本书基本收口阻断项。
- 第三轮后，收口任务从“风险补写”压缩为外部输入来源处理、交叉引用审稿和编号一致性检查。

## 第四轮收口记录

- 已精校 [E_external_input_theorem_index.md](E_external_input_theorem_index.md)，把 Frey 曲线局部性质、Satake、Arthur、几何 Satake 和 Fargues-Scholze 拆成更具体的外部输入条目。
- 已在 [SOURCES.md](SOURCES.md) 中补入 Frey-Hellegouarch 曲线来源入口。
- 当前剩余工作进一步收缩为编号一致性检查、交叉引用审稿和局部措辞精修。

## 本轮严格性审查记录

- 已修正第二章中全局类域论接口：有限像 Galois 表示只对应有限阶 Hecke 特征；一般非有限阶 Hecke quasi-character 应放在 Weil 侧。
- 已修正第三章小结，明确局部一般连续特征对应 Weil 参数，而不是普通 profinite Galois 表示。
- 已把第四章的基础群论口径从含混的 tdlc 表述收紧为 locally profinite group，并注明 Van Dantzig 定理的角色。
- 已把第五章 `GL(n)` 局部 Langlands 陈述改为“同构类之间的双射”。
- 已在第六章 Deligne 接口中显式声明 Frobenius 归一化风险。
- 已在第七章区分算术归一化和 unitary automorphic normalization，避免把 $L(f,s)$ 与标准自守 L 函数的不同 convention 混用。
- 已把第六、七、八章中模形式和椭圆曲线 Galois 表示的 Frobenius 写法改为算术 Frobenius，并明确与局部 Langlands 几何 Frobenius convention 的换算风险。
- 已在第九章区分 residual modularity、lift modularity、elliptic curve modularity 和 general Fontaine-Mazur expectation，避免把模性提升定理写成无条件黑箱。
- 已在第十章把 Ribet 降层拆为局部-整体相容、残余导子和 level 删除三个层次，避免把“降层”写成单步口号。
- 已补写第十一章，把一般 Langlands 的结构入口从 `GL(n)` 扩展到 connected reductive groups、root datum、dual group、L group、L homomorphism 和 unramified Satake parameter。
- 已在第十一章区分 Galois 型 L 群、局部 Weil 型 L 群和数域情形中仍属纲领性的全局 Langlands 群，避免把全局 L 群写成无条件已构造对象。
- 已补写第十二章，把局部 Langlands 从“参数对应表示”的粗略说法收紧为 coarse packet、enhanced parameter、component group、inner form 和 endoscopic compatibility 的分层表述。
- 已在第十二章区分定理性已知情形、一般猜想和后续章节只作为接口使用的外部输入，避免把一般 reductive 群 LLC 写成已完全证明的一一对应。
- 已补写第十三章，把全局自守表示、尖点条件、restricted tensor product、非分歧 Satake 参数、部分 Euler 乘积和完全 L 函数接成统一接口。
- 已在第十三章区分形式 Euler 乘积、局部因子定义、已知解析定理和一般 Langlands 解析猜想，避免把任意 $r:{}^LG\to\operatorname{GL}(V)$ 的解析延拓写成无条件结论。
- 已补写第十四章，把 `GL(n)` 的局部 LLC、Langlands 分类、全局标准和 Rankin-Selberg L 函数、强重数一、converse theorem、函数域 Lafforgue 定理和数域 Galois 表示接口集中整理。
- 已在第十四章区分函数域 `GL(n)` 全局定理、数域 regular algebraic 已知构造和数域完整全局 Langlands 猜想，避免把数域一般情形写成既成定理。
- 已补写第十五章，把函子性分为弱转移、强转移、L 函数相容性、`GL(N)` 目标 converse theorem、base change、automorphic induction、低阶 functorial lifts、endoscopy 和 Galois 表示侧复合。
- 已在第十五章区分一般 functoriality 猜想与已知特殊情形，避免把 symmetric powers、exterior powers、tensor products 或 endoscopic transfer 写成全体已知定理。
- 已补写第十六章，建立 trace formula、稳定轨道积分、endoscopic data、transfer factor、fundamental lemma 和 twisted trace formula 的接口。
- 已补写第十七章，建立 Arthur 参数、A-packet、multiplicity formula、standard transfer 和非 tempered 离散谱的接口。
- 已补写第十八至二十二章，建立几何 Langlands 主线：$\operatorname{Bun}_G$、Hecke 修改、affine Grassmannian、几何 Satake、Hecke eigensheaves、谱侧范畴和函数域桥梁。
- 已补写附录 A-E，集中记录代数数论、Haar 测度、光滑表示、模曲线维数公式和外部输入定理索引。
- 已新增全书定理索引 [THEOREM_INDEX.md](THEOREM_INDEX.md)，把主要结果标记为已证、证明草图、外部输入或猜想。
- 已新增章节依赖图 [DEPENDENCY_GRAPH.md](DEPENDENCY_GRAPH.md)，明确 `GL(1)`、费马应用、一般 Langlands 和几何 Langlands 四条阅读路径。
- 已新增核心习题解答 [SOLUTIONS.md](SOLUTIONS.md)，覆盖 restricted product、局部特征、Frobenius 归一化、Hecke 关系、Satake 参数、函子性和费马应用链。
- 已扩充附录 A：加入非 Archimedean 赋值、数域乘积公式证明、ray class groups、idele class group 商描述、norm subgroup theorem 口径和 Artin 导子基本性质。
- 已扩充附录 B：加入卷积结合律、开紧平均投影、商测度和 restricted product 积分公式。
- 已扩充附录 C：加入 Schur 引理、中心特征、smooth dual、有限长度和可容许性稳定性。
- 已扩充附录 D：加入 $X_0(2)$ 的指数、cusp 数、椭圆点数、genus 和权 $2$ 微分形式的计算。
- 已新增附录 F：固定局部紧 Abel 群 Fourier 分析、self-dual measure、Schwartz-Bruhat 空间、adeles 自对偶和 Poisson summation 在 Tate thesis 中的用法。
- 已新增附录 G：补 `GL(n)`、`SL(n)`、`PGL(n)`、classical dual groups、split L 群、restriction of scalars torus 和 determinant/symmetric square L 同态计算。
- 已新增附录 H：补 Hecke 双陪集算子、$\Gamma_0(N)$ 好/坏素数代表、Fourier 系数公式、Petersson 内积和 adelic Hecke algebra 比较。
- 已新增附录 I：补 Godement-Jacquet、Rankin-Selberg、Whittaker 模型、全局 unfolding、converse theorem 和函子性检测的积分接口。
- 已新增附录 J：补 degeneracy maps、old/new 分解、Atkin-Lehner involutions、Casselman newvector theorem、局部导子和费马应用中的级 $2$ 矛盾。
- 已新增附录 K：补 Galois deformation functor、局部变形条件、Selmer 群、Hecke algebra、$R=T$、Taylor-Wiles patching 和模性提升逻辑。
- 已新增附录 L：补 Eisenstein series、constant term formula、intertwining operators、continuous spectrum、residual spectrum、Arthur 参数和 trace formula 谱侧接口。
- 已新增附录 M：补 Langlands-Shahidi local coefficient、局部 $\gamma$ 因子、局部 L 因子、全局 Eisenstein 函数方程和函子性解析接口。
- 已新增附录 N：补 tori 的 LLC、$\operatorname{SL}_2$ packet 现象、Jacquet-Langlands 内形式、endoscopic datum、stable character 和 fundamental lemma 接口。
- 已新增附录 O：补 D-modules、six functors、kernel formalism、QCoh/IndCoh、singular support 和 categorical geometric Langlands 的谱侧技术口径。
- 已新增附录 P：补球 Hecke 代数、Cartan 分解、Satake 变换、spherical representations、`GL(n)` 非分歧 L 因子和几何 Satake 的函数迹接口。
- 已新增附录 Q：补 Bernstein-Zelevinsky segments、multisegments、Langlands quotient theorem、tempered/generic classification 和 `GL(n)` 局部因子相容接口。
- 已新增附录 R：补紧商 trace formula 核公式、Arthur truncation、weighted orbital integrals、谱展开、invariant trace formula、稳定化和 base change/endoscopic classification 应用接口。
- 已新增附录 S：补函数域双商、Hecke correspondences、shtukas、Drinfeld/Laurent Lafforgue 定理、V. Lafforgue excursion operators 和 sheaf-function 桥梁。
- 已新增附录 T：补模曲线 local systems、Hecke correspondences、Eichler-Shimura、Deligne 表示、weight two 椭圆曲线相容和 residual representations。
- 已新增附录 U：补 regular algebraic automorphic Galois representations、Shimura varieties cohomology、unitary group realization、p-adic Hodge comparison、局部-整体相容和 automorphy lifting 接口。
- 已新增附录 V：补 class formations、局部/全局 Artin reciprocity、norm subgroup theorem、ray class fields、conductors 和 `GL(1)` Langlands 的 character 形式。
- 已新增附录 W：补模曲线代数化、权二 cusp forms 与微分、$X_0(2)$ genus、Hecke correspondences、old/new decomposition、Atkin-Lehner signs 和费马应用中的级 $2$ 矛盾。
- 已新增附录 X：补 Arthur classification 的 classical groups 范围、self-duality sign、local/global packets、multiplicity formula、standard transfer、inner form refinements 和 beyond endoscopy 接口。
- 已新增附录 Y：补 Ran space、factorization objects、Beilinson-Drinfeld Grassmannian、fusion、几何 Satake、Hecke action 和 categorical geometric Langlands 的技术层。
- 已新增附录 Z：补局部调和分析、Harish-Chandra temperedness、characters、Plancherel、Bernstein decomposition、local Paley-Wiener 和 local character expansion 接口。
- 已新增附录 AA：补 Bruhat-Tits buildings、parahoric group schemes、hyperspecial subgroups、Cartan/Iwahori 分解、Moy-Prasad filtrations 和非分歧 LLC 的结构接口。
- 已新增附录 AB：补 derived stacks、cotangent complex、QCoh/IndCoh comparison、singular support、six functors、kernel formalism、renormalized D-modules 和 spectral action 接口。
- 已新增附录 AC：补 perfectoid spaces、diamonds、Fargues-Fontaine curve、$G$-bundles on FF curve、local Shimura varieties、local shtuka cohomology 和 Fargues-Scholze 几何局部 Langlands。
- 已扩充附录 F：补对偶测度缩放、有限 Abel 群 Fourier 反演、非 Archimedean 紧开陪集 Fourier 公式、$\mathbb A_\mathbb Q/\mathbb Q$ 基本域证明、idele 缩放 Poisson、经典 Poisson 的 adele 推导和 Tate theta 恒等式。
- 已扩充附录 A：补有限扩张中的 ramification index、residue degree、分解群-局部 Galois 群同构、惯性精确列、非分歧 Frobenius、lower ramification groups 和 Herbrand upper numbering 接口。
- 已新增附录 AD：补 Neron model、Kodaira-Neron classification、Tate curve、Ogg conductor formula、Tate algorithm 输出、坏约化局部 L 因子和 Frey 曲线 residual conductor 降到级 $2$ 的局部计算接口。
- 已新增附录 AE：补 `GL(2)` principal series、Steinberg twists、supercuspidals、admissible pairs、Weil-Deligne 参数、局部 L 因子、conductor 和椭圆曲线局部表示类型接口。

## 收口阶段补证计划

以下计划取代原先的横向扩写清单。每一项都必须服务 Langlands 主线闭合。

### A. 基本收口必需

1. 维护 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md)：Frobenius、Artin reciprocity、Satake、Haar、Fourier、L 函数变量、Tate twist、classical normalization 与 unitary normalization 的跨章比较必须回指该表。
2. 维护 `THEOREM_INDEX.md` 与 `E_external_input_theorem_index.md`：每个外部输入必须有来源、使用章节、状态和分级。
3. 维护 [MAINLINE_PROOF_CHAINS.md](MAINLINE_PROOF_CHAINS.md)：每一步标注已证、证明草图、外部输入或猜想。
4. 维持 [EXERCISE_COVERAGE.md](EXERCISE_COVERAGE.md) 的覆盖状态；新增习题必须服务四条主线闭环。
5. 维护 [CHAPTER_CLOSURE_AUDIT.md](CHAPTER_CLOSURE_AUDIT.md)：只保留会阻断阅读路径、外部输入状态或归一化一致性的缺口。

### B. 主线小补

1. 第三章精校 ray class characters、idele class characters 的 conductor 和 Dirichlet 特征比较；不证明完整 class formation。
2. 第七章精校 classical-to-adelic 比较的检查表和 Hecke 算子对应；strong approximation 仍作为外部输入。
3. 第八至十章精校 Frey 曲线导子、Ribet 降层假设和级 $2$ 矛盾的交叉引用；Taylor-Wiles 和 Ribet 证明仍作为外部输入。
4. 第十一至十四章精校 rank-one Satake、`GL(2)` 非分歧参数和 `GL(n)` 局部因子归一化例子。
5. 第十五至十七章精校 functoriality、trace formula、endoscopy、Arthur 参数之间的对象字典；不证明 Arthur trace formula。
6. 第十八至二十二章精校几何 Satake、Hecke eigensheaf 和谱侧范畴的最短对象链；D-module 和 derived stack 技术保持接口性质。

### C. 后置或另卷

1. 完整 LCA Fourier inversion、Plancherel、Poisson summation 和 Tate thesis 解析证明。
2. 完整 class formation、Herbrand 理论、different/discriminant 公式和 cohomological class field theory。
3. 完整模曲线代数化、Atkin-Lehner-Li old/new 分解和 Deligne 表示构造。
4. 完整 Neron 模型存在性、Tate algorithm 逐步证明和残数特征 $2,3$ 的全表。
5. 完整 Taylor-Wiles patching、Poitou-Tate、p-adic Hodge 局部变形环和 automorphy lifting 证明。
6. 完整 Harish-Chandra、Bruhat-Tits、Bernstein center、Plancherel 和 Paley-Wiener 理论。
7. 完整 Arthur trace formula、稳定化、fundamental lemma、twisted trace formula 和 Arthur-Mok 分类证明。
8. 完整 D-module、IndCoh、derived stack、six functors、factorization、perfectoid、diamond 和 Fargues-Fontaine 技术层。
