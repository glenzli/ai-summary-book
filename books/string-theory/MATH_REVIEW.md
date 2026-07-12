# 数学审查记录

本文档记录《String Theory》教材的严格性审查清单和阶段状态。

## 全书审查清单

- [x] 每个作用量是否声明 Lorentzian/Euclidean worldsheet。
- [x] 每个 CFT 公式是否声明 OPE 归一化。
- [x] 每个谱公式是否声明 normal ordering constant。
- [x] 每个 BRST statement 是否区分 nilpotency、closed state 和 exact state。
- [x] 每个对偶性是否标为物理猜想或外部输入，而不是数学定理。
- [x] 每个低能有效作用是否说明阶数：tree/genus、$\alpha'$ 或 $g_s$。

## 第一轮写作记录

- 已建立 [SKILL.md](SKILL.md)、[NOTATION.md](NOTATION.md)、[NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md)、[SOURCES.md](SOURCES.md)、[THEOREM_INDEX.md](THEOREM_INDEX.md)、[DEPENDENCY_GRAPH.md](DEPENDENCY_GRAPH.md) 和 [CLOSURE_STATUS.md](CLOSURE_STATUS.md)。
- 已完成第 0 至 20 章第一版入口和附录 A-E。
- 已将 no-ghost theorem、state-operator correspondence 和 moduli 分解标为外部输入。
- 当前风险：第 6 至 20 章仍需由入口版扩写为完整教材章，超弦和对偶性主线尚未闭合。

## 第二轮正式教材化记录

- 新增 [MAINLINE_PROOF_CHAINS.md](MAINLINE_PROOF_CHAINS.md)、[CONCEPTUAL_AUDIT.md](CONCEPTUAL_AUDIT.md)、[CHAPTER_CLOSURE_AUDIT.md](CHAPTER_CLOSURE_AUDIT.md)、[EXERCISE_COVERAGE.md](EXERCISE_COVERAGE.md) 和 [NUMBERING_CROSSREF_AUDIT.md](NUMBERING_CROSSREF_AUDIT.md)。
- 第 6 章已扩写为顶点算子、Koba-Nielsen 因子、Veneziano/Virasoro-Shapiro 振幅和因子化接口的教材章。
- 第 7 章已扩写为圆紧化质量公式、T-duality、D-brane 边界条件、Chan-Paton factors 和 D-brane tension 接口的教材章。
- 第 8 章已扩写为 RNS action、super-Virasoro algebra、ghost central charge、临界维数、NS/R 谱和 GSO projection 的教材章。
- 第 9 章已扩写为 type II chirality、R-R potentials、Green-Schwarz kappa symmetry 接口、light-cone 自由度计数和 BPS bound 的教材章。
- 第 10 章已扩写为 heterotic central charge accounting、lattice construction、current algebra、modular invariance 和 Green-Schwarz anomaly cancellation 的教材章。
- 第 11 章已扩写为 sigma-model beta functions、NS-NS effective action、frame transformation、type II/heterotic low-energy action 和展开参数的教材章。
- 第 12 章已扩写为 D-brane worldvolume fields、DBI 展开、WZ coupling、lower brane charge 和 D3-brane 低能理论的教材章。
- 第 13 章已扩写为 Calabi-Yau 定义、Ricci-flat 背景、holonomy supersymmetry、Hodge moduli counting 和 heterotic bundle 接口的教材章。
- 第 14 章已扩写为 S-duality、IIB $SL(2,\mathbb Z)$、IIA/M-theory、D0-KK 匹配、U-duality 和 protected tests 的教材章。
- 第 15 章已扩写为 genus expansion、moduli dimension、Beltrami/$b$-ghost measure、torus modular invariance 和 degeneration factorization 的教材章。
- 第 16 章已扩写为 topological twist、A/B model、localization、periods、mirror map 和 topological string free energy 的教材章。
- 第 17 章已扩写为 BPS index、D1-D5-P entropy、Cardy formula、Strominger-Vafa matching 和 attractor mechanism 接口的教材章。
- 第 18 章已扩写为 AdS/CFT 标准对偶、参数字典、D3 near-horizon limit、GKPW dictionary、bulk scalar dimension 和 symmetry matching 的教材章。
- 第 19 章已扩写为 flux quantization、GVW superpotential、tadpole cancellation、Kahler moduli stabilization 和 landscape 控制条件的教材章。
- 第 20 章已扩写为 QFT、几何、数论接口边界和全书定理状态回顾的收束章。
- 修正闭弦质量公式：闭弦应满足
  $$
  M^2=\frac4{\alpha'}(N-a)=\frac4{\alpha'}(\tilde N-a),\qquad N=\tilde N,
  $$
  在 level matching 后才可写为
  $$
  M^2=\frac2{\alpha'}(N+\tilde N-2a).
  $$
- 当前风险：全书主线已覆盖，但例题密度、附录公式表、局部证明完整度和习题数量仍未达到最终出版标准。

## 第二轮机械检查

- `git diff --check -- books/string-theory`：通过。
- 旧闭弦错误质量公式与禁用标记：未命中。
- Markdown 本地链接：67 个，缺失 0 个。
- 定理索引编号：146 个，缺失 0 个。
- 章节内定理/定义编号重复：0 个。

## 第三轮正式教材细节化记录

- 新增 [FORMAL_TEXTBOOK_COMPLETENESS.md](FORMAL_TEXTBOOK_COMPLETENESS.md)，将内容范围完整、内部完整和细节完整分成可审查标准。
- 第 1 章已扩写作用量变分、边界项、Noether theorem、Hilbert stress tensor、Weyl tracelessness、$B$-field 与 dilaton coupling。
- 第 2 章已扩写 Nambu-Goto/Polyakov 等价、局部对称性、Virasoro constraints、闭弦与开弦模展开、边界条件变分。
- 第 3 章已扩写 radial quantization、contour deformation、free boson OPE、Ward identity、Virasoro algebra 和 ghost number。
- 第 4 章已扩写开闭弦正则量子化、oscillator algebra、number operators、质量公式、临界条件和低能谱。
- 第 5 章已扩写 Polyakov path integral、metric fluctuation 分解、ghost action、BRST current、exact state decoupling 和 sphere ghost zero mode counting。
- 解答集已补齐第 1 至 5 章新增核心习题。

## 第四轮逐章细节化记录

- 第 9 章补充 type II massless field table、degrees of freedom matching、GS supersymmetric one-form、light-cone GS spectrum、D-brane parity 和 IIB self-dual five-form 说明。
- 第 10 章补充 heterotic lattice theta function、root states、partition function lattice 因子、Green-Schwarz factorization 的 $X_4/X_8$ 结构和 trace convention。
- 第 11 章补充 dilaton equation、R-R kinetic term 的 dilaton coupling、Einstein-Hilbert 振幅匹配和 $R^4$ 修正接口。
- 第 12 章补充 nonabelian SYM 低能极限、scalar vev 与 brane separation、anomaly inflow 接口。
- 第 13 章补充 quintic threefold、Hodge number 计数、Euler characteristic 和 harmonic forms 的零模展开。
- 第 14 章补充 M2 包裹圆的张力匹配和 D3-brane S-duality 自洽性。
- 第 15 章补充 torus fundamental domain、compact boson lattice sum 和 modular $T$ 对 level matching 的约束。
- 第 16 章补充 quintic mirror Picard-Fuchs operator、mirror theorem 接口和 mirror map 的局部形式。
- 第 17 至 20 章补充 wall crossing、Wald entropy、AdS/CFT two-point function、Wilson loop、no-scale potential、ISD flux condition 和接口矩阵。
- 全书当前 42 个 Markdown 文件，约 5430 行；定理索引 138 条，核心解答集超过 1000 行。

## 第五轮出版化细节记录

- 第 3 章补充 Virasoro highest-weight modules、Verma modules、null states 和低 level Gram matrix。
- 第 4 章补充 light-cone coordinates、light-cone physical oscillators 与 Lorentz algebra closure 的临界条件。
- 第 5 章修正 BRST charge mode 公式中的连接符号，并补充 genus-one torus vacuum amplitude 与临界玻色弦 integrand。
- 第 6 章补充 tree factorization、worldsheet degeneration 与 loop amplitude 接口。
- 第 7 章补充 Narain lattice、$O(d,d;\mathbb Z)$、orbifold twisted sectors 和 orientifold 接口。
- 第 8 章补充 bosonized superghost、picture-changing operator、NS vector 的不同 picture 表示和十维 spin field dimension。
- 全书当前 42 个 Markdown 文件，约 5770 行；定理索引 146 条，解答集 66 题。

## 最终内容收口审定记录

- 新增 [CONTENT_CLOSURE_AUDIT.md](CONTENT_CLOSURE_AUDIT.md) 和 [REFERENCE_MAP.md](REFERENCE_MAP.md)，分别固定内容收口判定与逐章资料源映射。
- 当前版本不再按“继续扩张主线”处理；第 0 至 20 章和附录 A-E 已形成教材内容闭包。
- 证明义务按 `P`、`S`、`E`、`C` 四类闭合：正文证明、标准物理推导说明、外部输入、物理猜想或对偶性原则。
- 机械检查结果：44 个 Markdown 文件，5917 行；Markdown 本地链接 78 个且缺失 0 个；定理索引 146 条且缺失 0 条；正文编号标签 357 个且重复 0 个；核心习题解答 66 题且重复 0 题。
- 剩余工作属于出版化增强：更多例题、图表、历史注、书末排印索引、附录公式表和局部标准物理推导说明扩写。

## 第六轮内容本体清理记录

- 新增 `INDEX.md`、`GLOSSARY.md` 和 `EXERCISE_INDEX.md`，统一总索引、压缩术语表和习题索引。
- 补齐序章练习 0.1、0.2 的解答，使正文 68 道习题与解答集 68 个条目一一对应。
- 统一第 1、2、5 章的“本章小结”标题格式，修正第 19 章小节编号顺序。
- 扩充附录 A-E：补充 Hodge star、characteristic classes、first-order CFT systems、modular functions、Sugawara construction、descent formalism、theta functions、spin structures、BPS bounds 和 supersymmetry variations。
- 新增 [WORKED_EXAMPLES.md](WORKED_EXAMPLES.md)，补充 8 个主线例题，覆盖 Polyakov variation、开弦谱、T-duality、BRST exact decoupling、DBI 展开、quintic 计数、AdS scalar dimension 和 no-scale potential。
- 机械检查结果：48 个 Markdown 文件，6641 行；Markdown 本地链接 262 个且缺失 0 个；定理索引 146 条且缺失 0 条；正文编号标签 403 个且重复 0 个；正文习题 68 道且解答 68 道。
