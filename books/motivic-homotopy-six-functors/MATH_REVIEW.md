# 数学审查记录

核查日期：2026-07-15
当前状态：第 00 至 24 章与附录 A-H 形成完整教材草稿。基础主链、第 09-18 章
cohomology/motives/transfers 主线及第 19-23 章扩展理论的 P0 外部输入均已完成
OET 复核和 locator 对账。自动化交叉引用、长篇习题详解与最终版式仍属于出版增强，
不影响教材内容闭合。

## 2026-07-15 教材化与扩展章终校

- 第 00-24 章已改为内容特定的自然导言和数学收束；固定的“本章目标/依赖/主线/小结”
  模板以及正文中的项目管理口吻已经移除。
- 第一章及附录 B 的 Nisnevich distinguished square 论证改用可表预层的推出、层化与
  Yoneda，不再把 representable presheaf 错当成在预层层面保持该推出。
- 第 19-20 章把 equivariant、genuine/lisse、scalloped stack、exceptional functor
  定义域与 torus concentration 的假设分别写明；Hoyois 与 Khan--Ravi 的 P0 定理已定位。
- 第二十一章区分整系数 nil-invariance、反演指数特征后的 universal-homeomorphism
  invariance 以及 perfect motivic theory 的研究边界，避免把三者合并成无条件等价。
- 第二十二章区分绝对 Betti、相对六操作 realization、pro-space etale realization、
  profinite etale realization、实点与 `C_2`-equivariant Betti theory；Tate sphere 的像以
  homotopy cofiber 表述。
- 第二十三章把 universal property 限定在明确的 ambient category 及其编码结构中，
  不再由 pullback formalism 自动推出定义之外的全部六操作相容性。

## 2026-07-12 purity/excess 交叉审计

- 核对 Deglise--Jin--Khan Proposition 2.5.4、Remark 2.5.5 与 Paragraph
  4.3.1 后，定理 6.12 的复合相容性降为
  `\operatorname{Ho}(\mathbf{SH})` 值逆变伪函子层，transverse base change
  降为 homotopy-category 中的自然变换交换方块；该来源明确没有完成
  infinity-category 层的自然变换增强，故正文不再声称 higher coherence。
- Smooth purity 由 Hoyois Theorem 6.18(2) 单独保留为 stable
  infinity-categorical natural equivalence，其三角影子与 DJK smooth class
  相符。
- 核对 DJK Paragraph 3.3.3、Propositions 3.3.4、4.2.2 后，excess formula
  明确要求 Cartesian square 中 `f,g` 均为 smoothable lci s-morphisms，并由
  法丛单射取得 locally free quotient `\xi`。系数版本与 proper push-pull 的
  额外假设已分别写入正文和 `FC-16.13` 账本条目；任意非 Tor-independent
  方块不再被概括为自动存在 excess 修正。

## 2026-07-11 OET 主链修订

- 宇宙口径改为 `\mathbb U\in\mathbb V` 的相对 presentability；固定
  finite-type-over-`B` 的 base-change-closed 默认基范畴，并区分普通 sheaf
  与 hypercompletion。
- 把 presheaf presentability、higher sheafification 和 small-generated
  localization 改为带 HTT locator 的外部基础输入，保留书内应用证明。
- 修正错误命题：在 symmetric monoidal category 中，`A\otimes B` 可逆
  确实推出 `A`、`B` 分别可逆；据此从 `T`-反演推出 ordinary suspension
  可逆与稳定性。
- 对象反演、3-symmetry、sequential/symmetric spectra 模型不再混写；
  Robalo 4.10、4.24、4.29 与 Hoyois 6.3、6.4、6.7 已定位。
- 六操作按方差和态射类重写：`f^*\dashv f_*` 对所有默认态射，
  `f_!\dashv f^!` 只对 separated morphisms，`f_\sharp` 只对 smooth。
- Base change 分为 exceptional、proper ordinary、smooth ordinary 三类；
  projection formula 分为 exceptional、proper ordinary、dualizable-coefficient
  ordinary 三类，并修正 ordinary projection map 的方向。
- Purity 分为 unstable homotopy purity、smoothable-lci purity
  transformation、smooth purity equivalence、coefficientwise `f`-purity 与
  absolute purity；删除 regular closed immersion 的无条件 functor equivalence。
- 按 exceptional 态射类把 smooth purity/ambidexterity 收紧到 smooth
  separated maps；补出 ambidexterity 的伴随证明，并修正 proper-smooth
  trace 的 `\Sigma^{T_f}` 扭曲及 Atiyah dual
  `f_\sharp\Sigma^{-T_f}\mathbb 1_X\simeq f_*\mathbb 1_X`。
- 第十六章统一使用 `\tau_f=\langle L_f\rangle` 的 virtual-tangent
  convention，删除把 dualization 错当作 K-theory additive inverse 的
  `T_f=-L_f` 写法，并把 fundamental-class 输入收紧到 smoothable lci。
- 定义并区分 compact 与 geometrically constructible objects，补六操作保持
  compactness 的伴随判据和精确 continuity 输入。
- 附录 F 补 stable infinity-natural transformation 与 bare triangulated
  isomorphism 的差别，并完整证明两种 compactness 口径在已有 enhancement
  下等价。
- 新增 `P0_REFERENCE_LOCATORS_BATCH_2.md`，同步更新 sources、theorem/
  locator ledgers、notation、index 和习题解答。

## 2026-07-11 第 09--18 章 P0 locator 闭合

- `H\mathbb Z` 构造、higher Chow/Milnor `K` 比较、`DM`-module 比较、
  `KGL/KH/cdh`、`MGL` universality/Hopkins--Morel 和 zero slice 均补到
  原始资料的定理号、基与系数假设。
- `DM` 比较明确分成 characteristic zero 的 monoidal triangulated
  equivalence 和反演 exponential characteristic 后的 presentably symmetric
  monoidal stable infinity-category equivalence。
- Finite correspondence 复合、Nisnevich sheafification、framed
  recognition、smoothable-lci Gysin、finite-etale stable norms、normed
  `HZ/KGL/MGL` 与 Morel `End(1)=GW` 均完成 locator 对账。
- 修正 finite syntomic 的 cotangent Tor-amplitude 为 `[-1,0]`；区分
  very-effective infinity-category 与 stable/triangulated category；修正
  `H^{0,0}` 的 multiplicative norm 示例为 split degree `d` 时 `n\mapsto n^d`。
- 新增 `P0_REFERENCE_LOCATORS_BATCH_3.md`，并同步正文、`SOURCES.md`、
  三份 theorem/locator 索引和内部闭合矩阵。

## 已达成

- 已固定默认基概形、小骨架、Nisnevich topology、`\mathbb A^1`-局部化、`T`-稳定化和六操作的符号口径。
- 第一至第二十四章已采用“定义-命题-证明-外部输入定理-练习”结构。
- Motivic 六操作、proper compatibility、localization、purity、ambidexterity、duality、base change 和 projection formula 已形成主干闭包。
- `H\mathbb Z`、`DM`、`KGL/KH`、`MGL`、slice filtration、finite/framed transfers、fundamental classes、norms、Milnor-Witt refinements、equivariant/stacky/log/perfect/realization/universal formalisms 已建立第一层严格口径。
- 附录 A-H 已补齐大小、Nisnevich topology、稳定化、mate calculus、代数几何背景、三角/稳定 infinity 翻译、资料源索引和低阶计算例子。
- `CHAPTER_DENSITY_AUDIT.md` 已逐章检查大纲态风险；除序章和研究边界章因性质为 B，其余主体章达到 A 级教材草稿密度。
- `TEACHING_CLOSURE_AUDIT.md` 已按内容、证明、引用、术语和教学可读性给出教材闭合判定。
- `TYPESETTING_AND_NUMBERING.md` 已固定编号、排版、证明格式和交叉引用规范。
- `INDEX.md` 已补全书主题索引。
- `EXERCISE_SOLUTIONS.md` 已给出 205 道练习的一版解答要点。
- `P0_REFERENCE_LOCATORS_BATCH_1.md` 已定位 Drew-Gallauer universal formalism、framed recognition、norms、fundamental classes/Gysin maps。
- `P0_REFERENCE_LOCATORS_BATCH_2.md` 已定位基础范畴论、稳定化、六操作、purity 与 triangulated shadow。
- `P0_REFERENCE_LOCATORS_BATCH_3.md` 已定位第 09-18 章教学主线，并把不参与主线的高级比较降为 P1。
- 内部可证明结论已经与外部深定理区分。
- 2025-2026 近期资料已单独放入研究边界，不作为无条件正文基础。

## 教学闭合判断

- 按“可作为完整教材阅读和教学使用”的标准，本书已经收口：读者可以从站点和局部化一路读到六操作、谱、转移、范数、framed recognition、基本类和现代扩展。
- 各章均有定义、命题、证明或外部输入标记、边界说明和练习；不再是目录或导览态。
- 深定理没有伪装成书内证明：六操作存在性、homotopy/smooth purity、
  Atiyah duality、`H\mathbb Z`/`DM` 比较、`KGL`/`MGL` 表示性、framed
  recognition、norms、fundamental classes、stack/equivariant/log/analytic
  扩展都标为外部输入或研究边界；smooth ambidexterity 则明确由 purity
  和伴随唯一性书内推出。
- 三批 P0 locator 及后续扩展章对账已覆盖基础、六操作、第 09-18 章主线以及
  equivariant、stacky 与相对 Betti realization。仍标为 P1 或 R 的结果不参与主线证明。

## 剩余增强项

- `\mathbf{SH}(S)` 的构造采用 infinity-categorical 教材口径；若面向模型范畴读者，可另加 Morel-Voevodsky/Jardine 模型比较补章。
- `T`、`\mathbb P^1/\infty` 与 `S^{1,1}` 的坐标级等价可增加更多计算例子。
- 第五至第八章采用的 Hoyois trivial-group package 已补精确 locator；若要
  并列采用 Ayoub/Cisinski-Deglise 的更广或不同 coefficient-system 版本，仍需
  逐项补 locator，不能直接互换假设。
- 第 09-18 章已降为 P1 的 etale/Adams/Chern-character/Hilbert/Tambara/
  quadratic-enumerative 结果若升级为正文主线，须按具体模型另补 locator；
  当前不得反向调用。
- 习题已有一版解答要点；若面向出版，可把 Nisnevich square、Thom space、localization triangle、finite correspondence 复合、Gysin excess 和 slice spectral sequence 等题扩展为长篇详解。

## 当前结论

数学上可以按“完整教材可读版”收口。若后续继续推进，应进入教学增强和出版校订，而不是继续扩充目录。
