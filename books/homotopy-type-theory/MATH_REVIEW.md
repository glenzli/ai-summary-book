# 数学审查记录

本文件记录《同伦类型论与单值基础》当前版本的数学审查结果。它不是正文，而是后续扩写时的质量控制清单。

## 当前状态

1.  完整教材第一版已建立  
    当前新增 `SKILL.md`、`README.md`、`SOURCES.md`、`NOTATION.md`、`MATH_REVIEW.md`、第 0-17 章和附录 A-C。写作约束已要求每个非平凡结果标注证明或验证状态。

2.  基础口径已固定  
    当前采用 intensional Martin-Lof type theory 加层级宇宙作为第一阶段基础。函数外延性、单值性、高阶归纳类型、截断和选择原则均不作为第 1-2 章的默认规则。

3.  资料源已初步核查  
    已记录 HoTT Book、Rijke 教材、Coq-HoTT、UniMath、Cubical Agda、Agda 官方文档、cubical type theory 论文、simplicial model 论文和 1Lab。涉及软件版本和近期研究的条目需要在后续扩写时继续核查。

4.  高级章节采用分层验证  
    第 9-17 章大量内容属于外部输入、机器形式化或研究边界。当前版本已把圆的基本群和合成上同调核心骨架降为证明核；但不把 Rezk 泛性质、cubical canonicity、稳定同伦论和 cohomology 高阶相干等长证明伪装成书内已证结果。

5.  基础证明核已加入  
    当前新增附录 D，展开固定端点路径归纳、$\Sigma$ 路径刻画、基点路径纤维可收缩、命题纤维依赖和、等价诱导准逆等证明。第 2、4、5 章已开始引用这些证明核，把部分“证明说明”提升为书内证明。

6.  等价证明核已加入  
    当前新增附录 E，展开可收缩 total space、复合 fiber 分解和等价复合证明。第五章“等价复合”已提升为书内证明；“逆等价也是等价”由附录 G.8 补为书内证明，E.8 已同步改正。

7.  外延性与截断证明核已加入  
    当前新增附录 F，展开命题是集合、命题间双向蕴含给出等价、命题外延性、子类型外延性和命题截断泛性质。第 6-8 章对应结论已改为引用书内证明核。

8.  等价定义、Universe 非集合性与 SIP 已补强  
    当前新增附录 G-I，分别处理等价定义比较、逆等价与同伦层级保持；布尔类型非平凡自等价与 universe 非集合性；结构等同性原则及群对象实例。第 5、7、8 章对应缺口已改为书内证明引用。

9.  一元代数 SIP 与证明义务登记已加入  
    当前新增附录 J-K。附录 J 把结构等同性原则专门化到一元代数签名，覆盖幺半群和群等常见结构；附录 K 登记仍需后续逐行展开或外部化的证明义务。

10. HIT 输入规则已精确登记  
    当前新增附录 L，集中列出命题截断、一般截断、集合商、圆、悬挂和 pushout 的形成、构造、递归/消去和计算规则。第 8-11 章已引用该输入规则表；一般 HIT 元理论仍作为第十六章外部输入。

11. 整数对象与 successor 等价已补强  
    当前新增附录 M，区分用于圆 encode-decode 的归纳整数和用于集合层代数的商整数，证明 successor/predecessor 互逆，并把第十一章的 code 覆盖构造接到单值性路径 $\mathsf{ua}(\mathsf{succEquiv}_{\mathbb Z})$。附录 W 已补全整数加法交换群律；encode/decode 和基本群运算相容性已分别在附录 N、V 补强。

12. 圆的 encode-decode 证明核已加入  
    当前新增附录 N，展开 $\mathsf{code}$ 覆盖、沿 loop 的 transport、$\mathsf{encode}$、$\mathsf{decode}$、两个互逆同伦和 $(\mathsf{base}=\mathsf{base})\simeq\mathbb Z$。剩余边界集中在 HIT computation 口径差异。

13. 同伦层级性质命题性已补入  
    当前新增附录 O，用函数外延性和可收缩性命题性证明 $\mathsf{isOfHLevel}_n(A)$ 是命题，特别推出 $\mathsf{isSet}(A)$ 是命题。附录 N.9 已改为引用该书内证明核。

14. 准逆相干化公式已展开  
    附录 G.4 已把普通准逆相干化为半伴随等价的 $\epsilon'$ 公式和三角相干证明写出。附录 M 与 N 中“准逆推出等价”的引用不再依赖未展开公式。

15. 单值范畴论和 Yoneda 证明核已加入  
    当前新增附录 P-Q。附录 P 展开预范畴、同构、$\mathsf{idtoiso}$、集合范畴单值性和代数结构范畴单值性的证明核；附录 Q 展开集合值预层、自然变换和 Yoneda 引理的双向构造。第 13-14 章已改为引用这些证明核。附录 X 已进一步补齐一般函子范畴、自然同构和函子范畴单值性；Rezk 完备化泛性质已由附录 AA 降为证明架构。

16. Rezk 完备化已精确化  
    当前新增附录 R，用 Yoneda 嵌入的本质像给出 Rezk 完备化构造蓝图，登记 Rezk 嵌入的 fully faithful、essentially surjective 和单值性证明路线；附录 AA 进一步把对单值目标范畴的泛性质降为 weak equivalence 限制函子证明架构。第十四章 Rezk 小节已改为引用这些附录。

17. 形式化库索引已版本化  
    当前新增附录 S，按 2026-06-29 核查 Coq-HoTT、UniMath 和 Cubical Agda 的具体 commit、模块路径和入口 identifier。K.5 中三项形式化库索引缺口已降级为“脚本级依赖图、工具链版本和定义翻译”义务。

18. 单值性推出函数外延性已精确外部化  
    当前新增附录 T，登记 UniMath 中 `univalenceStatement -> isweqtoforallpathsStatement` 的形式化链条和 Coq-HoTT 的 `Univalence_implies_Funext` 入口。第六章 6.11 不再只是泛泛证明说明。

19. 预层范畴和 Yoneda 嵌入已补齐  
    当前新增附录 U，证明自然变换 Hom 集合性，定义预层范畴、自然变换复合和 Yoneda 嵌入，并把附录 Q 的 Hom 等价提升为 fully faithful 函子表述。K.4.2 的具体缺口已关闭；一般函子范畴理论已由附录 X 补入。

20. 圆的基本群同构已补强  
    当前新增附录 V，证明 loop 幂与整数加法相容，并把附录 N 的 loop space 等价提升为 $\pi_1(\mathbb S^1,\mathsf{base})\cong\mathbb Z$ 的群同构证明核。附录 W 现已补入其所需的整数加法群律。

21. 整数加法群律已补齐  
    当前新增附录 W，证明加法与左右 successor/predecessor 平移相容，推出左右单位律、结合律、交换律和逆元律，从而把 M.14 从证明说明提升为书内证明核。K.3.1 剩余项已降级为机器化翻译义务。

22. 函子范畴与预层范畴单值性已补齐  
    当前新增附录 X，定义一般函子、自然变换、函子范畴、自然同构，证明函子范畴同构等价于自然同构，并证明目标单值推出函子范畴单值。R.2 现在由 P.10、X.10、X.11 给出；R.11 的泛性质证明架构见附录 AA。

23. 合成上同调骨架已补齐  
    当前新增附录 Y，把第十二章的 EM 型和上同调入口提升为严格证明核：定义非约化与约化上同调，证明阿贝尔群结构、反变函子性、悬挂同构和球面上同调计算，并登记 cup product、graded commutativity 和 Eilenberg-Steenrod 性质的 Cubical Agda 形式化入口。第十二章已改为引用 Y。

24. Cubical/HIT 元理论边界已精确化  
    当前新增附录 Z，区分对象语言、元语言和实现语言，说明公理化 HoTT、Cubical 口径、Glue/univalence、canonicity、normalization、HIT 语义和模型比较的使用边界。第十六章和 K.2 已改为引用 Z；剩余项是外部元理论证明本身，而不是教材边界不清。

25. Rezk 完备化泛性质已降为证明架构  
    当前新增附录 AA，定义 weak equivalence 和限制函子，证明自然变换由本质满像决定，并给出限制函子 fully faithful、essentially surjective 以及 Rezk 泛性质的证明架构。R.11 和第十四章已改为引用 AA；剩余义务集中在 AA.8-AA.10 的逐行 transport 与代表元相容计算。

26. 同伦层级向上闭包已补齐  
    当前新增附录 AB，证明可收缩类型的路径空间可收缩，并由自然数归纳证明 $\mathsf{isOfHLevel}_n(A)\to\mathsf{isOfHLevel}_{n+1}(A)$。第四章 4.13 已从证明说明改为书内证明。

27. Eckmann-Hilton 已补齐  
    当前新增附录 AC，先证明抽象 Eckmann-Hilton 引理，再在二重 loop space 上用纵向/横向复合和 interchange law 推出高阶同伦群交换性。第十二章 12.3 已从证明说明改为书内证明核。

28. $\mathsf{susp}(\mathbf 2)\simeq\mathbb S^1$ 已补齐  
    当前新增附录 AD，构造悬挂到圆和圆到悬挂的双向函数，并用 HIT 依赖消去和路径代数证明双向同伦。第十章 10.5 已从证明说明改为书内证明核。

29. 自然数与和类型离散性已补齐  
    当前新增附录 AE，证明自然数路径的 encode-decode、自然数集合性、和类型 no-confusion 与和类型保持集合性。附录 M.1 和 M.5 已从证明说明改为书内证明。

30. 极限唯一性与伴随形式已补齐  
    当前新增附录 AF，证明终对象之间唯一同构和单值范畴中终对象唯一到路径，并证明 Hom 等价形式与单位/余单位三角恒等式形式互相构造。第十四章 14.6、14.8 已从证明说明改为书内证明核。

31. 结构 transport 与代数 SIP 已补齐  
    当前新增附录 AG，证明沿 $\mathsf{ua}(e)$ transport 常元、运算和有限 arity 运算的计算公式，推出命题性公理代数签名的传统同构等价于规范结构等价，并证明相应结构范畴单值性。附录 I.6、I.7 和 P.11 已从证明说明改为书内证明。

32. Full subcategory 与本质像证明核已补齐  
    当前新增附录 AH，证明命题性 full subcategory 保持单值性，并证明函子的本质像核心限制 essentially surjective，且在原函子 fully faithful 时仍 fully faithful。附录 R.7、R.10 已从证明说明改为书内证明。

33. Pushout 等价不变性已补齐  
    当前新增附录 AI，构造等价 span 之间 pushout 的双向递归函数，并用 pushout 依赖消去证明双向同伦。第十章 10.12 已从证明说明改为 pushout 证明核；一般高阶同伦余极限函子性仍保留为 HIT 元理论边界。

## 已知风险

1.  形式化状态不是定理本身  
    Coq-HoTT、UniMath、Cubical Agda 和 1Lab 的口径不完全相同。后续不能把某库中的同名定理直接移植为另一基础中的定理，必须检查公理、universe 和 HIT 支持。

2.  Judgmental equality 的选择敏感  
    不同呈现会把某些 eta 规则作为 judgmental 规则或 propositional 定理。当前书稿保守处理：只把明确列出的 beta 规则作为 judgmental computation；eta 原则须单独声明。

3.  外部模型论不可替代内部证明  
    simplicial model、cubical model 和 canonicity 结果是元理论输入。它们可支撑一致性或计算解释，但不能在内部章节中当作普通项构造使用。

4.  “最新研究”需要版本化  
    HoTT 相关库和研究仍在变化。后续每次加入 2025-2026 以后结果时，必须记录核查日期、来源链接和形式化系统。

5.  完整教材第一版仍是草稿  
    当前版本补齐了教材级结构和核心内容，但尚未达到出版级逐行形式化。后续应按章节做独立数学审查，优先审查第 5-8 章等价与单值性、第 9-11 章 HIT 和圆的基本群、第 13-14 章单值范畴论。

## 当前外部输入定理

1.  HoTT 基础理论的一致性背景  
    当前只引用 simplicial model 和 cubical type theory 作为外部输入；前两章不依赖其具体证明。

2.  Coq-HoTT、UniMath、Cubical Agda 的机器形式化事实  
    当前已有附录 S 作为版本化入口索引；前两章正文不依赖任何库特定定理。若后续章节把某个库入口升级为正文依赖，必须补定义翻译和假设列表。

3.  单值性和函数外延性  
    第六章以后作为原则引入。若采用 HoTT Book 口径，可公理化；若采用 Cubical Agda 口径，应改写为 cubical primitive 的计算性后果。

4.  高阶归纳类型与截断  
    第八至十一章以规则格式使用 HIT、截断和 quotient。当前版本不证明一般 HIT 的元理论存在性，只在第十六章讨论模型与实现。

## 后续审查清单

- 新章节是否更新 `README.md` 阅读顺序。
- 新符号是否更新 `NOTATION.md`。
- 新来源是否更新 `SOURCES.md`。
- 每个定理是否标注“书内证明”“证明说明”“外部输入”“机器形式化”或“研究边界”。
- 是否误用了函数外延性、命题外延性、单值性、高阶归纳类型、截断、排中律或选择原则。
- 是否把拓扑空间直觉直接当作类型论证明。
