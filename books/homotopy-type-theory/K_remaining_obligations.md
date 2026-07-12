# 附录 K：剩余证明义务登记

## 目标

本附录登记文本出版口径下仍需读者注意的证明义务、外部输入和研究边界。一条义务的状态只能是：已关闭、书内证明核、外部输入保留、研究边界保留或出版审校项。

## K.1 核心链状态

**K.1.1 基础类型论、路径代数与等价。**
范围：第 1-5 章，附录 A、D、E、G、AB、O。
状态：已关闭为书内证明核。
说明：第 1 章已固定 judgmental/propositional equality、类型转换、非累积宇宙和小族层级；第 2 章从完整 J 明写 transport、$\mathsf{apd}$、常值 transport、依赖自然性和逆运输。路径归纳、$\Sigma$ 路径、fiber 收缩、等价复合、准逆相干化、等价保持 h-level 和 h-level 向上闭包均已有书内证明核。

**K.1.2 外延性、单值性与 SIP。**
范围：第 6-8 章，附录 F、H、I、J、AG、T。
状态：书内证明核 / 外部输入保留。
说明：第 6 章已按 $\mathcal U_i$ 区分 universe path 与小等价类型，明确不使用 resizing，并书内推出沿 $\mathsf{ua}_i$ 的 transport 公式。函数外延性、单值性、命题外延性和结构等同性原则均有明确引入位置。单值性推出函数外延性按 HoTT Book Theorems 4.9.4-4.9.5 保留为精确外部输入；附录 T 把该输入限制到基底与 fibers 同属一个单值 universe 的情形，并禁止非累积口径下的隐式 lift，不再把压缩路线当作书内证明。

**K.1.3 HIT、截断、圆与基本群。**
范围：第 8-11 章，附录 L、M、N、V、W、AD、AI。
状态：已关闭为书内证明核与输入规则。
说明：本书实际使用的命题截断、一般截断、集合商、圆、悬挂和 pushout 均以分层规则表给出。圆输入明确区分 judgmental 点计算与 propositional 普通/依赖路径计算；CHM 的强 cubical 计算只按其列举签名作为外部输入。圆的 encode-decode、整数加法群律和 $\pi_1(\mathbb S^1)\cong\mathbb Z$ 有证明核。一般 HIT schema 仍是第十六章外部元理论边界。

**K.1.4 单值范畴论。**
范围：第 13-14 章，附录 P、Q、U、X、R、AA、AF、AH。
状态：核心文本已收口。
说明：预范畴、单值范畴、Yoneda、预层范畴、函子范畴、终对象唯一性、伴随形式和 Rezk 完备化泛性质均已有证明核或证明架构。AA.8-AA.10 的 transport 细节仍是压缩证明部分，但不造成未标注依赖。

## K.2 高级接口边界

**K.2.1 合成上同调与稳定方向。**
范围：第 12 章，附录 Y、AM、AQ、AV、AZ、BN。
状态：高级接口 / 外部输入保留。
说明：EM 型塔、上同调群、cup product、谱、exact couple、Serre/AHSS/Adams 模板和 Steenrod/Ext 计算均作为接口保留。完整高阶相干、具体谱序列计算和强收敛证明不作为核心封稿义务。

**K.2.2 Blakers-Massey、Hopf fibration、Postnikov 与局部系数。**
范围：附录 AL、AP、AU、AY、BF、BJ、BK、BM。
状态：证明核 / 外部输入保留。
说明：fiber sequence、长正合列、join connectivity、pushout 路径空间、higher groups、Postnikov tower、cofiber 和局部系数均有接口。完整分类定理、具体 obstruction 计算和局部系数谱序列保留为高级研究边界。

**K.2.3 集合层代数、选择和构造性分析。**
范围：附录 BH、BI、BL、AK、AR、AW、BA、BO。
状态：接口 / 研究边界保留。
说明：商群、商环、局部化、有限集、基数、序数、逻辑原则、Cauchy/Dedekind 实数、连续性、紧致性、级数和积分已有定义接口和部分证明核。完整误差预算、选择原则独立性、积分定理和代数泛性质细节不阻塞核心 HoTT。

**K.2.4 模型、对象语言和几何边界。**
范围：第 15-17 章，附录 Z、AO、AN、AS、AX、BB、BC、BD、BE、BG、AT。
状态：外部输入 / 研究边界保留。
说明：第 15-16 章已对 simplicial 相对一致性、CCHM $\mathsf{Path}$/Glue univalence、CCHM canonicity、Cartesian cubical normalization 和 CHM 列举型 HIT 分别记录语法、元理论假设、精确结论及非结论。Cubical type theory、Rezk/Segal types、QIIT、2LTT、directed/simplicial type theory、cohesive HoTT、SDG/SAG 和 displayed/bicategory coherence 均不得回流为 L0-L5 的隐式规则。

## K.3 出版审校义务

**K.3.1 链接和 README 覆盖。**
状态：已通过。
说明：当前出版审计确认本地 Markdown 链接无缺失，README 覆盖顶层 Markdown 文件。

**K.3.2 章节结构。**
状态：已通过。
说明：第 0-17 章均具备“本章目标”“依赖前置知识”“本章小结”“练习”结构。

**K.3.3 跳步词纪律。**
状态：已通过。
说明：禁用跳步词只允许出现在 `SKILL.md` 的约束条款中；正文若出现类似词汇，必须改写为明确证明、外部输入或研究边界。

**K.3.4 后续允许修改。**
状态：边界规则。
说明：后续只允许做术语、编号、符号、来源、交叉引用和真实错误修正。除非直接关闭本附录中的义务，不再新增横向专题。
