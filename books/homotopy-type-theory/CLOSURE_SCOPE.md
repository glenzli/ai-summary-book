# 收口范围与封稿门槛

本文件固定《同伦类型论与单值基础》的文本出版收口范围。新口径下，本书不以书稿外材料作为封稿条件；封稿只检查数学文本、证明状态、来源、符号和边界是否一致。

## C.1 收口目标

本书的收口目标是一条可审查的 HoTT 教材链：

1.  内部语言：依赖类型论、恒等类型、路径归纳、归纳类型和宇宙；
2.  单值基础：等价、函数外延性、单值性、同伦层级、截断和集合商；
3.  基础合成同伦论：HIT、圆、悬挂、pushout、encode-decode、基本群、fiber/cofiber sequence；
4.  单值范畴论：预范畴、单值范畴、Yoneda、函子范畴、Rezk 完备化架构；
5.  模型与对象语言边界：cubical type theory、simplicial model、QIIT、2LTT、directed/cohesive HoTT 等作为外部输入或研究边界；
6.  高级接口：合成上同调、谱、Postnikov、模态、构造性实数、局部系数和谱序列作为分层接口。

## C.2 封稿等级

**等级 0：结构收口。**
目录、依赖、符号、来源和审查文件完整；不再新增主要方向。

**等级 1：出版草稿。**
每个定理都有证明状态标签；每个外部输入都有来源；每个新符号在 `NOTATION.md` 中登记；每个未关闭证明义务在 `K_remaining_obligations.md` 中登记。

**等级 2：严格出版候选。**
核心 HoTT 主线的关键证明说明已降为书内证明、精确外部输入或明确研究边界；高级接口全部显式标为外部输入或研究边界；无未解释公理偷用。

当前按 HoTT 教材自身主线计算，L0-L5 核心链已达到等级 2 的收口口径。

## C.3 固定核心

以下文件属于收口期的核心 HoTT 内部链：

- `01_dependent_type_theory_and_judgments.md`
- `02_identity_types_and_paths.md`
- `03_basic_inductive_types.md`
- `04_contractibility_and_hlevels.md`
- `05_equivalences_and_fibers.md`
- `06_function_extensionality_and_univalence.md`
- `07_univalence_consequences.md`
- `08_truncations_sets_quotients.md`
- `09_higher_inductive_types.md`
- `10_circle_suspension_pushouts.md`
- `11_fundamental_group_and_coverings.md`
- `13_univalent_categories.md`
- `14_yoneda_limits_adjunctions_rezk.md`
- `D_foundational_proof_kernel.md` 至 `AI_pushout_equivalence_invariance.md` 中对应核心证明核。

## C.4 固定高级接口

以下内容保留为教材高级接口，但封稿时不得伪装为完全内部证明：

- 合成上同调、EM 型、cup product、谱和谱序列；
- Blakers-Massey、Freudenthal、Hopf fibration、Postnikov、Whitehead；
- Steenrod algebra、Adams/Serre/AHSS 具体计算；
- Cauchy/Dedekind 实数、构造性分析、积分；
- directed/simplicial type theory、Rezk types、2LTT；
- cohesive HoTT、SDG、SAG；
- cubical/model/canonicity 元理论。

这些内容可以有书内证明核，但若依赖额外对象语言、模型、HIIT/QIIT 或经典稳定同伦论计算，必须保留外部输入或研究边界标签。

## C.5 禁止继续扩展的方向

收口模式下禁止新增以下类型内容，除非它直接关闭 `K_remaining_obligations.md` 中已登记的义务：

1.  新的研究领域概览；
2.  新的外部对象语言；
3.  新的几何、代数或分析专题；
4.  未与现有定理依赖相连的例子；
5.  未进入来源索引或证明义务表的外部输入。

## C.6 封稿门槛

封稿前必须同时满足：

1.  `README.md` 中所有本地链接存在；
2.  宽松占位语和跳步词扫描只允许命中约束文件中的禁止条款；
3.  每个正文和附录新增符号均在 `NOTATION.md` 中；
4.  每个来源型断言均在 `SOURCES.md` 或附录 S 中可定位；
5.  `K_remaining_obligations.md` 中每条义务有状态：关闭、外部输入保留或研究边界保留；
6.  不再出现未标注的选择、排中律、resizing、HIT、univalence 或函数外延性使用；
7.  高级接口没有回流为 L0-L5 的隐式证明前提。

## C.7 收口工作顺序

1.  固定范围和依赖分层；
2.  清理核心 HoTT 证明义务；
3.  清理 HIT 与圆的基本群证明义务；
4.  清理单值范畴论证明义务；
5.  把高级合成同伦论、分析和模型论内容统一标为接口、外部输入或研究边界；
6.  做全书术语、编号、符号、来源和交叉引用审校。

## C.8 当前结论

按 HoTT 核心而不是所有高级接口计算，C.7 的第 1-6 项已经完成到严格出版候选口径。剩余可做的工作只属于出版校对层面：术语、编号、符号表、来源表和交叉引用修正；除非发现核心证明链中的真实错误，不再新增数学方向。
