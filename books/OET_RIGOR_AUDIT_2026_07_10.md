# OET 本体严格性修订记录（2026-07-10 至 2026-07-12）

## 范围与口径

本轮覆盖 `books/` 下 14 本数学、数学物理与理论物理教材，不含 AI 综述和
《随机鹦鹉的自传》。共享判定标准见
[OET_RIGOR_STANDARD.md](OET_RIGOR_STANDARD.md)，执行约束见
[SKILL.md](SKILL.md)。

这里的“闭合”指教材本体闭合：对象与约定可追踪，主线命题有书内证明或
明确的外部输入边界，物理推导标明近似与适用域，引用能回到一手来源。
它不等于逐页排版已经达到 camera-ready，也不要求用 Lean 重建外部基础学科。

## 共享基础设施

1. 建立对象、良定义、定理陈述、证明责任、外部输入和物理近似的统一标准。
2. 新增 [audit_oet_rigor.py](audit_oet_rigor.py)，检查定理证明边界、占位符、
   本地链接、Markdown 围栏、显示公式、LaTeX 环境、兼容性宏、裸
   `coloneqq` 和畸形积分微分。
3. 各书的局部 `SKILL.md`、定理账本、来源索引、符号表、依赖图、练习答案
   与正文同步；深层结果只以精确外部输入进入证明链。

## 第一轮正文修订

- **范畴、operad 与 HoTT**：补齐宇宙/小性、Yoneda、伴随、Kan extension、
  对称替换、空纤维与零元操作、bar-cobar/Koszul 边界、dendroidal typing，
  并严格区分 judgmental equality、path equality、univalence 与 HIT 外部输入。
- **chromatic、motivic 与凝聚数学**：补齐局部化、height/type、fracture 与
  convergence，统一六函子方差、purity、Gysin、norm、solid/analytic/liquid
  主线；P0 外部定理均补到一手资料的定理、章节或页码定位。
- **Langlands、几何表示、HMS 与棱柱上同调**：收紧 Haar/reciprocity、
  Satake、trace formula、Fargues-Fontaine、卷积与分解定理、Fukaya
  (A_infty) 符号和解析输入、derived completion、Nygaard/syntomic 与
  比较定理的假设和完成化口径。
- **量子力学、相对论、弦论与 EFT**：补齐无界算子定义域、谱与连续态归一化、
  Dyson 余项、ADM/FLRW/Kerr 号差、世界面约束与 BRST/GSO 边界、EFT
  power counting、matching、RG、Hermiticity 和截断余项。

## 独立交叉审计修正

第一轮作者 agent 完成后，另由独立审校检查高风险链条，并修正：

1. motivic lci purity 只在来源实际支持的同伦三角/伪函子层陈述；excess
   intersection 补齐 smoothable-lci、excess bundle 与 proper 条件。
2. categorical geometric Langlands 固定
   (operatorname{DMod}_{1/2}(operatorname{Bun}_G)	o
   operatorname{IndCoh}_{mathcal N}(operatorname{LocSys}_{widehat G}))
   的来源方向；反向使用明确写逆等价。
3. Fargues-Fontaine 向量丛与 (G)-bundle 分类加入完备代数闭几何点假设，
   一般 perfectoid 基底改为 families 与 v-descent 边界。
4. 重 singlet matching 的 (M^{-4}) 展开补入同阶
   (-a^2kappa X^3/(2M^4)) 与两个独立小参数条件。
5. SMEFT 拉氏量按自伴/非自伴 sector 写成 Hermitian 形式，并统一 Weinberg
   与 Warsaw basis 的 dagger-pair 计数口径。
6. chromatic 主线十组外部输入补齐 Quillen、Landweber、
   Goerss-Hopkins-Miller、Devinatz-Hopkins、HKR、tmf、
   Gross-Hopkins、Picard 与 ANSS 的精确一手定位。

## 第一次提交前验证

- `python3 books/audit_oet_rigor.py --strict`：
  `errors=0 warnings=0`。
- `python3 books/category-theory/validate.py`：章节、练习与综合题闭合。
- `python3 books/quantum-mechanics/validate.py`：章节、外部输入、索引、
  练习与答案闭合。
- `git diff --check -- books`：无尾随空白或补丁格式错误。

## 第一次提交后的独立复审

第一次提交 `09b9837` 后，六个只读审校 agent 按互不重叠的学科组重新检查
全部 14 本教材。审校只接受可定位的数学/物理错误、假设缺失、类型错配、
归一化冲突和外部输入越界，不把可选扩写或排版偏好计为 finding。

复审共确认 36 项高/中等级问题，已全部修正：

- **Operad 与 HoTT**：修正自由 operad 权重、infinitesimal composition 的
  类型、bar-cobar counit、树骨架与 dendroidal degeneracy；消除固定端点 J
  的循环证明，收紧 univalence-to-funext 的宇宙范围并补齐 HIT 计算规则。
- **Motivic 与 condensed**：把 norm quotient 公式限制到来源实际覆盖的
  finite-etale 情形，补全 deformation-to-normal-cone 的开 blow-up 定义，
  并移除未定义的复 liquid 派生范畴记号。
- **Langlands**：修正 geometric Satake 的 Frobenius/Tate 次幂，
  分离普通与 renormalized half-twisted D-modules，统一局部 `GL(2)`
  induction/WD/L-factor 约定，收紧 local Shimura datum、卷积图、
  dual-root-datum 外部输入和 Dirichlet idele character。
- **HMS 与 prismatic**：补全 Novikov 值群、Liouville sector 和 sectorial
  descent 假设；把 etale comparison 的全局公式限制到 affine 情形，并为
  Hodge-Tate 有限生成性加入 Noetherian 假设。
- **量子力学与相对论**：补齐 Kato-Rellich 对称性、升降算符定义域、
  Dyson 连续性、绝对连续谱散射边界；统一 Noether/Hilbert 张量号差，并
  区分 ADM 能量、空间动量与不变质量。
- **弦论与 EFT**：把固定右动基态的 BPS 计数从普通 Cardy 定理中分离，
  加入 no-ghost 的非零动量边界；同步 EFT 的双插入振幅答案，并把物理
  有效性尺度 (M_{mathrm{gap}}) 与 Wilson 参考尺度
  (Lambda_{mathrm{ref}}) 分开。

范畴论组未发现新的高/中等级问题；其余组的 findings 均经过正文、答案、
索引和来源账本同步，而非只修改审计记录。

## 最终验证

- `python3 books/audit_oet_rigor.py --strict`：
  `errors=0 warnings=0`。
- `python3 books/category-theory/validate.py`：
  `validation=ok`，855 道章末/附录练习与 52 道综合题通过结构检查。
- `python3 books/quantum-mechanics/validate.py`：
  `validation=ok`，68 道章末/附录练习及答案、18 道综合题及提示闭合。
- 未闭合占位符定向扫描：无命中。
- `git diff --check -- books`：通过。

因此，本轮达到 OET 标准所指的教材本体闭合；仍不把它表述为逐页排版完成的
camera-ready 出版终稿。
