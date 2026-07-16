# 来源与外部输入账本

本账本固定本书实际使用的版本与定位。正文只有 EI-1 至 EI-9 是外部输入；其余定理必须在书内完整证明。外部输入的证明路线只解释如何从所列来源取得精确结论，不增加结论范围。若来源系统与正文系统不逐字相同，本账本必须说明增加或删去的规则，以及专门化为何有效。

## 参考版本与定位

| 编号 | 资料与版本 | 精确定位 | 本书用途 |
| --- | --- | --- | --- |
| S1 | Alan M. Turing, “On Computable Numbers, with an Application to the Entscheidungsproblem”, *Proceedings of the London Mathematical Society*, s2-42 (1936–1937), 230–265 | §§1–2 的机器定义；§6 的 universal computing machine | 通用模拟的历史原型 |
| S2 | Marvin L. Minsky, *Computation: Finite and Infinite Machines*, Prentice-Hall, 1967 | Ch. 14, “Very Simple Bases for Computability” | 寄存器机/计数器机的模型等价 |
| S3 | Hartley Rogers Jr., *Theory of Recursive Functions and Effective Computability*, McGraw-Hill, 1967；MIT Press 重印，1987 | Chs. 4–5，尤其参数化定理与可接受程序系统各节 | 程序枚举、通用函数、s-m-n、Rice 定理 |
| S4 | Henk Barendregt, *The Lambda Calculus: Its Syntax and Semantics*, revised ed., North-Holland, 1984 | Ch. 2 的替换与变元约定；Ch. 3, §3.2 的 Church–Rosser 定理 | α-等价、捕获避免替换、β-合流 |
| S5 | Benjamin C. Pierce, *Types and Programming Languages*, MIT Press, 2002 | Ch. 9, §9.3；Ch. 12, §12.1；Ch. 23, §§23.2–23.5 | STLC 类型安全与正规化；System F 规则 |
| S6 | Jean-Yves Girard, Yves Lafont, Paul Taylor, *Proofs and Types*, Cambridge University Press, 1989 | Chs. 4–6 的可归约性方法；Chs. 11–14 的二阶系统 | 强正规化和 System F 的补充定位 |
| S7 | Arthur Adjedj, Meven Lennon-Bertrand, Kenji Maillard, Pierre-Marie Pédrot, Loïc Pujet, “Martin-Löf à la Coq”, *CPP 2024*, 230–245, DOI 10.1145/3636501.3636951 | §2 的对象系统与 canonicity 口径；Theorem 4.1；Lemma 4.2；Theorems 6.1–6.3；归档工件 cpp24-submission, Zenodo 8367154 | 含 Π、Σ、Id、Nat、一个 predicative universe 与大消去的 MLTT 元理论 |
| S8 | Andreas Abel, Klaus Aehlig, Peter Dybjer, “Normalization by Evaluation for Martin-Löf Type Theory with One Universe”, *ENTCS* 173 (2007), 17–39, DOI 10.1016/j.entcs.2007.02.025 | §2 的演算；Theorem 4.9；Theorem 5.7；Corollary 5.9 | 单宇宙 NbE 的补充材料；不含 Σ/Id，不能单独承担 EI-5 |
| S9 | John C. Reynolds, “Types, Abstraction and Parametric Polymorphism”, in *Information Processing 83*, North-Holland, 1983, 513–523 | §§3–5 的关系解释与 Abstraction Theorem | 纯 System F 关系参数性 |
| S10 | Philip Wadler, “Theorems for Free!”, *FPCA 1989*, 347–359, DOI 10.1145/99370.99404 | §§2–3 的参数性实例和自由定理 | 参数性结论的教学实例 |
| S11 | Gordon D. Plotkin, *A Structural Approach to Operational Semantics*, DAIMI FN-19, Aarhus University, 1981；*Journal of Logic and Algebraic Programming* 60–61 (2004), 17–139 重印 | §§2–4 的转移系统、推导规则与结构操作语义 | 大小步和 SOS 规则口径 |
| S12 | Bard Bloom, Sorin Istrail, Albert R. Meyer, “Bisimulation Can’t Be Traced”, *JACM* 42(1) (1995), 232–268 | §2 的 GSOS 语言；congruence theorem | EI-7 的格式定理 |
| S13 | Jan Friso Groote, Frits Vaandrager, “Structured Operational Semantics and Bisimulation as a Congruence”, *Information and Computation* 100(2) (1992), 202–260 | §§2–5，尤其 ntyft/ntyxt 格式的 congruence 结果 | EI-7 的另一种精确格式 |
| S14 | Stephen Cole Kleene, “Origins of Recursive Function Theory”, *Annals of the History of Computing* 3(1) (1981), 52–67 | 关于递归函数、λ-定义性和 Church–Turing 论题的历史区分 | EI-1 中数学等价与方法论论题的边界 |
| S15 | Samson Abramsky, Achim Jung, “Domain Theory”, in *Handbook of Logic in Computer Science*, Vol. 3, Oxford University Press, 1994, 1–168 | §§2.1.5–2.1.6；Theorem 2.1.19；Lemma 2.1.20 | dcpo、连续函数、最小不动点和不动点归纳 |
| S16 | Glynn Winskel, *The Formal Semantics of Programming Languages*, MIT Press, 1993 | Ch. 2 的操作语义；Chs. 5–8 的归纳、域和指称语义；Ch. 7 的 while 语义 | 命令语言的操作/指称对应 |
| S17 | Gordon D. Plotkin, “LCF Considered as a Programming Language”, *Theoretical Computer Science* 5(3) (1977), 223–255 | §§3–5 的 PCF 操作语义、连续模型与充分性 | PCF computational adequacy |
| S18 | J. M. E. Hyland, C.-H. Luke Ong, “On Full Abstraction for PCF”, *Information and Computation* 163(2) (2000), 285–408, DOI 10.1006/inco.2000.2917 | 论文的 Full Abstraction Theorem 及 preceding definability results | PCF 游戏语义完全抽象 |
| S19 | Samson Abramsky, Radha Jagadeesan, Pasquale Malacaria, “Full Abstraction for PCF”, *Information and Computation* 163(2) (2000), 409–470, DOI 10.1006/inco.2000.2930 | §§2–4 的游戏模型；主 full-abstraction theorem | PCF 完全抽象的独立来源 |
| S20 | C. A. R. Hoare, “An Axiomatic Basis for Computer Programming”, *Communications of the ACM* 12(10) (1969), 576–580, DOI 10.1145/363235.363259 | pp. 576–579 的赋值、顺序、条件、while 规则及正确性解释 | 基本 Hoare 逻辑 |
| S21 | Stephen A. Cook, “Soundness and Completeness of an Axiom System for Program Verification”, *SIAM Journal on Computing* 7(1) (1978), 70–90, DOI 10.1137/0207005 | §§2–4 的解释、soundness 与 relative completeness 结果 | EI-9 |
| S22 | Eugenio Moggi, “Notions of Computation and Monads”, *Information and Computation* 93(1) (1991), 55–92, DOI 10.1016/0890-5401(91)90052-4 | §§2–3 的 monad 与 Kleisli 语义 | 第 6 章效应接口 |

## 外部输入登记

### EI-1：计数器机的通用解释器与经典模型等价

**精确陈述。** 对第 1 章定义 1.1–1.6 的计数器机编码，存在偏可计算通用函数 $U(e,x)$，满足 $U(e,x)=\varphi_e(x)$；计数器机、Turing 机、无类型 λ-可定义数值函数与一般递归函数给出同一偏函数类。

**来源定位。** 通用机见 S1 §6；计数器机与标准机器模型的互模拟见 S2 Ch. 14；可接受程序系统和模型不变性见 S3 Chs. 4–5。

**使用与边界。** 第 1 章使用通用模拟和函数类等价。有限互模拟是数学定理；“所有直观有效过程都落入该函数类”是 Church–Turing 论题，不作为已证数学结论，历史区分见 S14。

### EI-2：可接受程序系统的 s-m-n 参数定理

**精确陈述。** 对 EI-1 固定的可接受一元程序枚举，存在全可计算 $s:\mathbb N^2\to\mathbb N$，使 $\varphi_{s(e,a)}(x)=\varphi_e(\langle a,x\rangle)$。

**来源定位。** S3 Ch. 5 的参数化定理及可接受程序系统各节。正文不调用递归定理；因此递归定理不列入 EI-2 的契约。

**使用与边界。** 第 2 章 T2.3 用它把给定输入固化为程序索引。结论依赖固定枚举可接受，不能对任意人为编号成立。

### EI-3：无类型 λ 演算的 Church–Rosser 合流性

**精确陈述。** 对第 3 章定义 3.6 的完整 β-归约，若 $e\to_\beta^*a$ 且 $e\to_\beta^*b$，则存在 $c$ 使 $a\to_\beta^*c$ 且 $b\to_\beta^*c$，等式均按 α-等价读取。

**来源定位。** S4 Ch. 3, §3.2。该来源先固定变元约定和替换，再以平行归约证明 Church–Rosser。

**使用与边界。** 第 3 章只从合流性推出 β-正规形按 α-等价唯一，不使用标准化策略或可判定正规化。

### EI-4：纯 STLC 的强正规化

**精确陈述。** 对第 4 章定义 4.1、4.4 的纯 STLC，完整 β-归约允许在任意项上下文中收缩 redex。若 $\Gamma\vdash e:A$，则不存在从 $e$ 出发的无限完整 β-归约序列。

**来源定位。** S5 Ch. 12, §12.1 的可归约性证明；S6 Chs. 4–6 给出同一方法的证明论版本。第 4 章 preservation/progress 只依赖书内替换与反演，不依赖本 EI。

**系统对齐。** S5 可用 de Bruijn 表示而正文使用有名 α-等价类；两者由标准去名/复名双向翻译保持类型与 β 步，故该表示差异不改变定理。

### EI-5：一个 predicative Russell-style universe 的 intensional MLTT 元理论

**精确对象系统。** 本 EI 只指第 5 章定义 5.1–5.10：有 Π、Σ 及投影、intensional Id 及 J、Nat 及依赖递归器、一个 Russell-style 小类型宇宙 $\mathcal U$；允许大消去；有相应 β 计算和有类型的判断等价；没有 $\mathcal U:\mathcal U$、判断性 η、等式反射、UIP/K、一般递归、空类型或额外公理。

**精确结论。** 每个良类型项沿 β 与递归器计算规则弱头正规化；若 $\varnothing\vdash n:\mathsf{Nat}$，则 $n$ 判断等价于唯一 numeral。本书不从 EI-5 输入无 η 判断等价的判定算法。

**主来源定位。** S7 §2 的对象系统同时包含 Π、Σ、Nat、intensional Id、一个 predicative universe 和大消去；Theorem 4.1 与 Lemma 4.2 给出逻辑关系基本引理和弱头正规化，§6 的 Theorems 6.1–6.3 给出算法式系统的 soundness、completeness 与类型检查可判定性。形式化版本固定为 Zenodo 8367154，而非浮动的仓库主分支。

**规则差异及专门化。** S7 的展示系统另含空类型以及 Π/Σ 判断性 η。正文系统删除空类型构造子、消去规则和 η 规则，所以每个正文类型推导仍是来源系统中的推导。S7 Theorem 4.1 与 Lemma 4.2 给出的 reducibility 和弱头归约使用同一 β/递归器计算关系；限制到较小推导类仍成立。对闭 Nat 项，来源的 canonical weak-head 形状排除变量 neutral，删除空类型和 η 不会增加新形状，因此 canonicity 同样限制成立。这里不从来源的 βη 转换判定器推断正文 β 转换可判定。

**补充来源边界。** S8 精确处理一个 universe、Π、Nat 与 βη-NbE，但不含 Σ 或 Id；它只说明单宇宙 NbE 路线，不能单独承担本书 EI-5。无宇宙或仅 Π-fragment 的正规化结果同样不得替代 S7。

### EI-6：纯 System F 的关系参数性

**精确陈述。** 对第 6 章定义 6.1–6.3 的纯、无递归、无效应 System F，类型变量由二元关系解释，箭头和全称类型按关系逻辑递归解释；每个可类型项保持该关系解释。

**来源定位。** S9 §§3–5 的 Abstraction Theorem；具体自由定理见 S10 §§2–3。System F 语法与替换口径可与 S5 Ch. 23 对照。

**使用与边界。** 正文只取闭项 $\forall\alpha.\alpha\to\alpha$ 的结论。引用、异常、非终止、类型反射或严格性原语需要改变关系类别，均不由 EI-6 覆盖。

### EI-7：特定 SOS 格式的同余性结果

**精确陈述。** 对满足各来源明确正负前提限制的 GSOS 或 ntyft/ntyxt 规则系统，强双模拟是语言算子的同余关系。

**来源定位。** GSOS 见 S12 §2 及其 congruence theorem；ntyft/ntyxt 见 S13 §§2–5。S11 只提供一般 SOS 方法，不单独推出格式定理。

**使用与边界。** 第 7 章的确定性、大小步对应和 CEK 正确性均已在书内对具体规则证明，不调用 EI-7。EI-7 只标出把结果推广到任意语言规范时所需的外部条件。

### EI-8：PCF computational adequacy 与游戏语义完全抽象

**精确陈述。** 对标准 PCF 的 ground 观察，Scott 连续函数模型 computationally adequate；该模型一般不完全抽象。相应的游戏语义模型满足定义 10.4 的双向 full abstraction。

**来源定位。** Scott/连续模型及充分性见 S17 §§3–5；完全抽象见 S18 的 Full Abstraction Theorem 与 S19 的主 full-abstraction theorem。域论基础见 S15 §§2.1.5–2.1.6 和 Theorem 2.1.19。

**使用与边界。** 第 8 章只外用 PCF 结果；while 语言的最小不动点与操作/指称双向一致已在书内证明。第 10 章只比较 adequacy 与 full abstraction，不重建游戏模型。

### EI-9：Cook 相对完备性

**精确陈述。** 对确定的 while/ALGOL 型语言及部分正确性，若断言语言能表达所需 strongest postconditions 或 weakest liberal preconditions，并允许使用其解释中的所有真断言作为 consequence 前提，则每个语义有效 Hoare 三元组均可在 Hoare 系统中推导。

**来源定位。** 基本规则及其语义见 S20 pp. 576–579；相对完备性的解释、表达力假设和证明见 S21 §§2–4。

**使用与边界。** 该结论相对于断言理论 oracle，不给出一阶算术真理判定器，也不推出有效可判定的完全验证算法。第 9 章 T9.1 的 soundness 已在书内完整证明，不依赖 EI-9。
